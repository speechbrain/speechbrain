"""This library implements a LiGRU cell needed for the LiGRU.

LiGRU is a Gated Recurrent Unit introduced by Assoc. Prof Mirco 
Ravanelli in 2018 (see: https://arxiv.org/pdf/1803.10225.pdf).

Authors
 * Adel Moumen 2022
"""

import torch
import torch.autograd as autograd
import torch.nn as nn 

try:
    import cupy as cp
except ImportError:
    err_msg = "The optional dependency CuPy is needed to use LiGRU on CuPy\n"
    err_msg += "Cannot import CuPy.\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "If you use your localhost:\n"
    err_msg += "$ python -m pip install -U setuptools pip\n"
    err_msg += "$ pip install cupy-cudaXXX (XXX is your Cuda Toolkit version)\n"
    err_msg += "If you use conda:\n"
    err_msg += "$ conda install -c conda-forge cupy"
    err_msg += "for more details: https://docs.cupy.dev/en/stable/install.html"
    err_msg += "=============================\n"
    raise ImportError(err_msg)

_preamble_gradient_relu = '''
template <typename T> __device__ T gradient_activation_hcand(T x) { return (x > 0. ? 1. : 0.); }
'''

_preamble_gradient_leaky_relu = '''
template <typename T> __device__ T gradient_activation_hcand(T x) { 
    T negative_slope = 1e-2;  
    return (x > 0. ? 1. : negative_slope); 
}
'''

_preamble_gradient_tanh = '''
template <typename T> __device__ T gradient_activation_hcand(T x) { return 1 - x * x; }
'''

def transform_tensor_to_cupy(x):
    """Transform a PyTorch Tensor located on device="cuda" to a CuPy array. 
    
    Argument
    --------
        x : torch.Tensor
    """
    return cp.ascontiguousarray(cp.from_dlpack(torch.utils.dlpack.to_dlpack(x.detach())))


class _ligru_cell_cupy(autograd.Function):
    """This class redefine the backpropagation of a LiGRU cell and implement the backward using CuPy. 
    By doing so, we speed up the training by a factor of ~4x in comparison to the original implementation
    and ~2.5x in comparison to the jitted version. 
    """

    @staticmethod
    def forward(ctx, wx, u, ht, drop_mask, act):
        """Compute the hidden states over each time step and save the intermediates results for the backward.

        The utilisation of Tanh in the LiGRU cell is not recommended because it increase the instability and can leads to 
        nan values in the gradients.
        
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input. 
        u  : torch.Tensor
            Recurrent weight. 
        ht : torch.Tensor
            Hidden state. 
        drop_mask : torch.Tensor
            Dropout mask. 
        act : nn.Module
            Activation Function. 
            Only three possibilities for now:
                1) ReLU,
                2) Leaky ReLU with slope_parameter of 1e-2 (default parameter of PyTorch),
                3) Tanh. 
        """
        
        # save values for backward
        hiddens        = []
        candidate_gate = []
        update_gate    = []
        save_at        = []
        h_init         = ht

        # iterate over each timesteps
        for k in range(wx.shape[1]):
            
            gates  = wx[:, k] + ht @ u.T
            at, zt = gates.chunk(2, 1)
            zt     = torch.sigmoid(zt)
            hcand  = act(at) * drop_mask
            ht     = ht * zt + (1 - zt) * hcand

            hiddens.append(ht)
            candidate_gate.append(hcand)
            update_gate.append(zt)
            save_at.append(at)
        
        # stacks values
        ht    = torch.stack(hiddens, dim=1) 
        zt    = torch.stack(update_gate, dim=1)
        at    = torch.stack(save_at, dim=1)
        hcand = torch.stack(candidate_gate, dim=1)

        ctx.save_for_backward(h_init, u, wx, zt, at, ht, hcand, drop_mask)
        ctx.activation_function = act

        return ht

    @staticmethod
    def backward(ctx, grad_out):
        """Run the backward phase of the forward call defined above. This implementation
        is dependant of CuPy. 

        Arguments
        ---------
        ctx : context variable
        grad_out : torch.Tensor
        """
        h_init, u, wx, zt, at, ht, hcand, drop_mask,  = ctx.saved_tensors

        activation_function = ctx.activation_function

        # we need to reshape our h_init tensor if h_init doesnt match the shape.
        if h_init.shape[0] != ht[:, 0].shape[0]:
            h_init = h_init.repeat(ht[:, 0].shape[0], 1)
            
        # Tensor -> Cuda CuPy Array
        h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out, = (transform_tensor_to_cupy(x) for x in [h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out])

        # allocate memory space 
        dwx     = cp.empty_like(wx)
        du      = cp.zeros_like(u)
        dh      = cp.empty_like(h_init)
        dh_prev = cp.zeros_like(h_init)
        idx     = dwx.shape[2] // 2

        # find the activation function and load the appropriate preamble 
        activation_function_name = activation_function.__class__.__name__
        if activation_function_name  == "LeakyReLU":
            preamble_gradient = _preamble_gradient_leaky_relu    
        elif activation_function_name  == "tanh":
            preamble_gradient = _preamble_gradient_tanh    
        else:
            preamble_gradient = _preamble_gradient_relu

        _ligru_cell_backward_kernel = cp.ElementwiseKernel(
            'T grad_out, T dh_prev, T zt, T at, T drop_mask, T ht, T hcand',
            'T dh, T dat, T dzt, T grad_dh_prev',
            '''
            dh = grad_out + dh_prev;
            T temp = (1 - zt) * dh;
            dat = gradient_activation_hcand(at) * drop_mask * temp;
            dzt = (ht - hcand) * temp * zt;
            grad_dh_prev = dh * zt; 
            ''',
            '_ligru_cell_backward_kernel', 
            preamble=preamble_gradient
        )

        # iterate over each timesteps in reversed
        for t in reversed(range(dwx.shape[1])):

            ht_ = h_init if t - 1 < 0 else ht[:, t - 1]
            
            _ligru_cell_backward_kernel( 
                grad_out[:, t], 
                dh_prev, 
                zt[:, t], 
                at[:, t], 
                drop_mask, 
                ht_, 
                hcand[:, t], 
                dh, 
                dwx[:, t, :idx], 
                dwx[:, t, idx:], 
                dh_prev
            )

            dh_prev = dh_prev + dwx[:, t].dot(u)
            du     += dwx[:, t].T.dot(ht_)   

        return torch.from_dlpack(dwx), torch.from_dlpack(du), torch.from_dlpack(dh), None, None