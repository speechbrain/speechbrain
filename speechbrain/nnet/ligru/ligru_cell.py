"""This library implements a LiGRU cell needed for the LiGRU.

LiGRU is a Gated Recurrent Unit introduced by Assoc. Prof Mirco 
Ravanelli in 2018 (see: https://arxiv.org/pdf/1803.10225.pdf).

Authors
 * Adel Moumen 2022
"""

from tokenize import String
import torch
from torch import Tensor
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

_preamble_relu = '''
template <typename T> __device__ T gradient_activation_hcand(T x) { return (x > 0. ? 1. : 0.); }
'''

_preamble_leaky_relu = '''
template <typename T> __device__ T gradient_activation_hcand(T x) { 
    T negative_slope = 1e-2;  
    return (x > 0. ? 1. : negative_slope); 
}
'''

def transform_tensor_to_cupy(x: Tensor):
    """Transform a PyTorch Tensor located on device="cuda" to a CuPy array. 
    
    Argument
    --------
        x : torch.Tensor
    """
    return cp.ascontiguousarray(cp.from_dlpack(torch.utils.dlpack.to_dlpack(x.detach())))


class _ligru_cell_cupy(autograd.Function):
    """
    This class redefine the backpropagation of a LiGRU cell and implement the backward using CuPy. 
    By doing so, we speed up the training by a factor of 2x in comparison to the original implementation. 
    """

    @staticmethod
    def forward(ctx, wx: Tensor, u: Tensor, ht: Tensor, drop_mask: Tensor, act: nn.Module) -> Tensor:
        """Returns the hidden states for each time step and save the intermediates results for the backward.

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
            Only two possibilities for now:
                1) ReLU,
                2) Leaky ReLU with slope_parameter of 1e-2 (default parameter of PyTorch).
        """
        
        # save values for backward
        hiddens        = []
        candidate_gate = []
        update_gate    = []
        save_at        = []
        h_init         = ht

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
        
        ht    = torch.stack(hiddens, dim=1) 
        zt    = torch.stack(update_gate, dim=1)
        at    = torch.stack(save_at, dim=1)
        hcand = torch.stack(candidate_gate, dim=1)

        ctx.save_for_backward(h_init, u, wx, zt, at, ht, hcand, drop_mask)
        ctx.activation_function = act

        return ht



    @staticmethod
    def backward(ctx, grad_out):
        '''
        '''
        h_init, u, wx, zt, at, ht, hcand, drop_mask,  = ctx.saved_tensors

        activation_function = ctx.activation_function

        # we need to reshape our h_init tensor  
        if h_init.shape[0] != ht[:, 0].shape[0]:
            h_init = h_init.repeat(ht[:, 0].shape[0], 1)
            
        h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out, = (transform_tensor_to_cupy(x) for x in [h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out])

        dwx     = cp.empty_like(wx)
        du      = cp.zeros_like(u)
        dh      = cp.empty_like(h_init)
        dh_prev = cp.zeros_like(h_init)
        idx     = dwx.shape[2] // 2

        _preamble = _preamble_leaky_relu if activation_function.__class__.__name__  == "LeakyReLU" else _preamble_relu

        _ligru_cell_backward_kernel = cp.ElementwiseKernel(
            'T grad_out, T dh_prev, T zt, T at, T drop_mask, T ht, T hcand',
            'T dh, T dat, T dzt, T grad_dh_prev',
            '''
            dh = grad_out + dh_prev;
            T temp = (1 - zt) * dh;
            dat = (at > 0. ? 1. : 0.) * drop_mask * temp;
            dzt = (ht - hcand) * temp * zt;
            grad_dh_prev = dh * zt; 
            ''',
            '_ligru_cell_backward_kernel', 
            preamble=_preamble)

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
            du     += dwx[:, t].T.dot(ht_ )   

        return torch.from_dlpack(dwx), torch.from_dlpack(du), torch.from_dlpack(dh), None, None

if __name__ == "__main__":
    torch.manual_seed(22)
    w = torch.randn(16, 7, 256, device="cuda", requires_grad=True)
    u = torch.randn(256, 128, device="cuda", requires_grad=True)
    ht = torch.randn(16, 128, device="cuda", requires_grad=True)
    drop_mask = torch.randn(16, 128, device="cuda")
    act = nn.ReLU()

    out = _ligru_cell_cupy.apply(w, u, ht, drop_mask, act)
    out.sum().backward()

    print(w.grad)