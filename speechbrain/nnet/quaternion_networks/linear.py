"""Library implementing quaternion-valued linear transformation.

Authors
 * Titouan Parcollet 2020
"""

import torch
import logging
from speechbrain.nnet.quaternion_networks.quaternion_ops import (
    affect_init,
    unitary_init,
    quaternion_init,
    quaternion_linear_op,
    check_quaternion_input,
    quaternion_linear_rotation_op,
    QuaternionLinearCustomBackward,
)

logger = logging.getLogger(__name__)


class QuaternionLinear(torch.nn.Module):
    """ This function implements a fully connected quaternion-valued
        linear layer: y = Wx + b. y, W, x and b are thus quaternion
        numbers. A quaternion number is written as: r + xi + yj + zk.
        A tensor of quaternion numbers x = [batch, 32] can be understood as
        [batch, 0:7] = R, [batch, 8:15] = Xi, [batch, 16:23] = Yi, and
        [batch, 24:31] = Xi. Thus the features dimension is cut in four
        (must be dividible by 4).

    Arguments
    ---------
    n_neurons : int
        it is the number of output neurons (i.e, the dimensionality of the
        output). Please note that these are quaternion-valued neurons. If 256
        neurons are specified, the output dimension will be 1024.
    input_shape : tuple
        Expected size of the input.
    bias : bool
        if True, the additive bias b is adopted.
    init_criterion: str , optional
        Default: he.
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the quaternion-valued weights.
    weight_init: str, optional
        Default: quaternion.
        (quaternion, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights. "quaternion" will generate quaternion-valued
        weights following the init_criterion and the quaternion  polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Quaternion recurrent neural networks", Parcollet T.
    autograd: bool, optional
        Default: True.
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower. This only works with
        spinor = False.
    spinor: bool, optional
        Default: False.
        When True, the layer will be turned into a spinor layer. More precisely
        W*x will be turned into W*x*W-1. The input x will be rotated by W such
        as in a spinor neural network. However, x MUST be a quaternion with
        the real part equal to zero. (0 + xi + yj + zk). Indeed, the rotation
        operation only acts on the vector part. Note that W will always be
        normalized before the rotation to ensure the quaternion algebra.
        More details in: "Quaternion neural networks", Parcollet T.
    vector_scale: bool, optional
        Default: False.
        The vector_scale is only used when spinor = True. In the context of a
        spinor neural network, multiple rotations of the input vector x are
        performed and summed. Hence, the norm of the output vector always
        increases with the number of layers, making the neural network instable
        with deep configurations. The vector_scale parameters are learnable
        parameters that acts like gates by multiplying the output vector with
        a small trainable parameter.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin = QuaternionLinear(n_neurons=100, input_shape=inputs.shape)
    >>> output = lin(inputs)
    >>> output.shape
    torch.Size([10, 50, 400])
    """

    def __init__(
        self,
        n_neurons,
        input_shape,
        bias=True,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd=True,
        spinor=False,
        vector_scale=False,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.bias = bias
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd
        self.spinor = spinor
        self.vector_scale = vector_scale

        # When initialising with speechbrain the input_shape is an integer !
        # we need to transform it into a list it works with all the question ops
        if isinstance(input_shape, int):
            input_shape = [1, input_shape]

        # Check the quaternion_valued form of the input
        check_quaternion_input(input_shape)

        # Computing the quaternion dimensionality of the input
        self.in_features = input_shape[-1] // 4
        self.out_features = self.n_neurons

        # Defining the weights
        self.r_weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )
        self.i_weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )
        self.j_weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )
        self.k_weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )

        # Spinor specific parameters
        if self.spinor:
            self.zero_kernel = torch.nn.Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )

        if self.spinor and self.vector_scale:
            self.scale_param = torch.nn.Parameter(
                torch.Tensor(self.in_features, self.out_features)
            )
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        else:
            self.scale_param = None

        if self.bias:
            self.b = torch.nn.Parameter(torch.Tensor(4 * n_neurons))
            self.b.data.fill_(0)
        else:
            self.b = None

        # Managing the weight initialization and bias
        self.winit = {"quaternion": quaternion_init, "unitary": unitary_init}[
            self.weight_init
        ]

        # Initialise the weights
        affect_init(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.winit,
            init_criterion,
        )

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """

        if self.autograd:
            if self.spinor:
                out = quaternion_linear_rotation_op(
                    x,
                    self.r_weight,
                    self.i_weight,
                    self.j_weight,
                    self.k_weight,
                    self.b,
                    self.scale_param,
                    self.zero_kernel,
                )
            else:
                out = quaternion_linear_op(
                    x,
                    self.r_weight,
                    self.i_weight,
                    self.j_weight,
                    self.k_weight,
                    self.b,
                )
        else:
            out = QuaternionLinearCustomBackward.apply(
                x,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.b,
            )

        return out
