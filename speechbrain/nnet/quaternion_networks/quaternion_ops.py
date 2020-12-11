"""This library implements different operations needed by quaternion-
 valued architectures.
  This work is inspired by:
  "Quaternion neural networks" - Parcollet T.
  "Quaternion recurrent neural networks" - Parcollet T. et al.
  "Quaternion convolutional neural networks for end-to-end automatic speech
  recognition" - Parcollet T. et al.
  "Deep quaternion networks" - Gaudet Chase J. et al.

Authors
 * Titouan Parcollet 2020
"""

import torch
import math
from scipy.stats import chi


def quaternion_linear_op(input, r_weight, i_weight, j_weight, k_weight, bias):
    """
    Applies a quaternion linear transformation to the incoming data:
    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion
    representation so when we do torch.mm(Input,W) it's equivalent
    to W * Inputs.

    Arguments
    ---------
    input: torch.Tensor
        Quaternion input tensor to be transformed.
    r_weight: torch.Parameter
        Real part of the quaternion weight matrix of this layer.
    i_weight: torch.Parameter
        First imaginary part of the quaternion weight matrix of this layer.
    j_weight: torch.Parameter
        Second imaginarypart of the quaternion weight matrix of this layer.
    k_weight: torch.Parameter
        Third imaginary part of the quaternion weight matrix of this layer.
    bias: torch.Parameter
    """

    cat_kernels_4_r = torch.cat(
        [r_weight, -i_weight, -j_weight, -k_weight], dim=0
    )
    cat_kernels_4_i = torch.cat(
        [i_weight, r_weight, -k_weight, j_weight], dim=0
    )
    cat_kernels_4_j = torch.cat(
        [j_weight, k_weight, r_weight, -i_weight], dim=0
    )
    cat_kernels_4_k = torch.cat(
        [k_weight, -j_weight, i_weight, r_weight], dim=0
    )
    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k],
        dim=1,
    )

    # If the input is already [batch*time, N]
    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output + bias
        else:
            return output


def quaternion_linear_rotation_op(
    input, r_weight, i_weight, j_weight, k_weight, bias, scale
):
    """
    Applies a quaternion rotation transformation to the incoming data:
    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Works for unitary and non unitary weights (they will be normalised).
    The initial size of the input must be a multiple of 4 with the real part
    equal to zero. Rotations only affect the vector part of a quaternion.

    Arguments
    ---------
    input: torch.Tensor
        Quaternion input tensor to be transformed.
    r_weight: torch.Parameter
        Real part of the quaternion weight matrix of this layer.
    i_weight: torch.Parameter
        First imaginary part of the quaternion weight matrix of this layer.
    j_weight: torch.Parameter
        Second imaginarypart of the quaternion weight matrix of this layer.
    k_weight: torch.Parameter
        Third imaginary part of the quaternion weight matrix of this layer.
    bias: torch.Parameter
    scale: torch.Parameter
        In the context of a spinor neural network, multiple rotations of
        the input vector x are performed and summed. Hence, the norm of
        the output vector always increases with the number of layers, making
        the neural network instable with deep configurations. The scale
        parameters are learnable parameters that acts like gates by multiplying
        the output vector with a small trainable parameter.
    """

    # First we normalise the quaternion weights. Only unit quaternions are
    # valid rotations.
    square_r = r_weight * r_weight
    square_i = i_weight * i_weight
    square_j = j_weight * j_weight
    square_k = k_weight * k_weight

    norm = torch.sqrt(square_r + square_i + square_j + square_k) + 0.0001

    r_n_weight = r_weight / norm
    i_n_weight = i_weight / norm
    j_n_weight = j_weight / norm
    k_n_weight = k_weight / norm

    # See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for
    # the rest of the equations.
    norm_factor = 2.0

    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)

    ri = norm_factor * r_n_weight * i_n_weight
    rj = norm_factor * r_n_weight * j_n_weight
    rk = norm_factor * r_n_weight * k_n_weight

    ij = norm_factor * i_n_weight * j_n_weight
    ik = norm_factor * i_n_weight * k_n_weight

    jk = norm_factor * j_n_weight * k_n_weight

    if scale is not None:
        rot_kernel_1 = torch.cat(
            [
                scale * (1.0 - (square_j + square_k)),
                scale * (ij - rk),
                scale * (ik + rj),
            ],
            dim=0,
        )
        rot_kernel_2 = torch.cat(
            [
                scale * (ij + rk),
                scale * (1.0 - (square_i + square_k)),
                scale * (jk - ri),
            ],
            dim=0,
        )
        rot_kernel_3 = torch.cat(
            [
                scale * (ik - rj),
                scale * (jk + ri),
                scale * (1.0 - (square_i + square_j)),
            ],
            dim=0,
        )
    else:
        rot_kernel_1 = torch.cat(
            [1.0 - (square_j + square_k), (ij - rk), (ik + rj)], dim=0
        )
        rot_kernel_2 = torch.cat(
            [(ij + rk), 1.0 - (square_i + square_k), (jk - ri)], dim=0
        )
        rot_kernel_3 = torch.cat(
            [(ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=0
        )

    global_rot_kernel = torch.cat(
        [rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1
    )

    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output + bias
        else:
            return output


def quaternion_init(
    in_features, out_features, kernel_size=None, criterion="glorot"
):
    """Returns a matrix of quaternion numbers initialised with the method
        described in "Quaternion Recurrent Neural Network " - Parcollt T.

    Arguments
    ---------
    in_features : int
        Number of real values of the input layer (quaternion // 4)
    out_features : int
        Number of real values of the output layer (quaternion // 4)
    kernel_size : int
        Kernel_size for convolutional layers (ex: (3,3))
    criterion: str, (glorot, he)
    """

    if kernel_size is not None:
        receptive_field = torch.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == "glorot":
        s = 1.0 / torch.sqrt(2 * (fan_in + fan_out))
    else:
        s = 1.0 / torch.sqrt(2 * fan_in)

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = torch.prod(kernel_shape)
    v_i = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_j = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_k = torch.FloatTensor(number_of_weights).uniform_(-1, 1)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = torch.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = torch.FloatTensor(kernel_shape).uniform_(-math.pi, math.pi)

    weight_r = modulus * torch.cos(phase)
    weight_i = modulus * v_i * torch.sin(phase)
    weight_j = modulus * v_j * torch.sin(phase)
    weight_k = modulus * v_k * torch.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def unitary_init(in_features, out_features, kernel_size=None, criterion="he"):
    """Returns a matrix of unitary quaternion numbers.

    Arguments
    ---------
    in_features : int
        Number of real values of the input layer (quaternion // 4)
    out_features : int
        Number of real values of the output layer (quaternion // 4)
    kernel_size : int
        Kernel_size for convolutional layers (ex: (3,3))
    criterion: str, (glorot, he)
    """

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = torch.prod(kernel_shape)
    v_r = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_r = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_i = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_j = torch.FloatTensor(number_of_weights).uniform_(-1, 1)
    v_k = torch.FloatTensor(number_of_weights).uniform_(-1, 1)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = (
            torch.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2)
            + 0.0001
        )
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)


def affect_init(
    r_weight, i_weight, j_weight, k_weight, init_func, init_criterion
):
    """Applies the weight initialization function given to the parameters.

    Arguments
    ---------
    real_weight: torch.Parameters, (nb_quaternion_in, nb_quaternion_out)
    imag_weight: torch.Parameters, (nb_quaternion_in, nb_quaternion_out)
    init_func: function, (unitary_init, complex_init)
    criterion: str, (glorot, he)
    """

    r, i, j, k = init_func(
        r_weight.size(0),
        r_weight.size(1),
        r_weight.size(2),
        r_weight.size(3),
        None,
        init_criterion,
    )

    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def check_quaternion_input(input_shape):
    """Check the quaternion-valued shape for a linear layer.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    """

    if len(input_shape) not in {2, 3}:
        raise Exception(
            "Quaternion linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input_shape[-1]

    if nb_hidden % 1 != 0:
        raise Exception(
            "Quaternion Tensors must have a dimensions dividible by 4."
            " input.size()[1] = " + str(nb_hidden)
        )
