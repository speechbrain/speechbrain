from functools import partial
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from .shared import BackboneRegistry, ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, \
    DiffusionStepEmbedding, GaussianFourierProjection, FeatureMapDense, torch_complex_from_reim


def get_activation(name):
    if name == "silu":
        return nn.SiLU
    elif name == "relu":
        return nn.ReLU
    elif name == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Unknown activation: {name}")


class BatchNorm(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError("expected 4D or 3D input (got {}D input)".format(input.dim()))


class OnReIm(nn.Module):
    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x):
        return torch_complex_from_reim(self.re_module(x.real), self.im_module(x.imag))


# Code for DCUNet largely copied from Danilo's `informedenh` repo, cheers!

def unet_decoder_args(encoders, *, skip_connections):
    """Get list of decoder arguments for upsampling (right) side of a symmetric u-net,
    given the arguments used to construct the encoder.
    Args:
        encoders (tuple of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding)):
            List of arguments used to construct the encoders
        skip_connections (bool): Whether to include skip connections in the
            calculation of decoder input channels.
    Return:
        tuple of length `N` of tuples of (in_chan, out_chan, kernel_size, stride, padding):
            Arguments to be used to construct decoders
    """
    decoder_args = []
    for enc_in_chan, enc_out_chan, enc_kernel_size, enc_stride, enc_padding, enc_dilation in reversed(encoders):
        if skip_connections and decoder_args:
            skip_in_chan = enc_out_chan
        else:
            skip_in_chan = 0
        decoder_args.append(
            (enc_out_chan + skip_in_chan, enc_in_chan, enc_kernel_size, enc_stride, enc_padding, enc_dilation)
        )
    return tuple(decoder_args)


def make_unet_encoder_decoder_args(encoder_args, decoder_args):
    encoder_args = tuple(
        (
            in_chan,
            out_chan,
            tuple(kernel_size),
            tuple(stride),
            tuple([n // 2 for n in kernel_size]) if padding == "auto" else tuple(padding),
            tuple(dilation)
        )
        for in_chan, out_chan, kernel_size, stride, padding, dilation in encoder_args
    )

    if decoder_args == "auto":
        decoder_args = unet_decoder_args(
            encoder_args,
            skip_connections=True,
        )
    else:
        decoder_args = tuple(
            (
                in_chan,
                out_chan,
                tuple(kernel_size),
                tuple(stride),
                tuple([n // 2 for n in kernel_size]) if padding == "auto" else padding,
                tuple(dilation),
                output_padding,
            )
            for in_chan, out_chan, kernel_size, stride, padding, dilation, output_padding in decoder_args
        )

    return encoder_args, decoder_args


DCUNET_ARCHITECTURES = {
    "DCUNet-10": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1, 32,  (7, 5), (2, 2), "auto", (1,1)),
            (32, 64, (7, 5), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-16": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1,  32, (7, 5), (2, 2), "auto", (1,1)),
            (32, 32, (7, 5), (2, 1), "auto", (1,1)),
            (32, 64, (7, 5), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DCUNet-20": make_unet_encoder_decoder_args(
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1,  32, (7, 1), (1, 1), "auto", (1,1)),
            (32, 32, (1, 7), (1, 1), "auto", (1,1)),
            (32, 64, (7, 5), (2, 2), "auto", (1,1)),
            (64, 64, (7, 5), (2, 1), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 64, (5, 3), (2, 1), "auto", (1,1)),
            (64, 64, (5, 3), (2, 2), "auto", (1,1)),
            (64, 90, (5, 3), (2, 1), "auto", (1,1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
    "DilDCUNet-v2": make_unet_encoder_decoder_args(  # architecture used in SGMSE / Interspeech paper
        # Encoders:
        # (in_chan, out_chan, kernel_size, stride, padding, dilation)
        (
            (1,  32,   (4, 4), (1, 1), "auto", (1, 1)),
            (32, 32,   (4, 4), (1, 1), "auto", (1, 1)),
            (32, 32,   (4, 4), (1, 1), "auto", (1, 1)),
            (32, 64,   (4, 4), (2, 1), "auto", (2, 1)),
            (64, 128,  (4, 4), (2, 2), "auto", (4, 1)),
            (128, 256, (4, 4), (2, 2), "auto", (8, 1)),
        ),
        # Decoders: automatic inverse
        "auto",
    ),
}


@BackboneRegistry.register("dcunet")
class DCUNet(nn.Module):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--dcunet-architecture", type=str, default="DilDCUNet-v2", choices=DCUNET_ARCHITECTURES.keys(), help="The concrete DCUNet architecture. 'DilDCUNet-v2' by default.")
        parser.add_argument("--dcunet-time-embedding", type=str, choices=("gfp", "ds", "none"), default="gfp", help="Timestep embedding style. 'gfp' (Gaussian Fourier Projections) by default.")
        parser.add_argument("--dcunet-temb-layers-global", type=int, default=1, help="Number of global linear+activation layers for the time embedding. 1 by default.")
        parser.add_argument("--dcunet-temb-layers-local", type=int, default=1, help="Number of local (per-encoder/per-decoder) linear+activation layers for the time embedding. 1 by default.")
        parser.add_argument("--dcunet-temb-activation", type=str, default="silu", help="The (complex) activation to use between all (global&local) time embedding layers.")
        parser.add_argument("--dcunet-time-embedding-complex", action="store_true", help="Use complex-valued timestep embedding. Compatible with 'gfp' and 'ds' embeddings.")
        parser.add_argument("--dcunet-fix-length", type=str, default="pad", choices=("pad", "trim", "none"), help="DCUNet strategy to 'fix' mismatched input timespan. 'pad' by default.")
        parser.add_argument("--dcunet-mask-bound", type=str, choices=("tanh", "sigmoid", "none"), default="none", help="DCUNet output bounding strategy. 'none' by default.")
        parser.add_argument("--dcunet-norm-type", type=str, choices=("bN", "CbN"), default="bN", help="The type of norm to use within each encoder and decoder layer. 'bN' (real/imaginary separate batch norm) by default.")
        parser.add_argument("--dcunet-activation", type=str, choices=("leaky_relu", "relu", "silu"), default="leaky_relu", help="The activation to use within each encoder and decoder layer. 'leaky_relu' by default.")
        return parser

    def __init__(
        self,
        dcunet_architecture: str = "DilDCUNet-v2",
        dcunet_time_embedding: str = "gfp",
        dcunet_temb_layers_global: int = 2,
        dcunet_temb_layers_local: int = 1,
        dcunet_temb_activation: str = "silu",
        dcunet_time_embedding_complex: bool = False,
        dcunet_fix_length: str = "pad",
        dcunet_mask_bound: str = "none",
        dcunet_norm_type: str = "bN",
        dcunet_activation: str = "relu",
        embed_dim: int = 128,
        **kwargs
    ):
        super().__init__()

        self.architecture = dcunet_architecture
        self.fix_length_mode = (dcunet_fix_length if dcunet_fix_length != "none" else None)
        self.norm_type = dcunet_norm_type
        self.activation = dcunet_activation
        self.input_channels = 2  # for x_t and y -- note that this is 2 rather than 4, because we directly treat complex channels in this DNN
        self.time_embedding = (dcunet_time_embedding if dcunet_time_embedding != "none" else None)
        self.time_embedding_complex = dcunet_time_embedding_complex
        self.temb_layers_global = dcunet_temb_layers_global
        self.temb_layers_local = dcunet_temb_layers_local
        self.temb_activation = dcunet_temb_activation
        conf_encoders, conf_decoders = DCUNET_ARCHITECTURES[dcunet_architecture]

        # Replace `input_channels` in encoders config
        _replaced_input_channels, *rest = conf_encoders[0]
        encoders = ((self.input_channels, *rest), *conf_encoders[1:])
        decoders = conf_decoders
        self.encoders_stride_product = np.prod(
            [enc_stride for _, _, _, enc_stride, _, _ in encoders], axis=0
        )

        # Prepare kwargs for encoder and decoder (to potentially be modified before layer instantiation)
        encoder_decoder_kwargs = dict(
            norm_type=self.norm_type, activation=self.activation,
            temb_layers=self.temb_layers_local, temb_activation=self.temb_activation)

        # Instantiate (global) time embedding layer
        embed_ops = []
        if self.time_embedding is not None:
            complex_valued = self.time_embedding_complex
            if self.time_embedding == "gfp":
                embed_ops += [GaussianFourierProjection(embed_dim=embed_dim, complex_valued=complex_valued)]
                encoder_decoder_kwargs["embed_dim"] = embed_dim
            elif self.time_embedding == "ds":
                embed_ops += [DiffusionStepEmbedding(embed_dim=embed_dim, complex_valued=complex_valued)]
                encoder_decoder_kwargs["embed_dim"] = embed_dim

            if self.time_embedding_complex:
                assert self.time_embedding in ("gfp", "ds"), "Complex timestep embedding only available for gfp and ds"
                encoder_decoder_kwargs["complex_time_embedding"] = True
            for _ in range(self.temb_layers_global):
                embed_ops += [
                    ComplexLinear(embed_dim, embed_dim, complex_valued=True),
                    OnReIm(get_activation(dcunet_temb_activation))
                ]
        self.embed = nn.Sequential(*embed_ops)

        ### Instantiate DCUNet layers ###
        output_layer = ComplexConvTranspose2d(*decoders[-1])
        encoders = [DCUNetComplexEncoderBlock(*args, **encoder_decoder_kwargs) for args in encoders]
        decoders = [DCUNetComplexDecoderBlock(*args, **encoder_decoder_kwargs) for args in decoders[:-1]]

        self.mask_bound = (dcunet_mask_bound if dcunet_mask_bound != "none" else None)
        if self.mask_bound is not None:
            raise NotImplementedError("sorry, mask bounding not implemented at the moment")
            # TODO we can't use nn.Sequential since the ComplexConvTranspose2d needs a second `output_size` argument
        #operations = (output_layer, complex_nn.BoundComplexMask(self.mask_bound))
        #output_layer = nn.Sequential(*[x for x in operations if x is not None])

        assert len(encoders) == len(decoders) + 1
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.output_layer = output_layer or nn.Identity()

    def forward(self, spec, t) -> Tensor:
        """
        Input shape is expected to be $(batch, nfreqs, time)$, with $nfreqs - 1$ divisible
        by $f_0 * f_1 * ... * f_N$ where $f_k$ are the frequency strides of the encoders,
        and $time - 1$ is divisible by $t_0 * t_1 * ... * t_N$ where $t_N$ are the time
        strides of the encoders.
        Args:
            spec (Tensor): complex spectrogram tensor. 1D, 2D or 3D tensor, time last.
        Returns:
            Tensor, of shape (batch, time) or (time).
        """
        # TF-rep shape: (batch, self.input_channels, n_fft, frames)
        # Estimate mask from time-frequency representation.
        x_in = self.fix_input_dims(spec)
        x = x_in
        t_embed = self.embed(t+0j) if self.time_embedding is not None else None

        enc_outs = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x, t_embed)
            # UNet skip connection
            enc_outs.append(x)
        for (enc_out, dec) in zip(reversed(enc_outs[:-1]), self.decoders):
            x = dec(x, t_embed, output_size=enc_out.shape)
            x = torch.cat([x, enc_out], dim=1)

        output = self.output_layer(x, output_size=x_in.shape)
        # output shape: (batch, 1, n_fft, frames)
        output = self.fix_output_dims(output, spec)
        return output

    def fix_input_dims(self, x):
        return _fix_dcu_input_dims(
            self.fix_length_mode, x, torch.from_numpy(self.encoders_stride_product)
        )

    def fix_output_dims(self, out, x):
        return _fix_dcu_output_dims(self.fix_length_mode, out, x)


def _fix_dcu_input_dims(fix_length_mode, x, encoders_stride_product):
    """Pad or trim `x` to a length compatible with DCUNet."""
    freq_prod = int(encoders_stride_product[0])
    time_prod = int(encoders_stride_product[1])
    if (x.shape[2] - 1) % freq_prod:
        raise TypeError(
            f"Input shape must be [batch, ch, freq + 1, time + 1] with freq divisible by "
            f"{freq_prod}, got {x.shape} instead"
        )
    time_remainder = (x.shape[3] - 1) % time_prod
    if time_remainder:
        if fix_length_mode is None:
            raise TypeError(
                f"Input shape must be [batch, ch, freq + 1, time + 1] with time divisible by "
                f"{time_prod}, got {x.shape} instead. Set the 'fix_length_mode' argument "
                f"in 'DCUNet' to 'pad' or 'trim' to fix shapes automatically."
            )
        elif fix_length_mode == "pad":
            pad_shape = [0, time_prod - time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        elif fix_length_mode == "trim":
            pad_shape = [0, -time_remainder]
            x = nn.functional.pad(x, pad_shape, mode="constant")
        else:
            raise ValueError(f"Unknown fix_length mode '{fix_length_mode}'")
    return x


def _fix_dcu_output_dims(fix_length_mode, out, x):
    """Fix shape of `out` to the original shape of `x` by padding/cropping."""
    inp_len = x.shape[-1]
    output_len = out.shape[-1]
    return nn.functional.pad(out, [0, inp_len - output_len])


def _get_norm(norm_type):
    if norm_type == "CbN":
        return ComplexBatchNorm
    elif norm_type == "bN":
        return partial(OnReIm, BatchNorm)
    else:
        raise NotImplementedError(f"Unknown norm type: {norm_type}")


class DCUNetComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="bN",
        activation="leaky_relu",
        embed_dim=None,
        complex_time_embedding=False,
        temb_layers=1,
        temb_activation="silu"
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.temb_layers = temb_layers
        self.temb_activation = temb_activation
        self.complex_time_embedding = complex_time_embedding

        self.conv = ComplexConv2d(
            in_chan, out_chan, kernel_size, stride, padding, bias=norm_type is None, dilation=dilation
        )
        self.norm = _get_norm(norm_type)(out_chan)
        self.activation = OnReIm(get_activation(activation))
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            ops = []
            for _ in range(max(0, self.temb_layers - 1)):
                ops += [
                    ComplexLinear(self.embed_dim, self.embed_dim, complex_valued=True),
                    OnReIm(get_activation(self.temb_activation))
                ]
            ops += [
                FeatureMapDense(self.embed_dim, self.out_chan, complex_valued=True),
                OnReIm(get_activation(self.temb_activation))
            ]
            self.embed_layer = nn.Sequential(*ops)

    def forward(self, x, t_embed):
        y = self.conv(x)
        if self.embed_dim is not None:
            y = y + self.embed_layer(t_embed)
        return self.activation(self.norm(y))


class DCUNetComplexDecoderBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding=(0, 0),
        norm_type="bN",
        activation="leaky_relu",
        embed_dim=None,
        temb_layers=1,
        temb_activation='swish',
        complex_time_embedding=False,
    ):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.complex_time_embedding = complex_time_embedding
        self.temb_layers = temb_layers
        self.temb_activation = temb_activation

        self.deconv = ComplexConvTranspose2d(
            in_chan, out_chan, kernel_size, stride, padding, output_padding, dilation=dilation, bias=norm_type is None
        )
        self.norm = _get_norm(norm_type)(out_chan)
        self.activation = OnReIm(get_activation(activation))
        self.embed_dim = embed_dim
        if self.embed_dim is not None:
            ops = []
            for _ in range(max(0, self.temb_layers - 1)):
                ops += [
                    ComplexLinear(self.embed_dim, self.embed_dim, complex_valued=True),
                    OnReIm(get_activation(self.temb_activation))
                ]
            ops += [
                FeatureMapDense(self.embed_dim, self.out_chan, complex_valued=True),
                OnReIm(get_activation(self.temb_activation))
            ]
            self.embed_layer = nn.Sequential(*ops)

    def forward(self, x, t_embed, output_size=None):
        y = self.deconv(x, output_size=output_size)
        if self.embed_dim is not None:
            y = y + self.embed_layer(t_embed)
        return self.activation(self.norm(y))


# From https://github.com/chanil1218/DCUnet.pytorch/blob/2dcdd30804be47a866fde6435cbb7e2f81585213/models/layers/complexnn.py
class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
        super(ComplexBatchNorm, self).__init__()
        self.num_features        = num_features
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(num_features))
            self.register_buffer('RMi',  torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones (num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones (num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, x):
        xr, xi = x.real, x.imag
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return torch.view_as_complex(torch.stack([yr, yi], dim=-1))

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)
