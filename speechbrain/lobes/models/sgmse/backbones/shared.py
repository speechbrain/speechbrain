import functools
import numpy as np

import torch
import torch.nn as nn

from ..util.registry import Registry


BackboneRegistry = Registry("Backbone")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=16, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos of the features to avoid ambiguities.
            # Therefore, in this case the effective embed_dim is cut in half. For the complex-valued case,
            # we use complex numbers which each represent sin+cos directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2*np.pi
        if self.complex_valued:
            return torch.exp(1j * t_proj)
        else:
            return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class DiffusionStepEmbedding(nn.Module):
    """Diffusion-Step embedding as in DiffWave / Vaswani et al. 2017."""

    def __init__(self, embed_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos of the features to avoid ambiguities.
            # Therefore, in this case the effective embed_dim is cut in half. For the complex-valued case,
            # we use complex numbers which each represent sin+cos directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        self.embed_dim = embed_dim

    def forward(self, t):
        fac = 10**(4*torch.arange(self.embed_dim, device=t.device) / (self.embed_dim-1))
        inner = t[:, None] * fac[None, :]
        if self.complex_valued:
            return torch.exp(1j * inner)
        else:
            return torch.cat([torch.sin(inner), torch.cos(inner)], dim=-1)


class ComplexLinear(nn.Module):
    """A potentially complex-valued linear layer. Reduces to a regular linear layer if `complex_valued=False`."""
    def __init__(self, input_dim, output_dim, complex_valued):
        super().__init__()
        self.complex_valued = complex_valued
        if self.complex_valued:
            self.re = nn.Linear(input_dim, output_dim)
            self.im = nn.Linear(input_dim, output_dim)
        else:
            self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.complex_valued:
            return (self.re(x.real) - self.im(x.imag)) + 1j*(self.re(x.imag) + self.im(x.real))
        else:
            return self.lin(x)


class FeatureMapDense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        self.dense = ComplexLinear(input_dim, output_dim, complex_valued=complex_valued)

    def forward(self, x):
        return self.dense(x)[..., None, None]


def torch_complex_from_reim(re, im):
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


class ArgsComplexMultiplicationWrapper(nn.Module):
    """Adapted from `asteroid`'s `complex_nn.py`, allowing args/kwargs to be passed through forward().

    Make a complex-valued module `F` from a real-valued module `f` by applying
    complex multiplication rules:

    F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

    where `f1`, `f2` are instances of `f` that do *not* share weights.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Constructor of `f` in the formula above.  Called 2x with `*args`, `**kwargs`,
            to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return torch_complex_from_reim(
            self.re_module(x.real, *args, **kwargs) - self.im_module(x.imag, *args, **kwargs),
            self.re_module(x.imag, *args, **kwargs) + self.im_module(x.real, *args, **kwargs),
        )


ComplexConv2d = functools.partial(ArgsComplexMultiplicationWrapper, nn.Conv2d)
ComplexConvTranspose2d = functools.partial(ArgsComplexMultiplicationWrapper, nn.ConvTranspose2d)