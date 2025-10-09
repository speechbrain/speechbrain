"""Autoencoder implementation. Can be used for Latent Diffusion or in isolation

Authors
 * Artem Ploujnikov 2022
"""

from collections import namedtuple

import torch
from torch import nn

from speechbrain.dataio.dataio import clean_padding
from speechbrain.processing.features import GlobalNorm
from speechbrain.utils.data_utils import trim_as


class Autoencoder(nn.Module):
    """A standard interface for autoencoders

    Example
    -------
    >>> import torch
    >>> from torch import nn
    >>> from speechbrain.nnet.linear import Linear
    >>> class SimpleAutoencoder(Autoencoder):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.enc = Linear(n_neurons=16, input_size=128)
    ...         self.dec = Linear(n_neurons=128, input_size=16)
    ...
    ...     def encode(self, x, length=None):
    ...         return self.enc(x)
    ...
    ...     def decode(self, x, length=None):
    ...         return self.dec(x)
    >>> autoencoder = SimpleAutoencoder()
    >>> x = torch.randn(4, 10, 128)
    >>> x_enc = autoencoder.encode(x)
    >>> x_enc.shape
    torch.Size([4, 10, 16])
    >>> x_enc_fw = autoencoder(x)
    >>> x_enc_fw.shape
    torch.Size([4, 10, 16])
    >>> x_rec = autoencoder.decode(x_enc)
    >>> x_rec.shape
    torch.Size([4, 10, 128])
    """

    def encode(self, x, length=None):
        """Converts a sample from an original space (e.g. pixel or waveform) to a latent
        space

        Arguments
        ---------
        x: torch.Tensor
            the original data representation
        length: torch.Tensor
            a tensor of relative lengths
        """
        raise NotImplementedError

    def decode(self, latent):
        """Decodes the sample from a latent representation

        Arguments
        ---------
        latent: torch.Tensor
            the latent representation
        """
        raise NotImplementedError

    def forward(self, x):
        """Performs the forward pass

        Arguments
        ---------
        x: torch.Tensor
            the input tensor

        Returns
        -------
        result: torch.Tensor
            the result
        """
        return self.encode(x)


class VariationalAutoencoder(Autoencoder):
    """A Variational Autoencoder (VAE) implementation.

    Paper reference: https://arxiv.org/abs/1312.6114

    Arguments
    ---------
    encoder: torch.Module
        the encoder network
    decoder: torch.Module
        the decoder network
    mean: torch.Module
        the module that computes the mean
    log_var: torch.Module
        the module that computes the log variance
    len_dim: None
        the length dimension
    latent_padding: function
        the function to use when padding the latent variable
    mask_latent: bool
        where to apply the length mask to the latent representation
    mask_out: bool
        whether to apply the length mask to the output
    out_mask_value: float
        the mask value used for the output
    latent_mask_value: float
        the mask value used for the latent representation
    latent_stochastic: bool
        if true, the "latent" parameter of VariationalAutoencoderOutput
        will be the latent space sample
        if false, it will be the mean

    Example
    -------
    The example below shows a very simple implementation of
    VAE, not suitable for actual experiments:

    >>> import torch
    >>> from torch import nn
    >>> from speechbrain.nnet.linear import Linear
    >>> vae_enc = Linear(n_neurons=16, input_size=128)
    >>> vae_dec = Linear(n_neurons=128, input_size=16)
    >>> vae_mean = Linear(n_neurons=16, input_size=16)
    >>> vae_log_var = Linear(n_neurons=16, input_size=16)
    >>> vae = VariationalAutoencoder(
    ...     encoder=vae_enc,
    ...     decoder=vae_dec,
    ...     mean=vae_mean,
    ...     log_var=vae_log_var,
    ... )
    >>> x = torch.randn(4, 10, 128)

    `train_sample` encodes a single batch and then reconstructs
    it

    >>> vae_out = vae.train_sample(x)
    >>> vae_out.rec.shape
    torch.Size([4, 10, 128])
    >>> vae_out.latent.shape
    torch.Size([4, 10, 16])
    >>> vae_out.mean.shape
    torch.Size([4, 10, 16])
    >>> vae_out.log_var.shape
    torch.Size([4, 10, 16])
    >>> vae_out.latent_sample.shape
    torch.Size([4, 10, 16])

    .encode() will return the mean corresponding
    to the sample provided

    >>> x_enc = vae.encode(x)
    >>> x_enc.shape
    torch.Size([4, 10, 16])

    .reparameterize() performs the reparameterization
    trick

    >>> x_enc = vae.encoder(x)
    >>> mean = vae.mean(x_enc)
    >>> log_var = vae.log_var(x_enc)
    >>> x_repar = vae.reparameterize(mean, log_var)
    >>> x_repar.shape
    torch.Size([4, 10, 16])

    """

    def __init__(
        self,
        encoder,
        decoder,
        mean,
        log_var,
        len_dim=1,
        latent_padding=None,
        mask_latent=True,
        mask_out=True,
        out_mask_value=0.0,
        latent_mask_value=0.0,
        latent_stochastic=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mean = mean
        self.log_var = log_var
        self.len_dim = len_dim
        self.latent_padding = latent_padding
        self.mask_latent = mask_latent
        self.mask_out = mask_out
        self.out_mask_value = out_mask_value
        self.latent_mask_value = latent_mask_value
        self.latent_stochastic = latent_stochastic

    def encode(self, x, length=None):
        """Converts a sample from an original space (e.g. pixel or waveform) to a latent
        space

        Arguments
        ---------
        x: torch.Tensor
            the original data representation
        length: torch.Tensor
            the length of the corresponding input samples (optional)

        Returns
        -------
        latent: torch.Tensor
            the latent representation
        """
        encoder_out = self.encoder(x)
        return self.mean(encoder_out)

    def decode(self, latent):
        """Decodes the sample from a latent representation

        Arguments
        ---------
        latent: torch.Tensor
            the latent representation

        Returns
        -------
        result: torch.Tensor
            the decoded sample
        """
        return self.decoder(latent)

    def reparameterize(self, mean, log_var):
        """Applies the VAE reparameterization trick to get a latent space
        single latent space sample for decoding

        Arguments
        ---------
        mean: torch.Tensor
            the latent representation mean
        log_var: torch.Tensor
            the logarithm of the latent representation variance

        Returns
        -------
        sample: torch.Tensor
            a latent space sample
        """
        epsilon = torch.randn_like(log_var)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def train_sample(
        self, x, length=None, out_mask_value=None, latent_mask_value=None
    ):
        """Provides a data sample for training the autoencoder

        Arguments
        ---------
        x: torch.Tensor
            the source data (in the sample space)
        length: None
            the length (optional). If provided, latents and
            outputs will be masked
        out_mask_value: float
            the mask value used for the output
        latent_mask_value: float
            the mask value used for the latent tensor


        Returns
        -------
        result: VariationalAutoencoderOutput
            a named tuple with the following values
            rec: torch.Tensor
                the reconstruction
            latent: torch.Tensor
                the latent space sample
            mean: torch.Tensor
                the mean of the latent representation
            log_var: torch.Tensor
                the logarithm of the variance of the latent representation

        """
        if out_mask_value is None:
            out_mask_value = self.out_mask_value
        if latent_mask_value is None:
            latent_mask_value = self.latent_mask_value
        encoder_out = self.encoder(x)

        mean = self.mean(encoder_out)
        log_var = self.log_var(encoder_out)
        latent_sample = self.reparameterize(mean, log_var)
        if self.latent_padding is not None:
            latent_sample, latent_length = self.latent_padding(
                latent_sample, length=length
            )
        else:
            latent_length = length
        if self.mask_latent and length is not None:
            latent_sample = clean_padding(
                latent_sample, latent_length, self.len_dim, latent_mask_value
            )
        x_rec = self.decode(latent_sample)
        x_rec = trim_as(x_rec, x)
        if self.mask_out and length is not None:
            x_rec = clean_padding(x_rec, length, self.len_dim, out_mask_value)

        if self.latent_stochastic:
            latent = latent_sample
        else:
            latent, latent_length = self.latent_padding(mean, length=length)

        return VariationalAutoencoderOutput(
            x_rec, latent, mean, log_var, latent_sample, latent_length
        )


VariationalAutoencoderOutput = namedtuple(
    "VariationalAutoencoderOutput",
    ["rec", "latent", "mean", "log_var", "latent_sample", "latent_length"],
)

AutoencoderOutput = namedtuple(
    "AutoencoderOutput", ["rec", "latent", "latent_length"]
)


class NormalizingAutoencoder(Autoencoder):
    """A classical (non-variational) autoencoder that
    does not use reparameterization but instead uses
    an ordinary normalization technique to constrain
    the latent space

    Arguments
    ---------
    encoder: torch.nn.Module
        the encoder to be used
    decoder: torch.nn.Module
        the decoder to be used
    latent_padding: function
        Function to use when padding the latent tensor
    norm: torch.nn.Module
        the normalization module
    len_dim: int
        The time dimension, which the length applies to.
    mask_out: bool
        whether to apply the length mask to the output
    mask_latent: bool
        where to apply the length mask to the latent representation
    out_mask_value: float
        the mask value used for the output
    latent_mask_value: float
        the mask value used for the latent tensor

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from speechbrain.nnet.linear import Linear
    >>> ae_enc = Linear(n_neurons=16, input_size=128)
    >>> ae_dec = Linear(n_neurons=128, input_size=16)
    >>> ae = NormalizingAutoencoder(
    ...     encoder=ae_enc,
    ...     decoder=ae_dec,
    ... )
    >>> x = torch.randn(4, 10, 128)
    >>> x_enc = ae.encode(x)
    >>> x_enc.shape
    torch.Size([4, 10, 16])
    >>> x_dec = ae.decode(x_enc)
    >>> x_dec.shape
    torch.Size([4, 10, 128])
    """

    def __init__(
        self,
        encoder,
        decoder,
        latent_padding=None,
        norm=None,
        len_dim=1,
        mask_out=True,
        mask_latent=True,
        out_mask_value=0.0,
        latent_mask_value=0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_padding = latent_padding
        if norm is None:
            norm = GlobalNorm(length_dim=len_dim)
        self.norm = norm
        self.len_dim = len_dim
        self.mask_out = mask_out
        self.mask_latent = mask_latent
        self.out_mask_value = out_mask_value
        self.latent_mask_value = latent_mask_value

    def encode(self, x, length=None):
        """Converts a sample from an original space (e.g. pixel or waveform) to a latent
        space

        Arguments
        ---------
        x: torch.Tensor
            the original data representation
        length: torch.Tensor
            The length of each sample in the input tensor.

        Returns
        -------
        latent: torch.Tensor
            the latent representation
        """
        x = self.encoder(x)
        x = self.norm(x, lengths=length)
        return x

    def decode(self, latent):
        """Decodes the sample from a latent representation

        Arguments
        ---------
        latent: torch.Tensor
            the latent representation

        Returns
        -------
        result: torch.Tensor
            the decoded sample
        """
        return self.decoder(latent)

    def train_sample(
        self, x, length=None, out_mask_value=None, latent_mask_value=None
    ):
        """Provides a data sample for training the autoencoder

        Arguments
        ---------
        x: torch.Tensor
            the source data (in the sample space)
        length: torch.Tensor
            the length (optional). If provided, latents and
            outputs will be masked
        out_mask_value: float
            The value to use when masking the output.
        latent_mask_value: float
            The value to use when masking the latent tensor.

        Returns
        -------
        result: AutoencoderOutput
            a named tuple with the following values
            rec: torch.Tensor
                the reconstruction
            latent: torch.Tensor
                the latent space sample
        """
        if out_mask_value is None:
            out_mask_value = self.out_mask_value
        if latent_mask_value is None:
            latent_mask_value = self.latent_mask_value
        latent = self.encode(x, length=length)
        if self.latent_padding is not None:
            latent, latent_length = self.latent_padding(latent, length=length)
        else:
            latent_length = length
        if self.mask_latent and length is not None:
            latent = clean_padding(
                latent, latent_length, self.len_dim, latent_mask_value
            )
        x_rec = self.decode(latent)
        x_rec = trim_as(x_rec, x)
        if self.mask_out and length is not None:
            x_rec = clean_padding(x_rec, length, self.len_dim, out_mask_value)

        return AutoencoderOutput(x_rec, latent, latent_length)
