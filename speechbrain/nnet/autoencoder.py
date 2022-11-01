"""Autoencoder implementation. Can be used for Latent Diffusion or in isolation

Authors
 * Artem Ploujnikov 2022
"""

import torch
from torch import nn
from collections import namedtuple
from speechbrain.dataio.dataio import clean_padding
from speechbrain.utils.data_utils import trim_as


class Autoencoder(nn.Module):
    """A standard interface for autoencoders"""
    
    def encode(self, x):
        """Converts a sample from an original space (e.g. pixel or waveform) to a latent
        space
        
        Arguments
        ---------
        x: torch.Tensor
            the original data representation
            
        Returns
        -------
        latent: torch.Tensor
            the latent representation
        """
        raise NotImplementedError

    def decode(self, latent):
        """Decodes the sample from a latent repsresentation
        
        Arguments
        ---------
        latent: torch.Tensor
            the latent representation

        Returns
        -------
        result: torch.Tensor
            the decoded sample
        """
        raise NotImplementedError

    def forward(self, x):
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

    mask_value: float
        The value with which outputs and latents will be masked

    len_dim: None
        the length dimension
    
    mask_latent: bool
        where to apply the length mask to the latent representation

    mask_out: bool
        whether to apply the length mask to the output
    
    out_mask_value: float
        the mask value used for the output
    
    latent_mask_value: float
        the mask value used for the latent representation

    """
    def __init__(
        self,
        encoder,
        decoder,
        mean,
        log_var,
        len_dim=1,
        mask_latent=True,
        mask_out=True,
        out_mask_value=0.,
        latent_mask_value=0.
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder        
        self.mean = mean
        self.log_var = log_var
        self.len_dim = len_dim
        self.mask_latent = mask_latent
        self.mask_out = mask_out
        self.out_mask_value = out_mask_value
        self.latent_mask_value = latent_mask_value

    def encode(self, x):
        """Converts a sample from an original space (e.g. pixel or waveform) to a latent
        space
        
        Arguments
        ---------
        x: torch.Tensor
            the original data representation
            
        Returns
        -------
        latent: torch.Tensor
            the latent representation
        """
        encoder_out = self.encoder(x)
        return self.mean(encoder_out)

    def decode(self, latent):
        """Decodes the sample from a latent repsresentation
        
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
            a latent space sample"""
        epsilon = torch.randn_like(log_var)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def train_sample(self, x, length=None, out_mask_value=None, latent_mask_value=None):
        """Provides a data sample for training the autoencoder

        Arguments
        ---------
        x: torch.Tensor
            the source data (in the sample space)
        length: None
            the length (optional). If provided, latents and
            outputs will be masked

        
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
        encoder_out, _ = self.encoder(x)
        mean = self.mean(encoder_out)
        log_var = self.log_var(encoder_out)
        latent_sample = self.reparameterize(mean, log_var)
        if self.mask_latent and length is not None:
            latent_sample = clean_padding(latent_sample, length, self.len_dim, latent_mask_value)
        x_rec = self.decode(latent_sample)
        x_rec = trim_as(x_rec, x)
        if self.mask_out and length is not None:
            x_rec = clean_padding(x_rec, length, self.len_dim, out_mask_value)
        
        return VariationalAutoencoderOutput(x_rec, latent_sample, mean, log_var)


VariationalAutoencoderOutput = namedtuple("VariationalAutoencoderOutput", ["rec", "latent", "mean", "log_var"])