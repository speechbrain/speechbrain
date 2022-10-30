"""Autoencoder implementation. Can be used for Latent Diffusion or in isolation

Authors
 * Artem Ploujnikov 2022
"""

import torch
from torch import nn
from collections import namedtuple
from speechbrain.dataio.dataio import clean_padding_
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
    
    mask_latents: bool
        where to apply the length mask to the latent representation

    mask_output: bool
        whether to apply the length mask to the output
    """
    def __init__(self, encoder, decoder, mean, log_var, mask_value=0., len_dim=1, mask_latents=True, mask_output=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder        
        self.mean = mean
        self.log_var = log_var
        self.mask_value = mask_value
        self.len_dim = len_dim
        self.mask_latents = mask_latents
        self.mask_output = mask_output

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

    def train_sample(self, x, length=None):
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
        encoder_out, length = self.encoder(x)
        mean = self.mean(encoder_out)
        log_var = self.log_var(encoder_out)
        latent_sample = self.reparameterize(mean, log_var)
        if self.mask_latents and length is not None:
            clean_padding_(latent_sample, length, self.len_dim, self.mask_value)
        x_rec = self.decode(latent_sample)
        x_rec = trim_as(x_rec, x)
        if self.mask_output and length is not None:
            clean_padding_(latent_sample, length, self.len_dim, self.mask_value)
        
        return VariationalAutoencoderOutput(x_rec, latent_sample, mean, log_var)


VariationalAutoencoderOutput = namedtuple("VariationalAutoencoderOutput", ["rec", "latent", "mean", "log_var"])