"""
Speech quality assessment models based on self-supervised learning (SSL)
model finetuning

Authors
 * Artem Ploujnikov 2023
 * Yingzi Wang 2024
"""

import torch
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
)
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.pooling import StatisticsPooling


class BaselineSSLFinetune(nn.Module):
    """A baseline self-supervised learning representation fine-tuning
    model, inspired by the following:

    https://github.com/nii-yamagishilab/mos-finetune-ssl

    Arguments
    ---------
    base_model : torch.nn.Module
        The base model to be used

    feats_dim : int, optional
        The feature dimension. If omitted, it will be computed automatically

    Example
    -------
    >>> from speechbrain.lobes.models.eval.ssl import BaselineSSLFinetune
    >>> from speechbrain.nnet.linear import Linear
    >>> from torch import nn
    >>> import torch
    >>> class FakeBaseModel(nn.Module):
    ...     def __init__(self, output_size):
    ...         super().__init__()
    ...         self.lin = Linear(
    ...             input_size=1,
    ...             n_neurons=output_size
    ...         )
    ...     def forward(self, x, length):
    ...         return self.lin(x.unsqueeze(-1))
    >>> fake_base_model = FakeBaseModel(128)
    >>> model = BaselineSSLFinetune(
    ...     base_model=fake_base_model
    ... )
    >>> x = torch.randn(4, 100)
    >>> length = torch.ones(4)
    >>> scores = model(x, length)
    >>> scores.shape
    torch.Size([4, 1, 1])
    """

    def __init__(self, base_model, feats_dim=None):
        super().__init__()
        self.base_model = base_model
        if feats_dim is None:
            feats_dim = compute_feats_dim(base_model)
        self.feats_dim = feats_dim
        self.pool = StatisticsPooling(return_std=False)
        self.out = Linear(n_neurons=1, input_size=feats_dim)

    def forward(self, wav, length):
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The waveform (in the format understood by the base model)
            Typically (Batch x Time) or (Batch x Channel x Time)
        length : torch.Tensor
            A 1-D tensor of relative lengths

        Returns
        -------
        result : torch.Tensor
            a 1-D tensor with an estimated speech quality rating
            (the scale used depends on the training data)
        """
        x = self.base_model(wav, length)
        x = self.pool(x, length)
        x = self.out(x)
        return x


class TransformerRegression(nn.Module):
    """A simple extension of the SSL fine-tuning model that adds a non-autoregressive
    transformer layer on top of SSL representation followed by average pooling. The
    idea is to train a new model for the evaluation task instead of - or in addition to
    - attempting to update the weights of the base model

    Arguments
    ---------
    base_model : torch.nn.Module
        The base model converting an audio/speech signal to a latent representation
    feats_dim : int, optional
        The feature dimension. If omitted, it will be computed automatically
    d_model : int, optional
        The transformer model dimension
    d_ffn : int, optional
        The transformer feed-forward network dimension
    num_layers : int, optional
        The number of transformer layers
    nhead : int, optional
        The number of transformer heads
    activation : torch.nn.Module, optional
        The type of activation to use (defaults to LeakyRELU)
    dropout : float, optional
        The dropout probability
    max_len : int
        The maximum sequence length

    Example
    -------
    >>> from speechbrain.lobes.models.eval.ssl import TransformerRegression
    >>> from speechbrain.nnet.linear import Linear
    >>> from torch import nn
    >>> import torch
    >>> class FakeBaseModel(nn.Module):
    ...     def __init__(self, output_size):
    ...         super().__init__()
    ...         self.lin = Linear(
    ...             input_size=1,
    ...             n_neurons=output_size
    ...         )
    ...     def forward(self, x, length):
    ...         return self.lin(x.unsqueeze(-1))
    >>> fake_base_model = FakeBaseModel(128)
    >>> model = TransformerRegression(
    ...     base_model=fake_base_model
    ... )
    >>> x = torch.randn(4, 100)
    >>> length = torch.ones(4)
    >>> scores = model(x, length)
    >>> scores.shape
    torch.Size([4, 1, 1])
    """

    def __init__(
        self,
        base_model,
        feats_dim=None,
        d_model=512,
        d_ffn=2048,
        num_layers=3,
        nhead=4,
        activation=None,
        dropout=0.2,
        max_len=2500,
    ):
        super().__init__()
        self.base_model = base_model

        if activation is None:
            activation = nn.LeakyReLU

        if feats_dim is None:
            feats_dim = compute_feats_dim(base_model)
        self.feats_norm = BatchNorm1d(input_size=feats_dim)
        self.feat_proj = Linear(n_neurons=d_model, input_size=feats_dim)
        self.pos_emb = PositionalEncoding(input_size=d_model, max_len=max_len)

        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_model=d_model,
            d_ffn=d_ffn,
            activation=nn.LeakyReLU,
            dropout=dropout,
            normalize_before=True,
        )
        self.pool = StatisticsPooling(return_std=False)
        self.out_proj = Linear(n_neurons=1, input_size=d_model)

    def forward(self, wav, length):
        """Computes the forward pass

        Arguments
        ---------
        wav : torch.Tensor
            The waveform (in the format understood by the base model)
            Typically (Batch x Time) or (Batch x Channel x Time)
        length : torch.Tensor
            A 1-D tensor of relative lengths

        Returns
        -------
        result : torch.Tensor
            a 1-D tensor with an estimated speech quality rating
            (the scale used depends on the training data)
        """
        x = self.base_model(wav, length)
        x = self.feats_norm(x)
        pos_emb = self.pos_emb(x)
        x = self.feat_proj(x) + pos_emb
        abs_len = torch.round(length * x.shape[1])
        src_key_padding_mask = ~length_to_mask(abs_len).bool()
        x, _ = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pool(x)
        x = self.out_proj(x)
        return x


def compute_feats_dim(model):
    """Computes the feature dimension by feeding a fake tensor to the model

    Arguments
    ---------
    model : torch.nn.Module
        A model that takes audio input
    """
    device = next(model.parameters()).device
    wav = torch.randn(1, 1000, device=device)
    length = torch.tensor([1.0], device=device)
    out = model(wav, length)
    return out.size(-1)
