"""Utilities for discrete audio token models


Authors
 * Artem Ploujnikov 2023
"""

from functools import partial

import torch

from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.data_utils import batch_pad_right


def get_silence_token(
    model,
    sample_length=100000,
    extract_emb=True,
    device=None,
    model_kwargs=None,
):
    """Attempts to find out the silence tokens for a given model,
    if applicable

    Arguments
    ---------
    model : nn.Module
        A discrete token model, taking (wav, lengths) as arguments
    sample_length : int
        The length of the sample
    extract_emb : bool
        Whether to extract embeddings
    device : str | torch.Device
        The device to use
    model_kwargs : dict
        Additional arguments to pass to the model

    Returns
    -------
    silence_tokens : torch.Tensor
        The token(s) corresponding to silence

    silece_emb : torch.Tensor
        The embedding(s) corresponding to silence

    """
    if device is None:
        device = next(model.parameters()).device
    if model_kwargs is None:
        model_kwargs = {}

    audio = torch.zeros(1, sample_length, device=device)
    length = torch.ones(1, device=device)
    result = model(audio, length, **model_kwargs)
    tokens = result[0]
    silence_tokens = tokens.squeeze(0).mode(0).values
    silence_emb = None
    if extract_emb:
        if hasattr(model, "embeddings"):
            silence_emb = model.embeddings(
                silence_tokens[None, None, :]
            ).squeeze()
        else:
            heads = tokens.shape[-1]
            embs = result[1]
            mode_idx = [
                (tokens[0, :, head] == silence_tokens[head]).nonzero()[0].item()
                for head in range(heads)
            ]
            silence_emb = torch.stack(
                [embs[0, idx, head] for head, idx in enumerate(mode_idx)]
            )
    return silence_tokens, silence_emb


def feature_pad_to(tensor, length, padding=None):
    """Pads feature dimensions to the specified length with the specified padding,
    assuming a (Batch x Length x Features..) tensor

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be padded

    length : int
        The length to which the tensor will be padded

    padding : torch.Tensor, optional
        The padding tensor - if omitted, zero padding
        will be used

    Returns
    -------
    result : torch.Tensor
        The padded tensor
    """
    if padding is None:
        padding = torch.zeros(tensor.shape[1:])
    padding = padding[None, ...].expand(
        (length - tensor.size(0),) + tensor.shape[1:]
    )
    return torch.cat([tensor, padding], dim=0)


def batch_feature_pad(tensors, padding=None):
    """Similar to batch_pad_right but pads with the specified padding, which
    can be a vector or a tensor

    Arguments
    ---------
    tensors : list
        The list of tensors to be padded
    padding : torch.Tensor
        The padding tensor

    Returns
    -------
    result : torch.Tensor
        the padded tensor
    """
    lengths_abs = torch.tensor(
        [len(item) for item in tensors], device=tensors[0].device
    )
    max_length = lengths_abs.max()
    data = torch.stack(
        [feature_pad_to(item, max_length, padding) for item in tensors]
    )
    lengths = lengths_abs / max_length
    return data, lengths


def token_collate_fn(examples, silence_token, token_keys):
    """A customized collation function for audio tokens where
    the specified silence token will be used as padding - instead of
    zeros

    Arguments
    ---------
    examples : list
        A list of examples

    silence_token : torch.Tensor
        The token(s) representing silence

    token_keys : list
        The list of keys to which special padding will be applied

    Returns
    -------
    result : speechbrain.dataio.batch.PaddedBatch
        A padded batch
    """
    token_tensor_ids = {id(examples[0][key]) for key in token_keys}
    return PaddedBatch(
        examples,
        padding_func=_silence_padding,
        padding_kwargs={
            "silence_token": silence_token,
            "token_tensor_ids": token_tensor_ids,
        },
    )


def _silence_padding(values, silence_token, token_tensor_ids):
    """Computes silence padding for the specified tensors only.

    Arguments
    ---------
    values: torch.Tensor
        Audio features
    silence_token : torch.Tensor
        The silence token
    token_tensor_ids : enumerable
        The identifiers of token tensors to pad. This is
        needed because in PaddedBatch, the padding function
        is applied indiscriminately to all tensors

    Returns
    -------
    result : torch.Tensor
        The padded tensor
    """
    return (
        batch_feature_pad(values, silence_token)
        if id(values[0]) in token_tensor_ids
        else batch_pad_right(values)
    )


def use_silence_padding(dataloader_opts, silence_token, token_keys):
    """Overrides the collation function to add silence padding to
    audio token features

    Arguments
    ---------
    dataloader_opts : dict
        Dataloader options
    silence_token : torch.Tensor
        The tensor to be used as silence padding
    token_keys : torch.Tensor
        The keys to apply silence padding to

    Returns
    -------
    dataloader_opts : dict
        Updated data loader options
    """
    return {
        **dataloader_opts,
        "collate_fn": partial(
            token_collate_fn, silence_token=silence_token, token_keys=token_keys
        ),
    }
