"""Utilities for discrete token models


Authors
 * Artem Ploujnikov 2023
"""
import torch


def get_silence_token(model, sample_length=100000, device=None):
    """Attempts to find out the silence tokens for a given model,
    if applicable

    Arguments
    ---------
    model: nn.Module
        a discrete token model, taking (wav, lengths) as arguments

    Returns
    -------
    silence_tokens : torch.Tensor
        the token(s) corresponding to silence

    silece_emb : torch.Tensor
        the embedding(s) corresponding to silence

    """
    if device is None:
        device = next(model.parameters()).device

    audio = torch.zeros(1, sample_length, device=device)
    length = torch.ones(1, device=device)
    tokens, _ = model(audio, length)
    silence_tokens = tokens.squeeze(0).mode(0).values
    silence_emb = model.embeddings(silence_tokens[None, None, :]).squeeze()
    return silence_tokens, silence_emb
