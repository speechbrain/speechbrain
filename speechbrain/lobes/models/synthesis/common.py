"""Common functions

Authors
* Artem Ploujnikov 2021
"""

import torch
from torch import nn

class GuidedAttentionLoss(nn.Module):
    """
    A loss implementation that forces attention matrices to be
    near-diagonal, imposing progressively larger penalties for paying
    attention to regions far away from the diagonal). It is useful
    for sequence-to-sequence models in which the sequence of outputs
    is expected to corrsespond closely to the sequence of inputs,
    such as TTS or G2P

    https://arxiv.org/abs/1710.08969

    Arguments
    ---------
    sigma:
        the guided attention weight
    """
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.weight_factor = 2 * (sigma ** 2)

    def forward(self, attention, input_lengths, target_lengths):
        """
        Computes the guided attention loss for a single batch

        Arguments
        ---------
        attention: torch.Tensor
            A padded attention/alignments matrix
            (batch, inputs, outputs)
        input_lengths: torch.tensor
            A (batch, lengths) tensor of input lengths

        Returns
        -------
        loss: torch.Tensor
            A single-element tensor with the loss value
        """
        soft_mask = self.guided_attentions(input_lengths, target_lengths)
        return (attention * soft_mask).mean()

    def guided_attention(self, input_length, target_length):
        """
        Computes a single guided attention matrix for a single sample

        Arguments
        ---------
        input_length: int
            The input length
        target_length: int
            The target length

        Returns
        -------
        value: torch.Tensor
            The guided attention tensor for a single example
        """
        n, t = torch.meshgrid(
            torch.arange(input_length).to(input_length.device),
            torch.arange(target_length).to(input_length.device)
        )
        value = 1.0 - torch.exp(
            -((n / input_length - t / target_length) ** 2)
            / self.weight_factor)
        return value

    def guided_attentions(
        self, input_lengths, target_lengths, max_target_len=None
    ):
        """
        Computes guided attention matrices

        Arguments
        ---------
        input_lengths: torch.Tensor
            A tensor of input lengths
        target_lengths: torch.Tensor
            A tensor of target (spectrogram) length
        max_target_len: int
            The maximum target length


        Returns
        -------
        soft_mask: torch.Tensor
            The guided attention tensor
        """
        batch_size = len(input_lengths)
        max_input_len = input_lengths.max()
        if max_target_len is None:
            max_target_len = target_lengths.max().item()
        soft_mask = (
            torch.zeros((batch_size, max_target_len, max_input_len.item()))
            .float()
            .to(input_lengths.device)
        )
        # TODO: Attempt to vectorize here as well
        for b in range(batch_size):
            attention = self.guided_attention(
                input_lengths[b],
                target_lengths[b]
            ).T
            soft_mask[b, : attention.size(0), : attention.size(1)] = attention
        return soft_mask