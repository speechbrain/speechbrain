"""
All the code comes from: https://github.com/allenai/longformer and some change were done to fit SpeechBrain's architecture.

    Longformer is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).
    AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and
    engineering.

    The Longformer paper:
        @article{
        Beltagy2020Longformer,
        title={Longformer: The Long-Document Transformer},
        author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
        journal={arXiv:2004.05150},
        year={2020}
        }

    Parts of the code found herein were modified by: Jonathan Tremblay in order to fit SpeechBrain's interface.
"""

from typing import Union
from functools import lru_cache
import torch


def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device="cpu", dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)


@lru_cache()
def _get_invalid_locations_mask(
    w: int, d: Union[torch.Tensor, int], autoregressive: bool, device: str
):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(
            affected_seq_len, w, d
        )
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(
                affected_seq_len, w, d
            )
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = (
        None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    )
    return affected_seq_len, mask.bool().to(device), ending_mask


def mask_invalid_locations(
    input_tensor: torch.Tensor,
    w: int,
    d: Union[torch.Tensor, int],
    autoregressive: bool,
) -> torch.Tensor:
    affected_seq_len, beginning_mask, ending_mask = _get_invalid_locations_mask(
        w, d, autoregressive, input_tensor.device
    )
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :affected_seq_len, :, : w + 1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float("inf"))
    if not autoregressive:
        ending_input = input_tensor[:, -affected_seq_len:, :, -(w + 1) :]
        ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float("inf"))
