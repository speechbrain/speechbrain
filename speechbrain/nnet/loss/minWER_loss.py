"""
minWER loss implementation
and based on https://arxiv.org/pdf/1712.01818.pdf

Authors
 * Sung-Lin Yeh 2020
 * Abdelwahab Heba 2020
"""
import torch
import pytest

pytest.importorskip("torch_edit_distance_cuda")

try:
    import torch_edit_distance_cuda as core
except ImportError:
    err_msg = "The optional dependency pytorch-edit-distance is needed to use the minWER loss\n"
    err_msg += "cannot import torch_edit_distance_cuda. To use minWER loss\n"
    err_msg += "Please follow the instructions below or visit\n"
    err_msg += "https://github.com/1ytic/pytorch-edit-distance\n"
    err_msg += "================ Export UNIX var + Install =============\n"
    err_msg += "CUDAVER=cuda"
    err_msg += "export PATH=/usr/local/$CUDAVER/bin:$PATH"
    err_msg += "export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH"
    err_msg += (
        "export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH"
    )
    err_msg += "export CUDA_PATH=/usr/local/$CUDAVER"
    err_msg += "export CUDA_ROOT=/usr/local/$CUDAVER"
    err_msg += "export CUDA_HOME=/usr/local/$CUDAVER"
    err_msg += "export CUDA_HOST_COMPILER=/usr/bin/gcc-7"
    err_msg += "pip install torch_edit_distance"
    err_msg += ""
    raise ImportError(err_msg)


def minWER_loss(
    hypotheses,
    targets,
    hyps_lens,
    target_lens,
    hypotheses_scores,
    blank_index,
    separator_index=None,
    mode="Num_Word_Errors",
):
    """
    Compute minWER loss using torch_edit_distance.
    This implementation is based on the paper: https://arxiv.org/pdf/1712.01818.pdf (see section 3)
    We use levenshtein distance function from torch_edit_distance lib
    which allow us to compute W(y,y_hat) -> the number of word errors in a hypothesis.
    instead of the WER metric.

    Arguments
    ---------
    hypotheses : torch.Tensor
        Tensor (B, N, H) where H is the maximum
        length of tokens from N hypotheses each batch (B utt).
    targets : torch.Tensor
        Tensor (B, R) where R is the maximum
        length of tokens for each reference in batch (B utt).
    hyps_lens : torch.Tensor
        Tensor (B, N) representing the
        number of tokens for each hypothesis in batch (B utt).
    target_lens : torch.Tensor
        Tensor (B,) representing the
        number of tokens for each reference in batch (B utt).
    hypotheses_scores : torch.Tensor
        Tensor (B, N) where N is the maximum
        length of hypotheses from batch.
    blank : int
        blank index.
    separator_index : default None,
        otherwise specify the space index.
    mode : str, default "Num_word_Errors"
        for using the number of word errors in a hypothesis.
        Otherwise "WER" for using WER metric.

    Returns
    -------
    torch.tensor
        minWER loss
    """
    batch_size = hypotheses_scores.size(0)
    topk = hypotheses_scores.size(1)
    blank_index = torch.tensor([blank_index], dtype=torch.int).to(
        hypotheses.device
    )
    space_token = [] if separator_index is None else [separator_index]
    separator_index = torch.tensor(space_token, dtype=torch.int).to(
        hypotheses.device
    )

    # levenshtein_distance tensor will have 4D dimensions
    # [ ins, del, sub, utt_len ] so we use the first 3 values
    levenshtein_distance = core.levenshtein_distance(
        hypotheses.view(batch_size * topk, -1),
        torch.repeat_interleave(targets.to(torch.int32), repeats=topk, dim=0),
        hyps_lens.view(-1),
        torch.repeat_interleave(
            target_lens.to(torch.int32), repeats=topk, dim=0
        ),
        blank_index,
        separator_index,
    )
    # compute the number of word errors for each hypothesis.
    wers = torch.sum(levenshtein_distance[:, :3], 1, dtype=torch.float32)
    # if WER, then normalize by utt length
    if mode == "WER":
        wers /= levenshtein_distance[:, 3]

    wers = wers.view(batch_size, topk)
    avg_wers = torch.mean(wers, -1).unsqueeze(1)
    relative_wers = wers - avg_wers
    # compute softmax for the Nbest scores
    hypotheses_scores = hypotheses_scores.log_softmax(dim=-1)
    mWER_loss = torch.sum(hypotheses_scores * relative_wers, -1)

    return mWER_loss
