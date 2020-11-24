"""
minWER loss implementation
and based on https://arxiv.org/pdf/1712.01818.pdf

Authors
 * Sung-Lin Yeh 2020
 * Abdelwahab Heba 2020
"""
import torch

try:
    import torch_edit_distance_cuda as core
except ImportError:
    err_msg = "The optional dependency pytorch-edit-distance is needed to use the minWER loss\n"
    err_msg += "cannot import torch_edit_distance_cuda. To use minWER loss\n"
    err_msg += "Please follow the instructions below or visit\n"
    err_msg += "https://github.com/1ytic/pytorch-edit-distance"
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
    hyps_length,
    target_lens,
    hypotheses_scores,
    blank,
    space=None,
):
    """
    Compute minWER loss using torch_edit_distance.
    This implementation is based on the paper: https://arxiv.org/pdf/1712.01818.pdf (see section 3)
    We use levenshtein distance function from torch_edit_distance lib
    which allow us to compute W(y,y_hat) -> the number of word errors in a hypothesis.
    instead of the WER metric.

    Arguments
    ---------
    hypotheses (torch.Tensor): Tensor (N, H) where H is the maximum
        length of tokens from N hypotheses.
    targets (torch.Tensor): Tensor (N, R) where R is the maximum
        length of tokens from N references.
    hyps_lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each hypothesis.
    target_lens (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each reference.
    hypotheses_scores (torch.Tensor): Tensor (B, N) where N is the maximum
        length of hypotheses from batch.
    blank (int): blank indice.
    separator: default None, otherwise space indice (int).
    """
    blank = torch.tensor([blank], dtype=torch.int).to(hypotheses.device)
    space_token = [] if space is None else [space]
    space = torch.tensor(space_token, dtype=torch.int).to(hypotheses.device)

    wers = core.levenshtein_distance(
        hypotheses, targets, hyps_length, target_lens, blank, space,
    )
    wers = wers.view(hypotheses_scores.size(0), hypotheses_scores.size(1))
    avg_wers = torch.mean(wers, -1).unsqueeze(1)
    relative_wers = wers - avg_wers
    mWER_loss = torch.sum(hypotheses_scores * relative_wers, -1)

    return mWER_loss.mean()
