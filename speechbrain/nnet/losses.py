"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
"""

import torch
from torch import nn
import logging
import functools
from speechbrain.data_io.data_io import length_to_mask
from itertools import permutations


logger = logging.getLogger(__name__)


def transducer_loss(log_probs, targets, input_lens, target_lens, blank_index):
    """Transducer loss, see `speechbrain/nnet/transducer/transducer_loss.py`

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    """
    from speechbrain.nnet.transducer.transducer_loss import Transducer

    input_lens = (input_lens * log_probs.shape[1]).int()
    target_lens = (target_lens * targets.shape[1]).int()
    return Transducer.apply(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        reduction="mean",
    )


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.
    Permutation invariance is calculated over sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat: torch.Tensor
            tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss: torch.Tensor
            permutation invariant loss for current batch, tensor of shape [1]

        assigned_perm: tuple
            indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """

        Parameters
        ----------
        pred: torch.Tensor
            network prediction for current example, tensor of
            shape [..., sources].
        target: torch.Tensor
            target for current example, tensor of shape [..., sources].

        Returns
        -------
        loss: torch.Tensor
            permutation invariant loss for current example, tensor of shape [1]

        assigned_perm: tuple
            indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)
        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
            Arguments
            ---------
            tensor : torch.Tensor
                tensor to reorder given the optimal permutation, of shape
                [batch, ..., sources].
            p : list of tuples
                list of optimal permutations, e.g. for batch=2 and n_sources=3
                [(0, 1, 2), (0, 2, 1].

            Returns
            -------
            reordered: torch.Tensor
                reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor).to(tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss: torch.Tensor
                permutation invariant loss for current examples, tensor of
                shape [batch]

            perms: list
                list of indexes for optimal permutation of the inputs over
                sources.
                e.g. [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def ctc_loss(log_probs, targets, input_lens, target_lens, blank_index):
    """CTC loss

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    """
    input_lens = (input_lens * log_probs.shape[1]).int()
    target_lens = (target_lens * targets.shape[1]).int()
    log_probs = log_probs.transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
    )


def l1_loss(predictions, targets, length=None, allowed_len_diff=3):
    """Compute the true l1 loss, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor, same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> l1_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.1000)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.l1_loss, reduction="none")
    return compute_masked_loss(loss, predictions, targets, length)


def mse_loss(predictions, targets, length=None, allowed_len_diff=3):
    """Compute the true mean squared error, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor, same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> mse_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.0100)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.mse_loss, reduction="none")
    return compute_masked_loss(loss, predictions, targets, length)


def classification_error(
    probabilities, targets, length=None, allowed_len_diff=3
):
    """Computes the classification error at frame or batch level.

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> classification_error(probs, torch.tensor([1, 1]))
    tensor(0.5000)
    """
    if len(probabilities.shape) == 3 and len(targets.shape) == 2:
        probabilities, targets = truncate(
            probabilities, targets, allowed_len_diff
        )

    def error(predictions, targets):
        predictions = torch.argmax(probabilities, dim=-1)
        return (predictions != targets).float()

    return compute_masked_loss(error, probabilities, targets.long(), length)


def nll_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> nll_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if len(log_probabilities.shape) == 3:
        log_probabilities, targets = truncate(
            log_probabilities, targets, allowed_len_diff
        )
        log_probabilities = log_probabilities.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(torch.nn.functional.nll_loss, reduction="none")
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
    )


def truncate(predictions, targets, allowed_len_diff=3):
    """Ensure that predictions and targets are the same length.

    Arguments
    ---------
    predictions : torch.Tensor
        First tensor for checking length.
    targets : torch.Tensor
        Second tensor for checking length.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    """
    len_diff = predictions.shape[1] - targets.shape[1]
    if len_diff == 0:
        return predictions, targets
    elif abs(len_diff) > allowed_len_diff:
        raise ValueError(
            "Predictions and targets should be same length, but got %s and "
            "%s respectively." % (predictions.shape[1], targets.shape[1])
        )
    elif len_diff < 0:
        return predictions, targets[:, : predictions.shape[1]]
    else:
        return predictions[:, : targets.shape[1]], targets


def compute_masked_loss(
    loss_fn, predictions, targets, length=None, label_smoothing=0.0
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none")
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output Distributions.
        https://arxiv.org/abs/1701.06548
    """
    mask = torch.ones_like(targets)
    if length is not None:
        mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )
        if len(targets.shape) == 3:
            mask = mask.unsqueeze(2).repeat(1, 1, targets.shape[2])

    loss = torch.sum(loss_fn(predictions, targets) * mask) / torch.sum(mask)
    if label_smoothing == 0:
        return loss
    else:
        loss_reg = -torch.sum(
            torch.mean(predictions, dim=1) * mask
        ) / torch.sum(mask)
        return label_smoothing * loss_reg + (1 - label_smoothing) * loss


def get_si_snr(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(
        source, estimate_source, source_lengths
    )
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(
        estimate_source, perms, max_snr_idx
    )
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(
        s_estimate * s_target, dim=3, keepdim=True
    )  # [B, C, C, 1]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
    )  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (
        torch.sum(e_noise ** 2, dim=3) + EPS
    )
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum("bij,pij->bp", [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i] :] = 0
    return mask
