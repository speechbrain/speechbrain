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


def transducer_loss(
    log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"
):
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
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.
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
        reduction=reduction,
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


def ctc_loss(
    log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"
):
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
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'batchmean' | 'sum'.
    """
    input_lens = (input_lens * log_probs.shape[1]).int()
    target_lens = (target_lens * targets.shape[1]).int()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batchmean":
        reduction_loss = "sum"
    else:
        reduction_loss = reduction
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
        reduction=reduction_loss,
    )

    if reduction == "batchmean":
        return loss / targets.shape[0]
    else:
        return loss


def l1_loss(
    predictions, targets, length=None, allowed_len_diff=3, reduction="mean"
):
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
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> l1_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.1000)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.l1_loss, reduction="none")
    return compute_masked_loss(
        loss, predictions, targets, length, reduction=reduction
    )


def mse_loss(
    predictions, targets, length=None, allowed_len_diff=3, reduction="mean"
):
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
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> mse_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.0100)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.mse_loss, reduction="none")
    return compute_masked_loss(
        loss, predictions, targets, length, reduction=reduction
    )


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
    reduction="mean",
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
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.

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
        reduction=reduction,
    )


def BCE_loss(
    inputs,
    targets,
    length=None,
    weight=None,
    pos_weight=None,
    reduction="mean",
    allowed_len_diff=3,
    label_smoothing=0.0,
):
    """Computes binary cross-entropy (BCE) loss. It also applies the sigmoid
    function directly (this improves the numerical stability).

    Arguments
    ---------
    inputs : torch.Tensor
        The output before applying the final softmax
        Format is [batch, 1] or [batch, frames, 1]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    weight: torch.Tensor
        a manual rescaling weight if provided it’s repeated to match input
        tensor shape.
    pos_weight : torch.Tensor
        a weight of positive examples. Must be a vector with length equal to
        the number of classes.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction: str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.

    Example
    -------
    >>> inputs = torch.tensor([10.0, -6.0])
    >>> targets = torch.tensor([1, 0])
    >>> BCE_loss(inputs, targets)
    tensor(0.0013)
    """
    if len(inputs.shape) == 3:
        inputs, targets = truncate(inputs, targets, allowed_len_diff)
        inputs = inputs.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(
        torch.nn.functional.binary_cross_entropy_with_logits,
        weight=weight,
        pos_weight=pos_weight,
        reduction="none",
    )
    return compute_masked_loss(
        loss,
        inputs,
        targets.float(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def kldiv_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    pad_idx=0,
    reduction="mean",
):
    """Computes the KL-divergence error at batch level.
    This loss applies label smoothing directly on the targets

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
    reduction: str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'batchmean' | 'sum'.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> kldiv_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if label_smoothing > 0:
        if log_probabilities.dim() == 2:
            log_probabilities = log_probabilities.unsqueeze(1)

        bz, time, n_class = log_probabilities.shape
        targets = targets.long().detach()

        confidence = 1 - label_smoothing

        log_probabilities = log_probabilities.view(-1, n_class)
        targets = targets.view(-1)
        with torch.no_grad():
            true_distribution = log_probabilities.clone()
            true_distribution.fill_(label_smoothing / (n_class - 1))
            ignore = targets == pad_idx
            targets = targets.masked_fill(ignore, 0)
            true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

        loss = torch.nn.functional.kl_div(
            log_probabilities, true_distribution, reduction="none"
        )
        loss = loss.masked_fill(ignore.unsqueeze(1), 0)

        # return loss according to reduction specified
        if reduction == "mean":
            return loss.sum().mean()
        elif reduction == "batchmean":
            return loss.sum() / bz
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        return nll_loss(log_probabilities, targets, length, reduction=reduction)


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
    loss_fn,
    predictions,
    targets,
    length=None,
    label_smoothing=0.0,
    reduction="mean",
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
    reduction: str
        Specifies the reduction to apply to the output loss: 'mean' | 'sum'.

    """
    mask = torch.ones_like(targets)
    if length is not None:
        mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )
        if len(targets.shape) == 3:
            mask = mask.unsqueeze(2).repeat(1, 1, targets.shape[2])

    loss = torch.sum(loss_fn(predictions, targets) * mask)
    if reduction == "mean":
        loss = loss / torch.sum(mask)
    if reduction == "batchmean":
        loss = loss / targets.shape[0]

    if label_smoothing == 0:
        return loss
    else:
        loss_reg = -torch.sum(torch.mean(predictions, dim=1) * mask)
        if reduction == "mean":
            loss_reg = loss_reg / torch.sum(mask)
        if reduction == "batchmean":
            loss_reg = loss_reg / targets.shape[0]

        return label_smoothing * loss_reg + (1 - label_smoothing) * loss


def get_si_snr_with_pitwrapper(source, estimate_source):
    """
    This function wraps si_snr calculation with the speechbrain pit-wrapper

    Args:
        source: [B, T, C],
            where B is batch size, T is the length of the sources, C is the number of sources
            the ordering is made so that this loss is compatible with the class PitWrapper

        estimate_source: [B, T, C]
            the estimated source

    Example:
    >>> x = torch.arange(600).reshape(3, 100, 2)
    >>> xhat = x[:, :, (1, 0)]
    >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    >>> print(si_snr)
    tensor(135.2284)
    """

    pit_si_snr = PitWrapper(cal_si_snr)
    loss, perms = pit_si_snr(source, estimate_source)

    return loss.mean()


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR
    Arguments:
        source: [T, B, C],
            where B is batch size, T is the length of the sources, C is the number of sources
            the ordering is made so that this loss is compatible with the class PitWrapper

        estimate_source: [T, B, C]
            the estimated source

    Example:
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1]
    ).to(device)
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """
    Args:
        source: [T, B, C]
        source_lengths: [B]
    Returns:
        mask: [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask
