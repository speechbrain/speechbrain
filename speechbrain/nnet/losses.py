"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Hwidong Na 2020
 * Yan Gao 2020
 * Titouan Parcollet 2020
"""

import functools
import math
from collections import namedtuple
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.decoders.ctc import filter_ctc_output
from speechbrain.utils.data_utils import unsqueeze_as
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def transducer_loss(
    logits,
    targets,
    input_lens,
    target_lens,
    blank_index,
    reduction="mean",
    use_torchaudio=True,
):
    """Transducer loss, see `speechbrain/nnet/loss/transducer_loss.py`.

    Arguments
    ---------
    logits : torch.Tensor
        Predicted tensor, of shape [batch, maxT, maxU, num_labels].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len].
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the label indices.
    reduction : str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.
    use_torchaudio: bool
        If True, use Transducer loss implementation from torchaudio, otherwise,
        use Speechbrain Numba implementation.

    Returns
    -------
    The computed transducer loss.
    """
    input_lens = (input_lens * logits.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()

    if use_torchaudio:
        try:
            from torchaudio.functional import rnnt_loss
        except ImportError:
            err_msg = "The dependency torchaudio >= 0.10.0 is needed to use Transducer Loss\n"
            err_msg += "Cannot import torchaudio.functional.rnnt_loss.\n"
            err_msg += "To use it, please install torchaudio >= 0.10.0\n"
            err_msg += "==================\n"
            err_msg += "Otherwise, you can use our numba implementation, set `use_torchaudio=False`.\n"
            raise ImportError(err_msg)

        return rnnt_loss(
            logits,
            targets.int(),
            input_lens,
            target_lens,
            blank=blank_index,
            reduction=reduction,
        )
    else:
        from speechbrain.nnet.loss.transducer_loss import Transducer

        # Transducer.apply function take log_probs tensor.
        log_probs = logits.log_softmax(-1)
        return Transducer.apply(
            log_probs, targets, input_lens, target_lens, blank_index, reduction
        )


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

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
        super().__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ---------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]
        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
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
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]
        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
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
        assert len(loss_mat.shape) >= 2, (
            "Base loss should not perform any reduction operation"
        )
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
        Arguments
        ---------
        tensor : torch.Tensor
            torch.Tensor to reorder given the optimal permutation, of shape
            [batch, ..., sources].
        p : list of tuples
            List of optimal permutations, e.g. for batch=2 and n_sources=3
            [(0, 1, 2), (0, 2, 1].

        Returns
        -------
        reordered : torch.Tensor
            Reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor, device=tensor.device)
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
        loss : torch.Tensor
            Permutation invariant loss for current examples, tensor of
            shape [batch]
        perms : list
            List of indexes for optimal permutation of the inputs over
            sources.
            e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
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
    """CTC loss.

    Arguments
    ---------
    log_probs : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'batch',
        'batchmean', 'none'.
        See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed CTC loss.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batchmean":
        reduction_loss = "sum"
    elif reduction == "batch":
        reduction_loss = "none"
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
    elif reduction == "batch":
        N = loss.size(0)
        return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
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
        Target tensor with the same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed L1 loss.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> l1_loss(probs, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
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
        Target tensor with the same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed MSE loss.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> mse_loss(probs, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    tensor(0.0100)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.mse_loss, reduction="none")
    return compute_masked_loss(
        loss, predictions, targets, length, reduction=reduction
    )


def classification_error(
    probabilities, targets, length=None, allowed_len_diff=3, reduction="mean"
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
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed classification error.

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
        """Computes the classification error."""
        predictions = torch.argmax(probabilities, dim=-1)
        return (predictions != targets).float()

    return compute_masked_loss(
        error, probabilities, targets.long(), length, reduction=reduction
    )


def nll_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    weight=None,
    reduction="mean",
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    label_smoothing : float
        The amount of smoothing to apply to labels (default 0.0, no smoothing)
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    weight: torch.Tensor
        A manual rescaling weight given to each class.
        If given, has to be a Tensor of size C.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed NLL loss.

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
    loss = functools.partial(
        torch.nn.functional.nll_loss, weight=weight, reduction="none"
    )
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def bce_loss(
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
        Format is [batch[, 1]?] or [batch, frames[, 1]?].
        (Works with or without a singleton dimension at the end).
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    weight : torch.Tensor
        A manual rescaling weight if provided it’s repeated to match input
        tensor shape.
    pos_weight : torch.Tensor
        A weight of positive examples. Must be a vector with length equal to
        the number of classes.
    reduction: str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    label_smoothing : float
        The amount of smoothing to apply to labels (default 0.0, no smoothing)

    Returns
    -------
    The computed BCE loss.

    Example
    -------
    >>> inputs = torch.tensor([10.0, -6.0])
    >>> targets = torch.tensor([1, 0])
    >>> bce_loss(inputs, targets)
    tensor(0.0013)
    """
    # Squeeze singleton dimension so inputs + targets match
    if len(inputs.shape) == len(targets.shape) + 1:
        inputs = inputs.squeeze(-1)

    # Make sure tensor lengths match
    if len(inputs.shape) >= 2:
        inputs, targets = truncate(inputs, targets, allowed_len_diff)
    elif length is not None:
        raise ValueError("length can be passed only for >= 2D inputs.")
    else:
        # In 1-dimensional case, add singleton dimension for time
        # so that we don't run into errors with the time-masked loss
        inputs, targets = inputs.unsqueeze(-1), targets.unsqueeze(-1)

    # input / target cannot be 1D so bump weight up to match
    if weight is not None and weight.dim() == 1:
        weight = weight.unsqueeze(-1)

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
    """Computes the KL-divergence error at the batch level.
    This loss applies label smoothing directly to the targets

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    label_smoothing : float
        The amount of smoothing to apply to labels (default 0.0, no smoothing)
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    pad_idx : int
        Entries of this value are considered padding.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed kldiv loss.

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
        elif reduction == "batch":
            return loss.view(bz, -1).sum(1) / length
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        return nll_loss(log_probabilities, targets, length, reduction=reduction)


def distance_diff_loss(
    predictions,
    targets,
    length=None,
    beta=0.25,
    max_weight=100.0,
    reduction="mean",
):
    """A loss function that can be used in cases where a model outputs
    an arbitrary probability distribution for a discrete variable on
    an interval scale, such as the length of a sequence, and the ground
    truth is the precise values of the variable from a data sample.

    The loss is defined as
    loss_i = p_i * exp(beta * |i - y|) - 1.

    The loss can also be used where outputs aren't probabilities, so long
    as high values close to the ground truth position and low values away
    from it are desired

    Arguments
    ---------
    predictions: torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position
    targets: torch.Tensor
        a 1-D tensor in which each element is thr ground truth
    length: torch.Tensor
        lengths (for masking in padded batches)
    beta: torch.Tensor
        a hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster
    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)
    reduction: str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size

    Returns
    -------
    The masked loss.

    Example
    -------
    >>> predictions = torch.tensor(
    ...     [
    ...         [0.25, 0.5, 0.25, 0.0],
    ...         [0.05, 0.05, 0.9, 0.0],
    ...         [8.0, 0.10, 0.05, 0.05],
    ...     ]
    ... )
    >>> targets = torch.tensor([2.0, 3.0, 1.0])
    >>> length = torch.tensor([0.75, 0.75, 1.0])
    >>> loss = distance_diff_loss(predictions, targets, length)
    >>> loss
    tensor(0.2967)
    """
    return compute_masked_loss(
        functools.partial(
            _distance_diff_loss, beta=beta, max_weight=max_weight
        ),
        predictions=predictions,
        targets=targets,
        length=length,
        reduction=reduction,
        mask_shape="loss",
    )


def _distance_diff_loss(predictions, targets, beta, max_weight):
    """Computes the raw (unreduced) distance difference loss

    Arguments
    ---------
    predictions: torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position
    targets: torch.Tensor
        a 1-D tensor in which each element is thr ground truth
    beta: torch.Tensor
        a hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster
    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)

    Returns
    -------
    The raw distance loss.
    """
    batch_size, max_len = predictions.shape
    pos_range = (torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)).to(
        predictions.device
    )
    diff_range = (pos_range - targets.unsqueeze(-1)).abs()
    loss_weights = ((beta * diff_range).exp() - 1.0).clamp(max=max_weight)
    return (loss_weights * predictions).unsqueeze(-1)


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

    Returns
    -------
    predictions : torch.Tensor
    targets : torch.Tensor
        Same as inputs, but with the same shape.
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
    mask_shape="targets",
    reduction="mean",
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none").
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output
        Distributions. https://arxiv.org/abs/1701.06548
    mask_shape: torch.Tensor
        the shape of the mask
        The default is "targets", which will cause the mask to be the same
        shape as the targets

        Other options include "predictions" and "loss", which will use the
        shape of the predictions and the unreduced loss, respectively.
        These are useful for loss functions that whose output does not
        match the shape of the targets
    reduction : str
        One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
        single value and 'batch' returns one per item in the batch and
        'batchmean' is sum / batch_size and 'none' returns all.

    Returns
    -------
    The masked loss.
    """

    # Compute, then reduce loss
    loss = loss_fn(predictions, targets)

    if mask_shape == "targets":
        mask_data = targets
    elif mask_shape == "predictions":
        mask_data = predictions
    elif mask_shape == "loss":
        mask_data = loss
    else:
        raise ValueError(f"Invalid mask_shape value {mask_shape}")

    mask = compute_length_mask(mask_data, length)

    loss *= mask
    return reduce_loss(
        loss, mask, reduction, label_smoothing, predictions, targets
    )


def compute_length_mask(data, length=None, len_dim=1):
    """Computes a length mask for the specified data shape

    Arguments
    ---------
    data: torch.Tensor
        the data shape
    length: torch.Tensor
        the length of the corresponding data samples
    len_dim: int
        the length dimension (defaults to 1)

    Returns
    -------
    mask: torch.Tensor
        the mask

    Example
    -------
    >>> data = torch.arange(5)[None, :, None].repeat(3, 1, 2)
    >>> data += torch.arange(1, 4)[:, None, None]
    >>> data *= torch.arange(1, 3)[None, None, :]
    >>> data
    tensor([[[ 1,  2],
             [ 2,  4],
             [ 3,  6],
             [ 4,  8],
             [ 5, 10]],
    <BLANKLINE>
            [[ 2,  4],
             [ 3,  6],
             [ 4,  8],
             [ 5, 10],
             [ 6, 12]],
    <BLANKLINE>
            [[ 3,  6],
             [ 4,  8],
             [ 5, 10],
             [ 6, 12],
             [ 7, 14]]])
    >>> compute_length_mask(data, torch.tensor([1.0, 0.4, 0.8]))
    tensor([[[1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 1]],
    <BLANKLINE>
            [[1, 1],
             [1, 1],
             [0, 0],
             [0, 0],
             [0, 0]],
    <BLANKLINE>
            [[1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [0, 0]]])
    >>> compute_length_mask(data, torch.tensor([0.5, 1.0, 0.5]), len_dim=2)
    tensor([[[1, 0],
             [1, 0],
             [1, 0],
             [1, 0],
             [1, 0]],
    <BLANKLINE>
            [[1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 1]],
    <BLANKLINE>
            [[1, 0],
             [1, 0],
             [1, 0],
             [1, 0],
             [1, 0]]])
    """
    mask = torch.ones_like(data)
    if length is not None:
        length_mask = length_to_mask(
            (length * data.shape[len_dim] - 1e-6),
            max_len=data.shape[len_dim],
        )

        # Handle any dimensionality of input
        while len(length_mask.shape) < len(mask.shape):
            length_mask = length_mask.unsqueeze(-1)
        length_mask = length_mask.type(mask.dtype).transpose(1, len_dim)
        mask *= length_mask
    return mask


def reduce_loss(
    loss,
    mask,
    reduction="mean",
    label_smoothing=0.0,
    predictions=None,
    targets=None,
):
    """Performs the specified reduction of the raw loss value

    Arguments
    ---------
    loss : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none").
    mask : torch.Tensor
        Mask to apply before computing loss.
    reduction : str
        One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
        single value and 'batch' returns one per item in the batch and
        'batchmean' is sum / batch_size and 'none' returns all.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output
        Distributions. https://arxiv.org/abs/1701.06548
    predictions : torch.Tensor
        First argument to loss function. Required only if label smoothing is used.
    targets : torch.Tensor
        Second argument to loss function. Required only if label smoothing is used.

    Returns
    -------
    Reduced loss.
    """
    N = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / N
    elif reduction == "batch":
        loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

    if label_smoothing == 0:
        return loss
    else:
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss


def get_si_snr_with_pitwrapper(source, estimate_source):
    """This function wraps si_snr calculation with the speechbrain pit-wrapper.

    Arguments
    ---------
    source: torch.Tensor
        Shape is [B, T, C],
        Where B is the batch size, T is the length of the sources, C is
        the number of sources the ordering is made so that this loss is
        compatible with the class PitWrapper.
    estimate_source: torch.Tensor
        The estimated source, of shape [B, T, C]

    Returns
    -------
    loss: torch.Tensor
        The computed SNR

    Example
    -------
    >>> x = torch.arange(600).reshape(3, 100, 2)
    >>> xhat = x[:, :, (1, 0)]
    >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    >>> print(si_snr)
    tensor([135.2284, 135.2284, 135.2284])
    """

    pit_si_snr = PitWrapper(cal_si_snr)
    loss, perms = pit_si_snr(source, estimate_source)

    return loss


def get_snr_with_pitwrapper(source, estimate_source):
    """This function wraps snr calculation with the speechbrain pit-wrapper.

    Arguments
    ---------
    source: torch.Tensor
        Shape is [B, T, E, C],
        Where B is the batch size, T is the length of the sources, E is binaural channels, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.
    estimate_source: torch.Tensor
        The estimated source, of shape [B, T, E, C]

    Returns
    -------
    loss: torch.Tensor
        The computed SNR
    """

    pit_snr = PitWrapper(cal_snr)
    loss, perms = pit_snr(source, estimate_source)

    return loss


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments
    ---------
    source: torch.Tensor
        Shape is [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.
    estimate_source: torch.Tensor
        The estimated source, of shape [T, B, C]

    Returns
    -------
    The calculated SI-SNR.

    Example:
    ---------
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
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
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
        torch.sum(s_target**2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj**2, dim=0) / (
        torch.sum(e_noise**2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)


def cal_snr(source, estimate_source):
    """Calculate binaural channel SNR.

    Arguments
    ---------
    source: torch.Tensor
        Shape is [T, E, B, C]
        Where B is batch size, T is the length of the sources, E is binaural channels, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.
    estimate_source: torch.Tensor
        The estimated source, of shape [T, E, B, C]

    Returns
    -------
    Binaural channel SNR
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
    mask = get_mask(source, source_lengths)  # [T, E, 1]
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

    # Step 2. SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, E, B, C]
    s_estimate = zero_mean_estimate  # [T, E, B, C]
    # SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    # n_dim = [x for x in range(len(s_target.shape)-2)]
    snr_beforelog = torch.sum(s_target**2, dim=0) / (
        torch.sum((s_estimate - s_target) ** 2, dim=0) + EPS
    )
    snr = 10 * torch.log10(snr_beforelog + EPS)  # [B, C]

    return -snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : torch.Tensor
        Shape [T, B, C]
    source_lengths : torch.Tensor
        Shape [B]

    Returns
    -------
    mask : torch.Tensor
        Shape [T, B, 1]

    Example
    -------
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
    mask = source.new_ones(source.size()[:-1]).unsqueeze(-1).transpose(1, -2)
    B = source.size(-2)
    for i in range(B):
        mask[source_lengths[i] :, i] = 0
    return mask.transpose(-2, 1)


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity
    scale : float
        The scale for cosine similarity

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor(
    ...     [[1.0, -1.0], [-1.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
    ... )
    >>> targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:, 0] > predictions[:, 1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale : float
        The scale for cosine similarity.
    easy_margin : bool

    Example
    -------
    >>> outputs = torch.tensor(
    ...     [[1.0, -1.0], [-1.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
    ... )
    >>> targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:, 0] > predictions[:, 1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super().__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    loss_fn : Callable
        The LogSoftmax function to wrap.

    Example
    -------
    >>> outputs = torch.tensor(
    ...     [[1.0, -1.0], [-1.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
    ... )
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([[0], [1], [0], [1]])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor(
    ...     [[1.0, -1.0], [-1.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
    ... )
    >>> log_prob = LogSoftmaxWrapper(
    ...     AdditiveAngularMargin(margin=0.3, scale=32)
    ... )
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].
        length : torch.Tensor
            The lengths of the corresponding inputs.

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


def ctc_loss_kd(log_probs, targets, input_lens, blank_index, device):
    """Knowledge distillation for CTC loss.

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    log_probs : torch.Tensor
        Predicted tensor from student model, of shape [batch, time, chars].
    targets : torch.Tensor
        Predicted tensor from single teacher model, of shape [batch, time, chars].
    input_lens : torch.Tensor
        Length of each utterance.
    blank_index : int
        The location of the blank symbol among the character indexes.
    device : str
        Device for computing.

    Returns
    -------
    The computed CTC loss.
    """
    scores, predictions = torch.max(targets, dim=-1)

    pred_list = []
    pred_len_list = []
    for j in range(predictions.shape[0]):
        # Getting current predictions
        current_pred = predictions[j]

        actual_size = (input_lens[j] * log_probs.shape[1]).round().int()
        current_pred = current_pred[0:actual_size]
        current_pred = filter_ctc_output(
            list(current_pred.cpu().numpy()), blank_id=blank_index
        )
        current_pred_len = len(current_pred)
        pred_list.append(current_pred)
        pred_len_list.append(current_pred_len)

    max_pred_len = max(pred_len_list)
    for j in range(predictions.shape[0]):
        diff = max_pred_len - pred_len_list[j]
        for n in range(diff):
            pred_list[j].append(0)

    # generate soft label of teacher model
    fake_lab = torch.from_numpy(np.array(pred_list))
    fake_lab.to(device)
    fake_lab = fake_lab.int()
    fake_lab_lengths = torch.from_numpy(np.array(pred_len_list)).int()
    fake_lab_lengths.to(device)

    input_lens = (input_lens * log_probs.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        fake_lab,
        input_lens,
        fake_lab_lengths,
        blank_index,
        zero_infinity=True,
    )


def ce_kd(inp, target):
    """Simple version of distillation for cross-entropy loss.

    Arguments
    ---------
    inp : torch.Tensor
        The probabilities from student model, of shape [batch_size * length, feature]
    target : torch.Tensor
        The probabilities from teacher model, of shape [batch_size * length, feature]

    Returns
    -------
    The distilled outputs.
    """
    return (-target * inp).sum(1)


def nll_loss_kd(probabilities, targets, rel_lab_lengths):
    """Knowledge distillation for negative log-likelihood loss.

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    probabilities : torch.Tensor
        The predicted probabilities from the student model.
        Format is [batch, frames, p]
    targets : torch.Tensor
        The target probabilities from the teacher model.
        Format is [batch, frames, p]
    rel_lab_lengths : torch.Tensor
        Length of each utterance, if the frame-level loss is desired.

    Returns
    -------
    Computed NLL KD loss.

    Example
    -------
    >>> probabilities = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    >>> targets = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> rel_lab_lengths = torch.tensor([1.0])
    >>> nll_loss_kd(probabilities, targets, rel_lab_lengths)
    tensor(-0.7400)
    """
    # Getting the number of sentences in the minibatch
    N_snt = probabilities.shape[0]

    # Getting the maximum length of label sequence
    max_len = probabilities.shape[1]

    # Getting the label lengths
    lab_lengths = torch.round(rel_lab_lengths * targets.shape[1]).int()

    # Reshape to [batch_size * length, feature]
    prob_curr = probabilities.reshape(N_snt * max_len, probabilities.shape[-1])

    # Generating mask
    mask = length_to_mask(
        lab_lengths, max_len=max_len, dtype=torch.float, device=prob_curr.device
    )

    # Reshape to [batch_size * length, feature]
    lab_curr = targets.reshape(N_snt * max_len, targets.shape[-1])

    loss = ce_kd(prob_curr, lab_curr)
    # Loss averaging
    loss = torch.sum(loss.reshape(N_snt, max_len) * mask) / torch.sum(mask)
    return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss as used in wav2vec2.

    Reference
    ---------
    wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
    https://arxiv.org/abs/2006.11477

    Arguments
    ---------
    logit_temp : torch.Float
        A temperature to divide the logits.
    """

    def __init__(self, logit_temp):
        super().__init__()
        self.logit_temp = logit_temp

    def forward(self, x, y, negs):
        """Compute contrastive loss.

        Arguments
        ---------
        x : torch.Tensor
            Encoded embeddings with shape (B, T, C).
        y : torch.Tensor
            Feature extractor target embeddings with shape (B, T, C).
        negs : torch.Tensor
            Negative embeddings from feature extractor with shape (N, B, T, C)
            where N is number of negatives. Can be obtained with our sample_negatives
            function (check in lobes/wav2vec2).

        Returns
        -------
        loss : torch.Tensor
            The computed loss
        accuracy : torch.Tensor
            The computed accuracy
        """
        neg_is_pos = (y == negs).all(-1)
        y = y.unsqueeze(0)
        target_and_negatives = torch.cat([y, negs], dim=0)
        logits = torch.cosine_similarity(
            x.float(), target_and_negatives.float(), dim=-1
        ).type_as(x)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        # N, B, T -> T, B, N -> T*B, N
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))

        targets = torch.zeros(
            (logits.size(0)), dtype=torch.long, device=logits.device
        )
        loss = F.cross_entropy(
            logits / self.logit_temp, targets, reduction="sum"
        )
        accuracy = torch.sum(logits.argmax(-1) == 0) / (
            logits.numel() / logits.size(-1)
        )
        return loss, accuracy


class VariationalAutoencoderLoss(nn.Module):
    """The Variational Autoencoder loss, with support for length masking

    From Autoencoding Variational Bayes: https://arxiv.org/pdf/1312.6114.pdf

    Arguments
    ---------
    rec_loss: callable
        a function or module to compute the reconstruction loss
    len_dim: int
        the dimension to be used for the length, if encoding sequences
        of variable length
    dist_loss_weight: float
        the relative weight of the distribution loss (K-L divergence)

    Example
    -------
    >>> from speechbrain.nnet.autoencoders import VariationalAutoencoderOutput
    >>> vae_loss = VariationalAutoencoderLoss(dist_loss_weight=0.5)
    >>> predictions = VariationalAutoencoderOutput(
    ...     rec=torch.tensor([[0.8, 1.0], [1.2, 0.6], [0.4, 1.4]]),
    ...     mean=torch.tensor(
    ...         [[0.5, 1.0], [1.5, 1.0], [1.0, 1.4]],
    ...     ),
    ...     log_var=torch.tensor(
    ...         [[0.0, -0.2], [2.0, -2.0], [0.2, 0.4]],
    ...     ),
    ...     latent=torch.randn(3, 1),
    ...     latent_sample=torch.randn(3, 1),
    ...     latent_length=torch.tensor([1.0, 1.0, 1.0]),
    ... )
    >>> targets = torch.tensor([[0.9, 1.1], [1.4, 0.6], [0.2, 1.4]])
    >>> loss = vae_loss(predictions, targets)
    >>> loss
    tensor(1.1264)
    >>> details = vae_loss.details(predictions, targets)
    >>> details  # doctest: +NORMALIZE_WHITESPACE
    VariationalAutoencoderLossDetails(loss=tensor(1.1264),
                                      rec_loss=tensor(0.0333),
                                      dist_loss=tensor(2.1861),
                                      weighted_dist_loss=tensor(1.0930))
    """

    def __init__(self, rec_loss=None, len_dim=1, dist_loss_weight=0.001):
        super().__init__()
        if rec_loss is None:
            rec_loss = mse_loss
        self.rec_loss = rec_loss
        self.dist_loss_weight = dist_loss_weight
        self.len_dim = len_dim

    def forward(self, predictions, targets, length=None, reduction="batchmean"):
        """Computes the forward pass

        Arguments
        ---------
        predictions: speechbrain.nnet.autoencoders.VariationalAutoencoderOutput
            the variational autoencoder output
        targets: torch.Tensor
            the reconstruction targets
        length : torch.Tensor
            Length of each sample for computing true error with a mask.
        reduction: str
            The type of reduction to apply, default "batchmean"

        Returns
        -------
        loss: torch.Tensor
            the VAE loss (reconstruction + K-L divergence)
        """
        return self.details(predictions, targets, length, reduction).loss

    def details(self, predictions, targets, length=None, reduction="batchmean"):
        """Gets detailed information about the loss (useful for plotting, logs,
        etc.)

        Arguments
        ---------
        predictions: speechbrain.nnet.autoencoders.VariationalAutoencoderOutput
            the variational autoencoder output (or a tuple of rec, mean, log_var)
        targets: torch.Tensor
            targets for the reconstruction loss
        length : torch.Tensor
            Length of each sample for computing true error with a mask.
        reduction: str
            The type of reduction to apply, default "batchmean"

        Returns
        -------
        details: VAELossDetails
            a namedtuple with the following parameters
            loss: torch.Tensor
                the combined loss
            rec_loss: torch.Tensor
                the reconstruction loss
            dist_loss: torch.Tensor
                the distribution loss (K-L divergence), raw value
            weighted_dist_loss: torch.Tensor
                the weighted value of the distribution loss, as used
                in the combined loss

        """
        if length is None:
            length = torch.ones(targets.size(0))
        rec_loss, dist_loss = self._compute_components(predictions, targets)
        rec_loss = _reduce_autoencoder_loss(rec_loss, length, reduction)
        dist_loss = _reduce_autoencoder_loss(dist_loss, length, reduction)
        weighted_dist_loss = self.dist_loss_weight * dist_loss
        loss = rec_loss + weighted_dist_loss

        return VariationalAutoencoderLossDetails(
            loss, rec_loss, dist_loss, weighted_dist_loss
        )

    def _compute_components(self, predictions, targets):
        rec, _, mean, log_var, _, _ = predictions
        rec_loss = self._align_length_axis(
            self.rec_loss(targets, rec, reduction="none")
        )
        dist_loss = self._align_length_axis(
            -0.5 * (1 + log_var - mean**2 - log_var.exp())
        )
        return rec_loss, dist_loss

    def _align_length_axis(self, tensor):
        return tensor.moveaxis(self.len_dim, 1)


class AutoencoderLoss(nn.Module):
    """An implementation of a standard (non-variational)
    autoencoder loss

    Arguments
    ---------
    rec_loss: callable
        the callable to compute the reconstruction loss
    len_dim: int
        the dimension index to be used for length

    Example
    -------
    >>> from speechbrain.nnet.autoencoders import AutoencoderOutput
    >>> ae_loss = AutoencoderLoss()
    >>> rec = torch.tensor([[0.8, 1.0], [1.2, 0.6], [0.4, 1.4]])
    >>> predictions = AutoencoderOutput(
    ...     rec=rec,
    ...     latent=torch.randn(3, 1),
    ...     latent_length=torch.tensor([1.0, 1.0]),
    ... )
    >>> targets = torch.tensor([[0.9, 1.1], [1.4, 0.6], [0.2, 1.4]])
    >>> ae_loss(predictions, targets)
    tensor(0.0333)
    >>> ae_loss.details(predictions, targets)
    AutoencoderLossDetails(loss=tensor(0.0333), rec_loss=tensor(0.0333))
    """

    def __init__(self, rec_loss=None, len_dim=1):
        super().__init__()
        if rec_loss is None:
            rec_loss = mse_loss
        self.rec_loss = rec_loss
        self.len_dim = len_dim

    def forward(self, predictions, targets, length=None, reduction="batchmean"):
        """Computes the autoencoder loss

        Arguments
        ---------
        predictions: speechbrain.nnet.autoencoders.AutoencoderOutput
            the autoencoder output
        targets: torch.Tensor
            targets for the reconstruction loss
        length: torch.Tensor
            Length of each sample for computing true error with a mask
        reduction: str
            The type of reduction to apply, default "batchmean"

        Returns
        -------
        The computed loss.
        """
        rec_loss = self._align_length_axis(
            self.rec_loss(targets, predictions.rec, reduction="none")
        )
        return _reduce_autoencoder_loss(rec_loss, length, reduction)

    def details(self, predictions, targets, length=None, reduction="batchmean"):
        """Gets detailed information about the loss (useful for plotting, logs,
        etc.)

        This is provided mainly to make the loss interchangeable with
        more complex autoencoder loses, such as the VAE loss.

        Arguments
        ---------
        predictions: speechbrain.nnet.autoencoders.AutoencoderOutput
            the  autoencoder output
        targets: torch.Tensor
            targets for the reconstruction loss
        length : torch.Tensor
            Length of each sample for computing true error with a mask.
        reduction: str
            The type of reduction to apply, default "batchmean"

        Returns
        -------
        details: AutoencoderLossDetails
            a namedtuple with the following parameters
            loss: torch.Tensor
                the combined loss
            rec_loss: torch.Tensor
                the reconstruction loss
        """
        loss = self(predictions, targets, length, reduction)
        return AutoencoderLossDetails(loss, loss)

    def _align_length_axis(self, tensor):
        return tensor.moveaxis(self.len_dim, 1)


def _reduce_autoencoder_loss(loss, length, reduction):
    max_len = loss.size(1)
    if length is not None:
        mask = length_to_mask(length * max_len, max_len)
        mask = unsqueeze_as(mask, loss).expand_as(loss)
    else:
        mask = torch.ones_like(loss)
    reduced_loss = reduce_loss(loss * mask, mask, reduction=reduction)
    return reduced_loss


VariationalAutoencoderLossDetails = namedtuple(
    "VariationalAutoencoderLossDetails",
    ["loss", "rec_loss", "dist_loss", "weighted_dist_loss"],
)

AutoencoderLossDetails = namedtuple(
    "AutoencoderLossDetails", ["loss", "rec_loss"]
)


class Laplacian(nn.Module):
    """Computes the Laplacian for image-like data

    Arguments
    ---------
    kernel_size: int
        the size of the Laplacian kernel
    dtype: torch.dtype
        the data type (optional)

    Example
    -------
    >>> lap = Laplacian(3)
    >>> lap.get_kernel()
    tensor([[[[-1., -1., -1.],
              [-1.,  8., -1.],
              [-1., -1., -1.]]]])
    >>> data = torch.eye(6) + torch.eye(6).flip(0)
    >>> data
    tensor([[1., 0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 1., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 1., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0., 1.]])
    >>> lap(data.unsqueeze(0))
    tensor([[[ 6., -3., -3.,  6.],
             [-3.,  4.,  4., -3.],
             [-3.,  4.,  4., -3.],
             [ 6., -3., -3.,  6.]]])
    """

    def __init__(self, kernel_size, dtype=torch.float32):
        super().__init__()
        self.kernel_size = kernel_size
        self.dtype = dtype
        kernel = self.get_kernel()
        self.register_buffer("kernel", kernel)

    def get_kernel(self):
        """Computes the Laplacian kernel"""
        kernel = -torch.ones(
            self.kernel_size, self.kernel_size, dtype=self.dtype
        )
        mid_position = self.kernel_size // 2
        mid_value = self.kernel_size**2 - 1.0
        kernel[mid_position, mid_position] = mid_value
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, data):
        """Computes the Laplacian of image-like data

        Arguments
        ---------
        data: torch.Tensor
            a (B x C x W x H) or (B x C x H x W) tensor with image-like data

        Returns
        -------
        The transformed outputs.
        """
        return F.conv2d(data, self.kernel)


class LaplacianVarianceLoss(nn.Module):
    """The Laplacian variance loss - used to penalize blurriness in image-like
    data, such as spectrograms.

    The loss value will be the negative variance because the
    higher the variance, the sharper the image.

    Arguments
    ---------
    kernel_size: int
        the Laplacian kernel size

    len_dim: int
        the dimension to be used as the length

    Example
    -------
    >>> lap_loss = LaplacianVarianceLoss(3)
    >>> data = torch.ones(6, 6).unsqueeze(0)
    >>> data
    tensor([[[1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1.]]])
    >>> lap_loss(data)
    tensor(-0.)
    >>> data = (torch.eye(6) + torch.eye(6).flip(0)).unsqueeze(0)
    >>> data
    tensor([[[1., 0., 0., 0., 0., 1.],
             [0., 1., 0., 0., 1., 0.],
             [0., 0., 1., 1., 0., 0.],
             [0., 0., 1., 1., 0., 0.],
             [0., 1., 0., 0., 1., 0.],
             [1., 0., 0., 0., 0., 1.]]])
    >>> lap_loss(data)
    tensor(-17.6000)
    """

    def __init__(self, kernel_size=3, len_dim=1):
        super().__init__()
        self.len_dim = len_dim
        self.laplacian = Laplacian(kernel_size=kernel_size)

    def forward(self, predictions, length=None, reduction=None):
        """Computes the Laplacian loss

        Arguments
        ---------
        predictions: torch.Tensor
            a (B x C x W x H) or (B x C x H x W) tensor
        length: torch.Tensor
            The length of the corresponding inputs.
        reduction: str
            "batch" or None

        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        laplacian = self.laplacian(predictions)
        laplacian = laplacian.moveaxis(self.len_dim, 1)
        mask = compute_length_mask(laplacian, length).bool()
        if reduction == "batch":
            # TODO: Vectorize
            loss = torch.stack(
                [
                    item.masked_select(item_mask).var()
                    for item, item_mask in zip(laplacian, mask)
                ]
            )
        else:
            loss = laplacian.masked_select(mask).var()
        return -loss
