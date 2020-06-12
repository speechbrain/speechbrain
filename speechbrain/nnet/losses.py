"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
"""

import torch
import logging
import functools
from speechbrain.data_io.data_io import length_to_mask

import numpy as np
from speechbrain.decoders.ctc import filter_ctc_output
from speechbrain.utils.edit_distance import accumulatable_wer_stats


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


'''
The Functions below are For KD.
'''
def ctc_loss_kd(log_probs, targets, input_lens, blank_index, device):
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
    scores, predictions = torch.max(targets, dim=-1)

    pred_list = []
    pred_len_list = []
    for j in range(predictions.shape[0]):
        # Getting current predictions
        current_pred = predictions[j]

        actual_size = (input_lens[j] * log_probs.shape[1]).int()
        current_pred = current_pred[0:actual_size]
        current_pred = filter_ctc_output(list(current_pred.cpu().numpy()), blank_id=blank_index)
        current_pred_len = len(current_pred)
        pred_list.append(current_pred)
        pred_len_list.append(current_pred_len)

    max_pred_len = max(pred_len_list)
    for j in range(predictions.shape[0]):
        diff = max_pred_len - pred_len_list[j]
        for n in range(diff):
            pred_list[j].append(0)

    fake_lab = torch.from_numpy(np.array(pred_list))
    fake_lab.to(device)
    fake_lab = fake_lab.int()
    fake_lab_lengths = torch.from_numpy(np.array(pred_len_list)).int()
    fake_lab_lengths.to(device)

    input_lens = (input_lens * log_probs.shape[1]).int()
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
    return (- target * inp).sum(1)

def nll_loss_kd(
    probabilities,
    targets,
    rel_lab_lengths,
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
    # Getting the number of sentences in the minibatch
    N_snt = probabilities.shape[0]

    # Getting the maximum length of label sequence
    max_len = probabilities.shape[1]

    # Getting the label lengths
    lab_lengths = torch.round(rel_lab_lengths * targets.shape[1]).int()

    # Reshape to [batch_size * length, feature]
    prob_curr = probabilities.reshape(N_snt * max_len, probabilities.shape[-1])

    # Generating mask
    mask = length_to_mask(lab_lengths,
                          max_len=max_len,
                          dtype=torch.float,
                          device=prob_curr.device)

    # Reshape to [batch_size * length, feature]
    lab_curr = targets.reshape(N_snt * max_len, targets.shape[-1])

    loss = ce_kd(prob_curr, lab_curr)
    # Loss averaging
    loss = torch.sum(loss.reshape(N_snt, max_len) * mask) / torch.sum(mask)
    return loss

def compute_wer_list(prob, lab, lab_length, eos_index):
    # Getting the number of sentences in the minibatch
    N_snt = prob.shape[0]

    # Wer initialization
    wer = 0

    # Loop over all the sentences of the minibatch
    wer_list = []
    for j in range(N_snt):
        # getting the current probabilities and labels
        prob_curr = prob[j]
        lab_curr = lab[j]

        # NOTE temporary
        ## Avoiding padded time steps
        # actual_size_prob = int(
        #    torch.round(lengths[0][j] * prob_curr.shape[-1])
        # )

        actual_size_lab = int(
            torch.round(lab_length[j] * lab_curr.shape[0])
        )

        # prob_curr = prob_curr.narrow(-1, 0, actual_size_prob)
        lab_curr = lab_curr.narrow(-1, 0, actual_size_lab)

        # Computing the wer
        wer = compute_wer(prob_curr, lab_curr, eos_index=eos_index)
        wer_list.append(wer)

    wer_list = torch.tensor(wer_list)
    wer_list = torch.unsqueeze(wer_list, 0)
    return wer_list

def compute_wer_list_ctc(prob, lab, prob_length, lab_length, blank_index):
    # Getting the number of sentences in the minibatch
    N_snt = prob.shape[0]

    # Wer initialization
    wer = 0

    # Loop over all the sentences of the minibatch
    wer_list = []
    for j in range(N_snt):
        # getting the current probabilities and labels
        prob_curr = prob[j]
        lab_curr = lab[j]

        # NOTE temporary
        ## Avoiding padded time steps
        # actual_size_prob = int(
        #    torch.round(lengths[0][j] * prob_curr.shape[-1])
        # )

        actual_size_lab = int(
            torch.round(lab_length[j] * lab_curr.shape[0])
        )

        # prob_curr = prob_curr.narrow(-1, 0, actual_size_prob)
        lab_curr = lab_curr.narrow(-1, 0, actual_size_lab)

        # Computing the wer
        wer = compute_wer(prob_curr, lab_curr, prob_length[j], ctc_label=True, blank_index=blank_index)
        wer_list.append(wer)

    wer_list = torch.tensor(wer_list)
    wer_list = torch.unsqueeze(wer_list, 0)
    return wer_list


def compute_wer(prob, lab, length=None, ctc_label=False, blank_index=None, eos_index=None):

    if ctc_label:
        scores, predictions = torch.max(prob, dim=-1)

        actual_size = int(
            torch.round(length * predictions.shape[-1])
        )
        predictions = predictions[0:actual_size]

        # Converting labels and prediction to lists (faster)
        lab = lab.tolist()
        predictions = predictions.tolist()

        predictions = filter_ctc_output(
            predictions, blank_id=blank_index
        )

        # Computing the word error rate
        stats = accumulatable_wer_stats([lab], [predictions])

        # Getting the wer
        wer = stats["WER"]

        # Setting the max value of wer
        if wer > 100:
            wer = 100

        # Converting to a FloatTensor
        wer = torch.FloatTensor([wer])

        return wer

    else:
        # Computing predictions
        scores, predictions = torch.max(prob, dim=-1)

        # Converting labels and prediction to lists (faster)
        lab = lab.tolist()
        predictions = predictions.tolist()

        #elif "seq_nll" in self.cost_type:
        # NOTE: temporary
        predictions = filter_seq2seq_output(
            predictions, eos_id=eos_index
        )

        # Computing the word error rate
        stats = accumulatable_wer_stats([lab], [predictions])

        # Getting the wer
        wer = stats["WER"]

        # Setting the max value of wer
        if wer > 100:
            wer = 100

        # Converting to a FloatTensor
        wer = torch.FloatTensor([wer])

        return wer

def filter_seq2seq_output(string_pred, eos_id=-1, logger=None):
    if isinstance(string_pred, list):
        # Finding the first eos token
        try:
            eos_index = next(i for i, v in enumerate(string_pred) if v == eos_id)
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]

        return string_out