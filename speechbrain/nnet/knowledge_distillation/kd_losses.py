"""
Losses for knowledge distillation.

Authors
 * Yan Gao 2020
 * Titouan Parcollet 2020

"""

import torch
from speechbrain.data_io.data_io import length_to_mask
import numpy as np
from speechbrain.decoders.ctc import filter_ctc_output
from speechbrain.utils.edit_distance import accumulatable_wer_stats


def ctc_loss_kd(log_probs, targets, input_lens, blank_index, device):
    """Knowledge distillation for CTC loss

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
        device for computing.
    """
    scores, predictions = torch.max(targets, dim=-1)

    pred_list = []
    pred_len_list = []
    for j in range(predictions.shape[0]):
        # Getting current predictions
        current_pred = predictions[j]

        actual_size = (input_lens[j] * log_probs.shape[1]).int()
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
    """Simple version of distillation fro cross entropy loss.

    Arguments
    ---------
    inp : torch.Tensor
        The probabilities from student model, of shape [batch_size * length, feature]
    target : torch.Tensor
        The probabilities from teacher model, of shape [batch_size * length, feature]
    """
    return (-target * inp).sum(1)


def nll_loss_kd(
    probabilities, targets, rel_lab_lengths,
):
    """Knowledge distillation for negative log likelihood loss.

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    probabilities : torch.Tensor
        The predicted probabilities from student model.
        Format is [batch, frames, p]
    targets : torch.Tensor
        The target probabilities from teacher model.
        Format is [batch, frames, p]
    rel_lab_lengths : torch.Tensor
        Length of each utterance, if frame-level loss is desired.

    Example
    -------
    >>> probabilities = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    >>> targets = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> rel_lab_lengths = torch.tensor([1.])
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


def wer_ce(prob, lab, lab_length, eos_index):
    """Compute word (or character) error rate based on output of decoder (CE loss).

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    prob : torch.Tensor
        The probabilities from decoder output layer of a teacher model.
        Format is [batch, frames, p]
    lab : torch.Tensor
        The target label from dataset, of shape [batch, frames]
    lab_length : torch.Tensor
        Length of each target label sequence.
    eos_index : int
        The location of the eos symbol among the character indexes.

    Example
    -------
    >>> prob = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> lab = torch.tensor([[1, 1]])
    >>> lab_length = torch.tensor([1.])
    >>> eos_index = 0
    >>> wer_ce(prob, lab, lab_length, eos_index)
    tensor([[100.]])
    """
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

        actual_size_lab = int(torch.round(lab_length[j] * lab_curr.shape[0]))

        # prob_curr = prob_curr.narrow(-1, 0, actual_size_prob)
        lab_curr = lab_curr.narrow(-1, 0, actual_size_lab)

        # Computing the wer
        wer = compute_wer(prob_curr, lab_curr, eos_index=eos_index)
        wer_list.append(wer)

    wer_list = torch.tensor(wer_list)
    wer_list = torch.unsqueeze(wer_list, 0)
    return wer_list


def wer_ctc(prob, lab, prob_length, lab_length, blank_index):
    """Compute word (or character) error rate based on output of encoder (CTC loss).

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    prob : torch.Tensor
        The probabilities from encoder output layer of a teacher model.
        Format is [batch, frames, p]
    lab : torch.Tensor
        The target label from dataset, of shape [batch, frames]
    prob_length : torch.Tensor
        Length of each utterance.
    lab_length : torch.Tensor
        Length of each target label sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.

    Example
    -------
    >>> prob = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> lab = torch.tensor([[1, 1]])
    >>> prob_length = torch.tensor([1.])
    >>> lab_length = torch.tensor([1.])
    >>> blank_index = 0
    >>> wer_ctc(prob, lab, prob_length, lab_length, blank_index)
    tensor([[50.]])
    """
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

        actual_size_lab = int(torch.round(lab_length[j] * lab_curr.shape[0]))

        # prob_curr = prob_curr.narrow(-1, 0, actual_size_prob)
        lab_curr = lab_curr.narrow(-1, 0, actual_size_lab)

        # Computing the wer
        wer = compute_wer(
            prob_curr,
            lab_curr,
            prob_length[j],
            ctc_label=True,
            blank_index=blank_index,
        )
        wer_list.append(wer)

    wer_list = torch.tensor(wer_list)
    wer_list = torch.unsqueeze(wer_list, 0)
    return wer_list


def compute_wer(
    prob, lab, length=None, ctc_label=False, blank_index=None, eos_index=None
):
    """Compute word (or character) error rate for one sentence.

    Arguments
    ---------
    prob : torch.Tensor
        The probabilities from a teacher model.
        Format is [batch, frames, p]
    lab : torch.Tensor
        The target label from dataset, of shape [batch, frames]
    length : torch.Tensor
        Length of one utterance.
    ctc_label : Bool
        The flag if prob from encoder (CTC loss)
    blank_index : int
        The location of the blank symbol among the character indexes.
    eos_index : int
        The location of the eos symbol among the character indexes.

    Example
    -------
    >>> prob = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> lab = torch.tensor([[1, 1]])
    >>> length = torch.tensor([1])
    >>> compute_wer(prob, lab, length, eos_index=0)
    tensor([100.])
    """

    if ctc_label:
        scores, predictions = torch.max(prob, dim=-1)

        actual_size = int(torch.round(length * predictions.shape[-1]))
        predictions = predictions[0:actual_size]

        # Converting labels and prediction to lists (faster)
        lab = lab.tolist()
        predictions = predictions.tolist()

        predictions = filter_ctc_output(predictions, blank_id=blank_index)

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

        # filter predictions
        predictions = filter_seq2seq_output(predictions, eos_id=eos_index)

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


def filter_seq2seq_output(string_pred, eos_id=-1):
    """Extract the strings before eos.

    Arguments
    ---------
    string_pred : list
        Predictions of symbol indexes for one sentence.
    eos_id : int
        The location of the eos symbol among the character indexes.
    """
    if isinstance(string_pred, list):
        # Finding the first eos token
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]

        return string_out
