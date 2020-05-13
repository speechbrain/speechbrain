"""
Losses for training neural networks.

Author
------
Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn
from speechbrain.data_io.data_io import length_to_mask

logger = logging.getLogger(__name__)


class ComputeCost(nn.Module):
    """This function implements different losses for training neural
        networks. It supports NLL, MSE, L1 and CTC objectives.

    Arguments
    ---------
    cost_type: one of the following options. let's fix it!
        "nll": negative log-likelihood cost.
        "mse": mean squared error between the prediction and the target.
        "l1": l1 distance between the prediction and the target.
        "ctc": connectionist temporal classification, this loss sums
            up all the possible alignments between targets and predictions.
        "error": classification error.
        "wer": word error rate, computed with the edit distance algorithm.
    allow_lab_diff: int
        the number of tolerated differences between the label
        and prediction lengths. Minimal differences can be tolerated and
        could be due to different way of processing the signal. Big
        differences are likely due to an error.

    Example
    -------
    >>> import torch
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.activations import Softmax
    >>> mock_input = torch.rand([1, 660, 3])
    >>> model = Linear(n_neurons=4)
    >>> softmax = Softmax(apply_log=True)
    >>> cost = ComputeCost(cost_type='nll')
    >>> pred = softmax(model(mock_input, init_params=True))
    >>> label = torch.FloatTensor([0,1,3]).unsqueeze(0)
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = cost(pred, label, lengths)
    >>> out_cost.backward()
    """

    def __init__(
        self, cost_type, allow_lab_diff=3, blank_index=None,
    ):
        super().__init__()
        self.allow_lab_diff = allow_lab_diff
        self.cost_type = cost_type

        if cost_type == "nll":
            self.cost = torch.nn.NLLLoss(reduction="none")

        if cost_type == "error":
            self.cost = self._compute_error

        if cost_type == "mse":
            self.cost = nn.MSELoss(reduction="none")

        if cost_type == "l1":
            self.cost = nn.L1Loss(reduction="none")

        if cost_type == "ctc":
            if blank_index is None:
                raise ValueError("Must pass blank index for CTC")
            self.blank_index = blank_index
            self.cost = nn.CTCLoss(blank=self.blank_index)

    def forward(self, prediction, target, lengths):
        """Returns the cost function given predictions and targets.

        Arguments
        ---------
        prediction : torch.Tensor
            tensor containing the posterior probabilities
        target : torch.Tensor
            tensor containing the targets
        lengths : torch.Tensor
            tensor containing the relative lengths of each sentence
        """

        # Check on input and label shapes
        self._check_inp(prediction, target, lengths)

        # Computing actual target and prediction lengths
        pred_len, target_len = self._compute_len(prediction, target, lengths)
        target = target.to(prediction.device)

        if self.cost_type == "ctc":
            target = target.int()
            prediction = prediction.transpose(0, 1)
            loss = self.cost(prediction, target, pred_len, target_len)

        else:
            # Mask to avoid zero-padded time steps from  the total loss
            mask = length_to_mask(target_len)

            if self.cost_type in ["nll", "error"]:
                prediction = prediction[:, 0 : target.shape[1], :]
                prediction = prediction.reshape(
                    prediction.shape[0] * prediction.shape[1],
                    prediction.shape[2],
                )
                target = target.reshape(
                    target.shape[0] * target.shape[1]
                ).long()
                mask = mask.reshape(mask.shape[0] * mask.shape[1])

            if self.cost_type in ["mse", "l1"]:
                mask = mask.unsqueeze(2).repeat(1, 1, target.shape[2])

            loss = self.cost(prediction, target) * mask
            loss = torch.sum(loss * mask) / torch.sum(mask)

        return loss

    def _check_inp(self, prediction, target, lengths):
        """Peforms some check on prediction, targets and lengths.

        Arguments
        ---------
        prediction : torch.Tensor
            tensor containing the posterior probabilities
        target : torch.Tensor
            tensor containing the targets
        lengths : torch.Tensor
            tensor containing the relative lengths of each sentence
        """

        if "ctc" not in self.cost_type:

            # Shapes cannot be too different (max 3 time steps)
            diff = abs(prediction.shape[1] - target.shape[1])
            if diff > self.allow_lab_diff:
                err_msg = (
                    "The length of labels differs from the length of the "
                    "output probabilities. (Got %i vs %i)"
                    % (target.shape[1], prediction.shape[1])
                )

                logger.error(err_msg, exc_info=True)
            prediction = prediction[:, 0 : target.shape[1], :]

        else:

            if not isinstance(lengths, list):
                err_msg = (
                    "The third input to the compute_cost function must "
                    "be a list [wav_len, lab_len] when ctc is the cost. "
                )

                logger.error(err_msg, exc_info=True)

        # Regression case (no reshape)
        self.reshape = True
        if len(prediction.shape) == len(target.shape):
            self.reshape = False

    def _compute_len(self, prediction, target, lengths):
        """Compute the actual length of prediction and targets given
        the relative length tensor.

        Arguments
        ---------
        prediction : torch.Tensor
            tensor containing the posterior probabilities
        target : torch.Tensor
            tensor containing the targets
        lengths : torch.Tensor
            tensor containing the relative lengths of each sentence

        Returns
        -------
        pred_len : torch.Tensor
            tensor contaning the length of each sentence in the batch
        target_len : torch.Tensor
            tensor contaning the length of each target in the batch
        """

        if isinstance(lengths, list) and len(lengths) == 2:
            pred_len, target_len = lengths
        else:
            pred_len = target_len = lengths

        pred_len = torch.round(pred_len * prediction.shape[1]).int()
        target_len = torch.round(target_len * target.shape[1]).int()

        return pred_len, target_len

    def _compute_error(self, prob, lab):
        """Computes the classification error at frame level.

        Arguments
        ---------
        prob : torch.Tensor
            It is the tensor containing the posterior probabilities
            as [batch,prob]
        lab : torch.Tensor
            tensor containing the targets ([batch])

        """

        predictions = torch.max(prob, dim=-1)[1]
        error = torch.mean((predictions != lab).float())

        return error
