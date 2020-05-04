"""
Losses for training neural networks.

Author
------
Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class ComputeCost(nn.Module):
    """This function implements different cost functions for training neural
        networks. It supports NLL, MSE, L1 and CTC objectives.

    Arguments
    ---------
    cost_type: one of the following options
        "nll": negative log-likelihood cost.
        "mse": mean squared error between the prediction and the target.
        "l1": l1 distance between the prediction and the target.
        "ctc": connectionist temporal classification, this loss sums
            up all the possible alignments between targets and predictions.
        "error": classification error.
        "wer": word error rate, computed with the edit distance algorithm.
    avoid_pad: bool
        when True, the time steps corresponding to zero-padding
        are not included in the cost function computation.
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
    >>> for cost in out_cost:
    ...     cost.backward()
    """

    def __init__(
        self, cost_type, avoid_pad=None, allow_lab_diff=3, blank_index=None,
    ):
        super().__init__()
        self.cost_type = cost_type
        self.avoid_pad = avoid_pad
        self.allow_lab_diff = allow_lab_diff

        # if not specified, set avoid_pad to False
        if self.avoid_pad is None:
            self.avoid_pad = [False] * len(self.cost_type)

        # Adding cost functions is a list
        self.costs = []

        for cost_index, cost in enumerate(self.cost_type):

            if cost == "nll":
                self.costs.append(torch.nn.NLLLoss())

            if cost == "error":
                self.costs.append(self._compute_error)

            if cost == "mse":
                self.costs.append(nn.MSELoss())

            if cost == "l1":
                self.costs.append(nn.L1Loss())

            if cost == "ctc":
                self.blank_index = blank_index
                self.costs.append(nn.CTCLoss(blank=self.blank_index))
                self.avoid_pad[cost_index] = False

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

        target = target.to(prediction.device)

        # Regression case
        reshape = True

        if len(prediction.shape) == len(target.shape):
            reshape = False

        out_costs = []

        for i, cost in enumerate(self.costs):

            # Managing avoid_pad to avoid adding costs of padded time steps
            if self.avoid_pad[i]:

                # Getting the number of sentences in the minibatch
                N_snt = prediction.shape[0]

                # Loss initialization
                loss = 0

                # Loop over all the sentences of the minibatch
                for j in range(N_snt):

                    # Selecting sentence
                    prob_curr = prediction[j]
                    lab_curr = target[j]

                    # Avoiding padded time steps
                    actual_size = int(
                        torch.round(lengths[j] * lab_curr.shape[0])
                    )

                    prob_curr = prob_curr.narrow(0, 0, actual_size)
                    lab_curr = lab_curr.narrow(0, 0, actual_size)

                    if reshape:
                        lab_curr = lab_curr.long()

                    # Loss accumulation
                    loss = loss + cost(prob_curr, lab_curr)

                # Loss averaging
                loss = loss / N_snt

                # Appending current loss
                out_costs.append(loss)

            # Managing case in which we also include the cost of padded steps.
            # This can be use when the number of padding elements is small
            # (e.g, when we sort the sentences before creating the batches)
            else:

                # Reshaping
                prob_curr = prediction
                lab_curr = target

                # Managing ctc cost for sequence-to-sequence learning
                if self.cost_type[i] == "ctc":

                    # Cast lab_curr to int32 for using Cudnn computation
                    # In the case of using CPU training, int type is mondatory.
                    lab_curr = lab_curr.int()

                    # Permuting output probs
                    prob_curr = prob_curr.transpose(0, 1)

                    # Getting the actual lengths
                    input_lengths = torch.round(
                        lengths[0] * prob_curr.shape[0]
                    ).int()
                    lab_lengths = lengths[1] * target.shape[1]
                    lab_lengths = torch.round(lab_lengths).int()

                    # Compute CTC loss
                    ctc_cost = cost(
                        prob_curr, lab_curr, input_lengths, lab_lengths
                    )

                    out_costs.append(ctc_cost)

                else:

                    # Reshaping tensors when needed
                    if reshape:
                        lab_curr = target.reshape(
                            target.shape[0] * target.shape[1]
                        ).long()

                        prob_curr = prob_curr.reshape(
                            prob_curr.shape[0] * prob_curr.shape[1],
                            prob_curr.shape[2],
                        )

                    # Cost computation
                    out_costs.append(cost(prob_curr, lab_curr))

        if len(out_costs) == 1:
            out_costs = out_costs[0]

        return out_costs

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
