"""
Losses for training neural networks.

Author
------
Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn
from speechbrain.utils.edit_distance import accumulatable_wer_stats
from speechbrain.decoders.ctc import filter_ctc_output

logger = logging.getLogger(__name__)


class compute_cost(nn.Module):
    """
    -------------------------------------------------------------------------
    Description:
        This function implements different cost functions for training neural
        networks. It supports NLL, MSE, L1 and CTC objectives.

    Args:
        cost_type: one of the following options
            "nll": negative log-likelihood cost.
            "mse": mean squared error between the prediction and the target.
            "l1": l1 distance between the prediction and the target.
            "ctc": connectionist temporal classification, this loss sums
                up all the possible alignments between targets and predictions.
            "error": classification error.
            "wer": word error rate, computed with the edit distance algorithm.
        avoid_pad: when True, the time steps corresponding to zero-padding
            are not included in the cost function computation.
        allow_lab_diff: the number of tolerated differences between the label
            and prediction lengths. Minimal differences can be tolerated and
            could be due to different way of processing the signal. Big
            differences are likely due to an error.

     Example:
        >>> import torch
        >>> from speechbrain.nnet.architectures import linear
        >>> from speechbrain.nnet.architectures import activation
        >>> mock_input = torch.rand([1, 660, 3])
        >>> model = linear(n_neurons=4)
        >>> model.init_params(mock_input)
        >>> softmax = activation(act_type='log_softmax')
        >>> cost = compute_cost(cost_type='nll')
        >>> pred = softmax(model(mock_input))
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
                self.costs.append(self.compute_error)

            if cost == "mse":
                self.costs.append(nn.MSELoss())

            if cost == "l1":
                self.costs.append(nn.L1Loss())

            if cost == "ctc":
                self.blank_index = blank_index
                self.costs.append(nn.CTCLoss(blank=self.blank_index))
                self.avoid_pad[cost_index] = False

            if cost == "wer":
                self.costs.append(self.compute_wer)

    def forward(self, prediction, target, lengths):
        """
        Input: - prediction (type: torch.Tensor, mandatory)
                   the output of the neural network,

               - target (type: torch.Tensor, mandatory)
                   the label

               - lengths (type: torch.Tensor, mandatory)
                   the percentage of valid time steps for each batch.
                   Can be used for removing zero-padded steps from
                   the cost computation.

        Output: - out_costs(type; lst)
                   This function returns a list of scalar torch.Tensor
                   elements that contain the cost function to optimize.
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

        # Adding signal to gpu or cpu
        prediction = prediction.to(prediction.device)
        target = target.to(prediction.device)

        # Regression case
        reshape = True

        if len(prediction.shape) == len(target.shape):
            reshape = False

        out_costs = []

        # Loop over all the cost specified in self.costs
        for i, cost in enumerate(self.costs):

            if self.cost_type[i] == "wer":

                # Getting the number of sentences in the minibatch
                N_snt = prediction.shape[0]

                # Wer initialization
                wer = 0

                # Loop over all the sentences of the minibatch
                for j in range(N_snt):

                    # getting the current probabilities and labels
                    prob_curr = prediction[j]
                    lab_curr = target[j]

                    # Avoiding padded time steps
                    actual_size_prob = int(
                        torch.round(lengths[0][j] * prob_curr.shape[0])
                    )

                    actual_size_lab = int(
                        torch.round(lengths[1][j] * lab_curr.shape[0])
                    )

                    prob_curr = prob_curr.narrow(0, 0, actual_size_prob)
                    lab_curr = lab_curr.narrow(0, 0, actual_size_lab)

                    # Computing the wer
                    wer = wer + cost(prob_curr, lab_curr)

                # WER averaging
                wer = wer / N_snt

                # Appending current loss
                out_costs.append(wer)

                continue

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

                    # Reshaping
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
                    # cast lab_curr to int32 for using Cudnn computation
                    # In the case of using CPU training, int type is mondatory.
                    lab_curr = lab_curr.int()
                    # Permuting output probs
                    prob_curr = prob_curr.transpose(0, 1)

                    # Getting the input lengths
                    input_lengths = torch.round(
                        lengths[0] * prob_curr.shape[0]
                    ).int()

                    # Getting the label lengths
                    lab_lengths = lengths[1] * target.shape[1]
                    lab_lengths = torch.round(lab_lengths).int()

                    # CTC cost computation
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

    def compute_error(self, prob, lab):
        """
         ----------------------------------------------------------------------
         nnet.losses.compute_cost.compute_error (author: Mirco Ravanelli)

         Description:  This support function compute the classification error
                        given a set of posterior probabilities and a target.


         Input (call):
                        - prob (type:torch.Tensor, mandatory):
                            it is the tensor containing the posterior
                            probabilities in the following format:
                            [batch,prob]

                        - lab (type:torch.Tensor, mandatory):
                            it is the Floattensor containing the labels
                            in the following format:
                            [batch]

         Output (call): - error(type, torch.Tensor):
                           it is the classification error.


         Example:   import torch
                    from speechbrain.nnet.losses import compute_cost


                    # Definition of the loss
                    config={'class_name':'speechbrain.nnet.losses.compute_cost',
                            'cost_type':'nll'}

                    # Initialization of the loss function
                    cost=compute_cost(config)

                    # fake probabilities/labels
                    prob=torch.rand([4,3])
                    lab=torch.FloatTensor([1,0,2,0])
                    print(cost.compute_error(prob,lab))


         """
        # Computing predictions
        predictions = torch.max(prob, dim=-1)[1]

        # Computing classification error
        error = torch.mean((predictions != lab).float())

        return error

    def compute_wer(self, prob, lab):
        """
         ----------------------------------------------------------------------
         nnet.losses.compute_cost.compute_wer
         (authors: Aku Rouhe, Mirco Ravanelli)

         Description:  This support function computes the wer based on the
                       edit distance.


         Input (call):
                        - prob (type:torch.Tensor, mandatory):
                            it is the tensor containing the posterior
                            probabilities in the following format:
                            [batch,prob].

                        - lab (type:torch.Tensor, mandatory):
                            it is the FloatTensor containing the labels
                            in the following format:
                            [batch].


         Output (call): - wer(type, torch.Tensor):
                           it is the wer for the given input batch.


         Example:   import torch
                    from speechbrain.nnet.losses import compute_cost


                    # Definition of the loss
                    config={'class_name':'speechbrain.nnet.losses.compute_cost',
                            'cost_type':'nll'}

                    # Initialization of the loss function
                    cost=compute_cost(config)

                    # fake probabilities/labels
                    prob=torch.rand([4,3])
                    lab=torch.FloatTensor([1,0,2,0])
                    print(cost.compute_error(prob,lab))


         """
        # Computing predictions
        scores, predictions = torch.max(prob, dim=0)

        # Converting labels and prediction to lists (faster)
        lab = lab.tolist()
        predictions = predictions.tolist()

        # If the case of CTC, filter the predicted output
        if "ctc" in self.cost_type:
            predictions = filter_ctc_output(
                predictions, blank_id=self.blank_index
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
