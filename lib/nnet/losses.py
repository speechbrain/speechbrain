"""
 -----------------------------------------------------------------------------
 losses.py

 Description: This library implements different losses for training neural
              models.
 -----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from lib.utils.input_validation import check_opts, check_inputs
from lib.utils.logger import logger_write


class compute_cost(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.losses.compute_cost (author: Mirco Ravanelli)

     Description:  This function implements different cost functions for
                   training neural networks. It supports NLL, MSE, L1 and
                   CTC objectives.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - cost_type (one_of(nll,mse,l1,ctc,error), \
                               mandatory):
                               it is the type of cost used.

                               "nll": it is the standard negative
                               log-likelihood cost.

                               "mse": it is the mean squared error between
                                      the prediction and the target.

                               "l1":  it is the l1 distance between the
                                      prediction and the target.

                               "ctc":  it is the ctc function used for
                                        sequence-to-sequence learning. It sums
                                        up over all the possible alignments
                                        between targets and predictions.

                               "ctc":  it is the standard classification error.


                           - avoid_pad (bool, optional, Default: False):
                               when True, the time steps corresponding to
                               zero-padding are not included in the cost
                               function computation.

                          - allow_lab_diff (int(0,inf), Default:False):
                              It is the number of tollerable difference
                              between the label and prediction lengths.
                              Minimal differences can be tollerated and
                              could be due to different way of processing
                              the signal. Big differences and likely due
                              to an error.

                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing [predictions, target, lengths].
                       where prediction is the output of the neural network,
                       target is the label, while lengths contains the
                       percentage of valid time steps for each batch. The
                       latter can be used for removing zero-padded steps from
                       the cost computation.



     Output (call): - out_costs(type; lst)
                       This function returns a list of scalar torch.Tensor
                       elements that contain the cost function to optimize.



     Example:   import torch
                from lib.nnet.architectures import linear
                from lib.nnet.architectures import activation
                from lib.nnet.losses import compute_cost

                # Definition of a linear model
                inp_tensor = torch.rand([1,660,3])

                # config dictionary definition
                config={'class_name':'lib.nnet.architectures.linear',
                        'n_neurons':'4'}

                # Initialization of the linear class
                model=linear(config,first_input=[inp_tensor])


                # Definition of the log_softmax
                config={'class_name':'lib.nnet.architectures.activation',
                        'act_type':'log_softmax',
                        }

                # Initialization of the log_softmax class
                softmax=activation(config, first_input=[inp_tensor])


                # Definition of the loss
                config={'class_name':'lib.nnet.losses.compute_cost',
                        'cost_type':'nll'}

                # Initialization of the loss function
                cost=compute_cost(config)

                # Computatitions of the prediction for the current input
                pre_act=model([inp_tensor])
                pred = softmax([pre_act])

                # fake label
                label=torch.FloatTensor([0,1,3]).unsqueeze(0)
                lengths=torch.Tensor([1.0])

                out_cost= cost([pred,label,lengths])

                print(out_cost)

                # back propagation
                out_cost.backward()


     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(compute_cost, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "cost_type": ("one_of_list(nll,error,mse,l1,ctc)", "mandatory"),
            "avoid_pad": ("bool_list", "optional", "None"),
            "allow_lab_diff": ("int(0,inf)", "optional", "3"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = [
            "torch.Tensor",
            "torch.Tensor",
            ["torch.Tensor", "list"],
        ]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

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
                self.blank_index = first_input[0].shape[1] - 1
                self.costs.append(nn.CTCLoss(blank=self.blank_index))
                self.avoid_pad[cost_index] = False

    def forward(self, input_lst):

        # Reading input prediction
        prob = input_lst[0]

        # Reading the targets
        lab = input_lst[1]

        # Reading the relative lengths
        lengths = input_lst[2]

        # Check on input and label shapes
        if "ctc" not in self.cost_type:

            # Shapes cannot be too different (max 3 time steps)
            if abs(prob.shape[-1] - lab.shape[-1]) > self.allow_lab_diff:
                err_msg = (
                    "The length of labels differs from the length of the "
                    "output probabilities. (Got %i vs %i)"
                    % (lab.shape[-1], prob.shape[-1])
                )

                logger_write(err_msg, logfile=self.logger)

            prob = prob[:, :, 0: lab.shape[-1]]

        else:

            if not isinstance(lengths, list):

                err_msg = (
                    "The third input to the compute_cost function must "
                    "be a list [wav_len, lab_len] when ctc is the cost. "
                )

                logger_write(err_msg, logfile=self.logger)

        # Adding signal to gpu or cpu
        prob = prob.to(prob.device)
        lab = lab.to(prob.device)

        # Regression case
        reshape = True

        if len(prob.shape) == len(lab.shape):
            reshape = False

        out_costs = []

        # Loop over all the cost specified in self.costs
        for i, cost in enumerate(self.costs):

            # Managing avoid_pad to avoid adding costs of padded time steps
            if self.avoid_pad[i]:

                # Getting the number of sentences in the minibatch
                N_snt = prob.shape[0]

                # Loss initialization
                loss = 0

                # Loop over all the sentences of the minibatch
                for i in range(N_snt):

                    # Selecting sentence
                    prob_curr = prob[i]
                    lab_curr = lab[i]

                    # Avoiding padded time steps
                    actual_size = int(
                        torch.round(lengths[i] * lab_curr.shape[0])
                    )

                    prob_curr = prob_curr.narrow(-1, 0, actual_size)
                    lab_curr = lab_curr.narrow(-1, 0, actual_size)

                    # Reshaping
                    if reshape:
                        prob_curr = prob_curr.transpose(0, 1)
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
                prob_curr = prob
                lab_curr = lab

                # Managing ctc cost for sequence-to-sequence learning
                if self.cost_type[i] == "ctc":

                    prob_curr = prob_curr.permute(2, 0, 1)
                    input_lengths = torch.round(
                        lengths[0] * prob_curr.shape[0]
                    ).int()
                    lab_lengths = torch.round(lengths[1] * lab.shape[-1]).int()

                    # ctc cost computation
                    ctc_cost = cost(
                        prob_curr, lab_curr, input_lengths, lab_lengths
                    )

                    out_costs.append(ctc_cost)

                else:

                    # Reshaping tensors when needed
                    if reshape:
                        lab_curr = lab.reshape(
                            lab.shape[0] * lab.shape[1]
                        ).long()
                        prob_curr = prob.transpose(1, 2)
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
                    from lib.nnet.losses import compute_cost


                    # Definition of the loss
                    config={'class_name':'lib.nnet.losses.compute_cost',
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
