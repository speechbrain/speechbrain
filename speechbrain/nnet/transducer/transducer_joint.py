"""Library implementing transducer_joint.

Author
    Abdelwahab HEBA 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Transducer_joint(nn.Module):
    """Computes joint tensor between Transcription network & Prediction network

    Arguments
    ---------
    joint_network: if joint == "concat" Neural Modules applied after the joint
                   else None (not used if joint=="sum")
    joint : joint the two tensors by ("sum",or "concat") option
    nonlinearity: str
         Type of nonlinearity (tanh, relu)

    Example
    -------
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> from speechbrain.nnet.linear import Linear
    >>> joint_network = Linear(80)
    >>> TJoint = Transducer_joint(joint_network, joint="concat", nonlinearity="tanh")
    >>> input_TN = torch.randn((8, 200, 1, 40))
    >>> input_PN = torch.randn((8, 1, 12, 40))
    >>> output = TJoint(input_TN, input_PN, init_params=True)
    >>> output.shape
    torch.Size([8, 200, 12, 80])
    """

    def __init__(self, joint_network=None, joint="sum", nonlinearity="tanh"):
        super().__init__()
        self.joint_network = joint_network
        self.joint = joint
        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = torch.nn.ReLU()
        elif nonlinearity == "leakyrelu":
            self.nonlinearity = torch.nn.LeakyReLU()

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self.joint_network(first_input, init_params=True)

    def forward(self, input_TN, input_PN, init_params=False):
        """Returns the fusion of inputs tensors.

        Arguments
        ---------
        input_TN: torch.Tensor
           input from Transcription Network.

        input_PN: torch.Tensor
           input from Prediction Network.
        """
        assert input_TN.shape != 4, "Arg 1 must be a 4 dim tensor"
        assert input_PN.shape != 4, "Arg 2 must be a 4 dim tensor"

        if self.joint == "sum":
            joint = input_TN + input_PN

        if self.joint == "concat":
            if len(input_TN.shape) == 4:
                dim = len(input_TN.shape) - 1
                xs = input_TN
                ymat = input_PN
                sz = [
                    max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])
                ]
                xs = xs.expand(torch.Size(sz + [xs.shape[-1]]))
                ymat = ymat.expand(torch.Size(sz + [ymat.shape[-1]]))
                joint = torch.cat((xs, ymat), dim=dim)

            # use a joint_network it we do a concat
            if init_params:
                self.init_params(joint)

            joint = self.joint_network(joint)

        return self.nonlinearity(joint)
