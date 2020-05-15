"""Library implementing transducer_joint.

Author
    Abdelwahab HEBA 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Transducer_joint(nn.Module):
    """Computes joint between Transcription network & Prediction network
     
    Arguments
    ---------
    fusion : oneof("sum","concat") fusion option
        it is the dim of embedding (i.e, the dimensionality of the output)
    
    Example
    -------
    >>> TJoint = transducer_joint(joint="concat")
    >>> input_TN = torch.randn((8,200,1,40))
    >>> input_PN = torch.randn((8,1,12,40))
    >>> output = TJoint(input_TN,input_PN)
    >>> output.shape
    torch.Size([8, 200, 12 , 80])

    """

    def __init__(
        self,
        joint="sum"
    ):
        assert joint == "sum" or joint == "concat", "fusion must be one of ('sum','concat')"
        super().__init__()
        self.joint=joint

    def forward(self, input_TN, input_PN):
        """Returns the fusion of inputs tensors.

        Arguments
        ---------
        input_TN: torch.Tensor
           input from Transcription Network.

        input_PN: torch.Tensor
           input from Prediction Network.
        """
        if self.joint=="sum":
            return torch.tanh(input_TN+input_PN)
        if self.joint=="concat":
            if len(input_TN.shape)==4:
                dim =len(input_TN.shape)-1
                xs=input_TN
                ymat=input_PN
                sz = [max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])]
                xs = xs.expand(torch.Size(sz+[xs.shape[-1]])); ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
                joint = torch.cat((xs,ymat),dim=dim)
                return torch.tanh(joint)
            else:
                # for decoding, inputs shape will be (Batch,x_hidden)
                return torch.tanh(torch.cat((input_TN,input_PN),dim=0))