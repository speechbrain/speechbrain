"""Library implementing time delayed neural networks.

Author
    Nauman Dawalatabad 2020, Dannynis 2020
"""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


class TDNN(nn.Module):
    """This function implements a Time Delayed Neural Network (TDNN).

    This class implements TDNN layer

    Arguments
    ---------
    context: list
        It is the list representing indices of the context for the 1d convolution.
    input_dim: int
        It is the dimensionality of the input.
    output_dim: int
        it is the output of TDNN layer (Realtes to number of filters used)
    full_context: bool
        if True, a full context as per context is used.
    device: str

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 20])
    >>> tdnn = TDNN(context=[-2, 2], input_dim=20, output_dim=512)
    >>> out_tensor = tdnn(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 96, 512])
    """

    def __init__(
        self, context, input_dim, output_dim, full_context=True, device="cpu"
    ):
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._check_valid_context(context)
        self.kernel_width, context = self._get_kernel_width(
            context, full_context
        )
        self.register_buffer("context", torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1.0 / math.sqrt(input_dim)
        self.device = device
        self.kernel = nn.Parameter(
            torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(
                0, stdv
            )
        ).to(device)
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0, stdv)).to(
            device
        )

    def forward(self, x):
        # x shape is standard [batch, time_steps, features]
        # conv output shape is [batch, filters, conv]
        conv_out = self._selective_convolution(
            x, self.kernel, self.context, self.bias
        )

        # activation shape is [batch, conv, filters]
        activation = F.relu(conv_out).transpose(1, 2).contiguous()

        return activation

    def _selective_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context.
        Cannot directly use convolution because in case of only particular frames
        of context,one needs to select only those frames and perform a convolution
        across all batch items and all output dimensions of the kernel.
        """
        input_size = x.shape
        assert (
            len(input_size) == 3
        ), "Input tensor dimensionality is incorrect. Should be a 3D tensor"
        [batch_size, input_sequence_length, input_dim] = input_size

        x = x.transpose(1, 2).contiguous()

        # Allocate memory for output
        valid_steps = self._get_valid_steps(self.context, input_sequence_length)
        xs = self.bias.data.new(batch_size, kernel.shape[0], len(valid_steps))

        # Perform the convolution with relevant input frames
        # shifting blockwise: c is starting index and i is end index
        for c, i in enumerate(valid_steps):
            # set of feature vectors (context+i) to be convolved with the kernel
            features = torch.index_select(x, 2, context + i)

            # features = [batch, dim, context]; kernel= [out_dim, in_dim, kernel_width]; bias= [out_dim]
            xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
        return xs

    @staticmethod
    def _check_valid_context(context):
        """
        Check if the context provided is valid
        """
        assert context[0] <= context[-1], "Invalid context"

    @staticmethod
    def _get_kernel_width(context, full_context):
        """
        Returns the kernel width
        """
        if full_context:
            context = range(context[0], context[-1] + 1)
        return len(context), context

    @staticmethod
    def _get_valid_steps(context, input_sequence_length):
        """
        Returns the start and end points in the sequence
        """
        start = 0 if context[0] >= 0 else -1 * context[0]
        end = (
            input_sequence_length
            if context[-1] <= 0
            else input_sequence_length - context[-1]
        )
        return range(start, end)
