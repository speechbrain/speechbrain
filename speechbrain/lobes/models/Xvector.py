"""A popular speaker recognition and diarization model.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""

# import os
import torch  # noqa: F401
import speechbrain as sb


class Xvector(sb.nnet.Sequential):
    """This model extracts XVectors for speaker recognition and diarization.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dialations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 24])
    >>> compute_xvect = Xvector(input_shape=input_feats.shape)
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
    ):

        blocks = []

        # TDNN layers
        for block_index in range(tdnn_blocks):
            blocks.extend(
                [
                    lambda input_shape: sb.nnet.Conv1d(
                        input_shape=input_shape,
                        out_channels=tdnn_channels[block_index],
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    sb.nnet.BatchNorm1d,
                ]
            )

        # Statistical pooling
        blocks.append(sb.nnet.StatisticsPooling())

        # Final linear transformation
        blocks.append(
            lambda input_shape: sb.nnet.Linear(
                input_shape=input_shape,
                n_neurons=lin_neurons,
                bias=True,
                combine_dims=False,
            )
        )

        super().__init__(input_shape, *blocks)


class Classifier(sb.nnet.Sequential):
    """This class implements the last MLP on the top of xvector features.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 24])
    >>> compute_xvect = Xvector(input_shape=input_feats.shape)
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):
        blocks = []

        blocks.extend([activation(), sb.nnet.BatchNorm1d])

        for block_index in range(lin_blocks):
            blocks.extend(
                [
                    lambda input_shape: sb.nnet.Linear(
                        n_neurons=lin_neurons,
                        input_shape=input_shape,
                        bias=True,
                        combine_dims=False,
                    ),
                    activation(),
                    sb.nnet.BatchNorm1d,
                ]
            )

        # Final Softmax classifier
        blocks.extend(
            [
                sb.nnet.Linear(
                    input_size=lin_neurons, n_neurons=out_neurons, bias=True
                ),
                sb.nnet.Softmax(apply_log=True),
            ]
        )

        super().__init__(input_shape, *blocks)
