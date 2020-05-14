"""A popular speech model.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020,
    Titouan Parcollet 2020, Abdel 2020
"""
import os
import torch  # noqa: F401
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import Pooling
from speechbrain.nnet.activations import Softmax
from speechbrain.nnet.containers import Sequential, Nerve


class CRDNN(Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    The default CNN model is based on VGG.

    Arguments
    ---------
    output_size : int
        The length of the output (number of target classes).
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_overrides : mapping
        Additional parameters overriding the CNN parameters.
    rnn_blocks : int
        The number of recurrent neural blocks to include.
    rnn_overrides : mapping
        Additional parameters overriding the RNN parameters.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_overrides : mapping
        Additional parameters overriding the DNN parameters.

    CNN Block Parameters
    --------------------
        .. include:: cnn_block.yaml

    RNN Block Parameters
    --------------------
        .. include:: rnn_block.yaml

    DNN Block Parameters
    --------------------
        .. include:: dnn_block.yaml

    Example
    -------
    >>> model = CRDNN(output_size=40)
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 116, 40])
    """

    def __init__(
        self,
        output_size,
        cnn_blocks=1,
        cnn_overrides={},
        cnn_shortcuts="",
        rnn_blocks=1,
        rnn_overrides={},
        rnn_shortcuts="",
        dnn_blocks=1,
        dnn_overrides={},
        dnn_shortcuts="",
        time_pooling=False,
        time_pooling_stride=2,
        time_pooling_size=2,
    ):
        blocks = []

        model_dir = os.path.dirname(os.path.abspath(__file__))
        blocks.append(
            Nerve(
                param_file=os.path.join(model_dir, "cnn_block.yaml"),
                overrides=cnn_overrides,
                copies=cnn_blocks,
                shortcuts=cnn_shortcuts,
            )
        )

        if time_pooling:
            blocks.append(
                Pooling(
                    pool_type="max",
                    stride=time_pooling_stride,
                    kernel_size=time_pooling_size,
                    pool_axis=1,
                )
            )

        blocks.append(
            Nerve(
                param_file=os.path.join(model_dir, "rnn_block.yaml"),
                overrides=rnn_overrides,
                copies=rnn_blocks,
                shortcuts=rnn_shortcuts,
            )
        )

        blocks.append(
            Nerve(
                param_file=os.path.join(model_dir, "dnn_block.yaml"),
                overrides=dnn_overrides,
                copies=dnn_blocks,
                shortcuts=dnn_shortcuts,
            )
        )

        blocks.append(Linear(output_size, bias=False))
        blocks.append(Softmax(apply_log=True))

        super().__init__(*blocks)
