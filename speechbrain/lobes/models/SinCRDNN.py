"""A popular speech model.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020,
    Titouan Parcollet 2020, Abdel 2020
"""
import os
import torch  # noqa: F401
from speechbrain.nnet.pooling import Pooling
from speechbrain.nnet.containers import Sequential, ReplicateBlock


class SinCRDNN(Sequential):
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
    """

    def __init__(
        self,
        sinc_blocks=1,
        sinc_overrides={},
        cnn_blocks=1,
        cnn_overrides={},
        rnn_blocks=1,
        rnn_overrides={},
        dnn_blocks=1,
        dnn_overrides={},
        time_pooling=False,
        time_pooling_stride=2,
        time_pooling_size=2,
    ):
        blocks = []

        model_dir = os.path.dirname(os.path.abspath(__file__))

        blocks.append(
            ReplicateBlock(
                param_file=os.path.join(model_dir, "sincnet_block.yaml"),
                yaml_overrides=sinc_overrides,
                replication_count=sinc_blocks,
            )
        )

        # blocks.append(
        #    ReplicateBlock(
        #        param_file=os.path.join(model_dir, "cnn_block.yaml"),
        #        yaml_overrides=cnn_overrides,
        #        replication_count=cnn_blocks,
        #    )
        # )

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
            ReplicateBlock(
                param_file=os.path.join(model_dir, "rnn_block.yaml"),
                yaml_overrides=rnn_overrides,
                replication_count=rnn_blocks,
            )
        )

        blocks.append(
            ReplicateBlock(
                param_file=os.path.join(model_dir, "dnn_block.yaml"),
                yaml_overrides=dnn_overrides,
                replication_count=dnn_blocks,
            )
        )

        super().__init__(*blocks)
