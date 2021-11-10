"""TO BE UPDATED

Authors
 * Francesco Paissan, 2021
"""
import torch
import speechbrain as sb
from numpy import ceil


class PodNet(torch.nn.Module):
    """
    TO BE UPDATED
    """

    @staticmethod
    def _create_pod(
        in_shape_time=1500,
        in_filters=1,
        pod_cnn_filter=(30, 1),
        pod_pool_kernel=(2, 2),
        pod_pool_stride=(2, 2),
        pod_pool_pad=1,
        pod_dp_rate=0.5,
        filters_pod=100,
        pod_activation=torch.nn.ReLU(),
        pod_id="1",
    ) -> torch.nn.Sequential:
        pod = torch.nn.Sequential()
        pod.add_module(
            f"pod_{pod_id}_conv",
            sb.nnet.CNN.Conv2d(
                in_channels=in_filters,
                out_channels=filters_pod,
                kernel_size=pod_cnn_filter,
                padding="valid",
                # padding_mode="constant",
                bias=False,
                transpose=True,
            ),
        )

        pod.add_module(f"pod_{pod_id}_dropout", torch.nn.Dropout(p=pod_dp_rate))

        pod.add_module(
            f"pod_{pod_id}_batchnorm",
            sb.nnet.normalization.BatchNorm2d(
                input_size=filters_pod, momentum=0.01, affine=True,
            ),
        )

        pod.add_module(f"pod_{pod_id}_act", pod_activation)

        pod.add_module(
            f"pod_{pod_id}_maxpool",
            sb.nnet.pooling.Pooling2d(
                pool_type="max",
                kernel_size=pod_pool_kernel,
                stride=pod_pool_stride,
                padding=pod_pool_pad,
                pool_axis=[1, 2],
            ),
        )

        return pod, ceil(float(in_shape_time - pod_cnn_filter[0]) / 2 + 1)

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        n_pods=3,
        dropout=0.5,
        filters_pod=100,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        T = input_shape[1]
        C = input_shape[2]
        self.conv_module = torch.nn.Sequential()

        out_shapes = []
        pod1, temp = self._create_pod(
            in_shape_time=T,
            in_filters=1,
            pod_cnn_filter=(30, C),
            pod_pool_kernel=(2, 2),
            pod_pool_stride=(2, 2),
            pod_pool_pad=1,
            pod_dp_rate=dropout,
            filters_pod=filters_pod,
            pod_activation=activation,
            pod_id="1",
        )
        out_shapes.append(temp)

        self.conv_module.add_module("pod0", pod1)

        for i in range(1, 1 + n_pods):
            pod_i, temp = self._create_pod(
                in_shape_time=out_shapes[-1],
                in_filters=filters_pod,
                pod_cnn_filter=(30, 1),
                pod_pool_kernel=(2, 2),
                pod_pool_stride=(2, 2),
                pod_pool_pad=1,
                pod_dp_rate=dropout,
                filters_pod=filters_pod,
                pod_activation=torch.nn.ReLU(),
                pod_id=i,
            )
            out_shapes.append(temp)
            self.conv_module.add_module(f"pod{i}", pod_i)

        pod_last, temp = self._create_pod(
            in_shape_time=out_shapes[-1],
            in_filters=filters_pod,
            pod_cnn_filter=(30, 1),
            pod_pool_kernel=(2, 2),
            pod_pool_stride=(2, 2),
            pod_pool_pad=1,
            pod_dp_rate=0,
            filters_pod=filters_pod,
            pod_activation=torch.nn.ReLU(),
            pod_id="last",
        )
        out_shapes.append(temp)
        self.conv_module.add_module("pod_last", pod_last)

        # # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )

        dense_input_size = int(out_shapes[-1]) * filters_pod
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=int(dense_input_size),
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)
        x = self.dense_module(x)

        return x
