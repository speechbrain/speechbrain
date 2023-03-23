import torch
import torchaudio.transforms as T
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.pooling import Pooling1d


class Downsampler(torch.nn.Module):
    def forward(self, x):
        return self.downsampler(x)


class SignalDownsampler(Downsampler):
    def __init__(self, downsampling_factor, initial_sampling_rate):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.target_ds_rate = int(initial_sampling_rate / downsampling_factor)
        self.downsampler = T.Resample(
            initial_sampling_rate, self.target_ds_rate, dtype=torch.float32
        )


class Conv1DDownsampler(Downsampler):
    def __init__(self, downsampling_factor, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.downsampling_factor = downsampling_factor
        self.downsampler = Conv1d(
            stride=self.downsampling_factor,
            padding="valid",
            kernel_size=self.kernel_size,
            out_channels=1,
            input_shape=[None, None],
        )


class PoolingDownsampler(Downsampler):
    def __init__(
        self, downsampling_factor, kernel_size, padding=0, pool_type="avg"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_type = pool_type
        self.downsampling_factor = downsampling_factor
        self.downsampler = Pooling1d(
            stride=self.downsampling_factor,
            padding=self.padding,
            kernel_size=self.kernel_size,
            input_dims=3,
            pool_type=self.pool_type,
        )
