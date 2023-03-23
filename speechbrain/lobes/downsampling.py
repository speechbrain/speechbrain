"""
Combinations of processing algorithms to implement downsampling methods.

Authors
 * Salah Zaiem
"""
import torch
import torchaudio.transforms as T
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.pooling import Pooling1d


class Downsampler(torch.nn.Module):
    """ Wrapper for downsampling techniques
    """

    def forward(self, x):
        """ Downsampling function
        Arguments
        ---------
        x : tensor
            Speech samples of shape [B,n_samples] with B the batch size
        """

        return self.downsampler(x)


class SignalDownsampler(Downsampler):
    """Signal downsampling (Decimation)

    Arguments
    ---------
    downsampling_factor : int
        Factor of downsampling (i.e. ratio (length before ds / length after ds))
    initial_sampling_rate : int
        Sampling_rate of the input audios

    Example
    -------
    >>> sd = SignalDownsampler(2,16000)
    >>> a = torch.rand([8,28000])
    >>> a = sd(a)
    >>> print(a.shape)
    torch.Size([8, 14000])
    """

    def __init__(self, downsampling_factor, initial_sampling_rate):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.target_ds_rate = int(initial_sampling_rate / downsampling_factor)
        self.downsampler = T.Resample(
            initial_sampling_rate, self.target_ds_rate, dtype=torch.float32
        )


class Conv1DDownsampler(Downsampler):
    """1D Convolutional downsampling with a learned convolution

    Arguments
    ---------
    downsampling_factor : int
        Factor of downsampling (i.e. ratio (length before ds / length after ds))
    kernel_size : int
        Kernel size of the 1D filter (must be an odd integer)
    Example
    -------
    >>> sd = Conv1DDownsampler(3,161)
    >>> a = torch.rand([8,33000])
    >>> a = sd(a)
    >>> print(a.shape)
    torch.Size([8, 10947])
    """

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
    """1D Pooling downsampling (non-learned)

    Arguments
    ---------
    downsampling_factor : int
        Factor of downsampling (i.e. ratio (length before ds / length after ds))
    kernel_size : int
        Kernel size of the 1D filter (must be an odd integer)
    padding : int
        The number of padding elements to apply.
    pool_type : string
        Pooling approach, must be within ["avg","max"]
    Example
    -------
    >>> sd = PoolingDownsampler(3,41)
    >>> a = torch.rand([8,33000])
    >>> a = sd(a)
    >>> print(a.shape)
    torch.Size([8, 10987])
    """

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
