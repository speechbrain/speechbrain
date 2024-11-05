"""Basic feature pipelines.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Sarthak Yadav 2020
 * Sylvain de Langen 2024
"""

from dataclasses import dataclass
from typing import Optional

import torch

from speechbrain.nnet.CNN import GaborConv1d
from speechbrain.nnet.normalization import PCEN
from speechbrain.nnet.pooling import GaussianLowpassPooling
from speechbrain.processing.features import (
    DCT,
    STFT,
    ContextWindow,
    Deltas,
    Filterbank,
    spectral_magnitude,
)
from speechbrain.utils.autocast import fwd_default_precision
from speechbrain.utils.filter_analysis import FilterProperties


class Fbank(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: False)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: False)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 160000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 40)
        Number of Mel filters.
    filter_shape : str (default: triangular)
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor : float (default: 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor : float (default: 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default: 5)
        Number of frames of left context to add.
    right_frames : int (default: 5)
        Number of frames of right context to add.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=40,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames,
            right_frames=right_frames,
        )

    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals to transform to features.

        Returns
        -------
        fbanks : torch.Tensor
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)
        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            delta2 = self.compute_deltas(delta1)
            fbanks = torch.cat([fbanks, delta1, delta2], dim=2)
        if self.context:
            fbanks = self.context_window(fbanks)
        return fbanks

    def get_filter_properties(self) -> FilterProperties:
        # only the STFT affects the FilterProperties of the Fbank
        return self.compute_STFT.get_filter_properties()


class MFCC(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: True)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: True)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 16000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 23)
        Number of filters to use for creating filterbank.
    n_mfcc : int (default: 20)
        Number of output coefficients
    filter_shape : str (default 'triangular')
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor: bool (default 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor: float (default 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default 5)
        Number of frames of left context to add.
    right_frames : int (default 5)
        Number of frames of right context to add.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 660])
    """

    def __init__(
        self,
        deltas=True,
        context=True,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=23,
        n_mfcc=20,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_dct = DCT(input_size=n_mels, n_out=n_mfcc)
        self.compute_deltas = Deltas(input_size=n_mfcc)
        self.context_window = ContextWindow(
            left_frames=left_frames,
            right_frames=right_frames,
        )

    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, wav):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals to transform to features.

        Returns
        -------
        mfccs : torch.Tensor
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)
        mfccs = self.compute_dct(fbanks)
        if self.deltas:
            delta1 = self.compute_deltas(mfccs)
            delta2 = self.compute_deltas(delta1)
            mfccs = torch.cat([mfccs, delta1, delta2], dim=2)
        if self.context:
            mfccs = self.context_window(mfccs)
        return mfccs


class Leaf(torch.nn.Module):
    """
    This class implements the LEAF audio frontend from

    Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc. of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    window_len: float
        length of filter window in milliseconds
    window_stride : float
        Stride factor of the filters in milliseconds
    sample_rate : int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    input_shape : tuple
        Expected shape of the inputs.
    in_channels : int
        Expected number of input channels.
    min_freq : float
        Lowest possible frequency (in Hz) for a filter
    max_freq : float
        Highest possible frequency (in Hz) for a filter
    use_pcen: bool
        If True (default), a per-channel energy normalization layer is used
    learnable_pcen: bool:
        If True (default), the per-channel energy normalization layer is learnable
    use_legacy_complex: bool
        If False, torch.complex64 data type is used for gabor impulse responses
        If True, computation is performed on two real-valued torch.Tensors
    skip_transpose: bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    n_fft: int
        Number of FFT bins

    Example
    -------
    >>> inp_tensor = torch.rand([10, 8000])
    >>> leaf = Leaf(
    ...     out_channels=40, window_len=25., window_stride=10., in_channels=1
    ... )
    >>> out_tensor = leaf(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        out_channels,
        window_len: float = 25.0,
        window_stride: float = 10.0,
        sample_rate: int = 16000,
        input_shape=None,
        in_channels=None,
        min_freq=60.0,
        max_freq=None,
        use_pcen=True,
        learnable_pcen=True,
        use_legacy_complex=False,
        skip_transpose=False,
        n_fft=512,
    ):
        super().__init__()
        self.out_channels = out_channels
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.complex_conv = GaborConv1d(
            out_channels=2 * out_channels,
            in_channels=in_channels,
            kernel_size=window_size,
            stride=1,
            padding="same",
            bias=False,
            n_fft=n_fft,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            use_legacy_complex=use_legacy_complex,
            skip_transpose=True,
        )

        self.pooling = GaussianLowpassPooling(
            in_channels=self.out_channels,
            kernel_size=window_size,
            stride=window_stride,
            skip_transpose=True,
        )
        if use_pcen:
            self.compression = PCEN(
                self.out_channels,
                alpha=0.96,
                smooth_coef=0.04,
                delta=2.0,
                floor=1e-12,
                trainable=learnable_pcen,
                per_channel_smooth_coef=True,
                skip_transpose=True,
            )
        else:
            self.compression = None
        self.skip_transpose = skip_transpose

    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, x):
        """
        Returns the learned LEAF features

        Arguments
        ---------
        x : torch.Tensor of shape (batch, time, 1) or (batch, time)
            batch of input signals. 2d or 3d tensors are expected.

        Returns
        -------
        outputs : torch.Tensor
        """

        if not self.skip_transpose:
            x = x.transpose(1, -1)

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        outputs = self.complex_conv(x)
        outputs = self._squared_modulus_activation(outputs)
        outputs = self.pooling(outputs)
        outputs = torch.maximum(
            outputs, torch.tensor(1e-5, device=outputs.device)
        )
        if self.compression:
            outputs = self.compression(outputs)
        if not self.skip_transpose:
            outputs = outputs.transpose(1, -1)
        return outputs

    def _squared_modulus_activation(self, x):
        x = x.transpose(1, 2)
        output = 2 * torch.nn.functional.avg_pool1d(
            x**2.0, kernel_size=2, stride=2
        )
        output = output.transpose(1, 2)
        return output

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = 1
        else:
            raise ValueError(
                "Leaf expects 2d or 3d inputs. Got " + str(len(shape))
            )
        return in_channels


def upalign_value(x, to: int) -> int:
    """If `x` cannot evenly divide `to`, round it up to the next value that
    can."""

    assert x >= 0

    if (x % to) == 0:
        return x

    return x + to - (x % to)


@dataclass
class StreamingFeatureWrapperContext:
    """Streaming metadata for the feature extractor. Holds some past context
    frames."""

    left_context: Optional[torch.Tensor]
    """Cached left frames to be inserted as left padding for the next chunk.
    Initially `None` then gets updated from the last frames of the current
    chunk.
    See the relevant `forward` function for details."""


class StreamingFeatureWrapper(torch.nn.Module):
    """Wraps an arbitrary filter so that it can be used in a streaming fashion
    (i.e. on a per-chunk basis), by remembering context and making "clever" use
    of padding.

    Arguments
    ---------
    module : torch.nn.Module
        The filter to wrap; e.g. a module list that constitutes a sequential
        feature extraction pipeline.
        The module is assumed to pad its inputs, e.g. the output of a
        convolution with a stride of 1 would end up with the same frame count
        as the input.
    properties : FilterProperties
        The effective filter properties of the provided module. This is used to
        determine padding and caching.
    """

    def __init__(self, module: torch.nn.Module, properties: FilterProperties):
        super().__init__()

        self.module = module
        self.properties = properties

        if self.properties.causal:
            raise ValueError(
                "Causal streaming feature wrapper is not yet supported"
            )

        if self.properties.dilation != 1:
            raise ValueError(
                "Dilation not yet supported in streaming feature wrapper"
            )

    def get_required_padding(self) -> int:
        """Computes the number of padding/context frames that need to be
        injected at the past and future of the input signal in the forward pass.
        """

        return upalign_value(
            (self.properties.window_size - 1) // 2, self.properties.stride
        )

    def get_output_count_per_pad_frame(self) -> int:
        """Computes the exact number of produced frames (along the time
        dimension) per input pad frame."""

        return self.get_required_padding() // self.properties.stride

    def get_recommended_final_chunk_count(self, frames_per_chunk: int) -> int:
        """Get the recommended number of zero chunks to inject at the end of an
        input stream depending on the filter properties of the extractor.

        The number of injected chunks is chosen to ensure that the filter has
        output frames centered on the last input frames.
        See also :meth:`~StreamingFeatureWrapper.forward`.

        Arguments
        ---------
        frames_per_chunk : int
            The number of frames per chunk, i.e. the size of the time dimension
            passed to :meth:`~StreamingFeatureWrapper.forward`.

        Returns
        -------
        Recommended number of chunks.
        """

        return (
            upalign_value(self.get_required_padding(), frames_per_chunk)
            // frames_per_chunk
        )

    def forward(
        self,
        chunk: torch.Tensor,
        context: StreamingFeatureWrapperContext,
        *extra_args,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Forward pass for the streaming feature wrapper.

        For the first chunk, 0-padding is inserted at the past of the input.
        For any chunk (including the first), some future frames get truncated
        and cached to be inserted as left context for the next chunk in time.

        For further explanations, see the comments in the code.

        Note that due to how the padding is implemented, you may want to call
        this with a chunk worth full of zeros (potentially more for filters with
        large windows) at the end of your input so that the final frames have a
        chance to get processed by the filter.
        See :meth:`~StreamingFeatureWrapper.get_recommended_final_chunk_count`.
        This is not really an issue when processing endless streams, but when
        processing files, it could otherwise result in truncated outputs.

        Arguments
        ---------
        chunk : torch.Tensor
            Chunk of input of shape [batch size, time]; typically a raw
            waveform. Normally, in a chunkwise streaming scenario,
            `time = (stride-1) * chunk_size` where `chunk_size` is the desired
            **output** frame count.
        context : StreamingFeatureWrapperContext
            Mutable streaming context object; should be reused for subsequent
            calls in the same streaming session.
        *extra_args : tuple
        **extra_kwargs : dict
            Args to be passed to he module.

        Returns
        -------
        torch.Tensor
            Processed chunk of shape [batch size, output frames]. This shape is
            equivalent to the shape of `module(chunk)`.
        """

        feat_pad_size = self.get_required_padding()
        num_outputs_per_pad = self.get_output_count_per_pad_frame()

        # consider two audio chunks of 6 samples (for the example), where
        # each sample is denoted by 1, 2, ..., 6
        # so chunk 1 is 123456 and chunk 2 is 123456
        if context.left_context is None:
            # for the first chunk we left pad the input by two padding's worth of zeros,
            # and truncate the right, so that we can pretend to have right padding and
            # still consume the same amount of samples every time
            #
            # our first processed chunk will look like:
            # 0000123456
            #         ^^ right padding (truncated)
            #   ^^^^^^ frames that some outputs are centered on
            # ^^ left padding (truncated)
            chunk = torch.nn.functional.pad(chunk, (feat_pad_size * 2, 0))
        else:
            # prepend left context
            #
            # for the second chunk ownwards, given the above example:
            # 34 of the previous chunk becomes left padding
            # 56 of the previous chunk becomes the first frames of this chunk
            # thus on the second iteration (and onwards) it will look like:
            # 3456123456
            #         ^^ right padding (truncated)
            #   ^^^^^^ frames that some outputs are centered on
            # ^^ left padding (truncated)
            chunk = torch.cat((context.left_context, chunk), 1)

        # our chunk's right context will become the start of the "next processed chunk"
        # plus we need left padding for that one, so make it double
        context.left_context = chunk[:, -feat_pad_size * 2 :]

        feats = self.module(chunk, *extra_args, **extra_kwargs)

        # truncate left and right context
        feats = feats[:, num_outputs_per_pad:-num_outputs_per_pad, ...]

        return feats

    def get_filter_properties(self) -> FilterProperties:
        return self.properties

    def make_streaming_context(self) -> StreamingFeatureWrapperContext:
        return StreamingFeatureWrapperContext(None)
