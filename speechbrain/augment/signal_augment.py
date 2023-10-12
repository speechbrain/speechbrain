"""
Classes for Sequential Data Augmentation.

This module contains classes that generate sequence distortions for data augmentation.
For speech-specific augmentations, please refer to speechbrain.augment.speech_augment.

Authors:
    - Peter Plantinga (2020)
    - Mirco Ravanelli (2023)
"""

import torch


class DoClip(torch.nn.Module):
    """This function mimics audio clipping by clamping the input tensor.

    Arguments
    ---------
    clip_low : float
        The low end of amplitudes for which to clip the signal.
    clip_high : float
        The high end of amplitudes for which to clip the signal.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> clipper = DoClip(clip_low=0.01, clip_high=0.01)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> clipped_signal = clipper(signal.unsqueeze(0))
    >>> "%.2f" % clipped_signal.max()
    '0.01'
    """

    def __init__(self, clip_low=0.5, clip_high=1):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = torch.rand(1)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = waveforms.clamp(-clip_value, clip_value)

        return clipped_waveform


class RandAmp(torch.nn.Module):
    """This function multiples the signal by a random amplitude

    Arguments
    ---------
    amp_low : float
        The minumum amplitude multiplication factor.
    amp_high : float
        The maximum amplitude multiplication factor.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> rand_amp = RandAmp(amp_low=0.25, amp_high=1.75)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> output_signal = rand_amp(signal.unsqueeze(0))
    """

    def __init__(self, amp_low=0.5, amp_high=1.5):
        super().__init__()
        self.amp_low = amp_low
        self.amp_high = amp_high

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a frequency to drop
        rand_range = self.amp_high - self.amp_low
        amp = (
            torch.rand(waveforms.shape[0], device=waveforms.device) * rand_range
            + self.amp_low
        )
        amp = amp.unsqueeze(1)
        if len(waveforms.shape) == 3:
            amp = amp.unsqueeze(2)
        waveforms = waveforms * amp

        return waveforms


class ChannelDrop(torch.nn.Module):
    """This function drops random channels in the multi-channel nput waveform.

    Arguments
    ---------
    drop_rate : float
        The channel droput factor

    Example
    -------
    >>> signal = torch.rand(4, 256, 8)
    >>> ch_drop = ChannelDrop(drop_rate=0.5)
    >>> output_signal = ch_drop(signal)
    """

    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a frequency to drop
        waveforms = waveforms.detach().clone()
        x = torch.rand(waveforms.shape[-1], device=waveforms.device)
        channel_mask = x.ge(self.drop_rate)
        waveforms = waveforms * channel_mask.unsqueeze(0).unsqueeze(1)
        return waveforms


class ChannelSwap(torch.nn.Module):
    """This function randomly swaps N channels.

    Arguments
    ---------
    min_swap : int
        The mininum number of channels to swap.
    max_swap : int
        The maximum number of channels to swap.

    Example
    -------
    >>> signal = torch.rand(4, 256, 8)
    >>> ch_swap = ChannelSwap()
    >>> output_signal = ch_swap(signal)
    """

    def __init__(self, min_swap=0, max_swap=0):
        super().__init__()
        self.min_swap = min_swap
        self.max_swap = max_swap

        # Check arguments
        if self.min_swap < 0:
            raise ValueError("min_swap must be  >= 0.")
        if self.max_swap < 0:
            raise ValueError("max_swap must be  >= 0.")
        if self.max_swap < self.min_swap:
            raise ValueError("max_swap must be  >= min_swap")

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a frequency to drop
        waveforms = waveforms.detach().clone()
        rand_perm1 = torch.randperm(waveforms.shape[-1])
        rand_perm2 = torch.randperm(waveforms.shape[-1])
        N_swaps = torch.randint(
            low=self.min_swap, high=self.max_swap + 1, size=(1,)
        )

        if N_swaps < waveforms.shape[-1]:
            for i in range(N_swaps):
                store_channel = waveforms[:, :, rand_perm2[i]]
                waveforms[:, :, rand_perm2[i]] = waveforms[:, :, rand_perm1[i]]
                waveforms[:, :, rand_perm1[i]] = store_channel
        else:
            # Full swap
            waveforms = waveforms[:, :, rand_perm1]

        return waveforms


class RandomShift(torch.nn.Module):
    """This function shifts the input tensor by a random amount. Depending
    on the axis it can perform time pr channel shift.

    Arguments
    ---------
    min_shift : int
        The mininum channel shift.
    max_shift : int
        The maximum channel shift.
    dim: int
        The dimension to shift.

    Example
    -------
    >>> signal = torch.rand(4, 256, 8)
    >>> rand_shift =  RandomShift()
    >>> output_signal = rand_shift(signal)
    """

    def __init__(self, min_shift=0, max_shift=0, dim=1):
        super().__init__()
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.dim = dim

        # Check arguments
        if self.max_shift < self.min_shift:
            raise ValueError("max_shift must be  >= min_shift")

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a frequency to drop
        waveforms = waveforms.detach().clone()
        N_shifts = torch.randint(
            low=self.min_shift, high=self.max_shift + 1, size=(1,)
        )
        waveforms = torch.roll(waveforms, shifts=N_shifts.item(), dims=self.dim)
        return waveforms


class CutCat(torch.nn.Module):
    """This function combines segments (with equal length in time) of the time series contained in the batch.
    Proposed for EEG signals in https://doi.org/10.1016/j.neunet.2021.05.032.

    Arguments
    ---------
    num_segments : int
        The number of segments to combine.
    max_num_segments : int
        The maximum number of segments to combine. Default is 10.

    Example
    -------
    >>> signal = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1,))
    >>> cutcat =  CutCat()
    >>> output_signal = cutcat(signal)
    """

    def __init__(self, min_num_segments=2, max_num_segments=10):
        super().__init__()
        self.min_num_segments = min_num_segments
        self.max_num_segments = max_num_segments
        # Check arguments
        if self.max_num_segments < self.min_num_segments:
            raise ValueError("max_num_segments must be  >= min_num_segments")

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        waveforms = waveforms.detach().clone()
        if (
            waveforms.shape[0] > 1
        ):  # only if there are at least 2 examples in batch
            # rolling waveforms to point to segments of other examples in batch
            waveforms_rolled = torch.roll(waveforms, shifts=1, dims=0)
            # picking number of segments to use
            num_segments = torch.randint(
                low=self.min_num_segments,
                high=self.max_num_segments + 1,
                size=(1,),
            )
            # index of cuts (both starts and stops)
            idx_cut = torch.linspace(
                0, waveforms.shape[1], num_segments.item() + 1, dtype=torch.int
            )
            for i in range(idx_cut.shape[0] - 1):
                # half of segments from other examples in batch
                if i % 2 == 1:
                    start = idx_cut[i]
                    stop = idx_cut[i + 1]
                    waveforms[:, start:stop, ...] = waveforms_rolled[
                        :, start:stop, ...  # noqa: W504
                    ]

        return waveforms


def pink_noise_like(waveforms, alpha_low=1.0, alpha_high=1.0, sample_rate=50):
    """Creates a sequence of pink noise (also known as 1/f). The pink noise
    is obtained by multipling the spectrum of a white noise sequence by a
    factor (1/f^alpha).
    The alpha factor controls the decrease factor in the frequnecy domain
    (alpha=0 adds white noise, alpha>>0 adds low frequnecy noise). It is
    randomly sampled between alpha_low and alpha_high. With negative alpha this
    funtion generates blue noise.

    Arguments
    ---------
    waveforms : torch.Tensor
        The original waveform. It is just used to infer the shape.
    alpha_low : float
        The minimum value for the alpha spectral smooting factor.
    alpha_high : float
        The maximum value for the alpha spectral smooting factor.
    sample_rate : float
        The sample rate of the original signal.

    Example
    -------
    >>> waveforms = torch.randn(4,257,10)
    >>> noise = pink_noise_like(waveforms)
    >>> noise.shape
    torch.Size([4, 257, 10])
    """
    # Sampling white noise (flat spectrum)
    white_noise = torch.randn_like(waveforms)

    # Computing the fft of the input white noise
    white_noise_fft = torch.fft.fft(white_noise, dim=1)

    # Sampling the spectral smoothing factor
    rand_range = alpha_high - alpha_low
    alpha = (
        torch.rand(waveforms.shape[0], device=waveforms.device) * rand_range
        + alpha_low
    )

    # preparing the spectral mask (1/f^alpha)
    f = torch.linspace(
        0,
        sample_rate / 2,
        int(white_noise.shape[1] / 2),
        device=waveforms.device,
    )
    spectral_mask = 1 / torch.pow(f.unsqueeze(0), alpha.unsqueeze(1))

    # Avoid inf due to 1/0 division at f=0
    spectral_mask[:, 0] = spectral_mask[:, 1]

    # Mask for the upper part of the spectrum (f > sample_rate/2)
    spectral_mask_up = torch.flip(spectral_mask, dims=(1,))

    # Managing odd/even sequences
    if white_noise.shape[1] % 2:
        mid_element = spectral_mask[
            :, int(white_noise.shape[1] / 2) - 1
        ].unsqueeze(1)
        spectral_mask = torch.cat(
            [spectral_mask, mid_element, spectral_mask_up], dim=1
        )
    else:
        spectral_mask = torch.cat([spectral_mask, spectral_mask_up], dim=1)

    # Managing multi-channel inputs
    if len(white_noise.shape) == 3:
        spectral_mask = spectral_mask.unsqueeze(2)

    # Spectral masking
    pink_noise_fft = white_noise_fft * spectral_mask

    # Return to the time-domain
    pink_noise = torch.fft.ifft(pink_noise_fft, dim=1).real
    return pink_noise
