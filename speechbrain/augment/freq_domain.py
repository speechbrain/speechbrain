"""Frequency-Domain Sequential Data Augmentation Classes

This module comprises classes tailored for augmenting sequential data in the
frequency domain, such as spectrograms and mel spectrograms.
Its primary purpose is to enhance the resilience of neural models during the training process.

Authors:
- Peter Plantinga (2020)
- Mirco Ravanelli (2023)
"""

import torch
import random


class SpectrogramDrop(torch.nn.Module):
    """This class drops slices of the input spectrogram.

    Using `SpectrogramDrop` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to drop the
        spectrogram, in samples.
    drop_length_high : int
        The high end of lengths for which to drop the
        signal, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped.
    replace: str
        - 'zeros': Masked values are replaced with zeros.
        - 'mean': Masked values are replaced with the mean value of the spectrogram.
        - 'rand': Masked values are replaced with random numbers ranging between
                  the maximum and minimum values of the spectrogram.
        - 'cutcat': Masked values are replaced with chunks from other signals in the batch.
        - 'swap': Masked values are replaced with other chunks from the same sentence.
        - 'random_selection': A random selection among the approaches above.
    dim : int
        Corresponding dimension to mask. If dim=1, we apply time masking.
        If dim=2, we apply frequency masking.

    Example
    -------
    >>> # time-masking
    >>> drop = SpectrogramDrop(dim=1)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # frequency-masking
    >>> drop = SpectrogramDrop(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(
        self,
        drop_length_low=5,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="zeros",
        dim=1,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.replace = replace
        self.dim = dim

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        self.replace_opts = [
            "zeros",
            "mean",
            "rand",
            "cutcat",
            "swap",
            "random_selection",
        ]
        if self.replace not in self.replace_opts:
            raise ValueError(
                f"Invalid 'replace' option. Select one of {', '.join(self.replace_opts)}"
            )

    def forward(self, spectrogram):
        """
        Apply the DropChunk augmentation to the input spectrogram.

        This method randomly drops chunks of the input spectrogram to augment the data.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram of shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram of shape `[batch, time, fea]`.
        """

        # Manage 4D tensors
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.view(
                -1, spectrogram.shape[2], spectrogram.shape[3]
            )

        # Get the batch size
        batch_size, time_duration, fea_size = spectrogram.shape

        # Managing masking dimensions
        if self.dim == 1:
            D = time_duration
        else:
            D = fea_size

        # Randomly select the number of chunks to drop (same for all samples in the batch)
        n_masks = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(1,),
            device=spectrogram.device,
        )

        # Randomly sample the lengths of the chunks to drop
        mask_len = torch.randint(
            low=self.drop_length_low,
            high=self.drop_length_high,
            size=(batch_size, n_masks),
            device=spectrogram.device,
        ).unsqueeze(2)

        # Randomly sample the positions of the chunks to drop
        mask_pos = torch.randint(
            0,
            max(1, D, -mask_len.max()),
            (batch_size, n_masks),
            device=spectrogram.device,
        ).unsqueeze(2)

        # Compute the mask for the selected chunk positions
        arange = torch.arange(D, device=spectrogram.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2) if self.dim == 1 else mask.unsqueeze(1)

        # Determine the value to replace the masked chunks (zero or mean of the spectrogram)
        if self.replace == "random_selection":
            self.replace = random.choice(self.replace_opts[:-1])

        if self.replace == "zeros":
            spectrogram = spectrogram.masked_fill_(mask, 0.0)
        elif self.replace == "mean":
            mean = spectrogram.mean().detach()
            spectrogram = spectrogram.masked_fill_(mask, mean)
        elif self.replace == "rand":
            max_spectrogram = spectrogram.max().detach()
            min_spectrogram = spectrogram.min().detach()
            rand_spectrogram = torch.rand_like(spectrogram)
            rand_spectrogram = (
                rand_spectrogram * (max_spectrogram - min_spectrogram)
                + min_spectrogram
            )
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rand_spectrogram
        elif self.replace == "cutcat":
            rolled_spectrogram = torch.roll(spectrogram, shifts=1, dims=0)
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rolled_spectrogram
        elif self.replace == "swap":
            shift = torch.randint(
                low=1,
                high=spectrogram.shape[1],
                size=(1,),
                device=spectrogram.device,
            )
            rolled_spectrogram = torch.roll(
                spectrogram, shifts=shift.item(), dims=1
            )
            mask = mask.float()
            spectrogram = (1 - mask) * spectrogram + mask * rolled_spectrogram

        return spectrogram.view(*spectrogram.shape)


class Warping(torch.nn.Module):
    """
    Apply time or frequency warping to a spectrogram.

    If `dim=1`, time warping is applied; if `dim=2`, frequency warping is applied.
    This implementation selects a center and a window length to perform warping.
    It ensures that the temporal dimension remains unchanged by upsampling or
    downsampling the affected regions accordingly.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    warp_window : int, optional
        The width of the warping window. Default is 5.
    warp_mode : str, optional
        The interpolation mode for time warping. Default is "bicubic."
    dim : int, optional
        Dimension along which to apply warping (1 for time, 2 for frequency).
        Default is 1.

    Example
    -------
    >>> # Time-warping
    >>> warp = Warping()
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # Frequency-warping
    >>> warp = Warping(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(self, warp_window=5, warp_mode="bicubic", dim=1):
        super().__init__()
        self.warp_window = warp_window
        self.warp_mode = warp_mode
        self.dim = dim

    def forward(self, spectrogram):
        """
        Apply warping to the input spectrogram.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram with shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram with shape `[batch, time, fea]`.
        """

        # Set warping dimension
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        original_size = spectrogram.shape
        window = self.warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        len_original = spectrogram.shape[2]
        if len_original - window <= window:
            return spectrogram.view(*original_size)

        # Compute center and corresponding window
        c = torch.randint(window, len_original - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1

        # Update the left part of the spectrogram
        left = torch.nn.functional.interpolate(
            spectrogram[:, :, :c],
            (w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Update the right part of the spectrogram.
        # When the left part is expanded, the right part is compressed by the
        # same factor, and vice versa.
        right = torch.nn.functional.interpolate(
            spectrogram[:, :, c:],
            (len_original - w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Injecting the warped left and right parts.
        spectrogram[:, :, :w] = left
        spectrogram[:, :, w:] = right
        spectrogram = spectrogram.view(*original_size)

        # Transpose if freq warping is applied.
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        return spectrogram


class RandomShift(torch.nn.Module):
    """Shifts the input tensor by a random amount, allowing for either a time
    or frequency (or channel) shift depending on the specified axis.
    It is crucial to calibrate the minimum and maximum shifts according to the
    requirements of your specific task.
    We recommend using small shifts to preserve information integrity.
    Using large shifts may result in the loss of significant data and could
    potentially lead to misalignments with corresponding labels.

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
    >>> # time shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0,50,:] = 1
    >>> rand_shift =  RandomShift(dim=1, min_shift=-10, max_shift=10)
    >>> lenghts = torch.tensor([0.2, 0.8, 0.9,1.0])
    >>> output_signal, lenghts = rand_shift(signal,lenghts)

    >>> # frequency shift
    >>> signal = torch.zeros(4, 100, 80)
    >>> signal[0,:,40] = 1
    >>> rand_shift =  RandomShift(dim=2, min_shift=-10, max_shift=10)
    >>> lenghts = torch.tensor([0.2, 0.8, 0.9,1.0])
    >>> output_signal, lenghts = rand_shift(signal,lenghts)
    """

    def __init__(self, min_shift=0, max_shift=0, dim=1):
        super().__init__()
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.dim = dim

        # Check arguments
        if self.max_shift < self.min_shift:
            raise ValueError("max_shift must be  >= min_shift")

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        # Pick a frequency to drop
        N_shifts = torch.randint(
            low=self.min_shift,
            high=self.max_shift + 1,
            size=(1,),
            device=waveforms.device,
        )
        waveforms = torch.roll(waveforms, shifts=N_shifts.item(), dims=self.dim)

        # Update lenghts in the case of temporal shift.
        if self.dim == 1:
            lengths = lengths + N_shifts / waveforms.shape[self.dim]
            lengths = torch.clamp(lengths, min=0.0, max=1.0)

        return waveforms, lengths
