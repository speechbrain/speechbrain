"""Few components to support BEST RQ training as described in the
original paper: https://arxiv.org/pdf/2202.01855.

Authors
* Ryan Whetten 2024
* Titouan Parcollet 2025
"""

import random

import torch

from speechbrain.utils.data_utils import batch_pad_right


def compute_mask(shape, sample_lens, mask_prob, mask_length):
    """This function generates the masks of BEST-RQ.

    It generates a unique mask for the whole batch and based on the shorter utte
    rance. This is important as it may alter the training if the batch contains
    one small sentence and many large ones as only few frames will be masked.

    In particular, out of the smaller length passed to sample_lens, we will
    generate N masks with N = mask_prob * smallest_len. Hence, mask_prob is
    the probability for a frame to start a mask, and not to be masked.

    If a sentence length is 100 time steps, a mask_prob of 0.15 and a mask size
    of 4 would results in 100*0.15*4=60% of the frames being masked.

    Arguments
    ---------
    shape: tuple
        The shape of the input tensor to be masked. Usually (Batch, Time, Fea).
    sample_lens: list
        List of int corresponding to the number of frames of each sample in the
        batch. E.g. (12,13,14,20)
    mask_prob: float
        Probability for a frame to spawn a mask. Frames already masked cannot
        spawn new masks.
    mask_length: int
        Number of frames covered by a mask.

    Returns
    -------
    The computed mask

    Example
    -------
    >>> compute_mask((2,50,60), [40, 50], 0.15, 2).shape
    torch.Size([12])
    """
    min_sample_len = min(sample_lens)

    # int always floors the float number so adding + random.random()
    # makes it 50% change of rounding up and 50% of rounding down
    num_mask = int(mask_prob * min_sample_len + random.random())

    # make sure there is at least 1 mask
    if num_mask == 0:
        num_mask = 1

    permutation = torch.randperm(min_sample_len // mask_length) * mask_length
    selected_indices = permutation[:num_mask]
    selected_indices, _ = selected_indices.sort()

    idx = []
    for i in selected_indices:
        idx.append(torch.arange(start=i, end=i + mask_length))
    idx = torch.cat(idx)

    return idx


def brq_mask_collate_fn(
    samples_lst, get_out_len_fn, mask_prob, mask_length, n_mels
):
    """This creates a batch from a list of samples and also creates
    the mask that will be used to mask the inputs of BEST-RQ.
    To create the mask we need to know the output shape after the
    latent extractor, therefore the argument `get_out_len_fn`.
    One could also create masks per sample (when loading the audio file) and
    then collate them but at that time one doesn't know the length of the
    shortest sample in the batch (which determines the number of masked frames)
    so it's better this way.

    Arguments
    ---------
    samples_lst : list
        List of samples returned by the audio_pipeline.
    get_out_len_fn : function
        Function that calculates length of sample after it passes through feature extractor.
    mask_prob : float
        Probability for a frame to spawn a mask. Frames already masked cannot
        spawn new masks.
    mask_length : int
        Number of contiguous frames that will be masked.
    n_mels : int
        Number of Mels filterbanks in the last dimension of the input tensor.

    Returns
    -------
    wavs_padded : torch.Tensor, shape (B, T)
        Audio arrays with right-sided padding.
    wav_lens : torch.Tensor, shape (B,)
        For each sample the percentage of the array that is not padding.
    mask : torch.Tensor, shape (T)
        Mask with the indices to be masked in the input tensor.
    """
    wav_lst, latent_length_lst = [], []
    ids = []
    for sample in samples_lst:
        ids.append(sample["id"])
        sig = sample["sig"]
        wav_lst.append(sig)
        latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
        latent_length_lst.append(latent_length.item())
    bs = len(wav_lst)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    batch_time_len
    mask = compute_mask(
        (bs, batch_time_len, n_mels), latent_length_lst, mask_prob, mask_length
    )
    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask),
    )
