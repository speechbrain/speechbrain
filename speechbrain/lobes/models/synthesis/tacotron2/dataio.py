"""
 Data preprocessing for Tacotron2

 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021

"""

import speechbrain as sb
import torch
from speechbrain.lobes.models.synthesis.tacotron2.text_to_sequence import (
    text_to_sequence,
)

DEFAULT_TEXT_CLEANERS = ["english_cleaners"]


def encode_text(
    text_cleaners=None,
    takes="txt",
    provides=["text_sequences", "input_lengths"],
):
    """
    A pipeline function that encodes raw text into a tensor

    Arguments
    ---------
    text_cleaners: list
        an list of text cleaners to use
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    text_cleaners = DEFAULT_TEXT_CLEANERS

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(*provides)
    def f(txt):
        sequence = text_to_sequence(txt, text_cleaners)
        yield torch.tensor(sequence, dtype=torch.int32)
        yield torch.tensor(len(sequence), dtype=torch.int32)

    return f


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """A pipeline function that performs dynamic range compression (i.e. puts
    the supplied input, typically a spectrogram, on a logarithmic scale)

    Arguments
    ---------
    x: torch.Tensor
        the input tensor

    C: int
        compression factor

    clip_val: float
        the minimum value below which x values will be clipped

    Returns
    -------
    return: torch.tensor
        compressed results
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_compression_pipeline(
    C, clip_val, takes="mel", provides="mel_compressed"
):
    """A pipeline element that performs dynamic range compression (i.e. puts
    the supplied input, typically a spectrogram, on a logarithmic scale)

    x: torch.Tensor
        the input tensor

    C: int
        compression factor

    clip_val: float
        the minimum value below which x values will be clipped

    Returns
    -------
    result: DymamicItem
        a pipeline element

    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(x):
        return dynamic_range_compression(x, C, clip_val)

    return f


def dynamic_range_decompression(x, C=1):
    """Performs dynamic range compression (i.e. puts
    the supplied input, typically a spectrogram, on a logarithmic scale)

    Arguments
    ---------
    C: int
        compression factor used to compress
    """
    return torch.exp(x) / C


def dynamic_range_decompression_pipeline(
    C=1, takes="mel", provides="mel_decompressed"
):
    """A pipeline element that performs dynamic range decompression

    x: torch.Tensor
        the input tensor

    C: int
        compression factor

    clip_val: float
        the minimum value below which x values will be clipped

    Returns
    -------
    result: DymamicItem
        a pipeline element

    """

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(x):
        return dynamic_range_decompression(x, C)

    return f
