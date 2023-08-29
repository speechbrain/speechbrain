"""Utilities to assist with designing and training streaming models.

Authors
* Sylvain de Langen 2023
"""

import math
import torch
from typing import Callable, List


def split_fixed_chunks(
    x: torch.Tensor, chunk_size: int, dim: int = -1
) -> List[torch.Tensor]:
    """Split an input tensor `x` into a list of chunk tensors of size
    `chunk_size` alongside dimension `dim`.
    Useful for splitting up sequences with chunks of fixed sizes.

    If dimension `dim` cannot be evenly split by `chunk_size`, then the last
    chunk will be smaller than `chunk_size`.

    Arguments
    ---------
    x : torch.Tensor
        The tensor to split into chunks, typically a sequence or audio signal.

    chunk_size : int
        The size of each chunk, i.e. the max size of each chunk on dimension
        `dim`.

    dim : int
        Dimension to split alongside of, typically the time dimension.

    Returns
    -------
    List[torch.Tensor]
        A chunk list of tensors, see description and example.
        Guarantees `.size(dim) <= chunk_size`.

    Example
    -------
    >>> import torch
    >>> from speechbrain.utils.streaming import split_fixed_chunks
    >>> x = torch.zeros((16, 10000, 80))
    >>> chunks = split_fixed_chunks(x, 128, dim=1)
    >>> len(chunks)
    79
    >>> chunks[0].shape
    torch.Size([16, 128, 80])
    >>> chunks[-1].shape
    torch.Size([16, 16, 80])
    """

    num_chunks = math.ceil(x.size(dim) / chunk_size)
    split_at_indices = [(i + 1) * chunk_size for i in range(num_chunks - 1)]
    return torch.tensor_split(x, split_at_indices, dim=1)


def split_wav_lens(
    chunk_lens: List[int], wav_lens: torch.Tensor
) -> List[torch.Tensor]:
    """Converts a single `wav_lens` tensor into a list of `chunk_count` tensors,
    typically useful when chunking signals with `split_fixed_chunks`.

    `wav_lens` represents the relative length of each audio within a batch,
    which is typically used for masking. This function computes the relative
    length at chunk level.

    Arguments
    ---------
    chunk_lens : List[int]
        Length of the sequence of every chunk. For example, if `chunks` was
        returned from `split_fixed_chunks(x, chunk_size, dim=1)`, then this
        should be `[chk.size(1) for chk in chunks]`.

    wav_lens : torch.Tensor
        Relative lengths of audio within a batch. For example, for an input
        signal of 100 frames and a batch of 3 elements, `(1.0, 0.5, 0.25)`
        would mean the batch holds audio of 100 frames, 50 frames and 25 frames
        respectively.

    Returns
    -------
    List[torch.Tensor]
        A list of chunked wav_lens, see description and example.

    Example
    -------
    >>> import torch
    >>> from speechbrain.utils.streaming import split_wav_lens, split_fixed_chunks
    >>> x = torch.zeros((3, 20, 80))
    >>> chunks = split_fixed_chunks(x, 8, dim=1)
    >>> len(chunks)
    3
    >>> # 20 frames, 13 frames, 17 frames
    >>> wav_lens = torch.tensor([1.0, 0.65, 0.85])
    >>> chunked_wav_lens = split_wav_lens([c.size(1) for c in chunks], wav_lens)
    >>> chunked_wav_lens
    [tensor([1., 1., 1.]), tensor([1.0000, 0.6250, 1.0000]), tensor([1.0000, 0.0000, 0.2500])]
    >>> # wav 1 covers 62.5% (5/8) of the second chunk's frames
    """

    chunk_wav_lens = []

    seq_size = sum(chunk_lens)
    wav_lens_frames = wav_lens * seq_size

    chunk_start_frame = 0
    for chunk_len in chunk_lens:
        chunk_raw_len = (wav_lens_frames - chunk_start_frame) / chunk_len
        chunk_raw_len = torch.clamp(chunk_raw_len, 0.0, 1.0)
        chunk_wav_lens.append(chunk_raw_len)

        chunk_start_frame += chunk_len

    return chunk_wav_lens


def infer_dependency_matrix(
    model: Callable, seq_shape: tuple, in_stride: int = 1
):
    """
    Randomizes parts of the input sequence several times in order to detect
    dependencies between input frames and output frames, aka whether a given
    output frame depends on a given input frame.

    This can prove useful to check whether a model behaves correctly in a
    streaming context and does not contain accidental dependencies to future
    frames that couldn't be known in a streaming scenario.

    Note that this can get very computationally expensive for very long
    sequences.

    Furthermore, this expects inference to be fully deterministic, else false
    dependencies may be found. This also means that the model must be in eval
    mode, to inhibit things like dropout layers.

    Arguments
    ---------
    model : Callable
        Can be a model or a function (potentially emulating streaming
        functionality). Does not require to be a trained model, random weights
        should usually suffice.
    seq_shape : tuple
        The function tries inferring by randomizing parts of the input sequence
        in order to detect unwanted dependencies.
        The shape is expected to look like `[batch_size, seq_len, num_feats]`,
        where `batch_size` may be `1`.
    in_stride : int
        Consider only N-th input, for when the input sequences are very long
        (e.g. raw audio) and the output is shorter (subsampled, filters, etc.)

    Returns
    -------
    dependencies : torch.BoolTensor
        Matrix representing whether an output is dependent on an input; index
        using `[in_frame_idx, out_frame_idx]`. `True` indicates a detected
        dependency.
    """
    # TODO: document arguments

    bs, seq_len, feat_len = seq_shape

    base_seq = torch.rand(seq_shape)
    with torch.no_grad():
        base_out = model(base_seq)

        if not model(base_seq).equal(base_out):
            raise ValueError(
                "Expected deterministic model, but inferring twice on the same "
                "data yielded different results. Make sure that you use "
                "`eval()` mode so that it does not include randomness."
            )
    out_len, _out_feat_len = base_out.shape[1:]

    deps = torch.zeros(
        ((seq_len + (in_stride - 1)) // in_stride, out_len), dtype=torch.bool
    )

    for in_frame_idx in range(0, seq_len, in_stride):
        test_seq = base_seq.clone()
        test_seq[:, in_frame_idx, :] = torch.rand(bs, feat_len)

        with torch.no_grad():
            test_out = model(test_seq)

        for out_frame_idx in range(out_len):
            if not torch.allclose(
                test_out[:, out_frame_idx, :], base_out[:, out_frame_idx, :]
            ):
                deps[in_frame_idx // in_stride][out_frame_idx] = True

    return deps


def plot_dependency_matrix(deps):
    """
    Returns a matplotlib figure of a dependency matrix generated by
    `infer_dependency_matrix`.

    At a given point, a red square indicates that a given output frame (y-axis)
    was to depend on a given input frame (x-axis).

    For example, a fully red image means that all output frames were dependent
    on all the history. This could be the case of a bidirectional RNN, or a
    transformer model, for example.

    Arguments
    ---------
    deps : torch.BoolTensor
        Matrix returned by `infer_dependency_matrix` or one in a compatible
        format.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "red"])

    fig, ax = plt.subplots()

    ax.pcolormesh(
        torch.permute(deps, (1, 0)),
        cmap=cmap,
        vmin=False,
        vmax=True,
        edgecolors="gray",
        linewidth=0.5,
    )
    ax.set_title("Dependency plot")
    ax.set_xlabel("in")
    ax.set_ylabel("out")
    ax.set_aspect("equal")
    return fig
