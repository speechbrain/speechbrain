import math

import numpy as np
import pytest
import torch

from speechbrain.nnet.attention import memoise_at_least


def test_rel_pos_MHA(device):
    from speechbrain.nnet.attention import RelPosMHAXL

    bsz = 2
    emb_dim = 4
    k_len = [12, 10]
    q_len = [10, 12]
    bias = [True, False]
    head_dim = [4, None]

    for kl in k_len:
        for ql in q_len:
            for b in bias:
                for h in head_dim:
                    relpos = RelPosMHAXL(
                        emb_dim, num_heads=2, vbias=b, vdim=h
                    ).to(device)
                    q = torch.rand((bsz, ql, emb_dim), device=device)
                    k = torch.rand((bsz, kl, emb_dim), device=device)
                    pos_embs = torch.rand(
                        (1, 2 * kl - 1, emb_dim), device=device
                    )
                    relpos(q, k, k, pos_embs=pos_embs)


def rel_pos_enc_xl_slow(seq_len: int, emb_dim: int) -> np.ndarray:
    """
    Slow implementation of RelPosEncXL.make_pe.
    """
    result = np.zeros((2 * seq_len - 1, emb_dim))

    for index in range(2 * seq_len - 1):
        # rows run from the largest positive relative position down to the
        # largest negative one, as expected by `RelPosMHAXL.rel_shift`.
        position = seq_len - 1 - index

        # Implement (1) from https://arxiv.org/pdf/1706.03762 for a signed
        # position, as https://arxiv.org/pdf/1901.02860 does.
        for dimension_pair_index in range(emb_dim // 2):
            angle = position * 10000 ** (-2 * dimension_pair_index / emb_dim)
            result[index][dimension_pair_index * 2] = math.sin(angle)
            result[index][dimension_pair_index * 2 + 1] = math.cos(angle)

    return result


@pytest.mark.parametrize(
    "seq_len, emb_dim", [(1, 4), (2, 4), (7, 16), (16, 2), (33, 64)]
)
def test_rel_pos_enc_xl(device, seq_len, emb_dim):
    from speechbrain.nnet.attention import RelPosEncXL

    reference = rel_pos_enc_xl_slow(seq_len, emb_dim)

    # If opposite relative positions shared an embedding, the encoding could
    # not tell a key `d` steps ahead from one `d` steps behind, and this test
    # would be much weaker.
    centre = seq_len - 1
    assert seq_len == 1 or not np.allclose(
        reference[:centre], np.flip(reference[centre + 1 :], axis=0)
    )

    encoder = RelPosEncXL(emb_dim, use_legacy_symmetric=False).to(device=device)
    result = encoder.make_pe(seq_len)
    assert result.shape == (1, 2 * seq_len - 1, emb_dim)

    assert np.allclose(result[0].cpu().numpy(), reference, atol=1e-5)

    # the default stays symmetric, so that checkpoints trained before this was
    # made configurable keep behaving the same way
    legacy = RelPosEncXL(emb_dim).to(device=device).make_pe(seq_len)
    assert torch.equal(
        legacy[0, :centre], torch.flip(legacy[0, centre + 1 :], (0,))
    )


memoised_calls = []


@memoise_at_least(lambda x: math.ceil(x))
def memoisable(n: float, *args):
    memoised_calls.append((n,) + args)
    return n


def test_memoise_at_least():
    result = memoisable(5.5, "a", "b", "c")
    assert result == 6
    result = memoisable(5.5, "a", "b", "c")
    assert result == 6
    result = memoisable(5.9, "a", "b", "c")
    assert result == 6

    result = memoisable(2.1, "b", "c")
    assert result == 3

    result = memoisable(2.1, "b")
    assert result == 3

    result = memoisable(2.5, "b")
    assert result == 3

    result = memoisable(7.5)
    assert result == 8
    result = memoisable(7.1)
    assert result == 8

    assert memoised_calls == [(6, "a", "b", "c"), (3, "b", "c"), (3, "b"), (8,)]


def rope_rotate_slow(x: np.ndarray):
    """
    Slow implementation of rope_rotate.
    """
    batch_size, length, num_heads, num_dimensions = x.shape

    def dimension_pair_angle(
        dimension_pair_index: int, num_dimensions: int
    ) -> float:
        return 10000 ** (-2 * dimension_pair_index / num_dimensions)

    def make_rotation_matrix(time: int, num_dimensions: int) -> np.ndarray:
        assert num_dimensions / 2 == num_dimensions // 2

        result = np.zeros((num_dimensions, num_dimensions))

        # Implement (15) from https://arxiv.org/pdf/2104.09864 explicitly.
        for dimension_pair_index in range(num_dimensions // 2):
            angle = time * dimension_pair_angle(
                dimension_pair_index, num_dimensions
            )
            result[dimension_pair_index * 2][dimension_pair_index * 2] = (
                math.cos(angle)
            )
            result[dimension_pair_index * 2][
                dimension_pair_index * 2 + 1
            ] = -math.sin(angle)
            result[dimension_pair_index * 2 + 1][dimension_pair_index * 2] = (
                math.sin(angle)
            )
            result[dimension_pair_index * 2 + 1][
                dimension_pair_index * 2 + 1
            ] = math.cos(angle)

        return result

    # Initialise to a noticeable value in case of logic problems.
    result = -123456 * np.ones_like(x)

    for batch_index in range(batch_size):
        for time in range(length):
            rotation_matrix = make_rotation_matrix(time, num_dimensions).astype(
                x.dtype
            )
            for head_index in range(num_heads):
                result[batch_index][time][head_index] = (
                    rotation_matrix @ x[batch_index][time][head_index]
                )

    return result


@pytest.mark.parametrize(
    "batch_size, length, num_heads, num_dimensions",
    # length and num_dimensions are the most interesting to vary.
    [
        (1, 1, 1, 4),
        (2, 5, 3, 8),
        (3, 20, 2, 22),
        (2, 20, 2, 1024),
        (2, 170, 2, 22),
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype, torch_dtype, tolerance",
    [
        (np.float16, torch.float16, 1e-3),
        (np.float32, torch.float32, 1e-5),
        (np.float64, torch.float64, 1e-10),
    ],
)
def test_rope_rotate(
    device,
    numpy_dtype,
    torch_dtype,
    tolerance,
    batch_size,
    length,
    num_heads,
    num_dimensions,
):
    from speechbrain.nnet.attention import _rope_rotate

    generator: np.random.Generator = np.random.default_rng(
        seed=20250205 + batch_size + length + num_heads
    )

    x = generator.uniform(
        -1, +1, (batch_size, length, num_heads, num_dimensions)
    ).astype(numpy_dtype)

    result = _rope_rotate(torch.tensor(x, dtype=torch_dtype, device=device))
    assert result.dtype == torch_dtype
    result_np = result.cpu().numpy()

    reference = rope_rotate_slow(x)

    # If the result is the same as the reference, then the test is meaningless.
    # (But the first element is always rotated with an angle of 0.)
    assert length == 1 or not np.allclose(x, reference, atol=tolerance)

    assert np.allclose(result_np, reference, atol=tolerance)
