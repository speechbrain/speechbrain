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
            result[dimension_pair_index * 2][dimension_pair_index * 2 + 1] = (
                -math.sin(angle)
            )
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

    result = _rope_rotate(torch.tensor(x, dtype=torch_dtype))
    assert result.dtype == torch_dtype
    result_np = result.cpu().numpy()

    reference = rope_rotate_slow(x)

    # If the result is the same as the reference, then the test is meaningless.
    # (But the first element is always rotated with an angle of 0.)
    assert length == 1 or not np.allclose(x, reference, atol=tolerance)

    assert np.allclose(result_np, reference, atol=tolerance)
