"""
Test functions that turn a list of lengths into a mask:
* speechbrain.dataio.dataio.length_to_mask
* speechbrain.lobes.models.transformer.Transformer.get_mask_from_lengths
* speechbrain.nnet.losses.get_mask
* speechbrain.nnet.losses.compute_length_mask
* speechbrain.processing.features.make_padding_mask

Authors
 * Rogier van Dalen 2025
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

numpy_dtype = {
    torch.bool: np.bool_,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.float32: np.float32,
}


def reference_length_to_mask_implementation(
    max_length: int,
    lengths: np.ndarray,
    dtype: np.dtype,
    feature_value,
    padding_value,
):
    result = np.empty(lengths.shape + (max_length,), dtype=dtype)
    for location in np.ndindex(lengths.shape):
        length = int(lengths[location])
        assert length <= max_length
        result[location] = length * [feature_value] + (max_length - length) * [
            padding_value
        ]
    return result


def reference_length_to_mask(
    max_length: int,
    lengths: np.ndarray,
    length_dim: int,
    feature_value,
    padding_value,
    dtype=None,
):
    """
    Convert a list of lengths into a padding mask, in a slow and general way.
    The padding mask has an extra dimension compared to "lengths", replacing
    e.g. "3" by e.g. [1, 1, 1, 0, 0].

    Arguments
    ---------
    max_length : int
        The size of the new dimension.
    lengths : np.ndarray
        The actual lengths, as integer ndarray.
        The values should be no greater than max_length.
    length_dim : int
        The dimension that should be inserted.
    feature_value : Any
        The value that indicates that the feature vector should be used.
    padding_value : Any
        The value that indicates that the feature vector should not be used.
    dtype : np.dtype
        Numpy dtype for the output.
        If None, the dtype is computed from the types of feature_value and
        padding_value.

    Returns
    -------
    A Numpy ndarray shaped like "lengths" with an extra dimension.
    Each vector in the new dimension has first as many feature_value's as
    "lengths" indicates, and then padding_value's to fill up to "max_length".
    """

    if dtype is None:
        dtype = np.asarray([feature_value, padding_value]).dtype

    # Fill in values location by location.
    values = np.empty(lengths.shape + (max_length,), dtype=dtype)
    for location in np.ndindex(lengths.shape):
        length = int(lengths[location])
        assert length <= max_length
        values[location] = length * [feature_value] + (max_length - length) * [
            padding_value
        ]

    return np.moveaxis(values, source=-1, destination=length_dim)


def test_reference_length_to_mask():
    """Sanity-check reference implementation"""
    result = reference_length_to_mask(
        3,
        np.asarray(2),
        length_dim=0,
        feature_value=4,
        padding_value=7,
        dtype=np.int32,
    )
    assert result.dtype == np.int32
    assert result.shape == (3,)
    assert np.all(result == [4, 4, 7])

    result = reference_length_to_mask(
        3, np.asarray([2]), length_dim=1, feature_value=4, padding_value=7
    )
    assert result.dtype == np.int64
    assert result.shape == (1, 3)
    assert np.all(result == [[4, 4, 7]])

    lengths = np.asarray([2, 1, 3, 0])
    result = reference_length_to_mask(
        3, lengths, length_dim=-1, feature_value=4, padding_value=7
    )
    assert result.dtype == np.int64
    assert result.shape == (4, 3)
    assert np.all(result == [[4, 4, 7], [4, 7, 7], [4, 4, 4], [7, 7, 7]])

    # Using length_dim.
    lengths = np.asarray([2, 1, 3, 0])
    result = reference_length_to_mask(
        3, lengths, length_dim=0, feature_value=4, padding_value=7
    )
    assert result.dtype == np.int64
    assert result.shape == (3, 4)
    assert np.all(result == [[4, 4, 4, 7], [4, 7, 4, 7], [7, 7, 4, 7]])


def adversarial_fractions(
    device: torch.device, dtype: torch.dtype, max_value: int
) -> Dict[int, List[int]]:
    """
    Compute fractions that where numerical accuracy can be an issue for
    fractional lengths.

    Arguments
    ---------
    device: torch.device
        Torch device for which the fractions are adversarial.
    dtype: torch.dtype
        Torch dtype for which the fractions are adversarial.
    max_value: int
        Maximum value (inclusive) of denominator and numerator to be considered.

    Returns
    -------
    A dictionary from d to a list of n, where (n/d) * d != n when computed by
    Torch with the device and dtype passed in, and n < d.
    """
    indices = torch.arange(1, max_value + 1, dtype=dtype, device=device)
    numerator = torch.broadcast_to(indices, (max_value, max_value))
    denominator = torch.transpose(numerator, 0, 1)
    less_than_one = numerator < denominator
    inexact = ((numerator / denominator) * denominator) != numerator

    result = defaultdict(list)
    for n, d in zip(
        numerator[less_than_one * inexact],
        denominator[less_than_one * inexact],
    ):
        result[int(d)].append(int(n))
    return result


def generate_integer_lengths(
    generator: np.random.Generator,
    lengths_ndim: int,
    adversarial_device: Optional[torch.device] = None,
    fractional_dtype: Optional[torch.dtype] = None,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate example lengths that are integer, as well as their fractional
    counterparts.
    Most weights are randomly chosen, but some are chosen so that the fractional
    versions, when multiplied back, are not exactly integer.

    Arguments
    ---------
    generator: np.random.Generator
        The Numpy generator that ensures deterministic examples.
    lengths_ndim: int
        The number of dimensions that the length array should have.
    adversarial_device: Optional[torch.device]
        The Torch device used for finding lengths that are rounded
        incorrectly.
    fractional_dtype: Optional[torch.dtype]
        The Torch dtype used for checking rounding.

    Yields
    ------
    Examples (max_length, integer_lengths, fractional_lengths) where the
    fractional_lengths match integer_lengths.
    """

    def random_lengths(max_length=None):
        """Draw random lengths."""
        if max_length is None:
            max_length = int(generator.integers(low=1, high=10))
        lengths_shape = generator.integers(low=1, high=8, size=lengths_ndim)
        lengths = generator.integers(
            low=0, high=max_length + 1, size=lengths_shape
        )
        return max_length, lengths

    def generate_lengths():
        """Generate first adversarial and then random weights."""
        if adversarial_device is not None:
            assert fractional_dtype is not None
            deterministic_lengths = adversarial_fractions(
                adversarial_device, fractional_dtype, 32
            )
            for max_length, some_lengths in deterministic_lengths.items():
                _, lengths = random_lengths(max_length)
                # Overwrite the first random lengths in "lengths" with specific
                # lengths from "some_lengths".
                for index, location in zip(
                    range(len(some_lengths)), np.ndindex(lengths.shape)
                ):
                    lengths[location] = some_lengths[index]
                yield max_length, lengths
        while True:
            yield random_lengths()

    # Add fractional lengths.
    for max_length, lengths in generate_lengths():
        assert np.all(lengths <= max_length)
        # Generate fractional lengths with Torch to trigger the problematic
        # rounding.
        tensor_lengths = torch.tensor(
            lengths, device=adversarial_device, dtype=fractional_dtype
        )
        fractional_lengths = (tensor_lengths / max_length).cpu().numpy()
        yield max_length, lengths, fractional_lengths


def generate_fractional_lengths(
    generator: np.random.Generator, lengths_ndim: int
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate example lengths that are fractional, as well as their integer
    counterparts.
    Some fractional weights indicate half or quarter frames, like would be
    expected when subsampling.
    The integers are obtained by rounding up.
    Nondeterministic examples are unlikely to occur, but if they do, the
    generator can be seeded differently.

    Arguments
    ---------
    generator: np.random.Generator
        The Numpy generator that ensures deterministic examples.
    lengths_ndim: int
        The number of dimensions that the length array should have.

    Yields
    ------
    Examples (max_length, integer_lengths, fractional_lengths) where the
    fractional_lengths match integer_lengths.
    """

    def generate():
        while True:
            max_length = int(generator.integers(low=1, high=10))
            lengths_shape = generator.integers(low=1, high=8, size=lengths_ndim)
            # Draw fractional lengths that round to each length with the same
            # probability.
            fractional_lengths = (
                generator.random(size=lengths_shape) * (max_length + 1) - 0.5
            ) / max_length
            # Do ensure the fractional length is in [0, 1]
            yield max_length, np.clip(fractional_lengths, 0, 1)

            # Generate a common case: lengths divided by 2 or 4.
            # Make it possible to compute this exactly.
            max_length = 2 ** int(generator.integers(low=1, high=4))
            lengths_shape = generator.integers(low=1, high=8, size=lengths_ndim)
            quadruple_lengths = generator.integers(
                0, 4 * max_length + 1, size=lengths_shape
            )
            fractional_lengths = (
                quadruple_lengths.astype(np.float32) / 4.0 / max_length
            )
            yield max_length, fractional_lengths

    for max_length, fractional_lengths in generate():
        integer_lengths = np.ceil(fractional_lengths * max_length).astype(
            np.int32
        )
        assert np.all(integer_lengths <= max_length)
        yield max_length, integer_lengths, fractional_lengths


def generate_lengths(
    generator: np.random.Generator,
    number: int,
    lengths_ndim: int,
    adversarial_device: Optional[torch.device] = None,
    adversarial_dtype: Optional[torch.dtype] = None,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate example lengths as both integer and fractional lengths.
    They are drawn partly to encourage rounding errors.
    The integer weights can be found by rounding up the fractional weights
    re-multiplied with max_length.

    Arguments
    ---------
    generator: np.random.Generator
        The Numpy generator that ensures deterministic examples.
    number: int
        The number of examples to draw.
    lengths_ndim: int
        The number of dimensions that the length array should have.
    adversarial_device: torch.device | None
        The Torch device used for finding lengths that are rounded
        incorrectly.
    adversarial_dtype: torch.dtype | None
        The Torch dtype used for checking rounding.

    Yields
    ------
    Examples (max_length, integer_lengths, fractional_lengths) where the
    fractional_lengths match integer_lengths.
    """

    def generate():
        integer_examples = generate_integer_lengths(
            generator, lengths_ndim, adversarial_device, adversarial_dtype
        )
        float_examples = generate_fractional_lengths(generator, lengths_ndim)
        # Interleave examples.
        # Note that calls to the single generator will also be interleaved.
        # The behaviour of this function is deterministic, but if
        # generate_integer_lengths is changed, than float_examples will be
        # different.
        for example_1, example_2 in zip(integer_examples, float_examples):
            yield example_1
            yield example_2

    # Yield the first "number" items.
    for _, item in zip(range(number), generate()):
        yield item


def generate_lengths_optional_max_length(
    generator: np.random.Generator,
    number: int,
    lengths_ndim: int,
    adversarial_device: Optional[torch.device] = None,
    adversarial_dtype: Optional[torch.dtype] = None,
) -> Iterable[Tuple[int, Optional[int], np.ndarray, np.ndarray]]:
    """Generate example lengths with max_length sometimes None."""
    for max_length, int_lengths, float_lengths in generate_lengths(
        generator,
        number,
        lengths_ndim,
        adversarial_device,
        adversarial_dtype,
    ):
        yield (max_length, max_length, int_lengths, float_lengths)
        if int(np.max(int_lengths)) == max_length:
            yield (max_length, None, int_lengths, float_lengths)


### Actual tests for all the functions that convert lengths to masks.


def test_length_to_mask(device):
    """
    Test :func:`~speechbrain.dataio.dataio.length_to_mask`.
    The function `length_to_mask` takes integer `lengths` and an optional
    `max_length`.
    It also takes an explicit `dtype` and `device`.
    """
    from speechbrain.dataio.dataio import length_to_mask

    generator = np.random.default_rng(seed=20250424)

    for dtype in [torch.bool, torch.int, torch.int64]:
        examples = generate_lengths_optional_max_length(
            generator, number=50, lengths_ndim=1
        )
        for max_length, maybe_max_length, lengths, _ in examples:
            reference_mask = reference_length_to_mask(
                max_length,
                lengths,
                length_dim=1,
                feature_value=1,
                padding_value=0,
                dtype=numpy_dtype[dtype],
            )

            lengths_tensor = torch.tensor(lengths, device=device)
            mask = length_to_mask(
                lengths_tensor, maybe_max_length, dtype, device
            )

            mask_numpy = mask.cpu().numpy()

            assert mask.dtype == dtype
            assert mask_numpy.shape == reference_mask.shape
            assert np.all(mask_numpy == reference_mask)


def test_get_mask_from_lengths(device):
    """
    Test :func:`~speechbrain.lobes.models.transformer.Transformer.get_mask_from_lengths`.
    The function `get_mask_from_lengths` takes integer `lengths` and an optional `max_length`.
    It always returns a boolean mask tensor.
    """
    from speechbrain.lobes.models.transformer.Transformer import (
        get_mask_from_lengths,
    )

    generator = np.random.default_rng(seed=20250424)

    examples = generate_lengths_optional_max_length(
        generator, number=50, lengths_ndim=1
    )
    for max_length, maybe_max_length, lengths, _ in examples:
        reference_mask = reference_length_to_mask(
            max_length,
            lengths,
            length_dim=1,
            feature_value=False,
            padding_value=True,
            dtype=np.bool_,
        )

        lengths_tensor = torch.tensor(lengths, device=device)
        mask = get_mask_from_lengths(lengths_tensor, maybe_max_length)

        mask_numpy = mask.cpu().numpy()

        assert mask.dtype == torch.bool
        assert mask_numpy.shape == reference_mask.shape
        assert np.all(mask_numpy == reference_mask)


def test_get_mask(device):
    """
    Test :func:`~speechbrain.nnet.losses.get_mask`.
    The function `get_mask` takes a single-dimensional lengths tensor with integer lengths.
    It also takes a data tensor `[T, B, C]` from which the device and dtype are
    taken (not from the length tensor!).
    The features are at 1s and the padding at 0s.
    """
    from speechbrain.nnet.losses import get_mask

    generator = np.random.default_rng(seed=20250424)

    for dtype in [torch.bool, torch.int, torch.int64, torch.float32]:
        examples = generate_lengths(generator, number=50, lengths_ndim=1)
        for max_length, lengths, _ in examples:
            reference_mask = reference_length_to_mask(
                max_length,
                lengths,
                length_dim=0,
                feature_value=1,
                padding_value=0,
                dtype=numpy_dtype[dtype],
            )

            lengths_tensor = torch.tensor(
                lengths, device=device, dtype=torch.int32
            )
            num_features = 8
            # max_length is given as the size of the first dimension of "data".
            data = torch.ones(
                (max_length, len(lengths), num_features),
                dtype=dtype,
                device=device,
            )
            # [T, B, 1] -> [T, B]
            mask = get_mask(data, lengths_tensor).squeeze(2)

            mask_numpy = mask.cpu().numpy()

            assert mask.dtype == dtype
            assert mask_numpy.shape == reference_mask.shape
            assert np.all(mask_numpy == reference_mask)


def test_compute_length_mask(device):
    """
    Test :func:`~speechbrain.nnet.losses.compute_length_mask`.
    The function `compute_length_mask` adds an extra dimension to the mask so it
    matches the data shape.
    The result has the same dtype as the data.
    """
    from speechbrain.nnet.losses import compute_length_mask

    generator = np.random.default_rng(seed=20250424)

    # Only test dtypes that can hold lengths; so not bool.
    # Also, no floating-point types can be used.
    for dtype in [torch.bool, torch.int, torch.int64, torch.float32]:
        examples = generate_lengths(
            generator,
            number=50,
            lengths_ndim=1,
            adversarial_device=device,
            adversarial_dtype=torch.float32,
        )
        for length_dim in [0, 1]:
            for max_length, lengths, float_lengths in examples:
                reference_mask = reference_length_to_mask(
                    max_length,
                    lengths,
                    length_dim=length_dim,
                    feature_value=True,
                    padding_value=False,
                    dtype=numpy_dtype[dtype],
                )

                num_features = 8
                print("max_length", max_length)
                print("lengths.shape", lengths.shape)
                print("reference_mask.shape", reference_mask.shape)  # TODO
                reference_mask = np.broadcast_to(
                    reference_mask[:, :, np.newaxis],
                    reference_mask.shape + (num_features,),
                )

                lengths_tensor = torch.tensor(
                    float_lengths, device=device, dtype=torch.float32
                )
                # max_length is given as a dimension of "data"
                data_size = (
                    lengths.shape[:length_dim]
                    + (max_length,)
                    + lengths.shape[length_dim:]
                    + (num_features,)
                )
                data = torch.ones(data_size, dtype=dtype, device=device)
                mask = compute_length_mask(
                    data, lengths_tensor, len_dim=length_dim
                )

                mask_numpy = mask.cpu().numpy()

                print("mask", mask_numpy)

                assert mask.dtype == dtype
                assert mask_numpy.shape == reference_mask.shape
                assert np.all(mask_numpy == reference_mask)


def test_make_padding_mask(device):
    """
    Test :func:`~speechbrain.processing.features.make_padding_mask`.
    The function `make_padding_mask` takes fractional lengths between `0` and `1`.
    It always rounds the number of elements up.
    It adds a new dimension with length `1` for the length, so that if the tensor
    with features is the right shape, it can be multiplied straight away.
    The result always has dtype `bool`.
    """
    from speechbrain.processing.features import make_padding_mask

    generator = np.random.default_rng(seed=20250424)

    examples = generate_lengths(
        generator,
        number=50,
        lengths_ndim=1,
        adversarial_device=device,
        adversarial_dtype=torch.float32,
    )
    for max_length, lengths, float_lengths in examples:
        reference_mask = reference_length_to_mask(
            max_length,
            lengths,
            length_dim=1,
            feature_value=True,
            padding_value=False,
            dtype=np.bool_,
        )[:, :, np.newaxis]

        lengths_tensor = torch.tensor(float_lengths, device=device)
        num_features = 8
        # max_length is given as a dimension of "data".
        data_size = lengths.shape + (max_length, num_features)
        data = torch.ones(data_size, dtype=torch.float32, device=device)
        mask = make_padding_mask(data, lengths_tensor, length_dim=1)

        mask_numpy = mask.cpu().numpy()

        assert mask.dtype == torch.bool
        assert mask_numpy.shape == reference_mask.shape
        assert np.all(mask_numpy == reference_mask)
