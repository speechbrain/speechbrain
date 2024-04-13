import pytest

from speechbrain.utils.parallel import parallel_map

small_test_input = [1, 2, 3, 4, 5, 6] * 5
small_test_expected = [2, 4, 6, 8, 10, 12] * 5


def small_test_func(x):
    return x * 2


def raise_error(x):
    raise ValueError("dummy error")


def test_parallel_map():
    # test different chunk sizes
    for test_chunk_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        small_test_output = list(
            parallel_map(
                small_test_func,
                small_test_input,
                process_count=2,
                chunk_size=test_chunk_size,
            )
        )

        assert (
            small_test_output == small_test_expected
        ), f"chunk_size={test_chunk_size}"

    # test whether pbar off works properly
    small_test_output = list(
        parallel_map(small_test_func, small_test_input, progress_bar=False)
    )

    # test whether exceptions are forwarded properly
    with pytest.raises(ValueError):
        small_test_output = list(
            parallel_map(raise_error, small_test_input, progress_bar=False)
        )

    # test edge case: empty input
    assert list(parallel_map(small_test_func, [])) == []

    # trivial test for tqdm kwargs
    parallel_map(small_test_func, small_test_input, progress_bar_kwargs={})
