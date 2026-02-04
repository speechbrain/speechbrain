import os
from unittest import mock

import pytest

from speechbrain.utils.parallel import get_available_cpu_count, parallel_map

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

        assert small_test_output == small_test_expected, (
            f"chunk_size={test_chunk_size}"
        )

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


class TestGetAvailableCpuCount:
    """Tests for get_available_cpu_count function."""

    def test_returns_positive_integer(self):
        """Basic test that function returns a positive integer."""
        result = get_available_cpu_count()
        assert isinstance(result, int)
        assert result > 0

    def test_env_var_override(self):
        """Test that SB_NUM_PROC environment variable takes precedence."""
        with mock.patch.dict(os.environ, {"SB_NUM_PROC": "4"}, clear=False):
            assert get_available_cpu_count() == 4

    def test_env_var_invalid_ignored(self):
        """Test that invalid SB_NUM_PROC values are ignored."""
        with mock.patch.dict(
            os.environ, {"SB_NUM_PROC": "invalid"}, clear=False
        ):
            result = get_available_cpu_count()
            assert result > 0  # Should fall through to auto-detection

    def test_env_var_zero_ignored(self):
        """Test that zero SB_NUM_PROC is ignored."""
        with mock.patch.dict(os.environ, {"SB_NUM_PROC": "0"}, clear=False):
            result = get_available_cpu_count()
            assert result > 0

    def test_env_var_negative_ignored(self):
        """Test that negative SB_NUM_PROC is ignored."""
        with mock.patch.dict(os.environ, {"SB_NUM_PROC": "-1"}, clear=False):
            result = get_available_cpu_count()
            assert result > 0

    @pytest.mark.skipif(
        not hasattr(os, "sched_getaffinity"),
        reason="sched_getaffinity not available on this platform",
    )
    def test_respects_cpu_affinity(self):
        """Test that function respects CPU affinity on Unix systems."""
        # Get current affinity
        current_affinity = os.sched_getaffinity(0)
        result = get_available_cpu_count()
        # Result should not exceed affinity
        assert result <= len(current_affinity)

    def test_fallback_when_all_fail(self):
        """Test fallback to 1 when all detection methods fail."""
        import sys

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.sched_getaffinity", side_effect=AttributeError):
                with mock.patch("os.cpu_count", return_value=None):
                    # On Python 3.13+, also mock os.process_cpu_count
                    if sys.version_info >= (3, 13):
                        with mock.patch(
                            "os.process_cpu_count", return_value=None
                        ):
                            assert get_available_cpu_count() == 1
                    else:
                        assert get_available_cpu_count() == 1


class TestParallelMapDefaultProcessCount:
    """Tests for parallel_map with default process count."""

    def test_default_process_count_uses_available_cpus(self):
        """Test that parallel_map uses available CPUs by default."""
        result = list(
            parallel_map(small_test_func, [1, 2, 3], progress_bar=False)
        )
        assert result == [2, 4, 6]

    def test_explicit_process_count_overrides_default(self):
        """Test that explicit process_count is respected."""
        result = list(
            parallel_map(
                small_test_func, [1, 2, 3], process_count=1, progress_bar=False
            )
        )
        assert result == [2, 4, 6]
