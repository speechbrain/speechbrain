"""Tests for CachedHDF5DynamicItem

Authors:
 * Adel Moumen, 2025
"""

import numpy as np
import pytest
import torch

from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem
from speechbrain.utils.data_pipeline import provides, takes


def test_cached_hdf5_dynamic_item_basic(tmp_path):
    """Test CachedHDF5DynamicItem basic functionality."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    call_count = 0

    @takes("id", "limit")
    @provides("array")
    def count_to(id, limit):
        """Return np.arange(limit) for the given id."""
        nonlocal call_count
        call_count += 1
        return np.arange(limit)

    cached_func = CachedHDF5DynamicItem(
        cache_dir,
        takes=["id", "limit"],
        func=count_to,
        provides=["array"],
    )

    # First call should compute and cache
    result1 = cached_func("utt_id", 5)
    expected = np.arange(5)
    np.testing.assert_array_equal(result1, expected)
    assert call_count == 1
    assert "utt_id" in cached_func.hdf5file

    # Second call with same id should use cache
    result2 = cached_func("utt_id", 5)
    np.testing.assert_array_equal(result2, expected)
    assert call_count == 1  # Should not increment

    # Different id should compute again
    result3 = cached_func("utt_id2", 3)
    expected2 = np.arange(3)
    np.testing.assert_array_equal(result3, expected2)
    assert call_count == 2
    assert "utt_id2" in cached_func.hdf5file

    # Verify cache contains correct data
    cached_data1 = cached_func.hdf5file["utt_id"][:]
    np.testing.assert_array_equal(cached_data1, expected)
    cached_data2 = cached_func.hdf5file["utt_id2"][:]
    np.testing.assert_array_equal(cached_data2, expected2)

    # Clean up
    cached_func.hdf5file.close()


def test_cached_hdf5_dynamic_item_decorator(tmp_path):
    """Test CachedHDF5DynamicItem.cache decorator."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    call_count = 0

    @CachedHDF5DynamicItem.cache(cache_dir)
    @takes("id", "limit")
    @provides("array")
    def count_to(id, limit):
        """Return np.arange(limit) for the given id using cached HDF5 backend."""
        nonlocal call_count
        call_count += 1
        return np.arange(limit)

    # First call
    result1 = count_to("utt_id", 5)
    expected = np.arange(5)
    np.testing.assert_array_equal(result1, expected)
    assert call_count == 1
    assert "utt_id" in count_to.hdf5file

    # Second call should use cache
    result2 = count_to("utt_id", 5)
    np.testing.assert_array_equal(result2, expected)
    assert call_count == 1

    # Verify it's a CachedHDF5DynamicItem
    assert isinstance(count_to, CachedHDF5DynamicItem)

    # Clean up
    count_to.hdf5file.close()


def test_cached_hdf5_dynamic_item_validation(tmp_path):
    """Test CachedHDF5DynamicItem validation errors."""
    cache_dir = tmp_path / "cache"

    # Test decorator with non-DynamicItem
    with pytest.raises(ValueError, match="Can only cache a DynamicItem"):
        CachedHDF5DynamicItem.cache(cache_dir)(lambda x: x)


def test_cached_hdf5_dynamic_item_file_mode(tmp_path):
    """Test CachedHDF5DynamicItem file mode handling."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir, file_mode="a")
    @takes("id", "value")
    @provides("doubled")
    def double(id, value):
        """Return value * 2 stored in HDF5 cache."""
        return np.array([value * 2])

    # Create some cache entries
    result1 = double("id1", 5)
    assert result1[0] == 10

    # Change to read-only mode
    double.change_file_mode("r")
    assert double.file_mode == "r"

    # Should still be able to read from cache
    result2 = double("id1", 5)
    assert result2[0] == 10

    # Should not be able to write in read-only mode
    # h5py raises OSError when trying to create_dataset in read-only mode
    with pytest.raises((OSError, ValueError)):
        double("id2", 3)

    # Clean up
    double.hdf5file.close()


def test_cached_hdf5_dynamic_item_compression(tmp_path):
    """Test CachedHDF5DynamicItem with compression."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir, compression="gzip")
    @takes("id", "data")
    @provides("processed")
    def process_data(id, data):
        return data * 2

    input_data = np.array([1.0, 2.0, 3.0])
    result1 = process_data("compressed_id", input_data)
    expected = np.array([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result1, expected)

    # Second call should use cache
    result2 = process_data("compressed_id", input_data)
    np.testing.assert_array_equal(result2, expected)

    # Verify compression is set
    assert process_data.compression == "gzip"

    # Clean up
    process_data.hdf5file.close()


def test_cached_hdf5_dynamic_item_custom_filename(tmp_path):
    """Test CachedHDF5DynamicItem with custom cache filename."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    custom_filename = "my_cache.hdf5"

    @CachedHDF5DynamicItem.cache(cache_dir, cache_filename=custom_filename)
    @takes("id", "value")
    @provides("doubled")
    def double(id, value):
        """Return value * 2 stored in custom-named HDF5 cache."""
        return np.array([value * 2])

    result = double("test_id", 5)
    assert result[0] == 10

    # Verify custom filename is used
    expected_path = cache_dir / custom_filename
    assert expected_path.exists()

    # Clean up
    double.hdf5file.close()


def test_cached_hdf5_dynamic_item_cache_methods(tmp_path):
    """Test CachedHDF5DynamicItem internal cache methods."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir)
    @takes("id", "value")
    @provides("doubled")
    def double(id, value):
        """Return value * 2 and test low-level cache helpers."""
        return np.array([value * 2])

    # Test _is_cached
    assert not double._is_cached("test_id")
    result = double("test_id", 5)
    assert result[0] == 10
    assert double._is_cached("test_id")

    # Test _load
    loaded = double._load("test_id")
    np.testing.assert_array_equal(loaded, np.array([10]))

    # Test _cache
    double._cache(np.array([42]), "new_id")
    assert double._is_cached("new_id")
    loaded_new = double._load("new_id")
    np.testing.assert_array_equal(loaded_new, np.array([42]))

    # Clean up
    double.hdf5file.close()


def test_cached_hdf5_dynamic_item_torch_tensors(tmp_path):
    """Test CachedHDF5DynamicItem with PyTorch tensors."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir)
    @takes("id", "data")
    @provides("processed")
    def process_tensor(id, data):
        """Return tensor or array multiplied by 2, stored via HDF5."""
        # Convert to numpy for HDF5 storage
        if isinstance(data, torch.Tensor):
            return data.numpy() * 2
        return data * 2

    # Test with tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    result1 = process_tensor("tensor1", input_tensor)
    expected = np.array([2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result1, expected)

    # Second call should use cache
    result2 = process_tensor("tensor1", input_tensor)
    np.testing.assert_array_equal(result2, expected)

    # Clean up
    process_tensor.hdf5file.close()


def test_cached_hdf5_dynamic_item_multiple_items(tmp_path):
    """Test CachedHDF5DynamicItem with multiple cached items."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir)
    @takes("id", "value")
    @provides("squared")
    def square(id, value):
        """Return value ** 2 stored in a shared HDF5 cache file."""
        return np.array([value**2])

    # Create multiple cache entries
    results = {}
    for i in range(5):
        uid = f"item_{i}"
        result = square(uid, i)
        results[uid] = result[0]
        assert result[0] == i**2

    # Verify all are cached
    for i in range(5):
        uid = f"item_{i}"
        assert square._is_cached(uid)
        loaded = square._load(uid)
        assert loaded[0] == i**2

    # Verify all are in the same HDF5 file
    assert len(square.hdf5file.keys()) == 5

    # Clean up
    square.hdf5file.close()


def test_cached_hdf5_dynamic_item_inheritance(tmp_path):
    """Test that CachedHDF5DynamicItem properly inherits from CachedDynamicItem."""
    from speechbrain.utils.data_pipeline import CachedDynamicItem

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedHDF5DynamicItem.cache(cache_dir)
    @takes("id", "value")
    @provides("doubled")
    def double(id, value):
        return np.array([value * 2])

    # Should be instance of both classes
    assert isinstance(double, CachedHDF5DynamicItem)
    assert isinstance(double, CachedDynamicItem)

    # Should have HDF5-specific attributes
    assert hasattr(double, "hdf5file")
    assert hasattr(double, "file_mode")
    assert hasattr(double, "compression")

    # Clean up
    double.hdf5file.close()
