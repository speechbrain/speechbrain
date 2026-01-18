import pytest


def test_data_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline

    pipeline = DataPipeline(
        ["text"],
        dynamic_items=[
            {"func": lambda x: x.lower(), "takes": ["text"], "provides": "foo"},
            {"func": lambda x: x[::-1], "takes": "foo", "provides": ["bar"]},
        ],
        output_keys=["text", "foo", "bar"],
    )
    result = pipeline({"text": "Test"})
    print(result)
    assert result["bar"] == "tset"
    pipeline = DataPipeline(["foo", "bar"])
    pipeline.add_dynamic_item(
        func=lambda x, y: x + y, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.set_output_keys(["bar", "foobar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert result["foobar"] == 3
    pipeline = DataPipeline(["foo", "bar"])
    from unittest.mock import MagicMock, Mock

    watcher = Mock()
    pipeline.add_dynamic_item(
        provides="foobar", func=watcher, takes=["foo", "bar"]
    )
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(watcher, ["foo", "bar"], "foobar")
    pipeline.add_dynamic_item(lambda x: x, ["foobar"], ["truebar"])
    pipeline.set_output_keys(("truebar",))
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert result["truebar"] == 3
    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(
        func=watcher, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.add_dynamic_item(
        func=lambda x: x, takes=["foo"], provides="truebar"
    )
    pipeline.set_output_keys(("truebar",))
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert result["truebar"] == 1

    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(
        func=watcher, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.set_output_keys(("foobar", "foo"))
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert "foo" in result
    assert "foobar" in result
    assert "bar" not in result
    # Can change the outputs (continues previous tests)
    watcher.reset_mock()
    pipeline.set_output_keys(["bar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert "foo" not in result
    assert "foobar" not in result
    assert "bar" in result
    # Finally, can also still request any specific key:
    computed = pipeline.compute_specific(["foobar"], {"foo": 1, "bar": 2})
    assert watcher.called
    assert computed["foobar"] == 3

    # Output can be a mapping:
    # (Key appears outside, value is internal)
    pipeline.set_output_keys({"signal": "foobar"})
    result = pipeline({"foo": 1, "bar": 2})
    assert result["signal"] == 3


def test_takes_provides():
    from speechbrain.utils.data_pipeline import provides, takes

    @takes("a")
    @provides("b")
    def a_to_b(a):
        """Maps input ``a`` to ``b = a + 1``.

        Arguments
        ---------
        a : int or float
            Input value.

        Returns
        -------
        int or float
            The value ``a + 1``.
        """
        return a + 1

    assert a_to_b(1) == 2
    a_to_b.reset()
    # Normal dynamic item can be called twice:
    assert a_to_b(1) == 2
    assert a_to_b(1) == 2
    # And it knows what it needs:
    assert a_to_b.next_takes() == ("a",)
    # And it knows what it gives:
    assert a_to_b.next_provides() == ("b",)


def test_MIMO_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline, provides, takes

    @takes("text", "other-text")
    @provides("reversed", "concat")
    def text_pipeline(text, other):
        """Creates two text variants for the MIMO pipeline.

        Arguments
        ---------
        text : str
            Input text sequence.
        other : str
            Auxiliary text sequence.

        Returns
        -------
        tuple[str, str]
            A tuple ``(reversed, concat)`` where ``reversed`` is
            ``text[::-1]`` and ``concat`` is ``text + other``.
        """
        return text[::-1], text + other

    @takes("reversed", "concat")
    @provides("reversed_twice", "double_concat")
    def second_pipeline(rev, concat):
        """Yields second-stage text transforms for the MIMO pipeline.

        Arguments
        ---------
        rev : str
            Previously reversed text.
        concat : str
            Concatenated text from the first stage.

        Yields
        ------
        str
            First, the text reversed back (``rev[::-1]``).
        str
            Second, the concatenation repeated twice (``concat + concat``).
        """
        yield rev[::-1]
        yield concat + concat

    @provides("hello-world")
    def provider():
        """Provides a constant greeting string.

        Yields
        ------
        str
            The literal string ``\"hello-world\"``.
        """
        yield "hello-world"

    @takes("hello-world", "reversed_twice")
    @provides("message")
    def messenger(hello, name):
        """Formats a greeting message from its components.

        Arguments
        ---------
        hello : str
            Greeting prefix, e.g., ``\"hello-world\"``.
        name : str
            Name or identifier to greet.

        Returns
        -------
        str
            Formatted message ``f\"{hello}, {name}\"``.
        """
        return f"{hello}, {name}"

    pipeline = DataPipeline(
        ["text", "other-text"],
        dynamic_items=[second_pipeline, text_pipeline],
        output_keys=["text", "reversed", "reversed_twice"],
    )
    result = pipeline({"text": "abc", "other-text": "def"})
    assert result["reversed"] == "cba"
    assert result["reversed_twice"] == "abc"
    result = pipeline.compute_specific(
        ["concat"], {"text": "abc", "other-text": "def"}
    )
    assert result["concat"] == "abcdef"
    result = pipeline.compute_specific(
        ["double_concat"], {"text": "abc", "other-text": "def"}
    )
    assert result["double_concat"] == "abcdefabcdef"
    assert "concat" not in result

    # Add messenger but not provider, so "hello-world" is unaccounted for:
    pipeline.add_dynamic_item(messenger)
    with pytest.raises(RuntimeError):
        pipeline.compute_specific(
            ["message"], {"text": "abc", "other-text": "def"}
        )
    # Now add provider, so that the unaccounted for hello-world key gets accounted for.
    pipeline.add_dynamic_item(provider)
    result = pipeline.compute_specific(
        ["message"], {"text": "abc", "other-text": "def"}
    )
    assert result["message"] == "hello-world, abc"


def test_cached_dynamic_item(tmp_path):
    """Test CachedDynamicItem basic functionality."""

    import torch

    from speechbrain.utils.data_pipeline import (
        CachedDynamicItem,
        provides,
        takes,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Test basic caching
    call_count = 0

    @takes("id", "text")
    @provides("tokenized")
    def tokenize(id, text):
        """Tokenizes and normalizes input text for a given id.

        Arguments
        ---------
        id : str
            Unique identifier used as cache key.
        text : str
            Input text to be normalized and split into tokens.

        Returns
        -------
        list[str]
            List of lowercase tokens produced from ``text``.
        """
        nonlocal call_count
        call_count += 1
        return text.strip().lower().split()

    cached_tokenize = CachedDynamicItem(
        cache_dir, takes=["id", "text"], func=tokenize, provides=["tokenized"]
    )

    # First call should compute and cache
    result1 = cached_tokenize("utt1", "  Hello World  ")
    assert result1 == ["hello", "world"]
    assert call_count == 1
    assert (cache_dir / "utt1.pt").exists()

    # Second call with same id should use cache
    result2 = cached_tokenize("utt1", "  Hello World  ")
    assert result2 == ["hello", "world"]
    assert call_count == 1  # Should not increment

    # Different id should compute again
    result3 = cached_tokenize("utt2", "  Test Sentence  ")
    assert result3 == ["test", "sentence"]
    assert call_count == 2
    assert (cache_dir / "utt2.pt").exists()

    # Verify cache files contain correct data
    cached_data1 = torch.load(cache_dir / "utt1.pt")
    assert cached_data1 == ["hello", "world"]
    cached_data2 = torch.load(cache_dir / "utt2.pt")
    assert cached_data2 == ["test", "sentence"]


def test_cached_dynamic_item_decorator(tmp_path):
    """Test CachedDynamicItem.cache decorator."""
    from speechbrain.utils.data_pipeline import (
        CachedDynamicItem,
        provides,
        takes,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    call_count = 0

    @CachedDynamicItem.cache(cache_dir)
    @takes("id", "text")
    @provides("tokenized")
    def tokenize(id, text):
        nonlocal call_count
        call_count += 1
        return text.strip().lower().split()

    # First call
    result1 = tokenize("utt1", "  Hello World  ")
    assert result1 == ["hello", "world"]
    assert call_count == 1
    assert (cache_dir / "utt1.pt").exists()

    # Second call should use cache
    result2 = tokenize("utt1", "  Bonjour  ")
    assert result2 == ["hello", "world"]
    assert call_count == 1

    # Verify it's a CachedDynamicItem
    assert isinstance(tokenize, CachedDynamicItem)


def test_cached_dynamic_item_validation(tmp_path):
    """Test CachedDynamicItem validation errors."""
    from speechbrain.utils.data_pipeline import CachedDynamicItem

    cache_dir = tmp_path / "cache"

    # Test empty takes list
    with pytest.raises(
        ValueError, match="Expected 'takes' list to have at least one item"
    ):
        CachedDynamicItem(
            cache_dir, takes=[], func=lambda x: x, provides=["out"]
        )

    # Test first item not "id"
    with pytest.raises(
        ValueError, match="First item in 'takes' list must be 'id'"
    ):
        CachedDynamicItem(
            cache_dir, takes=["text"], func=lambda x: x, provides=["out"]
        )

    # Test decorator with non-DynamicItem
    with pytest.raises(ValueError, match="Can only cache a DynamicItem"):
        CachedDynamicItem.cache(cache_dir)(lambda x: x)


def test_cached_dynamic_item_torch_tensors(tmp_path):
    """Test CachedDynamicItem with PyTorch tensors."""
    import torch

    from speechbrain.utils.data_pipeline import (
        CachedDynamicItem,
        provides,
        takes,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedDynamicItem.cache(cache_dir)
    @takes("id", "data")
    @provides("processed")
    def process_tensor(id, data):
        """Applies a simple scaling transform and caches the result.

        Arguments
        ---------
        id : str
            Unique identifier used as cache key.
        data : torch.Tensor or number
            Input value or tensor to be doubled.

        Returns
        -------
        same as data
            The value ``data * 2``.
        """
        return data * 2

    # Test with tensor
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    result1 = process_tensor("tensor1", input_tensor)
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(result1, expected)

    # Second call should use cache
    result2 = process_tensor("tensor1", input_tensor)
    assert torch.allclose(result2, expected)

    # Verify cache file
    cached_tensor = torch.load(cache_dir / "tensor1.pt")
    assert torch.allclose(cached_tensor, expected)


def test_cached_dynamic_item_cache_methods(tmp_path):
    """Test CachedDynamicItem internal cache methods."""
    from speechbrain.utils.data_pipeline import (
        CachedDynamicItem,
        provides,
        takes,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @CachedDynamicItem.cache(cache_dir)
    @takes("id", "value")
    @provides("doubled")
    def double(id, value):
        """Doubles a scalar value to exercise CachedDynamicItem internals.

        Arguments
        ---------
        id : str
            Unique identifier used as cache key.
        value : int or float
            Input scalar to be doubled.

        Returns
        -------
        int or float
            The value ``value * 2``.
        """
        return value * 2

    # Test _is_cached
    assert not double._is_cached("test_id")
    result = double("test_id", 5)
    assert result == 10
    assert double._is_cached("test_id")

    # Test _uid2path
    path = double._uid2path("test_id")
    assert path == cache_dir / "test_id.pt"
    assert path.exists()

    # Test _load
    loaded = double._load("test_id")
    assert loaded == 10

    # Test _cache
    double._cache(42, "new_id")
    assert double._is_cached("new_id")
    loaded_new = double._load("new_id")
    assert loaded_new == 42
