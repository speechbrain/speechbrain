import pytest

from speechbrain.utils import fetching


def test_link_with_strategy_symlink_and_copy(tmp_path):
    # Create a source file
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("testdata")

    # Test COPY
    result = fetching.link_with_strategy(src, dst, fetching.LocalStrategy.COPY)
    assert dst.exists()
    assert dst.read_text() == "testdata"
    assert result == dst

    # Overwrite with SYMLINK
    dst.unlink()
    result = fetching.link_with_strategy(
        src, dst, fetching.LocalStrategy.SYMLINK
    )
    assert dst.is_symlink()
    assert dst.resolve() == src

    # Test NO_LINK
    result = fetching.link_with_strategy(
        src, dst, fetching.LocalStrategy.NO_LINK
    )
    assert result == src


def test_link_with_strategy_self_symlink(tmp_path):
    # Create a file that is a symlink to itself (simulate error)
    path = tmp_path / "loop.txt"
    path.write_text("content")
    path.unlink()
    path.symlink_to(path)
    with pytest.raises(ValueError):
        fetching.link_with_strategy(path, path, fetching.LocalStrategy.SYMLINK)


def test_guess_source_local_and_uri(tmp_path):
    # Local directory
    srcdir = tmp_path
    fetch_from, path = fetching.guess_source(str(srcdir))
    assert fetch_from == fetching.FetchFrom.LOCAL

    # URI
    fetch_from, path = fetching.guess_source("http://example.com")
    assert fetch_from == fetching.FetchFrom.URI

    # Huggingface fallback
    fetch_from, path = fetching.guess_source("facebook/wav2vec2-base-960h")
    assert fetch_from == fetching.FetchFrom.HUGGING_FACE


def test_fetch_local_file(tmp_path):
    # Setup local file and dest
    srcdir = tmp_path / "src"
    srcdir.mkdir()
    f = srcdir / "foo.txt"
    f.write_text("abc123")

    destdir = tmp_path / "dest"
    outpath = fetching.fetch(
        "foo.txt",
        str(srcdir),
        savedir=str(destdir),
        local_strategy=fetching.LocalStrategy.COPY,
    )
    assert outpath.exists()
    assert outpath.read_text() == "abc123"
    assert outpath.parent == destdir


def test_fetch_raises_on_bad_uri(tmp_path):
    # Should raise ValueError when fetching from URI without savedir
    with pytest.raises(ValueError):
        fetching.fetch("foo.txt", "http://example.com", savedir=None)


def test_fetch_raises_on_network_disallowed(tmp_path):
    destdir = tmp_path / "dest"
    destdir.mkdir()
    config = fetching.FetchConfig(allow_network=False)
    with pytest.raises(ValueError):
        fetching.fetch(
            "foo.txt",
            "http://example.com",
            savedir=str(destdir),
            fetch_config=config,
        )
