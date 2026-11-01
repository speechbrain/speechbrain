"""Regression tests for lazy module metadata and optional dependencies."""

import importlib
import sys
import warnings

import pytest

from speechbrain.utils.importutils import DeprecatedModuleRedirect, LazyModule


@pytest.fixture
def lazy_target(tmp_path, monkeypatch):
    package = tmp_path / "lazy_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "target.py"
    module.write_text("value = 42\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module
    for name in ("lazy_test_package.target", "lazy_test_package"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["absolute", "relative", "redirect"])
@pytest.mark.parametrize("missing_dependency", [False, True])
def test_file_probe_preserves_lazy_import(
    lazy_target, kind, missing_dependency
):
    if missing_dependency:
        lazy_target.write_text(
            "raise ImportError('optional dependency unavailable')\n",
            encoding="utf-8",
        )

    target = "lazy_test_package.target"
    if kind == "relative":
        proxy = LazyModule("target", "target", "lazy_test_package")
    elif kind == "redirect":
        proxy = DeprecatedModuleRedirect("old_lazy_target", target)
    else:
        proxy = LazyModule("target", target, None)

    with warnings.catch_warnings(record=True) as captured:
        # IPython's autoreloader probes this metadata on registered modules.
        assert not hasattr(proxy, "__file__")
        assert getattr(proxy, "__file__", None) is None
        assert proxy.lazy_module is None
        assert target not in sys.modules
    assert not captured

    if missing_dependency:
        with pytest.raises(ImportError, match="Lazy import") as error:
            _ = proxy.value
        assert "optional dependency unavailable" in str(error.value.__cause__)
        assert proxy.lazy_module is None
        return

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert proxy.value == 42
        assert proxy.__file__ == str(lazy_target)
        assert proxy.value == 42
    assert len(captured) == (1 if kind == "redirect" else 0)
    assert proxy.lazy_module is importlib.import_module(target)


def test_loaded_module_without_file():
    proxy = LazyModule("system", "sys", None)
    assert proxy.version == sys.version
    assert not hasattr(proxy, "__file__")
