import sys

import pytest


@pytest.fixture(params=["absolute", "relative", "deprecated"])
def lazy_module(request, tmp_path, monkeypatch):
    from speechbrain.utils.importutils import (
        DeprecatedModuleRedirect,
        LazyModule,
    )

    package_name = "lazy_test_package"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "target.py").write_text("value = 42\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    target_name = f"{package_name}.target"

    if request.param == "relative":
        module = LazyModule("target", "target", package_name)
    elif request.param == "deprecated":
        module = DeprecatedModuleRedirect("old_target", target_name)
    else:
        module = LazyModule("target", target_name, None)

    yield module, package_dir

    sys.modules.pop(target_name, None)
    sys.modules.pop(package_name, None)


def test_lazy_module_file_inspection(lazy_module):
    module, package_dir = lazy_module

    assert not hasattr(module, "__file__")
    assert "lazy_test_package.target" not in sys.modules

    if module.__name__ == "old_target":
        with pytest.warns(UserWarning, match="was deprecated"):
            assert module.value == 42
    else:
        assert module.value == 42

    assert module.__file__ == str(package_dir / "target.py")
    assert module.lazy_module is sys.modules["lazy_test_package.target"]


def test_lazy_module_file_inspection_without_optional_dependency(lazy_module):
    module, package_dir = lazy_module
    (package_dir / "target.py").write_text(
        "import missing_optional_dependency_for_lazy_test\n", encoding="utf-8"
    )

    assert not hasattr(module, "__file__")
    assert "lazy_test_package.target" not in sys.modules

    with pytest.raises(ImportError, match="Lazy import") as exc:
        _ = module.value
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
