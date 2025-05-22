"""
Module importing related utilities.

Author
 * Sylvain de Langen 2024
"""

import importlib
import inspect
import os
import sys
import warnings
from types import ModuleType
from typing import List, Optional


class LazyModule(ModuleType):
    """Defines a module type that lazily imports the target module, thus
    exposing contents without importing the target module needlessly.

    Arguments
    ---------
    name : str
        Name of the module.
    target : str
        Module to be loading lazily.
    package : str, optional
        If specified, the target module load will be relative to this package.
        Depending on how you inject the lazy module into the environment, you
        may choose to specify the package here, or you may choose to include it
        into the `name` with the dot syntax.
        e.g. see how :func:`~lazy_export` and :func:`~deprecated_redirect`
        differ.
    """

    def __init__(
        self,
        name: str,
        target: str,
        package: Optional[str],
    ):
        super().__init__(name)
        self.target = target
        self.lazy_module = None
        self.package = package

    def ensure_module(self, stacklevel: int) -> ModuleType:
        """Ensures that the target module is imported and available as
        `self.lazy_module`, also returning it.

        Arguments
        ---------
        stacklevel : int
            The stack trace level of the function that caused the import to
            occur, relative to the **caller** of this function (e.g. if in
            function `f` you call `ensure_module(1)`, it will refer to the
            function that called `f`).

        Raises
        ------
        AttributeError
            When the function responsible for the import attempt is found to be
            `inspect.py`, we raise an `AttributeError` here. This is because
            some code will inadvertently cause our modules to be imported, such
            as some of PyTorch's op registering machinery.

        Returns
        -------
        The target module after ensuring it is imported.
        """

        importer_frame = None

        # NOTE: ironically, calling this causes getframeinfo to call into
        # `findsource` -> `getmodule` -> ourselves here
        # bear that in mind if you are debugging and checking out the trace.
        # also note that `_getframe` is an implementation detail, but it is
        # somewhat non-critical to us.
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame to check if we should ignore "
                "importing a module lazily. This relies on a CPython "
                "implementation detail, report an issue if you see this with "
                "standard Python and include your version number."
            )

        if importer_frame is not None and importer_frame.filename.endswith(
            "/inspect.py"
        ):
            raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e

        return self.lazy_module

    def __repr__(self) -> str:
        return f"LazyModule(package={self.package}, target={self.target}, loaded={self.lazy_module is not None})"

    def __getattr__(self, attr):
        # NOTE: exceptions here get eaten and not displayed
        return getattr(self.ensure_module(1), attr)


class DeprecatedModuleRedirect(LazyModule):
    """Defines a module type that lazily imports the target module using
    :class:`~LazyModule`, but logging a deprecation warning when the import
    is actually being performed.

    This is only the module type itself; if you want to define a redirection,
    use :func:`~deprecated_redirect` instead.

    Arguments
    ---------
    old_import : str
        Old module import path e.g. `mypackage.myoldmodule`
    new_import : str
        New module import path e.g. `mypackage.mynewcoolmodule.mycoolsubmodule`
    extra_reason : str, optional
        If specified, extra text to attach to the warning for clarification
        (e.g. justifying why the move has occurred, or additional problems to
        look out for).
    """

    def __init__(
        self,
        old_import: str,
        new_import: str,
        extra_reason: Optional[str] = None,
    ):
        super().__init__(name=old_import, target=new_import, package=None)
        self.old_import = old_import
        self.extra_reason = extra_reason

    def _redirection_warn(self):
        """Emits the warning for the redirection (with the extra reason if
        provided)."""

        warning_text = (
            f"Module '{self.old_import}' was deprecated, redirecting to "
            f"'{self.target}'. Please update your script."
        )

        if self.extra_reason is not None:
            warning_text += f" {self.extra_reason}"

        # NOTE: we are not using DeprecationWarning because this gets ignored by
        # default, even though we consider the warning to be rather important
        # in the context of SB

        warnings.warn(
            warning_text,
            # category=DeprecationWarning,
            stacklevel=4,  # ensure_module <- __getattr__ <- python <- user
        )

    def ensure_module(self, stacklevel: int) -> ModuleType:
        should_warn = self.lazy_module is None

        # can fail with exception if the module shouldn't be imported, so only
        # actually emit the warning later
        module = super().ensure_module(stacklevel + 1)

        if should_warn:
            self._redirection_warn()

        return module


def find_imports(file_path: str, find_subpackages: bool = False) -> List[str]:
    """Returns a list of importable scripts in the same module as the specified
    file. e.g. if you have `foo/__init__.py` and `foo/bar.py`, then
    `files_in_module("foo/__init__.py")` then the result will be `["bar"]`.

    Not recursive; this is only applies to the direct modules/subpackages of the
    package at the given path.

    Arguments
    ---------
    file_path : str
        Path of the file to navigate the directory of. Typically the
        `__init__.py` path this is called from, using `__file__`.
    find_subpackages : bool
        Whether we should find the subpackages as well.

    Returns
    -------
    imports : List[str]
        List of importable scripts with the same module.
    """

    imports = []

    module_dir = os.path.dirname(file_path)

    for filename in os.listdir(module_dir):
        if filename.startswith("__"):
            continue

        if filename.endswith(".py"):
            imports.append(filename[:-3])

        if find_subpackages and os.path.isdir(
            os.path.join(module_dir, filename)
        ):
            imports.append(filename)

    return imports


def lazy_export(name: str, package: str):
    """Makes `name` lazily available under the module list for the specified
    `package`, unless it was loaded already, in which case it is ignored.

    Arguments
    ---------
    name : str
        Name of the module, as long as it can get imported with
        `{package}.{name}`.
    package : str
        The relevant package, usually determined with `__name__` from the
        `__init__.py`.

    Returns
    -------
    None
    """

    # already imported for real (e.g. utils.importutils itself)
    if hasattr(sys.modules[package], name):
        return

    setattr(sys.modules[package], name, LazyModule(name, name, package))


def lazy_export_all(
    init_file_path: str, package: str, export_subpackages: bool = False
):
    """Makes all modules under a module lazily importable merely by accessing
    them; e.g. `foo/bar.py` could be accessed with `foo.bar.some_func()`.

    Arguments
    ---------
    init_file_path : str
        Path of the `__init__.py` file, usually determined with `__file__` from
        there.
    package : str
        The relevant package, usually determined with `__name__` from the
        `__init__.py`.
    export_subpackages : bool
        Whether we should make the subpackages (subdirectories) available
        directly as well.
    """

    for name in find_imports(
        init_file_path, find_subpackages=export_subpackages
    ):
        lazy_export(name, package)


def deprecated_redirect(
    old_import: str,
    new_import: str,
    extra_reason: Optional[str] = None,
    also_lazy_export: bool = False,
) -> None:
    """Patches the module list to add a lazy redirection from `old_import` to
    `new_import`, emitting a `DeprecationWarning` when imported.

    Arguments
    ---------
    old_import : str
        Old module import path e.g. `mypackage.myoldmodule`
    new_import : str
        New module import path e.g. `mypackage.mycoolpackage.mynewmodule`
    extra_reason : str, optional
        If specified, extra text to attach to the warning for clarification
        (e.g. justifying why the move has occurred, or additional problems to
        look out for).
    also_lazy_export : bool
        Whether the module should also be exported as a lazy module in the
        package determined in `old_import`.
        e.g. if you had a `foo.bar.somefunc` import as `old_import`, assuming
        you have `foo` imported (or lazy loaded), you could use
        `foo.bar.somefunc` directly without importing `foo.bar` explicitly.
    """

    redirect = DeprecatedModuleRedirect(
        old_import, new_import, extra_reason=extra_reason
    )

    sys.modules[old_import] = redirect

    if also_lazy_export:
        package_sep_idx = old_import.rfind(".")
        old_package = old_import[:package_sep_idx]
        old_module = old_import[package_sep_idx + 1 :]
        if not hasattr(sys.modules[old_package], old_module):
            setattr(sys.modules[old_package], old_module, redirect)
