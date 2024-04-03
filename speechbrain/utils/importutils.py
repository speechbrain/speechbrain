"""
Module importing related utilities.

Author
 * Sylvain de Langen 2024
"""

from types import ModuleType
import importlib
import sys
from typing import Optional
import warnings


class LegacyModuleRedirect(ModuleType):
    """Defines a module type that lazily imports the target module (and warns
    about the deprecation when this happens), thus allowing deprecated
    redirections to be defined without immediately importing the target module
    needlessly.

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
        super().__init__(old_import)
        self.old_import = old_import
        self.new_import = new_import
        self.extra_reason = extra_reason
        self.lazy_module = None

    def _redirection_warn(self):
        """Emits the warning for the redirection (with the extra reason if
        provided)."""

        warning_text = (
            f"Module '{self.old_import}' was deprecated, redirecting to "
            f"'{self.new_import}'. Please update your script."
        )

        if self.extra_reason is not None:
            warning_text += f" {self.extra_reason}"

        # NOTE: we are not using DeprecationWarning because this gets ignored by
        # default, even though we consider the warning to be rather important
        # in the context of SB

        warnings.warn(
            warning_text,
            # category=DeprecationWarning,
            stacklevel=3,
        )

    def __getattr__(self, attr):
        # NOTE: exceptions here get eaten and not displayed

        if self.lazy_module is None:
            self._redirection_warn()
            self.lazy_module = importlib.import_module(self.new_import)

        return getattr(self.lazy_module, attr)


def deprecated_redirect(
    old_import: str, new_import: str, extra_reason: Optional[str] = None
) -> None:
    """Patches the module list to add a lazy redirection from `old_import` to
    `new_import`, emitting a `DeprecationWarning` when imported.

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

    sys.modules[old_import] = LegacyModuleRedirect(
        old_import, new_import, extra_reason=extra_reason
    )
