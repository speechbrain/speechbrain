"""
Module importing related utilities.

Author
 * Sylvain de Langen 2024
"""

from types import ModuleType
import importlib
import sys
import warnings


class LegacyModuleRedirect(ModuleType):
    def __init__(self, old_import, new_import):
        super().__init__(old_import)
        self.old_import = old_import
        self.new_import = new_import
        self.lazy_module = None

    def __getattr__(self, attr):
        if self.lazy_module is None:
            warnings.warn(
                f"Module '{self.old_import}' was deprecated, redirecting to '{self.new_import}'",
                category=DeprecationWarning,
                stacklevel=2,
            )
            self.lazy_module = importlib.import_module(self.new_import)

        # NOTE: exceptions here get eaten and not displayed

        return getattr(self.lazy_module, attr)


def deprecated_redirect(old_import: str, new_import: str) -> None:
    """Patches the module list to add a lazy redirection from `old_import` to
    `new_import`, emitting a `DeprecationWarning` when imported.

    Arguments
    ---------
    old_import: str
        Old module import path e.g. `mypackage.myoldmodule`
    new_import: str
        New module import path e.g. `mypackage.mynewcoolmodule.mycoolsubmodule`
    """

    sys.modules[old_import] = LegacyModuleRedirect(old_import, new_import)
