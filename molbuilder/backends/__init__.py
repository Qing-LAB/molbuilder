"""Back-compat shim.

The build backends moved to ``molbuilder.builders.backends``.  This
module re-exports the public surface and the per-engine submodules
so existing imports keep working:

    from molbuilder.backends import dispatch, available_backends
    from molbuilder.backends import _amber, _rdkit, _threedna

New code should prefer ``molbuilder.builders.backends`` directly.
"""

from molbuilder.builders.backends import (
    BackendUnavailable,
    auto_backend_name,
    available_backends,
    dispatch,
)
# Submodule access for code that does `from molbuilder.backends import _amber`
# etc.  The `as` binding makes the submodule visible as an attribute of this
# package without importing it again into a different namespace.
from molbuilder.builders.backends import _amber    as _amber
from molbuilder.builders.backends import _common   as _common
from molbuilder.builders.backends import _rdkit    as _rdkit
from molbuilder.builders.backends import _threedna as _threedna

__all__ = [
    "BackendUnavailable", "auto_backend_name",
    "available_backends", "dispatch",
]
