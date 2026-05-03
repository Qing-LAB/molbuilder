"""Back-compat shim.

The trajectory-log writer moved to ``molbuilder.trajectory_log``.
This module re-exports the public surface so existing imports
``from molbuilder.molwatch_log import write_initial_preview`` keep
working.  New code should import from ``molbuilder.trajectory_log``
directly.
"""

from molbuilder.trajectory_log import write_initial_preview
from molbuilder.trajectory_log import format as format

__all__ = ["write_initial_preview", "format"]
