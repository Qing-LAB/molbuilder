"""molbuilder.siesta -- SIESTA input generation.

Trajectory parsing is NOT here: it lives entirely in `molbuilder/parse/
engines/`.  This line said "and trajectory parsing" until 2026-09-18,
long after the move.

Submodules:
    input  -- render_fdf / spec_for / SiestaConfig (FDF generation)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.siesta import SiestaConfig`` keep working as the module
became a package.
"""

from ..config.siesta import SiestaConfig
from .input import (
    copy_pseudopotentials,
    find_psml,
    render_fdf,
)
__all__ = [
    "SiestaConfig",
    "copy_pseudopotentials",
    "find_psml",
    "render_fdf",
]
