"""molbuilder.siesta -- SIESTA input generation and trajectory parsing.

Submodules:
    input  -- render_fdf / convert / SiestaConfig (FDF generation)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.siesta import SiestaConfig`` keep working as the module
became a package.
"""

from ..config.siesta import SiestaConfig
from .input import (
    convert,
    copy_pseudopotentials,
    find_psml,
    render_fdf,
)
__all__ = [
    "SiestaConfig",
    "convert",
    "copy_pseudopotentials",
    "find_psml",
    "render_fdf",
]
