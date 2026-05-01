"""molbuilder.siesta -- SIESTA input generation and trajectory parsing.

Submodules:
    input  -- render_fdf / convert / SiestaConfig (FDF generation)
    parser -- SiestaParser for live-trajectory reading (added in Phase 2.1
              of the molwatch merge; not yet present)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.siesta import SiestaConfig`` keep working as the module
became a package.
"""

from .input import (
    Config,
    SiestaConfig,
    convert,
    copy_pseudopotentials,
    find_psml,
    render_fdf,
)
# Internal helpers exposed for tests.  Not part of the public API.
from .input import _auto_block_size, _detect_species, _wrap_into_cell

__all__ = [
    "Config",
    "SiestaConfig",
    "convert",
    "copy_pseudopotentials",
    "find_psml",
    "render_fdf",
]
