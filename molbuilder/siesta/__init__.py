"""molbuilder.siesta -- SIESTA input generation and trajectory parsing.

Submodules:
    input  -- render_fdf / convert / SiestaConfig (FDF generation)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.siesta import SiestaConfig`` keep working as the module
became a package.
"""

from ..config.siesta import Config, SiestaConfig
from .input import (
    convert,
    copy_pseudopotentials,
    find_psml,
    render_fdf,
)
# Internal helpers exposed for tests.  Not part of the public API.
# Drop _wrap_into_cell -- nobody imports it anymore (was used by an
# older test that's since been rewritten).  Listed in __all__ so
# pyflakes / mypy don't flag the imports as dead -- the test module
# `from molbuilder.siesta import _detect_species` is the real caller.
from .input import _auto_block_size, _detect_species

__all__ = [
    "Config",
    "SiestaConfig",
    "convert",
    "copy_pseudopotentials",
    "find_psml",
    "render_fdf",
    "_auto_block_size",
    "_detect_species",
]
