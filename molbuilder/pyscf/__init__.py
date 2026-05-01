"""molbuilder.pyscf -- PySCF script generation and trajectory parsing.

Submodules:
    input  -- render_script / convert / PySCFConfig (script generation)
    parser -- PySCFParser for live-trajectory reading (added in Phase 2.1
              of the molwatch merge; not yet present)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.pyscf import PySCFConfig`` work after the
``pyscf_input`` module became a ``pyscf`` package.
"""

from .input import (
    PySCFConfig,
    convert,
    render_script,
)

__all__ = [
    "PySCFConfig",
    "convert",
    "render_script",
]
