"""molbuilder.pyscf -- PySCF script generation.

Trajectory parsing is NOT here: it lives entirely in `molbuilder/parse/
engines/` (`parse/engines/pyscf.py`).  This line said "and trajectory
parsing" until 2026-09-18, long after the move.

Submodules:
    input  -- render_script / spec_for / PySCFConfig (script generation)

The public symbols of ``input`` are re-exported here so existing imports
``from molbuilder.pyscf import PySCFConfig`` work after the
``pyscf_input`` module became a ``pyscf`` package.
"""

from .input import (
    PySCFConfig,
    render_script,
)

__all__ = [
    "PySCFConfig",
    "render_script",
]
