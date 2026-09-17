"""molbuilder.pyscf -- PySCF script generation and trajectory parsing.

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
