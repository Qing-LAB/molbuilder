"""molbuilder.pyscf -- PySCF deck description.

Submodules:
    input  -- spec_for / PySCFConfig: the deck's spec, which `jobset prep`
              renders, writes and checks (`script_emit.prepare_deck`)

``PySCFConfig`` is re-exported here, so ``from molbuilder.pyscf import
PySCFConfig`` reads it.
"""

from .input import PySCFConfig

__all__ = ["PySCFConfig"]
