"""molbuilder.siesta -- SIESTA deck description.

Submodules:
    input  -- spec_for / SiestaConfig: the deck's spec, which `jobset prep`
              renders, writes and checks (`script_emit.prepare_deck`)

``SiestaConfig`` and the pseudopotential helpers of ``input`` are
re-exported here, so ``from molbuilder.siesta import SiestaConfig`` reads it.
"""

from ..config.siesta import SiestaConfig
from .input import (
    copy_pseudopotentials,
    find_psml,
)
__all__ = [
    "SiestaConfig",
    "copy_pseudopotentials",
    "find_psml",
]
