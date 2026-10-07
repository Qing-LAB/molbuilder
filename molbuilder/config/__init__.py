"""molbuilder.config -- engine-parameter dataclasses (L1 nouns).

These are pure data: each engine config is a ``@dataclass`` with field
metadata the validation pass reads (`web/form-schema.md` § 1a); the forms
are drawn from the catalogue.  The L2 deck specs (``molbuilder.siesta.input.spec_for``,
``molbuilder.pyscf.input.spec_for``) consume them; nothing in
this package imports from them.

Public symbols are also re-exported by the engine-package __init__s
(``molbuilder.siesta``, ``molbuilder.pyscf``), so
``from molbuilder.siesta import SiestaConfig`` reads it.
"""

from .pyscf  import PySCFConfig
from .siesta import SiestaConfig

__all__ = ["SiestaConfig", "PySCFConfig"]
