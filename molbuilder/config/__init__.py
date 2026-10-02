"""molbuilder.config -- engine-parameter dataclasses (L1 nouns).

These are pure data: each engine config is a ``@dataclass`` with field
metadata the validation pass reads (`web/form-schema.md` § 1a); the forms
are drawn from the catalogue.  The L2 generators (``molbuilder.siesta.input.render_fdf``,
``molbuilder.pyscf.input.render_script``) consume them; nothing in
this package imports from generators.

Public symbols are also re-exported by the engine-package __init__s
(``molbuilder.siesta``, ``molbuilder.pyscf``) so existing imports
``from molbuilder.siesta import SiestaConfig`` keep working.
"""

from .pyscf  import PySCFConfig
from .siesta import SiestaConfig

__all__ = ["SiestaConfig", "PySCFConfig"]
