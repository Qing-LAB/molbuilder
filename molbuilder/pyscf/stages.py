"""PySCF's answer at the **declare** sub-step — `script-preparation.md` § 4.

One of the seam's questions, about the job rather than the deck: *what facts
are its warm files conditioned on*.  It lives here beside `siesta/stages.py`
for the same reason that one does -- the engine knows, and the `jobset` layer
must not.  *What may this run reuse from an earlier one* is every engine's one
door over its warm-files table (`jobset.model.warm_for`).
"""
from __future__ import annotations

from typing import Dict



def traits_for(eff) -> Dict[str, str]:
    """The per-job facts a warm file can be conditioned on -- none on PySCF.

    SIESTA conditions its ``.CG`` on the optimizer both rungs use
    (``requires_same = "optimizer"``).  PySCF has one optimizer, geomeTRIC
    (`engines/pyscf.md` § 3), and
    its rules file conditions nothing on a trait, so there is no fact to
    report.
    """
    return {}

