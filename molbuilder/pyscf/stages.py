"""PySCF's answers at the **declare** sub-step — `script-preparation.md` § 4.

Two of the seam's fifteen questions, and both are about the job rather than the
deck: *what may this run reuse from an earlier one*, and *what must the wrapper
route on*.  They live here beside `siesta/stages.py` for the same reason that
one does -- the engine knows, and the `jobset` layer must not.
"""
from __future__ import annotations

from typing import Dict, List

from ..jobset.model import WarmFile



def traits_for(eff) -> Dict[str, str]:
    """The per-job facts a warm file can be conditioned on -- none on PySCF.

    SIESTA conditions its ``.CG`` on the optimizer both rungs use
    (``requires_same = "optimizer"``).  PySCF has one optimizer, geomeTRIC
    (`engines/pyscf.md` § 3), and
    its rules file conditions nothing on a trait, so there is no fact to
    report.
    """
    return {}


def warm_for(label: str, eff,
             calculation: str = "optimization",
             base_dir=None) -> List[WarmFile]:
    """What a PySCF stage takes from a run it continues.

    **The rows come from the rules file**, ``pyscf/warm-files.toml``
    (`job-contracts.md` § 4.2a) -- the checkpoint for every calculation, plus
    the optimized geometry for an optimization.  Nothing is listed here that
    the file does not say.

    **Gated on the stage actually continuing**, exactly as SIESTA's twin is:
    `run-identity.md` § 4 rule 2 makes ``restart`` the one field that says so,
    and the same answer gates the generated script's checkpoint and
    optimized-geometry reads.  Placing files for a ``clean`` stage would leave
    them sitting unread -- § 4's *"present but not honoured"*, the silent half
    of the pair.
    """
    from ..identity import continues
    from ..jobset.model import warm_declaration
    if not continues(eff):
        return []
    return warm_declaration("pyscf", label, calculation, base_dir)
