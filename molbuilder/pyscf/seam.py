"""PySCF's seam -- what PySCF supplies for `prep` to run its steps over it
(`execution/script-preparation.md` § 4): its deck's spec and its warm files.
The part its record in the engine table names (``seam``), resolved by
`jobset.engines.engine_seam`.  Floor 3 (`execution/architecture.md` § 2.1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..jobset.engines import EngineSeam


def pyscf_seam() -> EngineSeam:
    """PySCF's seam -- the part its record in the engine table names
    (``seam``).  NO ``provide_data`` and NO ``sibling_artifacts``, and both
    absences are ANSWERS rather than omissions (`script-preparation.md` § 4,
    W5): PySCF's basis sets ship inside PySCF, so there is no file to put in
    the calculation; and its script's own text instructs nothing to be run
    beside it, so there is no promise to keep.  NO ``shared_package``: there
    is nothing in the calculation for every job to link to.  NO
    ``describe_findings``: there is no data file to settle before one is
    written."""
    from ..jobset.engines import EngineSeam, relabel
    from ..runfiles import deck_role
    from .input import spec_for as _pyscf_spec
    from .stages import traits_for as _pyscf_traits
    from .stages import warm_for as _pyscf_warm
    return EngineSeam(spec_for=_pyscf_spec,
                      suffix=deck_role("pyscf"),
                      relabel=relabel("pyscf"),
                      warm_for=_pyscf_warm, traits_for=_pyscf_traits)


__all__ = ["pyscf_seam"]
