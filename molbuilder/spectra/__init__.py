"""The spectra ARTIFACT layer -- results, mode selection, Methods prose.

See ``docs/web/spectra.md`` for the full design contract.

Public surface (engine-agnostic):

  * :class:`molbuilder.spectra.results.SpectraResults`
  * :class:`molbuilder.spectra.results.ModeData`
  * :class:`molbuilder.spectra.results.ModeElectronicStructure`

THE PRODUCER LEFT (spectra-migration plan P3, 2026-08-21): the old
engine registry (``engine_base``), the PySCF engine class
(``pyscf_engine``) and the standalone generator (``pyscf_script``)
retired when the vibration calculation KIND became the one producer --
``molbuilder.pyscf.vibration_deck`` renders the deck through the
ordinary ``spec_for`` seam, composing the emitters that moved to
``molbuilder.pyscf.vibration_emitters``.  Every module left in this
package serves the ARTIFACT: ``results`` (the ``.spectra.json``
shape), ``selection`` (which modes get per-mode ES), ``methods``
(the Methods paragraph the deck header carries).
"""

# NO CONFIG RE-EXPORT.  `SpectraConfig` was retired 2026-08-22: a 33-field
# dataclass that nothing in production constructed, whose fields were
# PySCFConfig's plus four the vibration deck's config view supplies.  A
# spectra calculation is described by `PySCFConfig`; the shape the deck and
# the Methods fragment read is `pyscf.vibration_deck.VibrationConfigView`.
from .results import (
    ModeData,
    ModeElectronicStructure,
    SpectraResults,
    PHASE_EMPTY,
    PHASE_RUNNING,
    PHASE_COMPLETE,
)
from .selection import select_modes
from .methods import render_methods_md, extract_citation_keys

__all__ = [
    # Result types
    "ModeData",
    "ModeElectronicStructure",
    "SpectraResults",
    "PHASE_EMPTY",
    "PHASE_RUNNING",
    "PHASE_COMPLETE",
    # Mode-selection logic
    "select_modes",
    # Methods-paragraph composer
    "render_methods_md",
    "extract_citation_keys",
]
