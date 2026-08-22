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

# Config re-export matches the pattern at molbuilder/{siesta,pyscf}/__init__.py
# so callers can do `from molbuilder.spectra import SpectraConfig` without
# reaching into the config/ subpackage explicitly.
from ..config.spectra import SpectraConfig
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
    # Config -- re-exported from molbuilder.config.spectra
    "SpectraConfig",
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
