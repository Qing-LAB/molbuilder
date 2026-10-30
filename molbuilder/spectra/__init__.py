"""The vibration kind's own modules -- its result, how it is read, written,
derived and exported (`engines/vibration.md` § 6, the result file;
`web/spectra.md`, the page).

Public surface (engine-agnostic):

  * :class:`molbuilder.spectra.results.SpectraResults`
  * :class:`molbuilder.spectra.results.ModeData`
  * :class:`molbuilder.spectra.results.ModeElectronicStructure`

A kind's own modules (`execution/architecture.md` § 3.3): ``results``
(the ``.spectra.json`` shape and its one reader), ``datafile`` (its
``.spectra.nc``), ``normal_modes`` and ``vibrational_analysis`` (the
harmonic analysis both engines' results rest on;
``docs/science/normal-modes.md``), ``activity``, ``infrared``, ``raman``
and ``optical_data`` (the derivations on request), ``selection`` and
``methods`` (which modes get per-mode ES, the Methods paragraph),
``siesta_vibration`` (the SIESTA job's plan, record and finish steps),
``displacement_sweep`` and ``export``.  The PySCF deck is rendered by
``molbuilder.pyscf.vibration_deck`` through the ordinary ``spec_for``
seam; the PySCF script imports ``normal_modes`` from ``mb_pyscf.pyz``, a
SIESTA finish runs from ``mb_vibration.pyz``.
"""

# NO CONFIG RE-EXPORT.  A spectra calculation is described by `PySCFConfig`;
# the shape the deck and the Methods fragment read is
# `pyscf.vibration_deck.VibrationConfigView`.  A spectra-specific config
# dataclass would carry PySCFConfig's fields plus the four that config view
# already supplies -- a second description of the same calculation, free to
# drift from the one the job actually runs.
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
