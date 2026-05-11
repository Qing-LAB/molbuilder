"""Post-relaxation harmonic vibrational analysis + per-mode electronic
structure for the Spectra tab.

See ``docs/tabs/spectra/spec.md`` for the full design contract.

Public surface (engine-agnostic L1):

  * :class:`molbuilder.spectra.results.SpectraResults`
  * :class:`molbuilder.spectra.results.ModeData`
  * :class:`molbuilder.spectra.results.ModeElectronicStructure`

Engine-specific implementations (L2) register themselves on import
via :func:`molbuilder.spectra.engine_base.register_engine`.  The
PySCF engine is the only one shipped in v1; the SIESTA engine slot
is reserved.
"""

from .results import (
    ModeData,
    ModeElectronicStructure,
    SpectraResults,
)

__all__ = [
    "ModeData",
    "ModeElectronicStructure",
    "SpectraResults",
]
