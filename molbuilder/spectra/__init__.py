"""Post-relaxation harmonic vibrational analysis + per-mode electronic
structure for the Spectra tab.

See ``docs/web/spectra.md`` for the full design contract.

Public surface (engine-agnostic L1):

  * :class:`molbuilder.spectra.results.SpectraResults`
  * :class:`molbuilder.spectra.results.ModeData`
  * :class:`molbuilder.spectra.results.ModeElectronicStructure`

Engine-specific implementations (L2) register themselves on import
via :func:`molbuilder.spectra.engine_base.register_engine`.  The
PySCF engine is the only one shipped in v1; the SIESTA engine slot
is reserved.
"""

# Config re-export matches the pattern at molbuilder/{siesta,pyscf}/__init__.py
# so callers can do `from molbuilder.spectra import SpectraConfig` without
# reaching into the config/ subpackage explicitly.
from ..config.spectra import SpectraConfig
from .engine_base import (
    SpectraEngine,
    UnknownEngineError,
    register_engine,
    get_engine,
    registered_engines,
    unregister_engine,
)
from .results import (
    ModeData,
    ModeElectronicStructure,
    SpectraResults,
    PHASE_EMPTY,
    PHASE_RUNNING,
    PHASE_COMPLETE,
)
from .selection import select_modes, validate_selection
from .methods import render_methods_md, extract_citation_keys

# Import the PySCF engine for its side effect: the module's
# @register_engine decorator runs and the engine becomes
# discoverable via get_engine("pyscf").  Keep this AFTER the
# engine_base import (which defines the registry) and AFTER the
# `register_engine` re-export above (so the registry contract is
# stable by the time the engine imports it).
from . import pyscf_engine as _pyscf_engine  # noqa: F401

__all__ = [
    # Config (L1) -- re-exported from molbuilder.config.spectra
    "SpectraConfig",
    # Result types (L1)
    "ModeData",
    "ModeElectronicStructure",
    "SpectraResults",
    "PHASE_EMPTY",
    "PHASE_RUNNING",
    "PHASE_COMPLETE",
    # Engine plug-in surface (L2)
    "SpectraEngine",
    "UnknownEngineError",
    "register_engine",
    "get_engine",
    "registered_engines",
    "unregister_engine",
    # Mode-selection logic (L2)
    "select_modes",
    "validate_selection",
    # Methods-paragraph composer (L2)
    "render_methods_md",
    "extract_citation_keys",
]
