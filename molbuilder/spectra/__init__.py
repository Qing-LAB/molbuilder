"""The spectra ARTIFACT layer -- results, mode selection, Methods prose.

See ``docs/web/spectra.md`` for the full design contract.

Public surface (engine-agnostic):

  * :class:`molbuilder.spectra.results.SpectraResults`
  * :class:`molbuilder.spectra.results.ModeData`
  * :class:`molbuilder.spectra.results.ModeElectronicStructure`

THIS PACKAGE HOLDS NO PRODUCER: no engine registry, no engine class,
no standalone generator.  The vibration calculation KIND is the one
producer -- ``molbuilder.pyscf.vibration_deck`` renders the deck
through the ordinary ``spec_for`` seam, composing the emitters in
``molbuilder.pyscf.vibration_emitters``.  Every module in this
package serves the ARTIFACT: ``results`` (the ``.spectra.json``
shape), ``selection`` (which modes get per-mode ES), ``methods``
(the Methods paragraph the deck header carries) -- and the two
engine-agnostic derivations every engine's artifact rests on:
``activity`` (which modes are active in a channel) and ``normal_modes``
(which whole-body motions a vibration removes, and the one path that
removes them; ``docs/science/normal-modes.md``).  The first runs on the
host at serialisation; the second is spliced into a deck as source text,
so it may not lean on a module-scope name.
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
