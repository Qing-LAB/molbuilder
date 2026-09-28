"""Mode-selection logic for the Spectra tab's L4 step.

Pure functions, no I/O.  Spec § 8 + § 8.1 + § 2.5.3 codify the
semantics; this module is the executable form.

Public surface:

  * :func:`select_modes(modes, cfg, prior=None) -> List[int]` --
    given the full mode list from L2 + the user's config view
    + (optionally) the prior :class:`SpectraResults` from a
    previous run, return the 1-based indices of modes that should
    get per-mode displaced-geometry SCFs in this run.

The three selectors (`engines/vibration.md` § 4.8):

    skip      -> []  (no L4 work)
    all       -> every mode (respecting the freq filter)
    explicit  -> exactly cfg.es_explicit_indices (frequency
                 filter IGNORED -- the user named specific modes,
                 the window doesn't override)

(`top_n` and `threshold` ranked modes by Raman activity; retired by
decision 2026-09-23 and removed 2026-09-28, V1.6 -- the probe measures
d(eps)/dQ, which follows its own selection rule.)

The freq filter (cfg.freq_min_cm1 / cfg.freq_max_cm1, either
side optionally None) restricts `all` to the modes in that window.

Resume / non-destructive L4 (spec § 2.5.2): when ``prior`` is
provided, modes that already have ``electronic_structure``
populated in the prior results are excluded from the returned
list -- the engine skips them on this run (their ES data stays
in the file).  This is the "incrementally add modes" workflow.
"""

from __future__ import annotations

from typing import List, Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..pyscf.vibration_deck import VibrationConfigView
from .results import ModeData, SpectraResults


def select_modes(modes: List[ModeData],
                 cfg: "VibrationConfigView",
                 prior: Optional[SpectraResults] = None) -> List[int]:
    """Return the 1-based mode indices selected for L4 (per-mode
    displaced-geometry SCFs) under the configured selector + filter.

    Pure: does not consult disk, does not mutate inputs.  The
    engine's render_script feeds this into the script template; the
    test suite asserts each selector / filter / resume combination
    independently.

    Behaviour summary (full spec at § 8 + § 8.1):

      * cfg.es_mode_selection == "skip"  -> []
      * cfg.es_mode_selection == "all"   -> every mode (after freq filter)
      * cfg.es_mode_selection == "explicit" -> cfg.es_explicit_indices
                                            (freq filter IGNORED)

      * When ``prior`` is supplied, modes that already have ES data
        in the prior results are removed from the returned list --
        the engine skips them on this run (their ES persists).
    """
    sel = cfg.es_mode_selection

    # 1. Resolve the base set per selector.
    if sel == "skip":
        base: List[int] = []

    elif sel == "all":
        base = [m.index_1based for m in modes
                if _passes_freq_window(m, cfg)]

    elif sel == "explicit":
        # Frequency filter intentionally IGNORED -- the user named
        # specific modes, the window doesn't override (the archived
        # spec's § 8.1 rule, now stated here).  Index-range validation
        # is the KIND validator's job (validation/spectra.py's
        # frozen/explicit-index bounds; validate_selection retired), so
        # this path produces the raw user input; out-of-range indices
        # simply find no matching mode in the engine's later lookup.
        base = [int(i) for i in cfg.es_explicit_indices]

    else:
        # Unknown selector -- caller should have caught this via
        # the catalogue's `choices` metadata + the schema validator.
        # Defensive: produce an empty selection rather than crash;
        # (the kind's science refuses out-of-range explicit indices upstream)
        base = []

    # 2. Drop modes whose ES is already in `prior` (resume / additive
    #    L4 -- spec § 2.5.2).  Preserves existing work, only computes
    #    newly-requested modes.
    if prior is not None:
        already_done = {
            m.index_1based for m in prior.modes
            if m.electronic_structure is not None
        }
        base = [i for i in base if i not in already_done]

    # 3. De-duplicate while preserving order (user-supplied
    #    explicit lists can have repeats).
    seen: set = set()
    out: List[int] = []
    for i in base:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# (validate_selection retired 2026-08-21, C-spectra-config: its caller --
#  the old engine preflight's advisory layer -- died at P3, and the kind's
#  science (validation/spectra.py) carries the selector advisories now.
#  select_modes above stays deliberately: it is the REFERENCE
#  implementation the deck's inlined selector is parity-tested against,
#  tests/spectra/test_selection.py::TestSelectorEquivalence.)

def _passes_freq_window(m: ModeData, cfg: "VibrationConfigView") -> bool:
    """Return True iff the mode's frequency lies in
    ``[freq_min_cm1, freq_max_cm1]`` (either bound = None means no
    constraint on that side; both = None means no filter).

    Imaginary modes (negative frequency) are filtered by the same
    rule -- they're rarely transport-relevant anyway, and the
    user can opt them in by setting freq_min_cm1 to a negative
    value.  Documented in spec § 8.1 implicitly via "frequency"
    being the abs-of-eigenvalue with sign carrying imag info.
    """
    if cfg.freq_min_cm1 is not None and m.frequency_cm1 < cfg.freq_min_cm1:
        return False
    if cfg.freq_max_cm1 is not None and m.frequency_cm1 > cfg.freq_max_cm1:
        return False
    return True


__all__ = ["select_modes"]
