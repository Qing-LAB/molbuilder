"""Mode-selection logic for the Spectra tab's L4 step.

Pure functions, no I/O.  Spec § 8 + § 8.1 + § 2.5.3 codify the
semantics; this module is the executable form.

Public surface:

  * :func:`select_modes(modes, cfg, prior=None) -> List[int]` --
    given the full mode list from L2 + the user's :class:`SpectraConfig`
    + (optionally) the prior :class:`SpectraResults` from a
    previous run, return the 1-based indices of modes that should
    get per-mode displaced-geometry SCFs in this run.

  * :func:`validate_selection(modes, cfg, l3_done, prior=None)
    -> List[Issue]` -- pre-selection preflight; surfaces error /
    warn issues (selector / prior / soft-dep conflicts) before the
    script runs.

The five Model-2 selectors:

    none      -> []  (no L4 work)
    all       -> every mode (respecting the freq filter)
    top_n     -> top-N by raman_activity_a4_amu (respecting the
                 freq filter); REQUIRES L3 complete
    threshold -> modes with raman_activity > cfg.es_threshold
                 (respecting the freq filter); REQUIRES L3 complete
    explicit  -> exactly cfg.es_explicit_indices (frequency
                 filter IGNORED -- the user named specific modes,
                 the window doesn't override)

The freq filter (cfg.freq_min_cm1 / cfg.freq_max_cm1, either
side optionally None) composes with selector by INTERSECTION:
"top_n=10 within [500, 2000] cm⁻¹" means top-10-by-activity
*among* modes in that window.

Resume / non-destructive L4 (spec § 2.5.2): when ``prior`` is
provided, modes that already have ``electronic_structure``
populated in the prior results are excluded from the returned
list -- the engine skips them on this run (their ES data stays
in the file).  This is the "incrementally add modes" workflow.
"""

from __future__ import annotations

from typing import List, Optional

from ..issues import Issue
from ..config.spectra import SpectraConfig
from .results import ModeData, SpectraResults


def select_modes(modes: List[ModeData],
                 cfg: SpectraConfig,
                 prior: Optional[SpectraResults] = None) -> List[int]:
    """Return the 1-based mode indices selected for L4 (per-mode
    displaced-geometry SCFs) under the configured selector + filter.

    Pure: does not consult disk, does not mutate inputs.  The
    engine's render_script feeds this into the script template; the
    test suite asserts each selector / filter / resume combination
    independently.

    Behaviour summary (full spec at § 8 + § 8.1):

      * cfg.es_mode_selection == "none"  -> []
      * cfg.es_mode_selection == "all"   -> every mode (after freq filter)
      * cfg.es_mode_selection == "top_n" -> top cfg.es_top_n by Raman
                                            activity (after freq filter,
                                            requires L3 data)
      * cfg.es_mode_selection == "threshold" -> Raman activity above
                                            cfg.es_threshold (after
                                            freq filter, requires L3)
      * cfg.es_mode_selection == "explicit" -> cfg.es_explicit_indices
                                            (freq filter IGNORED)

      * When ``prior`` is supplied, modes that already have ES data
        in the prior results are removed from the returned list --
        the engine skips them on this run (their ES persists).
    """
    sel = cfg.es_mode_selection

    # 1. Resolve the base set per selector.
    if sel == "none":
        base: List[int] = []

    elif sel == "all":
        base = [m.index_1based for m in modes
                if _passes_freq_window(m, cfg)]

    elif sel == "top_n":
        ranked = _modes_with_activity_sorted(modes)
        # Apply freq window first, then take top-N -- order matters
        # for "top 10 within the window" semantics (spec § 8.1).
        windowed = [m for m in ranked if _passes_freq_window(m, cfg)]
        base = [m.index_1based for m in windowed[:max(0, int(cfg.es_top_n))]]

    elif sel == "threshold":
        base = [m.index_1based for m in modes
                if (_passes_freq_window(m, cfg)
                    and m.raman_activity_a4_amu is not None
                    and m.raman_activity_a4_amu > cfg.es_threshold)]

    elif sel == "explicit":
        # Frequency filter intentionally IGNORED -- the user named
        # specific modes, the window doesn't override (spec § 8.1).
        # Index range validation is the validator's job (see
        # validate_selection) so this path produces the raw user
        # input; out-of-range indices simply find no matching mode
        # in the engine's later lookup.
        base = [int(i) for i in cfg.es_explicit_indices]

    else:
        # Unknown selector -- caller should have caught this via
        # SpectraConfig's `choices` metadata + the schema validator.
        # Defensive: produce an empty selection rather than crash;
        # validate_selection will have already raised an error-Issue.
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


def validate_selection(modes: List[ModeData],
                       cfg: SpectraConfig,
                       l3_done: bool,
                       prior: Optional[SpectraResults] = None) -> List[Issue]:
    """Pre-selection preflight.  Returns Issues that the form's
    validation panel renders before script generation:

      * top_n / threshold selectors WITHOUT L3 complete   -> error
        (soft dep, spec § 2.5.3)
      * explicit indices out of [1, len(modes)]           -> error
      * freq_min > freq_max                                -> error
      * freq window producing 0 matching modes             -> warn
      * top_n > len(modes after window)                    -> warn

    Caller (the engine's preflight()) merges these with the
    engine-specific scientific warnings (grid level, displacement
    amplitude, etc.).
    """
    issues: List[Issue] = []

    # Frequency window sanity: ordering.
    fmin = cfg.freq_min_cm1
    fmax = cfg.freq_max_cm1
    if fmin is not None and fmax is not None and fmin > fmax:
        issues.append(Issue(
            severity="error",
            message=(f"freq_min_cm1 ({fmin}) > freq_max_cm1 ({fmax}); "
                     f"the window is empty by construction"),
            where="config.freq_window",
        ))
        # Don't bother computing the rest of the window-dependent
        # checks if the window itself is broken.
        return issues

    sel = cfg.es_mode_selection

    # Soft dep: top_n / threshold filter modes BY Raman activity, so
    # they require L3 complete (spec § 2.5.3).
    if sel in ("top_n", "threshold") and not l3_done:
        issues.append(Issue(
            severity="error",
            message=(f"es_mode_selection={sel!r} ranks modes by Raman "
                     f"activity, which requires phase_raman=complete "
                     f"in prior results.  Either run with "
                     f"compute_raman=True first, or pick selector "
                     f"'all' / 'explicit' for L4."),
            where="config.es_mode_selection",
        ))

    # Explicit-index range check.
    if sel == "explicit":
        valid_range = range(1, len(modes) + 1)
        bad = [i for i in cfg.es_explicit_indices if i not in valid_range]
        if bad:
            issues.append(Issue(
                severity="error",
                message=(f"es_explicit_indices contains out-of-range "
                         f"values {bad}; valid range is "
                         f"1..{len(modes)} for the current mode set"),
                where="config.es_explicit_indices",
            ))

    # Window-emptiness check (only when the window is meaningful for
    # this selector -- explicit ignores it).
    if sel in ("all", "top_n", "threshold") and (fmin is not None or fmax is not None):
        n_in_window = sum(1 for m in modes if _passes_freq_window(m, cfg))
        if n_in_window == 0:
            issues.append(Issue(
                severity="warn",
                message=(f"frequency window ["
                         f"{fmin if fmin is not None else '-inf'}, "
                         f"{fmax if fmax is not None else '+inf'}] cm⁻¹ "
                         f"contains zero modes; L4 will produce no ES "
                         f"data"),
                where="config.freq_window",
            ))

    # top_n > available modes (after window) -- warn, not error;
    # we silently clamp in select_modes but the user might not
    # realise.
    if sel == "top_n":
        n_in_window = sum(1 for m in modes if _passes_freq_window(m, cfg))
        if cfg.es_top_n > n_in_window > 0:
            issues.append(Issue(
                severity="warn",
                message=(f"es_top_n={cfg.es_top_n} exceeds the number "
                         f"of modes in the frequency window "
                         f"({n_in_window}); will select all {n_in_window}"),
                where="config.es_top_n",
            ))

    return issues


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _passes_freq_window(m: ModeData, cfg: SpectraConfig) -> bool:
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


def _modes_with_activity_sorted(modes: List[ModeData]) -> List[ModeData]:
    """Return modes sorted by Raman activity DESCENDING; modes
    without Raman activity (raman_activity_a4_amu = None) drop
    out entirely (top_n / threshold semantics: you can't rank by
    a quantity you haven't computed).  Ties broken by ascending
    mode index for determinism (spec § 8 table)."""
    with_activity = [m for m in modes if m.raman_activity_a4_amu is not None]
    return sorted(
        with_activity,
        key=lambda m: (-m.raman_activity_a4_amu, m.index_1based),
    )


__all__ = ["select_modes", "validate_selection"]
