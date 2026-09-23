"""Sidecar-aware validators (frozen-atoms / region labels).

Per docs/model/structure-molstruct.md the user attaches metadata to
a structure via a ``.molstruct.json`` sidecar: regions, frozen-atom
indices, generator-input echo.  Engines that don't consume one of
these labels MUST surface an INFO Issue so the user can see the
absorption was noticed but not silently dropped.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/science/validation.md  Function body +
signature are identical to the pre-split version.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def _check_frozen_atoms_consumed(struct: Structure, *,
                                   engine: str,
                                   honored: bool,
                                   reason_when_dropped: str = "",
                                   ) -> List[Issue]:
    """Three-stage contract: warn when ``Structure.frozen_atoms`` is
    populated (the user / sidecar asked SOMETHING be held fixed) but
    the engine emission path is going to silently drop the constraint.

    Two callsites:
      * SIESTA: ``honored`` is False when ``cfg.relax_type == 'none'``
        (no MD block emitted, so Geometry.Constraints does nothing).
      * PySCF: ``honored`` is False when ``cfg.optimize == False``
        (single-point energy, no relaxation) OR
        ``cfg.optimizer != 'geometric'`` (the only PySCF optimizer
        with constraint support).

    When honored is True we emit an INFO-severity Issue so the user
    sees an explicit "N atoms held fixed during relaxation" line in
    the preflight panel, not just a silent emission.
    """
    n = len(getattr(struct, "frozen_atoms", []) or [])
    if n == 0:
        return []
    if not honored:
        return [Issue(
            "warn",
            (f"Structure has {n} frozen atom(s) from /modify "
             f"(struct.frozen_atoms), but {engine} won't honor them: "
             f"{reason_when_dropped}.  Either change the config to a "
             f"mode that supports constraints, or clear the frozen "
             f"atoms in /modify if you want a free relaxation."),
            "config.frozen_atoms",
        )]
    return [Issue(
        "info",
        (f"{n} atom(s) held fixed during {engine} relaxation "
         f"(from struct.frozen_atoms / /modify sidecar)."),
        "config.frozen_atoms",
    )]


def check_electrode_labels_are_frozen(struct: Structure) -> List[Issue]:
    """**The first validation of a junction** — are the labeled electrodes
    the fixed atoms?

    `engines/transport.md` § 4 and the one-structure design: a junction is
    ONE file carrying the frozen leads at both ends and the relaxed bridge
    between them, and transport takes the lead atoms **out of that file** by
    their label. The leads must come through the relaxation untouched, or the
    self-energies attach to a geometry that is not the bulk they claim to be.

    **There is already a gate for that, and it is not this one.**
    `transport.wizard.extract_electrode_model` refuses a lead whose atoms are
    not frozen, and refuses one that moved when the caller hands it the
    geometry the relaxation started from. Those refusals are correct and they
    are also **too late**: they run when transport composes, which is after the
    relaxation has been paid for. Label the leads, forget to freeze them, and
    nothing objects until a metal junction's relaxation has already run and
    must be thrown away.

    So this asks the same question one step earlier, where it is cheap — and
    where it is still ACTIONABLE, because the run has not happened yet. That
    is the whole of what this adds; the compose-time refusal is what makes a
    bad junction impossible, and this is what makes it avoidable.

    **A warning, not a refusal**, and the line is worth stating. A structure
    carrying electrode labels is heading for transport, but it has not
    committed: a person may deliberately relax the whole junction once before
    freezing the leads for the run that counts. Refusing here would block that.
    The refusal belongs at compose, where transport IS the intent, and it is
    already there.
    """
    from ..config.transport import is_electrode_label
    from ..structure import FROZEN_LABEL

    regions = getattr(struct, "regions", None) or {}
    lead_idx = {i for name, idxs in regions.items()
                if is_electrode_label(name) for i in (idxs or ())}
    if not lead_idx:
        return []
    frozen = set(getattr(struct, "frozen_atoms", None) or ())
    loose = sorted(lead_idx - frozen)
    if not loose:
        return []
    els = getattr(struct, "elements", ())
    shown = ", ".join(f"{i} ({els[i]})" if i < len(els) else str(i)
                      for i in loose[:6])
    more = f" and {len(loose) - 6} more" if len(loose) > 6 else ""
    return [Issue(
        "warn",
        (f"{len(loose)} atom(s) carry an electrode label but are NOT frozen: "
         f"{shown}{more}.  A transport calculation takes the lead atoms out "
         f"of this structure by that label and treats them as pristine bulk, "
         f"so they must come through the relaxation unmoved -- and nothing "
         f"holds them.  Add them to \"frozen_atoms\" in /modify before "
         f"running this, or the relaxation will move the leads and the "
         f"junction cannot be composed afterwards (the composer refuses it, "
         f"by which point this run has been paid for)."),
        "structure.electrode_frozen",
    )]


def check_unconsumed_region_labels(struct: Structure, *, engine: str,
                                   calculation: str = "") -> List[Issue]:
    """Pattern B, re-homed (validation.md § 5; C-shared 2026-08-21): every
    region label this calculation does NOT consume is named explicitly.

    The /modify selection panel writes ``regions`` for transport workflows
    (L-electrode, bridge, ...); an optimization deck reads none of them,
    and silence would let a user believe their labels shaped the run.  The
    reserved frozen label is EXCLUDED: both engines consume it (SIESTA's
    ``Geometry.Constraints``, PySCF's geomeTRIC ``$freeze``), so warning
    about it would be the same false alarm E-M7.1 fixed on the vibration
    route.  This ran in two web endpoints until they were deleted; living
    HERE puts it on every deck route through the one settings gate.

    **WHAT IS CONSUMED DEPENDS ON THE KIND, and asking only the ENGINE got
    it exactly backwards for transport.**  This took ``engine`` alone, so
    every transport deck's ``.validation.txt`` said its
    ``L-electrode``/``bridge``/``R-electrode`` labels *"do NOT consume ...
    do not shape this calculation"* -- about the partition the entire
    five-rung ladder is built from (`engines/transport.md` § 4).  The
    warning that exists to stop a person believing their labels mattered
    was telling them the opposite of the truth.  `engines/transport.md`
    § 3.6a recorded it as a known wrong warning on 2026-09-16; the kind was
    already being passed to every validator (`validation/__init__` sets
    ``engine_kw["calculation"]``) and this function simply never asked.

    For transport the consumed set is `sort.PARTITION_LABELS` -- asked of
    the module that owns the partition, never re-listed here -- plus any
    ``*-electrode`` name, since the suffix convention is what makes
    ``tip-electrode`` a lead without a code change (§ 4).  Anything else is
    genuinely unread and is still named, which is § 4's own rule: *"a label
    this engine does not consume is WARNED about, never dropped in
    silence."*
    """
    from ..structure import FROZEN_LABEL
    regions = getattr(struct, "regions", None) or {}
    consumed = {FROZEN_LABEL}
    if calculation == "transport":
        from ..config.transport import is_electrode_label
        from ..transport.sort import PARTITION_LABELS
        consumed |= set(PARTITION_LABELS)
        inert = sorted(name for name, idxs in regions.items()
                       if idxs and name not in consumed
                       and not is_electrode_label(name))
        what = "transport ladder"
    else:
        inert = sorted(name for name, idxs in regions.items()
                       if idxs and name not in consumed)
        what = f"{engine} run"
    if not inert:
        return []
    return [Issue(
        "warn",
        (f"this structure carries region label(s) {inert}, which the "
         f"{what} does NOT consume -- they stay in the sidecar "
         f"for /transport but do not shape this calculation. "
         f"If you meant those atoms to be held fixed, assign them to "
         f"\"frozen_atoms\" in /modify."),
        "structure.regions",
    )]
