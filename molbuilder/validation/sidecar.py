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

from typing import Any, List, Mapping, Optional, Tuple

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
        (single-point energy, no relaxation).

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


def check_electrode_labels_are_frozen(struct: Structure, *,
                                      run: str = "the relaxation") -> List[Issue]:
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
         f"so they must come through {run} unmoved -- and nothing "
         f"holds them.  Add them to \"frozen_atoms\" in /modify before "
         f"running this, or {run} will move the leads and the "
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
        # AND THE ADVICE IS THE KIND'S TOO.  Telling a transport calculation
        # its labels "stay in the sidecar for /transport" is nonsense -- this
        # IS /transport -- and offering `frozen_atoms` is the wrong remedy
        # for a stray label on a junction.  Fixing WHICH labels are named and
        # leaving this sentence would have been half the defect.
        advice = ("The partition it reads is "
                  "L-electrode / R-electrode / bridge / buffer, plus any "
                  "*-electrode name; anything else rides along untouched. "
                  "Rename it to one of those if it was meant to be part of "
                  "the junction.")
    else:
        inert = sorted(name for name, idxs in regions.items()
                       if idxs and name not in consumed)
        what = f"{engine} run"
        advice = ("They stay in the sidecar for /transport but do not shape "
                  "this calculation. If you meant those atoms to be held "
                  "fixed, assign them to \"frozen_atoms\" in /modify.")
    if not inert:
        return []
    return [Issue(
        "warn",
        (f"this structure carries region label(s) {inert}, which the "
         f"{what} does NOT consume. {advice}"),
        "structure.regions",
    )]


def _ev(x) -> str:
    """A measured force in eV/Å, readable at both ends of the range: four
    decimals down to half a milli-eV/Å, an exponent below that.  A
    tolerance prints as the person typed it (``:g``)."""
    x = float(x)
    return f"{x:.4f}" if x >= 5e-4 else f"{x:.1e}"


def _held_sets_differ(struct: Structure, rec: Mapping[str, Any]
                      ) -> Optional[Tuple[int, int]]:
    """``(held then, held now)`` when a relaxation record's held set is not
    this structure's -- compared by the atoms themselves, never by an index,
    because a deck's copy may list them in another order; ``None`` when they
    are the same atoms or the record keeps no set."""
    keys_then = rec.get("held_atom_keys")
    if not isinstance(keys_then, list):
        return None
    lines = struct.geometry_lines()
    keys_now = sorted(lines[i] for i in (struct.frozen_atoms or [])
                      if 0 <= int(i) < len(lines))
    if sorted(keys_then) == keys_now:
        return None
    return len(keys_then), len(keys_now)


def _judged_force(rec: Mapping[str, Any]) -> Optional[float]:
    """The force a relaxation is judged by: the largest on the atoms it
    moved, when the record says; on every atom otherwise."""
    f_free = rec.get("max_force_free_ev_ang")
    return f_free if f_free is not None else rec.get("max_force_ev_ang")


def check_relaxation_record(struct: Structure, *, engine: str,
                            already_relaxed: bool,
                            force_tolerance_ev_ang: Optional[float],
                            level: Optional[Mapping[str, Any]] = None,
                            relaxed_by: Optional[Mapping[str, Any]] = None,
                            relax_stage: str = "relax"
                            ) -> List[Issue]:
    """The structure's own evidence against the person's statement
    (`engines/vibration.md` § 2.2, the record table; `model/parse.md`
    § 5b.1).

    ``info.relaxation`` is what a finished run said about the geometry it
    left; ``info.calculation`` the level of theory its deck stated.  Both
    ride the pair the Results tab exports.  This reads them beside the box
    -- every finding lands on ``config.already_relaxed`` -- and never
    refuses: the record informs the choice, it does not make it.  One
    finding per fact: the record's absence (ticked only), a geometry the
    record does not describe, a different engine or level of theory, the
    largest remaining force against THIS calculation's tolerance, a
    different held set.  Ticked, a disagreement is a warning; unticked,
    the same fact is information, because the ladder relaxes regardless.

    ``relaxed_by`` is the other question, asked at a force-constant stage
    whose ladder relaxed the coordinates itself: that stage's record
    (`engines/vibration.md` § 5.2a, V1.36) -- judged by
    :func:`_ladder_relaxation_findings`, never by the box's advice.
    """
    # THE one remedy for a structure stated relaxed that is not stationary
    # here -- the text the finish and the PySCF deck write too
    # (`vibrational_analysis.nonstationary_remedy`); four wordings of it stood
    # in this function until 2026-09-29 (the K6 review, R11).
    from ..spectra.vibrational_analysis import nonstationary_remedy
    _remedy = nonstationary_remedy(None, engine)
    if relaxed_by is not None:
        return _ladder_relaxation_findings(struct, relaxed_by,
                                           force_tolerance_ev_ang,
                                           stage=relax_stage)
    issues: List[Issue] = []
    where = "config.already_relaxed"
    info = getattr(struct, "info", None) or {}
    rec = info.get("relaxation")
    if not isinstance(rec, dict) or not rec.get("geometry_sha256"):
        if already_relaxed:
            issues.append(Issue(
                severity="info",
                message=(
                    "No relaxation record travels with this structure (its "
                    "metadata has no `relaxation` entry -- a pair exported "
                    "from a finished relaxation on the Results tab carries "
                    "one), so the statement stands on its own; the "
                    "run measures the forces at the starting geometry."),
                where=where))
        return issues
    disagree = "warn" if already_relaxed else "info"
    disagreed = False           # any fact below that says "not stationary here"
    rec_engine = rec.get("engine")

    if rec["geometry_sha256"] != struct.geometry_fingerprint():
        issues.append(Issue(
            severity=disagree,
            message=(
                f"This structure carries a relaxation record "
                f"({rec_engine or 'engine not recorded'}, "
                f"{rec.get('n_steps', '?')} geometry step(s)), but for a "
                f"different geometry -- another frame of that run, or edited "
                f"since -- so it does not vouch for these coordinates."
                + ("  The statement that the structure is relaxed stands on "
                   "its own; the run measures it." if already_relaxed
                   else "")),
            where=where))
        return issues
    # -- the level of theory: the engine, then the recorded contract ------
    if not rec_engine:
        issues.append(Issue(
            severity="info", where=where,
            message=("This structure's relaxation record does not say which "
                     "engine relaxed it, so the level of theory cannot be "
                     "compared with this calculation's.")))
    elif str(rec_engine).lower() != str(engine).lower():
        disagreed = True
        issues.append(Issue(
            severity=disagree,
            message=(
                f"This structure was relaxed on {rec_engine}; this is a "
                f"{engine} calculation -- a different level of theory, so "
                f"the geometry is not a stationary point here."
                + ("  " + _remedy if already_relaxed else
                   "  The ladder relaxes it at this level first.")),
            where=where))
    else:
        cal = info.get("calculation")
        recorded = (cal.get("contract") if isinstance(cal, dict) else None) or {}
        differing = []
        for key, mine in (level or {}).items():
            theirs = recorded.get(key)
            if theirs is None or mine is None:
                continue
            same = (str(theirs).strip().lower() == str(mine).strip().lower()
                    if isinstance(theirs, str) or isinstance(mine, str)
                    else abs(float(theirs) - float(mine)) <= 1e-9 * max(
                        1.0, abs(float(mine))))
            if not same:
                differing.append(f"{key}: relaxed with {theirs}, this run {mine}")
        if differing:
            disagreed = True
            issues.append(Issue(
                severity=disagree,
                message=(
                    "This structure was relaxed at a different level of "
                    "theory than this calculation runs (" + "; ".join(differing)
                    + "), so its geometry is not a stationary point here."
                    + ("  " + _remedy if already_relaxed else
                       "  The ladder relaxes it at this level first.")),
                where=where))
    # -- the held set ------------------------------------------------------
    held = _held_sets_differ(struct, rec)
    if held is not None:
        disagreed = True
        issues.append(Issue(
            severity=disagree, where=where,
            message=(f"The relaxation held {held[0]} atom(s); this "
                     f"calculation holds {held[1]}, and they are not "
                     f"the same atoms.  The free atoms are not the same "
                     f"set, so the geometry is not stationary for this "
                     f"calculation's free atoms."
                     + ("  " + _remedy if already_relaxed
                        else "  The ladder relaxes this set first."))))
    # -- the largest remaining force against THIS calculation's tolerance --
    judged = _judged_force(rec)
    rec_tol = rec.get("force_tolerance_ev_ang")
    who = (f"relaxed on {rec_engine or 'an engine the record does not name'}"
           + (f" to {float(rec_tol):g} eV/Å" if rec_tol is not None else "")
           + f" in {rec.get('n_steps', '?')} geometry step(s)")
    if judged is None:
        issues.append(Issue(severity="info", where=where,
                            message=f"This structure was {who}; the record "
                                    f"carries no final force to judge."))
    elif force_tolerance_ev_ang is None:
        issues.append(Issue(severity="info", where=where,
                            message=f"This structure was {who}; the largest "
                                    f"force left on the atoms it moved is "
                                    f"{_ev(judged)} eV/Å."))
    elif float(judged) <= float(force_tolerance_ev_ang):
        issues.append(Issue(
            severity="info", where=where,
            message=(f"This structure was {who}; the largest force left on "
                     f"the atoms it moved is {_ev(judged)} eV/Å, within "
                     f"this calculation's tolerance of "
                     f"{float(force_tolerance_ev_ang):g} eV/Å."
                     + ("" if (already_relaxed or disagreed) else
                        "  The record already meets this calculation's "
                        "criterion: the box may be ticked and the relaxation "
                        "skipped (a ladder already holding a `relax` stage "
                        "runs it regardless)."))))
    else:
        issues.append(Issue(
            severity=disagree, where=where,
            message=(f"This structure was {who}, but the largest force left "
                     f"on the atoms it moved is {_ev(judged)} eV/Å, "
                     f"above this calculation's tolerance of "
                     f"{float(force_tolerance_ev_ang):g} eV/Å."
                     + ("  The frequencies will be off unless it is relaxed "
                        "further.  " + _remedy if already_relaxed else
                        "  The ladder's relaxation tightens it."))))
    return issues


def _ladder_relaxation_findings(struct: Structure, rec: Mapping[str, Any],
                                force_tolerance_ev_ang: Optional[float], *,
                                stage: str) -> List[Issue]:
    """What a force-constant stage is told about the geometry its ladder's
    `relax` stage left (`engines/vibration.md` § 5.2a; plan V1.36, the
    user's word 2026-09-29).

    The fact at this stage is that stage's OUTCOME -- how far it got, against
    this calculation's tolerance -- and, when it stopped short, the one
    remedy: continue it.  Not the box's describe-time advice ("the ladder
    relaxes it first", "the box may be ticked", "untick the box"), which
    was written for the input and is moot once the ladder has relaxed; not
    the level of theory, which is this calculation's own by construction;
    and not the geometry's fingerprint, since the coordinates are that
    record's last frame.  The held set IS compared: a set changed between
    the stages leaves free atoms the relaxation never balanced.
    """
    issues: List[Issue] = []
    where = "config.already_relaxed"
    # THE one remedy text, the finish's and the PySCF deck's too
    # (`engines/vibration.md` § 5.5).
    from ..spectra.vibrational_analysis import nonstationary_remedy
    remedy = "  " + nonstationary_remedy(stage, "siesta")
    held = _held_sets_differ(struct, rec)
    if held is not None:
        issues.append(Issue(
            severity="warn", where=where,
            message=(f"The `{stage}` stage held {held[0]} atom(s); this "
                     f"stage holds {held[1]}, and they are not the same "
                     f"atoms, so the geometry is not stationary for this "
                     f"stage's free atoms." + remedy)))
    judged = _judged_force(rec)
    steps = rec.get("n_steps", "?")
    if judged is None:
        issues.append(Issue(
            severity="info", where=where,
            message=(f"The force constants are taken at the geometry the "
                     f"`{stage}` stage left ({steps} geometry step(s)); its "
                     f"record carries no final force to judge.")))
    elif force_tolerance_ev_ang is None:
        issues.append(Issue(
            severity="info", where=where,
            message=(f"The force constants are taken at the geometry the "
                     f"`{stage}` stage left: a largest force of "
                     f"{_ev(judged)} eV/Å on the atoms it moved, after "
                     f"{steps} geometry step(s).")))
    elif float(judged) <= float(force_tolerance_ev_ang):
        issues.append(Issue(
            severity="info", where=where,
            message=(f"The force constants are taken at the geometry the "
                     f"`{stage}` stage left: a largest force of "
                     f"{_ev(judged)} eV/Å on the atoms it moved, after "
                     f"{steps} geometry step(s), within this calculation's "
                     f"tolerance of {float(force_tolerance_ev_ang):g} "
                     f"eV/Å.")))
    else:
        issues.append(Issue(
            severity="warn", where=where,
            message=(f"The `{stage}` stage stopped with a largest force of "
                     f"{_ev(judged)} eV/Å on the atoms it moved, after "
                     f"{steps} geometry step(s) -- above this "
                     f"calculation's tolerance of "
                     f"{float(force_tolerance_ev_ang):g} eV/Å, so the force "
                     f"constants would be taken off a stationary point and "
                     f"the frequencies will be off, the low ones most."
                     + remedy)))
    return issues
