"""The frame-contract gate — structure-periodicity.md § 6.1 / § 6.2.

MODULE  periodicity_gate (L1; imports structure only)
ROLE    the ONE place periodicity state is defaulted and validated
USED-BY StructureCodec (load/save gate), web/blueprints/build.py (the
        unified periodicity door), tests/test_periodicity_gate.py

Contract (§ 6.1, decided 2026-07-29): the .xyz/.molstruct.json pair is the
only truth; resolved values are views and are never written back; NOTHING
here rewrites stored state.  The § 6.1 state table governs STORED state
(load/save) -- it says, for each state the pair can hold, whether it is legal
and what the user is told; ``apply_edit`` below governs LIVE edits per the
§ 6.2 v3 regime model:

  DERIVED regime: {structure size, vacuum, axis_kind} => {cell, origin}
  are computed views.  Editing vacuum / axis_kind RESETS to this regime
  (explicit cell + origin cleared) -- the box boundary moves, and the
  caller must warn the user BEFORE committing.

  MANUAL regime: an explicit cell demotes vacuum to reference-only; an
  explicit origin overrides the vacuum-derived corner ("origin first,
  then vacuum").  Upstream edits never silently contradict downstream
  state -- they reset it, loudly.

  Coordinate rewrites (calibrate) are NOT a periodicity edit: they live
  with the Modify ops (/api/modify/calibrate, molbuilder.modify.
  calibrate_to_cell).  Emission translates to the engine frame
  implicitly; nothing on the Cell page moves atoms.

State table (§ 6.1, stored state; right-handed cells only, det > 0).  Every
row ends in "legal" -- the gate reports, it does not repair:

  | stored state                | contained? | action                       |
  |-----------------------------|-----------|-------------------------------|
  | no cell, no origin          |     —     | derived; vacuum authoritative |
  | explicit cell, no origin    |    yes    | legal (imported crystal): the |
  |                             |           | corner IS the world origin    |
  | explicit cell, no origin    |    no     | legal: the corner is DERIVED  |
  |                             |           | (wrapping/centred) + info     |
  | explicit cell + origin      |    yes    | legal, user-owned, untouched  |
  | explicit cell + origin      |    no     | user-owned in BOTH halves:    |
  |                             |           | warn, never auto-fix          |

The corner respects the per-direction vacuum -- ``bbox_min − vacuum`` on
isolated axes, ``bbox_min`` on transport, ``0`` on periodic -- and the rule
lives ONCE, on ``Structure`` (``expected_cell_corner`` /
``cell_contains_atoms``).  This module used to re-export both under second
names; they were one-line delegates whose only callers were tests, so a
reader had two names for one rule and no way to tell which was authoritative.
Removed 2026-08-03; ask ``Structure``.

NOTHING IS MATERIALISED (decided 2026-07-29).  "No explicit origin" means
"derive the corner" — ``Structure.resolve_cell_origin`` answers it at every
seam — so this module never writes a resolved corner into the truth.  The
earlier behaviour DID (it healed ``cell_origin = expected_corner`` on load),
and that disagreed with the reset-origin op, which left the same state
alone: one state, two answers, and a save+reload silently changed what the
user had been shown.  The rule now lives once, on ``Structure``
(``expected_cell_corner`` / ``cell_contains_atoms``), and this module
delegates to it.

Notices (the machine-readable half of the contract).  Every entry is
``{"severity", "message", "where", "about"}`` -- FOUR keys.  It was
``level`` until 2026-09-10, which is why `task-handover.js` carried a
comment about an error arm that could never fire: it read ``severity``,
the only spelling `issues_to_json` ever used.  One name now.
``where`` is the stable id ``Issue`` carries, because the conditions now come
from ``cell.check`` and a finding has to be identifiable without reading its
prose; ``about`` is the subject, which is what decides where it is shown.
Callers surface notices; they never parse the message text.

A notice that reports state the gate corrected would have carried a marker;
NOTHING DOES, and nothing should: clause 1 forbids writing a resolved value
back, so there is no correction to mark.

Errors vs notices.  ``ValueError`` (mapped to HTTP 400 by the door) is raised
for a malformed payload, and for any error-severity finding ``cell.check``
returns -- this module no longer decides WHICH states those are, it asks
(``_refuse_on_error``).  The list used to be repeated here and enforced by
three hand-written checks in this file; they asked the same questions in
their own words, at their own thresholds, and are gone (cell-plan.md § 6a).
Everything else — including a box that does not contain its atoms under a
user-owned origin — is a notice, never an exception: the gate reports, the
user decides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .issues import Issue
from .structure import Structure

# Containment tolerance (Angstrom-scale in fractional projections).  Loose
# enough to forgive round-tripped floats, tight enough that "half the
# molecule outside the box" can never pass.
_EPS = 1e-6

#: The four ops the unified door accepts (§ 6.2 v3).  ``cell_origin`` with a
#: ``null`` payload is the "reset origin to default" button.
OPS = ("vacuum", "axis_kind", "cell", "cell_origin", "block")

#: The keys ``block`` accepts, which are exactly § 6.2's cell -- the vectors,
#: the anchor, how each axis is treated, how much vacuum an isolated axis gets.
#: Named once so the door and its refusal cannot disagree about the set.
BLOCK_KEYS = ("cell", "cell_origin", "axis_kind", "vacuum")


def _notice(level: str, message: str, where: str = "cell.edit") -> Dict[str, str]:
    """One notice: how loud, what it says, and WHAT IT IS ABOUT.

    ``about`` is the subject, and it is what decides where the message is shown
    -- a message about the box belongs beside the box, on the page where a user
    would go to change it.  Everything this module produces is about the cell,
    because this module IS the cell's gate; a notice from somewhere else says
    its own subject.

    It used to be missing, and the display worked out where to put a message
    from where it CAME FROM instead -- a load, or a cell edit.  That put a
    warning about an unusable box above the atom list whenever it arrived with
    a file, which is not where anybody can act on it.

    ``where`` IS THE STABLE ID (added 2026-08-03), the same one ``Issue``
    carries, because these entries now come from ``cell.check`` and a finding
    must be identifiable without reading its prose.  Its absence is why four
    tests matched on message TEXT -- pinning the wording of a sentence this
    module's own header says callers must never parse -- so a reworded message
    broke tests while a deleted check would not have.

    CONDITIONS take the checker's id (``cell.no_volume``, ``cell.atoms_outside``
    …).  RECEIPTS -- what an edit just did -- default to ``cell.edit``: they are
    not findings, they have no verdict, and nothing should key on an individual
    one.
    """
    # BUILT THROUGH `issues.Issue`, and `about` is DERIVED.
    #
    # This assembled `{"severity", "message", "where", "about"}` by hand until
    # 2026-09-09.  Two problems, both silent: `level` was never checked
    # against anything, and `about` stored a fact `where` already carried --
    # it was `where.split(".")[0]` in every case the tree produces
    # (`cell.no_volume` -> `cell`, `append.merge` -> `append`,
    # `slab.seam_ok` -> `slab`), so the pair could drift with nothing to say
    # so.  Constructing an `Issue` first borrows its validated `severity`, so
    # a typo'd level raises HERE instead of reaching the browser as a notice
    # no stylesheet matches.
    #
    # The wire key stays `level`, not `severity`: three browser modules read
    # `.level`, and the node harness that would check a rename SKIPS in this
    # environment (`#79`), so the wire rename waits on that decision rather
    # than going in unverified.
    return _wire(Issue(level, message, where))


def _refuse_on_error(s: Structure) -> None:
    """Raise on the first error the ONE checker finds in ``s``'s box.

    The edit door's half of § 8.2: the Cell page refuses the value you type,
    because its whole subject is that value and a good one entered straight
    after is accepted.  A structure that ARRIVED holding the same state is
    reported instead (``ok_structure_response``) -- same checker, two verdicts.

    This is what ``_require_right_handed`` and ``_too_small_axes`` used to do
    inline.  They asked the same two questions ``cell.check`` asks, in their own
    words and at their own thresholds, and a third copy lived in the emitter.
    """
    from .cell import resolve_and_check
    _rc, issues = resolve_and_check(s)
    for i in issues:
        if i.severity == "error":
            raise ValueError(i.message)


def notices_for_report(issues) -> List[Dict[str, str]]:
    """Findings as wire notices, for a door that REPORTS rather than refuses.

    THE ONE SERIALIZER.  Every notice on the wire is made here, from an
    ``Issue`` that ``cell.check`` produced -- so the id, the wording and the
    subject travel together and no door invents its own shape.
    ``ok_structure_response`` used to catch the gate's ``ValueError`` and
    rebuild a notice by hand from ``str(exc)``, which silently dropped the id
    and left the front end with a message it could not identify.

    **Error becomes warn here, deliberately.**  § 8.2: a request that is
    *loading or modifying* reports a bad box, with the structure, so the user
    can see the problem and fix it; only a request that is *generating
    something you would run* refuses.  The severity is not being softened --
    ``report(validate(...))`` still raises on the same finding at the emit
    door.  What changes is who is being answered.
    """
    return [
        _notice("warn" if i.severity == "error" else i.severity,
                i.message, i.where)
        for i in issues
    ]


def _wire(i: "Issue") -> Dict[str, str]:
    """One `Issue` as the wire notice the browser reads.

    THE ONLY PLACE the wire shape is written.  `about` is DERIVED from
    `where` -- it was a stored fourth key until 2026-09-09 and was
    `where.split(".")[0]` in every case, so the two could drift and nothing
    would say so.  `molview/ui.js` filters notices by it
    (`n.about === subject`), which is why it is emitted rather than dropped.

    The key is still `level`, not `severity`: three browser modules read
    `.level` and the node harness that would check a rename SKIPS here
    (`#79`), so the wire rename waits on that decision rather than going in
    unverified.
    """
    return {"severity": i.severity, "message": i.message,
            "where": i.where, "about": i.where.split(".")[0]}





def validate_periodicity(struct: Structure) -> Tuple[Structure, List[dict]]:
    """Check STORED periodicity against the § 6.1 table and REPORT.  Returns
    ``(struct, notices)``.

    THE ANSWER DOES NOT DEPEND ON HOW THE STATE ARRIVED.  There was a
    ``live_edit`` flag here, documented as selecting the explicit-origin
    manual-edit row -- and never read by a line of this function.  It was true
    once: the flag chose between healing a stored origin and accepting a typed
    one.  Healing left on 2026-07-29 and both branches became the same branch,
    so the flag went on being passed, and read, and believed for three days
    while doing nothing.  Removed 2026-08-02.  Row 5 is one row: a manual origin
    is user-owned, warned about, and never auto-fixed -- from disk and from the
    Cell page alike, which is the property the round-trip depends on.

    IT CORRECTS NOTHING, and must not.  Clause 1: `cell` / `cell_origin` hold
    only what the user set, and every resolved value is a VIEW that is never
    written back.  The struct comes out as it went in.

    It was called ``validate_and_heal`` until 2026-08-01 and that name outlived
    the behaviour: healing was removed on 2026-07-29 when materialising a
    resolved corner was found to corrupt a saved pair (the hemeC case named in
    the module header).  The name then had readers — and the author of this
    docstring — looking for a correction step that clause 1 forbids, and
    worrying about a marker (``kind: "heal"``) that two comments described and
    no code produced.  A function is named for what it does.

    The struct is still returned, and callers still adopt it, so that this stays
    the one seam every structure passes through rather than an optional check.

    Raises ``ValueError`` (the door maps it to HTTP 400) for a left-handed
    cell (``det <= 0``) or one no origin could make contain the structure.

    ONE LINE SINCE 2026-08-03 (cell-plan.md § 6a).  This function used to walk
    the § 6.1 state table itself -- five branches, two of which raised and three
    of which built notices by hand -- while ``validation/`` judged the same box
    separately in its own vocabulary.  It now does what every other consumer
    does: ``cell.resolve_and_check(struct)``, once, and hands the findings on.

    The row-by-row reasoning did not disappear; it moved into ``cell.check``
    where it is stated once and reaches BOTH surfaces.  What did disappear is
    this function's ability to disagree with the validator about the same box.
    """
    from .cell import resolve_and_check

    _rc, issues = resolve_and_check(struct)
    # ERRORS STILL RAISE HERE, because this door's callers are mid-edit and a
    # state that cannot be represented has to become an HTTP 400 rather than a
    # structure nobody can act on (§ 8.2's "the Cell page refuses the value you
    # type").  The LOADING doors call ``cell.check`` directly and report the
    # same Issues instead -- one checker, two verdicts, exactly as the contract
    # says.
    fatal = [i for i in issues if i.severity == "error"]
    if fatal:
        raise ValueError(fatal[0].message)
    return struct, [_notice(i.severity, i.message, i.where) for i in issues]


def _reset_to_derived(s: Structure, what: str,
                      notices: List[dict]) -> None:
    """Shared § 6.2 v3 upstream-edit semantics: editing {vacuum,
    axis_kind} moves the box back to the DERIVED regime — explicit cell +
    origin cleared, boundary recomputed from the structure + vacuum.

    Refuses when the derived box would be DEGENERATE.  On an ISOLATED axis with
    no vacuum set that can no longer happen — the § 6.1 default gives a flat or
    linear molecule a real 3 Å-per-side gap (``Structure.effective_vacuum``), so
    a planar structure resets cleanly now instead of being told to set a vacuum
    first.  With a vacuum of 0 explicitly SET it can still happen, and must: the
    typed value is never overridden.  A TRANSPORT axis still refuses: its length
    is the captured device length, vacuum does not apply there, and a zero-extent
    bbox cannot reproduce it.

    Says nothing about the default gap itself.  ``validate_periodicity`` runs on
    the RESULT of every edit (build.py::api_structure_periodicity) and reports it
    there, for every hand-over rather than only for an edit — a second producer
    here just delivered the same sentence twice."""
    if s.n_atoms:
        ext = s.positions.max(axis=0) - s.positions.min(axis=0)
        kinds = s.axis_kind or ("isolated",) * 3
        eff = s.effective_vacuum()
        for i, kind in enumerate(kinds):
            pad = 2.0 * float(eff[i]) if kind == "isolated" else 0.0
            if float(ext[i]) + pad < 1e-6:
                raise ValueError(
                    f"cannot reset to the derived box: axis {i} would be "
                    f"degenerate (a '{kind}' axis whose structure extent is ~0, "
                    f"and vacuum does not apply to it). Keep an explicit cell "
                    f"for that direction.")
    had_manual = s.cell is not None or s.cell_origin is not None
    s.cell = None
    s.cell_origin = None
    if had_manual:
        notices.append(_notice(
            "warn",
            f"{what} changed → the box returned to the DERIVED regime: the "
            "explicit cell and origin were reset, and the boundary is now "
            "recomputed from the structure size + per-direction vacuum "
            "(molecule centred on isolated axes)."))


def _apply_block(s: Structure, payload: Any,
                 notices: List[dict]) -> Structure:
    """Set § 6.2's whole cell at once -- the ``block`` op.

    THE CELL IS ONE FACT THAT TRAVELS TOGETHER (molview.md § 6.2), and the four
    single-field ops could only be spent one request at a time.  A panel that
    wanted to change two of them had to send two, which is not atomic: the
    second can be refused after the first has landed, leaving a box the user
    never asked for and a form that no longer describes it.  Worse, some pairs
    are unreachable in EITHER order -- turning an axis periodic needs the
    explicit cell already stored, and clearing that cell needs the axis already
    non-periodic -- so "become a periodic crystal" and "go back to a derived
    box" were two-step journeys through a state the gate refuses.

    So this one sets all four and checks ONCE, at the end.  The intermediate
    states never exist, which is why the interlocks that make the step-at-a-time
    ops refuse are simply not in the way here -- the same rules are still
    enforced, on the result, below.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"op 'block' takes the whole cell as an object with "
            f"{list(BLOCK_KEYS)} -- a partial block is what this op exists to "
            f"replace")
    unknown = sorted(set(payload) - set(BLOCK_KEYS))
    if unknown:
        raise ValueError(
            f"op 'block' got unknown key(s) {unknown}; expected "
            f"{list(BLOCK_KEYS)}")

    kinds = tuple(str(k) for k in (payload.get("axis_kind") or []))
    if len(kinds) != 3 or any(
            k not in ("isolated", "transport", "periodic") for k in kinds):
        raise ValueError("axis_kind must be 3 of isolated|transport|periodic")

    raw_cell = payload.get("cell")
    if raw_cell is None:
        cell = None
    else:
        try:
            cell = np.asarray(raw_cell, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            raise ValueError("cell must be a 3×3 matrix of numbers (Å)") from None

    # THE ONE PAIRING RULE, stated on the whole block rather than on the order
    # the fields arrived in: a periodic direction is a lattice, and there is
    # nothing to derive one from.
    if "periodic" in kinds and cell is None:
        raise ValueError(
            "a periodic axis needs an explicit cell — a derived bounding box "
            "is not a lattice (§ 4)")

    raw_origin = payload.get("cell_origin")
    if raw_origin is None:
        origin = None
    else:
        try:
            origin = [float(x) for x in raw_origin]
        except (TypeError, ValueError):
            raise ValueError("cell_origin must be 3 numbers (Å), or null "
                             "to derive it") from None
        if len(origin) != 3:
            raise ValueError("cell_origin must be 3 numbers (Å), or null "
                             "to derive it")
        if cell is None:
            raise ValueError(
                "cell_origin is only meaningful with an explicit cell — the "
                "derived box computes its own corner (§ 3c)")

    raw_vac = payload.get("vacuum")
    if raw_vac is None:
        vac = None
    else:
        try:
            vac = [float(x) for x in raw_vac]
        except (TypeError, ValueError):
            raise ValueError("vacuum must be 3 non-negative floats (Å), or "
                             "null to clear it") from None
        if len(vac) != 3 or any(x < 0 for x in vac):
            raise ValueError("vacuum must be 3 non-negative floats (Å), or "
                             "null to clear it")

    s.cell = cell
    s.cell_origin = None if cell is None else (
        None if origin is None else tuple(origin))
    s.axis_kind = kinds
    s.vacuum = None if vac is None else tuple(vac)
    s.__post_init__()
    # ONE CHECK, ON THE RESULT.  Every refusal the field-at-a-time ops raise
    # about a bad box is this same checker; asking it here asks it of the box
    # the user actually described.
    _refuse_on_error(s)

    # A RECEIPT: which regime the box is now in, because that is the fact the
    # panel's own switch is about and the one thing a user cannot read off the
    # numbers.
    if cell is None:
        notices.append(_notice(
            "info",
            "the box is derived: the structure's extent plus the vacuum on "
            "each side, centred on the structure. Vacuum is authoritative."))
    else:
        # THE DERIVED CORNER IS NOT ALWAYS A NUMBER.  `resolve_cell_origin`
        # answers None to mean THE WORLD ORIGIN, no shift -- the box already
        # sits where the atoms are (structure.py § 3c) -- and
        # `np.asarray(None, dtype=float)` is `nan`, so a sentence built without
        # this branch reads "a derived corner at nan".
        corner = None if origin is not None else s.resolve_cell_origin()
        if origin is not None:
            where = "the origin you set"
        elif corner is None:
            where = "the world origin"
        else:
            where = ("a derived corner at "
                     + str(np.round(np.asarray(corner, dtype=float), 4).tolist()))
        notices.append(_notice(
            "info",
            f"the box is the explicit cell, anchored at {where}. Vacuum "
            f"values are reference-only from now on (§ 6.1)."))
    return s


def apply_edit(struct: Structure, op: str,
               payload: Any) -> Tuple[Structure, List[dict]]:
    """The § 6.2 v3 unified door: one entry point for the Cell-page edits.
    Returns (new struct, notices).  Raises ``ValueError`` on contract
    violations (the caller maps to HTTP 400).  Coordinates are NEVER
    touched here (calibrate is a Modify op, not a periodicity edit)."""
    if op not in OPS:
        raise ValueError(f"unknown periodicity op {op!r}; one of {OPS}")
    if struct.n_atoms == 0:
        raise ValueError(
            "no atoms loaded — load a structure before editing its box")
    s = struct.copy()
    notices: List[dict] = []
    kinds = s.axis_kind or ("isolated",) * 3

    if op == "block":
        return _apply_block(s, payload, notices), notices

    if op == "vacuum":
        # ``null`` CLEARS -- the third state the model gained on 2026-08-03.
        # molview.md § 9.5 has always documented this payload as "null
        # clears"; until vacuum became Optional there was nothing to clear
        # TO, and this branch raised "must be 3 non-negative floats".
        if payload is None:
            v = None
        else:
            try:
                v = [float(x) for x in payload]
            except (TypeError, ValueError):
                raise ValueError(
                    "vacuum must be 3 non-negative floats (Å), or null to "
                    "clear it") from None
            if len(v) != 3 or any(x < 0 for x in v):
                raise ValueError(
                    "vacuum must be 3 non-negative floats (Å), or null to "
                    "clear it")
        if "periodic" in kinds:
            raise ValueError(
                "cannot re-derive the box while an axis is periodic (a "
                "bounding box is not a lattice) — make the axis isolated "
                "first, or edit the cell explicitly (§ 6.2)")
        s.vacuum = None if v is None else tuple(v)
        if v is None:
            # A RECEIPT, not an explanation: what the default now is, and what
            # it leaves between images, is a CONDITION of the resulting box,
            # and ``cell.check`` says it on this same response (the
            # door re-validates the result).  Saying it twice is what
            # ``_floor_notices`` used to do.
            notices.append(_notice("info", "vacuum cleared."))
        _reset_to_derived(s, "vacuum", notices)
        s.__post_init__()
        return s, notices

    if op == "axis_kind":
        new_kinds = tuple(str(k) for k in (payload or []))
        if len(new_kinds) != 3 or any(
                k not in ("isolated", "transport", "periodic")
                for k in new_kinds):
            raise ValueError(
                "axis_kind must be 3 of isolated|transport|periodic")
        if "periodic" in new_kinds:
            # Entering (or staying in) a periodic direction needs a real
            # lattice: keep an existing explicit cell (respected), else
            # refuse — there is nothing to derive one from.
            if s.cell is None:
                raise ValueError(
                    "a periodic axis needs an explicit commensurate cell "
                    "first — set the cell, then the axis kind (§ 4)")
            s.axis_kind = new_kinds
            s.__post_init__()
            notices.append(_notice(
                "info",
                "axis kinds updated; the existing explicit cell is kept "
                "(respected) and vacuum stays reference-only."))
            return s, notices
        s.axis_kind = new_kinds
        _reset_to_derived(s, "periodicity", notices)
        s.__post_init__()
        return s, notices

    if op == "cell":
        if payload is None:
            if "periodic" in kinds:
                raise ValueError(
                    "cannot clear the cell while an axis is periodic — a "
                    "derived bounding box is not a lattice (§ 4)")
            s.cell = None
            s.cell_origin = None
            s.__post_init__()
            notices.append(_notice(
                "info",
                "explicit cell cleared; the box is derived again "
                "(structure size + 2·vacuum, molecule centred) and vacuum "
                "is authoritative."))
            return s, notices
        try:
            cell = np.asarray(payload, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            raise ValueError(
                "cell must be a 3×3 matrix of numbers (Å)") from None
        s.cell = cell
        if s.cell_origin is not None:
            # v3 precedence: respect the existing explicit ORIGIN first.
            s.__post_init__()
            _refuse_on_error(s)
            # RECEIPT ONLY. Whether the result contains the structure is a
            # CONDITION, and conditions are answered once, by validate_periodicity
            # on the result (molview.md § 6.8).  Saying it here too put the same
            # fact in the answer twice, in two wordings.
            notices.append(_notice(
                "info",
                "explicit cell set; the existing origin is respected. "
                "Vacuum values are reference-only (§ 6.1)."))
            return s, notices
        # ... then respect the VACUUM: anchor at the expected corner.
        # A cell the structure cannot fit for ANY origin is REFUSED, not
        # stored — a stored-but-invalid cell locked every later door
        # (review finding, 2026-07-29).
        # No explicit origin: leave it unset — the corner is DERIVED from the
        # structure + vacuum by ``resolve_cell_origin``, so the box wraps the
        # structure without storing a computed value as truth (§ 6.1 clause 1).
        s.__post_init__()
        # A cell the structure cannot fit for ANY origin is REFUSED, not
        # stored — a stored-but-invalid cell locked every later door (review
        # finding, 2026-07-29).  Asked of the ONE checker now, so the rule and
        # its wording live in a single place.
        _refuse_on_error(s)
        notices.append(_notice(
            "info",
            "explicit cell set (origin first, then vacuum — § 6.2); no "
            "explicit origin, so the corner stays DERIVED at "
            f"{np.round(np.asarray(s.resolve_cell_origin(), dtype=float), 4).tolist()}"
            " and the box wraps the structure. Vacuum values are "
            "reference-only from now on."))
        return s, notices

    # op == "cell_origin"
    if s.cell is None:
        raise ValueError(
            "cell_origin is only meaningful with an explicit cell — "
            "the derived box computes its own corner (§ 3c)")
    if payload is None:
        # "Reset origin to default": literal cell_origin = None.  The corner is
        # then DERIVED again (structure + vacuum), so the box keeps wrapping the
        # structure — it does NOT jump to (0,0,0).  Same rule the load seam
        # applies to the same state (§ 6.1 row 3).
        s.cell_origin = None
        s.__post_init__()
        notices.append(_notice(
            "info",
            "cell_origin cleared — the box corner is DERIVED again at "
            f"{np.round(np.asarray(s.resolve_cell_origin(), dtype=float), 4).tolist()}"
            " (bbox_min − vacuum per isolated axis), so the box still wraps "
            "the structure. The other parameters have their freedom back; a "
            "vacuum / periodicity edit re-derives the whole box."))
        return s, notices
    origin = [float(x) for x in payload]
    if len(origin) != 3:
        raise ValueError("cell_origin must be 3 floats (Å)")
    s.cell_origin = np.asarray(origin, dtype=float)
    s.__post_init__()
    # The validator DECIDES here; it does not report here. Its notices are
    # CONDITIONS and the door answers those once, on the result (molview.md
    # § 6.8) -- merging them in as well put the same containment warning in one
    # answer twice, word for word.
    s, conditions = validate_periodicity(s)
    # A RECEIPT ONLY.  This used to append a warn behind `if not conditions:`
    # -- so the sentence saying why your vacuum stopped mattering was dropped
    # precisely when the box also had a problem (cell-plan.md § 3c).  That fact
    # is a CONDITION now (``cell.vacuum_ignored``), reported on every hand-over
    # whenever it is true, and the clearances ride with the containment finding
    # that is about them.
    notices.append(_notice("info", "cell_origin set."))
    return s, notices
