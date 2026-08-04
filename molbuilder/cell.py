"""The one process line for the unit cell — resolve once, check once.

MODULE  cell (L1; imports ``structure`` + ``issues`` and nothing else)
ROLE    the ONE place a box is worked out, and the ONE place it is judged
USED-BY periodicity_gate (the hand-over gate + the edit door), validation/
        (the engine validators fold these in), siesta/input.py (the emitter),
        web/blueprints (through the gate)

Contract: docs/model/structure-periodicity.md § 6.1 / § 6.1a, and the finding
contract in docs/science/validation.md § 4.1 (R1–R6).

WHY THIS MODULE EXISTS (decided 2026-08-03 — cell-plan.md § 6a).  Two jobs had
grown many hands, and the hands disagreed.

**Working the box out** was spread over six methods on ``Structure`` that call
each other -- ``resolve_cell_origin`` → ``expected_cell_corner`` →
``effective_vacuum``, and ``resolve_cell`` → ``effective_vacuum`` again.  One
box computed its effective vacuum three or four times, and a caller could enter
at any of the six and get a partial view of the answer.  Those methods remain
(they are the arithmetic, and ``Structure`` owns its own fields); what changes
is that **nobody outside composes them any more** -- they compose here, once,
into a :class:`ResolvedCell`.

**Judging the box** was worse, because it was TWO systems.  The gate emitted
``{level, message, about}`` notices; the validators emitted ``Issue``s with a
``where``; and three call sites raised bare ``ValueError``.  The same fact --
*this box has no volume* -- was decided in FOUR places at TWO thresholds
(``structure.py`` 1e-8, the gate 1e-8, the gate's reset path 1e-6, the emitter
1e-6).  Only the delivery contract held it together, by hand.

So: one resolver, one checker, one vocabulary.  ``where`` is the stable id, as
R1–R6 already require, which is why the Cell page and the Generate preflight can
finally show the same finding.

WHAT IS NOT HERE, DELIBERATELY.

* **Engine checks.**  ``cell.vacuum_thin`` is SIESTA's advice,
  ``cell.periodic_in_gas_phase`` is PySCF's, ``cell.image_distance`` is a
  geometry measurement.  They already are Issues with a ``where`` and they
  already reach both surfaces; they stay in ``validation/`` where the engine
  knowledge is.  This module owns the facts that are true of a box regardless
  of what will be run on it.
* **Receipts.**  "explicit cell cleared", "vacuum cleared" -- what an edit just
  DID.  A receipt is true for a moment and then meaningless, which is not what a
  finding is.  They stay on the gate, and the door returns them beside these.
* **Repair.**  Clause 1 forbids writing a resolved value back.  Nothing here
  mutates; ``resolve`` reads and ``check`` reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .issues import Issue
from .structure import Structure

#: THE zero-volume threshold, in Å³.  One constant, where there were two.
#:
#: 1e-6 rather than 1e-8: it was the value the EMITTER used, and the emitter is
#: the last line of defence before a cell reaches SIESTA, which builds
#: reciprocal vectors from it and fails outright on a singular one.  Unifying
#: downward would have loosened the guard that matters most.  The difference is
#: academic in Å³ -- a real cell is 10²–10⁴, and even a deliberately tiny one is
#: ~1 -- so both values only ever catch true degeneracy.
ZERO_VOLUME_TOL = 1e-6

#: Containment tolerance in fractional coordinates: loose enough to forgive a
#: round-tripped float, tight enough that "half the molecule outside" cannot
#: pass.  Mirrors ``periodicity_gate._EPS``.
_CONTAIN_EPS = 1e-6


@dataclass(frozen=True)
class ResolvedCell:
    """Everything true about a structure's box, worked out ONCE.

    Consumers read fields.  Nobody re-derives, and nobody calls the six
    ``Structure`` resolvers directly any more -- that is the whole point.

    ``box`` and ``corner`` are ``None`` only when the box could not be worked
    out at all (``unresolvable`` then says why, and :func:`check` turns it into
    an error).  Everything else is always populated, so a caller never has to
    branch on absence to ask a simple question.
    """

    #: The 3×3 lattice the calculation will use, or None if unresolvable.
    box: Optional[np.ndarray]
    #: The world-space low corner the box emanates from.  Never None when
    #: ``box`` is set: the "no explicit origin" case is RESOLVED here rather
    #: than handed on as an absence, which is what let two seams disagree
    #: before (§ 6.1, "no explicit origin means derive the corner").
    corner: Optional[np.ndarray]
    #: The per-side gap the box was actually built from -- what the user set,
    #: or the default where they set nothing.
    vacuum: Tuple[float, float, float]
    #: What the structure itself says: the typed triple, or None for "nobody
    #: chose one".  The pair (this, ``vacuum``) is what makes "(default)"
    #: answerable without a second opinion.
    stated_vacuum: Optional[Tuple[float, float, float]]
    axis_kind: Tuple[str, str, str]
    #: "manual" when an explicit cell decides the box (vacuum is then
    #: reference-only), "derived" when the molecule + vacuum + kinds do.
    regime: str
    #: Axes whose gap is the default because no vacuum was set.
    defaulted_axes: Tuple[int, ...]
    #: |det(box)|, in Å³.  0.0 when unresolvable.
    volume: float
    #: True when every atom sits inside the box along every NON-periodic axis.
    #: Along a periodic axis, atoms outside are legitimate images, so they are
    #: never counted against containment (§ 2).
    contains_atoms: bool
    #: Set when the box could not be worked out; the reason, in the user's
    #: words.  :func:`check` promotes it to an error Issue.
    unresolvable: Optional[str] = None
    #: True when the user stored a ``cell_origin`` of their own.  A user-owned
    #: origin is never rewritten, and a box that misses its atoms under one is
    #: reported differently from one under a corner we derived.
    origin_is_user_owned: bool = False
    #: det(box) > 0.  A left-handed cell is not a representable state: SIESTA's
    #: reciprocal vectors come out mirrored.
    right_handed: bool = True
    #: Non-periodic axes whose FRACTIONAL extent exceeds 1 — the structure
    #: cannot fit along them for ANY origin, so moving the corner cannot help.
    #: Distinct from ``contains_atoms``, which is about the corner it HAS.
    unfittable_axes: Tuple[int, ...] = ()
    #: True when an explicit cell stores no origin, so the corner above was
    #: worked out rather than stored.  Reported, never written back (§ 6.1).
    corner_was_derived: bool = False
    #: Per-axis (near, far) gap in Å between the structure and the box faces.
    #: Negative means atoms poke out of that face.  Empty without a box.
    clearances: Tuple[Tuple[float, float], ...] = ()
    #: Would the box contain the atoms if it sat at the WORLD origin?  This is
    #: the imported-crystal test, and it is a different question from
    #: ``contains_atoms``, which asks about the corner the box actually has.
    #: True here means the corner is the world origin and nothing was worked
    #: out, so there is nothing to disclose.
    contains_at_world_origin: bool = True

    @property
    def has_volume(self) -> bool:
        return self.volume > ZERO_VOLUME_TOL

    @property
    def is_manual(self) -> bool:
        return self.regime == "manual"


def resolve(struct: Structure) -> ResolvedCell:
    """Work the box out, once, and say everything that is true of it.

    Never raises for a *structure* problem: a periodic axis with no lattice is
    a legitimate thing to have loaded and be about to fix, so it comes back as
    ``unresolvable`` rather than an exception.  That is what lets one code path
    serve both "refuse to generate" and "open it so it can be corrected"
    (§ 8.2's two verdicts) instead of the caller catching exceptions to tell
    them apart.
    """
    kinds = tuple(struct.axis_kind or ("isolated", "isolated", "isolated"))
    stated = (tuple(float(v) for v in struct.vacuum)
              if struct.vacuum is not None else None)
    eff = struct.effective_vacuum()
    regime = "manual" if struct.cell is not None else "derived"
    defaulted = tuple(struct.defaulted_vacuum_axes())

    common = dict(
        vacuum=eff,
        stated_vacuum=stated,
        axis_kind=kinds,                       # type: ignore[arg-type]
        regime=regime,
        defaulted_axes=defaulted,
        origin_is_user_owned=struct.cell_origin is not None,
    )

    if struct.n_atoms == 0:
        # Nothing to wrap.  Not an error -- a blank viewer is a legal state --
        # so it resolves to "no box" and every check stays quiet.
        return ResolvedCell(box=None, corner=None, volume=0.0,
                            contains_atoms=True, **common)

    try:
        box = np.asarray(struct.resolve_cell(), dtype=float)
    except ValueError as exc:
        # The one structural impossibility ``resolve_cell`` raises for: a
        # `periodic` axis with no explicit lattice.  A bounding box is not a
        # lattice, and one can never be invented from coordinates (§ 3).
        return ResolvedCell(box=None, corner=None, volume=0.0,
                            contains_atoms=True, unresolvable=str(exc),
                            **common)

    corner = struct.resolve_cell_origin()
    if corner is None:
        corner = np.zeros(3, dtype=float)      # the world origin, resolved
    else:
        corner = np.asarray(corner, dtype=float)

    frac_at_corner = _fractional(struct, box, corner)
    return ResolvedCell(
        box=box,
        corner=corner,
        volume=abs(float(np.linalg.det(box))),
        contains_atoms=_contains(frac_at_corner, kinds),
        right_handed=float(np.linalg.det(box)) > 0.0,
        unfittable_axes=_unfittable(struct, box, kinds),
        corner_was_derived=(struct.cell is not None
                            and struct.cell_origin is None),
        clearances=_clearances(frac_at_corner, box),
        contains_at_world_origin=_contains(
            _fractional(struct, box, np.zeros(3)), kinds),
        **common,
    )


def _fractional(struct: Structure, box: np.ndarray,
                origin: np.ndarray) -> Optional[np.ndarray]:
    """Atom positions in fractions of the box, measured from ``origin``.

    Triclinic-safe: solves ``boxᵀ · frac = pos − origin`` rather than dividing
    by row norms, which would be wrong the moment a lattice vector is not
    axis-aligned.  ``None`` when the box is singular and nothing can be solved.
    """
    if struct.n_atoms == 0:
        return None
    try:
        rel = struct.positions.astype(float) - np.asarray(
            origin, dtype=float).reshape(1, 3)
        return np.linalg.solve(box.T, rel.T).T
    except np.linalg.LinAlgError:
        return None


def _contains(frac: Optional[np.ndarray], kinds: Sequence[str]) -> bool:
    """Every atom inside ``[0, 1)`` along every NON-periodic axis.

    Along a periodic axis an atom outside the cell is a legitimate image the
    engine wraps, so containment is never required there (§ 2) — requiring it
    everywhere made real crystals unopenable.
    """
    if frac is None:
        return False                            # singular box: nothing fits
    for i, kind in enumerate(kinds):
        if kind == "periodic":
            continue
        if not (np.all(frac[:, i] >= -_CONTAIN_EPS)
                and np.all(frac[:, i] <= 1.0 + _CONTAIN_EPS)):
            return False
    return True


def _unfittable(struct: Structure, box: np.ndarray,
                kinds: Sequence[str]) -> Tuple[int, ...]:
    """Non-periodic axes the structure cannot fit along for ANY origin.

    The distinction from containment is the actionable one: a contained-ness
    failure is fixed by moving the corner, this one cannot be — the cell is
    simply too short along that axis.
    """
    frac = _fractional(struct, box, np.zeros(3))
    if frac is None:
        return ()
    return tuple(
        i for i, kind in enumerate(kinds)
        if kind != "periodic"
        and float(frac[:, i].max() - frac[:, i].min()) > 1.0 + 2 * _CONTAIN_EPS
    )


def _clearances(frac: Optional[np.ndarray],
                box: np.ndarray) -> Tuple[Tuple[float, float], ...]:
    """Per-axis (near, far) gap in Å between the structure and the box faces."""
    if frac is None:
        return ()
    lens = np.linalg.norm(box, axis=1)
    return tuple(
        (float(frac[:, i].min() * lens[i]),
         float((1.0 - frac[:, i].max()) * lens[i]))
        for i in range(3)
    )


# --------------------------------------------------------------------- #
#  The one checker                                                      #
# --------------------------------------------------------------------- #

def check(rc: ResolvedCell) -> List[Issue]:
    """Every fact that is true of a box, whatever will be run on it.

    Severity is the whole verdict mechanism: ``report(issues)`` raises on
    `error`, so a generating door refuses and a loading door -- which calls
    ``report(..., raise_on_error=False)`` or simply passes the list on --
    reports.  Neither decides for itself what a bad box costs (§ 8.2).
    """
    out: List[Issue] = []

    if rc.unresolvable:
        out.append(Issue("error", rc.unresolvable, "cell.unresolvable"))
        return out                              # nothing else can be said

    if rc.box is None:
        return out                              # empty structure: no box, no facts

    # ``has_volume`` first: a degenerate box has det == 0, which is not
    # "left-handed" -- it is the no-volume finding below, and reporting both
    # would give one cause two names.  Left-handed means genuinely MIRRORED
    # (det < 0), which is a different repair: swap two vectors.
    if not rc.right_handed and rc.has_volume:
        det = float(np.linalg.det(rc.box))
        out.append(Issue(
            "error",
            f"The cell must be right-handed (det > 0); this one has "
            f"det = {det:.6g}. Swap two lattice vectors, or negate one.",
            "cell.left_handed"))
        # SHORT-CIRCUIT, as the gate's `_require_right_handed` did by raising:
        # in a mirrored frame the fractional coordinates run backwards, so
        # containment and clearances describe a box nobody has.  Fix the
        # handedness and the rest can be asked honestly.
        return out

    if rc.unfittable_axes:
        names = ", ".join("abc"[i] for i in rc.unfittable_axes)
        out.append(Issue(
            "error",
            f"The structure is longer than the cell along {names}, so no "
            f"corner can make it fit — moving the origin cannot help. Enlarge "
            f"the cell along those axes, or clear it to have the box derived "
            f"from the molecule.",
            "cell.unfittable"))

    if not rc.has_volume:
        lengths = " × ".join(f"{v:g}" for v in np.linalg.norm(rc.box, axis=1))
        out.append(Issue(
            "error",
            f"This box has no volume ({lengths} Å): there is nothing for the "
            f"calculation to happen in. An axis has no length — either the "
            f"molecule is flat along it and the vacuum there is zero, or a "
            f"typed cell has a zero row. Your values are kept exactly as you "
            f"set them; give that axis a non-zero gap, or type a cell with a "
            f"real length on it.",
            "cell.no_volume"))

    # YOUR VACUUM IS DOING NOTHING -- said whenever it is true, which is the
    # point (cell-plan.md § 3c).  It used to live on the cell_origin RECEIPT
    # behind `if not conditions:`, so the one sentence explaining why a number
    # you typed stopped mattering was dropped exactly when the box ALSO had a
    # problem -- the moment you most needed it.  A condition is not a receipt:
    # it is true until the regime changes, so it belongs here and is answered
    # on every hand-over.
    #
    # Silent when no vacuum was set: there is no expectation to correct.
    if rc.is_manual and rc.stated_vacuum is not None:
        typed = ", ".join(f"{v:g}" for v in rc.stated_vacuum)
        out.append(Issue(
            "info",
            f"The vacuum you set ({typed} Å per side) is not being used: you "
            f"typed a cell, and an explicit cell IS the box. Vacuum only sizes "
            f"a box that is worked out from the molecule. Clear the cell to go "
            f"back to that, or edit the cell itself.",
            "cell.vacuum_ignored"))

    # ONLY IN THE DERIVED REGIME.  Under an explicit cell the vacuum is
    # reference-only (§ 6.2), so announcing a default gap would describe a
    # number that never reaches the calculation -- a molecule in a hand-typed
    # 30 Å box would be told the box came from a 3 Å default.  Same trap as
    # ``cell.vacuum_thin``, which had to learn the same lesson.
    if rc.defaulted_axes and not rc.is_manual:
        gap = rc.vacuum[rc.defaulted_axes[0]]
        where = ", ".join("abc"[i] for i in rc.defaulted_axes)
        plural = "axes" if len(rc.defaulted_axes) > 1 else "axis"
        out.append(Issue(
            "info",
            f"No vacuum was set, so the default {gap:g} Å per side is used on "
            f"the isolated {plural} {where} — which leaves {2 * gap:g} Å "
            f"between the molecule and its periodic image. That is enough for "
            f"the box to be well-formed, not enough for a converged "
            f"isolated-molecule run: those usually want ≥ 8 Å per side. Set a "
            f"vacuum to choose your own.",
            "cell.vacuum_defaulted"))

    # ONE CAUSE, ONE FINDING.  A box with no volume fails containment by
    # construction -- nothing fits in nothing -- so reporting both would hand
    # the user two problems to chase where there is one to fix.  The volume is
    # the cause; containment is its shadow.
    # ``unfittable`` already says the cell is too short, and says it better --
    # it names the axes and rules out moving the corner.  Repeating it as a
    # containment warning would be the same defect twice.
    if not rc.contains_atoms and rc.has_volume and not rc.unfittable_axes:
        gaps = ", ".join(f"{'abc'[i]}: ({n:.2f}, {f:.2f})"
                         for i, (n, f) in enumerate(rc.clearances))
        if rc.origin_is_user_owned:
            out.append(Issue(
                "warn",
                f"The box does not contain the structure along a non-periodic "
                f"axis — per-axis (near, far) clearances in Å: {gaps}. BOTH "
                f"the cell and its origin are yours, so neither is changed "
                f"for you, and vacuum is not respected under an origin you "
                f"typed. Move the origin, widen the cell, or clear the origin "
                f"to have the corner worked out.",
                "cell.atoms_outside"))
        else:
            out.append(Issue(
                "warn",
                f"The box does not contain the structure along a non-periodic "
                f"axis — per-axis (near, far) clearances in Å: {gaps}. The "
                f"corner was worked out to wrap the atoms and still does not "
                f"fit them. Widen the cell, or clear it to have the box "
                f"derived from the molecule.",
                "cell.atoms_outside"))

    # A CONDITION, not a complaint: the state is legal (§ 6.1 row 3) and
    # nothing was written back.  It is said because the corner is a number the
    # user never typed, and the box is drawn and emitted from it.
    # Silent for the IMPORTED-CRYSTAL case (§ 6.1 row 2): the box already
    # contains the atoms where they sit, so the corner IS the world origin and
    # nothing was worked out.  Announcing a derivation that did not happen is
    # noise on the commonest state a crystal file has.
    if (rc.corner_was_derived and not rc.contains_at_world_origin
            and rc.contains_atoms and rc.has_volume):
        out.append(Issue(
            "info",
            f"The cell stores no origin, so the box corner was worked out: "
            f"{np.round(rc.corner, 4).tolist()} (bbox_min − vacuum per "
            f"isolated axis; the structure centred where the per-side vacuum "
            f"does not fit). The box wraps the structure, nothing was "
            f"changed, and vacuum stays reference-only under an explicit "
            f"cell.",
            "cell.corner_derived"))

    return out


def resolve_and_check(struct: Structure) -> Tuple[ResolvedCell, List[Issue]]:
    """The whole line in one call — what almost every caller wants."""
    rc = resolve(struct)
    return rc, check(rc)
