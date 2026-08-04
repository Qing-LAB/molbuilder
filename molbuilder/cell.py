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


def resolve(struct: Structure, *,
            box: Optional[np.ndarray] = None) -> ResolvedCell:
    """Work the box out, once, and say everything that is true of it.

    ``box`` overrides what the structure resolves to -- for the one caller that
    has a different one: a generator emitting a lattice it chose itself
    (``render_fdf`` passes ``cell=`` into ``validate``).  The checker must judge
    the box that will ACTUALLY be used, or it is answering about a box nobody
    will run.  Omitting it is the normal case and asks the structure.

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

    # NOTE the regime is NOT touched by ``box``.  A box handed in changes WHICH
    # box is measured; it says nothing about whether the USER typed a cell.
    # Conflating the two made `cell.vacuum_ignored` tell a user "your vacuum is
    # not being used: you typed a cell" when they had typed no cell at all and
    # the generator had merely passed the derived box in for checking.
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

    if box is not None:
        box = np.asarray(box, dtype=float).reshape(3, 3)
    try:
        if box is None:
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

    # TWO fractional projections, and only two: one from the corner this box
    # actually has, one from the world origin.  Everything below derives from
    # those, and the determinant is taken once.
    #
    # It was THREE, plus two determinants -- `_unfittable` projected at the
    # world origin while `contains_at_world_origin` projected the same thing
    # again, inside the module whose whole claim is that the box is worked out
    # ONCE.  Solving a 3x3 per atom is cheap; a module that does not keep its
    # own promise is not.
    frac_at_corner = _fractional(struct, box, corner)
    frac_at_origin = _fractional(struct, box, np.zeros(3))
    det = float(np.linalg.det(box))
    return ResolvedCell(
        box=box,
        corner=corner,
        volume=abs(det),
        contains_atoms=_contains(frac_at_corner, kinds),
        right_handed=det > 0.0,
        unfittable_axes=_unfittable(frac_at_origin, kinds),
        corner_was_derived=(struct.cell is not None
                            and struct.cell_origin is None),
        clearances=_clearances(frac_at_corner, box),
        contains_at_world_origin=_contains(frac_at_origin, kinds),
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


def _unfittable(frac: Optional[np.ndarray],
                kinds: Sequence[str]) -> Tuple[int, ...]:
    """Non-periodic axes the structure cannot fit along for ANY origin.

    The distinction from containment is the actionable one: a contained-ness
    failure is fixed by moving the corner, this one cannot be — the cell is
    simply too short along that axis.

    Measured on the SPAN of the fractional coordinates, which no choice of
    origin changes -- so the caller passes the projection it already made
    rather than this making a second one.
    """
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
            f"This cell is mirrored: its three vectors turn the wrong way "
            f"round (left-handed, det = {det:.6g}). Fix it by swapping any two "
            f"of the three rows, or by flipping the sign of one.",
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
            f"The molecule is longer than the cell along {names}, so it "
            f"cannot fit wherever you put the box. Make the cell bigger along "
            f"{names}, or clear the cell and let the box be sized around the "
            f"molecule.",
            "cell.unfittable"))

    if not rc.has_volume:
        lengths = " × ".join(f"{v:g}" for v in np.linalg.norm(rc.box, axis=1))
        out.append(Issue(
            "error",
            f"This box is flat ({lengths} Å), so there is no space for the "
            f"calculation to happen in. One side has no length — either the "
            f"molecule is flat that way and you set no vacuum there, or the "
            f"cell you typed has a row of zeros. Add vacuum on that side, or "
            f"give that row a real length. Nothing you set has been changed.",
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
    #
    # ALSO SILENT WHEN THE STORED VACUUM IS ALL ZEROS.  A zero vacuum being
    # ignored changes nothing about the box, so the sentence is true and
    # useless -- it corrects an expectation nobody has, and it was firing on the
    # commonest state a typed cell is in.  What earns the interruption is a
    # NUMBER THE USER CHOSE that has stopped counting.
    if (rc.is_manual and rc.stated_vacuum is not None
            and any(float(v) != 0.0 for v in rc.stated_vacuum)):
        typed = ", ".join(f"{v:g}" for v in rc.stated_vacuum)
        out.append(Issue(
            "info",
            f"Your vacuum ({typed} Å) is not being used, because you typed a "
            f"cell and that cell is the box. Vacuum only applies when the box "
            f"is worked out from the molecule. To use it again, clear the "
            f"cell.",
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
            f"No vacuum set, so {plural} {where} got the default {gap:g} Å on "
            f"each side — leaving {2 * gap:g} Å between your molecule and the "
            f"next copy of it. That is enough to make a valid box, but usually "
            f"too little for an accurate result: 8 Å per side or more is "
            f"typical. Set a vacuum to choose your own.",
            "cell.vacuum_defaulted"))

    # ONE CAUSE, ONE FINDING.  A box with no volume fails containment by
    # construction -- nothing fits in nothing -- so reporting both would hand
    # the user two problems to chase where there is one to fix.  The volume is
    # the cause; containment is its shadow.
    # ``unfittable`` already says the cell is too short, and says it better --
    # it names the axes and rules out moving the corner.  Repeating it as a
    # containment warning would be the same defect twice.
    if not rc.contains_atoms and rc.has_volume and not rc.unfittable_axes:
        gaps = ", ".join(f"{'abc'[i]} {n:.2f}/{f:.2f}"
                         for i, (n, f) in enumerate(rc.clearances))
        if rc.origin_is_user_owned:
            out.append(Issue(
                "warn",
                f"Some atoms are outside the box. Room to spare at each end, "
                f"in Å — a negative number means atoms stick out that side: "
                f"{gaps}. You typed both the cell and its corner, so neither "
                f"is adjusted for you. Move the corner, make the cell bigger, "
                f"or clear the corner and have it worked out for you.",
                "cell.atoms_outside"))
        else:
            out.append(Issue(
                "warn",
                f"Some atoms are outside the box. Room to spare at each end, "
                f"in Å — a negative number means atoms stick out that side: "
                f"{gaps}. The corner was already placed to wrap the molecule "
                f"as well as it can, so the cell itself is too small. Make it "
                f"bigger, or clear it and have the box sized around the "
                f"molecule.",
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
            f"Your cell does not say where its corner sits, so it was placed "
            f"at {np.round(rc.corner, 4).tolist()} to wrap the molecule. The "
            f"molecule fits, and nothing you set was changed.",
            "cell.corner_derived"))

    return out


def resolve_and_check(struct: Structure, *,
                      box: Optional[np.ndarray] = None
                      ) -> Tuple[ResolvedCell, List[Issue]]:
    """The whole line in one call — what almost every caller wants."""
    rc = resolve(struct, box=box)
    return rc, check(rc)
