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


# --------------------------------------------------------------------- #
#  A layered slab's own periodic repeat                                 #
#                                                                       #
#  Contract: docs/science/junction-cell.md.  Two callers derive the      #
#  same number for two boxes -- the junction cell that flanks a molecule #
#  (modify.add_slab) and the bulk-lead cell TranSIESTA needs            #
#  (transport.wizard.extract_electrode_model).  It lived in the wizard   #
#  first; it is here so the second caller reuses it instead of growing   #
#  a second copy that can disagree.                                      #
# --------------------------------------------------------------------- #

#: Two atoms whose z differ by less than this are the same atomic layer.
#: 0.5 Å is well below any real interlayer spacing (Au(111) ≈ 2.35 Å)
#: and well above relaxation jitter within a layer.
LAYER_TOL_ANG = 0.5


def detect_layers(z, tol_ang: float = LAYER_TOL_ANG) -> List[float]:
    """Cluster z-coordinates into atomic layers; return sorted layer
    centroids (low → high).

    Single-linkage along the sorted z axis: a new layer starts whenever
    the gap to the previous atom exceeds ``tol_ang``.  Returns the mean
    z of each cluster.
    """
    zs = np.sort(np.asarray(z, dtype=float))
    if zs.size == 0:
        return []
    layers: List[List[float]] = [[float(zs[0])]]
    for val in zs[1:]:
        if float(val) - layers[-1][-1] > tol_ang:
            layers.append([float(val)])
        else:
            layers[-1].append(float(val))
    return [float(np.mean(layer)) for layer in layers]


def bulk_z_period(layer_z: Sequence[float]) -> Tuple[float, float, int]:
    """Propose the bulk repeat from the layer centroids.

    Returns ``(z_period, d_interlayer, n_layers)`` where
    ``d_interlayer`` is the *median* adjacent-layer spacing (robust to a
    slightly off top/bottom layer) and
    ``z_period = z_span + d_interlayer`` so the slab tiles seamlessly:
    the next periodic image's first layer lands exactly one interlayer
    spacing above the current top layer.

    The median is measured on the slab AS BUILT, so an ``inter_layer_offset``
    override is honoured without being passed in.

    Raises ``ValueError`` on fewer than 2 layers (the repeat is
    undeterminable from a single layer — the caller must supply it).
    """
    n = len(layer_z)
    if n < 2:
        raise ValueError(
            "cannot derive a bulk z-period from a single atomic layer; "
            "pass an explicit z_period (the lead's bulk lattice repeat)")
    diffs = np.diff(np.asarray(layer_z, dtype=float))
    d_interlayer = float(np.median(diffs))
    z_span = float(layer_z[-1] - layer_z[0])
    z_period = z_span + d_interlayer
    return z_period, d_interlayer, n


# --------------------------------------------------------------------- #
#  Reading a lattice constant back OUT of a relaxed result                #
#  (archive/2026-09-01-modify-redesign-plan.md § 3.3)                                  #
# --------------------------------------------------------------------- #

#: Fractional width of the nearest-neighbour shell.  6% holds a relaxed
#: crystal's jitter and is far narrower than the gap to the second shell
#: (√2 ≈ 1.414 d), so a "neighbour" cannot quietly become a next-nearest one.
NN_SHELL_TOL = 0.06

#: Below this, two positions are the same point, not two atoms.  Only used to
#: drop an atom's distance to ITSELF at the identity translation -- its
#: distances to its own periodic IMAGES are real and are counted, which is what
#: makes this correct on a one-atom primitive cell where the nearest neighbour
#: IS an image.
_SAME_POINT_ANG = 1e-9


@dataclass(frozen=True)
class FccMeasurement:
    """What a relaxed bulk result says its lattice constant is.

    ``a`` is the CONVENTIONAL cubic edge, from ``a = √2 · d_nn``.
    """

    #: Å — the closest two atoms get, periodic images included.
    d_nn: float
    #: Å — the conventional cubic edge implied by ``d_nn``.
    a: float
    #: How many neighbours an atom has in that shell; the MEDIAN over atoms,
    #: so a surface layer cannot drag the answer down.  12 for bulk fcc.
    coordination: int
    #: Å — the next distinct distance, or ``None`` if there is not one.  In
    #: fcc it sits at ``√2 · d_nn``; that ratio is the cheap catch for a
    #: distorted or non-cubic result.
    second_shell: Optional[float]
    #: How many atoms the measurement was taken over.
    n_atoms: int


def measure_fcc(positions, cell) -> FccMeasurement:
    """Measure the fcc lattice constant from ATOMS, not from the cell.

    **Why the atoms.**  The cell of a relaxed result may be conventional cubic
    (edge ``a``), primitive rhombohedral (edge ``a/√2``), or the user's own
    m×n×N layered lead cell (edge ``m·a/√2``) -- three different relations to
    ``a``, and the file does not say which.  Reading the cell means guessing
    the user's convention.  The nearest-neighbour distance assumes nothing:
    whatever the box, the closest two atoms in an fcc crystal are ``a/√2``
    apart.

    Distances are taken under the **minimum image convention**: fractional
    deltas are wrapped to the nearest image and then checked against the 27
    surrounding translations, which is exact rather than merely usually right
    on a skewed cell.

    NOT :func:`validation.geometry._min_image_distance`, and the difference
    matters: that one EXCLUDES the identity translation on purpose, because it
    asks "how close does this molecule sit to its periodic copies" -- an
    artefact question.  This one asks for the nearest neighbour, which in a
    supercell is overwhelmingly an in-cell atom.  Handing this job to that
    function returns the distance across the box boundary and calls it a bond.

    Raises ``ValueError`` on fewer than two atoms or a cell with no volume --
    both are cases where there is no distance to report rather than a distance
    that happens to be wrong.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    box = np.asarray(cell, dtype=float).reshape(3, 3)
    n = int(pos.shape[0])
    if n < 1:
        # ONE ATOM IS ENOUGH, and the first version of this guard said
        # `n < 2`.  A primitive fcc cell holds exactly one atom whose twelve
        # nearest neighbours are its own periodic images -- precisely the
        # case the image handling below exists for -- so refusing it hid a
        # real bug in that handling behind a plausible sentence.  The
        # comment two blocks up already promised the opposite; a test
        # settled which of the two was the mistake.
        raise ValueError("there are no atoms of that element to measure")
    if abs(float(np.linalg.det(box))) < ZERO_VOLUME_TOL:
        raise ValueError(
            "this cell has no volume, so there are no periodic images to "
            "measure against")

    inv = np.linalg.inv(box)
    whole = np.array([(i, j, k)
                      for i in (-1, 0, 1)
                      for j in (-1, 0, 1)
                      for k in (-1, 0, 1)], dtype=float) @ box   # (27, 3)

    d_nn = float("inf")
    per_atom: List[np.ndarray] = []
    for i in range(n):
        frac = (pos - pos[i]) @ inv            # (n, 3)
        frac -= np.round(frac)                 # nearest image
        near = frac @ box                      # (n, 3), cartesian
        d = np.linalg.norm(near[:, None, :] + whole[None, :, :], axis=2)
        d = d.reshape(-1)
        # Only the exact self-at-the-identity-translation is dropped.  Taking a
        # minimum over the 27 images first would have dropped an atom's own
        # images WITH it -- and on a one-atom primitive cell those are the only
        # neighbours there are, so the function would have had nothing to
        # measure and said so, wrongly.  (Caught on the first run, 2026-08-30.)
        d = d[d > _SAME_POINT_ANG]
        if d.size:
            per_atom.append(d)
            d_nn = min(d_nn, float(d.min()))

    if not per_atom or not np.isfinite(d_nn):
        raise ValueError("no two atoms are a measurable distance apart")

    shell = d_nn * (1.0 + NN_SHELL_TOL)
    counts = [int((d <= shell).sum()) for d in per_atom]
    # A crystal whose every measured distance falls inside the first shell has
    # no second shell to report -- one image cage and nothing beyond it.  That
    # is a `None`, not an empty concatenate.
    beyond = [d[d > shell] for d in per_atom]
    beyond = [chunk for chunk in beyond if chunk.size]
    second = float(np.concatenate(beyond).min()) if beyond else None

    return FccMeasurement(
        d_nn=d_nn,
        a=float(np.sqrt(2.0) * d_nn),
        coordination=int(np.median(counts)) if counts else 0,
        second_shell=second,
        n_atoms=n,
    )


# --------------------------------------------------------------------- #
#  Does the boundary continue the crystal?                                #
#  (archive/2026-09-01-bench-and-junction-plan.md § 2.4)                               #
# --------------------------------------------------------------------- #

#: Two atoms closer than this across the boundary are colliding, not bonded.
SEAM_COLLISION_ANG = 0.5

#: How far a lateral offset may sit from a reference and still be called it.
#: 0.3 A is well under the smallest real registry step -- a/sqrt(6) = 1.67 A
#: on fcc(111), the tightest of the three (fcc(100) a/2 = 2.04, fcc(110) 2.50).
#: (1.44 A = a/(2*sqrt(2)) is the fcc(110) INTERLAYER spacing, measured along
#: z; it is not a registry step, and it is not (111)'s.)
#: and well over any relaxation jitter in a frozen outer layer.
SEAM_STEP_TOL_ANG = 0.3

#: How many interlayer spacings of room mean the boundary is VACUUM rather
#: than a seam.
#:
#: A boundary is only a seam when the two faces are close enough to be
#: bonded.  Open it wider and there is no crystal there to continue -- there
#: is a free surface, which is what a slab calculation WANTS.
#:
#: Without this the registry test answers a question nobody asked: a slab
#: built beside a molecule gets a box tall enough for both, and its faces came
#: back `continues` across 7.5 Å of empty space because the two layers
#: happened to land on matching sites.  Registry agreement across vacuum is a
#: coincidence, not a crystal.
#:
#: 1.5 sits above any real relaxation (the measured `Au-BDT-Au` seam holds at
#: 1.0000 -- 2.4008 Å across a 2.4006 Å spacing) and far below the several
#: spacings of room that even a thin vacuum layer opens.
SEAM_VACUUM_FACTOR = 1.5


@dataclass(frozen=True)
class SeamVerdict:
    """What the periodic boundary does to the crystal."""

    #: ``continues`` · ``eclipsed`` · ``twin`` · ``collision`` ·
    #: ``vacuum`` · ``unknown``
    verdict: str
    #: Å — the closest approach across the boundary.  NOT the same number as
    #: :attr:`z_room`: where consecutive layers sit laterally offset, the
    #: nearest atom across the boundary is further than the layers are apart.
    gap: float
    #: Å — the VERTICAL room at the boundary, top face to the imaged bottom
    #: one.  This is what decides collision and vacuum, because it is what
    #: "is there space for a layer here" actually asks.
    z_room: float
    #: Å — the lateral offset across the boundary, shortest representative.
    seam_step: Tuple[float, float]
    #: Å — the same measurement between the top two layers INSIDE the slab.
    #: Reported for diagnosis only: the verdict is NOT reached by comparing
    #: these two vectors (see :func:`classify_seam` for why that fails).
    slab_step: Tuple[float, float]
    #: One sentence naming what is wrong, or ``None`` when it continues.
    message: Optional[str] = None
    #: Layers per stacking period, MEASURED from this structure -- not taken
    #: from :data:`STACKING_PERIOD`.  Compare it against the plane's expected
    #: period to catch a slab too thin to determine its own stacking: two
    #: layers of (111) are ``A,B``, which is fcc and hcp alike, so the boundary
    #: continues them as a 2-period stack that is not gold's.  ``None`` when no
    #: period could be measured.
    period: Optional[int] = None


def _shortest_lateral(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """The shortest lateral displacement from layer ``a`` to layer ``b``.

    Not a centroid difference and not a corner-atom difference: a layer is a
    finite patch, so translating it changes which atom sits at the corner and
    changes the centroid too when the patches differ.  The shortest
    atom-to-atom displacement is canonical for two lattices of the same shape
    -- it is the registry offset reduced to its smallest representative, with
    no choice of marker to get wrong.

    THE SEAM AND THE IN-SLAB REFERENCE ARE MEASURED BY THIS SAME FUNCTION,
    which is what makes them comparable: whatever convention the reduction
    lands on applies to both, so the modulo-the-lattice ambiguity cancels
    instead of having to be resolved.
    """
    d = b_xy[None, :, :] - a_xy[:, None, :]              # (na, nb, 2)
    flat = d.reshape(-1, 2)
    return flat[int(np.argmin(np.linalg.norm(flat, axis=1)))]


def _failing_condition(n_layers: int, period: Optional[int]) -> str:
    """Which of `junction-cell.md`'s two conditions this structure breaks.

    The contract requires the warning to NAME the failing condition, and the
    condition does not follow from the verdict -- the real `Au-BDT-Au`
    junction is `eclipsed` with 6 layers per side, which is a whole number of
    periods, so § 3.1 HOLDS there and only the mirror is wrong.  Reading the
    condition off the verdict called that one backwards.

    So it is measured: the layer count is the one that fails when the layers
    are not a whole number of stacking periods; otherwise the placement is.
    Only the structure as handed over is examined -- for a junction that is
    both electrodes together, so this counts every layer, not one side's.
    """
    if period and n_layers % period:
        return (f"The LAYER COUNT is what fails (junction-cell.md § 3.1): "
                f"{n_layers} layers is not a whole number of "
                f"{period}-layer stacking periods")
    return ("The PLACEMENT is what fails (junction-cell.md § 3.2): the layer "
            "count is a whole number of stacking periods, so it is where the "
            "layers were put -- mirrored rather than translated")


def _coincide(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """How far layer ``b`` sits from layer ``a`` laterally, in Angstrom.

    Zero means the two layers occupy the SAME registry -- the same lateral
    sites, however the patches are wrapped.
    """
    return float(np.linalg.norm(_shortest_lateral(a_xy, b_xy)))


def classify_seam(positions, cell) -> SeamVerdict:
    """Does the cell boundary continue the crystal, and if not, how not?

    Compares the BOTTOM layer's image one cell up against the layers actually
    in the slab, and reports **which one it lands on**.

    **The distance alone is not the test** (`junction-cell.md` § 3.1): a twin
    has the correct bulk bond length, so a distance check passes it.  Only the
    registry separates continuation from a twin.

    **AND THE REGISTRY IS NOT TESTED BY ARITHMETIC.**  Two earlier versions
    compared the seam's lateral step against the in-slab step -- first
    directly, then reduced modulo the cell -- and both were wrong for the same
    reason the slab builder hit: consecutive layers step by `period` DIFFERENT
    vectors that agree only modulo the PRIMITIVE lattice, while the cell on
    hand is the SUPERCELL's, m times larger.  A perfectly continuous 3-layer
    Au(111) boundary came back `unknown`.

    So this asks the question that has an answer without any lattice
    arithmetic: **which layer of this slab does the imaged one coincide
    with?**  In a crystal of period `p`, the layer above the top is a repeat
    of the one `p` below it, so:

    ==================  ==========================================
    the image lands on  verdict
    ==================  ==========================================
    layer ``N - p``     ``continues`` -- the crystal carries on
    the top layer       ``eclipsed`` -- stacking repeats, not advances
    layer ``N - 2``     ``twin`` -- right bond, reversed stacking
    ==================  ==========================================

    ``p`` is the slab's OWN period, measured here rather than passed in, so
    the plane never has to be stated and a strained or unusual slab is judged
    on what it is.

    A WARNING, NEVER A REFUSAL, is the caller's job -- an eclipsed seam is
    wrong for a periodic crystal and harmless in a relaxation whose outer
    layers are frozen, and only the caller knows which it is running.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    box = np.asarray(cell, dtype=float).reshape(3, 3)

    def unknown(why):
        return SeamVerdict("unknown", float("nan"), float("nan"),
                            (0.0, 0.0), (0.0, 0.0), why)

    if pos.shape[0] < 2:
        return unknown("fewer than two atoms: there is no seam to classify")
    layers = detect_layers(pos[:, 2])
    if len(layers) < 2:
        return unknown(
            "this structure has one atomic layer, so there is nothing for the "
            "boundary to continue")

    def _at(z):
        return pos[np.abs(pos[:, 2] - z) < LAYER_TOL_ANG]

    sets = [_at(z) for z in layers]
    if any(len(a) == 0 for a in sets):
        return unknown("a layer came back empty")
    N = len(sets)
    top = sets[-1]
    up = sets[0] + box[2]

    gap = float(np.linalg.norm(
        up[None, :, :] - top[:, None, :], axis=2).min())
    # THE ROOM AT THE BOUNDARY IS VERTICAL, and the two are not the same
    # number: where consecutive layers sit laterally offset, the nearest atom
    # ACROSS the boundary is further than the layers are apart.  Measuring
    # room with the 3-D distance called a perfectly padded fcc(110) boundary
    # vacuum (2.04 Å of reach across a 1.44 Å spacing), and called a box with
    # its padding removed -- layers at the SAME z, 1.66 Å apart sideways -- a
    # continuation.  So room is measured up, and `gap` is reported as the
    # closest approach it is.
    z_room = float(up[:, 2].min() - top[:, 2].max())
    seam = _shortest_lateral(top[:, :2], up[:, :2])
    slab = _shortest_lateral(sets[-2][:, :2], top[:, :2])
    pair = (tuple(float(v) for v in seam), tuple(float(v) for v in slab))

    spacing = float(np.median(np.diff(layers))) if len(layers) >= 2 else 0.0

    if z_room < SEAM_COLLISION_ANG:
        return SeamVerdict(
"collision", gap, z_room, *pair, period=None,
            message=(f"the boundary leaves {z_room:.2f} Å of room -- the box is "
                     f"not padded, and no engine can use it "
                     f"(science/junction-cell.md § 1)"))

    if spacing > 0 and z_room > SEAM_VACUUM_FACTOR * spacing:
        return SeamVerdict(
"vacuum", gap, z_room, *pair, period=None,
            message=(f"the boundary is {z_room:.2f} Å of vacuum, against a "
                     f"{spacing:.2f} Å layer spacing -- a free surface, not a "
                     f"seam.  Nothing continues across it, and for a slab "
                     f"nothing should"))

    # THE SLAB'S OWN STACKING PERIOD, measured: the smallest step at which a
    # layer repeats.  Measured rather than taken from STACKING_PERIOD so the
    # plane never has to be stated, and so a slab that is not what it claims
    # is judged on what it is.
    #
    # The search runs over the slab's layers PLUS the imaged one, because a
    # slab exactly one period tall has no repeat inside it -- layer 0 comes
    # back only across the boundary.  Leaving the image out found no period
    # for exactly the three slabs that continue most perfectly (3-layer (111),
    # 2-layer (100) and (110)) and called all three wrong.
    stack = sets + [up]
    period = next((p for p in range(1, N + 1)
                   if _coincide(stack[0][:, :2], stack[p][:, :2])
                   < SEAM_STEP_TOL_ANG), None)

    if _coincide(top[:, :2], up[:, :2]) < SEAM_STEP_TOL_ANG:
        return SeamVerdict(
"eclipsed", gap, z_room, *pair, period=period,
            message=("the layers across the boundary sit on the same sites, "
                     "so the stacking repeats where it should advance.  "
                     + _failing_condition(N, period) +
                     ".  Wrong for a periodic crystal, harmless in a "
                     "relaxation whose outer layers are frozen"))
    if period is not None \
            and _coincide(stack[N - period][:, :2], up[:, :2]) \
            < SEAM_STEP_TOL_ANG:
        return SeamVerdict(
"continues", gap, z_room, *pair, period=period)
    if N >= 2 and _coincide(sets[-2][:, :2], up[:, :2]) < SEAM_STEP_TOL_ANG:
        return SeamVerdict(
"twin", gap, z_room, *pair, period=period,
            message=("the boundary is a mirror twin: the bond length is right "
                     "but the stacking reverses there, so the crystal does "
                     "not continue.  " + _failing_condition(N, period) +
                     ".  Change the layer count or the starting registry"))
    return SeamVerdict(
"unknown", gap, z_room, *pair, period=period,
        message=("the layer across the boundary matches none of this slab's "
                 "own layers, so the boundary is neither a continuation, a "
                 "twin, nor an eclipse"))


#: Layers per stacking period, by fcc surface.  (111) is ABCABC, the
#: others ABAB -- so a seam only continues the crystal when the layer
#: count is a whole multiple (junction-cell.md § 3.1).
#:
#: Served to the Slab panel by /api/modify/meta, which is the whole
#: reason it is a table rather than a function: the note re-renders as the
#: user types a layer count, so the arithmetic happens client-side and only
#: the crystallography travels.  A plane absent from this table means "not
#: known" -- the note then says nothing instead of asserting a verdict.
STACKING_PERIOD = {"111": 3, "100": 2, "110": 2}
