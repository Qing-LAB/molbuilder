"""The frame-contract gate — structure-periodicity.md § 6.1 / § 6.2.

MODULE  periodicity_gate (L1; imports structure only)
ROLE    the ONE place periodicity state is defaulted, validated, and healed
USED-BY StructureCodec (load/save gate), web/blueprints/build.py (the
        unified periodicity door), tests/test_periodicity_gate.py

Contract (§ 6.1, decided 2026-07-29): the .xyz/.molstruct.json pair is the
only truth; resolved values are views and are never written back; ALL
healing happens here.  The § 6.1 heal table governs STORED state (load/
save); ``apply_edit`` below governs LIVE edits per the § 6.2 v3 regime
model:

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

Heal table (§ 6.1, stored state; right-handed cells only, det > 0):

  | stored state                | contained? | action                       |
  |-----------------------------|-----------|-------------------------------|
  | no cell, no origin          |     —     | derived; vacuum authoritative |
  | explicit cell, no origin    |    yes    | legal (imported crystal)      |
  | explicit cell, no origin    |    no     | heal origin -> expected corner|
  | explicit cell + origin      |    yes    | legal, user-owned, never healed|
  | explicit cell + origin      |    no     | stored pair -> heal + notice; |
  |                             |           | live edit -> accept + warn    |

``expected_corner`` respects the per-direction vacuum: ``bbox_min −
vacuum`` on isolated axes, ``bbox_min`` on transport, ``0`` on periodic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .structure import Structure

# Containment tolerance (Angstrom-scale in fractional projections).  Loose
# enough to forgive round-tripped floats, tight enough that "half the
# molecule outside the box" can never pass.
_EPS = 1e-6

#: The four ops the unified door accepts (§ 6.2 v3).  ``cell_origin`` with a
#: ``null`` payload is the "reset origin to default" button.
OPS = ("vacuum", "axis_kind", "cell", "cell_origin")


def _notice(level: str, message: str) -> Dict[str, str]:
    return {"level": level, "message": message}


def expected_corner(struct: Structure) -> np.ndarray:
    """The § 6.1 'correct corner': the low corner that wraps the structure
    honouring the per-direction vacuum on isolated axes."""
    lo = struct.positions.min(axis=0).astype(float)
    kinds = struct.axis_kind or ("isolated",) * 3
    out = np.zeros(3, dtype=float)
    for i, kind in enumerate(kinds):
        if kind == "isolated":
            out[i] = lo[i] - float(struct.vacuum[i])
        elif kind == "transport":
            out[i] = lo[i]
        else:                              # periodic: phase convention = 0
            out[i] = 0.0
    return out


def _fractional(struct: Structure, origin: np.ndarray) -> np.ndarray:
    """Fractional coordinates of every atom relative to (origin, cell).
    Triclinic-safe: solves cell.T @ frac = (pos - origin)."""
    rel = struct.positions.astype(float) - origin.reshape(1, 3)
    return np.linalg.solve(np.asarray(struct.cell, dtype=float).T, rel.T).T


def contains_atoms(struct: Structure,
                   origin: Optional[np.ndarray] = None) -> bool:
    """True when every atom sits inside ``[origin, origin + cell)`` along
    every NON-PERIODIC axis (with tolerance).  Along a periodic axis atoms
    outside the cell are legitimate periodic images (SIESTA wraps them), so
    containment is NEVER required there — requiring it made real crystals
    and junction files unopenable (review finding, 2026-07-29).
    ``origin=None`` means the world origin (imported-crystal semantics)."""
    if struct.cell is None or struct.n_atoms == 0:
        return True
    o = (np.zeros(3) if origin is None
         else np.asarray(origin, dtype=float).reshape(3))
    frac = _fractional(struct, o)
    kinds = struct.axis_kind or ("isolated",) * 3
    for i in range(3):
        if kinds[i] == "periodic":
            continue
        if not (np.all(frac[:, i] >= -_EPS)
                and np.all(frac[:, i] <= 1.0 + _EPS)):
            return False
    return True


def _too_small_axes(struct: Structure) -> List[int]:
    """Non-periodic axes whose FRACTIONAL extent exceeds 1 — the structure
    cannot fit along them for ANY origin choice.  Measured along the cell
    axes (triclinic-safe), never bbox-vs-row-norm."""
    if struct.cell is None or struct.n_atoms == 0:
        return []
    frac = _fractional(struct, np.zeros(3))
    kinds = struct.axis_kind or ("isolated",) * 3
    out = []
    for i in range(3):
        if kinds[i] == "periodic":
            continue
        if float(frac[:, i].max() - frac[:, i].min()) > 1.0 + 2 * _EPS:
            out.append(i)
    return out


def clearances(struct: Structure, origin: Optional[np.ndarray]) -> List[
        Tuple[float, float]]:
    """Per-axis (near, far) gaps in Angstrom between the structure and the
    box faces, measured along each cell axis (for the § 6.2 origin-edit
    warning).  Negative = atoms poke out that face."""
    o = (np.zeros(3) if origin is None
         else np.asarray(origin, dtype=float).reshape(3))
    frac = _fractional(struct, o)
    lens = np.linalg.norm(np.asarray(struct.cell, dtype=float), axis=1)
    out = []
    for i in range(3):
        out.append((float(frac[:, i].min() * lens[i]),
                    float((1.0 - frac[:, i].max()) * lens[i])))
    return out


def _clearance_text(struct: Structure, origin) -> str:
    gaps = clearances(struct, origin)
    return ", ".join(f"axis {i}: ({n:.2f}, {f:.2f})"
                     for i, (n, f) in enumerate(gaps))


def _require_right_handed(cell: np.ndarray) -> None:
    det = float(np.linalg.det(cell))
    if det <= 0.0:
        raise ValueError(
            f"cell must be right-handed (det > 0); got det = {det:.6g}. "
            f"Swap two lattice vectors or negate one (§ 6.1).")


def validate_and_heal(struct: Structure, *,
                      live_edit: bool = False) -> Tuple[Structure, List[dict]]:
    """Apply the § 6.1 heal table to STORED state.  Returns
    (possibly-corrected struct, notices).  ``live_edit=True`` = the
    explicit-origin manual-edit row: accept as typed, warn, never auto-fix.
    Raises ``ValueError`` for a left-handed cell or one too small to
    contain its atoms."""
    notices: List[dict] = []
    if struct.cell is None or struct.n_atoms == 0:
        return struct, notices                      # derived: nothing stored
    cell = np.asarray(struct.cell, dtype=float)
    _require_right_handed(cell)
    origin = struct.cell_origin
    if contains_atoms(struct, origin):
        return struct, notices                      # legal rows 2 / 4
    if origin is not None:
        # Row 5 (both halves): an EXPLICIT origin is user-owned — warn,
        # never overwrite.  (Healing a stored manual origin silently
        # flipped what the user typed on the next round-trip — review
        # finding, 2026-07-29.)
        notices.append(_notice(
            "warn",
            "the box does NOT contain the structure along a non-periodic "
            "axis — per-axis (near, far) clearances in Å: "
            + _clearance_text(struct, origin)
            + ". The manual origin is kept as typed (user-owned); vacuum "
            "values are not respected under a manual origin."))
        return struct, notices
    # Row 3: no origin stored — heal to the expected corner, if it can fit.
    bad = _too_small_axes(struct)
    if bad:
        raise ValueError(
            "explicit cell cannot contain the structure along "
            f"non-periodic axis(es) {bad} for ANY origin (fractional "
            "extent > 1); enlarge the cell along those axes or clear it "
            "to return to the derived box.")
    corner = expected_corner(struct)
    healed = struct.copy()
    healed.cell_origin = corner.copy()
    healed.__post_init__()
    if not contains_atoms(healed, corner):
        # The requested vacuum pushes atoms past the far face (cell fits
        # the structure but not structure + vacuum): centre instead.
        frac = _fractional(struct, np.zeros(3))
        lens = np.linalg.norm(cell, axis=1)
        lo = struct.positions.min(axis=0)
        kinds = struct.axis_kind or ("isolated",) * 3
        for i in range(3):
            if kinds[i] == "periodic":
                continue
            ext = float(frac[:, i].max() - frac[:, i].min()) * lens[i]
            corner[i] = lo[i] - max(0.0, (lens[i] - ext)) / 2.0
        healed.cell_origin = corner.copy()
        healed.__post_init__()
    heal_note = _notice(
        "warn",
        "healed: the explicit cell did not contain the structure, so "
        "cell_origin was set to the wrapping corner "
        f"{np.round(corner, 4).tolist()} (bbox_min − vacuum per isolated "
        "axis, centred where the per-side vacuum does not fit; § 6.1).")
    heal_note["kind"] = "heal"        # machine-readable: state was MODIFIED
    notices.append(heal_note)
    return healed, notices


def _reset_to_derived(s: Structure, what: str,
                      notices: List[dict]) -> None:
    """Shared § 6.2 v3 upstream-edit semantics: editing {vacuum,
    axis_kind} moves the box back to the DERIVED regime — explicit cell +
    origin cleared, boundary recomputed from the structure + vacuum.

    Refuses when the derived box would be DEGENERATE (a zero-extent axis
    with zero vacuum — e.g. a planar structure, or a transport axis whose
    captured device length the derived bbox cannot reproduce): resetting
    would ship a zero-volume lattice (review finding, 2026-07-29)."""
    if s.n_atoms:
        ext = s.positions.max(axis=0) - s.positions.min(axis=0)
        kinds = s.axis_kind or ("isolated",) * 3
        for i, kind in enumerate(kinds):
            pad = 2.0 * float(s.vacuum[i]) if kind == "isolated" else 0.0
            if float(ext[i]) + pad < 1e-6:
                raise ValueError(
                    f"cannot reset to the derived box: axis {i} would be "
                    f"degenerate (structure extent ~0 and no vacuum). Set "
                    f"a non-zero vacuum on that axis first, or keep an "
                    f"explicit cell.")
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

    if op == "vacuum":
        v = [float(x) for x in (payload or [])]
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("vacuum must be 3 non-negative floats (Å)")
        if "periodic" in kinds:
            raise ValueError(
                "cannot re-derive the box while an axis is periodic (a "
                "bounding box is not a lattice) — make the axis isolated "
                "first, or edit the cell explicitly (§ 6.2)")
        s.vacuum = tuple(v)
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
        _require_right_handed(cell)
        s.cell = cell
        if s.cell_origin is not None:
            # v3 precedence: respect the existing explicit ORIGIN first.
            s.__post_init__()
            if contains_atoms(s, s.cell_origin):
                notices.append(_notice(
                    "info",
                    "explicit cell set; the existing origin is respected. "
                    "Vacuum values are reference-only (§ 6.1)."))
            else:
                notices.append(_notice(
                    "warn",
                    "explicit cell set and the existing origin respected — "
                    "but the box does NOT contain the structure. Per-axis "
                    "(near, far) clearances in Å: "
                    + _clearance_text(s, s.cell_origin) + "."))
            return s, notices
        # ... then respect the VACUUM: anchor at the expected corner.
        # A cell the structure cannot fit for ANY origin is REFUSED, not
        # stored — a stored-but-invalid cell locked every later door
        # (review finding, 2026-07-29).
        bad = _too_small_axes(s)
        if bad:
            raise ValueError(
                "the new cell cannot contain the structure along "
                f"non-periodic axis(es) {bad} (fractional extent > 1); "
                "enlarge the cell along those axes.")
        s, heal_notes = validate_and_heal(s)     # anchors via the heal path
        notices.extend(heal_notes)
        notices.append(_notice(
            "info",
            "explicit cell set (origin first, then vacuum — § 6.2). "
            "Vacuum values are reference-only from now on."))
        return s, notices

    # op == "cell_origin"
    if s.cell is None:
        raise ValueError(
            "cell_origin is only meaningful with an explicit cell — "
            "the derived box computes its own corner (§ 3c)")
    if payload is None:
        # "Reset origin to default": literal cell_origin = None.  The view
        # falls back to imported-crystal semantics (world origin) until a
        # vacuum/periodicity edit re-derives the box or a new origin is set.
        s.cell_origin = None
        s.__post_init__()
        notices.append(_notice(
            "info",
            "cell_origin cleared. The box is drawn from the world origin "
            "(imported-crystal semantics) until you update vacuum / "
            "periodicity (re-derives the box) or set a new origin; the "
            "other parameters have their freedom back."))
        return s, notices
    origin = [float(x) for x in payload]
    if len(origin) != 3:
        raise ValueError("cell_origin must be 3 floats (Å)")
    s.cell_origin = np.asarray(origin, dtype=float)
    s.__post_init__()
    s, edit_notes = validate_and_heal(s, live_edit=True)
    notices.extend(edit_notes)
    if not edit_notes:
        notices.append(_notice(
            "warn",
            "cell_origin set. Vacuum values are NOT respected under a "
            "manual origin — only the unit-cell parameters are. Per-axis "
            "(near, far) clearances in Å: "
            + _clearance_text(s, s.cell_origin) + "."))
    return s, notices
