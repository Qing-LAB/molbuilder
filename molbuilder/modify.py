"""Structure-modification helpers for the Modify tab.

Public surface (all pure functions; each returns a new ``Structure``):

    delete_atoms(struct, indices)              -> Structure
    add_atom(struct, element, anchor_index, offset, *,
             atom_name=None, residue_name="MOD", residue_id=None)
                                                -> Structure
    orient_along_axis(struct, anchor_indices, axis="z", *,
                      angle=0.0, center="midpoint")
                                                -> Structure
    rotate_around_axis(struct, axis="z", angle=0.0, *, center="origin")
                                                -> Structure
    add_slab(struct, element, plane, size, *, start_registry=0,
             start_z=0.0, grow="+z", stacking="continue",
             orthogonal=False, offset=(0.0, 0.0),
                             lattice_constant=None)
                                                -> Structure

Every function preserves the per-atom metadata (``atom_names``,
``residue_ids``, ``residue_names``, ``chain_ids``) -- new atoms get
sensible defaults (atom_name = element symbol, a new residue id, residue
name "ELC" for electrode atoms, chain "A").  None of the functions mutate
their input.

Used by:
    * ``molbuilder modify`` CLI subcommand (in ``cli.py``)
    * ``/api/modify/op`` web endpoint (Modify tab in the Build UI)
    * Direct Python users building nanojunction geometries
      (Au-thiol-Au and similar) for transport-DFT runs.

Geometry conventions for the nanojunction workflow:

    * After ``orient_along_axis(struct, [a0, a1], "z")`` the atom at
      ``a0`` sits at the origin and ``a1`` is on +z.  This is the
      DFT-transport convention -- z is the transport direction; the
      two electrodes extend along ±z.
    * ``add_slab`` places from ABSOLUTE coordinates -- a stated
      ``start_z``, a growth direction and a starting registry -- and reads
      no selection at all.  **It is the only slab builder.**
    * ``add_electrode_slab`` was the other one, and it placed RELATIVE to a
      selection: the centroid of ``center_indices``, plus a
      ``contact_distance``, with ``side="-z"`` mirroring the slab.  That
      mirror is the accidental layer-order flip
      ``archive/2026-09-01-bench-and-junction-plan.md`` § 2.3 records, and the redesign
      set out to make it *unreachable rather than switchable* -- which it
      did for the browser on 2026-08-30, while this function kept it
      reachable from the CLI for two more days.  Deleted 2026-09-01
      (``archive/2026-09-01-modify-redesign-plan.md`` § 3.4b); ``--electrode`` is now the
      same convenience expressed over ``add_slab``, so the anchor-relative
      grammar survives and the placement logic does not.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cell as _cell
from .structure import Structure, copy_annotations, remap_annotations


# --------------------------------------------------------------------- #
#  Atom-level edits                                                     #
# --------------------------------------------------------------------- #


def _reindex_transport_metadata(
    struct: Structure, keep: Sequence[int],
) -> Tuple["dict[str, List[int]]", "dict"]:
    """Remap ``struct.regions`` and the extensible ``struct.annotations``
    channels after a slice/delete operation, dropping any index that fell off
    and renumbering survivors to their new 0-based position
    (``model/structure-annotations.md`` § 2.1).

    ONE pass over the label store -- reserved labels are in it and remap by the
    same rule.  Until 2026-07-31 frozen was a second store and needed its own
    line here, in lockstep with this one; a remap that forgot it silently
    constrained the wrong atoms.

    Used by ``delete_atoms`` (the only modify-op that changes the index space).
    Pure-passthrough ops (translate / rotate / orient) carry labels through
    verbatim without this.
    """
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_regions = {}
    for label, idxs in struct.regions.items():
        remapped = [old_to_new[i] for i in idxs if i in old_to_new]
        if remapped:
            new_regions[label] = remapped
    new_annotations = remap_annotations(struct.annotations, old_to_new)
    return new_regions, new_annotations


def delete_atoms(struct: Structure, indices: Sequence[int]) -> Structure:
    """Return a new ``Structure`` with the given atom indices removed.

    Indices may be in any order; duplicates are tolerated.  All per-atom
    metadata arrays are sliced consistently.  ``frozen_atoms`` and
    ``regions`` are reindexed to the post-delete atom-numbering: a
    frozen atom that survives keeps its frozen flag but at a possibly
    smaller index; a frozen atom that was deleted is dropped.
    """
    keep = sorted(set(range(struct.n_atoms)) - set(int(i) for i in indices))
    if len(keep) == struct.n_atoms:
        # No-op (or all indices were out of range / already absent).
        return struct.copy()
    new_regions, new_annotations = _reindex_transport_metadata(
        struct, keep)
    return Structure(
        elements=     [struct.elements[i]      for i in keep],
        positions=    struct.positions[keep].copy(),
        atom_names=   [struct.atom_names[i]    for i in keep],
        residue_ids=  [struct.residue_ids[i]   for i in keep],
        residue_names=[struct.residue_names[i] for i in keep],
        chain_ids=    [struct.chain_ids[i]     for i in keep],
        title=struct.title,
        regions=new_regions,
        annotations=new_annotations,
        **struct._carry_periodicity(),   # deleting atoms keeps the lattice
    )


def add_atom(
    struct: Structure,
    element: str,
    anchor_index: int,
    offset: Sequence[float],
    *,
    atom_name: Optional[str] = None,
    residue_name: str = "MOD",
    residue_id: Optional[int] = None,
) -> Structure:
    """Return a new ``Structure`` with one atom appended.

    The new atom's position is ``struct.positions[anchor_index] + offset``.

    By default a fresh residue id is allocated (``max(residue_ids) + 1``)
    so the added atom isn't lumped into the anchor's residue -- handy
    when the user wants to delete just-added atoms in one shot, and
    matches the way molbuilder treats added thiol caps / explicit
    hydrogens.  Pass ``residue_id`` explicitly to land the new atom in
    a specific residue (SP-E) -- e.g. building a polyatomic side-chain
    cap (-COOH = 4 atoms) where all four atoms share one residue id:

        s = add_atom(s, "C", anchor_index=ag, offset=(1.5, 0, 0))
        rid = s.residue_ids[-1]
        s = add_atom(s, "O", anchor_index=s.n_atoms - 1,
                     offset=(0.6, 1.0, 0), residue_id=rid)
        s = add_atom(s, "O", anchor_index=s.n_atoms - 2,
                     offset=(0.6, -1.0, 0), residue_id=rid)
        s = add_atom(s, "H", anchor_index=s.n_atoms - 1,
                     offset=(0.6, -1.0, 0), residue_id=rid)

    Parameters
    ----------
    struct
        The structure to extend.
    element
        Chemical symbol of the new atom ("H", "Au", ...).
    anchor_index
        Atom whose position the offset is measured from.
    offset
        ``(dx, dy, dz)`` in Angstroms.
    atom_name
        PDB-style atom name; default = ``element``.
    residue_name
        Residue name for the new atom; default ``"MOD"``.
    residue_id
        If provided, place the new atom in this existing residue.
        Default ``None`` -> allocate a fresh residue id.
    """
    if not (0 <= anchor_index < struct.n_atoms):
        raise IndexError(
            f"anchor_index {anchor_index} out of range for "
            f"{struct.n_atoms}-atom structure"
        )
    # Scientific guard: the element must be a real periodic-table symbol.
    # Without this a typo ("Xx") or a mis-cased symbol ("AU") rides silently
    # into the Structure and only detonates much later in the SIESTA/PySCF
    # emitters (KeyError on ase.data.atomic_numbers).  Canonicalise to
    # Element-case first so "au" / "AU" become "Au" rather than being rejected.
    from ase.data import atomic_numbers as _atomic_numbers
    element = str(element).strip().capitalize()
    if element not in _atomic_numbers:
        raise ValueError(
            f"unknown element symbol {element!r}; expected a periodic-table "
            f"symbol like 'H', 'C', or 'Au'"
        )
    offset_arr = np.asarray(offset, dtype=float).reshape(3)
    # ADVISORY-NOT-ENFORCING (validation contract): a (near-)zero offset places
    # the new atom on top of the anchor.  We do NOT block it -- close / coincident
    # atoms are surfaced non-blockingly by ``validate_geometry`` (the
    # ``geometry.min_distance`` finding) while editing, and the GENERATION gate
    # (``report(validate(...))`` in the SIESTA/PySCF emitters) enforces at emit
    # time.  So the user can build an intermediate structure and fix it later.
    # Only genuinely-invalid input (a non-element symbol, above) is rejected here.
    new_pos = struct.positions[anchor_index] + offset_arr
    if residue_id is None:
        new_residue_id = (max(struct.residue_ids) if struct.residue_ids else 0) + 1
    else:
        new_residue_id = int(residue_id)
    # New atom inherits a fresh index (= old n_atoms) and is NOT frozen
    # by default + NOT a member of any region.  Existing frozen_atoms +
    # region indices carry through unchanged: their atom-index space
    # only grows at the high end, so no remap needed.
    return Structure(
        elements=struct.elements + [element],
        positions=np.vstack([struct.positions, new_pos[None, :]]),
        atom_names=struct.atom_names + [atom_name or element],
        residue_ids=struct.residue_ids + [new_residue_id],
        residue_names=struct.residue_names + [residue_name],
        chain_ids=struct.chain_ids + [struct.chain_ids[anchor_index]],
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        annotations=copy_annotations(struct.annotations),
        **struct._carry_periodicity(),   # appending an atom keeps the lattice
    )


# --------------------------------------------------------------------- #
#  Whole-structure orientation                                          #
# --------------------------------------------------------------------- #


_AXES = ("x", "y", "z")


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation that takes unit vector ``a`` to unit ``b``
    (Rodrigues' formula).  Both inputs must be unit-length and 3-D.

    Handles the antiparallel edge case (a = -b) by rotating 180 degrees
    around any axis perpendicular to ``a``.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        # a and b are already aligned.
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # Antiparallel: rotate 180 around any axis perp to a.
        # Pick the axis whose dot with a is smallest, project out a.
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        # 180-degree rotation around `axis` via Rodrigues with theta=pi:
        #   R = 2*outer(axis, axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _moved_subset(struct: Structure, R, t, indices: Sequence[int]) -> Structure:
    """A copy of ``struct`` with ONLY ``indices`` mapped by ``x -> x @ Rᵀ + t``.

    THE BOX IS NOT TOUCHED, and that is the whole difference from
    :meth:`Structure.affine`.  Moving part of a structure moves those atoms
    RELATIVE to the ones that stayed; a lattice that followed them would stop
    describing the atoms it was drawn around.  A rigid whole-structure move is
    the other operation -- the box goes with the atoms because nothing moved
    relative to anything else.  Two operations, not one with a flag.

    Labels and channels are index-keyed and the index space is UNCHANGED here
    (no atom is added or removed), so they carry through untouched: the atom at
    index i before is the atom at index i after.
    """
    keep = sorted({int(i) for i in indices})
    n = struct.n_atoms
    for i in keep:
        if not 0 <= i < n:
            raise ValueError(
                f"atom index {i} out of range [0, {n}) for a {n}-atom structure")
    out = struct.copy()
    if keep:
        pos = out.positions
        pos[keep] = pos[keep] @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)
    return out


def translate(struct: Structure, vec: Sequence[float], *,
              indices: "Sequence[int] | None" = None) -> Structure:
    """Translate ``struct`` by ``vec`` (Angstrom).

    With ``indices``, ONLY those atoms move and the box stays where it is.
    Without, the whole structure moves rigidly and the box's world-space corner
    goes with it (``Structure.translated`` -> ``affine``).
    """
    vec = np.asarray(vec, dtype=float).reshape(3)
    if indices is None:
        return struct.translated(vec)
    return _moved_subset(struct, np.eye(3), vec, indices)


def orient_along_axis(
    struct: Structure,
    anchor_indices: Tuple[int, int],
    axis: str = "z",
    *,
    angle: float = 0.0,
    center: str = "midpoint",
) -> Structure:
    """Return a new ``Structure`` rotated so the vector from the first
    anchor atom to the second forms angle ``angle`` (degrees) with the
    given target axis.

    Parameters
    ----------
    anchor_indices
        ``(a0, a1)`` -- two distinct atom indices.  The vector
        ``positions[a1] - positions[a0]`` is rotated to point at angle
        ``angle`` from the +``axis`` direction.
    axis
        Target axis: "x", "y", or "z" (default).
    angle
        Tilt angle in **degrees** between the anchor pair vector and
        the target axis after rotation.  Default ``0.0`` puts the
        anchor pair exactly along the target axis (the canonical
        molecular-junction case).  Non-zero values tilt the molecule
        by that many degrees in a fixed default plane:

          * ``axis="z"`` -> tilt happens in the **xz-plane** (anchor
            pair vector becomes ``(sin θ * d, 0, cos θ * d)``);
          * ``axis="x"`` -> tilt in the **xy-plane**
            (``(cos θ * d, sin θ * d, 0)``);
          * ``axis="y"`` -> tilt in the **yz-plane**
            (``(0, cos θ * d, sin θ * d)``).

        For any other tilt direction, follow this call with
        :func:`rotate_around_axis` to spin the tilted molecule around
        the target axis.
    center
        How to translate the structure after rotation:

        * ``"midpoint"`` (default): place the midpoint of ``a0`` and
          ``a1`` at the origin.  This is what pair-mode electrode
          construction relies on -- the gap is centred on the
          molecule's anchor-pair midpoint.
        * ``"first"``: place ``a0`` at the origin.  The anchor pair
          extends from origin to ``(angle-rotated unit vector) * |d|``.
        * ``"none"``: leave translation unchanged after rotation; only
          rotate.  Use when the caller will translate explicitly.
    """
    if axis not in _AXES:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
    if center not in ("first", "midpoint", "none"):
        raise ValueError(
            f"center must be 'first', 'midpoint', or 'none'; got {center!r}"
        )
    a0, a1 = int(anchor_indices[0]), int(anchor_indices[1])
    if a0 == a1:
        raise ValueError("anchor_indices must be two distinct atom indices")
    for i in (a0, a1):
        if not (0 <= i < struct.n_atoms):
            raise IndexError(
                f"anchor index {i} out of range for {struct.n_atoms}-atom structure"
            )

    p0 = struct.positions[a0]
    p1 = struct.positions[a1]
    direction = p1 - p0
    length = np.linalg.norm(direction)
    if length < 1e-9:
        raise ValueError(
            "anchor atoms are coincident; cannot define an orientation axis"
        )

    # Build the target unit vector, accounting for the requested tilt.
    theta_rad = np.radians(float(angle))
    sin_t, cos_t = np.sin(theta_rad), np.cos(theta_rad)
    if axis == "z":
        target = np.array([sin_t, 0.0, cos_t])
    elif axis == "x":
        target = np.array([cos_t, sin_t, 0.0])
    else:  # axis == "y"
        target = np.array([0.0, cos_t, sin_t])

    R = _rotation_matrix_from_a_to_b(direction, target)
    # The post-rotation recentre is a translation of the ROTATED frame: for "first"
    # the rotated anchor a0 lands at the origin; for "midpoint" the rotated anchor
    # midpoint does.  Express the whole op as the affine ``x -> x @ Rᵀ + t``.
    if center == "first":
        t = -(struct.positions[a0] @ R.T)
    elif center == "midpoint":
        t = -0.5 * ((struct.positions[a0] + struct.positions[a1]) @ R.T)
    else:  # "none"
        t = np.zeros(3)
    # orient is ALWAYS a whole-structure rotation (the anchors only DEFINE the
    # rotation; every atom moves), so the unit-cell box rotates WITH the atoms --
    # lattice vectors + the cell_origin corner (structure-periodicity.md § 6),
    # via the ONE affine primitive.  (A partial-selection edit never reaches orient:
    # its role is "anchor", so applyOp always takes the whole-structure path.)
    return struct.affine(R, t)


def rotate_around_axis(
    struct: Structure,
    axis: str = "z",
    angle: float = 0.0,
    *,
    center: str = "origin",
    indices: "Sequence[int] | None" = None,
) -> Structure:
    """Rotate every atom by ``angle`` degrees around the named axis.

    Useful after :func:`orient_along_axis` to spin a tilted molecule
    around the transport axis to a different azimuth -- e.g. to tilt
    the molecule in the yz-plane instead of the default xz-plane.

    Parameters
    ----------
    axis
        Rotation axis: "x", "y", or "z" (default).
    angle
        Rotation angle in **degrees**.  Right-hand rule (positive
        angle = counter-clockwise when looking down the axis from +
        toward origin).  Default ``0.0`` is a no-op.
    center
        Pivot point for the rotation:

        * ``"origin"`` (default) -- the rotation axis passes through
          the world origin.  This is non-commutative with
          ``add_atom`` / ``translate`` ops that leave the molecule
          off-origin: the molecule then swings on a wide arc rather
          than rotating in place.  Use when you want a global
          coordinate-system rotation (e.g. immediately after
          ``orient_along_axis(center="midpoint")`` which already
          placed the anchor pair at the origin).
        * ``"centroid"`` -- the rotation axis passes through the
          atom-coordinate mean.  The molecule rotates in place
          regardless of its current global position.  Use for the
          typical "spin this molecule by N degrees" intent.
    """
    if axis not in _AXES:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
    if center not in ("origin", "centroid"):
        raise ValueError(
            f"center must be 'origin' or 'centroid'; got {center!r}"
        )
    theta_rad = np.radians(float(angle))
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    if axis == "z":
        R = np.array([[ c, -s, 0.0],
                      [ s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
    elif axis == "x":
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,  c,  -s],
                      [0.0,  s,   c]])
    else:  # y
        R = np.array([[ c, 0.0,  s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0,  c]])
    # Rotation about a pivot p is the affine ``x -> (x - p) @ Rᵀ + p`` = ``x @ Rᵀ + t``
    # with ``t = p - p @ Rᵀ``.
    #
    # WHOLE STRUCTURE: routed through ``Structure.affine`` so the whole box --
    # lattice VECTORS (cell) AND the world-space CORNER (cell_origin) -- rotates
    # WITH the atoms (structure-periodicity.md § 6); nothing moved relative to
    # anything else, and a box left behind would stop wrapping the atoms.
    #
    # A SUBSET: only those atoms turn, and the box stays.  The pivot is the
    # SELECTION's centroid, because "spin this piece in place" is what a partial
    # rotate means.  (Until 2026-07-31 the browser got this by shipping the
    # selected atoms as their own cell-less document and mapping the answer back
    # -- the second of § 11.7's two exceptions.  The route takes the atoms now,
    # so the whole structure goes out and the whole structure comes back.)
    if indices is not None:
        subset = sorted({int(i) for i in indices})
        pivot = (struct.positions[subset].mean(axis=0) if center == "centroid"
                 else np.zeros(3))
        return _moved_subset(struct, R, pivot - pivot @ R.T, subset)
    if center == "centroid":
        pivot = struct.positions.mean(axis=0)
        t = pivot - pivot @ R.T
    else:  # origin
        t = np.zeros(3)
    return struct.affine(R, t)


# --------------------------------------------------------------------- #
#  Crystal electrode layers                                             #
# --------------------------------------------------------------------- #


# Supported FCC metals for electrode construction.
#
# The actual lattice-constant values live in ``molbuilder/data/fcc_lattice.json``
# (with the citation chain in ``molbuilder/data/README.md``); this module
# loads them at import time.  Keeping the values out of source code means a
# user can edit the JSON to override a constant (or pass an in-tree DFT-
# equilibrium value) without touching code.
#
# The list is **deliberately closed** to the canonical metal-electrode
# materials used in single-molecule junction / NEGF transport DFT work.
# An open list would silently accept non-FCC elements (e.g. Fe / Cr would
# get geometrically nonsensical "FCC" slabs) -- by restricting at the API
# boundary we make the wrong-symmetry case a clear ``ValueError`` instead.
#
# Future-extension note: when a BCC / HCP electrode is needed, add a
# parallel data file (``bcc_lattice.json`` / ``hcp_lattice.json``) and a
# sibling ``add_slab_bcc`` / ``..._hcp`` function rather than
# overloading this one.  Same closed-list rule applies.


def _data_dir_candidates() -> List[_Path]:
    """Search order for fundamental-data files.

    1. ``$MOLBUILDER_DATA_DIR`` if set (user override that survives
       ``pip install --upgrade``).  See ``molbuilder/data/README.md``.
    2. The packaged ``molbuilder/data/`` directory.
    """
    candidates: List[_Path] = []
    env = _os.environ.get("MOLBUILDER_DATA_DIR")
    if env:
        candidates.append(_Path(env))
    candidates.append(_Path(__file__).parent / "data")
    return candidates


def load_fcc_lattice_full() -> dict:
    """Load the full FCC lattice-constant table from ``fcc_lattice.json``.

    Each metal carries ``a_experimental`` (Wyckoff 1963) and ``a_pbe``
    (Haas-Tran-Blaha 2009) -- the two LITERATURE references, which is what
    a shared table is for.

    Returns the metals dict directly: ``{symbol: {a_experimental: float,
    a_pbe: float, name: str, system: str}}``.

    **v3 (2026-08-30) dropped ``a_pbe_siesta_psml``.**  It was null for
    every metal and nothing in the codebase could write it: its only homes
    were the packaged file and a machine-wide ``MOLBUILDER_DATA_DIR``
    override, so the "Your bulk run" control it fed greyed itself out --
    correctly -- from the day it shipped.  A lattice constant measured in
    the user's own SIESTA+PSML setup belongs to ONE optimization run, not
    to a table every project shares, so it is read from that run's result
    instead (``POST /api/modify/lattice-from-run``).

    **v2 files still load**, carrying a column this returns nothing for:
    a user's overriding data dir must not stop working because a column
    they never filled went away.  v1 ("a" only) still raises.
    """
    last_error: Optional[Exception] = None
    for candidate_dir in _data_dir_candidates():
        path = candidate_dir / "fcc_lattice.json"
        if not path.is_file():
            continue
        try:
            with open(path) as fh:
                data = _json.load(fh)
        except (_json.JSONDecodeError, OSError) as exc:
            last_error = RuntimeError(
                f"failed to read FCC lattice table at {path!s}: {exc}"
            )
            continue
        if not isinstance(data, dict) or "metals" not in data:
            last_error = RuntimeError(
                f"FCC lattice table at {path!s} missing required 'metals' key"
            )
            continue
        fmt = data.get("_format", "")
        if not ("v2" in fmt or "v3" in fmt):
            raise RuntimeError(
                f"FCC lattice table at {path!s} is neither v2 nor v3 (got "
                f"{fmt!r}).  Each metal must carry a_experimental and "
                f"a_pbe; the v1 'a'-only schema is no longer supported."
            )
        metals: dict = {}
        for sym, entry in data["metals"].items():
            try:
                a_exp = float(entry["a_experimental"])
                a_pbe = float(entry["a_pbe"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"FCC lattice entry {sym!r} in {path!s} is malformed: {exc}"
                ) from exc
            metals[sym] = {
                "a_experimental": a_exp,
                "a_pbe":          a_pbe,
                "name":           entry.get("name", sym),
                "system":         entry.get("system", "fcc"),
            }
        if not metals:
            raise RuntimeError(
                f"FCC lattice table at {path!s} contains zero entries"
            )
        return metals
    raise RuntimeError(
        f"could not locate fcc_lattice.json under any of: "
        f"{[str(p) for p in _data_dir_candidates()]}.  "
        f"Last error: {last_error}"
    )


def _load_fcc_lattice() -> dict:
    """Back-compat shim: return ``{symbol: a_experimental_float}``.

    This was the v1 loader's return shape, kept here so callers that
    only need "the default experimental lattice constant" don't have
    to know about v2's per-XC fields.  Callers that need to pick
    between experimental / PBE / user-measured values should hit
    ``load_fcc_lattice_full`` directly (the web meta endpoint does).
    """
    full = load_fcc_lattice_full()
    return {sym: entry["a_experimental"] for sym, entry in full.items()}


# Closed list of supported FCC electrode metals.  Hardcoded (rather
# than derived from ``fcc_lattice.json``) so a missing or
# misconfigured ``MOLBUILDER_DATA_DIR`` doesn't crash module import
# (SP-C).  The JSON file holds the lattice CONSTANTS for these
# elements; membership in the closed list is a design decision
# (Au / Ag / Cu / Ni / Pt / Pd are the typical molecular-electronics
# choices) and lives here.  Adding a new metal is a two-step change:
# update this tuple AND extend the JSON.
SUPPORTED_FCC_ELEMENTS: Tuple[str, ...] = ("Au", "Ag", "Cu", "Ni", "Pt", "Pd")

SUPPORTED_FCC_PLANES: Tuple[str, ...] = ("100", "110", "111")

#: Which values of ``orthogonal`` each fcc surface can actually be built with
#: (science/junction-cell.md § 2b).
#:
#: ASE implements a non-orthogonal cell for fcc(111) only; asking for one on
#: (100) or (110) raises ``NotImplementedError``.  So the flag is a real
#: choice on ONE surface, and a UI that offers it as a free checkbox on all
#: three lets the user pick a combination that cannot exist -- which is what
#: it did: the box starts unchecked, and unchecked is the one setting a (100)
#: slab cannot be built with, so the default request came back a 400.
#:
#: What is NOT here: fcc(111)'s orthogonal cell additionally needs an even
#: ``n``.  That depends on the size rather than the surface, so it stays
#: passed through to ASE and surfaced verbatim (`_build_ase_slab`) instead of
#: being restated -- one door per fact.
#:
#: Hardcoded for the same reason ``SUPPORTED_FCC_ELEMENTS`` is, and with a
#: guard the elements list does not have:
#: ``tests/test_fcc_cell_shapes.py`` builds every combination and asserts ASE
#: agrees, so this cannot drift from the library it describes.
FCC_ORTHOGONAL_CHOICES: Dict[str, Tuple[bool, ...]] = {
    "100": (True,),
    "110": (True,),
    "111": (False, True),
}

# Lattice-constant table is loaded lazily on first call to
# :func:`_get_fcc_lattice` and cached for the rest of the process.
# Lazy loading lets ``import molbuilder.modify`` succeed even when the
# data file is missing -- only operations that actually need a lattice
# constant (build_electrode_slab, etc.) surface the error, and only
# at the moment they need it.
_FCC_LATTICE_A_CACHE: Optional[dict] = None


def _get_fcc_lattice() -> dict:
    """Return the FCC lattice-constant dict, loading on first call."""
    global _FCC_LATTICE_A_CACHE
    if _FCC_LATTICE_A_CACHE is None:
        _FCC_LATTICE_A_CACHE = _load_fcc_lattice()
    return _FCC_LATTICE_A_CACHE


# Metal-element-aware default contact distances (Å) loaded from
# ``data/contact_distance.json``.  **Read by nobody in the package since
# 2026-09-01**, when `add_electrode_slab` -- the one caller -- was deleted:
# `add_slab` takes an absolute `start_z`, so no builder asks "how far from
# the molecule should this metal sit" any more.  The TABLE is kept because
# it is measured physics a person needs when typing `--electrode @contact=`,
# and `default_contact_distance` is the published way to ask; wiring it in
# as that flag's default is a decision, not a cleanup.  Au-S
# canonical 2.40 Å is wrong for Pt-N (2.05) or Ag-S (2.50), so the
# element-aware default is a real win over the previous hardcoded
# 2.4 default.  Lazy-load + cache pattern matches the FCC lattice
# table above.
_CONTACT_DISTANCE_CACHE: Optional[dict] = None


def _load_contact_distance() -> dict:
    """Load the per-metal default contact-distance table from
    ``contact_distance.json``.  Returns ``{element_symbol: float_A}``.
    """
    last_error: Optional[Exception] = None
    for candidate_dir in _data_dir_candidates():
        path = candidate_dir / "contact_distance.json"
        if not path.is_file():
            continue
        try:
            with open(path) as fh:
                data = _json.load(fh)
        except (_json.JSONDecodeError, OSError) as exc:
            last_error = RuntimeError(
                f"failed to read contact-distance table at {path!s}: {exc}"
            )
            continue
        if not isinstance(data, dict) or "metals" not in data:
            last_error = RuntimeError(
                f"contact-distance table at {path!s} missing 'metals' key"
            )
            continue
        out: dict = {}
        for sym, entry in data["metals"].items():
            try:
                out[sym] = float(entry["d"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"contact-distance entry {sym!r} in {path!s} "
                    f"is malformed: {exc}"
                ) from exc
        if not out:
            raise RuntimeError(
                f"contact-distance table at {path!s} contains zero entries"
            )
        return out
    raise RuntimeError(
        f"could not locate contact_distance.json under any of: "
        f"{[str(p) for p in _data_dir_candidates()]}.  "
        f"Last error: {last_error}"
    )


def _get_contact_distance() -> dict:
    """Return the per-metal default contact-distance dict, loading on
    first call.  Cached for the rest of the process."""
    global _CONTACT_DISTANCE_CACHE
    if _CONTACT_DISTANCE_CACHE is None:
        _CONTACT_DISTANCE_CACHE = _load_contact_distance()
    return _CONTACT_DISTANCE_CACHE


def default_contact_distance(element: str) -> float:
    """Element-aware default contact distance (Å) for a metal anchor.

    Falls back to 2.4 (canonical Au-S) when the element isn't in the
    table -- preserves backward compatibility with callers that
    relied on the old hardcoded default for unsupported metals.
    """
    table = _get_contact_distance()
    return table.get(element, 2.4)


def _check_fcc_element(element: str) -> None:
    """Raise ``ValueError`` if the requested element isn't on the
    supported list.  The list is closed (see ``SUPPORTED_FCC_ELEMENTS``)
    so the UI dropdown and the Python API stay in lockstep.
    """
    if element not in SUPPORTED_FCC_ELEMENTS:
        raise ValueError(
            f"unsupported electrode element {element!r}; "
            f"supported FCC metals are: "
            f"{', '.join(SUPPORTED_FCC_ELEMENTS)}.  "
            f"The builder is fcc-only: a bcc or hcp metal has a different "
            f"stacking period and registry, so it is a second set of seam "
            f"rules rather than a table entry "
            f"(docs/science/junction-cell.md § 2a)."
        )


def _ase_slab_builder(plane: str):
    """Resolve plane string to ASE's slab builder.  Imported lazily so
    `import molbuilder.modify` doesn't pull ASE for callers that only
    use ``delete_atoms`` / ``add_atom`` / ``orient_along_axis``.

    The builders are the canonical, well-tested ASE functions for FCC
    surfaces; we lean on them rather than re-implementing the lattice
    geometry (principle #8 in ``docs/design.md``: "Don't reinvent
    wheels").  Per-(plane, orthogonal) compatibility is enforced by
    :func:`_validate_orthogonal_compat` before the builder is called.
    """
    from ase.build import fcc100, fcc110, fcc111
    builders = {"100": fcc100, "110": fcc110, "111": fcc111}
    if plane not in builders:
        raise ValueError(
            f"unsupported crystal plane {plane!r}; expected '100', '110', or '111'"
        )
    return builders[plane]


def _build_ase_slab(element: str, plane: str, size: Tuple[int, int, int],
                    orthogonal: bool, a: float):
    """Call ASE's ``fcc{100,110,111}`` builder and pass through any
    error message verbatim, prefixed with our operation context.

    ASE's own messages already explain the constraint clearly
    (e.g. *"Can't make orthorhombic cell with size=(3, 3, 1).
    Second number in size must be even"*); we just add the molbuilder
    context (which element / plane / orthogonal mode triggered it)
    and re-raise as ``ValueError`` so callers (and the future Modify
    UI) can display the message inline as a hint.

    See ``science/junction-cell.md`` § 2b for the per-(plane, orthogonal)
    compatibility table (``FCC_ORTHOGONAL_CHOICES`` above) -- determined
    empirically, but enforced by passing through to ASE rather than
    re-implementing the rules.  The size-dependent half of the constraint
    (fcc(111) orthogonal needs an even ``n``) is deliberately NOT in that
    table and reaches the user only through this pass-through.
    """
    builder = _ase_slab_builder(plane)
    try:
        return builder(element, size=size, orthogonal=orthogonal, a=a)
    except (ValueError, AssertionError, NotImplementedError) as exc:
        raise ValueError(
            f"ASE rejected the {element} fcc({plane}) slab with "
            f"size={size}, orthogonal={orthogonal}: {exc}.  "
            f"Adjust (m, n) to satisfy ASE's constraint, or flip "
            f"the orthogonal switch (see docs/web/tabs.md)."
        ) from exc


# --------------------------------------------------------------------- #
#  The new slab builder (archive/2026-09-01-modify-redesign-plan.md § 3)              #
# --------------------------------------------------------------------- #


def _finish_slab(struct, metal_pos, element, full):
    """Append placed metal atoms and capture the box they imply.

    EXTRACTED 2026-08-30 so two builders could share it rather than each
    carrying a copy.  There is one builder now -- ``add_slab`` -- and this
    is still its own function, because what it does (the metadata, the
    cell, its origin, the axis kinds) is a different question from WHERE
    the slab goes, and mixing the two is what made the placement bug in the
    builder this outlived hard to see.

    **It took two more parameters until 2026-09-01**, and both existed only
    for ``add_electrode_slab``: ``pad_interlayer_gap``, which added one
    layer spacing to the captured ``c``, and ``d_interlayer``, the spacing
    to use when a monolayer had none to measure.  ``junction-cell.md`` § 6
    retired that padding on the user's decision -- **`c` is measured and
    set, never invented** -- and ``add_slab`` had already been passing
    ``False``.  With the old builder gone the flag had one value, so it and
    the number it guarded went with it.
    """
    # Assemble metadata for the new metal atoms.
    n_new = metal_pos.shape[0]
    new_residue_id = (max(struct.residue_ids) if struct.residue_ids else 0) + 1

    # Capture the electrode's cell (structure-periodicity.md § 4 -- fixes the old
    # discard).  In-plane (x,y) = the ASE slab's lattice (rows 0,1; hexagonal for
    # fcc(111)).  axis_kind = (periodic, periodic, transport): the transport z is
    # electrode-matched, never tiled/k-sampled.  Overwrites any prior cell -- the
    # electrode defines the junction's in-plane periodicity.  Skipped if the z
    # extent is degenerate (would make a singular cell); the caller can set one.
    #
    # z is the atoms' extent PLUS ONE INTERLAYER SPACING (science/junction-cell.md
    # § 1).  The extent alone puts the bottom atom's image at z_min + c = z_max --
    # exactly on the top atom, at zero distance -- and SIESTA stops.  The spacing
    # comes from cell.bulk_z_period, the same derivation the electrode wizard uses
    # for the bulk lead (§ 5), measured on the slab AS BUILT so an
    # ``inter_layer_offset`` override is honoured without being passed in.
    all_pos = np.vstack([struct.positions, metal_pos])
    z_extent = float(all_pos[:, 2].max() - all_pos[:, 2].min())
    slab_cell = np.asarray(full.get_cell(), dtype=float)
    elc_cell = None
    elc_axis_kind = None
    elc_cell_origin = None
    if z_extent > 1e-6:
        # `c` IS THE ATOMS' EXTENT, VERBATIM.  A block here used to add one
        # layer spacing to it; `junction-cell.md` § 6 retired that on the
        # user's decision and the caller had already stopped asking for it.
        z_len = z_extent
        elc_cell = np.array([
            [slab_cell[0, 0], slab_cell[0, 1], 0.0],
            [slab_cell[1, 0], slab_cell[1, 1], 0.0],
            [0.0, 0.0, z_len],
        ], dtype=float)
        elc_axis_kind = ("periodic", "periodic", "transport")
        # cell_origin (structure-periodicity.md § 6): the captured cell is built
        # AROUND atoms that straddle the origin (the molecule stays pinned there;
        # the slabs sit at +/- gap/2).  Anchor the cell at the structure's LOW
        # CORNER so the box WRAPS the atoms WITHOUT moving them -- z runs
        # [z_min, z_min + z_len), so the padding opens at the TOP, which is where
        # the two faces meet.  render_fdf then shifts atoms by -cell_origin into
        # [0, cell) for SIESTA; the `calibrate` op bakes that shift when the user
        # wants it in the stored coords.
        elc_cell_origin = all_pos.min(axis=0).astype(float)

    # New electrode atoms are appended at indices [old_n, old_n + n_new).
    # Existing frozen_atoms + region indices carry through unchanged; the
    # new electrode atoms are NOT auto-frozen and NOT auto-tagged with a
    # region label (callers who want either can post-process the result).
    return Structure(
        elements=list(struct.elements) + [element] * n_new,
        positions=np.vstack([struct.positions, metal_pos]),
        cell=elc_cell,
        cell_origin=elc_cell_origin,
        axis_kind=elc_axis_kind,
        atom_names=list(struct.atom_names) + [element] * n_new,
        residue_ids=list(struct.residue_ids) + [new_residue_id] * n_new,
        residue_names=list(struct.residue_names) + ["ELC"] * n_new,
        chain_ids=list(struct.chain_ids) + ["A"] * n_new,
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        annotations=copy_annotations(struct.annotations),
    )


def add_slab(
    struct: Structure,
    element: str,
    plane: str,
    size: Tuple[int, int, int],
    *,
    start_registry: int = 0,
    start_z: float = 0.0,
    grow: str = "+z",
    stacking: str = "continue",
    orthogonal: bool = False,
    offset: Tuple[float, float] = (0.0, 0.0),
    lattice_constant: Optional[float] = None,
) -> Structure:
    """Append ONE fcc slab, placed absolutely (redesign plan § 3).

    Everything is stated; nothing is inferred from a selection.  ``offset``
    and ``start_z`` are measured from the WORLD ORIGIN -- the origin of the
    3-D window's own coordinate system -- so the same numbers place the same
    slab whatever is currently picked.

    Parameters that are not ``add_electrode_slab``'s
    ------------------------------------------------
    start_registry
        Which stacking registry the layer AT ``start_z`` sits on, as an
        index: 0=A, 1=B, 2=C.  Taken modulo the surface's period
        (``cell.STACKING_PERIOD``), so (100) and (110) have two choices and
        (111) three -- *"if available"* falls out of the period rather than
        needing a table.
    start_z
        The z of that starting layer, in Angstrom, absolute.
    grow
        ``"+z"`` or ``"-z"`` -- which way the remaining layers go from the
        starting one.
    stacking
        What the registry does when growing DOWNWARD.  ``"continue"`` walks
        it backwards with the growth direction, so the layers below A are
        C then B -- what a real fcc crystal has below an A layer.
        ``"mirror"`` walks it forwards regardless, which is the same slab
        flipped in z.

        **Both are real fcc**: the lattice is centrosymmetric, so a mirrored
        slab is a perfectly good crystal.  They differ only where two slabs
        MEET -- grown apart from A, ``continue`` gives ``...B C A | A B C...``
        and ``mirror`` gives ``...C B A | A B C...``.  Growing ``+z`` the two
        are identical, and this argument is why the parameter exists at all:
        the redesign plan first claimed that stating the registry made the
        choice unreachable, when it had only made it unstated
        (§ 3.2, corrected 2026-08-30 at the user's prompt).

    What it deliberately does NOT take
    ----------------------------------
    ``center_indices`` (placement is absolute), ``contact_distance`` (the
    starting z is given outright), ``side`` (``grow`` says it, and says it
    without mirroring by accident), and ``gap`` (there is no pair -- one
    slab per call, so nothing has to guess where the other one goes).
    """
    m, n, n_layers = (int(v) for v in size)
    if n_layers <= 0:
        return struct.copy()
    if grow not in ("+z", "-z"):
        raise ValueError(f"grow must be '+z' or '-z'; got {grow!r}")
    if stacking not in ("continue", "mirror"):
        raise ValueError(
            f"stacking must be 'continue' or 'mirror'; got {stacking!r}")
    _check_fcc_element(element)
    # THE PLANE IS CHECKED BEFORE THE REGISTRY, and the order is the whole
    # point.  The registry lookup below also rejects an unknown plane, but it
    # says "no stacking period is known for fcc(101)" -- which names neither
    # what is wrong nor what is allowed.  `_ase_slab_builder` names the closed
    # list.  The deleted `add_electrode_slab` checked the plane first and
    # this one did not, so its removal briefly cost the better message
    # (caught by its own test on 2026-09-01, repointed here).
    _ase_slab_builder(plane)
    a = (lattice_constant if lattice_constant is not None
         else _get_fcc_lattice()[element])

    period = _cell.STACKING_PERIOD.get(plane)
    if period is None:                              # pragma: no cover
        # Unreachable while STACKING_PERIOD covers the three supported
        # planes; kept because the two lists are separate facts and this is
        # the message if they ever disagree.
        raise ValueError(
            f"no stacking period is known for fcc({plane}), so a start "
            f"registry cannot be interpreted")
    k0 = int(start_registry) % period

    # BUILD TALL, TRIM, THEN MOVE AS ONE PIECE (user, 2026-08-30: "you can
    # create a super set that has more layers, trim it as needed, and offset
    # precisely with mirroring as needed").
    #
    # THE REGISTRY IS WHICH SLICE YOU TAKE, and nothing moves sideways at all.
    # Superset layer j already sits on registry j mod period -- ASE put it
    # there -- so choosing where the window starts chooses the registry, and
    # the result is a CONTIGUOUS SLICE OF A REAL CRYSTAL by construction.
    #
    # Two earlier attempts moved atoms laterally instead, and both were built
    # on a false premise: that there is one lateral "step" from one layer to
    # the next.  There is not.  Measured on ASE's own untouched Au(111) slab,
    # consecutive layer centroids walk by three DIFFERENT vectors repeating
    # with period 3 -- [-1.4418, 0.8324], [0, -1.6648], [1.4418, 0.8324] --
    # because each layer is wrapped into the cell.  Any single "step" is one
    # of three, and shifting by it lands a finite patch a lattice vector away
    # from where it was meant to be.  Trimming asks the question that has an
    # answer.
    tall = n_layers + period - 1
    full = _build_ase_slab(element, plane, (m, n, tall), orthogonal, a)
    all_pos = np.asarray(full.positions, dtype=float)
    zs = sorted({round(float(z), 6) for z in all_pos[:, 2]})

    # WHICH LAYER LANDS ON `start_z`, and therefore which window carries the
    # registry the caller asked for.  Growing up, or mirrored, it is the
    # window's BOTTOM layer; growing down by continuing, its TOP.
    if grow == "-z" and stacking == "continue":
        first = (k0 - n_layers + 1) % period
    else:
        first = k0
    keep_z = zs[first:first + n_layers]
    if len(keep_z) < n_layers:                      # pragma: no cover - guarded by `tall`
        raise RuntimeError(
            f"a {tall}-layer slab could not supply {n_layers} layers from "
            f"registry {k0}; this is a builder bug, not a bad request")
    lo, hi = keep_z[0] - 1e-6, keep_z[-1] + 1e-6
    metal_pos = all_pos[(all_pos[:, 2] >= lo) & (all_pos[:, 2] <= hi)].copy()

    # EVERYTHING FROM HERE IS RIGID -- it acts on the whole slice, never on a
    # layer.  A rigid motion of a crystal is a crystal, so the class of bug
    # that per-layer editing invites is unreachable rather than guarded.
    z_rel = metal_pos[:, 2] - metal_pos[:, 2].min()
    if grow == "+z":
        metal_pos[:, 2] = start_z + z_rel
    elif stacking == "mirror":
        # Reflected about the starting surface: the sequence reads the same
        # way outward from it as an upward slab does.
        metal_pos[:, 2] = start_z - z_rel
    else:
        # Translated only.  The layers below `start_z` are the ones ASE
        # already put below -- the crystal carries on downward because it was
        # never taken apart.
        metal_pos[:, 2] = start_z - (z_rel.max() - z_rel)

    # PLACEMENT IS ABSOLUTE, and the reference is THE SUPERSET'S centroid --
    # not the slice's.
    #
    # THE SLICE'S CENTROID IS REGISTRY-DEPENDENT, and using it silently
    # cancelled part of the registry the caller asked for.  Layer j's lateral
    # offset repeats with the stacking period, so the mean over a window of L
    # layers is independent of where the window starts ONLY when L is a whole
    # number of periods.  At any other L the correction differed per registry:
    # measured on Au(111), a 4-layer slab at registry B sat 1.249 A from the
    # registry-A one where the true step is a/sqrt(6) = 1.665 A -- off-lattice,
    # and past SEAM_STEP_TOL_ANG, so `classify_seam` then called the pair
    # `unknown` for a junction the user had every reason to think was one
    # crystal.
    #
    # The superset is the same slab for every registry, so its centroid is a
    # constant: `offset` means one thing, and two registries differ by exactly
    # the lattice step between them.  `tall` is at least one full period, so
    # this reference is itself period-averaged rather than arbitrary.
    metal_pos[:, :2] += np.asarray(offset, dtype=float) \
        - all_pos[:, :2].mean(axis=0)

    # NO PADDING.  `c` IS MEASURED AND SET, NEVER INVENTED
    # (`junction-cell.md` § 6, rewritten 2026-08-31).
    #
    # This used to add one interlayer spacing for you.  The switch that was
    # supposed to expose that decision made it instead -- it defaulted on, and
    # its note deliberately withheld the number -- so the one value deciding
    # whether a junction was a crystal was computed out of sight.
    #
    # What the builder knows, it sets: `a` and `b` are the crystal's own
    # in-plane vectors, straight from the slab ASE built.  What it does not
    # know, it leaves: `c` comes out as the atoms' extent, which is a
    # COLLISION until a person sets it on the Cell page.  That is deliberate
    # -- the missing step is visible where it is taken, and `classify_seam`
    # names it on every build rather than a builder guessing.
    return _finish_slab(struct, metal_pos, element, full)


def calibrate_to_cell(struct: Structure) -> Structure:
    """Move the structure into its cell's SIESTA coordinate frame (§ 3c).

    The unified "last step" before saving / handing to SIESTA: bake the
    generation-time shift into the STORED coordinates so the viewer box, the saved
    ``.xyz``, and the emitted FDF all agree.  Translate every atom by
    ``-resolve_cell_origin()`` so the atoms sit in ``[0, cell)`` with the cell
    anchored at ``(0,0,0)``, and materialise the resolved cell as the explicit cell
    (``cell_origin`` cleared).

    Idempotent: a second call is a no-op (origin is already 0).  Optional --
    ``render_fdf`` applies the SAME shift on the fly, so an un-calibrated structure
    still generates correct SIESTA input; calibration just lets the user SEE and
    SAVE the exact coordinate frame SIESTA will use.

    Raises ``ValueError`` for a ``periodic`` axis with no explicit cell (you cannot
    materialise a commensurate lattice from a bounding box; § 3).
    """
    if struct.n_atoms == 0:
        return struct.copy()
    resolved = struct.resolve_cell()          # explicit cell, or derived bbox+vacuum
    origin = struct.resolve_cell_origin()
    shift = (-np.asarray(origin, dtype=float)
             if origin is not None else np.zeros(3, dtype=float))
    return Structure(
        elements=list(struct.elements),
        positions=struct.positions + shift,
        cell=(resolved.copy() if resolved is not None else None),
        cell_origin=None,                     # atoms now in [0, cell); cell at origin
        axis_kind=struct.axis_kind,
        vacuum=struct.vacuum,
        pbc=struct.pbc,
        atom_names=list(struct.atom_names),
        residue_ids=list(struct.residue_ids),
        residue_names=list(struct.residue_names),
        chain_ids=list(struct.chain_ids),
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        annotations=copy_annotations(struct.annotations),
    )


__all__ = [
    "delete_atoms",
    "add_atom",
    "orient_along_axis",
    "rotate_around_axis",
    "calibrate_to_cell",
    "SUPPORTED_FCC_ELEMENTS",
    "SUPPORTED_FCC_PLANES",
    "FCC_ORTHOGONAL_CHOICES",
]
