"""Structure-modification helpers for the Modify tab.

Public surface (all pure functions; each returns a new ``Structure``):

    delete_atoms(struct, indices)              -> Structure
    add_atom(struct, element, anchor_index, offset)
                                                -> Structure
    orient_along_axis(struct, anchor_indices, axis="z", center="first")
                                                -> Structure
    add_electrode_slab(struct, element, plane, layer_sizes, anchor_index,
                       gap, side="+z", lattice_constant=None)
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
    * ``add_electrode_slab(side="+z")`` places the first (closest)
      layer at z = anchor.z + gap, with subsequent layers extending
      outward in the +z direction.  Use ``side="-z"`` for the bottom
      electrode.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure, copy_annotations, remap_annotations


# --------------------------------------------------------------------- #
#  Atom-level edits                                                     #
# --------------------------------------------------------------------- #


def _reindex_transport_metadata(
    struct: Structure, keep: Sequence[int],
) -> Tuple[List[int], "dict[str, List[int]]", "dict"]:
    """Remap ``struct.frozen_atoms``, ``struct.regions`` AND the
    extensible ``struct.annotations`` channels after a slice/delete
    operation, dropping any index that fell off and renumbering survivors
    to their new 0-based position (atom-annotations.md § 2.1).

    Used by ``delete_atoms`` (the only modify-op that changes the
    index space).  Pure-passthrough ops (translate / rotate / orient)
    can carry frozen_atoms + regions through verbatim without this.
    """
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_frozen = [old_to_new[i] for i in struct.frozen_atoms if i in old_to_new]
    new_regions = {}
    for label, idxs in struct.regions.items():
        remapped = [old_to_new[i] for i in idxs if i in old_to_new]
        if remapped:
            new_regions[label] = remapped
    new_annotations = remap_annotations(struct.annotations, old_to_new)
    return new_frozen, new_regions, new_annotations


def compute_selection_remap_after_delete(
    struct: Structure,
    indices: Sequence[int],
) -> List[Optional[int]]:
    """Return the per-atom remap that maps every PRE-delete index to
    its POST-delete index (or ``None`` if the atom was removed).

    Wire shape: a flat list where ``remap[old_index] == new_index``,
    or ``None`` when the atom was deleted.  Length equals the pre-
    delete ``struct.n_atoms``.  Workspace-state protocol § 4.5 +
    § 5.1 — emitted under ``extra["selection_remap"]`` by the
    ``/api/modify/delete`` endpoint.

    Why a list, not a dict: JSON int keys round-trip to strings;
    a list keeps the wire shape compact and the client doesn't
    have to parse keys.  ``len(remap)`` is the pre-op atom count
    (an implicit sanity check for the client).

    Used by the Phase 4+ workspace dispatcher to correctly remap
    the user's selection across a Delete that shrinks the
    structure.  Pre-Phase-3 the client's naive
    ``selection.filter(i < new_n_atoms)`` silently dropped
    surviving high-index atoms (e.g. selecting atom 2, deleting
    atom 0, ended up with empty selection instead of new index 1).

    >>> # Structure with atoms [O, H, H], delete index 0 (the O):
    >>> # surviving atoms become new [H, H] at indices [0, 1].
    >>> compute_selection_remap_after_delete(s, [0])
    [None, 0, 1]
    """
    indices_set = set(int(i) for i in indices)
    keep = sorted(set(range(struct.n_atoms)) - indices_set)
    old_to_new = {old: new for new, old in enumerate(keep)}
    return [old_to_new.get(i) for i in range(struct.n_atoms)]


def compute_selection_remap_after_add(
    struct: Structure,
) -> List[int]:
    """Return the identity remap for an ``add_atom`` op.

    ``add_atom`` appends at index ``struct.n_atoms``; every pre-op
    index keeps its value.  Length equals the pre-op ``struct.n_atoms``
    (the new atom appears at the next index, which has no pre-op
    counterpart).

    Why emit it at all when it's the identity: the Phase 4+ workspace
    dispatcher applies the same per-op rule (look up
    ``extra["selection_remap"]``) regardless of op class — emitting
    the identity for add keeps the dispatcher's per-op branch table
    flat instead of growing a "no remap means preserve" special case.
    """
    return list(range(struct.n_atoms))


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
    new_frozen, new_regions, new_annotations = _reindex_transport_metadata(
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
        frozen_atoms=new_frozen,
        annotations=new_annotations,
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
    offset_arr = np.asarray(offset, dtype=float).reshape(3)
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
        frozen_atoms=list(struct.frozen_atoms),
        annotations=copy_annotations(struct.annotations),
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
    new_pos = struct.positions @ R.T

    if center == "first":
        new_pos = new_pos - new_pos[a0]
    elif center == "midpoint":
        midpoint = 0.5 * (new_pos[a0] + new_pos[a1])
        new_pos = new_pos - midpoint
    # "none": no translation

    # Pure rotation: atom indices are unchanged so frozen_atoms +
    # regions carry through verbatim.
    return Structure(
        elements=list(struct.elements),
        positions=new_pos,
        atom_names=list(struct.atom_names),
        residue_ids=list(struct.residue_ids),
        residue_names=list(struct.residue_names),
        chain_ids=list(struct.chain_ids),
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        frozen_atoms=list(struct.frozen_atoms),
        annotations=copy_annotations(struct.annotations),
    )


def rotate_around_axis(
    struct: Structure,
    axis: str = "z",
    angle: float = 0.0,
    *,
    center: str = "origin",
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
    if center == "centroid":
        # Rotate about the atom centroid by subtracting it, rotating
        # in the centroid frame, then translating back.
        centroid = struct.positions.mean(axis=0)
        new_pos = (struct.positions - centroid) @ R.T + centroid
    else:
        new_pos = struct.positions @ R.T
    # Pure rotation: atom indices are unchanged so frozen_atoms +
    # regions carry through verbatim.
    return Structure(
        elements=list(struct.elements),
        positions=new_pos,
        atom_names=list(struct.atom_names),
        residue_ids=list(struct.residue_ids),
        residue_names=list(struct.residue_names),
        chain_ids=list(struct.chain_ids),
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        frozen_atoms=list(struct.frozen_atoms),
        annotations=copy_annotations(struct.annotations),
    )


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
# sibling ``add_electrode_slab_bcc`` / ``..._hcp`` function rather than
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


def _load_fcc_lattice_full() -> dict:
    """Load the full FCC lattice-constant table from ``fcc_lattice.json``.

    v2 schema (2026-06-18 onward): each metal carries
    ``a_experimental`` (Wyckoff 1963), ``a_pbe`` (Haas-Tran-Blaha 2009),
    and ``a_pbe_siesta_psml`` (user-measured, nullable until populated
    via a bulk-cell relax in the user's specific SIESTA+PSML setup).

    Returns the metals dict directly: ``{symbol: {a_experimental: float,
    a_pbe: float, a_pbe_siesta_psml: Optional[float], name: str,
    system: str}}``.  Format check is strict; v1 ("a" only) files raise.
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
        if "v2" not in fmt:
            raise RuntimeError(
                f"FCC lattice table at {path!s} is not v2 (got "
                f"{fmt!r}).  Each metal must carry a_experimental, "
                f"a_pbe, a_pbe_siesta_psml; the v1 'a'-only schema "
                f"is no longer supported."
            )
        metals: dict = {}
        for sym, entry in data["metals"].items():
            try:
                a_exp = float(entry["a_experimental"])
                a_pbe = float(entry["a_pbe"])
                a_psml_raw = entry.get("a_pbe_siesta_psml")
                a_psml = float(a_psml_raw) if a_psml_raw is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"FCC lattice entry {sym!r} in {path!s} is malformed: {exc}"
                ) from exc
            metals[sym] = {
                "a_experimental":    a_exp,
                "a_pbe":             a_pbe,
                "a_pbe_siesta_psml": a_psml,
                "name":              entry.get("name", sym),
                "system":            entry.get("system", "fcc"),
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
    ``_load_fcc_lattice_full`` directly (the web meta endpoint does).
    """
    full = _load_fcc_lattice_full()
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
# ``data/contact_distance.json``.  Used by ``add_electrode_slab``
# when the caller doesn't override ``contact_distance``: Au-S
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
            f"For BCC / HCP electrodes, see "
            f"docs/tabs/molbuilder.md § 'Off-scope'."
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

    See ``docs/tabs/molbuilder.md § 8`` for the per-(plane, orthogonal)
    compatibility table -- determined empirically once, but enforced
    by passing through to ASE rather than re-implementing the rules.
    """
    builder = _ase_slab_builder(plane)
    try:
        return builder(element, size=size, orthogonal=orthogonal, a=a)
    except (ValueError, AssertionError, NotImplementedError) as exc:
        raise ValueError(
            f"ASE rejected the {element} fcc({plane}) slab with "
            f"size={size}, orthogonal={orthogonal}: {exc}.  "
            f"Adjust (m, n) to satisfy ASE's constraint, or flip "
            f"the orthogonal switch (see docs/tabs/molbuilder.md § 8)."
        ) from exc


def add_electrode_slab(
    struct: Structure,
    element: str,
    plane: str,
    size: Tuple[int, int, int],
    anchor_index: int,
    *,
    contact_distance: Optional[float] = None,
    side: str = "+z",
    orthogonal: bool = False,
    offset: Tuple[float, float] = (0.0, 0.0),
    lattice_constant: Optional[float] = None,
    inter_layer_offset: Optional[float] = None,
) -> Structure:
    """Append a single FCC electrode slab on one side of an anchor atom.

    Single-electrode primitive.  For the canonical pair-electrode
    junction, use :func:`add_symmetric_electrodes` (it takes
    ``gap`` = electrode-to-electrode distance, the meaningful junction
    parameter) instead of calling this twice.

    The slab is built by ASE's ``fcc{100,110,111}`` builder with
    ``size=(m, n, n_layers)`` where every layer has the same lateral
    ``(m, n)`` (uniform across layers).

    The whole slab is translated so:

      * The slab's lateral centroid sits at ``(anchor.x + offset[0],
        anchor.y + offset[1])``.
      * The closest layer's z is ``anchor.z + sign * contact_distance``
        (sign = +1 for ``side="+z"``, -1 for ``"-z"``); subsequent
        layers extend outward at the slab's natural inter-layer
        spacing.

    Parameters
    ----------
    struct
        Existing structure (typically the molecule + already on-axis).
    element
        Metal symbol; must be one of ``SUPPORTED_FCC_ELEMENTS``
        (Au / Ag / Cu / Ni / Pt / Pd).
    plane
        "100" / "110" / "111".
    size
        ``(m, n, n_layers)`` -- in-plane repeat counts and number of
        layers.  Uniform across all layers.  ``n_layers == 0`` returns
        ``struct`` unchanged.
    anchor_index
        Atom whose (x, y, z) defines the electrode placement reference.
    contact_distance
        Distance (Å) from the anchor atom along the side direction to
        the closest electrode layer's z plane.  Default 2.4 Å (a
        typical Au-S contact distance).  Single-electrode-only param
        -- the pair version uses ``gap`` (electrode-to-electrode
        distance) and computes contact distances internally.
    side
        ``"+z"`` (default) or ``"-z"``.
    orthogonal
        Cell-shape choice.  ``False`` (default) requests the primitive
        in-plane unit cell (hexagonal parallelogram for fcc(111);
        rectangular for fcc(100) / fcc(110), which is also their
        primitive cell).  ``True`` forces an orthogonal/rectangular
        supercell -- relevant only for fcc(111), where it imposes
        ASE's "second-axis repeat count must be even" constraint.
        Compatibility is enforced **by passing through to ASE**; if
        the (plane, orthogonal, size) tuple is rejected, ASE's own
        error message is re-raised as a ``ValueError`` so the caller
        / UI can display it as a hint and the user can adjust manually.
        See ``docs/tabs/molbuilder.md § 8``.
    offset
        ``(Δx, Δy)`` lateral shift in Å applied to the slab's centroid
        relative to the anchor's xy.  Default ``(0.0, 0.0)`` puts the
        slab centroid directly above (or below) the anchor -- the atop
        site for that anchor.  Non-zero values shift the slab so the
        anchor sits at a bridge site (between two surface atoms),
        a hollow site (in a 3-fold hollow on fcc(111)), or wherever
        the user wants the molecule-surface contact point to fall.
    lattice_constant
        Override the value loaded from
        ``molbuilder/data/fcc_lattice.json`` (Å).
    inter_layer_offset
        Override the slab's natural inter-layer spacing (Å).  Default
        ``None`` lets the ASE slab decide (the experimentally-correct
        spacing for the chosen lattice constant + plane).  Useful for
        strained-distance studies where the contact's interlayer
        spacing is intentionally different from bulk.

    Returns
    -------
    Structure
        ``struct`` with the electrode atoms appended.  The electrode
        atoms get residue name ``"ELC"`` and a fresh residue id; the
        atom_names are set to the element symbol.
    """
    m, n, n_layers = (int(s) for s in size)
    if n_layers <= 0:
        return struct.copy()
    # Element-aware default: 2.4 A is Au-S; Pt-N wants 2.05, Ag-S
    # 2.50, etc.  Loaded from data/contact_distance.json so users
    # can audit / override per molbuilder/data/README.md.
    if contact_distance is None:
        contact_distance = default_contact_distance(element)
    if side not in ("+z", "-z"):
        raise ValueError(f"side must be '+z' or '-z'; got {side!r}")
    if not (0 <= anchor_index < struct.n_atoms):
        raise IndexError(
            f"anchor_index {anchor_index} out of range for "
            f"{struct.n_atoms}-atom structure"
        )
    _check_fcc_element(element)
    a = lattice_constant if lattice_constant is not None else _get_fcc_lattice()[element]

    # Use ASE's slab builder (principle #8: "Don't reinvent wheels").
    # The ``orthogonal`` flag is user-selectable; ``_build_ase_slab``
    # passes it straight through to ASE and re-raises ASE's error
    # verbatim (with operation context) if the (plane, orthogonal,
    # size) tuple is unsupported -- the user reads the error and
    # adjusts (m, n) manually.  Slab is uniform in (m, n) across
    # layers, so the returned atom set is exactly what we want -- no
    # per-layer cropping needed.
    full = _build_ase_slab(
        element, plane,
        size=(m, n, n_layers),
        orthogonal=orthogonal,
        a=a,
    )
    metal_pos = np.asarray(full.positions, dtype=float).copy()

    # Identify per-layer z values for the gap-anchor and the optional
    # inter-layer override.  ASE orders atoms by layer; we round to
    # absorb FP noise and group.
    z_vals = metal_pos[:, 2]
    z_unique = sorted({round(float(z), 4) for z in z_vals})
    z_min = z_unique[0]

    # Sanity-check ASE's slab orientation: every atom in a given
    # layer must have the same z (within FP tolerance), i.e. the slab
    # surface normal is parallel to z.  ASE's ``fcc{100,110,111}``
    # builders honour this convention today; the assertion is a guard
    # so a future ASE convention change (or a misinterpreted custom
    # ``lattice_constant``) can't silently produce a tilted slab,
    # which would invalidate transport-DFT lead self-energy
    # assumptions.
    for z_layer in z_unique:
        in_layer = metal_pos[np.abs(metal_pos[:, 2] - z_layer) < 1e-3, 2]
        if len(in_layer) and float(np.std(in_layer)) > 1e-6:
            raise RuntimeError(
                f"ASE slab is not z-perpendicular (layer at z={z_layer} "
                f"has within-layer z-std {np.std(in_layer):.2e} > 1e-6). "
                f"Surface normal must be along z for transport-DFT "
                f"electrodes.  Aborting; this is a build-tool regression."
            )

    # Translate so the slab's lateral centroid (average xy of every
    # electrode atom) lands at ``(anchor.x + offset[0], anchor.y +
    # offset[1])``.  The default ``offset=(0, 0)`` puts the slab
    # centroid directly over the anchor (atop site).  Non-zero
    # ``offset`` shifts the slab laterally so the user can park the
    # anchor over a bridge / hollow site -- the slider in the Modify
    # UI dials this in interactively.
    anchor = struct.positions[anchor_index]
    offset_x, offset_y = float(offset[0]), float(offset[1])
    slab_centroid_xy = metal_pos[:, :2].mean(axis=0)
    metal_pos[:, 0] += (anchor[0] + offset_x) - slab_centroid_xy[0]
    metal_pos[:, 1] += (anchor[1] + offset_y) - slab_centroid_xy[1]

    # z-positioning: closest layer at anchor.z + sign * gap, others
    # extending outward at the slab's natural inter-layer spacing
    # (which ASE has already set correctly for the chosen lattice
    # constant + plane).
    sign = +1.0 if side == "+z" else -1.0
    if side == "+z":
        metal_pos[:, 2] += (anchor[2] + contact_distance) - z_min
    else:
        # Mirror across the closest-layer z so the stack extends -z.
        z_in = metal_pos[:, 2] - z_min
        metal_pos[:, 2] = anchor[2] - contact_distance - z_in

    # Optional per-layer-spacing override (rare; pulls layers together
    # or pushes them further out for strained-distance studies).
    # Rescales spacing around the closest layer's z.  Only meaningful
    # for n_layers >= 2.  Group atoms into per-layer z bands at 1e-9
    # precision so a precise ``inter_layer_offset`` round-trips exactly
    # (the rounded ``z_unique`` from earlier was 4-decimal, too coarse).
    if inter_layer_offset is not None and n_layers > 1:
        z_layers_precise = sorted({round(float(z), 9) for z in metal_pos[:, 2]})
        if len(z_layers_precise) > 1:
            natural_spacing = abs(z_layers_precise[1] - z_layers_precise[0])
            if natural_spacing > 1e-9:
                scale = inter_layer_offset / natural_spacing
                closest_z = anchor[2] + sign * contact_distance
                metal_pos[:, 2] = closest_z + (metal_pos[:, 2] - closest_z) * scale

    # Assemble metadata for the new metal atoms.
    n_new = metal_pos.shape[0]
    new_residue_id = (max(struct.residue_ids) if struct.residue_ids else 0) + 1

    # Capture the electrode's cell (structure-periodicity.md § 4 -- fixes the old
    # discard).  In-plane (x,y) = the ASE slab's lattice (rows 0,1; hexagonal for
    # fcc(111)); z = the DEVICE length (combined atom extent), NOT the slab's
    # periodic z.  axis_kind = (periodic, periodic, transport): the transport z is
    # electrode-matched, never tiled/k-sampled.  Overwrites any prior cell -- the
    # electrode defines the junction's in-plane periodicity.  Skipped if the z
    # extent is degenerate (would make a singular cell); the caller can set one.
    all_pos = np.vstack([struct.positions, metal_pos])
    z_extent = float(all_pos[:, 2].max() - all_pos[:, 2].min())
    slab_cell = np.asarray(full.get_cell(), dtype=float)
    elc_cell = None
    elc_axis_kind = None
    if z_extent > 1e-6:
        elc_cell = np.array([
            [slab_cell[0, 0], slab_cell[0, 1], 0.0],
            [slab_cell[1, 0], slab_cell[1, 1], 0.0],
            [0.0, 0.0, z_extent],
        ], dtype=float)
        elc_axis_kind = ("periodic", "periodic", "transport")

    # New electrode atoms are appended at indices [old_n, old_n + n_new).
    # Existing frozen_atoms + region indices carry through unchanged; the
    # new electrode atoms are NOT auto-frozen and NOT auto-tagged with a
    # region label (callers who want either can post-process the result).
    return Structure(
        elements=list(struct.elements) + [element] * n_new,
        positions=np.vstack([struct.positions, metal_pos]),
        cell=elc_cell,
        axis_kind=elc_axis_kind,
        atom_names=list(struct.atom_names) + [element] * n_new,
        residue_ids=list(struct.residue_ids) + [new_residue_id] * n_new,
        residue_names=list(struct.residue_names) + ["ELC"] * n_new,
        chain_ids=list(struct.chain_ids) + ["A"] * n_new,
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        frozen_atoms=list(struct.frozen_atoms),
        annotations=copy_annotations(struct.annotations),
    )


# --------------------------------------------------------------------- #
#  Convenience: symmetric junction in one call                          #
# --------------------------------------------------------------------- #


def add_symmetric_electrodes(
    struct: Structure,
    element: str,
    plane: str,
    size: Tuple[int, int, int],
    anchor_indices: Optional[Tuple[int, int]] = None,
    *,
    gap: float = 8.0,
    orthogonal: bool = False,
    offset: Tuple[float, float] = (0.0, 0.0),
    lattice_constant: Optional[float] = None,
) -> Structure:
    """Add a symmetric pair of FCC electrodes -- one on +z, one on -z --
    with the inter-slab midpoint at the origin.

    Geometry (the canonical case, ``anchor_indices=None``):

      top closest layer at  z = +gap/2
      bot closest layer at  z = -gap/2
      both slabs lateral-centred on (offset[0], offset[1])

    Each slab's layer planes are perpendicular to z (the surface
    normal points along z, per ASE's fcc{100,110,111} convention);
    the line joining the two slab centroids IS the z-axis.  The
    user's workflow is: Center the molecule at origin,
    rotate it (Pose ops) so it sits the way they want in the gap,
    then call this with the desired ``gap`` -- the slabs are always
    placed symmetrically around the origin so the user has full
    control over molecule-vs-slab geometry via the molecule's pose
    alone.

    ``gap`` is the **canonical "junction gap"** -- the empty z-space
    between the two electrodes' closest layers.

    Parameters
    ----------
    anchor_indices
        ``None`` (default, recommended) -- slabs centred on the
        origin; no atom selection required.

        ``(a_top, a_bot)`` -- legacy mode that places the slab midpoint
        at the anchor-pair midpoint instead of the origin.  ``a_top``
        is the +z anchor, ``a_bot`` is the -z anchor.  Useful when the
        molecule is NOT pre-centred and the caller wants the slabs to
        flank a specific atom pair instead.  The contact distance is
        ``(gap - |z_top - z_bot|) / 2`` so the slabs still sit at
        ``mid.z ± gap/2``.
    gap
        Total junction gap (Å), electrode-to-electrode along z.  Default
        8.0 Å is a typical value for thiol-anchored small molecules.
    offset
        ``(Δx, Δy)`` lateral shift in Å applied to BOTH slabs.  Default
        ``(0, 0)`` puts the slabs' lateral centroid on the (xy) origin.
    orthogonal, lattice_constant
        Same as :func:`add_electrode_slab`; applied to both sides.

    For asymmetric junctions (different size / metal / offset per side,
    or stepped contacts), call :func:`add_electrode_slab` twice with
    different parameters.
    """
    if anchor_indices is None:
        # Anchorless / origin-centred mode (the recommended workflow).
        # The slabs sit at z = ±gap/2 around the origin; xy is centred
        # on (offset_x, offset_y).  We still need ATOMS in struct so
        # the molecule is preserved through the add_electrode_slab
        # call; the GEOMETRIC CENTROID of the molecule (atom-coordinate
        # mean) plays the role of the synthetic anchor, with a
        # compensating offset so the slab lands on the absolute
        # z = ±gap/2 plane regardless of where the molecule sits.
        if struct.n_atoms == 0:
            raise ValueError(
                "cannot add electrodes to an empty structure; load a "
                "molecule first (Build tab, /api/build/load, or one "
                "of the molbuilder build CLIs)"
            )
        if gap <= 0.0:
            raise ValueError(
                f"gap = {gap:.3f} Å must be > 0"
            )
        # Reject a gap that's too small for the molecule's z-extent.
        # The molecule has to fit in the empty space between the slabs;
        # we leave a ``mol_z_margin`` cushion on each side because a
        # typical M-X contact bond is ~2.0-2.4 Å.  Rejecting up front
        # gives the user an actionable message instead of a downstream
        # min-distance validation error.
        mol_z = struct.positions[:, 2]
        mol_z_extent = float(mol_z.max() - mol_z.min())
        mol_z_margin = 1.5     # Å, generous floor on M-X contact bond
        min_gap = mol_z_extent + 2.0 * mol_z_margin
        if gap < min_gap:
            raise ValueError(
                f"gap = {gap:.2f} Å is too small for the molecule's "
                f"z-extent ({mol_z_extent:.2f} Å). Need ≥ "
                f"{min_gap:.2f} Å, or shorten / re-orient the "
                f"molecule so its z-extent is smaller."
            )
        # add_electrode_slab places its slab using ``positions[anchor_index]``
        # as the geometric reference: lateral xy = anchor.xy + offset,
        # closest-layer z = anchor.z + sign * contact_distance.  We
        # pass ``anchor_index=0`` and compute ``offset`` and
        # ``contact_distance`` relative to atom 0 so the cancellation
        # is consistent: slab xy lands at the user's offset (default
        # 0,0), closest layers at absolute ±gap/2 (independent of
        # atom 0's location, which is just a placeholder).  The
        # CENTROID is used only for the pre-flight z-extent guard
        # above; it's not the placement reference.
        ref = struct.positions[0]
        contact_top = (gap / 2.0) - float(ref[2])
        contact_bot = (gap / 2.0) + float(ref[2])
        delta_xy = (
            float(offset[0]) - float(ref[0]),
            float(offset[1]) - float(ref[1]),
        )
        out = add_electrode_slab(
            struct, element, plane, size, anchor_index=0,
            contact_distance=contact_bot, side="-z",
            orthogonal=orthogonal, offset=delta_xy,
            lattice_constant=lattice_constant,
        )
        out = add_electrode_slab(
            out, element, plane, size, anchor_index=0,
            contact_distance=contact_top, side="+z",
            orthogonal=orthogonal, offset=delta_xy,
            lattice_constant=lattice_constant,
        )
        return out

    a_top, a_bot = int(anchor_indices[0]), int(anchor_indices[1])
    if a_top == a_bot:
        raise ValueError("anchor_indices must be two distinct atoms")
    for i in (a_top, a_bot):
        if not (0 <= i < struct.n_atoms):
            raise IndexError(
                f"anchor index {i} out of range for {struct.n_atoms}-atom structure"
            )
    p_top = struct.positions[a_top]
    p_bot = struct.positions[a_bot]
    # Enforce the (top, bot) convention: top must have higher z than bot.
    # Without this check the math produces overlapping slabs when the user
    # accidentally swaps the order.
    if p_top[2] < p_bot[2]:
        raise ValueError(
            f"anchor_indices=(a_top={a_top}, a_bot={a_bot}) but the "
            f"labelled top anchor sits at z={p_top[2]:.3f} which is lower "
            f"than the labelled bottom at z={p_bot[2]:.3f}.  Pass "
            f"(top, bot) where positions[top].z > positions[bot].z, or "
            f"orient the molecule with `orient_along_axis(..., "
            f"center='midpoint')` first to put the +z anchor on +z."
        )
    mid = 0.5 * (p_top + p_bot)
    anchor_sep_z = abs(p_top[2] - p_bot[2])
    contact = (gap - anchor_sep_z) / 2.0
    if contact <= 0.0:
        raise ValueError(
            f"gap = {gap:.3f} Å is smaller than the anchor pair's z-extent "
            f"({anchor_sep_z:.3f} Å); the electrodes would overlap the "
            f"molecule.  Increase gap, or re-orient the molecule with a "
            f"smaller tilt angle so the anchor pair is more z-aligned."
        )

    # We can't just call add_electrode_slab with each anchor because that
    # would lateral-centre each slab on its own anchor (which differs in
    # x, y for a tilted molecule).  We want both slabs centred on the
    # ANCHOR-PAIR MIDPOINT in xy, collinear along z.
    #
    # Equivalent trick: make a synthetic anchor at (mid.x, mid.y, p_*.z)
    # for each side and call add_electrode_slab.  But that requires a
    # real atom -- so instead we replicate the placement math here.

    # Use add_electrode_slab via the actual top/bot anchors, with an
    # offset that compensates for the lateral difference between the
    # anchor and the desired (mid.x, mid.y).
    delta_top_xy = (
        mid[0] - p_top[0] + offset[0],
        mid[1] - p_top[1] + offset[1],
    )
    delta_bot_xy = (
        mid[0] - p_bot[0] + offset[0],
        mid[1] - p_bot[1] + offset[1],
    )

    out = add_electrode_slab(
        struct, element, plane, size, a_bot,
        contact_distance=contact, side="-z",
        orthogonal=orthogonal, offset=delta_bot_xy,
        lattice_constant=lattice_constant,
    )
    out = add_electrode_slab(
        out, element, plane, size, a_top,
        contact_distance=contact, side="+z",
        orthogonal=orthogonal, offset=delta_top_xy,
        lattice_constant=lattice_constant,
    )
    return out


__all__ = [
    "delete_atoms",
    "add_atom",
    "orient_along_axis",
    "rotate_around_axis",
    "add_electrode_slab",
    "add_symmetric_electrodes",
    "SUPPORTED_FCC_ELEMENTS",
    "SUPPORTED_FCC_PLANES",
]
