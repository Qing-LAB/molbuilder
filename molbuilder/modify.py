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

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure


# --------------------------------------------------------------------- #
#  Atom-level edits                                                     #
# --------------------------------------------------------------------- #


def delete_atoms(struct: Structure, indices: Sequence[int]) -> Structure:
    """Return a new ``Structure`` with the given atom indices removed.

    Indices may be in any order; duplicates are tolerated.  All per-atom
    metadata arrays are sliced consistently.
    """
    keep = sorted(set(range(struct.n_atoms)) - set(int(i) for i in indices))
    if len(keep) == struct.n_atoms:
        # No-op (or all indices were out of range / already absent).
        return Structure(
            elements=list(struct.elements),
            positions=struct.positions.copy(),
            atom_names=list(struct.atom_names),
            residue_ids=list(struct.residue_ids),
            residue_names=list(struct.residue_names),
            chain_ids=list(struct.chain_ids),
            title=struct.title,
        )
    return Structure(
        elements=     [struct.elements[i]      for i in keep],
        positions=    struct.positions[keep].copy(),
        atom_names=   [struct.atom_names[i]    for i in keep],
        residue_ids=  [struct.residue_ids[i]   for i in keep],
        residue_names=[struct.residue_names[i] for i in keep],
        chain_ids=    [struct.chain_ids[i]     for i in keep],
        title=struct.title,
    )


def add_atom(
    struct: Structure,
    element: str,
    anchor_index: int,
    offset: Sequence[float],
    *,
    atom_name: Optional[str] = None,
    residue_name: str = "MOD",
) -> Structure:
    """Return a new ``Structure`` with one atom appended.

    The new atom's position is ``struct.positions[anchor_index] + offset``.
    A new residue id is allocated (``max(residue_ids) + 1``) so the added
    atom isn't lumped into the anchor's residue -- handy when the user
    wants to delete just-added atoms in one shot, and matches the way
    molbuilder treats added thiol caps / explicit hydrogens.

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
    """
    if not (0 <= anchor_index < struct.n_atoms):
        raise IndexError(
            f"anchor_index {anchor_index} out of range for "
            f"{struct.n_atoms}-atom structure"
        )
    offset_arr = np.asarray(offset, dtype=float).reshape(3)
    new_pos = struct.positions[anchor_index] + offset_arr
    new_residue_id = (max(struct.residue_ids) if struct.residue_ids else 0) + 1
    return Structure(
        elements=struct.elements + [element],
        positions=np.vstack([struct.positions, new_pos[None, :]]),
        atom_names=struct.atom_names + [atom_name or element],
        residue_ids=struct.residue_ids + [new_residue_id],
        residue_names=struct.residue_names + [residue_name],
        chain_ids=struct.chain_ids + [struct.chain_ids[anchor_index]],
        title=struct.title,
    )


# --------------------------------------------------------------------- #
#  Whole-structure orientation                                          #
# --------------------------------------------------------------------- #


_AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


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
    if axis not in _AXIS_VECTORS:
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

    return Structure(
        elements=list(struct.elements),
        positions=new_pos,
        atom_names=list(struct.atom_names),
        residue_ids=list(struct.residue_ids),
        residue_names=list(struct.residue_names),
        chain_ids=list(struct.chain_ids),
        title=struct.title,
    )


def rotate_around_axis(
    struct: Structure,
    axis: str = "z",
    angle: float = 0.0,
) -> Structure:
    """Rotate every atom by ``angle`` degrees around the given axis,
    which passes through the origin.

    Useful after :func:`orient_along_axis` to spin a tilted molecule
    around the transport axis to a different azimuth -- e.g. to tilt
    the molecule in the yz-plane instead of the default xz-plane.

    Parameters
    ----------
    axis
        Rotation axis: "x", "y", or "z" (default).  Passes through
        the origin -- callers wanting rotation around a different
        point should translate first / after.
    angle
        Rotation angle in **degrees**.  Right-hand rule (positive
        angle = counter-clockwise when looking down the axis from +
        toward origin).  Default ``0.0`` is a no-op.
    """
    if axis not in _AXIS_VECTORS:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
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
    return Structure(
        elements=list(struct.elements),
        positions=struct.positions @ R.T,
        atom_names=list(struct.atom_names),
        residue_ids=list(struct.residue_ids),
        residue_names=list(struct.residue_names),
        chain_ids=list(struct.chain_ids),
        title=struct.title,
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
import json as _json
import os as _os
from pathlib import Path as _Path


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


def _load_fcc_lattice() -> dict:
    """Load the FCC lattice-constant table from ``fcc_lattice.json``.

    Walks ``_data_dir_candidates()`` and returns the first found file's
    contents.  Format is checked at parse time -- a missing file or a
    schema mismatch raises a clear error so a misconfigured override
    surfaces immediately at import.
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
        metals: dict = {}
        for sym, entry in data["metals"].items():
            try:
                metals[sym] = float(entry["a"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"FCC lattice entry {sym!r} in {path!s} is malformed: {exc}"
                ) from exc
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


_FCC_LATTICE_A: dict = _load_fcc_lattice()

# Public tuple of supported electrode elements.  CLI / web layers read
# this to populate dropdowns and validate input -- single source of truth.
SUPPORTED_FCC_ELEMENTS: Tuple[str, ...] = tuple(_FCC_LATTICE_A.keys())

SUPPORTED_FCC_PLANES: Tuple[str, ...] = ("100", "110", "111")


def _check_fcc_element(element: str) -> None:
    """Raise ``ValueError`` if the requested element isn't on the
    supported list.  The list is closed (see ``SUPPORTED_FCC_ELEMENTS``)
    so the UI dropdown and the Python API stay in lockstep.
    """
    if element not in _FCC_LATTICE_A:
        raise ValueError(
            f"unsupported electrode element {element!r}; "
            f"supported FCC metals are: "
            f"{', '.join(SUPPORTED_FCC_ELEMENTS)}.  "
            f"For BCC / HCP electrodes, see "
            f"docs/spec/modify-tab.md § 'Off-scope'."
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

    See ``docs/spec/modify-tab.md § 8`` for the per-(plane, orthogonal)
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
            f"the orthogonal switch (see docs/spec/modify-tab.md § 8)."
        ) from exc


def add_electrode_slab(
    struct: Structure,
    element: str,
    plane: str,
    size: Tuple[int, int, int],
    anchor_index: int,
    *,
    contact_distance: float = 2.4,
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
        See ``docs/spec/modify-tab.md § 8``.
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
        return Structure(
            elements=list(struct.elements),
            positions=struct.positions.copy(),
            atom_names=list(struct.atom_names),
            residue_ids=list(struct.residue_ids),
            residue_names=list(struct.residue_names),
            chain_ids=list(struct.chain_ids),
            title=struct.title,
        )
    if side not in ("+z", "-z"):
        raise ValueError(f"side must be '+z' or '-z'; got {side!r}")
    if not (0 <= anchor_index < struct.n_atoms):
        raise IndexError(
            f"anchor_index {anchor_index} out of range for "
            f"{struct.n_atoms}-atom structure"
        )
    _check_fcc_element(element)
    a = lattice_constant if lattice_constant is not None else _FCC_LATTICE_A[element]

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
    return Structure(
        elements=list(struct.elements) + [element] * n_new,
        positions=np.vstack([struct.positions, metal_pos]),
        atom_names=list(struct.atom_names) + [element] * n_new,
        residue_ids=list(struct.residue_ids) + [new_residue_id] * n_new,
        residue_names=list(struct.residue_names) + ["ELC"] * n_new,
        chain_ids=list(struct.chain_ids) + ["A"] * n_new,
        title=struct.title,
    )


# --------------------------------------------------------------------- #
#  Convenience: symmetric junction in one call                          #
# --------------------------------------------------------------------- #


def add_symmetric_electrodes(
    struct: Structure,
    element: str,
    plane: str,
    size: Tuple[int, int, int],
    anchor_indices: Tuple[int, int],
    *,
    gap: float = 8.0,
    orthogonal: bool = False,
    offset: Tuple[float, float] = (0.0, 0.0),
    lattice_constant: Optional[float] = None,
) -> Structure:
    """Add a symmetric pair of FCC electrodes -- one on +z, one on -z --
    flanking the molecule's anchor pair.

    Geometry:

      mid    = 0.5 * (positions[a_top] + positions[a_bot])
      gap    = electrode-to-electrode distance (closest layer to closest
               layer), measured along z
      top closest layer at  z = mid.z + gap/2
      bot closest layer at  z = mid.z - gap/2
      both slabs lateral-centred on (mid.x + offset[0], mid.y + offset[1])

    The two electrodes are **collinear along z**, even when the anchor
    pair vector is tilted off the z-axis.  This matches real junction
    geometry where the metal contacts are crystallographic and the
    molecule fits in whatever pose it wants between them.

    ``gap`` here is the **canonical "junction gap"** -- the empty z-space
    between the two electrodes.  Internally each side gets the
    contact distance ``(gap - anchor_separation_z) / 2``, where
    ``anchor_separation_z = abs(positions[a_top].z - positions[a_bot].z)``.
    If the gap is smaller than the anchor pair's z-extent, this raises
    ``ValueError`` so the user adjusts before getting an overlapping
    structure.

    Parameters
    ----------
    anchor_indices
        ``(a_top, a_bot)`` -- the +z anchor first, the -z anchor second.
        After ``orient_along_axis(..., center="midpoint")`` the +z
        anchor is the one with the larger z coordinate.
    gap
        Total junction gap (Å), electrode-to-electrode along z.  Default
        8.0 Å is a typical value for thiol-anchored small molecules.
    orthogonal, offset, lattice_constant
        Same as :func:`add_electrode_slab`; applied to both sides.

    For asymmetric junctions (different size / metal / offset per side,
    or stepped contacts), call :func:`add_electrode_slab` twice with
    different parameters.
    """
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
