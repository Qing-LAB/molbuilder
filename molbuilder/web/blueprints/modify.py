"""Modify blueprint -- per-atom edit-op routes for the Modify tab.

Routes (no url_prefix; each carries its own full path):

    (``POST /api/modify/load`` was retired in commit ``7105ae8``;
     XYZ canonicalisation is now part of ``/api/build/load``'s
     `_struct_from_body` path so both tabs share one validator.)

    POST /api/modify/delete                delete_atoms(indices)
    POST /api/modify/add_atom              add_atom(element, anchor,
                                                    offset)
    POST /api/modify/orient                orient_along_axis(anchors,
                                                    axis, angle, center)
    POST /api/modify/rotate                rotate_around_axis(axis,
                                                    angle)
    POST /api/modify/translate             rigid translate ({dx,dy,dz}
                                                    or {recenter:true})
    POST /api/modify/electrode             add_electrode_slab (single
                                            mode: one slab on +z or -z
                                            of one anchor)
    POST /api/modify/symmetric_electrodes  add_symmetric_electrodes
                                            (pair mode: collinear-z
                                            slabs separated by gap)
    GET  /api/modify/meta                  dropdown enums (FCC
                                            elements + planes) for
                                            the UI; reads from the
                                            single-source-of-truth
                                            tuples in molbuilder.modify.

M3 covered the per-atom ops; M4 added orient + rotate; M5 added
the two electrode endpoints below.  The /api/modify/* surface is
now feature-complete relative to the molbuilder.modify Python API.

JSON body shape (shared by every op):

    {
      "xyz":            "<xyz string>",
      "atom_names":     [...],   # optional; len == n_atoms
      "residue_ids":    [...],   # optional
      "residue_names":  [...],   # optional
      "chain_ids":      [...],   # optional
      ...op-specific args...
    }

The metadata fields are optional: when present, they override the
defaults that ``Structure.from_xyz`` populates.  Sending them lets a
chain of ops preserve PDB-style atom names / residue ids across
round-trips through XYZ (per the spec's "Per-atom metadata is
preserved" invariant in `docs/web/tabs.md` § 5).

JSON response shape (shared):

    {
      "ok":            True,
      "xyz":           "...",
      "elements":      [...],
      "atom_names":    [...],
      "residue_ids":   [...],
      "residue_names": [...],
      "chain_ids":     [...],
      "n_atoms":       N,
      "title":         "...",
      "issues":        [{severity, message, where}, ...],
    }

This matches `/api/build/load` so the front end can keep one
``applyStructure(response)`` path for both initial load and
modify-op responses.

Body parsing, response building, validation, and the error
response shape all live in ``_shared.py`` so the build, modify,
and (future) any electrode-handoff blueprints share one wire
contract.  If a wire shape changes here, both blueprints' tests
catch it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, current_app, jsonify, request

from ._shared import (
    err as _err,
    finite_float as _finite_float,
    ok_structure_response as _ok_response,
    struct_from_body as _struct_from_body,
)

import numpy as np

from molbuilder.cell import (
    STACKING_PERIOD,
    classify_seam as _classify_seam,
    measure_fcc as _measure_fcc,
)
from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    SUPPORTED_FCC_PLANES,
    FCC_ORTHOGONAL_CHOICES,
    add_atom as _add_atom,
    add_electrode_slab as _add_electrode_slab,
    add_symmetric_electrodes as _add_symmetric_electrodes,
    calibrate_to_cell as _calibrate_to_cell,
    delete_atoms as _delete_atoms,
    orient_along_axis as _orient_along_axis,
    rotate_around_axis as _rotate_around_axis,
    translate as _translate,
    add_slab as _add_slab,
    load_fcc_lattice_full as _load_fcc_lattice_full,
)


bp = Blueprint("modify", __name__)


# --------------------------------------------------------------------- #
#  /api/modify/meta                                                     #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/meta", methods=["GET"])
def api_modify_meta():
    """Return enums the Modify-tab UI needs for its dropdowns.

    The single source of truth for the supported FCC elements and
    surface planes is ``molbuilder.modify``; the web form populates
    its <select> / radio groups by hitting this endpoint at page
    init rather than hardcoding the lists in the HTML template.

    Anti-pattern dodged: parallel field-metadata tables in the web
    layer that drift from the Python tuples.  Adding a new metal in
    ``molbuilder.modify`` reaches the UI automatically.
    """
    # Lattice table: per-element a_experimental + a_pbe + the nullable
    # a_pbe_siesta_psml (populated by the user when they run a bulk-
    # cell relax with their specific Au.psml/etc.).  UI renders a
    # 3-way radio per element so the user can pick the value matching
    # their XC + pseudopotential.  Failures here surface as a
    # diagnostic + an empty table so the UI degrades to the prior
    # behavior (always experimental, no radio).
    lattice_table: Dict[str, Any] = {}
    lattice_error: Optional[str] = None
    try:
        from molbuilder.modify import load_fcc_lattice_full
        lattice_table = load_fcc_lattice_full()
    except Exception as exc:                                # pragma: no cover -- defensive
        lattice_error = str(exc)
    # Layers per stacking period, by surface -- so the Junction panel can say
    # whether the chosen layer count makes a whole period without carrying its
    # own copy of the crystallography (science/junction-cell.md § 3.1).  Same
    # anti-drift reason as the two lists above.
    return jsonify({
        "ok":              True,
        "fcc_elements":    list(SUPPORTED_FCC_ELEMENTS),
        "fcc_planes":      list(SUPPORTED_FCC_PLANES),
        "lattice_table":   lattice_table,
        "lattice_error":   lattice_error,
        "stacking_period": dict(STACKING_PERIOD),
        # Which cell shapes each surface can be built with -- so the Slab
        # panel can stop offering a combination ASE cannot make, without
        # carrying its own copy of the rule (science/junction-cell.md § 2b).
        "orthogonal_choices": {p: list(v)
                               for p, v in FCC_ORTHOGONAL_CHOICES.items()},
    })


# --------------------------------------------------------------------- #
#  /api/modify/delete                                                   #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/delete", methods=["POST"])
def api_modify_delete():
    """Drop the named atom indices from the structure.

    Body: ``{xyz, [...metadata...], indices: List[int]}``.
    Out-of-range indices are silently ignored (matches
    :func:`molbuilder.modify.delete_atoms` behaviour) so the UI can
    fire the op even when its selection model is briefly stale.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    indices = body.get("indices")
    if not isinstance(indices, list):
        return _err("missing or non-list 'indices'", 400)
    try:
        indices_int: List[int] = [int(i) for i in indices]
    except (TypeError, ValueError):
        return _err("'indices' must be a list of integers", 400)
    try:
        new_struct = _delete_atoms(struct, indices_int)
    except (ValueError, IndexError) as exc:
        return _err(f"delete_atoms failed: {exc}", 400)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview.md § 11.1, "Effect on atom count") -- a cleared selection
    # can never mis-point at a shifted index, so the server computes no remap.
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/add_atom                                                 #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/add_atom", methods=["POST"])
def api_modify_add_atom():
    """Append one atom relative to an anchor.

    Body: ``{xyz, [...metadata...], element, anchor_index,
             offset: [dx, dy, dz], atom_name?, residue_name?,
             residue_id?}``.

    Defaults match :func:`molbuilder.modify.add_atom`: ``atom_name``
    falls back to ``element``, ``residue_name`` to ``"MOD"``,
    ``residue_id`` to ``max(residue_ids) + 1`` (a fresh residue so
    the new atom is easy to delete in one shot).  Pass an explicit
    ``residue_id`` to land the new atom in an existing residue
    (e.g. building a polyatomic side-chain cap).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    element = body.get("element")
    if not isinstance(element, str) or not element.strip():
        return _err("missing or empty 'element'", 400)
    anchor_index = body.get("anchor_index")
    try:
        anchor_index = int(anchor_index)
    except (TypeError, ValueError):
        return _err("'anchor_index' must be an integer", 400)
    if not (0 <= anchor_index < struct.n_atoms):
        return _err(
            f"anchor_index {anchor_index} out of range for "
            f"{struct.n_atoms}-atom structure",
            400,
        )
    offset = body.get("offset")
    if (not isinstance(offset, (list, tuple))) or len(offset) != 3:
        return _err("'offset' must be a 3-element [dx, dy, dz] list", 400)
    try:
        offset_f = [float(v) for v in offset]
    except (TypeError, ValueError):
        return _err("'offset' entries must be numeric", 400)

    atom_name = body.get("atom_name")
    residue_name = body.get("residue_name") or "MOD"
    residue_id: Optional[int] = body.get("residue_id")
    if residue_id is not None:
        try:
            residue_id = int(residue_id)
        except (TypeError, ValueError):
            return _err("'residue_id' must be an integer", 400)

    try:
        new_struct = _add_atom(
            struct, element.strip(), anchor_index, offset_f,
            atom_name=atom_name, residue_name=residue_name,
            residue_id=residue_id,
        )
    except (ValueError, IndexError) as exc:
        return _err(f"add_atom failed: {exc}", 400)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview.md § 11.1, "Effect on atom count").
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/orient                                                   #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/orient", methods=["POST"])
def api_modify_orient():
    """Rotate the structure so the anchor-pair vector forms ``angle``
    degrees with the chosen target axis.

    Body: ``{xyz, [...metadata...], anchors: [a0, a1], axis?, angle?,
             center?}``.

    Defaults match :func:`molbuilder.modify.orient_along_axis`:
    ``axis="z"`` (transport-DFT convention), ``angle=0.0`` (anchor
    pair lands exactly along the axis), ``center="midpoint"``
    (anchor-pair midpoint at the origin -- the geometry pair-mode
    electrode placement expects).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    anchors = body.get("anchors")
    if (not isinstance(anchors, list)) or len(anchors) != 2:
        return _err(
            "'anchors' must be a 2-element [a0, a1] list of atom indices",
            400,
        )
    try:
        a0, a1 = int(anchors[0]), int(anchors[1])
    except (TypeError, ValueError):
        return _err("anchor indices must be integers", 400)
    if a0 == a1:
        return _err("anchors must be two distinct atom indices", 400)
    if not (0 <= a0 < struct.n_atoms and 0 <= a1 < struct.n_atoms):
        return _err(
            f"anchor indices ({a0}, {a1}) out of range for "
            f"{struct.n_atoms}-atom structure",
            400,
        )
    axis = (body.get("axis") or "z").strip().lower()
    if axis not in ("x", "y", "z"):
        return _err(f"axis must be 'x', 'y', or 'z'; got {axis!r}", 400)
    center = (body.get("center") or "midpoint").strip().lower()
    if center not in ("midpoint", "first", "none"):
        return _err(
            f"center must be 'midpoint', 'first', or 'none'; got {center!r}",
            400,
        )
    try:
        angle_f = _finite_float("angle", body.get("angle", 0.0))
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        new_struct = _orient_along_axis(
            struct, (a0, a1), axis=axis, angle=angle_f, center=center,
        )
    except ValueError as exc:
        return _err(f"orient_along_axis failed: {exc}", 400)
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/rotate                                                   #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/rotate", methods=["POST"])
def api_modify_rotate():
    """Rotate by ``angle`` degrees (right-hand rule) about the named axis.

    Body: ``{xyz, [...metadata...], axis, angle, center?, indices?}``.

    ``center`` picks what the axis passes through -- ``"origin"``
    (default) or ``"centroid"``.  ``indices`` turns ONLY those atoms,
    about their own centroid, leaving the box where it is; omitted, the
    whole structure turns and the box turns with it.

    Useful for redirecting a tilted molecule's azimuth after an
    ``orient`` op with non-zero angle (e.g. spin a tilt from the
    xz-plane into the yz-plane via ``axis=z, angle=90``).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    axis = (body.get("axis") or "").strip().lower()
    if axis not in ("x", "y", "z"):
        return _err(f"axis must be 'x', 'y', or 'z'; got {axis!r}", 400)
    try:
        angle_f = _finite_float("angle", body.get("angle"))
    except ValueError as exc:
        return _err(str(exc), 400)
    center = (body.get("center") or "origin").strip().lower()
    if center not in ("origin", "centroid"):
        return _err(
            f"center must be 'origin' or 'centroid'; got {center!r}",
            400,
        )
    # `indices` -> turn ONLY those atoms about their own centroid, box
    # untouched.  Absent -> the whole structure turns and the box turns with it.
    indices = body.get("indices")
    if indices is not None and not isinstance(indices, list):
        return _err("'indices' must be a list of atom indices", 400)
    try:
        new_struct = _rotate_around_axis(
            struct, axis=axis, angle=angle_f, center=center, indices=indices,
        )
    except ValueError as exc:
        return _err(f"rotate_around_axis failed: {exc}", 400)
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/translate                                                #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/translate", methods=["POST"])
def api_modify_translate():
    """Translate the structure, or a named subset of its atoms.

    THREE modes (``recenter`` wins over the others if both are sent):

    * ``{recenter: true}`` -- translate so a centroid lands on the
      origin.  WHOSE centroid is decided by ``indices``, exactly as
      it is for a ``{dx, dy, dz}`` translate: with a group, that
      group's own centroid moves to the origin and only those atoms
      move; without one, the whole structure's centroid does and the
      box travels with it.  Useful after adding electrode slabs
      shifts the structure off-axis: re-anchoring the centroid
      makes mouse-zoom feel sane and aligns subsequent slab ops
      against a predictable origin.
    * ``{dx, dy, dz}`` (Å) -- translate EVERY atom by the given
      vector.  Each component defaults to 0.  The box goes with them:
      a rigid translation moves ``cell_origin`` by the same vector
      (``Structure.affine``), so the structure sits in the cell
      exactly as it did before and containment cannot change.
    * ``{dx, dy, dz, indices: [...]}`` -- move ONLY those atoms.  The
      box is NOT moved, because only part of what it contains did.
      This is the mode that can put atoms outside an explicit cell,
      and the reason the response is validated on the way out.

    Rigid in every mode: bonds, angles, residue assignments and atom
    ORDER are preserved, so a caller's selection indices survive the
    round-trip -- only coordinates change.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    # `indices` -> move ONLY those atoms, box untouched.  Absent (or empty) ->
    # the whole structure moves rigidly and the box goes with it.  The route
    # takes the atoms so the caller sends the WHOLE structure either way
    # (molview.md § 11.7: one path in, one path out).
    #
    # READ BEFORE THE RECENTER BRANCH, and that placement is the whole fix.
    # The browser sends the selection for Center exactly as it does for
    # Translate -- `applyOp` injects it from `OPERATIONS.translate.group` one
    # layer below either call site, so both bodies carry `indices`.  The
    # recenter branch used to return above this line, so the key was never
    # read and Center silently centred the whole structure while every other
    # op on the tab honoured the group.
    indices = body.get("indices")
    if indices is not None and not isinstance(indices, list):
        return _err("'indices' must be a list of atom indices", 400)
    group = indices or None

    if bool(body.get("recenter", False)):
        # Center IS a Translate whose vector the server computes.  One path,
        # so the two cannot drift: `modify.translate` already decides what
        # moves -- a named group moves alone and the box stays; no group and
        # the structure moves rigidly with its box.
        try:
            positions = (struct.positions[group] if group is not None
                         else struct.positions)
            if len(positions) == 0:
                raise ValueError("no atoms to centre")
            new_struct = _translate(struct, -positions.mean(axis=0),
                                    indices=group)
        except (ValueError, IndexError) as exc:
            return _err(f"recenter failed: {exc}", 400)
        return _ok_response(new_struct)
    try:
        dx = _finite_float("dx", body.get("dx", 0.0))
        dy = _finite_float("dy", body.get("dy", 0.0))
        dz = _finite_float("dz", body.get("dz", 0.0))
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        new_struct = _translate(struct, (dx, dy, dz), indices=indices)
    except ValueError as exc:
        return _err(f"translate failed: {exc}", 400)
    return _ok_response(new_struct)


@bp.route("/api/modify/calibrate", methods=["POST"])
def api_modify_calibrate():
    """Calibrate coordinates to the cell (structure-periodicity.md § 3c).

    The unified "last step": translate every atom by ``-resolve_cell_origin()`` so the
    atoms sit inside ``[0, cell)`` with the cell anchored at ``(0,0,0)``, and
    materialise the resolved cell as the explicit cell (``cell_origin`` cleared).  Rigid
    + count-preserving (selection indices survive); idempotent.  Lets the user SEE and
    SAVE the exact coordinate frame SIESTA will use -- generation applies the same shift
    on the fly, so this is optional.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        new_struct = _calibrate_to_cell(struct)
    except ValueError as exc:
        return _err(f"calibrate failed: {exc}", 400)
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  Electrode-shared parsing helpers                                     #
# --------------------------------------------------------------------- #


def _parse_electrode_common(body):
    """Validate and unpack the fields the two electrode endpoints
    share: element, plane, size, orthogonal, offset, lattice_constant,
    pad_interlayer_gap.

    Returns ``(element, plane, size, orthogonal, offset, lattice_constant,
    pad_interlayer_gap)``
    on success, or raises ``ValueError`` (the caller turns into HTTP
    400 via :func:`._err`).
    """
    element = body.get("element")
    if not isinstance(element, str) or element not in SUPPORTED_FCC_ELEMENTS:
        raise ValueError(
            f"element must be one of {SUPPORTED_FCC_ELEMENTS}; "
            f"got {element!r}"
        )
    plane = body.get("plane")
    if not isinstance(plane, str) or plane not in SUPPORTED_FCC_PLANES:
        raise ValueError(
            f"plane must be one of {SUPPORTED_FCC_PLANES}; got {plane!r}"
        )
    size = body.get("size")
    if (not isinstance(size, list)) or len(size) != 3:
        raise ValueError(
            "'size' must be a 3-element [m, n, n_layers] list"
        )
    try:
        m, n, n_layers = (int(size[0]), int(size[1]), int(size[2]))
    except (TypeError, ValueError):
        raise ValueError("'size' entries must be integers")
    if m < 1 or n < 1 or n_layers < 1:
        raise ValueError(
            f"'size' components must all be >= 1; got ({m}, {n}, {n_layers})"
        )
    orthogonal = bool(body.get("orthogonal", False))
    offset = body.get("offset", [0.0, 0.0])
    if (not isinstance(offset, list)) or len(offset) != 2:
        raise ValueError("'offset' must be a 2-element [dx, dy] list")
    try:
        offset_t = (float(offset[0]), float(offset[1]))
    except (TypeError, ValueError):
        raise ValueError("'offset' entries must be numeric")
    lattice_constant = body.get("lattice_constant")
    if lattice_constant is not None:
        try:
            lattice_constant = float(lattice_constant)
        except (TypeError, ValueError):
            raise ValueError("'lattice_constant' must be numeric or null")
    # Default TRUE, matching the builder: an un-padded box collides with its
    # own periodic image (science/junction-cell.md § 6).  A client that omits
    # the key must get the correct cell, not the historical one.
    pad_gap = bool(body.get("pad_interlayer_gap", True))
    return (element, plane, (m, n, n_layers), orthogonal, offset_t,
            lattice_constant, pad_gap)


# --------------------------------------------------------------------- #
#  /api/modify/electrode  (single mode)                                 #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/electrode", methods=["POST"])
def api_modify_electrode():
    """Append one FCC slab, centred on the selected atom group.

    Body: ``{xyz, [...metadata...], element, plane, size:[m,n,n_layers],
             center_indices?, contact_distance?, side?, orthogonal?,
             offset?, lattice_constant?, inter_layer_offset?}``.

    ``center_indices`` is the selected atoms whose centroid the slab
    centres on (1 -> that atom, 2 -> midpoint, N -> centroid); omit /
    empty -> the world origin.  ``side`` is ``"+z"`` or ``"-z"``;
    ``contact_distance`` is the centre-to-closest-layer distance.  Per-side single mode is the
    one to use when the user wants asymmetric junctions (different
    metal / size on each side); for canonical pair junctions, prefer
    :func:`api_modify_symmetric_electrodes` which takes ``gap``
    directly.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        element, plane, size, orthogonal, offset, lat_a, pad_gap = \
            _parse_electrode_common(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    # Slab CENTRE = the centroid of ``center_indices`` (the user's selection);
    # omitted / empty -> the origin.  Same rule as the symmetric op (1 atom ->
    # that atom, 2 -> midpoint, N -> centroid).
    center_indices = body.get("center_indices")
    center_idx: Optional[List[int]]
    if center_indices is None:
        center_idx = None
    else:
        if not isinstance(center_indices, list):
            return _err(
                "'center_indices' must be omitted or a list of atom indices", 400)
        try:
            center_idx = [int(i) for i in center_indices]
        except (TypeError, ValueError):
            return _err("'center_indices' entries must be integers", 400)
        for i in center_idx:
            if not (0 <= i < struct.n_atoms):
                return _err(
                    f"center index {i} out of range for {struct.n_atoms}-atom "
                    f"structure", 400)
    try:
        contact_distance = _finite_float(
            "contact_distance", body.get("contact_distance", 2.4))
    except ValueError as exc:
        return _err(str(exc), 400)
    if contact_distance <= 0.0:
        return _err("'contact_distance' must be > 0 Å", 400)
    side = (body.get("side") or "+z").strip()
    if side not in ("+z", "-z"):
        return _err(f"'side' must be '+z' or '-z'; got {side!r}", 400)
    inter_layer_offset = body.get("inter_layer_offset")
    if inter_layer_offset is not None:
        try:
            inter_layer_offset = float(inter_layer_offset)
        except (TypeError, ValueError):
            return _err("'inter_layer_offset' must be numeric or null", 400)
    try:
        new_struct = _add_electrode_slab(
            struct, element, plane, size, center_idx,
            contact_distance=contact_distance,
            side=side,
            orthogonal=orthogonal,
            pad_interlayer_gap=pad_gap,
            offset=offset,
            lattice_constant=lat_a,
            inter_layer_offset=inter_layer_offset,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_electrode_slab failed: {exc}", 400)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview.md § 11.1, "Effect on atom count").
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/symmetric_electrodes  (pair mode; canonical junction)    #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/symmetric_electrodes", methods=["POST"])
def api_modify_symmetric_electrodes():
    """Append a collinear-z pair of FCC slabs perpendicular to z.

    Body: ``{xyz, [...metadata...], element, plane, size:[m,n,n_layers],
             gap?, anchors?, orthogonal?, offset?, lattice_constant?}``.

    ``gap`` is the canonical electrode-to-electrode z-distance (the
    empty space between the two slabs' closest layers).

    Slab placement (canonical, ``anchors`` omitted):

      top closest layer at z = +gap/2
      bot closest layer at z = -gap/2
      lateral xy centroid    = ``offset`` (default origin)

    The midpoint of the two slabs is at the world origin -- the
    Body: ``{xyz, [...metadata...], element, plane, size:[m,n,n_layers],
             gap?, center_indices?, orthogonal?, offset?, lattice_constant?}``.

    The junction CENTRE is the centroid of ``center_indices`` (the atoms the user
    selected): 1 atom centres on that atom, 2 on their midpoint, N on their
    centroid.  Omitted / empty -> the origin (0, 0, 0).  The molecule is NOT
    moved; if it doesn't fit the gap, ``validate_geometry`` advises (non-blocking).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        element, plane, size, orthogonal, offset, lat_a, pad_gap = \
            _parse_electrode_common(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    # Junction CENTRE = the centroid of ``center_indices`` (the user's selection);
    # omitted / empty -> the origin.  This generalises the old two-anchor midpoint
    # to any-size group (1 atom -> that atom, 2 -> midpoint, N -> centroid).
    center_indices = body.get("center_indices")
    center_idx: Optional[List[int]]
    if center_indices is None:
        center_idx = None
    else:
        if not isinstance(center_indices, list):
            return _err(
                "'center_indices' must be omitted or a list of atom indices", 400)
        try:
            center_idx = [int(i) for i in center_indices]
        except (TypeError, ValueError):
            return _err("'center_indices' entries must be integers", 400)
        for i in center_idx:
            if not (0 <= i < struct.n_atoms):
                return _err(
                    f"center index {i} out of range for {struct.n_atoms}-atom "
                    f"structure", 400)
    try:
        gap = _finite_float("gap", body.get("gap", 8.0))
    except ValueError as exc:
        return _err(str(exc), 400)
    if gap <= 0.0:
        return _err("'gap' must be > 0 Å", 400)
    try:
        new_struct = _add_symmetric_electrodes(
            struct, element, plane, size, center_idx,
            gap=gap,
            orthogonal=orthogonal,
            pad_interlayer_gap=pad_gap,
            offset=offset,
            lattice_constant=lat_a,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_symmetric_electrodes failed: {exc}", 400)
    except Exception as exc:  # noqa: BLE001 -- surface the real error as JSON, not a 500 HTML page
        current_app.logger.exception("symmetric_electrodes: unexpected error")
        return _err(
            f"add_symmetric_electrodes failed ({type(exc).__name__}): {exc}", 500)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview.md § 11.1, "Effect on atom count").
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/slab                                                     #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/slab", methods=["POST"])
def api_modify_slab():
    """Append ONE fcc slab, placed absolutely (redesign plan § 3).

    Body: ``{structure, element, plane, m, n, layers, start_registry,
    start_z, grow, stacking, orthogonal, dx, dy, lattice_constant?}``.

    IT READS NO SELECTION, and that is the design rather than an omission:
    ``dx``, ``dy`` and ``start_z`` are measured from the world origin, so the
    same numbers place the same slab whatever the user happens to have
    picked.  The client's OPERATIONS table says so too (`molview.md` § 11.1),
    which is why no `indices` key reaches here.

    Beside ``/api/modify/electrode``, not replacing it: the old panel stays
    until this one is proven (§ 3.4 lists what goes when it is).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)

    element = body.get("element")
    plane = body.get("plane")
    if not isinstance(element, str) or not element:
        return _err("missing 'element'", 400)
    if plane not in SUPPORTED_FCC_PLANES:
        return _err(f"'plane' must be one of {list(SUPPORTED_FCC_PLANES)}", 400)

    def _int(key, default, lo=None):
        raw = body.get(key, default)
        try:
            val = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a whole number; got {raw!r}")
        if lo is not None and val < lo:
            raise ValueError(f"'{key}' must be >= {lo}; got {val}")
        return val

    try:
        m = _int("m", 1, 1)
        n = _int("n", 1, 1)
        layers = _int("layers", 1, 0)
        start_registry = _int("start_registry", 0, 0)
        start_z = _finite_float("start_z", body.get("start_z", 0.0))
        dx = _finite_float("dx", body.get("dx", 0.0))
        dy = _finite_float("dy", body.get("dy", 0.0))
    except ValueError as exc:
        return _err(str(exc), 400)

    lat_raw = body.get("lattice_constant")
    lat_a = None
    if lat_raw is not None:
        try:
            lat_a = _finite_float("lattice_constant", lat_raw)
        except ValueError as exc:
            return _err(str(exc), 400)
        if lat_a <= 0:
            return _err("'lattice_constant' must be > 0 Å", 400)

    try:
        new_struct = _add_slab(
            struct, element, plane, (m, n, layers),
            start_registry=start_registry,
            start_z=start_z,
            grow=body.get("grow", "+z"),
            stacking=body.get("stacking", "continue"),
            orthogonal=bool(body.get("orthogonal", False)),
            offset=(dx, dy),
            lattice_constant=lat_a,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_slab failed: {exc}", 400)
    except Exception as exc:  # noqa: BLE001 -- surface it as JSON, not an HTML 500
        current_app.logger.exception("slab: unexpected error")
        return _err(f"add_slab failed ({type(exc).__name__}): {exc}", 500)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview.md § 11.1, "Effect on atom count").
    return _ok_response(new_struct,
                        extra={"notices": _seam_notices(new_struct, element,
                                                        plane)})


#: The seam's subject.  NOT ``"cell"``: that routes to the Cell page, and a
#: seam is not fixed there -- it is fixed by changing the layer count or the
#: placement, both of which are in this panel.  Saying its own subject is what
#: `periodicity_gate._notice` prescribes for a notice from another module, and
#: any subject but ``"cell"`` lands in the general place, which is visible from
#: either page.
_SEAM_ABOUT = "slab"


def _seam_notice(level: str, verdict: str, message: str):
    """One seam receipt, in the wire shape every notice uses.

    ``where`` is the STABLE ID, so a finding is identifiable without reading
    its prose -- the reason that field exists (periodicity_gate, 2026-08-03).
    """
    return {"level": level, "message": message,
            "where": f"slab.seam_{verdict}", "about": _SEAM_ABOUT}


def _seam_notices(struct, element: str, plane: str):
    """What the periodic boundary does to the crystal, as a note.

    A WARNING, NEVER A REFUSAL (bench-and-junction-plan.md § 2.4): an eclipsed
    seam is wrong for a periodic crystal and harmless in a relaxation whose
    outer layers are frozen, and only the user knows which they are running.

    **This has to be said at BUILD TIME, and this is the only moment it can
    be.**  Measured on the real `Au-BDT-Au` junction: the seam is bit-identical
    before and after relaxation -- 2.4008 Å, step (0.000, 0.000) both times --
    because the layers that form it are exactly the ones
    ``Geometry.Constraints`` pins.  No later check can catch it, because
    nothing later changes it.

    The measuring lives in `cell.classify_seam`; the phrasing lives here, the
    same split `_lattice_notes` uses.

    **It travels in the RECEIPTS slot**, which `ok_structure_response`
    documents as "what the edit did first, what is now true after it" and
    which had no caller until this one -- so the warning reaches the screen
    through `applyOp`'s existing handoff (`model-jobs.js`: "the structure AND
    what the server found true of it, in one handoff") and needs no display
    code of its own.  A second `notes` key beside it, which is what this
    returned at first, would have been a second door onto the same fact -- and
    the panel dropped it on the floor, because nothing was reading that door.
    """
    cell = getattr(struct, "cell", None)
    if cell is None:
        return []
    metal = np.asarray(
        [p for sym, p in zip(struct.elements, struct.positions)
         if sym == element], dtype=float)
    if metal.shape[0] < 2:
        return []
    seam = _classify_seam(metal, cell)

    if seam.verdict == "continues":
        want = STACKING_PERIOD.get(plane)
        if want and seam.period and seam.period != want:
            # Thin enough that it does not determine its own stacking: two
            # layers of (111) are A,B, which is fcc and hcp alike.
            return [_seam_notice("warn", "too_thin", (
                f"the boundary continues the stacking, but as a "
                f"{seam.period}-layer repeat -- fcc({plane}) has "
                f"{want}.  This slab is too thin to be the crystal you "
                f"asked for; add layers"))]
        return [_seam_notice("info", "continues", (
            f"the crystal continues across the periodic boundary: layers "
            f"{seam.z_room:.3f} Å apart, nearest atoms {seam.gap:.3f} Å"))]

    if seam.verdict == "vacuum":
        # Not a warning: an open face is what a slab calculation wants.
        return [_seam_notice("info", "vacuum", seam.message)]
    return [_seam_notice("warn", seam.verdict,
                         f"{seam.verdict}: {seam.message}")]


# --------------------------------------------------------------------- #
#  /api/modify/lattice-from-run                                         #
# --------------------------------------------------------------------- #


#: How far the second shell may sit from √2·d before it is worth a note.
#: 3% is wider than any relaxed crystal's jitter and far narrower than the
#: distortion that would make the fcc reading wrong.
_SECOND_SHELL_TOL = 0.03

#: fcc's own coordination number.  Anything else means the file is probably
#: not the bulk crystal the user meant to point at.
_FCC_COORDINATION = 12

#: How many atoms of the named element this route will measure over.
#:
#: The measurement is exact and therefore O(n²) -- every atom against every
#: atom's 27 surrounding images.  Measured: 500 atoms 0.5 s, 1372 atoms 5.2 s,
#: growing as the square from there, so an unbounded request can hang the
#: server on a file picked by accident.
#:
#: The cap is at the ROUTE, not in `cell.measure_fcc`: a module measures
#: whatever it is handed (the reasoning web-api.md § 2.1 gives for where a
#: fence belongs), and a request-time budget is the route's business.  It
#: REFUSES rather than sampling, because a silently truncated answer to "what
#: is this crystal's lattice constant" is worse than being asked for a smaller
#: cell -- and a lattice constant does not need a thousand atoms.
_MAX_ATOMS = 1000


@bp.route("/api/modify/lattice-from-run", methods=["POST"])
def api_modify_lattice_from_run():
    """Read a lattice constant back out of the user's own relaxed result.

    Body: ``{path, element?}`` -> ``{ok, element, a, d_nn, coordination,
    second_shell_ratio, n_atoms, source, notes}``.

    THE DIVISION OF LABOUR IS THE USER'S OWN (plans/modify-redesign-plan.md
    § 3.3): *"the user needs to make sure this setup is correct, and the
    backend just extracts the lattice from that result."*  So **they**
    guarantee the pseudopotential, basis and mesh cutoff; this measures what
    the file says and reports what looks odd, without refusing on their
    behalf.

    IT MEASURES THE ATOMS, NOT THE CELL, and `cell.measure_fcc` carries why:
    the box may be conventional, primitive, or their own layered lead cell,
    and the file does not say which.

    Two refusals, because guessing would be worse than stopping: a file with
    **no cell** (no periodic images, so on a small cell the measured minimum
    is simply wrong), and **more than one element with none named**.
    Everything else is a note.
    """
    from .files import _PickerError, _resolve_within_roots

    body = request.get_json(silent=True) or {}
    raw = body.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return _err("missing 'path'", 400)
    # THE FENCE IS AT THE ROUTE (web-api.md § 2.1): an untrusted path is
    # resolved through the one primitive before anything opens it.
    try:
        path = _resolve_within_roots(raw)
    except _PickerError as exc:
        return _err(exc.message, exc.status)
    if not path.is_file():
        return _err(f"{path.name} is not a file", 400)

    element = body.get("element")
    if element is not None and not isinstance(element, str):
        return _err("'element' must be a chemical symbol", 400)
    element = element.strip() if isinstance(element, str) else None

    try:
        elements, positions, cell = _read_relaxed_result(path)
    except (ValueError, OSError) as exc:
        return _err(f"could not read {path.name}: {exc}", 400)
    except Exception as exc:  # noqa: BLE001 -- a bad file is the user's, not a crash
        # A malformed sidecar, a truncated deck, a Z this build has no symbol
        # for: the readers raise their own kinds, and the ones they do not
        # name still have to reach the user as JSON.  The house pattern in
        # this file (the electrode routes) logs and answers, rather than
        # letting Flask render an HTML 500 into a fetch() that expects JSON.
        current_app.logger.exception("lattice-from-run: unexpected read error")
        return _err(
            f"could not read {path.name} ({type(exc).__name__}): {exc}", 400)

    if cell is None:
        return _err(
            f"{path.name} carries no unit cell. A lattice constant is measured "
            f"against the crystal's periodic images, so a file without a cell "
            f"has none to measure against.", 400)

    present = sorted(set(elements))
    if element is None:
        if len(present) != 1:
            return _err(
                f"{path.name} holds {len(present)} elements "
                f"({', '.join(present)}); name which one to measure.", 400)
        element = present[0]
    elif element not in present:
        return _err(
            f"{path.name} holds no {element} (it holds "
            f"{', '.join(present) or 'nothing'}).", 400)

    keep = [i for i, sym in enumerate(elements) if sym == element]
    if len(keep) > _MAX_ATOMS:
        return _err(
            f"{path.name} holds {len(keep)} {element} atoms; this measurement "
            f"is exact and grows as the square of the count, so it is capped "
            f"at {_MAX_ATOMS}. A lattice constant does not need more than a "
            f"few unit cells — point at a smaller relaxed cell.", 400)
    try:
        measured = _measure_fcc(np.asarray(positions)[keep], cell)
    except ValueError as exc:
        return _err(str(exc), 400)

    notes = _lattice_notes(measured, element)
    ratio = (measured.second_shell / measured.d_nn
             if measured.second_shell else None)
    return jsonify({
        "ok":                 True,
        "element":            element,
        "a":                  measured.a,
        "d_nn":               measured.d_nn,
        "coordination":       measured.coordination,
        "second_shell_ratio": ratio,
        "n_atoms":            measured.n_atoms,
        "source":             path.name,
        "notes":              notes,
    })


def _read_relaxed_result(path):
    """``(elements, positions_ang, cell_ang_or_None)`` from a result file.

    Two readers, and both already existed (§ 3.3): SIESTA's ``.XV`` through
    ``transport.compose.read_xv``, and everything else through
    ``StructureCodec``, which is the ONE authority on the ``.xyz`` +
    ``.molstruct.json`` pair.  Nothing here parses a file itself.
    """
    if path.suffix.lower() == ".xv":
        # THE PARSE MODULE'S READER, not `transport.compose`'s.  Reading files
        # is the parse module's job, and this one returns the cell as a
        # first-class field, which is the thing being measured against.
        #
        # Choosing between them used to matter for a second reason: the two
        # carried DIFFERENT Bohr radii, so the same file gave coordinates 4e-7
        # apart depending on which was asked.  That is fixed -- every
        # conversion in the tree now reads `molbuilder/constants.py` -- and is
        # recorded here because this route is where it surfaced, as a test
        # comparing the two answers and failing by 1.6e-6 Å.
        from molbuilder.parse.coords.siesta_xv import (
            SiestaXVError, read_xv, read_xv_cell,
        )
        try:
            struct = read_xv(path)
            cell = read_xv_cell(path)
        except SiestaXVError as exc:
            raise ValueError(str(exc)) from exc
        return list(struct.elements), struct.positions, cell

    from molbuilder.workingcopy_structure import StructureCodec
    struct = StructureCodec().read(path)
    return list(struct.elements), struct.positions, struct.cell


def _lattice_notes(measured, element: str):
    """What looks odd about this file, as notes rather than refusals.

    The setup is the user's to own, so nothing here stops the answer being
    used — each row says what was expected, what was found, and what that
    usually means, and leaves the judgement where it belongs.
    """
    notes = []

    if measured.coordination != _FCC_COORDINATION:
        notes.append({
            "level": "warn",
            "message": (
                f"Each {element} has {measured.coordination} neighbours at this "
                f"distance; bulk fcc has {_FCC_COORDINATION}. That usually means "
                f"the file is not the bulk crystal you meant — a slab with "
                f"vacuum, a surface, or a defect."),
        })

    if measured.second_shell is None:
        notes.append({
            "level": "warn",
            "message": (
                "There is no second neighbour shell to check, so the fcc "
                "signature could not be confirmed."),
        })
    else:
        ratio = measured.second_shell / measured.d_nn
        if abs(ratio - np.sqrt(2.0)) > _SECOND_SHELL_TOL:
            notes.append({
                "level": "warn",
                "message": (
                    f"The second shell sits at {ratio:.3f}×the first; fcc puts "
                    f"it at √2 = 1.414. A different ratio means this is not a "
                    f"cubic close-packed crystal."),
            })

    # AND HOW IT COMPARES, which is the cross-check that catches the one
    # mistake anyone makes: a second-shell pair reads a factor √2 high and
    # lands ~41% from both references.  Done here rather than in the panel
    # because the table is already loaded here, and two homes for one
    # subtraction is one home too many.
    try:
        table = _load_fcc_lattice_full().get(element, {})
    except Exception:                      # noqa: BLE001 -- a missing table is not this route's failure
        table = {}
    for key, label in (("a_experimental", "experimental"), ("a_pbe", "PBE")):
        ref = table.get(key)
        if not ref:
            continue
        off = (measured.a - ref) / ref * 100.0
        notes.append({
            "level": "info" if abs(off) < 5.0 else "warn",
            "message": f"{off:+.1f}% from {label} ({ref:.4f} Å)",
        })
    return notes
