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
preserved" invariant in `docs/tabs/molbuilder.md` § 5).

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
    apply_labels_to_struct as _apply_labels_to_struct,
)

from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    SUPPORTED_FCC_PLANES,
    add_atom as _add_atom,
    add_electrode_slab as _add_electrode_slab,
    add_symmetric_electrodes as _add_symmetric_electrodes,
    delete_atoms as _delete_atoms,
    orient_along_axis as _orient_along_axis,
    rotate_around_axis as _rotate_around_axis,
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
    return jsonify({
        "ok":            True,
        "fcc_elements":  list(SUPPORTED_FCC_ELEMENTS),
        "fcc_planes":    list(SUPPORTED_FCC_PLANES),
        "lattice_table": lattice_table,
        "lattice_error": lattice_error,
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
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
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
    # change (molview-module.md §19.3.2) -- a cleared selection can never
    # mis-point at a shifted index, so the server does not compute a remap.
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
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
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
    # change (molview-module.md §19.3.2).
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
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
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
    """Rotate every atom by ``angle`` degrees (right-hand rule) around
    the named axis through the origin.

    Body: ``{xyz, [...metadata...], axis, angle}``.

    Useful for redirecting a tilted molecule's azimuth after an
    ``orient`` op with non-zero angle (e.g. spin a tilt from the
    xz-plane into the yz-plane via ``axis=z, angle=90``).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
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
    try:
        new_struct = _rotate_around_axis(
            struct, axis=axis, angle=angle_f, center=center,
        )
    except ValueError as exc:
        return _err(f"rotate_around_axis failed: {exc}", 400)
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  /api/modify/translate                                                #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/translate", methods=["POST"])
def api_modify_translate():
    """Translate every atom rigidly.

    Two modes (mutually exclusive; ``recenter`` wins if both):

    * ``{recenter: true}`` -- translate so the geometric centroid
      lands on the origin.  Useful after adding electrode slabs
      shifts the structure off-axis: re-anchoring the centroid
      makes mouse-zoom feel sane and aligns subsequent slab ops
      against a predictable origin.
    * ``{dx, dy, dz}`` (Å) -- translate by the given vector.
      Each component defaults to 0.

    The op is rigid: bonds, angles, residue assignments, and
    selection indices are all preserved (callers should keep their
    ``state.selected`` indices across the round-trip; only the
    coordinates change).
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
    if bool(body.get("recenter", False)):
        try:
            new_struct = struct.centered()
        except ValueError as exc:
            return _err(f"recenter failed: {exc}", 400)
        return _ok_response(new_struct)
    try:
        dx = _finite_float("dx", body.get("dx", 0.0))
        dy = _finite_float("dy", body.get("dy", 0.0))
        dz = _finite_float("dz", body.get("dz", 0.0))
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        new_struct = struct.translated((dx, dy, dz))
    except ValueError as exc:
        return _err(f"translate failed: {exc}", 400)
    return _ok_response(new_struct)


# --------------------------------------------------------------------- #
#  Electrode-shared parsing helpers                                     #
# --------------------------------------------------------------------- #


def _parse_electrode_common(body):
    """Validate and unpack the fields the two electrode endpoints
    share: element, plane, size, orthogonal, offset, lattice_constant.

    Returns ``(element, plane, size, orthogonal, offset, lattice_constant)``
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
    return (element, plane, (m, n, n_layers), orthogonal, offset_t,
            lattice_constant)


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
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
    try:
        element, plane, size, orthogonal, offset, lat_a = \
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
            offset=offset,
            lattice_constant=lat_a,
            inter_layer_offset=inter_layer_offset,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_electrode_slab failed: {exc}", 400)
    # No selection_remap: the client CLEARS the selection on any atom-count
    # change (molview-module.md §19.3.2).
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
    _label_notice = _apply_labels_to_struct(struct, body)
    if _label_notice:
        return _err(_label_notice, 400)
    try:
        element, plane, size, orthogonal, offset, lat_a = \
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
    # change (molview-module.md §19.3.2).
    return _ok_response(new_struct)
