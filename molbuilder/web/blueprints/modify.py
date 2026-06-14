"""Modify blueprint -- per-atom edit-op routes for the Modify tab.

Routes (no url_prefix; each carries its own full path):

    POST /api/modify/load                  validate / canonicalise XYZ
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

from typing import List, Optional, Tuple

from flask import Blueprint, jsonify, request

from ._shared import (
    err as _err,
    finite_float as _finite_float,
    ok_structure_response as _ok_response,
    struct_from_body as _struct_from_body,
)

from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    SUPPORTED_FCC_PLANES,
    add_atom as _add_atom,
    add_electrode_slab as _add_electrode_slab,
    add_symmetric_electrodes as _add_symmetric_electrodes,
    compute_selection_remap_after_add as _selection_remap_after_add,
    compute_selection_remap_after_delete as _selection_remap_after_delete,
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
    return jsonify({
        "ok":           True,
        "fcc_elements": list(SUPPORTED_FCC_ELEMENTS),
        "fcc_planes":   list(SUPPORTED_FCC_PLANES),
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
    # Workspace-state Phase 3 (2026-06-07): emit selection_remap so
    # the client dispatcher can correctly translate the user's
    # selection across the index shift.  Pre-fix, the client's
    # naive "filter to in-range" check silently dropped surviving
    # high-index atoms (selecting atom 2, deleting atom 0, ended
    # up with empty selection instead of new index 1).
    return _ok_response(new_struct, extra={
        "selection_remap": _selection_remap_after_delete(
            struct, indices_int),
    })


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
    # Workspace-state Phase 3: identity remap (add appends at the
    # next index, every pre-op index survives unchanged).  We emit
    # it anyway so the client dispatcher can apply one uniform
    # per-op rule (look up ``extra["selection_remap"]``) instead of
    # growing a "no remap means preserve" special case for add ops.
    return _ok_response(new_struct, extra={
        "selection_remap": _selection_remap_after_add(struct),
    })


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
    """Append one FCC slab on either side of one anchor.

    Body: ``{xyz, [...metadata...], element, plane, size:[m,n,n_layers],
             anchor_index, contact_distance?, side?, orthogonal?,
             offset?, lattice_constant?, inter_layer_offset?}``.

    ``side`` is ``"+z"`` or ``"-z"``; ``contact_distance`` is the
    anchor-to-closest-layer distance.  Per-side single mode is the
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
        element, plane, size, orthogonal, offset, lat_a = \
            _parse_electrode_common(body)
    except ValueError as exc:
        return _err(str(exc), 400)
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
            struct, element, plane, size, anchor_index,
            contact_distance=contact_distance,
            side=side,
            orthogonal=orthogonal,
            offset=offset,
            lattice_constant=lat_a,
            inter_layer_offset=inter_layer_offset,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_electrode_slab failed: {exc}", 400)
    # Phase 3: slab appends pre-op atoms keep their indices —
    # identity remap so the client dispatcher's per-op rule stays
    # uniform across every atom-count-changing op.
    return _ok_response(new_struct, extra={
        "selection_remap": _selection_remap_after_add(struct),
    })


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
    user is expected to centre the molecule (``/api/modify/translate``
    with ``recenter=true``) and pose it (Pose subtab) before the
    slab op so the molecule sits in the gap exactly the way they
    want it.  Anchors aren't needed: slab placement is fully
    determined by ``gap`` and (optionally) ``offset``.

    Legacy mode: pass ``anchors=[a_top, a_bot]`` to centre the slab
    midpoint on the anchor-pair midpoint instead of the origin.
    Useful when the molecule is NOT pre-centred and the caller wants
    the slabs to flank a specific atom pair.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    try:
        element, plane, size, orthogonal, offset, lat_a = \
            _parse_electrode_common(body)
    except ValueError as exc:
        return _err(str(exc), 400)
    anchors = body.get("anchors")
    anchor_indices: Optional[Tuple[int, int]]
    if anchors is None:
        anchor_indices = None
    else:
        if (not isinstance(anchors, list)) or len(anchors) != 2:
            return _err(
                "'anchors' must be omitted or a 2-element [a_top, a_bot] list",
                400,
            )
        try:
            a_top, a_bot = int(anchors[0]), int(anchors[1])
        except (TypeError, ValueError):
            return _err("anchor indices must be integers", 400)
        if a_top == a_bot:
            return _err("anchors must be two distinct atom indices", 400)
        if not (0 <= a_top < struct.n_atoms and 0 <= a_bot < struct.n_atoms):
            return _err(
                f"anchor indices ({a_top}, {a_bot}) out of range for "
                f"{struct.n_atoms}-atom structure",
                400,
            )
        anchor_indices = (a_top, a_bot)
    try:
        gap = _finite_float("gap", body.get("gap", 8.0))
    except ValueError as exc:
        return _err(str(exc), 400)
    if gap <= 0.0:
        return _err("'gap' must be > 0 Å", 400)
    try:
        new_struct = _add_symmetric_electrodes(
            struct, element, plane, size, anchor_indices,
            gap=gap,
            orthogonal=orthogonal,
            offset=offset,
            lattice_constant=lat_a,
        )
    except (ValueError, NotImplementedError) as exc:
        return _err(f"add_symmetric_electrodes failed: {exc}", 400)
    # Phase 3: identity remap for the pre-op atoms (slabs appended
    # at the end) — same rule as add_atom / electrode_slab.
    return _ok_response(new_struct, extra={
        "selection_remap": _selection_remap_after_add(struct),
    })
