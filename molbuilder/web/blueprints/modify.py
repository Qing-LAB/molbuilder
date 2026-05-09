"""Modify blueprint -- per-atom edit-op routes for the Modify tab.

Routes (no url_prefix; each carries its own full path):

    POST /api/modify/load        validate / canonicalise an XYZ payload
    POST /api/modify/delete      delete_atoms(indices)
    POST /api/modify/add_atom    add_atom(element, anchor, offset)
    POST /api/modify/orient      orient_along_axis(anchors, axis,
                                                   angle, center)
    POST /api/modify/rotate      rotate_around_axis(axis, angle)

M3 covers the per-atom ops.  M4 added /api/modify/orient and
/api/modify/rotate; M5 adds the electrode endpoints.

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
preserved" invariant in `docs/spec/modify-tab.md` § 5).

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

from typing import List, Optional

from flask import Blueprint, request

from ._shared import (
    err as _err,
    ok_structure_response as _ok_response,
    struct_from_body as _struct_from_body,
)

from molbuilder.modify import (
    add_atom as _add_atom,
    delete_atoms as _delete_atoms,
    orient_along_axis as _orient_along_axis,
    rotate_around_axis as _rotate_around_axis,
)


bp = Blueprint("modify", __name__)


# --------------------------------------------------------------------- #
#  /api/modify/load                                                     #
# --------------------------------------------------------------------- #


@bp.route("/api/modify/load", methods=["POST"])
def api_modify_load():
    """Validate an XYZ payload + echo back the canonical re-parsed
    structure.  Catches malformed input early so a chain of edit ops
    doesn't proceed against a half-broken structure.
    """
    body = request.get_json(silent=True) or {}
    try:
        struct = _struct_from_body(body)
    except (ValueError, Exception) as exc:        # noqa: BLE001
        return _err(f"could not parse xyz: {exc}", 400)
    return _ok_response(struct)


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
    except Exception as exc:                       # noqa: BLE001
        return _err(f"delete_atoms failed: {exc}", 400)
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
    except Exception as exc:                       # noqa: BLE001
        return _err(f"add_atom failed: {exc}", 400)
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
    angle = body.get("angle", 0.0)
    try:
        angle_f = float(angle)
    except (TypeError, ValueError):
        return _err("angle must be a number (degrees)", 400)
    try:
        new_struct = _orient_along_axis(
            struct, (a0, a1), axis=axis, angle=angle_f, center=center,
        )
    except (ValueError, Exception) as exc:        # noqa: BLE001
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
    angle = body.get("angle")
    try:
        angle_f = float(angle)
    except (TypeError, ValueError):
        return _err("angle must be a number (degrees)", 400)
    try:
        new_struct = _rotate_around_axis(struct, axis=axis, angle=angle_f)
    except Exception as exc:                       # noqa: BLE001
        return _err(f"rotate_around_axis failed: {exc}", 400)
    return _ok_response(new_struct)
