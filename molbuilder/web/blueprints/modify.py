"""Modify blueprint -- per-atom edit-op routes for the Modify tab.

Routes (no url_prefix; each carries its own full path):

    POST /api/modify/load        validate / canonicalise an XYZ payload
    POST /api/modify/delete      delete_atoms(indices)
    POST /api/modify/add_atom    add_atom(element, anchor, offset)

M3 covers the per-atom ops.  M4 adds /api/modify/orient and
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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from molbuilder.modify import (
    add_atom as _add_atom,
    delete_atoms as _delete_atoms,
)
from molbuilder.structure import Structure
from molbuilder.validation import validate_geometry


bp = Blueprint("modify", __name__)


def _issues_to_json(issues):
    """Same serialiser as build.py.  Duplicated here rather than
    imported across blueprints; if the schema changes, both copies
    need to update -- but they're trivially small and not worth a
    cross-blueprint shim."""
    return [{"severity": i.severity, "message": i.message, "where": i.where}
            for i in issues]


def _struct_from_body(body: Dict[str, Any]) -> Structure:
    """Reconstruct a Structure from the canonical body shape.

    Reads ``xyz`` (required) and the four optional metadata lists.
    A metadata list is honoured only when its length matches the
    atom count; otherwise the default from ``Structure.from_xyz``
    (atom_names = elements, residue_ids = [1]*n, residue_names =
    ["MOL"]*n, chain_ids = ["A"]*n) is kept so a malformed metadata
    array can't corrupt the result.
    """
    xyz = body.get("xyz") or ""
    if not isinstance(xyz, str) or not xyz.strip():
        raise ValueError("missing or empty 'xyz'")
    title = body.get("title") or None
    s = Structure.from_xyz(xyz, title=title)
    n = s.n_atoms
    for attr in ("atom_names", "residue_ids", "residue_names", "chain_ids"):
        v = body.get(attr)
        if v is None:
            continue
        if not isinstance(v, list) or len(v) != n:
            # Silently drop malformed metadata; the default from
            # from_xyz is already in place.  An explicit error here
            # would be friendlier UX but the spec keeps the wire
            # shape forgiving.
            continue
        setattr(s, attr, list(v))
    return s


def _ok_response(struct: Structure):
    """Serialise a Structure + run validate_geometry into the canonical
    response shape (matches /api/build/load + adds an issues array)."""
    issues = validate_geometry(struct)
    return jsonify({
        "ok":            True,
        "xyz":           struct.to_xyz(),
        "elements":      list(struct.elements),
        "atom_names":    list(struct.atom_names),
        "residue_ids":   list(struct.residue_ids),
        "residue_names": list(struct.residue_names),
        "chain_ids":     list(struct.chain_ids),
        "n_atoms":       struct.n_atoms,
        "n_residues":    struct.n_residues,
        "title":         struct.title or "",
        "issues":        _issues_to_json(issues),
    })


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


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
