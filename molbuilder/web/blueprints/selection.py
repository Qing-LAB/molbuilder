"""Selection blueprint -- rule evaluator + atoms + sidecar I/O (L2).

The selection system is layered:

  * L1 (:mod:`molbuilder.selection`) -- pure-Python rule dataclasses
    + evaluator + JSON round-trip.  Independent of Flask, used by
    engines + tests + this blueprint.
  * **L2 (this module)** -- HTTP endpoints that turn a JSON rule
    tree + a structure path into a list of selected atom indices,
    plus the atom-list read and sidecar save endpoints.  Stateless:
    every request is self-contained, the server stores no per-user
    selection state.
  * L3 (``lib/molview/_selection-store.js`` since Phase 9 /
    2026-06-13) -- workspace-internal JS state holder (atoms,
    selection, filters, mode, error).  One process-wide instance
    owned by the workspace dispatcher; external consumers reach
    it via ``window.molbuilder.workspace.selection.*``
    (=``ws.selection.*``).  Posts to L2 only on ``applyFilter``
    and ``writeLabel``; click toggles are client-side.
  * L4 (``lib/selection-panel.js`` + ``lib/selection/viewer-adapter.js``)
    -- DOM panel + 3Dmol overlay/click consumer.  Both consume
    the L3 store via ``ws.selection.*`` and call its mutators on
    user action.

See ``docs/web/molview.md`` for the full module
contract, including the public API surface of the store.

Endpoints
---------

``POST /api/selection/atoms``
    Return the atom list for a structure (one row per atom, with
    element + optional PDB metadata + region tags + fixed flag).
    The panel fetches this once per structure load to populate the
    card's scrollable atom list.

``POST /api/selection/eval``
    Evaluate a rule against a structure on disk.

    Body::

        {
          "structure_path": "/abs/path/to/relaxed.xyz"   (or .pdb),
          "rule": {<rule-json>}
        }

    Response::

        {
          "selected_indices": [0, 1, 2, ...],
          "count": N,
          "n_atoms_total": M
        }

The structure_path is validated against the same allow-list as the
files blueprint (path must resolve inside a configured root); on
failure the response is HTTP 400 with a JSON error.

Reading the structure: dispatch by file extension --
``.xyz`` -> :func:`Structure.from_xyz`,
``.pdb`` -> :func:`Structure.from_pdb`.
Any other extension is rejected at the endpoint boundary with a
clear "unsupported structure extension" error.  If a
``<basename>.molstruct.json`` sidecar sits next to the structure
file, its ``regions`` + ``frozen_atoms`` are applied to the
Structure so :class:`ByRegion` rules can resolve and the
reserved ``frozen`` label on each atom-list row reflects the sidecar.
Missing sidecar is fine -- selection still works for everything
that doesn't reference a region.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from molbuilder.selection import (
    Rule, SelectionError, evaluate,
    from_json as rule_from_json,
)
from molbuilder.structure import Structure

# NO PATH VALIDATOR, AND NOTHING THAT READS A FILE.  This blueprint took a
# `structure_path`, resolved it inside the picker roots and read the pair
# itself until 2026-09-07 -- a second reader that had drifted to applying
# only the sidecar's `regions`.  The browser sends the atoms it is looking
# at; there is no path to fence any more.
from .files import _PickerError

bp = Blueprint("selection", __name__)


# --------------------------------------------------------------------- #
#  Structure loader                                                     #
# --------------------------------------------------------------------- #


_SUPPORTED_STRUCTURE_SUFFIXES = (".xyz", ".pdb")


def _bad_request(msg: str, status: int = 400):
    # Uniform error envelope (`web/web-api.md` § 1 -- the four
    # status buckets; decided 2026-05-25).  Both ``ok``
    # and ``error`` are load-bearing so a future client wrapper that
    # checks ``body.ok`` (as opposed to HTTP status) reads consistently
    # across blueprints.
    return jsonify({"ok": False, "error": msg}), status


def _parse_request_payload(req) -> Dict[str, Any]:
    """Common payload extraction + shape validation."""
    if not req.is_json:
        raise _PickerError(400, "request body must be JSON")
    payload = req.get_json(silent=True)
    if not isinstance(payload, dict):
        raise _PickerError(400, "request body must be a JSON object")
    return payload


def _load_rule_from_payload(payload: Dict[str, Any]) -> Rule:
    raw = payload.get("rule")
    if raw is None:
        raise _PickerError(400, "missing 'rule' in request body")
    try:
        return rule_from_json(raw)
    except SelectionError as exc:
        raise _PickerError(400, f"invalid rule: {exc}")


def _struct_from_atoms(atoms: list) -> Structure:
    """Build a Structure from the workspace's IN-MEMORY atom list (the store's
    ``atoms``) so a filter evaluates against MEMORY, not the stale saved file
    (`web/molview.md` § 9.5). The filter rules are label/element/index/
    residue only -- no geometry -- so positions are placeholders. The workspace
    module (ws.*) is the single source of truth."""
    if not isinstance(atoms, list):
        raise ValueError("'atoms' must be a list")
    elements: list = []
    regions: Dict[str, list] = {}
    residue_names: list = []
    for i, a in enumerate(atoms):
        if not isinstance(a, dict):
            raise ValueError(f"atoms[{i}] must be an object")
        elements.append(str(a.get("element") or "X"))
        for label in (a.get("labels") or a.get("regions") or []):
            if isinstance(label, str) and label:
                regions.setdefault(label, []).append(i)
        residue_names.append(
            str(a.get("residueName") or a.get("residue_name") or "MOL"))
    n = len(elements)
    return Structure(
        elements=elements,
        positions=[[0.0, 0.0, 0.0] for _ in range(n)],
        regions={k: sorted(set(v)) for k, v in regions.items()},
        residue_names=residue_names,
    )


@bp.route("/api/selection/eval", methods=["POST"])
def selection_eval():
    """Evaluate a rule against the workspace and return the selected indices.

    Preferred body (Modify): ``{atoms: [...store atoms...], rule}`` -- evaluate
    against the IN-MEMORY workspace (`web/molview.md` § 9.5), so filters
    reflect unsaved edits.  Legacy/Results body ``{structure_path, rule}`` still
    loads the file on disk (a saved result legitimately lives there)."""
    try:
        payload = _parse_request_payload(request)
        # THE BROWSER SENDS THE ATOMS IT IS LOOKING AT.  MolView holds the
        # structure; a filter is answered against that, never against a file
        # -- the file on disk may not be what is on screen, which is the bug
        # `71729ff9` fixed ("assign a label in the panel, filter by it, and
        # the filter read the stale saved file").  That commit added this
        # branch and left the disk one beside it, for a case Results never
        # used; the disk branch and its private reader are gone 2026-09-07.
        atoms = payload.get("atoms")
        if not isinstance(atoms, list):
            return _bad_request("missing 'atoms'")
        try:
            struct = _struct_from_atoms(atoms)
        except ValueError as exc:
            return _bad_request(f"invalid 'atoms': {exc}")
        rule = _load_rule_from_payload(payload)
        try:
            indices = evaluate(rule, struct)
        except SelectionError as exc:
            return _bad_request(f"evaluation failed: {exc}")
        return jsonify({
            "ok":               True,
            "selected_indices": sorted(indices),
            "count":            len(indices),
            "n_atoms_total":    len(struct.elements),
        })
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


