"""Selection blueprint -- rule evaluator (L2).

The selection system is layered:

  * L1 (:mod:`molbuilder.selection`) -- pure-Python rule dataclasses
    + evaluator + JSON round-trip.  Independent of Flask, used by
    engines + tests + this blueprint.
  * **L2 (this module)** -- HTTP endpoint that turns a JSON rule
    tree into a list of selected atom indices.  Stateless:
    every request is self-contained, the server stores no per-user
    selection state.

See ``docs/web/molview.md`` for the full module
contract, including the public API surface of the store.

Endpoints
---------

``POST /api/selection/eval``
    Evaluate a rule against the workspace's atoms.

    Response::

        {
          "selected_indices": [0, 1, 2, ...],
          "count": N,
          "n_atoms_total": M
        }
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from molbuilder.selection import (
    Rule, SelectionError, evaluate,
    from_json as rule_from_json,
)
from molbuilder.structure import Structure

# NO PATH VALIDATOR, AND NOTHING THAT READS A FILE: the browser sends the
# atoms it is looking at.
from .files import _PickerError

bp = Blueprint("selection", __name__)


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
    atom_names: list = []
    chain_ids: list = []
    for i, a in enumerate(atoms):
        if not isinstance(a, dict):
            raise ValueError(f"atoms[{i}] must be an object")
        element = str(a.get("element") or "X")
        elements.append(element)
        for label in (a.get("labels") or a.get("regions") or []):
            if isinstance(label, str) and label:
                regions.setdefault(label, []).append(i)
        residue_names.append(
            str(a.get("residueName") or a.get("residue_name") or "MOL"))
        # WHAT THE CALLER ACTUALLY HOLDS, when it holds it.  `by_atom_name`
        # and `by_chain_id` are rule kinds (`selection.py` ByAtomName /
        # ByChainId); a caller without the columns gets `Structure`'s
        # defaults -- atom name = element symbol, chain = "A".
        atom_names.append(str(a.get("atomName") or a.get("atom_name")
                              or element))
        chain_ids.append(str(a.get("chainId") or a.get("chain_id") or "A"))
    n = len(elements)
    return Structure(
        elements=elements,
        positions=[[0.0, 0.0, 0.0] for _ in range(n)],
        regions={k: sorted(set(v)) for k, v in regions.items()},
        residue_names=residue_names,
        atom_names=atom_names,
        chain_ids=chain_ids,
    )


@bp.route("/api/selection/eval", methods=["POST"])
def selection_eval():
    """Evaluate a rule against the workspace and return the selected indices.

    Preferred body (Modify): ``{atoms: [...store atoms...], rule}`` -- evaluate
    against the IN-MEMORY workspace (`web/molview.md` § 9.5), so filters
    reflect unsaved edits."""
    try:
        payload = _parse_request_payload(request)
        # THE BROWSER SENDS THE ATOMS IT IS LOOKING AT.  MolView holds the
        # structure; a filter is answered against that, never against a file
        # -- the file on disk may not be what is on screen.
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


