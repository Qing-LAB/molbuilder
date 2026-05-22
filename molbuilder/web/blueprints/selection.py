"""Selection blueprint -- evaluator + click-toggle endpoints (L2).

The selection system is layered:

  * L1 (:mod:`molbuilder.selection`) -- pure-Python rule dataclasses
    + evaluator + JSON round-trip.  Independent of Flask, used by
    engines + tests + this blueprint.
  * **L2 (this module)** -- HTTP endpoints that turn a JSON rule
    tree + a structure path into a list of selected atom indices,
    plus the atom-list read and sidecar save endpoints.  Stateless:
    every request is self-contained, the server stores no per-user
    selection state.
  * L3 (``lib/selection/store.js``) -- canonical JS state holder
    (atoms, selection, filters, mode, error).  Singleton; the SOLE
    cross-module signal bus for L4.  Posts to L2 only on
    ``applyFilter`` and ``writeLabel``; click toggles are
    client-side.
  * L4 (``lib/selection-panel.js`` + ``lib/selection/viewer-adapter.js``)
    -- DOM panel + 3Dmol overlay/click consumer.  Both subscribe
    to the L3 store and call its mutators on user action.

See ``docs/protocols/atom-selection.md`` for the full module
contract, including the public API surface of the store.

Endpoints
---------

``POST /api/selection/atoms``
    Return the atom list for a structure (one row per atom, with
    element + optional PDB metadata + region tags + fixed flag).
    The panel fetches this once per structure load to populate the
    card's scrollable atom list.

``POST /api/selection/save``
    Persist a materialised selection into the structure's
    ``.molstruct.json`` sidecar.  Body:

        {
          "structure_path": "...",
          "target":         "L-electrode" | "R-electrode" | "bridge"
                            | "frozen_atoms" | "<new region label>",
          "indices":        [0, 1, 2, ...],
          "rule":           {<rule-json>}   # optional, stored as
                                            # selection_rules[target]
        }

    Behaviour:
      * Creates the sidecar from scratch if none exists.
      * Assigning to a region SETS that region's membership to the
        given indices (REPLACE semantics) -- but does NOT remove
        those atoms from other regions.  An atom may carry
        multiple labels (e.g. ``"L-electrode"`` + ``"interface"``).
        Use the per-tag × button in the atom list to remove a
        single atom from one label.
      * ``frozen_atoms`` is independent of regions; assigning to
        it overwrites the existing frozen-atom list verbatim.
      * Empty ``indices`` removes the target region (or clears
        frozen_atoms) entirely.
      * Writes atomically via ``parsers.molstruct_json.save``.
      * Returns the updated sidecar dict so the client can refresh
        its atom list without an extra round-trip.

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

``POST /api/selection/toggle``
    Toggle an atom's membership in the rule's ``ByClick`` clause.
    This is the bookkeeping the user asked for: 3Dmol fires
    ``onclick(atom)`` with atom index N, JS posts that index here,
    Python flips its membership in the rule, returns the canonical
    rule + the new evaluated set.

    The bookkeeping rule:

      * If the existing rule already has a top-level ``ByClick`` (or
        an ``Or`` whose operands include one), N is toggled in that
        clause.
      * If no ``ByClick`` clause exists, one is created and the
        rule is wrapped in ``Or(existing_rule, ByClick([N]))`` (so
        the algorithmic part of the selection is preserved).
      * The ``All()`` rule is a special case: clicking on top of
        "everything" produces ``Minus(All(), ByClick([N]))`` --
        the click DEselects (since N was already selected).  This
        is the same semantics whenever the index is already in the
        evaluated set: a click deselects rather than re-selects.

    Body::

        {
          "structure_path": "...",
          "rule": {<rule-json>},
          "index": N
        }

    Response::

        {
          "rule": {<canonical-rule-json>},
          "selected_indices": [...],
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
``is_frozen`` flag on each atom-list row reflects the sidecar.
Missing sidecar is fine -- selection still works for everything
that doesn't reference a region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from molbuilder import selection
from molbuilder.parsers import molstruct_json
from molbuilder.selection import (
    All, ByClick, Minus, Or, Rule, SelectionError, evaluate,
    from_json as rule_from_json, to_json as rule_to_json,
)
from molbuilder.structure import Structure

# Reuse the files-blueprint path validator -- same allow-list semantics,
# same error type.  Internal helper but selection has identical needs
# (read a file inside the configured roots, fail loudly on traversal
# attempts), so importing beats forking the rules.
from .files import _PickerError, _resolve_within_roots

bp = Blueprint("selection", __name__)


# --------------------------------------------------------------------- #
#  Structure loader                                                     #
# --------------------------------------------------------------------- #


_SUPPORTED_STRUCTURE_SUFFIXES = (".xyz", ".pdb")


def _parse_structure_text(resolved, text: str) -> Structure:
    """Dispatch by extension to the right Structure parser.

    Centralised so :func:`_load_structure` and the save endpoint
    apply the same XYZ-vs-PDB choice (the save endpoint reads the
    file twice -- once for the hash, once for n_atoms).
    """
    ext = resolved.suffix.lower()
    if ext == ".xyz":
        return Structure.from_xyz(text)
    if ext == ".pdb":
        return Structure.from_pdb(text)
    raise _PickerError(
        400,
        f"unsupported structure extension {ext!r}; "
        f"selection endpoints accept {list(_SUPPORTED_STRUCTURE_SUFFIXES)}",
    )


def _load_structure(structure_path: str) -> Structure:
    """Resolve ``structure_path`` inside the allowed roots, load it,
    and apply any sidecar regions so :class:`ByRegion` works.

    Accepts ``.xyz`` and ``.pdb``; the parser is picked by file
    extension via :func:`_parse_structure_text`.  The sidecar
    (``<stem>.molstruct.json``) sits next to whichever file is
    loaded -- the sidecar key is stem-based so an XYZ and a PDB
    with the same stem CAN share a sidecar, but in practice each
    file gets its own.

    Raises :class:`_PickerError` on path-validation failure, or
    ValueError if the file is unreadable / malformed.
    """
    resolved = _resolve_within_roots(structure_path)
    if not resolved.exists():
        raise _PickerError(404, f"file not found: {resolved}")
    if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
        raise _PickerError(
            400,
            f"unsupported structure extension {resolved.suffix!r}; "
            f"selection endpoints accept "
            f"{list(_SUPPORTED_STRUCTURE_SUFFIXES)}",
        )
    text = resolved.read_text()
    struct = _parse_structure_text(resolved, text)

    # Apply sidecar regions if one is next to the XYZ.  Sidecar
    # failures here are non-fatal: a missing or malformed sidecar
    # just means ByRegion won't resolve, which the selection
    # evaluator will surface as a clean error if the user's rule
    # actually references a region.
    sidecar = molstruct_json.sidecar_path_for(resolved)
    if sidecar.exists():
        try:
            data = molstruct_json.load(sidecar)
            molstruct_json.apply_to_structure(struct, data)
        except molstruct_json.MolstructJsonError:
            # Silently ignore -- a stale sidecar shouldn't block the
            # user from using element-/index-based selection rules
            # against the underlying XYZ.  The /modify save-side
            # validates and rewrites the sidecar.
            pass
    return struct


# --------------------------------------------------------------------- #
#  Toggle bookkeeping (pure function over rule trees)                    #
# --------------------------------------------------------------------- #


def _toggle_click(rule: Rule, index: int, currently_selected: bool) -> Rule:
    """Return a new rule tree with ``index`` toggled in its click
    bookkeeping.

    Contract:

      * ``currently_selected = True``  -> deselect: result must
        exclude ``index`` from its evaluation.
      * ``currently_selected = False`` -> select: result must
        include ``index`` in its evaluation.

    Strategy:

      * Find the top-level ``ByClick`` clause if there is one (either
        the rule itself or one of the immediate ``Or`` operands).
        Add or remove ``index`` from its tuple.
      * If there is no ``ByClick`` clause, wrap the rule:
          * selecting    -> ``Or(rule, ByClick((index,)))``
          * deselecting  -> ``Minus(rule, ByClick((index,)))``
        These are the smallest tree-edits that achieve the desired
        evaluation change, and they keep the rule auditable (the
        UI can show a "clicked: [...]" row separately from the
        algorithmic clauses).
    """
    # Case 1: the entire rule is a ByClick.  Toggle in place.
    if isinstance(rule, ByClick):
        return _byclick_with_toggled(rule, index)

    # Case 2: rule is an Or whose operands include a ByClick.  Edit
    # that operand; preserve the rest.
    if isinstance(rule, Or):
        new_operands = []
        found = False
        for op in rule.operands:
            if isinstance(op, ByClick) and not found:
                new_operands.append(_byclick_with_toggled(op, index))
                found = True
            else:
                new_operands.append(op)
        if found:
            return Or(tuple(new_operands))

    # Case 3: rule has no ByClick clause.  Add one.  Selecting wraps
    # in Or; deselecting wraps in Minus.
    if currently_selected:
        return Minus(rule, ByClick((index,)))
    return Or((rule, ByClick((index,))))


def _byclick_with_toggled(clause: ByClick, index: int) -> ByClick:
    """Return a new ByClick with ``index`` toggled in its indices
    tuple (sorted to keep the rule canonical)."""
    indices = set(clause.indices)
    if index in indices:
        indices.discard(index)
    else:
        indices.add(index)
    return ByClick(tuple(sorted(indices)))


# --------------------------------------------------------------------- #
#  HTTP endpoints                                                       #
# --------------------------------------------------------------------- #


def _bad_request(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


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


@bp.route("/api/selection/atoms", methods=["POST"])
def selection_atoms():
    """Return the atom list for ``structure_path`` with per-atom
    labels (element, optional PDB metadata, region tags, fixed flag).

    The selection panel fetches this once whenever the active
    structure changes, then renders one row per atom in the card's
    scrollable list.  Selection state (which atoms are currently
    selected) is layered on top by the panel client-side, using
    indices from ``/api/selection/eval`` -- the two endpoints are
    separable so a structure load doesn't re-evaluate every rule
    just to populate the list.

    Body:  ``{"structure_path": "/abs/path/to.xyz"}`` (or ``.pdb``)
    Response::

        {
          "n_atoms": 11,
          "atoms": [
            {
              "index":         0,
              "element":       "Au",
              "atom_name":     "AuL",       # optional
              "residue_name":  "LEL",       # optional
              "chain_id":      "L",         # optional
              "regions":       ["L-electrode"],  # may be empty
              "is_frozen":     false
            },
            ...
          ]
        }
    """
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")
        struct = _load_structure(path)

        n = len(struct.elements)
        # Build a reverse index: atom -> [region_label, ...].  Most
        # atoms have zero or one label; a sidecar where the same atom
        # appears in two regions would have been rejected at load
        # time, but we still defend against it here by appending.
        atom_to_regions: Dict[int, list] = {}
        regions = getattr(struct, "regions", {}) or {}
        for label, idxs in regions.items():
            for idx in idxs:
                atom_to_regions.setdefault(idx, []).append(label)

        frozen_set = set(getattr(struct, "frozen_atoms", []) or [])

        # Empty Structure metadata lists round-trip as empty arrays
        # via :class:`Structure`'s dataclass; we still guard with the
        # `or []` so a future "make these Optional" refactor doesn't
        # bite the route.
        atom_names    = struct.atom_names    or []
        residue_names = struct.residue_names or []
        chain_ids     = struct.chain_ids     or []

        atoms = []
        for i in range(n):
            row: Dict[str, Any] = {
                "index":    i,
                "element":  struct.elements[i],
                "regions":  atom_to_regions.get(i, []),
                "is_frozen": i in frozen_set,
            }
            # PDB-derived metadata is optional; omit when empty so the
            # JSON stays compact for plain-XYZ structures (the common
            # /modify case).
            if i < len(atom_names)    and atom_names[i]:
                row["atom_name"]    = atom_names[i]
            if i < len(residue_names) and residue_names[i]:
                row["residue_name"] = residue_names[i]
            if i < len(chain_ids)     and chain_ids[i]:
                row["chain_id"]     = chain_ids[i]
            atoms.append(row)

        return jsonify({"n_atoms": n, "atoms": atoms})
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


@bp.route("/api/selection/save", methods=["POST"])
def selection_save():
    """Persist a materialised selection into the .molstruct.json
    sidecar next to the structure XYZ.  See module docstring for
    the body shape + behaviour contract."""
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")

        target = payload.get("target")
        if not isinstance(target, str) or not target.strip():
            return _bad_request("missing or empty 'target'")
        target = target.strip()

        indices = payload.get("indices")
        if not isinstance(indices, list):
            return _bad_request("'indices' must be a list of ints")
        # ``int(i)`` would silently truncate floats (``int(1.7) -> 1``)
        # and accept numeric strings, both of which let bad client
        # input quietly land in the sidecar.  Require true ints (and
        # explicitly reject bool, since ``isinstance(True, int)`` is
        # True in Python but ``True`` is never a meaningful atom
        # index).
        for i in indices:
            if not isinstance(i, int) or isinstance(i, bool):
                return _bad_request(
                    "'indices' must contain ints only; "
                    f"got {type(i).__name__}={i!r}"
                )
        indices = sorted(set(indices))

        rule_payload = payload.get("rule")  # optional

        resolved = _resolve_within_roots(path)
        if not resolved.exists():
            return _bad_request(f"file not found: {resolved}", 404)
        if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
            return _bad_request(
                f"unsupported structure extension {resolved.suffix!r}; "
                f"selection/save accepts "
                f"{list(_SUPPORTED_STRUCTURE_SUFFIXES)}"
            )

        # Read the structure once to learn n_atoms + structure_hash.
        # Dispatch by extension via the same helper as the load path
        # so XYZ vs PDB stays consistent across endpoints.
        try:
            struct = _parse_structure_text(resolved, resolved.read_text())
        except _PickerError as exc:
            return _bad_request(exc.message, exc.status)
        n_atoms = len(struct.elements)

        for idx in indices:
            if not 0 <= idx < n_atoms:
                return _bad_request(
                    f"indices contain out-of-range value {idx} "
                    f"(structure has {n_atoms} atoms)"
                )

        sidecar_path = molstruct_json.sidecar_path_for(resolved)
        new_hash = molstruct_json.sha256_of_file(resolved)

        # Load existing sidecar if present.  A hash mismatch means
        # the user edited the XYZ since last save; we let the new
        # hash win (the sidecar's regions might be stale but we
        # don't have enough context to recover them, so the
        # honest answer is "the user's most recent action takes
        # precedence").
        existing_regions: Dict[str, list] = {}
        existing_frozen: list = []
        existing_rules: Dict[str, Any] = {}
        if sidecar_path.exists():
            try:
                existing = molstruct_json.load(sidecar_path)
                existing_regions = dict(existing.get("regions") or {})
                existing_frozen  = list(existing.get("frozen_atoms") or [])
                existing_rules   = dict(existing.get("selection_rules") or {})
            except molstruct_json.MolstructJsonError as e:
                # A corrupt sidecar carries user work the server
                # cannot read but ALSO cannot replace safely: writing
                # a fresh sidecar here would overwrite the user's
                # prior regions / frozen_atoms / rules with only the
                # current save's target, silently destroying
                # everything else.  Refuse the save and ask the user
                # to rename / inspect the file -- their action ("save
                # my work") fails loudly instead of erasing data.
                return jsonify({
                    "ok":    False,
                    "error": (
                        f"sidecar at {sidecar_path.name} is unreadable "
                        f"({e}); rename or delete it before saving, "
                        f"or restore it from version control"
                    ),
                }), 409

        # Apply the assignment.  Multi-label model: regions are
        # freeform tags; assigning to one does NOT remove atoms
        # from other regions.  An atom can carry both
        # ``"L-electrode"`` and ``"interface"``.  Engines that
        # need disjoint regions (e.g. transport) enforce that as a
        # separate preflight at engine-load time.
        if target == "frozen_atoms":
            existing_frozen = indices
            if rule_payload is not None:
                existing_rules["frozen_atoms"] = rule_payload
            elif not indices:
                # Clearing frozen_atoms drops the stale rule too --
                # otherwise next-load re-evaluating the rule would
                # silently undo the clear.
                existing_rules.pop("frozen_atoms", None)
        else:
            if indices:
                # Assign = SET the region's membership to the given
                # indices (REPLACE semantics).  No prune from other
                # regions: overlap is allowed.  To remove a single
                # atom from a region, use the per-tag × button in
                # the atom list which POSTs the new list directly.
                existing_regions[target] = indices
                if rule_payload is not None:
                    existing_rules[target] = rule_payload
            else:
                # Assigning empty = remove this region entirely.
                existing_regions.pop(target, None)
                existing_rules.pop(target, None)

        try:
            new_payload = molstruct_json.to_dict(
                n_atoms_total   = n_atoms,
                structure_hash  = new_hash,
                regions         = existing_regions,
                frozen_atoms    = existing_frozen,
                selection_rules = existing_rules,
                created_by      = "molbuilder selection panel",
            )
        except molstruct_json.MolstructJsonError as exc:
            return _bad_request(f"sidecar build failed: {exc}")

        try:
            molstruct_json.save(sidecar_path, new_payload)
        except OSError as exc:
            return _bad_request(f"sidecar write failed: {exc}", 500)

        return jsonify({
            "ok":              True,
            "sidecar_path":    str(sidecar_path),
            "n_atoms_total":   n_atoms,
            "regions":         new_payload["regions"],
            "frozen_atoms":    new_payload["frozen_atoms"],
            "selection_rules": new_payload["selection_rules"],
        })
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


@bp.route("/api/selection/eval", methods=["POST"])
def selection_eval():
    """Evaluate a rule against a structure; return the selected
    indices + count.  See module docstring for body shape."""
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")
        struct = _load_structure(path)
        rule = _load_rule_from_payload(payload)
        try:
            indices = evaluate(rule, struct)
        except SelectionError as exc:
            return _bad_request(f"evaluation failed: {exc}")
        return jsonify({
            "selected_indices": sorted(indices),
            "count":            len(indices),
            "n_atoms_total":    len(struct.elements),
        })
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


@bp.route("/api/selection/toggle", methods=["POST"])
def selection_toggle():
    """Toggle ``index`` in the rule's click bookkeeping; return the
    canonical rule + the new evaluated set.  See module docstring
    for body shape + toggle semantics."""
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")
        struct = _load_structure(path)
        rule = _load_rule_from_payload(payload)
        idx = payload.get("index")
        # ``isinstance(True, int)`` is True in Python; reject bool
        # explicitly so a ``{"index": true}`` payload doesn't toggle
        # atom index 1 (same guard as the save endpoint at line 399).
        if not isinstance(idx, int) or isinstance(idx, bool):
            return _bad_request("'index' must be an integer")
        n = len(struct.elements)
        if not 0 <= idx < n:
            return _bad_request(
                f"index {idx} out of range [0, {n})"
            )

        # Evaluate the EXISTING rule first to decide whether the
        # click should select or deselect.  This is the bookkeeping
        # decision: if the user clicked an already-selected atom,
        # they're asking to deselect.
        try:
            existing = evaluate(rule, struct)
        except SelectionError as exc:
            return _bad_request(f"evaluation failed: {exc}")
        was_selected = idx in existing

        new_rule = _toggle_click(rule, idx, currently_selected=was_selected)
        try:
            new_indices = evaluate(new_rule, struct)
        except SelectionError as exc:
            return _bad_request(f"evaluation failed: {exc}")

        return jsonify({
            "rule":             rule_to_json(new_rule),
            "selected_indices": sorted(new_indices),
            "count":            len(new_indices),
            "n_atoms_total":    n,
        })
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)
