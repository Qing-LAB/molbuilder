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
  * L3 (``lib/workspace/_selection-store-impl.js`` since Phase 9 /
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

See ``docs/protocols/molview-module.md`` for the full module
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
``is_frozen`` flag on each atom-list row reflects the sidecar.
Missing sidecar is fine -- selection still works for everything
that doesn't reference a region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from molbuilder import selection
from molbuilder.sidecars import molstruct as molstruct_json
from molbuilder.selection import (
    Rule, SelectionError, evaluate,
    from_json as rule_from_json,
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
        except molstruct_json.MolstructJsonError:
            # A corrupt sidecar shouldn't block the user from using
            # element-/index-based selection rules against the
            # underlying XYZ.  The save-side validates and rewrites
            # the sidecar.
            pass
        else:
            # 2026-06-12 (sidecar/XYZ desync fix): tolerantly filter
            # out-of-range indices instead of calling
            # ``apply_to_structure`` (which raises on
            # ``sidecar.n_atoms_total != len(struct.elements)``).
            #
            # The desync happens when ``writeLabel`` (Assign click)
            # commits the sidecar with the workspace's IN-MEMORY
            # atom count BEFORE the user clicks Save — and then the
            # Save fails (disk full, permissions, network drop) or
            # the user closes the browser without ever saving.  The
            # sidecar then references atoms that never made it to
            # disk.
            #
            # Old behaviour: ``apply_to_structure`` raised on the
            # mismatch, the ``except`` block above silently swallowed
            # it, and ALL labels disappeared from the UI — including
            # the ones whose indices were still in range for the
            # XYZ on disk.  The user had no signal anything was
            # wrong.
            #
            # New behaviour: drop indices ≥ ``len(struct.elements)``
            # (the orphaned ones that reference atoms that never
            # persisted), keep the rest.  Empty regions after the
            # filter are dropped.  Engines that need a strict-
            # validity check (transport script generator, etc.)
            # call ``apply_to_structure`` directly and keep the
            # fail-fast semantics; only the interactive web load
            # is forgiving.
            struct_n = len(struct.elements)
            filtered_regions = {}
            for name, idxs in (data.get("regions") or {}).items():
                kept = [i for i in (idxs or [])
                        if isinstance(i, int) and 0 <= i < struct_n]
                if kept:
                    filtered_regions[name] = sorted(set(kept))
            filtered_frozen = sorted({
                i for i in (data.get("frozen_atoms") or [])
                if isinstance(i, int) and 0 <= i < struct_n
            })
            struct.regions      = filtered_regions
            struct.frozen_atoms = filtered_frozen
    return struct


def _expose_frozen_as_region(struct) -> None:
    """Expose ``struct.frozen_atoms`` as a synthetic region named
    ``frozen_atoms`` on the IN-MEMORY struct so the ``By label``
    filter in the selection panel can resolve "frozen" to the
    frozen-atom set via the standard ``ByRegion`` rule.

    2026-06-12: split out of ``_load_structure`` because the
    synthetic region would otherwise leak into ``/api/selection/
    atoms``'s response (every frozen atom would carry a
    ``"frozen_atoms"`` label tag in addition to the ``is_frozen``
    flag, double-rendering in the panel).  Only ``/api/selection/
    eval`` (the rule-resolution path) needs the synthetic — call
    this just before ``evaluate``.

    Mutates ``struct.regions`` in place.  ``frozen_atoms`` is
    reserved as a sidecar key (the sidecar keeps it separate from
    ``regions``), so a user-defined region with that name can't
    exist — ``setdefault`` is enough.
    """
    frozen = list(getattr(struct, "frozen_atoms", []) or [])
    if frozen:
        regions = getattr(struct, "regions", None)
        if regions is None:
            struct.regions = {"frozen_atoms": frozen}
        else:
            regions.setdefault("frozen_atoms", frozen)


# --------------------------------------------------------------------- #
#  HTTP endpoints                                                       #
# --------------------------------------------------------------------- #


def _bad_request(msg: str, status: int = 400):
    # Uniform error envelope (projects-sidebar.md § 12 + design.md
    # 2026-05-25 "uniform {ok,error} envelope" decision).  Both ``ok``
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
        # Atoms-list construction lives in ``_shared.atoms_list`` so
        # the disk-read path here AND the in-memory modifier-op path
        # (every /api/modify/* response via structure_to_dict) emit
        # the SAME wire shape.  Pre-2026-06-07 the two paths drifted:
        # this route returned the canonical shape, modify responses
        # returned a structure-minus-atoms shape, and the front-end's
        # selection store could only sync from the disk-read path —
        # so modifier ops left it stale.
        from ._shared import atoms_list as _atoms_list
        rows = _atoms_list(struct)
        # structure-periodicity.md: surface the sidecar's periodicity so a
        # reopened structure restores it (the .json sits next to the .xyz on the
        # server; the viewer never parses -- the host reads it here).  `cell` is
        # kept for the Phase 1 Results-viewer consumer; `periodicity` is the full
        # {cell, axis_kind, vacuum, kgrid} the Modify path-based load reads.
        # Absent / malformed sidecar -> both null.
        cell = None
        periodicity = None
        try:
            _sc = molstruct_json.sidecar_path_for(_resolve_within_roots(path))
            if _sc.exists():
                _sd = molstruct_json.load(_sc)
                cell = _sd.get("cell")
                periodicity = {
                    "cell":      cell,
                    "axis_kind": _sd.get("axis_kind"),
                    "vacuum":    _sd.get("vacuum"),
                    "kgrid":     _sd.get("kgrid"),
                }
        except Exception:
            cell = None
            periodicity = None
        return jsonify({"ok": True, "n_atoms": len(rows), "atoms": rows,
                        "cell": cell, "periodicity": periodicity})
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


def _struct_from_atoms(atoms: list) -> Structure:
    """Build a Structure from the workspace's IN-MEMORY atom list (the store's
    ``atoms``) so a filter evaluates against MEMORY, not the stale saved file
    (molview-migration-plan.md A5b). The filter rules are label/element/index/
    residue only -- no geometry -- so positions are placeholders. The workspace
    module (ws.*) is the single source of truth."""
    if not isinstance(atoms, list):
        raise ValueError("'atoms' must be a list")
    elements: list = []
    regions: Dict[str, list] = {}
    frozen: list = []
    residue_names: list = []
    for i, a in enumerate(atoms):
        if not isinstance(a, dict):
            raise ValueError(f"atoms[{i}] must be an object")
        elements.append(str(a.get("element") or "X"))
        for label in (a.get("labels") or a.get("regions") or []):
            if isinstance(label, str) and label:
                regions.setdefault(label, []).append(i)
        if a.get("isFrozen") or a.get("is_frozen"):
            frozen.append(i)
        residue_names.append(
            str(a.get("residueName") or a.get("residue_name") or "MOL"))
    n = len(elements)
    return Structure(
        elements=elements,
        positions=[[0.0, 0.0, 0.0] for _ in range(n)],
        regions={k: sorted(set(v)) for k, v in regions.items()},
        frozen_atoms=sorted(set(frozen)),
        residue_names=residue_names,
    )


@bp.route("/api/selection/eval", methods=["POST"])
def selection_eval():
    """Evaluate a rule against the workspace and return the selected indices.

    Preferred body (Modify): ``{atoms: [...store atoms...], rule}`` -- evaluate
    against the IN-MEMORY workspace (molview-migration-plan.md A5b), so filters
    reflect unsaved edits.  Legacy/Results body ``{structure_path, rule}`` still
    loads the file on disk (a saved result legitimately lives there)."""
    try:
        payload = _parse_request_payload(request)
        atoms = payload.get("atoms")
        if isinstance(atoms, list):
            try:
                struct = _struct_from_atoms(atoms)
            except ValueError as exc:
                return _bad_request(f"invalid 'atoms': {exc}")
        else:
            path = payload.get("structure_path")
            if not isinstance(path, str) or not path:
                return _bad_request("missing 'atoms' or 'structure_path'")
            struct = _load_structure(path)
        _expose_frozen_as_region(struct)
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


