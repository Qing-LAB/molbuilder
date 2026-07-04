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
    reserved as a sidecar target name (``selection_save`` treats
    it specially), so a user-defined region with that name can't
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

        # 2026-06-09 (modify-then-label fix): the client MAY pass
        # ``n_atoms`` (the workspace's in-memory atom count).  When
        # provided, the server validates indices against THAT instead
        # of the on-disk file.  Before this change, a user who added
        # electrodes (workspace n=11) then labelled an electrode atom
        # (idx=5) got a 400 because the disk file still had n=3 —
        # forcing a Save-first workflow that defeats the point of
        # in-memory edits.  With the client-provided n_atoms the
        # sidecar can be written for the workspace state directly;
        # on Save the XYZ + sidecar persist together.
        client_n_atoms = payload.get("n_atoms")
        if client_n_atoms is not None:
            # Strict type check first — bool slips past ``isinstance(x, int)``
            # in Python because True == 1, so guard explicitly.
            if (not isinstance(client_n_atoms, int)
                    or isinstance(client_n_atoms, bool)):
                return _bad_request(
                    "'n_atoms' must be an integer when provided; "
                    f"got {type(client_n_atoms).__name__}={client_n_atoms!r}"
                )
            # Reject negatives + cap at a generous-but-defensible maximum
            # so a hostile / buggy client can't poison the sidecar with
            # ``n_atoms_total = 999_999_999``, which would (a) bloat the
            # JSON, (b) fail every future apply_to_structure n_atoms check,
            # and (c) brick the file until manual sidecar cleanup.
            # 1,000,000 atoms is well above any realistic chemistry use
            # case (Au-BDT-Au junctions are ~10²-10³ atoms; the largest
            # systems molbuilder is intended for are ~10⁴-10⁵ atoms).
            if not 0 <= client_n_atoms <= 1_000_000:
                return _bad_request(
                    f"'n_atoms' out of range [0, 1_000_000]: "
                    f"got {client_n_atoms}"
                )

        resolved = _resolve_within_roots(path)
        if not resolved.exists():
            return _bad_request(f"file not found: {resolved}", 404)
        if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
            return _bad_request(
                f"unsupported structure extension {resolved.suffix!r}; "
                f"selection/save accepts "
                f"{list(_SUPPORTED_STRUCTURE_SUFFIXES)}"
            )

        if client_n_atoms is not None:
            # Trust the client's atom count — the workspace is the
            # source of truth for the in-memory state.  Hash will be
            # recomputed from disk for the sidecar's stored hash
            # (matches whatever is on disk now; will be re-stored on
            # the next save if the user persists the modified XYZ).
            n_atoms = client_n_atoms
        else:
            # Legacy path: validate against the file on disk.  Kept
            # for callers (CLI, tests) that don't know n_atoms ahead
            # of time.
            try:
                struct = _parse_structure_text(
                    resolved, resolved.read_text())
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

        # Serialise the read-modify-write cycle.  Two concurrent saves
        # on the same sidecar otherwise lose one update: both read the
        # same starting state, both compute disjoint mutations, the
        # second write clobbers the first.  See
        # ``molstruct_json.with_lock`` for the full rationale.  The
        # lock spans EVERY operation that reads or writes
        # ``sidecar_path`` -- ``load``, the in-memory mutation, and
        # ``save`` -- so a concurrent saver waits for our write to
        # commit and then reads the merged state.
        with molstruct_json.with_lock(sidecar_path):
            # Load existing sidecar if present.  A hash mismatch means
            # the user edited the XYZ since last save; we let the new
            # hash win (the sidecar's regions might be stale but we
            # don't have enough context to recover them, so the
            # honest answer is "the user's most recent action takes
            # precedence").
            existing_regions: Dict[str, list] = {}
            existing_frozen: list = []
            existing_rules: Dict[str, Any] = {}
            existing_cell = None
            existing_pbc = None
            if sidecar_path.exists():
                try:
                    existing = molstruct_json.load(sidecar_path)
                    existing_regions = dict(existing.get("regions") or {})
                    existing_frozen  = list(existing.get("frozen_atoms") or [])
                    existing_rules   = dict(existing.get("selection_rules") or {})
                    existing_cell    = existing.get("cell")
                    existing_pbc     = existing.get("pbc")
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
                    # carry the lattice through the load-modify-write so a
                    # region edit doesn't strip an existing cell.
                    cell            = existing_cell,
                    pbc             = existing_pbc,
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


@bp.route("/api/selection/refresh-hash", methods=["POST"])
def selection_refresh_hash():
    """Recompute the sidecar's ``structure_hash`` against the
    CURRENT bytes of the on-disk XYZ + rewrite the sidecar.  Used
    by ``structureSave.save()`` to keep the sidecar's hash in
    sync after a Save persists modifier-op atoms to disk.

    Body (JSON)::

      { "structure_path": "/abs/path/to/file.xyz" }

    Behaviour:

    * No sidecar on disk -> no-op (``ok=true, refreshed=false``)
      so callers can fire-and-forget without branching on a
      sidecar-presence preflight.
    * Sidecar exists -> hash recomputed, regions / frozen_atoms /
      selection_rules / n_atoms_total preserved verbatim, file
      rewritten atomically.

    Returns::

      { "ok": True, "refreshed": bool,
        "structure_hash": "<new sha256>" | null }
    """
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")

        resolved = _resolve_within_roots(path)
        if not resolved.exists():
            return _bad_request(f"file not found: {resolved}", 404)
        if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
            return _bad_request(
                f"unsupported structure extension {resolved.suffix!r}; "
                f"selection/refresh-hash accepts "
                f"{list(_SUPPORTED_STRUCTURE_SUFFIXES)}"
            )

        sidecar_path = molstruct_json.sidecar_path_for(resolved)
        if not sidecar_path.exists():
            # No sidecar to refresh — fire-and-forget no-op.
            return jsonify({
                "ok":             True,
                "refreshed":      False,
                "structure_hash": None,
            })

        new_hash = molstruct_json.sha256_of_file(resolved)

        # 2026-06-12 (n_atoms_total stale fix): re-parse the XYZ to
        # get the current atom count.  Before this, refresh-hash
        # preserved the sidecar's OLD ``n_atoms_total`` unconditionally
        # — but a Save after a modifier op (e.g. add electrode brings
        # 3 atoms -> 11) without a prior Assign would leave the
        # sidecar claiming 3 atoms while the XYZ on disk had 11.
        # Next load's ``apply_to_structure`` then raised
        # MolstructJsonError (mismatch), which ``_load_structure``
        # swallows silently — and the user's labels disappeared with
        # no visible error.  Both ``structure_hash`` and
        # ``n_atoms_total`` track the file bytes; refresh BOTH.
        try:
            new_struct = _parse_structure_text(
                resolved, resolved.read_text())
        except _PickerError as exc:
            return _bad_request(exc.message, exc.status)
        new_n_atoms = len(new_struct.elements)

        # Serialise the read-modify-write cycle against concurrent
        # /api/selection/save writers — same lock as selection_save.
        with molstruct_json.with_lock(sidecar_path):
            try:
                existing = molstruct_json.load(sidecar_path)
            except molstruct_json.MolstructJsonError as exc:
                # A corrupt sidecar shouldn't block Save itself;
                # surface the error but let the caller decide
                # whether to ignore it (the XYZ landed correctly,
                # the sidecar drift is a separate problem).
                return jsonify({
                    "ok":             False,
                    "error":          (
                        f"sidecar at {sidecar_path.name} is unreadable "
                        f"({exc}); the XYZ saved correctly but the "
                        f"sidecar hash could not be refreshed"
                    ),
                    "refreshed":      False,
                    "structure_hash": None,
                }), 409

            # If the new XYZ has fewer atoms than the sidecar's
            # existing region/frozen indices reference, ``to_dict``
            # will reject the rebuild (out-of-range indices vs
            # n_atoms_total).  Drop the now-invalid entries rather
            # than let the whole refresh fail: the user just shrunk
            # the structure and the dropped labels can't reference
            # atoms that no longer exist.  Mirrors the silent
            # filter ``adoptAtoms`` does in the JS store.
            existing_regions = {}
            for name, idxs in (existing.get("regions") or {}).items():
                filt = [i for i in (idxs or []) if 0 <= i < new_n_atoms]
                if filt:
                    existing_regions[name] = filt
            existing_frozen = [
                i for i in (existing.get("frozen_atoms") or [])
                if 0 <= i < new_n_atoms
            ]
            # Drop selection_rules whose target was just dropped —
            # the rule names a region that no longer exists.  Rules
            # for surviving regions + ``frozen_atoms`` pass through.
            existing_rules = {
                t: r for t, r in (existing.get("selection_rules") or {}).items()
                if t in existing_regions or t == "frozen_atoms"
            }

            try:
                new_payload = molstruct_json.to_dict(
                    n_atoms_total   = new_n_atoms,
                    structure_hash  = new_hash,
                    regions         = existing_regions,
                    frozen_atoms    = existing_frozen,
                    selection_rules = existing_rules,
                    # preserve the lattice across the hash-refresh rewrite.
                    cell            = existing.get("cell"),
                    pbc             = existing.get("pbc"),
                    created_by      = "molbuilder save (hash refresh)",
                )
            except molstruct_json.MolstructJsonError as exc:
                return _bad_request(
                    f"sidecar rebuild failed: {exc}", 500)

            try:
                molstruct_json.save(sidecar_path, new_payload)
            except OSError as exc:
                return _bad_request(
                    f"sidecar write failed: {exc}", 500)

        return jsonify({
            "ok":             True,
            "refreshed":      True,
            "structure_hash": new_hash,
        })
    except _PickerError as exc:
        return _bad_request(exc.message, exc.status)


@bp.route("/api/selection/save-sidecar", methods=["POST"])
def selection_save_sidecar():
    """Atomically REPLACE the entire .molstruct.json sidecar with a
    client-provided payload.  Used by Save-as to propagate the
    workspace's labels to a new destination without merging with
    any stale sidecar that happened to exist at the new path.

    Body (JSON)::

      {
        "structure_path": "/abs/path/to/file.xyz",
        "n_atoms":        <int>,
        "regions":        {"L-electrode": [0, 1], ...} or null,
        "frozen_atoms":   [5, 6, ...] or null,
      }

    Behaviour:

    * Existing sidecar at the path -> REPLACED entirely (no merge
      with prior regions / frozen_atoms / selection_rules).
    * Missing sidecar -> created from scratch.
    * ``structure_hash`` is recomputed from the on-disk XYZ bytes
      so the new sidecar matches whatever XYZ the client just
      wrote.
    * ``selection_rules`` is reset to {} — Save-as carries no
      assumption that prior rules still apply against the new XYZ
      bytes.

    Returns::

      { "ok": True,
        "sidecar_path": "...",
        "n_atoms_total": <int>,
        "regions": {...},
        "frozen_atoms": [...] }

    See ``docs/protocols/save-flow.md`` §4.3 for the Save-as label-
    propagation contract.
    """
    try:
        payload = _parse_request_payload(request)
        path = payload.get("structure_path")
        if not isinstance(path, str) or not path:
            return _bad_request("missing 'structure_path'")

        # n_atoms validation mirrors /api/selection/save's gate.
        n_atoms = payload.get("n_atoms")
        if (not isinstance(n_atoms, int)
                or isinstance(n_atoms, bool)):
            return _bad_request(
                "'n_atoms' must be an integer; "
                f"got {type(n_atoms).__name__}={n_atoms!r}"
            )
        if not 0 <= n_atoms <= 1_000_000:
            return _bad_request(
                f"'n_atoms' out of range [0, 1_000_000]: got {n_atoms}"
            )

        regions = payload.get("regions") or {}
        if not isinstance(regions, dict):
            return _bad_request(
                "'regions' must be a dict of {label: [indices]} "
                f"when provided; got {type(regions).__name__}"
            )
        # Validate each region: name is non-empty string, indices
        # is list of in-range ints.
        validated_regions: Dict[str, list] = {}
        for label, idxs in regions.items():
            if not isinstance(label, str) or not label.strip():
                return _bad_request(
                    "every region key must be a non-empty string; "
                    f"got {label!r}"
                )
            if not isinstance(idxs, list):
                return _bad_request(
                    f"regions[{label!r}] must be a list of ints; "
                    f"got {type(idxs).__name__}"
                )
            for idx in idxs:
                if (not isinstance(idx, int)
                        or isinstance(idx, bool)):
                    return _bad_request(
                        f"regions[{label!r}] indices must be ints; "
                        f"got {type(idx).__name__}={idx!r}"
                    )
                if not 0 <= idx < n_atoms:
                    return _bad_request(
                        f"regions[{label!r}] index {idx} out of "
                        f"range [0, {n_atoms})"
                    )
            validated_regions[label.strip()] = sorted(set(idxs))

        frozen = payload.get("frozen_atoms") or []
        if not isinstance(frozen, list):
            return _bad_request(
                "'frozen_atoms' must be a list of ints when provided; "
                f"got {type(frozen).__name__}"
            )
        validated_frozen: list = []
        for idx in frozen:
            if not isinstance(idx, int) or isinstance(idx, bool):
                return _bad_request(
                    "'frozen_atoms' must contain ints only; "
                    f"got {type(idx).__name__}={idx!r}"
                )
            if not 0 <= idx < n_atoms:
                return _bad_request(
                    f"'frozen_atoms' index {idx} out of range "
                    f"[0, {n_atoms})"
                )
            validated_frozen.append(idx)
        validated_frozen = sorted(set(validated_frozen))

        # Periodicity (structure-periodicity.md) rides with the whole-store save
        # (workspace-contract.md §4.0).  Optional dict {cell, axis_kind, vacuum,
        # kgrid}; to_dict validates the values.  Absent -> written as null.
        per = payload.get("periodicity") or {}
        if not isinstance(per, dict):
            return _bad_request(
                "'periodicity' must be a dict when provided; "
                f"got {type(per).__name__}")

        resolved = _resolve_within_roots(path)
        if not resolved.exists():
            return _bad_request(f"file not found: {resolved}", 404)
        if resolved.suffix.lower() not in _SUPPORTED_STRUCTURE_SUFFIXES:
            return _bad_request(
                f"unsupported structure extension {resolved.suffix!r}; "
                f"selection/save-sidecar accepts "
                f"{list(_SUPPORTED_STRUCTURE_SUFFIXES)}"
            )

        sidecar_path = molstruct_json.sidecar_path_for(resolved)
        new_hash = molstruct_json.sha256_of_file(resolved)

        # Serialise under the same with_lock that /api/selection/save
        # uses so concurrent writers don't race.
        with molstruct_json.with_lock(sidecar_path):
            try:
                new_payload = molstruct_json.to_dict(
                    n_atoms_total   = n_atoms,
                    structure_hash  = new_hash,
                    regions         = validated_regions,
                    frozen_atoms    = validated_frozen,
                    cell            = per.get("cell"),
                    axis_kind       = per.get("axis_kind"),
                    vacuum          = per.get("vacuum"),
                    kgrid           = per.get("kgrid"),
                    # Save-as starts fresh — no rule carries over,
                    # since the user's intent is "snapshot the
                    # current workspace labels into a new sidecar".
                    selection_rules = {},
                    created_by      = "molbuilder save panel (save-as)",
                )
            except molstruct_json.MolstructJsonError as exc:
                return _bad_request(f"sidecar build failed: {exc}")

            try:
                molstruct_json.save(sidecar_path, new_payload)
            except OSError as exc:
                return _bad_request(
                    f"sidecar write failed: {exc}", 500)

        return jsonify({
            "ok":              True,
            "sidecar_path":    str(sidecar_path),
            "n_atoms_total":   n_atoms,
            "regions":         new_payload["regions"],
            "frozen_atoms":    new_payload["frozen_atoms"],
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


