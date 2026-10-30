"""Build blueprint -- structure-construction + emitter routes.

Routes (registered with no url_prefix; each carries its own full path):

    POST /api/build/molecule        build a Structure from sequence/SMILES/name
    POST /api/build/load            load an existing XYZ / PDB into a Structure
    POST /api/build/preflight       fast validate-only path (no rendering)
    GET  /api/build/schema/<engine> form-rendering schema for the SIESTA /
                                    PySCF describing tabs (engine ∈ {siesta,
                                    pyscf}; ?calculation=vibration narrows to
                                    the kind's items)
    POST /api/structure/{analyze,periodicity}
    POST /api/task-setup/{handover,prep,save,bench-grid,prep-plan}
    GET  /api/task-setup/{folder,machines,sweepable,columns,presets}

These endpoints share a single Flask app instance with the watch
blueprint at ``molbuilder/web/blueprints/watch.py``.  Two top-level
routes stay on the app itself rather than on this blueprint:

    GET  /                     redirect to the first tab
    GET  /api/health           liveness

JSON shape:

  /api/build/molecule -- body: {"kind": "peptide|dna|rna|smiles|name",
                                "input": "<sequence-or-smiles-or-name>",
                                ...optional kind-specific knobs}
                         returns: {"ok": True, "xyz": "...", "pdb": "...",
                                   "n_atoms": N, "summary": "...",
                                   "title": "...", "elements": [...]}

      DNA / RNA tri-state add_hydrogens semantics:
        "auto"  (default) -- backend-aware: 3DNA gets H, AmberTools/3DNA-fiber
                             keeps the backend's existing H placement.
        "on"              -- always invoke chemistry.add_hydrogens.
        "off"              -- skip H addition entirely.
        true              -- back-compat alias for "auto" (NOT "on").
        false             -- back-compat alias for "off".

  /api/build/load     -- body: multipart with "file" field
                         OR JSON {"text": "...", "format": "auto"|"xyz"|"pdb",
                                  "filename": "<optional>"}
                         returns: same shape as /api/build/molecule
                                  plus "source_format": "xyz"|"pdb"

  The preflight reads the structure
  through ``_shared.struct_from_body`` -- the atoms as NUMBERS with their
  facts beside them, which is what the browser holds and what every other
  structure door already takes.
"""

from __future__ import annotations

import dataclasses

import json
import pathlib
from datetime import datetime
from typing import Any, Dict, Optional

from ...chemistry import BackendUnavailable as _BackendUnavailable
from ...issues import Issue
from flask import Blueprint, jsonify, request

from ._shared import (
    config_from_params as _config_from_params,
    catalogue_to_form_schema as _catalogue_to_form_schema,
    engine_key_for as _engine_key_for,
    issues_to_json as _issues_to_json,
    ok_structure_response,
    struct_from_body as _struct_from_body,
)

from molbuilder.nucleic import build_dna, build_rna
from molbuilder.peptide import build_peptide
from molbuilder.pubchem import build_from_name
from molbuilder.smiles import build_from_smiles
from molbuilder.engine_table import (EngineTableError, config_class,
                                     engine as _engine)
from molbuilder.structure import Structure
from molbuilder.validation import validate
from .files import _resolve_within_roots, _PickerError


bp = Blueprint("build", __name__)


# Map kind -> builder.  Keeps the dispatch tight; per-kind URL paths
# would be one route each (an internal refactor option for later --
# the dispatch table here makes that mechanical when wanted).
_BUILDERS = {
    "peptide": build_peptide,
    "dna":     build_dna,
    "rna":     build_rna,
    "smiles":  build_from_smiles,
    "name":    build_from_name,
}


def _sniff_structure_format(text: str) -> str:
    """Return ``"xyz"`` or ``"pdb"`` for raw structure text.

    Relies on the format's own first-line rule instead of a byte-window
    scan: a real PDB file's HEADER / TITLE / REMARK lines can push the
    first ATOM record far into the file.

    Rule: XYZ's first non-blank line is an atom count (positive int).
    Anything else (PDB headers, plain text, empty) is treated as PDB.
    Caller still wraps ``Structure.from_pdb`` in try/except, so a
    misclassified blob fails with a clear "could not parse" 400.
    """
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return "xyz" if int(line) > 0 else "pdb"
        except ValueError:
            return "pdb"
    return "pdb"


@bp.route("/api/structure/analyze", methods=["POST"])
def api_structure_analyze():
    """The electronic state of a structure, for exactly what a form says.

    Body (JSON)::

      "structure": {elements, positions, metadata[, info]}
      "kind":  "optimization" | "vibration" | "transport"   (stated by every tab)
      "forms": {"<engine>": {net_charge, spin_treatment,
                             unpaired_electrons, method}}      (optional)

    The structure is THE ENVELOPE the page would hand over -- the one its
    viewer holds, the same the preflight and the hand-over send -- so the
    answer is about the structure the deck will be written for, its cell,
    its axis kinds and the record of the run it came from included.

    Returns the structure's chemistry FACTS (``analyze_structure``: the atom
    count, the metals and their usual spins) and, per engine, the
    ``ElectronicState`` the class resolves for that form's own items -- each
    value with where it came from (`science/chemistry-correctness.md`
    § 2a.5).  A blank item is the instruction "work it out", so the card and
    the chip show the answer the deck will carry, not a suggestion to copy:
    this route and the deck writers and the checks read the same class, and
    cannot disagree.  With no ``forms``, each engine that runs the kind is
    answered with every item blank.
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body.get("structure"), dict):
        return jsonify({"ok": False,
                        "error": "no structure given: send the one the page "
                                 "holds, in the envelope -- {\"structure\": "
                                 "{elements, positions, metadata}}"}), 400
    from ._shared import struct_from_body
    try:
        struct = struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not restore structure: {exc}"}), 400
    return _analyze_response(struct, body)


def _analyze_response(struct, body):
    """The ONE analyze answer for a structure and the forms' items."""
    from dataclasses import asdict
    from molbuilder.chemistry import analyze_structure
    from molbuilder.electronic_state import electronic_state
    from molbuilder import kind_table

    kind = body.get("kind") or ""
    if not kind:
        return _unstated(kind=kind)
    if kind not in kind_table.kinds():
        return jsonify({"ok": False,
                        "error": f"kind must be one of "
                                 f"{', '.join(kind_table.kinds())}, "
                                 f"not {kind!r}"}), 400
    forms = body.get("forms")
    if forms is not None and not isinstance(forms, dict):
        return jsonify({"ok": False,
                        "error": "`forms` maps an engine to its form's "
                                 "items"}), 400
    runs = kind_table.kind(kind).engines
    if not forms:
        forms = {engine: {} for engine in runs}
    # The facts need every label to name an element (the answer is an
    # electron count): an unknown symbol is a 400 with the parser's own
    # message, never a 500.
    try:
        facts = analyze_structure(struct)
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc).strip("'")}), 400
    states = {}
    for engine, items in forms.items():
        # WHO RUNS WHICH KIND is the kind table's (its parts, § 2a.3):
        # a form for an engine that does not run this kind has no state to
        # ask for.
        if engine not in runs:
            return jsonify({"ok": False,
                            "error": f"no {engine!r} form for a {kind} "
                                     f"calculation: {kind} runs on "
                                     f"{', '.join(runs)}"}), 400
        # The form's items become the engine's config through the ONE
        # params door -- the same coercion the preflight and the hand-over
        # use -- so a blank means exactly what it means there.
        try:
            cfg = _config_from_params(config_class(engine), items or {}, kind)
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        states[engine] = electronic_state(struct, cfg, kind=kind).as_dict()
    return jsonify({
        "ok":          True,
        "kind":        kind,
        "n_atoms":     facts.n_atoms,
        "metals":      facts.metals,
        "metal_hints": [asdict(h) for h in facts.metal_hints],
        "state":       states,
        # EACH ENGINE'S NAME, from its one home (the engine table): the
        # card heads each state with it.
        "engine_words": {e: _engine(e).words for e in states},
    })


@bp.route("/api/build/molecule", methods=["POST"])
def api_build_molecule():
    body = request.get_json(silent=True) or {}
    kind = (body.get("kind") or "").strip().lower()
    text = (body.get("input") or "").strip()
    if kind not in _BUILDERS:
        return jsonify({"ok": False,
                        "error": f"Unknown kind {kind!r}; "
                                 f"valid: {sorted(_BUILDERS)}"}), 400
    if not text:
        return jsonify({"ok": False, "error": "empty input"}), 400
    backend_used: str | None = None
    h_mode_used: str | None = None
    # THE BACKEND ASKED FOR, when a kind lets a person choose one.
    # Bound here, before any builder runs, because the missing-backend
    # answer below names it -- and a builder that is not DNA/RNA can
    # reach that answer too (a peptide whose hydrogens nothing here can
    # add).
    requested: str | None = None
    build_warnings: list[str] = []
    try:
        # DNA / RNA accept extra knobs (backend / form / terminal).
        if kind in ("dna", "rna"):
            requested = body.get("backend", "auto")
            # add_hydrogens is tri-state: auto / on / off.  The web
            # form sends a string ("auto" by default).  We accept
            # bool too for back-compat with older client code.
            h_mode_raw = body.get("add_hydrogens", "auto")
            if isinstance(h_mode_raw, bool):
                h_mode_used = "auto" if h_mode_raw else "off"
            else:
                h_mode_used = str(h_mode_raw).lower()
                if h_mode_used not in ("auto", "on", "off"):
                    return jsonify({
                        "ok": False,
                        "error": (
                            f"add_hydrogens must be 'auto'/'on'/'off' "
                            f"(or bool); got {h_mode_raw!r}"
                        ),
                    }), 400
            kwargs = {
                "backend":  requested,
                "form":     body.get("form",     "B" if kind == "dna" else "A"),
                "terminal": body.get("terminal", "OH"),
                "add_hydrogens": h_mode_used,
                "protonate_phosphates":
                    bool(body.get("protonate_phosphates", True)),
            }
            # relax_clashes (DNA explicit-duplex): opt-in force-field relief of a
            # mismatched pair's steric overlap.  build_dna ignores it for ss / RNA.
            if kind == "dna":
                kwargs["relax_clashes"] = bool(body.get("relax_clashes", False))
            # Resolve "auto" before the build so the UI can display
            # which backend actually ran -- this matches dispatch()'s
            # selection logic exactly (see auto_backend_name docstring).
            if requested == "auto":
                from molbuilder.builders.backends import auto_backend_name
                backend_used = auto_backend_name()
            else:
                backend_used = requested
            # Capture builder RuntimeWarnings (e.g. a mismatched-duplex steric
            # CLASH, or the amber "extended polymer" note) so the UI can surface
            # them -- otherwise they'd only reach the server log.
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as _caught:
                _warnings.simplefilter("always")
                struct = _BUILDERS[kind](text, **kwargs)
            build_warnings = [str(w.message) for w in _caught]
        elif kind in ("smiles", "name"):
            # RDKit-first, OpenBabel-fallback (Name lookup resolves to SMILES then
            # builds, so it rides the same chain): surface WHICH engine produced
            # the geometry so the user knows when they're on the lower-fidelity path.
            struct, backend_used = _BUILDERS[kind](text, return_backend=True)
        else:
            struct = _BUILDERS[kind](text)
    except _BackendUnavailable as exc:
        # NOT A SERVER FAULT.  `web-api.md` § 1 defines 5xx as *"an I/O
        # error, an engine that fell over, a bug"* -- a backend the person
        # chose not to install is none of those, and answering 500 tells
        # them molbuilder crashed when it did not.  It is the contract's
        # ADVISORY case, word for word: *"the request was well-formed and
        # the validator refused it"* -> 200 with `ok: false`.
        #
        # `reason` and `backend` are machine-readable so the page can act
        # on it -- the Modify tab offers all four backends whether or not
        # they are installed, and this is what would let it stop.  The
        # message itself is unchanged: `design.md` requires it to name the
        # preconditions checked, the download URL, the licence terms and
        # the fallbacks, and it does.
        return jsonify({"ok": False, "reason": "backend_unavailable",
                        # WHAT IS MISSING, when one engine is the answer -- a
                        # backend, or the hydrogen engines -- else the
                        # request (`chemistry.BackendUnavailable.missing`).
                        "backend": exc.missing or requested or "auto",
                        "error": str(exc)}), 200
    except ImportError as exc:
        return jsonify({"ok": False,
                        "error": f"missing dependency: {exc}"}), 500
    except Exception as exc:
        # web-api.md § 1, *Status codes* (server fault -> 5xx): an
        # unhandled exception from the
        # builder dispatch is server fault, not protocol error.
        # The user's input passed shape validation (kind + input)
        # before reaching here; whatever went wrong is on us.
        return jsonify({"ok": False, "error": str(exc)}), 500

    # THE GENERATOR SIGNS ITS WORK, AND THE SIGNATURE IS SELECTABLE.
    #
    # A structure built from a SMILES string, a PubChem name or a sequence
    # comes back carrying one region over every atom, named for the text that
    # produced it.  So "select the thing I just built" is one click on a label
    # the user already recognises, and the provenance rides in the sidecar with
    # everything else instead of living only in a status line that the next
    # load erases.
    #
    # THE TRAILING `#` MARKS IT MOLBUILDER'S.  Region labels are ONE
    # namespace, shared by the user's own labels and by the ones molbuilder
    # reads -- `L-electrode` and `R-electrode` among them, the leads
    # (`transport.sort.ELECTRODE_LABELS`).  The name generator takes whatever
    # a person types, and with the `#` after it the label is never a lead
    # name, whatever was typed; and a machine-written label stays told apart
    # from a hand-written one, which is the thing a shared namespace
    # otherwise loses (`model/structure-annotations.md` § 5.1).  Measured to
    # survive both persistence paths -- the
    # `.molstruct.json` pair and the deck's ATOM-METADATA block, whose lines
    # are already `#`-prefixed comments and whose readers strip a prefix rather
    # than splitting on the character.
    if text:
        struct.regions = dict(struct.regions or {},
                              **{f"{text}#": list(range(struct.n_atoms))})

    # Route through the canonical ``ok_structure_response`` helper.
    # Endpoint-specific keys (pdb, summary, backend_used,
    # add_hydrogens_mode) land BOTH at the top level (for every JS
    # consumer that reads them off the response root) AND in the
    # canonical ``extra`` sub-dict.  Issues + canonical atoms
    # come from the helper — one validate_geometry pass.
    return ok_structure_response(struct, extra={
        # build/molecule's legacy contract: title defaults to the
        # build kind ("smiles" / "dna" / …) when the Structure
        # itself carries no title (most builders don't set one).
        # Override via extra rather than mutating struct.title so
        # downstream code that reuses the Structure sees the
        # canonical (empty) title.
        "title":             struct.title or kind,
        "pdb":               struct.to_pdb(),
        "summary":           struct.summary(),
        "backend_used":      backend_used,
        # Tri-state H-add decision actually used (echoes the
        # request, or "auto" when not explicitly requested).  None
        # for non-nucleic builds (peptide/SMILES/name) where the
        # kwarg doesn't apply.
        "add_hydrogens_mode": h_mode_used,
        # Builder warnings (e.g. a mismatched-duplex steric clash) for the UI to
        # surface; empty list when the build was clean.
        "build_warnings":     build_warnings,
    })


@bp.route("/api/structure/periodicity", methods=["POST"])
def api_periodicity():
    """The unified periodicity door (structure-periodicity.md § 6.2): ONE
    entry point for the Cell-page edits — ``vacuum`` / ``axis_kind`` /
    ``cell`` / ``box_corner`` and ``block`` (``periodicity_gate.OPS``) —
    through the frame-contract gate.

    Body: ``{"structure": <envelope>, "op": <one of OPS>, "payload": ...}`` --
    THE ENVELOPE every other structure door takes (web-api.md § 1), so a caller
    holding coordinates as numbers never writes a coordinate document to ask a
    question about them (molview.md § 11.7).  ``payload`` is required (may be
    ``null``) for ``cell`` / ``box_corner``, where ``null`` means "clear it" --
    omitting the key is an error rather than a silent clear.

    Response: ``{ok, periodicity, notices}`` -- ``periodicity`` is the cell block
    exactly as ``/api/build/load`` sends it (``cell`` / ``engine_offset`` /
    ``axis_kind`` / ``vacuum`` plus the resolved views beside them), so the
    client adopts it verbatim through the same path a load takes and there is one
    shape for the block rather than two.  ``notices`` is a list of
    ``{severity, message, where, about}`` for the Cell page (molview.md
    § 6.8).  There is no
    "the gate changed this" marker and there should not be: clause 1 forbids the
    gate writing a resolved value back, so nothing is ever changed to mark.

    ``op`` is one of ``vacuum`` / ``axis_kind`` / ``cell`` / ``box_corner`` /
    ``block``.  The first four set one field -- ``box_corner`` the origin the
    person assigns, stored as the offset (§ 6.0); ``block`` takes the WHOLE
    cell -- ``{cell, box_corner, axis_kind, vacuum}`` -- and checks it once, which is
    the only way to describe a change to two of them atomically (§ 6.2).

    400 on: an unknown op, a missing payload for ``cell`` / ``box_corner``, a
    malformed envelope, and every contract violation the gate raises (a
    left-handed cell, a cell no origin could make fit, a degenerate derived
    box, a periodic axis with no explicit cell)."""
    from molbuilder.periodicity_gate import apply_edit, OPS, validate_periodicity
    body = request.get_json(silent=True) or {}
    op = body.get("op")
    if op not in OPS:
        return jsonify({"ok": False,
                        "error": f"'op' must be one of {list(OPS)}"}), 400
    if not isinstance(body.get("structure"), dict):
        return jsonify({"ok": False,
                        "error": "missing or invalid 'structure' envelope "
                                 "(need {elements, positions, metadata})"}), 400
    if op in ("cell", "box_corner") and "payload" not in body:
        # For these ops a null payload is a DESTRUCTIVE action (clear /
        # reset) -- a dropped key must not be indistinguishable from an
        # explicit clear.
        return jsonify({"ok": False,
                        "error": f"op '{op}' requires an explicit "
                                 f"'payload' (use null to clear/reset)"}), 400
    try:
        struct = _struct_from_body(body)
        # NOT GATED ON THE INCOMING STATE.  This is the page a bad box is
        # repaired on -- the load door admits one for exactly that reason --
        # so refusing the edit because the box is still bad would leave it
        # unfixable.  The RESULT is gated below, which is what
        # molview.md § 6.8 actually asks for.
        new_struct, receipts = apply_edit(struct, op, body.get("payload"))
        # The CONDITIONS are re-derived on the RESULT, so "the box does not
        # contain the structure" is answered about the box that now exists.
        new_struct, conditions = validate_periodicity(new_struct)
        # Receipts first (what the edit did), then conditions (what is now
        # true), then WHAT THE BOUNDARY DOES TO THE CRYSTAL -- the verdict the
        # slab build gives, on the box just set (`science/junction-cell.md`
        # § 6.1: the loop closes where `c` is set).
        from .modify import box_seam_notices
        notices = (list(receipts) + list(conditions)
                   + box_seam_notices(new_struct))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 -- malformed envelope -> 400
        return jsonify({"ok": False, "error": str(exc)}), 400
    # The block the structure itself assembles -- raw values and the resolved
    # views together -- so a field added to it reaches this door with no edit
    # here, and the client cannot be handed a block missing the resolved half.
    return jsonify({
        "ok": True,
        "periodicity": new_struct.to_wire()["periodicity"],
        # THE RECORD RIDES WITH THE BLOCK.  `apply_edit` marks the recorded
        # contract outdated (a box edit is an edit), and this is the only
        # answer the caller adopts -- so without `info` here the mark would
        # be decided in Python and then set a second time in the browser.
        # One decider; `model-jobs.js` takes it from here.
        "info": new_struct.info,
        "notices": notices,
    })


@bp.route("/api/structure/save", methods=["POST"])
def api_structure_save():
    """FILE-ONLY save through the ONE authority (``model/structure.md`` § 2.4), the
    symmetric inverse of the file-only load below.  The browser hands the SETTLED model
    as the structure envelope (web-api.md § 1); the SERVER writes the ``<stem>.xyz`` + ``<stem>.molstruct.json``
    pair via ``StructureCodec.write``.  Python owns the pairing, the write order/atomicity,
    AND the sidecar schema -- ``write`` stamps ``schema_version`` + the ``geometry_hash`` of what it writes
    (``molstruct.to_dict``), so the pair the load door reads back is VALID.  The browser
    never authors the sidecar envelope (the drift that made a browser-written sidecar
    unloadable).  Body: ``{"path": "<project-relative .xyz>", "structure": {...},
    "overwrite": bool}``.  Returns ``{ok:true, path, notices}`` | ``{ok:false, needsOverwrite:true}``
    (409, drives the tab's overwrite dialog) | ``{ok:false, error}`` -- and the
    periodicity gate runs here exactly as on export: the same
    refusal is the same 400, the same verdicts ride ``notices``."""
    from molbuilder.web.blueprints.files import _resolve_within_roots, _PickerError
    from molbuilder.workingcopy_structure import StructureCodec
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    overwrite = bool(body.get("overwrite"))
    if not isinstance(path, str) or not path:
        return jsonify({"ok": False, "error": "missing or invalid 'path'"}), 400
    # THE STRUCTURE, not a document the browser wrote: the one-path rule
    # (molview.md § 11.7) cannot be true while a door accepts bytes.  The structure arrives as the envelope every other door
    # takes (web-api.md § 1), and the SERVER writes both files from it.
    try:
        struct = _struct_from_body(body)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # WRITTEN AS THE EXPORT WRITES IT (`_shared.for_writing`): in the user order,
    # its box judged -- a save and a download cannot produce different bytes,
    # nor disagree about a refusal.  Runs BEFORE the overwrite gate, so
    # nobody is asked to confirm an overwrite for a save that will be
    # refused; the verdicts ride the response as `notices`.
    from ._shared import for_writing
    struct, notices = for_writing(struct)
    try:
        resolved = _resolve_within_roots(path)   # save-as target need not exist yet
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    # Overwrite gate: the GEOMETRY file's existence drives the tab's overwrite dialog
    # (mirrors the /api/files/write 409 contract) -- refuse unless the caller confirmed.
    if resolved.exists() and not overwrite:
        return jsonify({"ok": False, "needsOverwrite": True,
                        "error": f"file already exists: {path}"}), 409
    try:
        StructureCodec().write(struct, resolved)
    except ValueError as exc:          # a frame set to a .pdb
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 -- disk / permission -> 500
        return jsonify({"ok": False, "error": f"could not save {path}: {exc}"}), 500
    return jsonify({"ok": True, "path": path, "notices": notices})


@bp.route("/api/structure/export", methods=["POST"])
def api_structure_export():
    """The pair a save would write -- NAMED, and returned instead of written.

    Same generator, different destination: :func:`api_structure_save` puts
    ``StructureCodec.pair`` on disk and this hands it back through
    ``StructureCodec.files``, so a structure saved into a project and the same
    structure downloaded are byte-identical BY CONSTRUCTION rather than by two
    code paths agreeing.

    That division exists because the browser cannot produce the pair itself. The
    sidecar's envelope -- ``schema_version``, and the ``geometry_hash`` pinning
    it to its geometry -- is the codec's.

    Body: the ENVELOPE (web-api.md § 1) plus an optional stem --
    ``{"structure": {...}, "name": "<stem>"}``; a frame set's frames ride
    inside the envelope (``model/structure.md`` § 2.2e).

    WHO NAMES WHAT.  ``name`` is a STEM and nothing else (``wire_frame40-120``,
    no extension), because only the caller knows what an export IS: which
    structure, which frames, chosen at which moment.  The SUFFIX is the server's,
    because it follows from the format, which ``pair()`` already decided --
    plain XYZ, a block per frame -- and a caller that appends its own is
    answering a question that has an answer.  A missing / empty / path-shaped ``name`` falls back to
    ``structure``; only the last path component is ever used, and nothing here
    touches the filesystem.

    Returns ``{ok, files: [{name, text}], frames, notices}`` -- ``frames`` the
    count the files hold -- each entry is a
    file as it would exist on disk, under the name it would exist as.  One entry
    means the structure carries no metadata worth keeping, which is exactly when
    a save writes no ``.json`` either (``no .json == empty metadata``)."""
    from molbuilder.workingcopy_structure import StructureCodec
    body = request.get_json(silent=True) or {}
    if not isinstance(body.get("structure"), dict):
        return jsonify({"ok": False,
                        "error": "missing or invalid 'structure' envelope "
                                 "(need {geometry: {elements, positions}, metadata})"}), 400
    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # WRITTEN AS THE SAVE WRITES IT (`_shared.for_writing`): in the user order,
    # its box judged -- a cell it refuses leaves as a 400 carrying the gate's
    # sentence, not as a 500.
    from ._shared import for_writing
    struct, notices = for_writing(struct)
    # THE STEM, reduced to its last component.  This never reaches the
    # filesystem -- ``files()`` builds names in memory -- but it does reach the
    # browser as a download name, so a path-shaped one is flattened rather than
    # passed on.
    raw_name = str(body.get("name") or "").replace("\\", "/")
    stem = raw_name.rsplit("/", 1)[-1].strip()
    if not stem or stem in (".", ".."):
        stem = "structure"
    # PLAIN XYZ, whatever the stem says: `name` is a stem, and the suffix
    # follows from the format, which is this door's (web-api.md, *the export
    # door answers with named files*).
    made = StructureCodec().files(struct, stem, fmt="xyz")
    return jsonify({"ok": True,
                    "files": [{"name": path.name,
                               "text": blob.decode("utf-8")}
                              for path, blob in made],
                    "frames": struct.n_frames,
                    "notices": notices})


def _chosen_frame(whole: Structure, body: Dict[str, Any]) -> Structure:
    """The frame a load answers with -- ``StructureCodec.frame_choice``, the
    one rule (``model/structure.md`` § 2.3): frame 0 unless the body names
    ``frame`` (0-based), or the whole set with ``frames: true``.  Every
    answer also says how many frames the file holds (``n_frames``), so a tab
    can ask which (``tabs.md`` § 2)."""
    from molbuilder.workingcopy_structure import StructureCodec
    frame = body.get("frame")
    frames = body.get("frames", False)
    if not isinstance(frames, bool):
        raise ValueError("'frames' is true for the whole set, or absent")
    return StructureCodec.frame_choice(whole, frame=frame, frames=frames)


@bp.route("/api/build/load", methods=["POST"])
def api_build_load():
    """Accept one of three JSON bodies:
      * {"path": "<project-relative structure file>"} -- the FILE-ONLY
        load: the SERVER reads the .xyz(/.pdb) + its paired
        .molstruct.json through StructureCodec.read (the ONE authority
        owns the file access AND the pairing).  This is how a project
        file is opened -- no raw text, no browser-side sidecar path.  A
        lone file is atoms and coordinates, said in a
        ``structure.lone_file`` notice quoting its comment line.
      * {"structure": <envelope>} -- a tab PUTTING BACK the structure it
        was showing; answered whole, no frame chosen.
      * {"text": "...", "filename": "<optional>"} -- raw-geometry IMPORT
        (a paste / upload with no persisted file yet); atoms only, the
        format read off the filename's suffix or sniffed.  An XYZ text is a
        lone file, and says so in the same ``structure.lone_file`` notice.
    A path or a text load answers frame 0, or ``frame`` (0-based), or the
    whole set with ``frames: true``; every answer says ``n_frames``, the
    count the file holds.  Returns the same JSON shape as
    /api/build/molecule so the front end can treat the result identically.

    A path load answers the file AS IT IS -- a pair in an engine's order
    with its order record -- unless ``restore: true``, which an editor's open
    asks, and the transport report, which names its selections in the
    user order: then in the user order, the codec's ``in_user_order`` read, said in a
    ``structure.restored`` notice naming the order in its words
    (``model/overview.md`` § 2.2a, I8).
    """
    # FILE-ONLY load through the ONE authority (``model/structure.md`` § 2.4): a project
    # ``path`` means the SERVER reads the .xyz + paired .molstruct.json via
    # StructureCodec.read -- Python owns the file access + the .xyz<->.molstruct
    # pairing, so there is NO raw-text hand-crafting and NO browser-side sidecar
    # derivation.  ``to_wire`` (via ok_structure_response) emits the enriched atoms
    # + periodicity + annotations in one response.
    _pbody = request.get_json(silent=True) or {}
    _path = _pbody.get("path")
    if _path:
        from molbuilder.web.blueprints.files import (
            _resolve_within_roots, _PickerError)
        from molbuilder.workingcopy_structure import StructureCodec
        try:
            _resolved = _resolve_within_roots(_path)
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        if not _resolved.exists():
            return jsonify({"ok": False, "error": f"no such file: {_path}"}), 404
        retired: Dict[str, Any] = {}
        read_said: list = []
        # A PAIR WHOSE GEOMETRY CHANGED OPENS HERE, said: this is the door the
        # Molbuilder tab fixes it through (`structure-molstruct.md` § 3) --
        # and so does a run whose order nothing states, for a viewer (I9).
        # AN EDITOR HOLDS ONLY THE USER ORDER (`model/overview.md` § 2.2a, I8):
        # its open asks for the file restored, said, and a run whose order
        # nothing states is refused it -- as the transport report's drawing
        # does, its selections named in the user order; any other viewer is
        # answered the file as it is, its order record attached.
        from molbuilder.workingcopy_structure import (GEOMETRY_CHANGED,
                                                      ORDER_UNSTATED)
        restore = _pbody.get("restore") is True
        try:
            whole = StructureCodec().read(
                _resolved, frames=True, retired_out=retired,
                said_out=read_said,
                opened=((GEOMETRY_CHANGED,) if restore
                        else (GEOMETRY_CHANGED, ORDER_UNSTATED)),
                in_user_order=restore)
        except Exception as exc:  # noqa: BLE001 -- parse/sidecar error -> 400
            return jsonify(
                {"ok": False, "error": f"could not load {_path}: {exc}"}), 400
        try:
            struct = _chosen_frame(whole, _pbody)
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        # The conditions for THIS structure are produced on the way out by
        # `ok_structure_response`, which validates every structure it sends.
        # What only the read knows is a retired corner it did not apply: said
        # here, naming it, because a person who typed it assigns it again on
        # the Cell page (plan § 5q D14).
        from molbuilder.issues import Issue
        from molbuilder.periodicity_gate import notices_for_report
        # WHAT THE READ SAID, each notice at its level (`workingcopy_structure
        # .READ_NOTICES`): a lone file, a changed geometry, a run whose order
        # nothing states, a structure put back in the user order.
        said = notices_for_report(read_said)
        if retired.get("cell_origin") is not None:
            corner = ", ".join(f"{float(v):g}" for v in retired["cell_origin"])
            said += notices_for_report([Issue(
                "info",
                f"This file stored a box origin ({corner}) the way molbuilder "
                f"no longer reads it, so it was not applied: the atoms are "
                f"centred in the cell (Automatic). If you placed the box "
                f"there, set the origin again on the Cell page.",
                "cell.origin_retired")])
        return ok_structure_response(struct, extra={
            "source_format": ("pdb" if str(_resolved).lower().endswith(".pdb")
                              else "xyz"),
            "title": struct.title or _resolved.name,
            "n_frames": whole.n_frames,
            **({"notices": said} if said else {}),
        })

    # A STRUCTURE PUT BACK, with no file and no text behind it.  A tab that
    # showed a structure before the page was left hands the SAME envelope every
    # edit posts -- atoms as numbers, the facts beside them -- and gets the same
    # answer a file load gives, so a restored viewer is indistinguishable from a
    # freshly-loaded one.  Nothing is parsed: the one deserialiser rebuilds it,
    # and refuses a malformed envelope rather than half-building a structure.
    if isinstance(_pbody.get("structure"), dict):
        from ._shared import struct_from_body
        # `exportFile`'s EXACT INVERSE: the envelope comes back whole -- a
        # frame set with every frame -- and asks for no frame.
        try:
            whole = struct = struct_from_body(_pbody)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False,
                            "error": f"could not restore structure: {exc}"}), 400
        # THE BOX CAME IN THE ENVELOPE, like everything else about these atoms,
        # and `from_dict` applied it.  Nothing more to apply.  A load APPLIES
        # rather than refuses -- a bad box is reported with the answer
        # (`ok_structure_response`) so the user can open the structure and fix
        # it in the Cell page; refusing would make it unopenable, and so
        # unfixable.
        return ok_structure_response(struct, extra={
            "source_format": "xyz",
            "title": struct.title or "restored structure",
            "n_frames": whole.n_frames,
        })

    text: str = ""
    fmt: str = "auto"
    filename: str = ""
    # THE TEXT CARRIES ATOMS, AND ONLY ATOMS (plan § 5q D15).
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    text = body.get("text") or ""
    filename = body.get("filename") or ""

    if not text.strip():
        return jsonify({"ok": False, "error": "empty input"}), 400

    if fmt == "auto":
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("xyz", "pdb"):
            fmt = ext
        else:
            # Sniff by content -- the ONE shared rule (XYZ's first non-blank line is
            # a POSITIVE atom count; anything else is PDB).  Delegate to the same
            # helper the /api/build/molecule path uses so the two never disagree on an
            # edge case (e.g. a leading "0" line: int("0")>0 is False -> pdb).
            fmt = _sniff_structure_format(text)

    try:
        if fmt == "xyz":
            whole = Structure.from_xyz(text, title=filename or None)
        elif fmt == "pdb":
            whole = Structure.from_pdb(text, title=filename or None)
        else:
            return jsonify({"ok": False,
                            "error": f"unknown format {fmt!r}; "
                                     "expected xyz or pdb"}), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse {fmt}: {exc}"}), 400
    try:
        struct = _chosen_frame(whole, body)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # AN XYZ TEXT IS A LONE FILE -- atoms only, nothing beside it -- and says
    # so as the path branch does, in the same sentence (`model/structure.md`
    # § 2.3).
    said: list = []
    if fmt == "xyz":
        from molbuilder.periodicity_gate import notices_for_report
        from molbuilder.workingcopy_structure import (LONE_FILE,
                                                      lone_xyz_said,
                                                      read_notice)
        said = notices_for_report([read_notice(
            LONE_FILE, lone_xyz_said(filename or "The pasted text", text))])

    # Route through the canonical ``ok_structure_response`` helper.
    # Per-atom payload, legacy aliases, validate-pass issues, and the
    # ``extra`` sub-dict all come from the helper in a single call.  Endpoint extras (pdb, summary, the
    # actual parsed format, title fallback) override the
    # canonical defaults at both the top level and the canonical
    # ``extra`` sub-dict — same threading rule for every key.
    return ok_structure_response(struct, extra={
        # /api/build/load's legacy contract: title defaults to the
        # filename (or format name) when the input carries none.
        # Override via extra so downstream code that reuses the
        # Structure sees the canonical (possibly empty) title.
        "title":         struct.title or (filename or fmt),
        "pdb":           struct.to_pdb(),
        "summary":       struct.summary(),
        # Override the canonical XYZ default with the actually-
        # parsed format; the helper threads this through to both
        # the top level and the ``extra`` sub-dict.
        "source_format": fmt,
        "n_frames":      whole.n_frames,
        **({"notices": said} if said else {}),
    })


@bp.route("/api/build/schema/<engine>", methods=["GET"])
def api_build_schema(engine: str):
    """A parameter form's schema, from the catalogue
    (``_shared.catalogue_to_form_schema``; `web/form-schema.md` § 1).

    ``engine`` is ``siesta`` or ``pyscf`` -- that engine's form -- or
    ``common``: the settings one row serves on every engine of the kind,
    drawn once on a page offering the kind on several (`form-schema.md`
    § 1.3), their electronic state apart: ``?surface=state`` gives it
    alone, and ``common`` without a surface the rest.  ``?surface=own``
    narrows an engine's form to its own settings, the common ones left
    out.  Anything else is a 404 naming what is known, never a default
    response.
    """
    engine = (engine or "").strip().lower()
    # Each form's id prefix -- the ids of one page's parts never collide.
    prefixes = {"siesta": "p", "pyscf": "py", "common": "common"}
    if engine not in prefixes:
        return jsonify({
            "ok": False,
            "error": (f"unknown form {engine!r}; expected one of "
                      f"{sorted(prefixes)}"),
        }), 404
    calculation = str(request.args.get("calculation") or "")
    if not calculation:
        return _unstated(calculation=calculation)
    refused = _not_in_tables(calculation=calculation)
    if refused:
        return refused
    surface = request.args.get("surface") or None
    if engine == "common":
        if surface not in (None, "state"):
            return jsonify({"ok": False,
                            "error": f"surface {surface!r}: the common form "
                                     f"is its electronic state (state), or "
                                     f"the rest of it"}), 400
        engine_arg, surface = None, surface or "common"
    elif surface not in (None, "own"):
        return jsonify({"ok": False,
                        "error": f"surface {surface!r}: an engine's form is "
                                 f"whole, or its own settings (own)"}), 400
    else:
        engine_arg = engine
    return jsonify({
        "ok": True,
        "schema": _catalogue_to_form_schema(engine_arg, prefixes[engine],
                                            calculation=calculation,
                                            surface=surface),
    })


def _unstated(**said):
    """A 400 naming what a request left out.  The engine and the calculation
    kind are stated by every caller -- the page states both -- and never
    supplied here."""
    missing = [k for k, v in said.items() if not v]
    return jsonify({"ok": False,
                    "error": (f"the request states no "
                              f"{' and no '.join(missing)} -- every caller "
                              f"states it; nothing is assumed")}), 400


def _not_in_tables(*, calculation: str, engine: Optional[str] = None):
    """A 400 in the tables' own sentence (`execution/architecture.md`
    § 3.3) for a kind the kind table does not hold and, with ``engine``,
    an engine the engine table does not hold or one the kind does not run
    on -- ``None`` when the tables hold them.  The route's boundary: the
    door refuses, the route says it as its own."""
    from molbuilder.kind_table import KindTableError, kind as _kind
    try:
        asked = _kind(calculation)
        if engine is not None:
            _engine(engine)
            asked.on(engine)
    except (KindTableError, EngineTableError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return None


@bp.route("/api/build/preflight", methods=["POST"])
def api_build_preflight():
    """Cheap validation-only endpoint for the live UI hint panel.

    Body: the structure envelope (``structure`` per `_struct_from_body`)
    plus ``engine`` (``"siesta"``/``"pyscf"``), ``params`` (the config
    dict), and ``calculation`` (the kind the validators branch on).

    Returns ``{"ok": True, "issues": [{"severity", "message",
    "where"}, ...]}``; on bad input returns ``{"ok": False, "error":
    ...}`` with HTTP 400.

    Rationale: the build form has many knobs whose interactions
    matter (k-grid vs vacuum padding, hybrid functional vs grid
    level, charged peptide without explicit charge override, ...).
    This endpoint
    runs ``validate(struct, cfg)`` without rendering FDF / PySCF
    text -- much cheaper -- so the UI can call it on debounced form
    input and update a structured issues panel live.
    """
    body = request.get_json(silent=True) or {}
    engine = (body.get("engine") or "").strip().lower()
    params: Dict[str, Any] = body.get("params") or {}
    # The calculation KIND rides the same live check, stated by every tab:
    # validate() composes the kind's own science from it, so the Spectrum
    # tab's panel shows the SAME verdict prep's settings gate gives later --
    # the browser hears it while the person is still at the form.
    calculation = str(body.get("calculation") or "")
    if not calculation:
        return _unstated(calculation=calculation)

    refused = _not_in_tables(calculation=calculation, engine=engine)
    if refused is not None:
        return refused

    # THE STRUCTURE ARRIVES AS DATA, through the one reader every structure
    # door shares: the atoms as numbers and the facts beside them.
    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # Preflight must see the labels + the model's periodicity truth.
    from ._shared import periodicity_checked_for_emit
    struct = periodicity_checked_for_emit(struct)

    try:
        cfg = _config_from_params(config_class(engine), params, calculation)
    except Exception as exc:
        # The "bad params" branch returns ``ok: False`` to match the
        # web-api.md envelope contract; the UI's ``!body.ok`` gate then
        # renders the issue + the user sees the parse error in the
        # issues panel.
        return jsonify({
            "ok":     False,
            "error":  f"bad parameters: {exc}",
            "issues": [Issue("error", f"bad parameters: {exc}",
                             "config").to_json()],
        }), 400

    return jsonify({
        "ok": True,
        "issues": _issues_to_json(
            validate(struct, cfg, calculation=calculation), cfg=cfg,
            calculation=calculation),
    })


# --------------------------------------------------------------------- #
#  Hand-over to Task setup                                              #
# --------------------------------------------------------------------- #

#: The hand-over file's own schema.  **Not** the description's
#: (``task.SCHEMA``), and the difference is the point: this file is
#: deliberately INCOMPLETE -- it carries what the parameter tab knows, and the
#: stages are Task setup's to settle (`engines/stages.md` § 6.5a).  A file
#: claiming the description's schema while failing its reader is worse than one that
#: says what it is; ``check_schema`` refuses a wrong artifact BY NAME, so this
#: cannot be mistaken for a description anywhere.
TASK_HANDOVER_SCHEMA = "molbuilder/task-handover@1"

#: What the hand-over is called on disk.  The extension is LAST on purpose --
#: `task.1st.json`, not `task.json.1st` -- so the editor's suffix map gives it
#: JSON highlighting (`lib/codemirror-load.js`), and so nothing looking for
#: `task.json` finds it.  That second half matters more than it looks:
#: `checkpoint.py::_BUNDLE_DESCRIPTORS` treats a `task.json` as the marker that
#: a folder "declares itself the root of one multi-directory unit of work", so
#: writing a premature one would make the folder claim to be a calculation root
#: before it is one (`checkpointing.md` L1).
from molbuilder.runfiles import TASK_HANDOVER_FILE as TASK_HANDOVER_NAME  # noqa: E402 -- the catalogue's name
from molbuilder.job_request import FILENAME as _JOB_REQUEST_FILENAME  # noqa: E402
from molbuilder.task import FILENAME as _TASK_FILENAME  # noqa: E402
from molbuilder.job_request import SCHEMA as _JOB_REQUEST_SCHEMA_NOW  # noqa: E402

#: WHAT A HAND-OVER'S PROPOSAL IS BUILT FROM -- the keys the hand-over route
#: writes beside the parameters, each with its type, which Task setup reads
#: to draw the description and its job request (`task-setup.md` § 9a).  A
#: file without one is one an older molbuilder wrote: its reader says so.
_PROPOSAL_KEYS = (("task_schema", str), ("task_shape", str),
                  ("job_request_schema", str), ("calculation", str),
                  ("engine", dict), ("proposed_stages", list),
                  ("proposed_varies", list), ("proposed_groups", dict))


def _proposal_refused(over) -> None:
    """Refuse a hand-over the page cannot finish, in its reader's words
    (`task-setup.md` § 9a): a key its proposal is built from missing
    (:data:`_PROPOSAL_KEYS`), or a kind on an engine the kind table does not
    hold (`kind_table.kind(...).on`, the description reader's own
    refusal)."""
    from molbuilder.kind_table import kind
    missing = [k for k, typ in _PROPOSAL_KEYS
               if not isinstance(over.get(k), typ)]
    if not missing and not over["engine"].get("name"):
        missing = ["engine.name"]
    if missing:
        raise ValueError(f"it carries no {', '.join(missing)} -- an older "
                         f"molbuilder wrote it: send it again from its tab.")
    kind(over["calculation"]).on(over["engine"]["name"])


@bp.route("/api/task-setup/handover", methods=["POST"])
def api_task_setup_handover():
    """RENDER the parameter tab's work, for the browser to write.

    **This writes nothing.**  `web/projects.md` § 1 puts raw bytes in the
    content-blind file layer that *"every tab can use"* -- `writeFile` /
    `deleteEntry` -- and a tab that opens files itself bypasses the roots
    guard, the uniform `{ok, ...}` envelope and the sidebar re-list that come
    with it.  So this returns the TEXTS and the caller puts them where the
    user chose, through `projects.writeFile`.

    What is genuinely server-side is the render: only Python can turn a config
    into `<label>.template.toml`, because `template_with_values` narrows the
    catalogue and fills in the answers.

    Four files, and none of them is a runnable anything:

      * ``<label>.template.toml`` -- every parameter with the value in force.
        Without it there is no path from the form to a calculation at all.
      * ``task.1st.json`` -- what the tab knows about the calculation ITSELF:
        the engine, the structure it is of, and what it is called.
      * ``<label>.source.xyz`` + ``<label>.source.molstruct.json`` -- THE
        STRUCTURE, from
        ``StructureCodec``, the same generator ``/api/structure/export`` uses.
        ``molview.md`` § 11.7: the server writes every file, so the pair a
        person downloads and the pair that lands here cannot differ.

    **This is a hand-over, not a description.**  `tabs.md` forbids an in-memory
    "send to tab" hand-off, and this obeys it -- the transfer goes through disk,
    so the receiving tab reads files like every other reader and nothing depends
    on state you cannot see in the folder.  Task setup finishes the job: it
    takes the stages, and on a successful save writes the real ``task.json``
    -- the one layout stated in it (`web/task-setup.md` § 4) -- and removes
    this file.
    """
    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "").lower()
    # The hand-over carries the calculation KIND (handover-procedure § 6:
    # "the hand-over is a Send button on the same endpoint").  The template narrows by it, and
    # the receiving tab writes it into task.json.
    calculation = str(body.get("calculation") or "")
    if not engine or not calculation:
        return _unstated(engine=engine, calculation=calculation)
    refused = _not_in_tables(calculation=calculation, engine=engine)
    if refused:
        return refused
    from molbuilder.kind_table import kind as _kind
    _sender = _kind(calculation).handover
    if _sender is None:
        # NO HAND-OVER FOR A KIND THAT DESCRIBES ON ITS OWN TAB (the kind
        # table's ``handover``, absent; user ruling 2026-08-29 for the
        # composite): nothing is awaiting -- its stages and identity are
        # fixed by design -- so this door refuses by name.
        return jsonify({"ok": False,
                        "error": f"{_kind(calculation).with_article} "
                                 f"describes on its own tab -- nothing is "
                                 f"awaiting"}), 400
    from molbuilder.web.tabs import TABS
    _sender = next(t["label"] for t in TABS if t["key"] == _sender)

    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    params: Dict[str, Any] = body.get("params") or {}
    try:
        cfg = _config_from_params(config_class(engine), params, calculation)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE ENGINE'S DATA FILES, SETTLED BEFORE ITS CALCULATION IS WRITTEN
    # (`handover-procedure.md` § 2.2, plan § 5w K20): the findings its seam
    # adds (`EngineSeam.describe_findings` -- SIESTA's pseudopotentials),
    # asked of the folder the files go into -- files already beside it
    # count -- and a refusal while one is an error, its findings for the tab
    # to put beside the field.  The sender names the folder (`dest`); a
    # caller that names none is asked about the library alone, as the live
    # check is.
    # THE FOLDER'S RULES ARE THE SERVER'S, as the transport describe door's
    # are (`engines/transport.md` § 3.8.2a): a calculation lives under a
    # topic -- project / topic / its folder, three levels inside the picker's
    # root (`job-contracts.md` § 2.5; the file layer's one depth door,
    # `files._depth_inside_root`) -- and one calculation per folder
    # (`describe.refuse_described`, the rule `molbuilder describe` asks).  A sender
    # that names no folder is answered about the library alone.
    dest_dir = None
    if body.get("dest"):
        from molbuilder.describe import DescribeError, refuse_described
        from .files import calculation_folder_refusal
        try:
            dest_dir = _resolve_within_roots(str(body["dest"]))
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status
        too_high = calculation_folder_refusal(dest_dir)
        if too_high:
            return jsonify({"ok": False, "error": too_high}), 400
        try:
            refuse_described(dest_dir, shown=str(body["dest"]))
        except DescribeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
    from molbuilder.issues import calling
    from molbuilder.jobset.engines import engine_seam
    describe_findings = engine_seam(engine).describe_findings
    if describe_findings is not None:
        with calling("describe_findings", engine=engine,
                     where="the hand-over"):
            found, refusal = describe_findings(struct, cfg,
                                               calculation=calculation,
                                               dest_dir=dest_dir)
        if refusal:
            return jsonify({
                "ok": False,
                "error": refusal,
                "findings": _issues_to_json(found, cfg=cfg,
                                            calculation=calculation),
            }), 400

    from molbuilder.identity import normalise_id
    from molbuilder.template import template_with_values
    from molbuilder.template_file import (
        template_filename as _template_filename)

    # WHAT THIS CALCULATION IS CALLED — the identity the person typed, and the
    # destination folder only when they did not.
    #
    # `run-identity.md` § 4: *"The label is the SystemLabel / JOB literal.
    # There is no second name."*
    #
    # Which field carries the identity is the ENGINE's to say, and its record
    # says so (the engine table's ``identity_field``).  The folder name is
    # still the answer when the field is untouched, because the schema default
    # is a placeholder (`siesta`, `pyscf_relax`) and naming a calculation after
    # a placeholder is worse than naming it after the folder somebody chose.
    _field = _engine(engine).identity_field
    _identity = str(getattr(cfg, _field, "") or "")
    _placeholder = str(
        type(cfg).__dataclass_fields__[_field].default or "")
    typed = (_identity if _identity and _identity != _placeholder
             else str(body.get("name") or ""))
    if not typed:
        # REFUSED BY NAME, never defaulted (a hand-over was called
        # "calculation" until 2026-10-19).
        return jsonify({"ok": False,
                        "error": "name: the calculation has no name -- the "
                                 "parameter tab's identity field states it"}), 400
    label = normalise_id(typed)
    formula = str(getattr(struct, "formula", "") or "")

    # AND THE TEMPLATE CARRIES THE SAME ONE.  Choosing the label above is only
    # half of "there is no second name": the template's identity field is what
    # the ENGINE writes its files under, so if it kept the placeholder while
    # `task.json` took the folder's name, the split would simply reappear from
    # the other side.  Normalisation happens once and the result is stored
    # (§ 3 rule 1) — this is the storing.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, **{_field: label})

    # WHAT THE PERSON'S FORM SENT is theirs (`template.md` § 6.6
    # obligation 2): the form sends the fields that hold a value
    # (`form-schema.md` § 1.1), and the calculation's own name.  Every other
    # item is nobody's choice.
    sources = {k: "person" for k, v in params.items()
               if v is not None and v != ""}
    sources[_field] = "person"
    try:
        template_text = template_with_values(cfg, engine=engine,
                                             calculation=calculation,
                                             sources=sources)
    except Exception as exc:                      # a bad value, named
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE STRUCTURE ITSELF, from the one generator.  `molview.md` § 11.7: the
    # server writes every file, because a browser-authored pair drifts from the
    # server's.  So this asks
    # `StructureCodec` for the pair exactly as `/api/structure/export` does, and
    # the two are byte-identical by construction rather than by agreement.
    #
    # The STEM is ours (the calculation's label); the SUFFIXES are the codec's,
    # because the format follows from the frame count and the pairing rule has
    # one home (`model/structure.md` § 2.4).  A caller appending `.xyz` here
    # would be answering a question that already has an answer.
    from molbuilder.workingcopy_structure import StructureCodec
    from ._shared import for_writing
    # A CALCULATION'S STRUCTURE IS WRITTEN AS A SAVE WRITES IT
    # (`_shared.for_writing`): in the user order, its box judged -- AS PREP WILL
    # PLACE IT (`cell.entering`, structure-periodicity.md § 6.0): an
    # engine's origin is not carried into the calculation, so a flush face
    # it left is not this calculation's.  The pair is written as given, its
    # origin's record with it.
    from molbuilder.cell import entering
    struct, struct_notices = for_writing(struct, judged_as=lambda s: entering(
        s, keep_engine_frame=bool(getattr(cfg, "already_relaxed", False))))
    try:
        # ``<label>.source.xyz`` -- the dotted segment is the reservation
        # (`job-contracts.md` § 6.3): every identity is validated
        # dot-free, so no engine output, which stems its files on an
        # identity, can ever take this name -- `<label>.xyz` is SIESTA's own
        # (WriteCoorXmol).  The one call both
        # writers make -- `molbuilder describe` too (`StructureCodec.source_files`).
        made = StructureCodec().source_files(struct, label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    structure_files = [{"name": path.name, "text": blob.decode("utf-8")}
                       for path, blob in made]
    # `source` names the COORDINATE document only.  The sidecar beside it is
    # found by the pairing rule, which has one home (`model/structure.md`
    # § 2.4) and is the codec's -- naming it here would be a second copy of a
    # rule this file does not own.
    geometry_name = next((f["name"] for f in structure_files
                          if not f["name"].endswith(".json")), "")

    from molbuilder.jobset.group import fresh_groups
    from molbuilder.jobset.ladder import fresh_stages
    from molbuilder.task import derive_run, fresh_varies
    proposed = fresh_stages(calculation, engine)
    # WHAT THE HAND-OVER WAITS ON -- the key Task setup still asks: the
    # stages.  The layout is never asked: there is one, and Save writes it
    # (`web/task-setup.md` § 4).
    awaiting = "stages"
    groups = fresh_groups(proposed, engine=engine, calculation=calculation,
                          template_text=template_text)
    from molbuilder.job_request import SCHEMA as _JOB_REQUEST_SCHEMA
    from molbuilder.task import LAYOUT as _TASK_LAYOUT
    from molbuilder.task import SCHEMA as _TASK_SCHEMA
    handover = {
        "schema":    TASK_HANDOVER_SCHEMA,
        # THE DESCRIPTION'S OWN SCHEMA STRING, for the page to write -- the
        # page spells none (`task.SCHEMA`, one home; 2026-10-19) -- its job
        # request's (`job_request.SCHEMA`), and the one layout the
        # description states as its `shape` (`task.LAYOUT`, `web/task-setup.md`
        # § 4).
        "task_schema": _TASK_SCHEMA,
        "task_shape": _TASK_LAYOUT,
        "job_request_schema": _JOB_REQUEST_SCHEMA,
        # JSON has no comments, so the file carries a line that says what it
        # is.  It IS read by a person -- Task setup shows it in the editor --
        # and a file whose whole job is to be handed between two surfaces
        # should not need a document open beside it to be understood.
        # The SENDER is the calculation kind's tab (E-B9): both tabs post
        # through the same shared door (lib/task-handover.js).
        "_what":     f"A hand-over from the {_sender} tab, not a "
                     "description. It carries the parameters (in the .template.toml "
                     "beside it) plus what this calculation is OF. It is missing "
                     f"`{awaiting}` on purpose -- Task setup asks for it, and "
                     "on a successful save writes the real task.json "
                     "and deletes this file. Nothing runs from it. The structure it is OF is the "
                     "file `structure.source` names in this same folder, written "
                     "by the server's one codec: the .xyz, "
                     "which carries the coordinates and the cell, and the "
                     ".molstruct.json beside it carries the region labels and frozen "
                     "atoms.",
        "engine":    {"name": engine},
        # THE ID THROUGH THE CODEC (`task.derive_run`, the one place an id is
        # made), with the proposed ladder's names, as the save derives it.
        "run":       dataclasses.asdict(derive_run(
                         typed, formula,
                         created=datetime.now().astimezone().isoformat(timespec="seconds"),
                         stage_names=tuple(st.name for st in proposed))),
        # WHAT THIS IS OF -- by NAME, pointing at files in this same folder.
        # These names come from the structure that was sent, so they cannot
        # disagree with it (`molview.md` § 9.3a).
        "structure": {"source":  geometry_name,
                      "formula": formula,
                      "atoms":   len(getattr(struct, "elements", []) or []),
                      # THE PAIR'S FINGERPRINT, which the description keeps
                      # and every prep compares (`engines/stages.md` § 6.3).
                      "fingerprint": StructureCodec.fingerprint(made)},
        # Stated rather than omitted so a reader of the file knows what it
        # is waiting on (above).
        "awaiting":  [awaiting],
    }
    # THE KIND RIDES THE HAND-OVER, every kind -- the receiving tab writes
    # it into task.json.
    handover["calculation"] = calculation
    # THE PROPOSED LADDER RIDES IT TOO: the fresh ladder's one owner
    # (`ladder.fresh_stages` -- a vibration's own, read from the box; the
    # ordinary one-stage start otherwise) -- and each stage's group, the job
    # request's, beside it, by the one numbering rule
    # (`jobset.group.fresh_groups`), as `molbuilder describe` makes them.
    # Task setup shows both as what will be written; it re-derives nothing.
    handover["proposed_stages"] = [
        {"name": st.name, "overrides": dict(st.overrides)} for st in proposed]
    handover["proposed_groups"] = {st.name: g
                                   for st, g in zip(proposed, groups)}
    # ...AND WHAT THOSE STAGES VARY, by the one rule `molbuilder describe` writes
    # with (`task.varies_for` over their overrides -- nothing, for a fresh
    # one-stage ladder): the page copies it and seeds no columns of its own
    # (2026-10-18; the page had seeded every stage-group item, so the same
    # hand-over made two descriptions by the door it went through).
    handover["proposed_varies"] = list(fresh_varies(proposed))

    return jsonify({
        "ok":            True,
        "label":         label,
        # THE door (`template_file.template_filename`), not a literal suffix.
        "template_name": _template_filename(label),
        "template_text": template_text,
        "handover_name": TASK_HANDOVER_NAME,
        "handover_text": json.dumps(handover, indent=2) + "\n",
        # A hand-over already in the folder is replaced by this one -- said,
        # so the page can say it.
        "handover_here": bool(dest_dir is not None
                              and (dest_dir / TASK_HANDOVER_NAME).is_file()),
        # Each entry is a file as it would exist on disk, under the name it
        # would exist as -- nothing is left for the browser to work out.
        "structure_files": structure_files,
        "notices": struct_notices,
    })


@bp.route("/api/task-setup/prep", methods=["POST"])
def api_task_setup_prep():
    """Run `prep` for the groups picked -- the SAME function the terminal
    runs.

    **Why a browser may trigger this at all**, when
    `project-layout.md` § 2.2 says the deck cannot be finished in the
    browser: that section's argument is about WHOSE FACTS the deck is
    rendered from, not about which surface presses the button.  `prep`
    needs three inputs: two portable (the template, the description) and
    one the target machine's (`molbuilder.json`).  A named target's record
    supplies the machine half -- that is what `environments/<name>.json`
    IS -- so preparing FOR Sol FROM here is the case
    `preparing-for-another-machine.md` exists for, and preparing for THIS
    machine is the ordinary one.

    What this door does NOT do is submit.  `prep` writes files into the
    calculation and can be run again; `launch` spends a queue slot and
    refuses batch submission by design (one job per invocation, by hand).
    That line is the user's (2026-08-24) and it is where it is because
    the two verbs differ in what they cost to get wrong.

    ``groups`` are the description's group numbers picked, each group one
    job (`job-system.md`, *The task*) -- ``stages``, one stage, for a
    benchmark; the answer is each stage's, ``answers``, in order, and
    ``plan_id`` the plan they make together.

    ``plan: true`` is the entry's PREVIEW (`job-system.md` § 5.0): the plan,
    stopped before the save -- what it would write, the launch the header
    and the run script would carry, what the stage builds on -- or the
    refusal prep would give, with nothing saved, written or recorded.
    Nothing is prepared unseen -- the rule the launch door keeps
    (`submission.md` S4) -- and a Prep names the preview's plan
    (``plan_id``) and is refused when the plan it makes now differs.

    **Everything else is the entry's** (W55 B3, D15): the description, the
    groups, the machine, every refusal in its words.
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "the body is a JSON object"}), 400
    # WORDS, as the page sends them: a field of another type is refused in
    # words, never a 500.
    for key in ("dest", "kind", "target", "plan_id"):
        if body.get(key) is not None and not isinstance(body.get(key), str):
            return jsonify({"ok": False,
                            "error": f"`{key}` is a string"}), 400
    dest_raw = body.get("dest")
    kind = str(body.get("kind") or "").strip()
    target = (body.get("target") or "").strip() or None
    if kind not in ("task", "bench"):
        return jsonify({"ok": False,
                        "error": "kind must be 'task' or 'bench'"}), 400
    groups, stages = body.get("groups"), body.get("stages")
    if kind == "task" and (
            not isinstance(groups, list) or not groups
            or not all(isinstance(g, int) and not isinstance(g, bool)
                       for g in groups)):
        return jsonify({"ok": False,
                        "error": "`groups` is a list of the group numbers "
                                 "picked"}), 400
    if kind == "bench" and (
            not isinstance(stages, list) or len(stages) != 1
            or not isinstance(stages[0], str) or not stages[0].strip()):
        return jsonify({"ok": False,
                        "error": "`stages` is the one stage a benchmark "
                                 "measures"}), 400
    preview = bool(body.get("plan"))
    plan_id = (str(body.get("plan_id") or "").strip() or None)
    # NOTHING IS PREPARED UNSEEN (`job-system.md` § 5.0): a Prep names the
    # plan its preview showed, and the entry refuses one that differs.
    if not preview and not plan_id:
        return jsonify({"ok": False,
                        "error": "a Prep names the plan its preview showed "
                                 "(plan_id) -- preview first "
                                 "(job-system.md § 5.0)."}), 400
    if not dest_raw:
        return jsonify({"ok": False,
                        "error": "no calculation folder given"}), 400
    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not dest.is_dir():
        return jsonify({"ok": False,
                        "error": f"not a directory: {dest_raw}"}), 400

    # The tab labels the local machine `(this machine)`, which is a label
    # and not a name; `LOCAL_TARGET` is the name.  Translated here so the
    # browser sends what it shows and the server speaks one vocabulary.
    from molbuilder.scheduler.record import LOCAL_TARGET
    if target in ("(this machine)", LOCAL_TARGET):
        target = LOCAL_TARGET

    # ---- THE ONE ENTRY (`job-system.md` § 5.3), whole ------------------- #
    # The command line's own prep: the preflight, the save, the steps, the
    # attempt, the transport carry, the launch agreement and their ledger
    # lines -- or, previewed, the same plan stopped before the save -- and
    # its answer returned whole, for the tab to show (`task-setup.md`
    # § 11.1).  The save is the entry's own and asks nothing
    # (`checkpointing.md` § 9).
    from molbuilder.calculation import NotACalculation, open as open_calc
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.prep import prep_task
    try:
        calc = open_calc(dest)
    except (NotACalculation, OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        answers = prep_task(calc, kind, groups=(groups or ()),
                            stage=(stages[0].strip() if kind == "bench"
                                   else None),
                            target=target, allocation=Resources(),
                            preview=preview, plan_id=plan_id)
    except PrepError as exc:
        # Refused, not repaired -- the reader's own words, as the terminal
        # gives them -- WITH what the entry had found by then: the preflight's
        # notes and what the inputs said (a bench's cells left out, which
        # a refusal may point at).  A refused prep wrote nothing.
        return jsonify({
            "ok": False, "error": str(exc),
            "findings": [i.to_json() for i in exc.findings],
            "notes": list(exc.notes),
        }), 400
    except Exception as exc:                      # pragma: no cover
        return jsonify({"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}), 500
    # THE MACHINE IT IS FOR, in the tab's word: the entry's answer -- the
    # one named, else the one the calculation's copy of its record names.
    said = [a.as_dict(dest) for a in answers]
    for d in said:
        if d.get("machine") == LOCAL_TARGET:
            d["machine"] = "(this machine)"
    # WHAT COMES NEXT, as the terminal's prep ends: the launch with nothing
    # named, which sends the group that can go -- or, the prep opening no
    # run of its own (`prep.opened_no_run`),
    # the ladder's own next step.  Launching stays at the terminal
    # (`task-setup.md` § 1).
    from molbuilder.jobset.commands import launch_lines
    from molbuilder.jobset.prep import opened_no_run
    from molbuilder.jobset.runstatus import jobset_status, render_steps
    ladder_next = opened_no_run(answers)
    nxt = ([] if (preview and not ladder_next) or not said else
           render_steps(jobset_status(open_calc(dest))).splitlines()
           if ladder_next else
           launch_lines("task", base=dest)
           if kind == "task" else
           launch_lines("bench", stages[0].strip(), base=dest))
    # THE PLAN, the prep's own -- named on the answers it wrote; a pick
    # with nothing to write names none.
    return jsonify({"ok": True, "answers": said, "next": nxt,
                    "plan_id": next((d["plan_id"] for d in said
                                     if d.get("plan_id")), None)})


@bp.route("/api/task-setup/save", methods=["POST"])
def api_task_setup_save():
    """Write the description and its job request a person has been reading,
    and resolve a hand-over (`execution/job-request.md` § 7).

    **The BUFFERS are the source.**  The editor is where a description and
    its job request are checked and corrected before they are written
    (`web/task-setup.md` § 9a), so this takes each text as edited -- never a
    re-serialisation of a parsed model, which would silently discard
    whatever was typed in the editor.

    **Refused rather than repaired.**  Each text goes through its shipped
    reader (`task.from_text`, `job_request.from_text` against the
    description), so a record that does not parse, names a field the schema
    does not know, or carries no stage is refused with the reason -- the
    same answer the CLI gives, from the same code.  A browser that "fixed" a
    description would be the second, drifting writer this design exists to
    avoid.

    **The hand-over resolves in one direction.**  On success `task.json` exists
    and `task.1st.json` is removed, so the next visit finds one description and
    no ambiguity about which file is current (`engines/stages.md` § 6.5a).
    Removed only AFTER the write succeeds: the reverse order loses the
    parameters if the write fails.
    """
    body = request.get_json(silent=True) or {}
    dest_raw = str(body.get("dest") or "")
    text     = body.get("text")
    request_text = body.get("request_text")
    if not dest_raw:
        return jsonify({"ok": False, "error": "no destination folder given"}), 400
    if not isinstance(text, str) or not text.strip():
        return jsonify({"ok": False, "error": "nothing to save"}), 400
    if not isinstance(request_text, str) or not request_text.strip():
        return jsonify({"ok": False, "error":
                        "no job request to save beside the description -- "
                        "the two are written together "
                        "(execution/job-request.md § 7)"}), 400

    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not dest.is_dir():
        return jsonify({"ok": False, "error": f"not a directory: {dest_raw}"}), 400
    # A FILE CHANGED ON DISK SINCE THE PAGE READ IT IS NEVER WRITTEN OVER
    # (`web/task-setup.md` § 8): the page sends what each record was when
    # the folder answer served it, compared before anything else.
    read_as = body.get("read_as")
    if not (isinstance(read_as, dict) and {"task", "request"} <= set(read_as)):
        return jsonify({"ok": False, "error":
                        "the save names no read_as -- what each record was "
                        "when the page read it; a write that cannot say "
                        "what it writes over is refused (task-setup.md § 8)"
                        }), 400
    on_disk = _read_as(dest)
    stale = [name for which, name in _RECORD_NAMES
             if read_as.get(which) != on_disk[which]]
    if stale:
        return jsonify({"ok": False, "error":
                        f"{' and '.join(stale)} changed on disk since this "
                        f"page read it -- another tab's Save, a `jobset "
                        f"migrate`, a hand edit.  Nothing was saved or "
                        f"written.  Reload the page to read it as it is now; "
                        f"the edits here are not kept, so copy any you want "
                        f"from the editor first."
                        }), 409

    # The description's NAME is `task.py`'s to spell, like its bytes
    # (`task-description.md` § 6.4 -- one reader, so the two surfaces cannot
    # produce different files).
    from molbuilder import job_request as _job_request
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import from_text, write_task
    from molbuilder.task import of as task_of

    # Validate by READING both before anything lands in the calculation
    # folder: each record's one reader (`task.from_text`,
    # `job_request.from_text` against the description, A16), the same gates
    # `prep` reads through.
    try:
        task = from_text(text)
        job_req = _job_request.from_text(request_text, task)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # GATE ③ FIRES HERE TOO (G-1b): a description naming an unknown field,
    # a value outside its bounds, or a bench point that fits no item is
    # refused here rather than at prep, on the cluster.  Same function the CLI runs (`validation.task.preflight`),
    # so the two surfaces cannot disagree; the template beside the
    # description adds the sequence findings when it is already there.
    from molbuilder.template_file import of as _template_of
    from molbuilder.validation.task import (preflight as _task_preflight,
                                            config_class_for as _cfg_cls_for)
    try:
        _tpl_file = _template_of(dest, task.label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    _pf = _task_preflight(
        task,
        template_text=(_tpl_file.text
                       if _tpl_file is not None else None),
        request=job_req)
    # THE CONFIG CLASS RIDES WITH THE FINDINGS.  Without it `_issues_to_json`
    # omits `card` and the page has no card to put a finding on
    # (`web/ui-contract.md` § 5.1).
    _pf_cfg = _cfg_cls_for(task)
    _pf_errs = [i for i in _pf if i.severity == "error"]
    if _pf_errs:
        return jsonify({
            "ok": False,
            "error": "the description or its job request fails its own "
                     "preflight (engines/stages.md § 6.6; "
                     "execution/job-request.md § 7):\n  - "
                     + "\n  - ".join(i.message for i in _pf_errs),
            "findings": _issues_to_json(_pf, cfg=_pf_cfg,
                                        calculation=task.calculation),
        }), 400

    # ONE CALCULATION PER FOLDER (`job-contracts.md` § 2.1 Rule 1).  A folder already
    # describing a DIFFERENT calculation is not a folder this save may land in:
    # the ids say they are different calculations, and overwriting one with the
    # other orphans every warm file and output already keyed to it.
    #
    # Compared by RUN ID, not by path: the id is `<label>_<formula>` and is the
    # one thing that says which calculation a folder is
    # (`run-identity.md` § 2.0a).  Re-saving the SAME calculation is the
    # ordinary case and must stay free.
    from molbuilder.calculation import NotACalculation, open as open_calc
    prior_calc = None
    try:
        prior_calc = open_calc(dest)
        prior = prior_calc.task
    except NotACalculation:
        # DESCRIBED, NO JOB REQUEST YET -- or nothing described, a hand-over
        # (`None`): the door has read it already, so it reads.
        prior = task_of(dest)
    except (OSError, ValueError) as exc:
        # A CALCULATION ITS READERS REFUSE is not written over: their words,
        # as the folder answer gives them (`refused`).
        return jsonify({"ok": False, "error": str(exc)}), 409
    if prior is not None and prior.run.id != task.run.id:
        return jsonify({
            "ok": False,
            "error": f"this folder already describes a different "
                     f"calculation ({prior.run.id!r}); saving {task.run.id!r} "
                     f"here would orphan its results. One calculation per folder — "
                     f"pick or make another.",
        }), 409
    # A LAUNCHED GROUP'S MEMBERS ARE FIXED (`web/task-setup.md` § 5): a
    # stage put into one, or a member launched with it given another group,
    # is refused in the one rule's words (`ready.launched_without`,
    # `ready.joined_launched`).  A stage that has run taken out of the
    # description is refused below, with every change to what it computes.
    if prior_calc is not None and prior_calc.jobset is not None:
        import dataclasses as _dc
        from molbuilder.jobset.ready import joined_launched, launched_without
        from molbuilder.runrecord import LaunchRecordError
        after = _dc.replace(prior_calc, task=task, request=job_req)
        for n in job_req.numbers:
            try:
                hit = launched_without(after, n)
            except LaunchRecordError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 409
            if hit and (hit[1] or any(job_req.group_of(x) is not None
                                      for x in hit[2])):
                return jsonify({"ok": False, "error": joined_launched(
                    after, n, *hit)}), 409

    # AN EXTENSION'S LIMITS, AT THIS DOOR TOO (`engines/transport.md`
    # § 2a.16, *Extending it*; `task-setup.md` § 3): the rungs it keeps,
    # its voltages, its rename and its label stay -- another junction is
    # another calculation, refused above by its id -- the one rule the Transport
    # tab's Describe asks (`transport.reuse.Extension`), each change
    # measured against the folder's own description, so what the command
    # line wrote is never refused for what it holds.  A finished
    # calculation that no longer reads leaves its groups prepared here
    # (§ 2a.16, *At prep*), nothing to keep.
    if getattr(task, "reuses", ""):
        from molbuilder.issues import Issue
        from molbuilder.projects import find_projects_root
        from molbuilder.transport.reuse import Extension, ReuseError
        try:
            ext = Extension.of(task.reuses,
                               tree_root=find_projects_root(dest))
        except ReuseError:
            ext = None
        no = ext.refusal(task, against=prior) if ext is not None else None
        if no is not None:
            return jsonify({"ok": False, "error": no.message,
                            "findings": _issues_to_json([Issue(
                                "error", no.message,
                                )])}), 400

    # WHAT A STAGE THAT HAS RUN COMPUTES IS NOT CHANGED (`web/task-setup.md`
    # § 8; the user, 2026-10-27): its settings, its points, or the stage
    # taken out, each read from the folder's records before and after --
    # refused with the way back, the state before its group's first launch;
    # its run card alone may change (`execution/job-request.md` § 5).
    if prior_calc is not None:
        import dataclasses as _dc
        from molbuilder.jobset.ready import changed_once_run
        from molbuilder.runrecord import LaunchRecordError
        try:
            no_run = changed_once_run(
                prior_calc, _dc.replace(prior_calc, task=task,
                                        request=job_req))
        except (LaunchRecordError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        if no_run is not None:
            return jsonify({"ok": False, "error": no_run}), 409

    # NOTHING TO SAVE (`task-setup.md` § 8; the user, 2026-10-23): the
    # records sent, each read by its reader, are the files' own read the
    # same way, whatever their spelling -- so a text the page holds
    # unedited is one, as the page's own rule says (`edited`) -- no state
    # saved, nothing written, and the page keeps what it shows.
    from molbuilder.task import path_of as _task_path
    _tp, _rp = _task_path(dest), _job_request.path_of(dest)
    try:
        _same = (_tp.is_file() and _rp.is_file()
                 and from_text(_tp.read_text()) == task
                 and _job_request.from_text(_rp.read_text(), task) == job_req)
    except Exception:
        _same = False              # a file its reader refuses is not this one
    if _same:
        return jsonify({
            "ok": True, "unchanged": True, "saved": None, "wrote": [],
            "said": (f"Nothing to save -- {TASK_FILENAME} and "
                     f"{_job_request.FILENAME} already hold what this page "
                     f"sends."),
            "handover_name": TASK_HANDOVER_NAME,
            "handover_here": (dest / TASK_HANDOVER_NAME).is_file(),
            "stages": [st.name for st in task.stages],
            "findings": _issues_to_json(_pf, cfg=_pf_cfg,
                                        calculation=task.calculation)})

    # THE FOLDER'S STATE, SAVED FIRST -- always, by the one function prep
    # calls (`checkpointing.md` § 9; user, 2026-10-03, B10: "make this
    # consistent with B1").  A state that cannot be saved stops the write:
    # it is the one a redo restores.
    from molbuilder.checkpoint import CheckpointError, save_before
    from molbuilder.runtime_config import RuntimeConfigError
    try:
        kept = save_before(dest, "saving the description",
                           engine=task.engine)
    except (CheckpointError, RuntimeConfigError) as exc:
        return jsonify({"ok": False,
                        "error": f"No state was saved, so nothing was "
                                 f"written: {exc}"}), 409

    try:
        write_task(dest, task)                   # atomic (persist.write_json)
        _job_request.write(dest, job_req)        # each file whole
    except OSError as exc:
        return jsonify({"ok": False, "error": f"could not write: {exc}"}), 500

    # The hand-over's REMOVAL is the browser's, through
    # `projects.deleteEntry` -- moving bytes is the content-blind layer's job
    # (`projects.md` § 1), and unlinking here would bypass its guard and the
    # sidebar re-list.  Reported so the caller knows whether to.
    return jsonify({
        "ok":            True,
        "saved":         kept.said(),
        "wrote":         [TASK_FILENAME, _job_request.FILENAME],
        "handover_name": TASK_HANDOVER_NAME,
        "handover_here": (dest / TASK_HANDOVER_NAME).is_file(),
        "stages":        [st.name for st in task.stages],
        # Gate ③'s non-refusing findings (sequence warnings and the
        # fingerprint note): the save proceeded, and the reader deserves
        # what the CLI would have echoed.
        "findings":      _issues_to_json(_pf, cfg=_pf_cfg,
                                         calculation=task.calculation),
    })


@bp.route("/api/task-setup/value", methods=["POST"])
def api_task_setup_value():
    """THE SETTING DOOR, FOR A PAGE (`engines/template.md` § 5.4;
    `web/web-api.md`): what was typed into a cell or a card row, read as its
    item's declared type (`template.read_text`) and judged in its home
    (`template.accepts`) -- ``{ok, value, refusal}``, the value in the
    item's spelling or the one refusal, said as a sentence.  The page keeps
    no reader and no check of its own.

    Body ``{engine, calculation, item, text, home, stage?}``: ``home`` one
    of `template.HOMES`; ``text`` what was typed -- ``""`` a blank, answered
    ``blank: true`` (nothing typed: the caller removes the row's value),
    ``null`` the row's *unset*, judged as one.  A run card's ``time`` and
    ``domain`` are read by the job request's reader, as the allocation's
    are (`job_request.read_lane_ask`)."""
    from molbuilder import template as _T
    from molbuilder.job_request import LANE_ASKS, read_lane_ask
    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "").lower()
    kind = str(body.get("calculation") or "")
    name = str(body.get("item") or "")
    home = str(body.get("home") or "")
    if not (engine and kind and name and home):
        return _unstated(engine=engine, calculation=kind, item=name,
                         home=home)
    refused = _not_in_tables(calculation=kind, engine=engine)
    if refused:
        return refused
    if home not in _T.HOMES:
        return jsonify({"ok": False, "error": (
            f"home {home!r} is not one of {', '.join(_T.HOMES)}")}), 400
    stage = body.get("stage") or None
    text = body.get("text")
    if isinstance(text, str) and not text.strip():
        return jsonify({"ok": True, "value": None, "blank": True,
                        "refusal": None})
    if name in LANE_ASKS and home == "run_card":
        # THE RUN'S OWN TWO ASKS, read by the job request's reader as the
        # allocation's are (`job-request.md` § 3.2a).
        try:
            return jsonify({"ok": True, "value": read_lane_ask(name, text),
                            "refusal": None})
        except ValueError as exc:
            return jsonify({"ok": True, "value": None,
                            "refusal": f"{exc}."})
    try:
        item = _T.one(_T.catalogue(), name, engine=engine)
    except (KeyError, ValueError):
        item = None
    if item is None:
        return jsonify({"ok": False, "error": (
            f"{name!r} is not a setting of {engine}")}), 400
    value, why = _T.read_text(item, text)
    why = why or _T.accepts(item, value, home, engine=engine, kind=kind,
                            stage=stage)
    if why:
        return jsonify({"ok": True, "value": None,
                        "refusal": f"{name} = {_T.shown(text)}{why}."})
    return jsonify({"ok": True, "value": value, "refusal": None})


@bp.route("/api/task-setup/sweepable", methods=["GET"])
def api_task_setup_sweepable():
    """The parameters a benchmark may sweep, for the Task-setup picker.

    **Not the form schema, and the difference is the rule.**
    ``catalogue_to_form_schema`` filters the ``staging`` group out — a
    parameter form does not ask how many ranks the scheduler granted — but
    those are exactly the knobs a benchmark measures.  So this reads the
    catalogue directly and applies § 6.8's rule instead:

      > A key must name a field the engine already declares sweepable — the
        ``execution`` category, which `template.md` § 6.2 defines as *"knobs
        that change speed and not the answer"*.

    Sweeping anything outside it means each point silently measures a
    DIFFERENT calculation, and the comparison is meaningless.

    Each item says whether it is the launch shape (``allocation``, which a
    prep flag may state too), and its homes (`template.home_of`) -- the
    Measure card offers what a benchmark row may hold.
    """
    engine = str(request.args.get("engine") or "").lower()
    if not engine:
        return _unstated(engine=engine)
    try:
        _engine(engine)
    except EngineTableError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    from molbuilder import template as _T
    parsed = _T.catalogue()
    # THE KIND NARROWS IT, as it narrows the columns: a kind's run card
    # offers only what that kind carries (`template.md` § 6.3's sibling
    # rule).
    kind = str(request.args.get("calculation") or "")
    if not kind:
        return _unstated(calculation=kind)
    out = []
    for it in _T.select(parsed, engine=engine, calculation=kind):
        if not _T.is_run_setting(it):
            continue
        out.append({
            "name":            it.name,
            "label":           it.label or it.name,
            "help":            it.help or "",
            "machine_answers": it.allocation,
            # A ROW ON EVERY RUN CARD FROM THE START, blank until stated
            # (`web/task-setup.md` § 6.2b), on the rungs that read it --
            # the catalogue's `run_card` and `stages`.
            "run_card":        it.run_card,
            "stages":          list(it.stages),
            # WHERE A VALUE OF IT MAY LIVE (`template.home_of`): the
            # Measure card offers what a benchmark row may hold.
            "homes":           sorted(_T.home_of(it, kind)),
            # THE VALUE SHAPE (user, 2026-08-20): the machine card births
            # a row at its value in force and offers an enum's choices or
            # a bool's two values from a dropdown -- it can only do either
            # by asking the catalogue.
            "type":            it.type or "",
            "choices":         (list(_T.offered(it, engine, kind))
                                if it.choices else None),
            # THE ONE DEFAULT (`template.default_of`): the kind's own, never
            # the general one -- what the hover calls *Recommended*.
            "default":         _T.default_of(it, kind),
            # The unit rides too: the machine card's default/help text
            # reads m.unit, and max_memory_mb rendered unitless without
            # it (allocation items never pass through the columns door,
            # so this payload is their only source).
            "unit":            it.unit or None,
        })
    # THE RUN'S OWN TWO ASKS, with their words (`job_request.LANE_ASK_ROWS`): the
    # page offers them from here and keeps no table (`task-setup.md` § 6.2b).
    from molbuilder.job_request import LANE_ASK_ROWS
    return jsonify({"ok": True, "engine": engine, "items": out,
                    "lane_asks": [dict(r) for r in LANE_ASK_ROWS]})


def _column_items(engine: str, kind: str):
    """The catalogue items that may be a COLUMN of the stage table for this
    (engine, kind) -- **the one membership rule**, read by the columns route
    and by the presets route (`web/task-setup.md` § 5, § 9), so a preset can
    never offer to fill a column the table would refuse to add.

    `engines/stages.md` § 6.2: *"Any setting the description is allowed to
    hold may become a column. The ones it is not allowed to hold may not."*
    """
    from molbuilder import template as _T
    for it in _T.select(_T.catalogue(), engine=engine):
        # THE membership rule, asked of the setting door rather than
        # restated here (`engines/template.md` § 5.4): a column is a home a
        # stage's override may hold -- never a run setting (the run card's),
        # an item this kind does not carry, one the rung fixes, or one every
        # stage shares.
        if _T.why_not_in(it, "override", engine=engine, kind=kind) is None:
            yield it


@bp.route("/api/task-setup/columns", methods=["GET"])
def api_task_setup_columns():
    """Which parameters may become a column of the stage table.

    `engines/stages.md` § 6.2: *"Any setting the description is allowed to hold
    may become a column. The ones it is not allowed to hold may not."*  There is
    no separate list — § 1.2 already says a stage may name any field of the
    shared schema but a run setting, which is the rung's run card (§ 6.8d;
    the machine's answers among them, `template.md` § 7), a `shared` value
    and a `role` one.  Those rules give the set with nothing left to decide,
    and it is the same membership `prep` applies when it accepts or refuses an
    override: a column offered here is a column `prep` will take.

    **Why this is not `/api/build/schema`.**  That is the PARAMETER FORM's
    schema, and it filters the whole `staging` group out on purpose — a form
    does not ask a person how many ranks the scheduler granted
    (`form-schema.md` § 1.3).  Filtering a panel and limiting a table are
    different jobs.

    `group` rides along because it is still the right answer to a different
    question — which columns the table STARTS with (§ 1.3).
    """
    engine = str(request.args.get("engine") or "").lower()
    _calc_kind = str(request.args.get("calculation") or "")
    if not engine or not _calc_kind:
        return _unstated(engine=engine, calculation=_calc_kind)
    refused = _not_in_tables(calculation=_calc_kind, engine=engine)
    if refused:
        return refused

    from molbuilder import template as _T
    out = []
    for it in _column_items(engine, _calc_kind):
        out.append({
            "name":    it.name,
            "label":   it.label or it.name,
            "help":    it.help or "",
            "unit":    it.unit or "",
            # THE KIND'S OWN DEFAULT (`template.recommended_for`).
            "default": _T.default_of(it, _calc_kind),
            "group":   it.group or "",
            # THE VALUE SHAPE (user, 2026-08-20): the stage table's cell
            # editor renders a dropdown for an enum or a bool, and it can
            # only ask the catalogue.
            "type":    it.type or "",
            # ...OF THE CHOICES THIS KIND MAY TAKE on this engine (the
            # catalogue's `offered`, template.md § 6.3a): a vibration's
            # relaxation cell offers three relaxers, never dynamics.
            "choices": (list(_T.offered(it, engine, _calc_kind))
                        if it.choices else None),
            # THE SAME WRITER THE FORM USES.
            "engine_key": _engine_key_for(it),
            # WHICH RUNGS may carry their own value (template.md § 6.4).
            # Empty means any -- the ordinary case, and the optimization
            # ladder's behaviour.  A composite kind's rungs are different
            # programs on different cells, so a transmission window is the
            # transmission's and an electrode's k-density an electrode's;
            # without this the table has to guess which rung an override
            # belongs to.
            "stages": list(it.stages),
        })
    # ...BY THE RUNG'S ROLE, and how this kind names its rungs' roles rides
    # with the columns as data (`template.stage_role_rule`, plan § 5w K4):
    # the table maps each row to its role with it and keeps no rule of its
    # own.  `null` for a kind without roles -- every rung reads every item.
    return jsonify({"ok": True, "engine": engine, "items": out,
                    "roles": _T.stage_role_rule(engine, _calc_kind)})


def _folder_template(folder, label, *, found=None, engine=None,
                     kind=None) -> dict:
    """What the folder's template answers -- the payload, not the response.

    ``found`` is the template as the calculation opened it (A16); with
    none, ``label`` is the hand-over's name and the door is asked
    (`template_file.of`) -- ``None`` for a folder that is neither, which has no
    template to show.  ``engine`` and ``kind``, a described folder's, add
    ``shared``: every value shared by every stage of that kind, echoed
    read-only with whose it is (`task-setup.md` § 3, `engines/template.md`
    § 6.6 obligation 3).
    """
    from molbuilder.template import (SOURCE_WORDS, read_template, select,
                                     shared_scope_words)
    from molbuilder.template_file import of as template_of
    if found is None:
        if label is None:
            return {"ok": True, "name": None, "values": {}}
        try:
            found = template_of(folder, label)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if found is None:
            return {"ok": True, "name": None, "values": {}}

    try:
        tmpl = read_template(found.text)
    except Exception as exc:
        # A template that does not parse is the user's to fix, and saying which
        # file beats an empty table that looks like "nothing was sent".
        return {"ok": False, "name": found.path.name,
                "error": f"{found.path.name}: {exc}"}

    # Through `select` -- `engines/template.md` § 8.0 owns the rule.
    values = {it.name: it.value for it in select(tmpl) if it.is_set}
    # ...AND WHOSE EACH ONE IS, in the words every surface says it
    # (`template.SOURCE_WORDS`, `engines/template.md` § 6.6 obligation 2):
    # the hover's *"450.0 Ry -- you set this"*.  A file written before the
    # key existed says *not recorded*, never a guess.
    said = {it.name: SOURCE_WORDS[it.source or "unrecorded"]
            for it in select(tmpl) if it.is_set}
    # EVERY VALUE SHARED BY EVERY STAGE, read-only, with whose it is: not a
    # column of the stage table (`shared`, § 5), so the one place on this
    # page a reader sees them -- in the catalogue's category order, by each
    # one's label.  A
    # value nobody gave reads as the template has it: none, *not chosen*.
    shared = []
    if engine and kind:
        from molbuilder.template import catalogue
        mine = {it.name: it for it in select(tmpl)}
        for it in select(catalogue(), engine=engine, calculation=kind,
                         shared=True):
            if kind not in it.shared:
                continue
            own = mine.get(it.name)
            shared.append({
                "name": it.name, "label": it.label or it.name,
                "unit": it.unit,
                "value": own.value if own is not None and own.is_set else None,
                # ...AND WHAT A BLANK ONE IS, as the catalogue calls unset
                # (`null_label`, *(auto)*) and what the engine then does
                # (`not_set`): blank is resolved, never "nothing".
                "null_label": it.null_label, "not_set": it.not_set,
                "said": SOURCE_WORDS[(own.source if own is not None else None)
                                     or "unrecorded"],
                # ...and which rungs it is for, when not every one reads it.
                "scope": shared_scope_words(it)})
    return {"ok": True, "name": found.path.name, "values": values, "said": said,
            "shared": shared}


def _folder_provenance(dest) -> dict:
    """This machine's `molbuilder.json` -- which file it is, what it supplies,
    and what is wrong with where it was read from -- for the folder card.

    THE MACHINE RECORD IS NOT HERE.  Which record answers depends on the
    machine a prep names, and at a calculation's first prep on whether its
    copy exists yet: the prep entry reads it once, at its checkpoint 4, and
    its answer -- a preview's too -- carries that table
    (`job-system.md` § 5.0)."""
    from molbuilder.runtime_config import config_provenance
    try:
        prov = config_provenance(project_dir=dest)
    except Exception as exc:                      # a malformed config
        return {"ok": False, "error": str(exc)}
    # THE WARNINGS TRAVEL WITH THE PROVENANCE.  The terminal prints `shadow` (a `molbuilder.json` sitting unread in a working
    # directory) and the config's mode finding, which `mode_warning` carries in
    # the same words (`placement.machine_config_finding`); a page that showed
    # the resolved path WITHOUT them would tell a person their config is fine
    # while the file they are editing is ignored.
    return {
        "ok": True,
        "sources": [s for s in prov.get("sources") or []
                    if s.get("scope") != "environment"],
        "effective": prov.get("effective") or {},
        "shadow": prov.get("shadow"),
        "mode_warning": prov.get("mode_warning"),
    }


def _row_kinds(job_req, bench) -> dict:
    """``{name: "chosen" | "machine" | "measured"}`` for the bench as it is
    being edited, from prep's own split of the lane."""
    from molbuilder.jobset.prep_inputs import _declared_execution_pins
    try:
        pins, axes, value_axes = _declared_execution_pins(job_req, bench)
    except Exception:                                  # noqa: BLE001
        # A description mid-edit may not read; the cells above still say.
        return {}
    return {**{n: "chosen" for n in pins}, **{n: "machine" for n in axes},
            **{n: "measured" for n in value_axes}}


@bp.route("/api/task-setup/bench-grid", methods=["POST"])
def api_task_setup_bench_grid():
    """The bench grid this description would produce, cell by cell, each
    with its fit on the target's queue -- the one picked on the page, else
    each of the record's (`task-setup.md` § 6.2a).

    **The same list the terminal prints, as data** -- served rather than
    recomputed, because a browser enumerating the grid a second way would
    be exactly the drifting second decider `generator.md` § 4.3a's rebuild
    removed.  `bench_inputs` is the one enumerator; this hands it the
    axes and collects its report.

    POST ``{dest, target?, bench, domain?}``.  ``bench`` is the axis map AS
    IT IS BEING EDITED -- the card's model, not what is saved -- so the list
    tracks the person's typing instead of the last write; ``domain`` the
    queue picked on the page, likewise -- ``""`` for none picked -- which
    the cells are fit on in place of the description's
    (`prep_inputs.bench_inputs`, ``queue``).

    Answers 200 with ``cells`` even when none is kept: *nothing here fits*
    is a result to show, not a failure.  400 is for a description this
    cannot resolve at all (a bad axis name, an engine with no bench lane),
    and carries the reader's own words.
    """
    body = request.get_json(silent=True) or {}
    dest_raw = str(body.get("dest") or "")
    if not dest_raw:
        return jsonify({"ok": False, "error": "no folder given"}), 400
    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not dest.is_dir():
        return jsonify({"ok": False,
                        "error": f"not a directory: {dest_raw}"}), 400
    bench = body.get("bench")
    if bench is not None and not isinstance(bench, dict):
        return jsonify({"ok": False,
                        "error": "bench: must be an object of "
                                 "axis -> points"}), 400
    # The picker offers "(this machine)" as a LABEL, not a name; `LOCAL_TARGET`
    # is the name.  Translated exactly as the prep door translates it, so both
    # doors speak one vocabulary.
    from molbuilder.scheduler.record import LOCAL_TARGET
    target = body.get("target") or None
    if target in ("(this machine)", LOCAL_TARGET):
        target = LOCAL_TARGET

    from molbuilder.jobset.prep_inputs import bench_inputs
    domain = body.get("domain")
    if domain is not None and not isinstance(domain, str):
        return jsonify({"ok": False,
                        "error": "domain: must be a queue's name"}), 400
    rows: list = []
    from molbuilder.calculation import NotACalculation, open as open_calc
    try:
        calc = open_calc(dest)
    except (NotACalculation, OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        # THE SAME REPORT the prep's notes print, as data: the card refreshes
        # it on every keystroke, so it asks for the rows and not the notes.
        # One function, two renderings.
        bench_inputs(calc, target, bench_override=bench, report=rows,
                     **({"queue": domain or None} if "domain" in body
                        else {}))
    except Exception as exc:                      # noqa: BLE001
        # A grid where nothing is left raises, and its report is still the
        # answer worth showing -- the rows left out say why.  The COUNT
        # is computed, never assumed: `bench_inputs` fills the report
        # before its last few refusals, so a raise can follow cells that
        # were kept, and writing 0 here would report them as left out.
        if rows:
            return jsonify({"ok": True, "cells": rows,
                            "kept": sum(1 for r in rows if not r["why"]),
                            "rows": _row_kinds(calc.request, bench),
                            "note": str(exc)})
        return jsonify({"ok": False, "error": str(exc)}), 400
    # EACH ROW'S CLASS, by the one rule prep splits the lane with
    # (`prep_inputs._declared_execution_pins`, `generator.md` § 4.3a): a
    # one-point non-machine entry is CHOSEN (the trials' value), a
    # machine-answered one the grid's axis, a multi-point one MEASURED.  The
    # page paints what this says and decides nothing (2026-10-19).
    return jsonify({"ok": True, "cells": rows,
                    "kept": sum(1 for r in rows if not r["why"]),
                    "rows": _row_kinds(calc.request, bench)})


@bp.route("/api/task-setup/prep-plan", methods=["POST"])
def api_task_setup_prep_plan():
    """What a `prep` would write, stage by stage — `task-setup.md` § 7.1.

    **The names come from the producer, never from the page.**  A list
    composed in the browser would be a second answer free to disagree with
    the thing it describes.  `naming.stage_home` is the one namer (decision
    27): a stage's folder is its token.

    The allocation each stage would carry is the job request's, folded by
    the same function the prep itself resolves through -- so the row cannot
    promise a wall the run will not ask for.

    POST ``{task, request, dest?}`` -- the description and its job request
    AS THEY ARE BEING EDITED, not what is saved, for the reason the grid
    door gives: the cards' edits live in the browser's model until the
    person saves -- and the folder, whose files hold the stages' numbers (a
    stage that has files keeps its own, W38 F4).  Nothing here writes --
    but what the status door records of a run's ending as it reads it, as
    for every reader of a run (the memory need, ``need``).
    """
    body = request.get_json(silent=True) or {}
    raw = body.get("task")
    raw_request = body.get("request")
    if not isinstance(raw, dict):
        return jsonify({"ok": False,
                        "error": "task: must be the description object"}), 400
    if not isinstance(raw_request, dict):
        return jsonify({"ok": False,
                        "error": "request: must be the job request object"
                        }), 400
    from molbuilder import job_request as _job_request
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runfiles import manifest
    from molbuilder.task import Task
    try:
        task = Task.from_dict(raw)
        job_req = _job_request.from_dict(raw_request, task)
    except Exception as exc:                      # noqa: BLE001
        # A description mid-edit is often unreadable, and that is ordinary.
        # Its own words, so the card can say why rather than going blank.
        return jsonify({"ok": False, "error": str(exc)}), 400

    # ONE allocation for the calculation (`job-request.md` § 3.1), and the
    # run card each rung runs at (§ 3.2).  Two reads because they are two
    # questions; neither is recomputed here.
    from molbuilder.job_request import LANE_ASKS
    from molbuilder.jobset.prep_inputs import fold_allocation
    def _ask_of(stage: str) -> dict:
        # THE EFFECTIVE ASK, by the one fold prep uses (`fold_allocation`):
        # the run card's own asks over the calculation's -- one line per
        # stage, said by the server (2026-10-18; the page had printed both
        # and composed which the wrapper carries).
        chosen = {k: v for k, v in job_req.run_condition(stage).items()
                  if k in LANE_ASKS}
        res, _src = fold_allocation(None, job_req.allocation, chosen)
        return {"domain": res.domain, "time": res.time, "mem": res.mem}
    # The plan card reads the RECORDS as posted, mid-edit, with no
    # machine resolved -- so it reports the condition as WRITTEN, without
    # asking the enumerator, which needs a target.  THE FOLDER, when the page
    # names it (`dest`), is where the stages' numbers are read: a stage that
    # has files keeps its number (`naming.stage_home`, W38 F4), so the
    # plan names the folders prep will write, not the description's places.
    folder = None
    if body.get("dest"):
        try:
            folder = _resolve_within_roots(str(body["dest"]))
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status

    # THE CALCULATION THE FOLDER HOLDS, as saved: what its runs held (the
    # memory need) and which of its stages run as a run of points (the
    # walk's files) -- a folder that holds none has neither.
    saved = None
    if folder is not None:
        from molbuilder import calculation as _calculation
        try:
            saved = _calculation.open(folder)
        except (_calculation.NotACalculation, OSError, ValueError):
            saved = None

    # THE MEMORY EACH STAGE NEEDS, measured on its runs, at the ranks its
    # card states as edited (`job-request.md` § 3.1a) -- asked by the page
    # when the folder or a rank count changes (``need``), read in the
    # calculation the folder holds: a folder that holds none has nothing
    # measured.
    needs = {}
    if body.get("need"):
        from molbuilder.jobset.memory import need as _need
        for st in task.stages:
            needs[st.name] = _need(
                saved, st.name,
                job_req.run_condition(st.name).get("mpi_np"))
    rows = []
    for st in task.stages:
        home = stage_home(folder, task, st.name)
        files = manifest(home.label, home.token, engine=task.engine,
                         calculation=task.calculation)
        if _runs_points(saved, st.name):
            # A RUN OF POINTS IS SENT AS ITS WALK (`execution/runs-of-points.md`
            # § 2.4, the launch folder): its header is the walk's, in the
            # stage's `launch/`, never one of its own.
            sbatch = home.names.name(".sbatch")
            files = ([f for f in files if f["name"] != sbatch]
                     + _walk_files(home.names.stem))
        rows.append({"stage": st.name, "dir": home.token,
                     **({"need": needs[st.name].to_wire()} if needs else {}),
                     "ask": _ask_of(st.name),
                     # PER STAGE, because the condition is (§ 3.2): the
                     # calculation's card with this rung's laid over it.
                     "chosen": {k: v for k, v in
                                job_req.run_condition(st.name).items()
                                if k not in LANE_ASKS},
                     # AND WHAT THIS RUNG WILL BE NAMED (user, 2026-09-07).
                     # Every name comes out of the grammar (`runfiles`
                     # § 2.2a) with THIS rung's token, so the card cannot
                     # show a spelling the writers do not use -- which is
                     # the failure that module exists for.  The engine
                     # filters it: a SIESTA run is not told about `.py`.
                     # ...ON THE LABEL ITS FILES CARRY: a transport lead's
                     # own stem (`StageHome.label`, `job-contracts.md` § 2.1
                     # rule 2).  A file that lands outside the stage's
                     # folder says where -- a prep's log, at the root.
                     "files": _placed(files, ("stage", "run", "launch"))})
    bench = None
    if job_req.bench:
        # Every axis, with its points.  A row of length one is a DECISION and
        # measures one cell; the card says which is which (§ 6.2b).
        #
        # AND WHERE THE SWEEP LANDS, asked of `bench_container`, which is the
        # ONE spelling of that rule: `<NN>_<stage>/bench`.
        #
        # ONE ENTRY PER RUNG, because that is what a sweep is: `prep bench`
        # takes a stage, the container lives inside the stage it measures,
        # and each entry here is a command someone actually runs.  Joining
        # them into one cell is wrong twice -- it hides which
        # rung each belongs to, and the card's directory column is
        # `max-content` shared down the list, so a joined cell widens that
        # column for every row, which is exactly what the stylesheet says
        # must not happen ("a rung that states nothing and one that states
        # four values must not move the two columns a reader scans down").
        #
        # `rows` cannot be empty here: `stages.md` § 6.5 refuses a
        # calculation with no stages, so there is no empty case to invent a
        # location for.
        from molbuilder.paths import bench_container
        bench = {"axes": {k: list(v) for k, v in job_req.bench.items()},
                 "allocation": {"domain": job_req.allocation.domain,
                                "time": job_req.allocation.time,
                                "mem": job_req.allocation.mem},
                 "rungs": [{"stage": r["stage"],
                            "dir": bench_container(r["dir"])}
                           for r in rows]}
    # WHAT PREP WRITES FOR THE WHOLE RUN, beside the per-stage rows above
    # (user, 2026-09-07: *"we should add a card that list all the generated
    # data file from the setup ... more the ones we designed to be
    # generated"*).  These are MOLBUILDER's own records -- not the engine's
    # outputs, which belong to the run and are listed by the Results tab.
    #
    # FROM THE CATALOGUE, like the stage rows: its fixed-name rows at the
    # calculation's root that prep writes (`runfiles.fixed`).
    from molbuilder.runfiles import fixed
    bundle = fixed("calculation", when=("prep",),
                   calculation=task.calculation, engine=task.engine)
    # AND WHAT THE CALCULATION WRITES ONCE, beside the per-rung lists: the
    # deck's own files that carry no stage token.  The SETUP's three
    # (`.template.toml` and the source pair) are asked for by moment rather
    # than subtracted here -- the card above already lists them, with the one
    # thing this cannot say: whether they are there yet.
    # And what `summarize task` writes there from results that exist -- the
    # transport record, a sweep's comparison -- whose own line says so.
    # Each that lands elsewhere than the calculation's folder says where:
    # a vibration's result, at each run's root.
    once = _placed(manifest(task.label, None, engine=task.engine,
                            when=("prep", "run", "summarize"),
                            calculation=task.calculation), ("calculation",))
    out = {"ok": True, "stages": rows,
           "bench": bench, "bundle": bundle, "once": once,
           "warm": _warm_in_effect(task, folder)}
    if needs:
        from molbuilder.jobset.memory import largest
        out["largest_need"] = largest(list(needs.values())).to_wire()
    return jsonify(out)


def _placed(rows, home) -> list:
    """A card group's file rows (`runfiles.manifest`), each that lands
    outside the group's own place -- ``home``, the levels its head names --
    keeping the catalogue's words for where it does (``where``); the rest
    saying nothing of it (`web/task-setup.md` § 7.2)."""
    return [dict(r, where=(r["where"] if r["level"] not in home else ""))
            for r in rows]


def _runs_points(calc, stage: str) -> bool:
    """Whether ``stage`` of the calculation as saved runs as a run of
    points -- its points as prepared, by its kind's declaration
    (`jobset.points.rung_points_of`); ``False`` with no calculation, and
    for a stage it does not hold or whose points do not read."""
    if calc is None:
        return False
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.points import rung_points_of
    try:
        return bool(rung_points_of(calc, stage))
    except (OSError, ValueError, PrepError):
        return False


def _walk_files(stem: str) -> list:
    """A RUN OF POINTS' WALK, in its stage's `launch/` -- the walk script,
    its header and its log (`execution/runs-of-points.md` § 2.4, the
    launch folder): each named by the one composer
    (`runfiles.GroupNames.of_points`) and said in the catalogue's words,
    its row found as every reader finds a launch group's file
    (`runfiles.row_for`)."""
    from molbuilder.runfiles import (LAUNCH_DIR, LEVEL_WHERE, GroupNames,
                                     row_for)
    gn = GroupNames.of_points(stem)
    out = []
    for tail in (".run.sh", ".sbatch", ".log"):
        name = f"{LAUNCH_DIR}/{gn.name(tail)}"
        a = row_for(name)
        out.append({"name": name, "what": a.what, "when": a.when,
                    "level": a.level, "where": LEVEL_WHERE[a.level],
                    "only": a.only})
    return out


def _warm_in_effect(task, folder) -> "dict | None":
    """WHICH RESTART-FILE LIST THIS CALCULATION FOLLOWS -- the one door's
    answer (`warmfiles.warm_list`, `job-contracts.md` § 4.2a): its own copy
    beside ``task.json``, or molbuilder's for its engine -- and where a
    custom copy goes, which the card says (`web/task-setup.md` § 7.2; plan
    W36 ⑧: "provide information on the task setup web ui").  ``None`` for an
    engine with no list."""
    from molbuilder.warmfiles import WarmFilesError, path_of, warm_list
    try:
        in_effect = warm_list(str(task.engine), None, folder)
    except WarmFilesError:
        return None
    return {"path": in_effect.path, "own": in_effect.own,
            "copy_to": (str(path_of(folder))
                        if folder is not None else None)}


def _folder_continue_from(calc) -> dict:
    """``{stage: choices}`` for every stage that continues from another by
    default -- what Task setup's **Continue from** offers
    (`continuation.continue_from_choices`); a stage that continues from
    nothing is absent.  Fail-soft per stage, like the folder's other parts."""
    from molbuilder.jobset.continuation import continue_from_choices
    out = {}
    for st in calc.task.stages:
        try:
            got = continue_from_choices(calc, st.name)
        except Exception as exc:                  # noqa: BLE001
            got = {"error": str(exc)}
        if got is not None:
            out[st.name] = got
    return out


def _folder_walks(calc) -> dict:
    """``{stage: words}`` -- each stage that runs as a run of points, its
    one job in prep's own words (`prep.walk_words`): the run card says
    what its wall is -- one for the whole run, or each point's, the job
    asking their sum (`execution/runs-of-points.md` § 1.2, *wall*).  A
    stage whose points do not read is left out, as the folder's other
    parts leave out what they cannot read."""
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.prep import walk_words
    out = {}
    for st in calc.task.stages:
        if not _runs_points(calc, st.name):
            continue
        try:
            out[st.name] = walk_words(calc, st.name)
        except (OSError, ValueError, PrepError):
            continue
    return out


def _folder_attempts(calc) -> dict:
    """How many runs of each stage were launched -- the payload.  A RUN IS
    ONE LAUNCH of the stage, `run-<n>/` -- a swept rung's too, its points
    inside it (`engines/transport.md` § 2a.11) -- and only the newest can be
    unlaunched, prep's or one a refused send opened
    (`materialize.newest_launched`), so it is not counted.  A stage whose
    newest launch record does not read is left out: its count is not
    known."""
    from molbuilder.jobset.materialize import newest_launched
    from molbuilder.jobset.naming import stage_home
    from molbuilder.paths import attempt_dir, attempts_in
    from molbuilder.runrecord import LaunchRecordError
    dest, task = calc.base, calc.task
    stages = {}
    for st in task.stages:
        home = stage_home(dest, task, st.name)
        ns = attempts_in(home.dir)
        try:
            newest = newest_launched(home.dir)
        except LaunchRecordError:
            continue
        stages[st.name] = {"attempts": (
            len(ns) - (newest != attempt_dir(home.dir, ns[-1])) if ns
            else 0)}
    return {"stages": stages}


@bp.route("/api/task-setup/folder", methods=["GET"])
def api_task_setup_folder():
    """**What is this folder?** — Task setup's one per-directory answer.

    `web/task-setup.md` § 2.1 is the rule this exists to make keepable: *"the
    page holds no state of its own… the folder is the only link."*

    **The page is a function of (folder, engine, machine), in that
    dependency order.**  This door answers the first.  The engine's
    vocabulary (`columns` / `presets` / `sweepable`) and the machine's
    records are shared caches on their own keys and are NOT here — they do
    not change when the folder does, and folding them in would make every
    directory change refetch them.  `bench-grid` is not here either: it
    takes `{dest, target}`, so it belongs to the PAIR and is invalidated
    when either moves.

    **The answer names its subject**, and that is the half a reset list
    cannot cover.  There are two ways a page shows the wrong folder — state
    that LINGERS, and an answer that LANDS LATE — and clearing things fixes
    only the first.  `dir` is here so a consumer that has moved on can
    discard it, which is `calcdir.json`'s rule (`project-layout.md` § 1.4a)
    applied to the wire: a record names its own place and the reader checks.

    **Composed, never recomputed.**  A part that fails carries its own `error` instead
    of failing the answer: a malformed template must not cost you the
    description beside it.
    """
    from molbuilder.task import FILENAME as TASK_FILENAME

    dir_raw = str(request.args.get("dir") or "")
    if not dir_raw:
        return jsonify({"ok": False, "error": "no folder given"}), 400
    try:
        folder = _resolve_within_roots(dir_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not folder.is_dir():
        return jsonify({"ok": False,
                        "error": f"not a directory: {dir_raw}"}), 400

    def _read_handover():
        """The hand-over file through its one reader: `persist.read_json`,
        the schema this module writes it with (`TASK_HANDOVER_SCHEMA`) and
        what its proposal is built from (:func:`_proposal_refused`) checked
        -- a file of another major, one that does not read, or one the page
        cannot finish is said in the reader's words, never read as this
        one's; the page refuses it as it opens."""
        from molbuilder.persist import check_schema, read_json
        f = folder / TASK_HANDOVER_NAME
        if not f.is_file():
            return None
        try:
            obj = read_json(f)
            check_schema(obj.get("schema") if isinstance(obj, dict) else None,
                         TASK_HANDOVER_SCHEMA, label=f.name)
            _proposal_refused(obj)
            return obj
        except Exception as exc:                  # noqa: BLE001
            return {"error": f"{TASK_HANDOVER_NAME}: {exc}"}

    def _described():
        """The description, through its one door (`task.of`,
        `execution/architecture.md` § 3.2, A16) -- the `Task` a
        molbuilder-written `task.json` holds, ``None`` with none, or the
        words its reader refused one with."""
        from molbuilder.task import of as task_of
        try:
            return task_of(folder)
        except Exception as exc:                  # noqa: BLE001
            return {"error": str(exc)}

    # THE CALCULATION, OPENED ONCE for every part below that reads it
    # (A16); a folder that is no calculation yet -- a hand-over, or
    # empty -- has none, and the description alone is asked for.
    from molbuilder.calculation import NotACalculation, open as open_calc
    # A CALCULATION ITS READERS REFUSE -- a template holding a run setting,
    # a plan written before a rule, each naming `jobset migrate` -- is
    # answered with their words (`refused`), which the page shows and Save
    # refuses with; it is never read as a calculation not yet prepared.
    _calc = refused = None
    try:
        _calc = open_calc(folder)
    except NotACalculation:
        pass
    except (OSError, ValueError) as exc:
        refused = str(exc)

    # `task.json` and the hand-over are mutually exclusive by design: a save
    # writes the first and deletes the second (`task-setup.md` § 3), so the
    # page's MODE follows from which is here rather than from a flag it has
    # to keep.
    _task = _calc.task if _calc is not None else _described()
    described = (_task.to_dict() if hasattr(_task, "to_dict") else _task)
    handover = None if described is not None else _read_handover()

    def _requested():
        """The job request beside a description that reads, through its
        door (`job_request.of`, against that description, A16) -- its
        object, ``None`` with none, or the words its reader refused it
        with (`execution/job-request.md` § 7)."""
        if _calc is not None:
            return _calc.request.to_dict()
        if not hasattr(_task, "stages"):
            return None
        from molbuilder import job_request as _job_request
        try:
            found = _job_request.of(folder, _task)
        except Exception as exc:                  # noqa: BLE001
            return {"error": str(exc)}
        return None if found is None else found.to_dict()
    requested = _requested()
    # THE LABEL THE TEMPLATE IS NAMED ON: the description's, else the
    # hand-over's name through the normaliser the hand-over named it with.
    if hasattr(_task, "label"):
        _label = _task.label
    elif isinstance(handover, dict) and (handover.get("run") or {}).get("name"):
        from molbuilder.identity import normalise_id
        _label = normalise_id(handover["run"]["name"])
    else:
        _label = None
    # THE STRUCTURE PAIR THE DESCRIPTION NAMES -- its geometry, and the
    # sidecar beside it by the pairing rule, which has one home
    # (`sidecars.molstruct.sidecar_path_for`, `model/structure.md` § 2.4):
    # the page names neither file itself.
    from molbuilder.sidecars.molstruct import sidecar_path_for
    _doc = described if isinstance(described, dict) else (
        handover if isinstance(handover, dict) else {})
    _source = str((_doc.get("structure") or {}).get("source") or "")
    structure_pair = ({"geometry": pathlib.Path(_source).name,
                       "sidecar": sidecar_path_for(
                           pathlib.Path(_source).name).name}
                      if _source else None)

    return jsonify({
        "ok": True,
        # THE SUBJECT.  Every folder-scoped answer says which folder it is
        # about; a consumer on another one discards it rather than painting.
        "dir": str(folder),
        "mode": ("description" if described is not None
                 else "handover" if handover is not None else "empty"),
        "description": described,
        # ITS JOB REQUEST, as its reader read it -- an object, `{error}`, or
        # null with none (a description written by hand): then the groups
        # a new one takes, by the one rule (`group.default_groups`), for
        # the page to write with Save (`execution/job-request.md` § 7).
        "request": requested,
        "proposed_groups": (_default_groups_of(folder, _task)
                            if requested is None and hasattr(_task, "stages")
                            else None),
        "handover": handover,
        # THE FILES' NAMES, said here so the page spells none.
        "task_name": TASK_FILENAME,
        "request_name": _JOB_REQUEST_FILENAME,
        "request_schema": _JOB_REQUEST_SCHEMA_NOW,
        "handover_name": TASK_HANDOVER_NAME,
        # WHICH FILES ARE HERE -- names only.  The "what gets written" card
        # asks *is the file this description names actually present*, which
        # is a fact about the folder and belongs in the folder's answer.
        "files": sorted(e.name for e in folder.iterdir() if e.is_file()),
        "structure_pair": structure_pair,
        "template": _folder_template(
            folder, _label,
            found=_calc.template if _calc is not None else None,
            engine=getattr(_task, "engine", None),
            kind=getattr(_task, "calculation", None)),
        "provenance": _folder_provenance(folder),
        # Only a calculation has stages, so only then is there anything to
        # count.
        "attempts": (_folder_attempts(_calc) if _calc is not None
                     else {"stages": {}}),
        # EACH RUN OF POINTS' ONE JOB, in prep's words -- what its run
        # card's wall is (`web/task-setup.md` § 6.2b).
        "walks": _folder_walks(_calc) if _calc is not None else {},
        # WHAT EACH STAGE CAN CONTINUE FROM -- its default, the runs of the
        # stage before it with what each was, and `--cold` where it is a
        # choice (`job-system.md` § 5.4, plan W37): the Continue-from choice
        # reads the folder's answer, so it shows what prep would take.
        "continue_from": (_folder_continue_from(_calc)
                          if _calc is not None else {}),
        # WHY THIS DESCRIPTION HAS NO BENCH, or null -- the prep entry's own
        # answer (`prep_inputs.bench_refusal`), so the page offers the
        # Measure step exactly where `prep bench` would take it.
        "bench_refusal": _folder_bench_refusal(_task, handover),
        # WHAT EACH RECORD WAS WHEN THIS ANSWER READ IT, for Save to compare
        # (`web/task-setup.md` § 8).
        "read_as": _read_as(folder),
        # WHAT ITS READERS REFUSE OF THE CALCULATION, or null.
        "refused": refused,
        # WHAT THE KIND FIXES of the stage table -- a transport description's
        # five rungs, in order -- by the reader's own answer
        # (the kind table), so the page offers no stage-table control
        # the reader refuses (`web/task-setup.md` § 5.3); null for every
        # other kind.
        "fixed": _folder_fixed(_task),
        # AN EXTENSION'S LIMITS -- a transport calculation that reuses a
        # finished one: the rungs that run once, fixed by it, and the words
        # the page says them in (`web/task-setup.md` § 3;
        # `engines/transport.md` § 2a.16, *Extending it*); null otherwise.
        "extension": _folder_extension(_task, folder),
        # HOW A TRANSPORT DESCRIPTION TREATS ITS BIAS LIST, by the one rule
        # the record names it with (`Task.treatment`); null for every other
        # kind.  The "What came over" card echoes it (`task-setup.md` § 3).
        "treatment": getattr(_task, "treatment", None),
        # WHAT THE CITED JUNCTION'S FRAMES ARE -- the count and, for a mode's
        # set, its definition -- by the describe door's own helper
        # (`frame_average.answer`; `engines/transport.md` § 3.8.1); null
        # for every other kind.  The "What came over" card draws it.
        "frame_set": _folder_frame_set(_task, folder),
        # THE MACHINE IT IS SET TO -- its first prep's, read from its copy
        # of the record, or null before that (`configuration.md` M-3): the
        # page shows it fixed rather than offering a choice that can only be
        # refused (W55 B4, D12).
        "set_to": _folder_set_to(folder),
        # WHICH STAGES ARE PREPARED AS A BENCHMARK -- each with the prep
        # entry's own sentence (`prep.prepared_already`), which the page
        # shows in place of a Prep it would refuse; a task's stages are the
        # ladder's below.
        "prepared": (_folder_prepared(_calc) if _calc is not None
                     else {"bench": {}}),
        # THE TASK'S LADDER, as `prep task` shows it before it asks: each
        # stage described, prepared or launched with the ready door's words,
        # and the groups `prep task` offers (`jobset.ready`) -- and THE
        # GROUPS: each stage's number, whether the rule lets it share the
        # group above and why not, and where each group stands
        # (`engines/stages.md` § 2; `web/task-setup.md` § 5.3, § 11).
        **(_folder_ladder(_calc) if _calc is not None
           else {"ladder": {"stages": [], "offer": []},
                 "groups": {"stages": [], "groups": []}, "steps": []}),
    })


def _default_groups_of(folder, task) -> dict:
    """``{stage: group}`` -- the groups a description with no job request
    takes, by the one rule a new calculation is numbered with
    (`group.default_groups`), against the template beside it when it
    reads."""
    from molbuilder.jobset.group import default_groups
    from molbuilder.template_file import of as _template_of
    try:
        tf = _template_of(folder, task.label)
    except ValueError:
        tf = None
    groups = default_groups(task, None, tf.text if tf is not None else None)
    return {st.name: g for st, g in zip(task.stages, groups)}


def _folder_fixed(task) -> Optional[dict]:
    """The stages a kind's part on its engine fixes (the kind table),
    ``{"stages"}``, as the stage table reads them; ``None`` for a pair that
    fixes none, or no description."""
    from molbuilder.kind_table import pair
    fixed = pair(getattr(task, "calculation", ""),
                 getattr(task, "engine", ""))
    return (None if fixed is None or not fixed.stages
            else {"stages": list(fixed.stages)})


def _folder_extension(task, folder) -> Optional[dict]:
    """An extension's limits, for a transport calculation that reuses a
    finished one -- the one door's wire form (`transport.reuse.Extension`:
    the rungs it keeps, what it offers and what each control says) -- or
    ``None``; ``{"error"}`` when the finished calculation does not read
    (`engines/transport.md` § 2a.16)."""
    reuses = getattr(task, "reuses", "") or ""
    if not reuses:
        return None
    from molbuilder.jobset.reuse import SHARING_RUNG
    from molbuilder.projects import find_projects_root
    from molbuilder.transport.reuse import Extension, ReuseError
    try:
        # ...AND THE RUNG WHOSE WINDOW A FIT IS WRITTEN INTO, as the
        # Transport tab's extend answer names it.
        return dict(Extension.of(reuses, tree_root=find_projects_root(
            pathlib.Path(folder))).to_wire(), window=SHARING_RUNG)
    except ReuseError as exc:
        return {"error": str(exc)}


def _folder_ladder(calc) -> dict:
    """``{"ladder": {stages, offer}, "groups": {stages, groups}}`` -- the
    ready door's answer for every stage (`ready.ladder`) and the groups
    `prep task` offers (`ready.prep_offer`), the one `prep task` and
    `status` read; and the groups: each stage's number, why it may not join
    the group above, and whether it was launched (its group is then fixed);
    each group's members, where it stands, and whether its prepared jobs
    hold its members (`web-api.md`); and ``steps`` -- each group's next
    step, as `jobset status` prints it under *each group, and what is next*
    (`runstatus.jobset_status`), so the page says it whatever it last did
    (`task-setup.md` § 11).  Fail-soft, like the folder's other parts."""
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.group import why_not_join
    from molbuilder.jobset.ready import task_state, why_group_fixed
    from molbuilder.jobset.runstatus import every_run, jobset_status
    try:
        state = task_state(calc)
        task, text = state.task, state.template_text
        answers, gs = state.stages, state.groups
        # THE STATUS DOOR, once: each group's next step, and each stage's
        # runs copied in from a finished calculation (`.copied-from`) --
        # every one, each by its folder, as the root's every run reads them
        # (`runstatus.every_run`).
        status = jobset_status(calc)
        copied = {s.name: [{"dir": r["dir"], "copied_from": r["copied_from"]}
                           for r in every_run(calc.base, status.calc.task, s)
                           if r["copied_from"]]
                  for s in status.stages}
        # EACH STAGE'S ROW; the reader's check (`group.check_groups`)
        # refuses at Save what the rule forbids after edits.
        rows = []
        job_req = calc.request
        number = [job_req.group_of(s.name) for s in task.stages]
        for i, (st, a) in enumerate(zip(task.stages, answers)):
            # WHY THIS ROW MAY NOT JOIN THE GROUP ABOVE, as the records
            # stand: the rule's own sentence (`group.why_not_join`); a
            # launched group above is the page's to apply, from `launched`
            # and `why_fixed`, to every row as edited (`task-setup.md` § 5).
            above = [s.name for j, s in enumerate(task.stages[:i])
                     if i and number[j] == number[i - 1]]
            why_same = (why_not_join(task, job_req, st.name, above, text)
                        if above else None)
            rows.append({"stage": st.name, "group": number[i],
                         "launched": a.launched,
                         "why_same": why_same,
                         "why_fixed": (why_group_fixed(calc, st.name)
                                       if a.launched else None)})
        return {
            "ladder": {"stages": [{"stage": a.stage, "group": a.group,
                                   "state": a.state, "detail": a.detail,
                                   "why": a.why,
                                   "copied": copied.get(a.stage, [])}
                                  for a in answers],
                       "offer": state.prep_offer},
            "groups": {"stages": rows, "groups": [
                {"number": g.number, "members": g.members,
                 "state": ("launched" if g.launched else
                           "prepared" if g.prepared or g.changed
                           else "not prepared"),
                 # ...IN THE WORDS EVERY SURFACE SHOWS (`GroupAnswer.said`).
                 "said": g.said,
                 # A PREPARED JOB HOLDING OTHER MEMBERS: prepared again,
                 # formed anew (`job-system.md`, *The task*, step 3).
                 "matches": not (g.changed or g.members_changed),
                 "preparable": g.preparable, "why": g.why} for g in gs]},
            "steps": [s.to_dict() for s in status.steps]}
    except (OSError, ValueError, PrepError) as exc:
        # WHAT ITS READERS REFUSE, said on the card; a defect in this code
        # is not one of them.
        return {"ladder": {"error": str(exc), "stages": [], "offer": []},
                "groups": {"error": str(exc), "stages": [], "groups": []},
                "steps": []}


def _folder_frame_set(task, folder):
    """The cited junction's frames, as the describe door answers them
    (`frame_average.answer`), for a description of a kind that cites its
    structure (the kind table's ``description.cites`` -- a transport
    calculation's junction); ``None`` for every other kind.  A citation that
    does not read says why, in its words -- the folder still opens."""
    from molbuilder.kind_table import kind as _kind
    cites = _kind(task.calculation).description.cites if task else None
    if not cites:
        return None
    from molbuilder.projects import find_projects_root
    from molbuilder.frame_average import answer_said
    from molbuilder.transport.compose import (CitedPair, ComposeError,
                                              resolve_citation)
    cited_at = ((getattr(task, "slots", None) or {}).get(cites))
    if not cited_at:
        return None
    # THE CALCULATION'S OWN TREE, as prep resolves the same citation
    # (`projects.find_projects_root`), never the server's.
    tree = find_projects_root(pathlib.Path(folder))
    if tree is None:
        return {"n_frames": None, "definition": None,
                "why": f"{folder} is in no projects tree, so its citation "
                       f"{cited_at} names nothing"}
    try:
        _, cited = resolve_citation(str(cited_at), tree)
    except ComposeError as exc:
        return {"n_frames": None, "definition": None, "why": str(exc)}
    return answer_said(
        cited.structure if isinstance(cited, CitedPair) else None)


def _folder_set_to(folder):
    """The machine the calculation is set to, or ``None`` before its first
    prep -- the name its copy of the record carries, in the tab's words:
    this machine is `(this machine)` there, as every answer here says it."""
    from molbuilder.scheduler.record import LOCAL_TARGET, calculation_machine
    name = calculation_machine(folder)
    return "(this machine)" if name == LOCAL_TARGET else name


def _folder_prepared(calc) -> dict:
    """``{"bench": {stage: why}}``: the stages prepared as a benchmark, and
    the prep entry's own sentence for each -- what a prep of it would
    answer, the way back in it (`job-system.md` § 5.0).  Fail-soft, like
    the folder's other parts."""
    from molbuilder.jobset.prep import prepared_already
    try:
        said = {}
        for st in calc.task.stages:
            why = prepared_already(calc, st.name)
            if why:
                said[st.name] = why
        return {"bench": said}
    except Exception as exc:                      # noqa: BLE001
        # THE SAME SHAPE, AND THE ERROR IN IT (`web-api.md`: one part
        # failing carries its own `error`).
        return {"error": str(exc), "bench": {}}


@bp.route("/api/task-setup/commands", methods=["POST"])
def api_task_setup_commands():
    """**What a person types for a group, or a stage** -- a group's prep, as
    chosen (the machine), and its
    launch (what a group of one continues from); a benchmark's lines; a
    member's launch again alone (D4) -- composed by the terminal's own
    composer (`jobset/commands.stage_lines`), so a line on the page is one
    the terminal would print and `launch` would take (`job-system.md`
    § 5.3, *what molbuilder prints, you can type*; W55 B4).

    POST ``{dest, kind, groups | stages, from?, cold?, target?, prep?}``
    -> ``{ok, lines}`` -- ``groups`` one, or several picked for one prep
    (its line alone); ``stages`` one.  A group prepared as described has no
    prep line, as `jobset status` prints it -- unless ``prep`` asks for a
    Prep's lines, which prepare it again.  Writes nothing.
    """
    from molbuilder.jobset.commands import stage_lines
    from molbuilder.scheduler.record import LOCAL_TARGET
    body = request.get_json(silent=True) or {}
    dest_raw = str(body.get("dest") or "")
    kind = str(body.get("kind") or "task")
    groups, stages = body.get("groups"), body.get("stages")
    picked = (isinstance(groups, list) and len(groups) >= 1
              and all(isinstance(g, int) and not isinstance(g, bool)
                      for g in groups))
    one_stage = (isinstance(stages, list) and len(stages) == 1
                 and isinstance(stages[0], str) and stages[0])
    if (not dest_raw or kind not in ("task", "bench")
            or not (picked or one_stage)
            or (kind == "bench" and not one_stage)):
        return jsonify({"ok": False,
                        "error": "dest, kind (task|bench) and groups (group "
                                 "numbers) or stages (one name) are "
                                 "required"}), 400
    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    target = body.get("target") or None
    if target == "(this machine)":
        target = LOCAL_TARGET
    prepared = False
    if picked and kind == "task":
        from molbuilder.calculation import open as open_calculation
        from molbuilder.jobset.ready import task_state
        try:
            calc = open_calculation(dest)
            state = task_state(calc)
        except (OSError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        # PREPARED AS DESCRIBED, the ready door's answer (`GroupAnswer`):
        # its prepared jobs hold its members.
        gs = [g for g in state.groups if g.number in groups]
        # A GROUP NO VERB TAKES -- its launched members changed, or a
        # sequence still going holds it -- has its way on for lines, the
        # answer's own words, as `status` prints them.
        stuck = next((g for g in gs if g.members_changed or g.held), None)
        if stuck is not None:
            from molbuilder.jobset.commands import said_lines
            return jsonify({"ok": True, "lines": said_lines(stuck.why)})
        # A PREPARED GROUP'S LINES are its launch alone -- unless the lines
        # asked for are a Prep's (`prep`), which prepares it again.
        prepared = (len(groups) == 1 and len(gs) == 1 and gs[0].prepared
                    and not body.get("prep"))
        # ...AND A PREP NAMES EVERY GROUP IT FORMS (`GroupAnswer.prep_with`).
        if not prepared:
            groups = sorted({n for g in gs for n in g.prep_with})
    elif one_stage and kind == "bench":
        from molbuilder.calculation import open as open_calculation
        prepared = stages[0] in _folder_prepared(
            open_calculation(dest)).get(kind, {})
    return jsonify({"ok": True, "lines": stage_lines(
        kind, base=dest, groups=(groups if picked else ()),
        stages=(stages if one_stage and not picked else ()),
        from_attempt=(body.get("from") or None), cold=bool(body.get("cold")),
        target=target, prepared=prepared)})


#: Each record Save writes, by the key `read_as` names it with.
_RECORD_NAMES = (("task", _TASK_FILENAME), ("request", _JOB_REQUEST_FILENAME))


def _read_as(folder) -> dict:
    """What each record is on disk -- ``{task, request}``, each the SHA-256
    of the file's bytes, or ``None`` when it is not there: what the folder
    answer serves and Save compares (`web/task-setup.md` § 8)."""
    from ...persist import sha256_of
    out = {}
    for which, name in _RECORD_NAMES:
        path = pathlib.Path(folder) / name
        out[which] = sha256_of(path) if path.is_file() else None
    return out


def _folder_bench_refusal(task, handover):
    """The folder answer's ``bench_refusal``: the entry's reason this
    calculation takes no bench -- a description's, or a hand-over's from
    the engine and kind it carries (`web/task-setup.md` § 11.1), whose
    reader has checked the pair (:func:`_proposal_refused`) -- or ``None``
    when there is neither to ask."""
    from molbuilder.jobset.prep_inputs import bench_refusal
    if hasattr(task, "stages"):
        return bench_refusal(str(task.engine), str(task.calculation))
    if isinstance(handover, dict) and "error" not in handover:
        return bench_refusal(handover["engine"]["name"],
                             handover["calculation"])
    return None


@bp.route("/api/task-setup/machines", methods=["GET"])
def api_task_setup_machines():
    """Which machines a calculation could be prepared FOR.

    The tab does not decide this and does not have its own rule for it:
    `preparing-for-another-machine.md` § 4 says the choice is the user's
    whenever more than one machine could be meant, and this serves the same
    list the CLI refusal names.

    Each entry is a machine record written by `jobset probe --write --name
    NAME` on the machine it describes -- MEASUREMENTS (scheduler, cores,
    reachable domains), which is why they can be trusted from here at all:
    the numbers were taken there, not guessed from this server.

    ``this_machine`` is always a candidate and is listed with the others,
    because "prepare for the box I am on" is a choice like any other and
    omitting it would make the common case look unavailable.
    """
    from molbuilder.scheduler import known_machines, choice_required

    # One list, one rule -- both live in `molbuilder.scheduler` so the
    # browser and `jobset machines` cannot disagree about which machines
    # exist, which can be read, or when a choice is required.
    machines = known_machines()
    return jsonify({"ok": True, "machines": machines,
                    "choice_required": choice_required(machines)})


@bp.route("/api/task-setup/presets", methods=["GET"])
def api_task_setup_presets():
    """The shipped tier presets, for filling a stage's row.

    These are the SAME tiers the engine's shipped ladder is built from
    (`jobset.ladder.engine_tiers`, the engine record's ``presets``), so a
    stage filled here and a stage of the default ladder cannot drift --
    `engines/tuning.md` § 4 is the authority for what number each tier
    carries, and this serves it rather than restating it.  Every tier is
    offered whatever a strategy enables: the enable-mask is the ladder's
    business, not this menu's.

    **Offered per KIND, by the columns' own membership rule.** A tier is a
    set of values for named fields; it is offered for this folder's kind
    only when every one of those fields may be a column of the table
    (`_column_items`) -- `task-setup.md` § 9: *"a preset that half-applied
    would be worse than one that refused"*.  The relaxation tiers are
    columns of an optimization and of a vibration's stage (its relaxation's
    settings), and of NO transport rung (`stages` routes each rung its own items;
    `engines/transport.md` § 2a.7: a rung carries its role's profile), so a
    transport description gets an empty menu and its rows draw none.
    """
    engine = str(request.args.get("engine") or "").lower()
    kind = str(request.args.get("calculation") or "")
    if not engine or not kind:
        return _unstated(engine=engine, calculation=kind)
    try:
        _engine(engine)
    except EngineTableError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    from molbuilder.jobset.ladder import engine_tiers
    out = [{"tier": tier, "name": name, "values": values}
           for tier, name, values in engine_tiers(engine)]
    columns = {it.name for it in _column_items(engine, kind)}
    out = [ps for ps in out if set(ps["values"]) <= columns]
    return jsonify({"ok": True, "engine": engine, "calculation": kind,
                    "presets": out})
