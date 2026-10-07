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

import typing
import json
import pathlib
from datetime import datetime
from typing import Any, Dict

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
from molbuilder.config.pyscf  import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
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
    from molbuilder.electronic_state import (KINDS, electronic_state,
                                             engines_for)

    kind = body.get("kind") or ""
    if not kind:
        return _unstated(kind=kind)
    if kind not in KINDS:
        return jsonify({"ok": False,
                        "error": f"kind must be one of {', '.join(KINDS)}, "
                                 f"not {kind!r}"}), 400
    forms = body.get("forms")
    if forms is not None and not isinstance(forms, dict):
        return jsonify({"ok": False,
                        "error": "`forms` maps an engine to its form's "
                                 "items"}), 400
    runs = engines_for(kind)
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
        # WHO RUNS WHICH KIND is `electronic_state.engines_for`'s (§ 2a.3):
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
            cfg = (_siesta_config_from_params(items or {}, kind)
                   if engine == "siesta"
                   else _pyscf_config_from_params(items or {}, kind))
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
        # Receipts first (what the edit did), then conditions (what is now true).
        notices = list(receipts) + list(conditions)
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
    AND the sidecar schema -- ``write`` stamps ``schema_version`` + a real ``structure_hash``
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
    # THE SAME GATE THE EXPORT DOOR RUNS: a save and a download cannot
    # produce different bytes -- and they must not disagree about a refusal
    # either.  Runs BEFORE the overwrite gate, so nobody is asked to confirm
    # an overwrite for a save that will be refused; the verdicts ride the
    # response as `notices`, exactly as they do on export.
    from ._shared import checked_periodicity
    struct, notices = checked_periodicity(struct)
    try:
        resolved = _resolve_within_roots(path)   # save-as target need not exist yet
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    # Overwrite gate: the GEOMETRY file's existence drives the tab's overwrite dialog
    # (mirrors the /api/files/write 409 contract) -- refuse unless the caller confirmed.
    if resolved.exists() and not overwrite:
        return jsonify({"ok": False, "needsOverwrite": True,
                        "error": f"file already exists: {path}"}), 409
    frames = body.get("frames")
    if frames is not None and not isinstance(frames, list):
        return jsonify({"ok": False,
                        "error": "'frames' must be a list of coordinate lists"}), 400
    try:
        StructureCodec().write(struct, resolved, frames=frames)
    except ValueError as exc:          # a frame that does not carry these atoms
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
    sidecar's envelope -- ``schema_version``, and the ``structure_hash`` pinning
    it to its geometry -- is the codec's.

    Body: the ENVELOPE (web-api.md § 1) plus two optional keys --
    ``{"structure": {...}, "name": "<stem>", "frames": [...]}``.

    WHO NAMES WHAT.  ``name`` is a STEM and nothing else (``wire_frame40-120``,
    no extension), because only the caller knows what an export IS: which
    structure, which frames, chosen at which moment.  The SUFFIX is the server's,
    because it follows from the format and the format follows from the frame
    count, which ``pair()`` already decided -- a caller that appends its own is
    answering a question that has an answer.  A missing / empty / path-shaped ``name`` falls back to
    ``structure``; only the last path component is ever used, and nothing here
    touches the filesystem.

    Returns ``{ok, files: [{name, text}], frames, notices}`` -- each entry is a
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
    # The same gate every other structure door runs, so what leaves here is
    # judged by the same rules a save is judged by -- and a cell it refuses
    # leaves as a 400 carrying the gate's sentence, not as a 500.
    from ._shared import checked_periodicity
    struct, notices = checked_periodicity(struct)
    # THE FRAMES, when a range was asked for (molview.md § 11.3).  They ride
    # BESIDE the envelope rather than inside it -- the same shape
    # ``/api/build/load`` takes on the way in: one structure carrying the
    # identity and the metadata, plus the coordinates of the frames wanted.
    # Absent, this is the single-frame export.
    frames = body.get("frames")
    if frames is not None and not isinstance(frames, list):
        return jsonify({"ok": False,
                        "error": "'frames' must be a list of coordinate lists"}), 400
    # THE STEM, reduced to its last component.  This never reaches the
    # filesystem -- ``files()`` builds names in memory -- but it does reach the
    # browser as a download name, so a path-shaped one is flattened rather than
    # passed on.
    raw_name = str(body.get("name") or "").replace("\\", "/")
    stem = raw_name.rsplit("/", 1)[-1].strip()
    if not stem or stem in (".", ".."):
        stem = "structure"
    try:
        made = StructureCodec().files(struct, stem, frames=frames)
    except ValueError as exc:          # a frame that does not carry these atoms
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True,
                    "files": [{"name": path.name,
                               "text": blob.decode("utf-8")}
                              for path, blob in made],
                    "frames": len(frames) if frames else 1,
                    "notices": notices})


@bp.route("/api/build/load", methods=["POST"])
def api_build_load():
    """Accept either:
      * multipart/form-data with a single file field "file", or
      * JSON {"path": "<project-relative structure file>"} -- the
        FILE-ONLY load: the SERVER reads the .xyz(/.pdb) + its paired
        .molstruct.json through StructureCodec.read (the ONE authority
        owns the file access AND the pairing).  This is how a project
        file is opened -- no raw text, no browser-side sidecar path.
      * JSON {"text": "...", "format": "xyz"|"pdb"|"auto",
              "filename": "<optional>"} -- raw-geometry IMPORT (a paste /
        upload with no persisted file yet); metadata-less.
    Returns the same JSON shape as /api/build/molecule so the front
    end can treat the result identically.
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
        try:
            struct = StructureCodec().read(_resolved, retired_out=retired)
        except Exception as exc:  # noqa: BLE001 -- parse/sidecar error -> 400
            return jsonify(
                {"ok": False, "error": f"could not load {_path}: {exc}"}), 400
        # The conditions for THIS structure are produced on the way out by
        # `ok_structure_response`, which validates every structure it sends.
        # What only the read knows is a retired corner it did not apply: said
        # here, naming it, because a person who typed it assigns it again on
        # the Cell page (plan § 5q D14).
        said = []
        if retired.get("cell_origin") is not None:
            from molbuilder.issues import Issue
            from molbuilder.periodicity_gate import notices_for_report
            corner = ", ".join(f"{float(v):g}" for v in retired["cell_origin"])
            said = notices_for_report([Issue(
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
        try:
            struct = struct_from_body(_pbody)
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
            struct = Structure.from_xyz(text, title=filename or None)
        elif fmt == "pdb":
            struct = Structure.from_pdb(text, title=filename or None)
        else:
            return jsonify({"ok": False,
                            "error": f"unknown format {fmt!r}; "
                                     "expected xyz or pdb"}), 400
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse {fmt}: {exc}"}), 400

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
    })


@bp.route("/api/build/schema/<engine>", methods=["GET"])
def api_build_schema(engine: str):
    """Form-rendering schema for the SIESTA or PySCF Build panel.

    Returns the JSON-friendly shape produced by
    ``_shared.catalogue_to_form_schema()`` -- see the helper docstring
    for the exact field/section layout.  The Build tab's JS calls
    this once on page load and renders the form panel directly from
    the returned schema; no static HTML field declarations are
    duplicated.

    ``engine`` is constrained to {"siesta", "pyscf"} so a typo
    surfaces as a clean 404 instead of leaking a default response.
    """
    engine = (engine or "").strip().lower()
    cls_map = {
        "siesta": (SiestaConfig, "p"),
        "pyscf":  (PySCFConfig,  "py"),
    }
    if engine not in cls_map:
        return jsonify({
            "ok": False,
            "error": (
                f"unknown engine {engine!r}; "
                f"expected one of {sorted(cls_map)}"
            ),
        }), 404
    _cls, id_prefix = cls_map[engine]
    calculation = str(request.args.get("calculation") or "")
    if not calculation:
        return _unstated(calculation=calculation)
    # Built from the CATALOGUE (`web/form-schema.md` § 1), not from the config
    # class: a parameter is defined in molbuilder/data/catalogue.template.toml,
    # and the class is a translator on the way OUT to an engine.  The renderer
    # is unchanged -- it takes whatever schema it is handed.
    return jsonify({
        "ok": True,
        "schema": _catalogue_to_form_schema(engine, id_prefix,
                                            calculation=calculation),
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

    if engine not in ("siesta", "pyscf"):
        return jsonify({
            "ok": False,
            "error": f"engine must be 'siesta' or 'pyscf'; got {engine!r}",
        }), 400

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
        if engine == "siesta":
            cfg = _siesta_config_from_params(params, calculation)
        else:
            cfg = _pyscf_config_from_params(params, calculation)
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
            validate(struct, cfg, calculation=calculation), cfg=cfg),
    })


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


_PYSCF_HINTS  = typing.get_type_hints(PySCFConfig)


def _siesta_config_from_params(params: Dict[str, Any],
                               calculation: str) -> SiestaConfig:
    """A SiestaConfig from a form's payload for a ``calculation`` of that
    kind -- the one door in `_shared`, which the transport describe door's
    shared panel reads through too."""
    from ._shared import siesta_config_from_params
    return siesta_config_from_params(params, calculation)


def _pyscf_config_from_params(params: Dict[str, Any],
                              calculation: str) -> PySCFConfig:
    """A PySCFConfig from a form's payload for a ``calculation`` of that
    kind, through the same door (`_shared.config_from_params`): a blank
    is not chosen -- a blank solvent is the gas phase, the field's own
    default.  Dispersion's ``"none"`` is its value for no correction,
    kept as itself (the field's note)."""
    return _config_from_params(PySCFConfig, params, _PYSCF_HINTS,
                               calculation)


# --------------------------------------------------------------------- #
#  Hand-over to Task setup                                              #
# --------------------------------------------------------------------- #

#: The hand-over file's own schema.  **Not** ``molbuilder/task@1``, and the
#: difference is the point: this file is deliberately INCOMPLETE -- it carries
#: what the parameter tab knows and cannot carry ``shape``, which is required
#: with no default because inferring it "would hand somebody a directory tree
#: they never asked for" (`engines/stages.md` § 6.7).  A file claiming
#: ``molbuilder/task@1`` while failing its own reader is worse than one that
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


@bp.route("/api/task-setup/handover", methods=["POST"])
def api_task_setup_handover():
    """RENDER the parameter tab's work, for the browser to write.

    **This writes nothing.**  `web/projects.md` § 1 puts raw bytes in the
    content-blind file layer that *"every tab can use"* -- `writeFile` /
    `safeSave` / `deleteEntry` -- and a tab that opens files itself bypasses the
    roots guard, the lock, the uniform `{ok, ...}` envelope and the sidebar
    re-list that come with it.  So this returns the two TEXTS and the caller
    puts them where the user chose, through `projects.safeSave`.

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
    on state you cannot see in the folder.  Task setup finishes the job: it asks
    for the shape, takes the stages, and on a successful save writes the real
    ``task.json`` and removes this file.
    """
    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "").lower()
    # The hand-over carries the calculation KIND (handover-procedure § 6:
    # "the hand-over is a Send button on the same endpoint").  The template narrows by it, and
    # the receiving tab writes it into task.json.
    calculation = str(body.get("calculation") or "")
    if not engine or not calculation:
        return _unstated(engine=engine, calculation=calculation)
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False,
                        "error": f"unknown engine {engine!r}"}), 400
    if calculation == "transport":
        # NO HAND-OVER FOR THE COMPOSITE (user ruling 2026-08-29):
        # nothing is awaiting -- the stages, the shape and the identity
        # are all fixed by design -- so the Transport tab DESCRIBES
        # directly through its own door and this one refuses by name.
        return jsonify({"ok": False,
                        "error": "transport describes on its own tab "
                                 "(POST /api/transport/describe writes "
                                 "the whole task.json -- nothing is "
                                 "awaiting)"}), 400

    try:
        struct = _struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    params: Dict[str, Any] = body.get("params") or {}
    try:
        cfg = (_siesta_config_from_params(params, calculation)
               if engine == "siesta"
               else _pyscf_config_from_params(params, calculation))
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # THE PSEUDOPOTENTIALS, SETTLED BEFORE A SIESTA CALCULATION IS WRITTEN
    # (`handover-procedure.md` § 2.2, plan § 5w K20): the check `prep` makes,
    # asked of the folder the files go into -- pseudopotentials already
    # beside it count -- and a refusal until every element is covered, its
    # findings for the tab to put beside the field.  The sender names the
    # folder (`dest`); a caller that names none is asked about the library
    # alone, as the live check is.
    if engine == "siesta":
        dest_dir = None
        if body.get("dest"):
            try:
                dest_dir = _resolve_within_roots(str(body["dest"]))
            except _PickerError as exc:
                return jsonify({"ok": False, "error": exc.message}), exc.status
        from molbuilder.validation.siesta import pseudopotential_findings
        psml = pseudopotential_findings(struct, cfg, calculation=calculation,
                                        dest_dir=dest_dir)
        if any(i.severity == "error" for i in psml):
            return jsonify({
                "ok": False,
                "error": "the pseudopotentials are not settled: SIESTA needs "
                         "a .psml file for every element, so this would "
                         "write a calculation that cannot start -- the "
                         "findings are beside the pseudopotential field "
                         "(handover-procedure.md 2.2)",
                "findings": _issues_to_json(psml, cfg=cfg),
            }), 400

    from molbuilder.identity import normalise_id, run_id
    from molbuilder.template import (template_filename as _template_filename,
                                     template_with_values)

    # WHAT THIS CALCULATION IS CALLED — the identity the person typed, and the
    # destination folder only when they did not.
    #
    # `run-identity.md` § 4: *"The label is the SystemLabel / JOB literal.
    # There is no second name."*
    #
    # Which field carries the identity is the ENGINE's to say, and it says so
    # (`RestartGroup.field`) — no `if engine ==` here.  The folder name is
    # still the answer when the field is untouched, because the schema default
    # is a placeholder (`siesta`, `pyscf_relax`) and naming a calculation after
    # a placeholder is worse than naming it after the folder somebody chose.
    from molbuilder.config.pyscf import PYSCF_RESTART_GROUP
    from molbuilder.config.siesta import SIESTA_RESTART_GROUP
    _group = SIESTA_RESTART_GROUP if engine == "siesta" else PYSCF_RESTART_GROUP
    _identity = str(getattr(cfg, _group.field, "") or "")
    _placeholder = str(
        type(cfg).__dataclass_fields__[_group.field].default or "")
    typed = (_identity if _identity and _identity != _placeholder
             else (str(body.get("name") or "") or "calculation"))
    label = normalise_id(typed)
    formula = str(getattr(struct, "formula", "") or "")

    # AND THE TEMPLATE CARRIES THE SAME ONE.  Choosing the label above is only
    # half of "there is no second name": the template's identity field is what
    # the ENGINE writes its files under, so if it kept the placeholder while
    # `task.json` took the folder's name, the split would simply reappear from
    # the other side.  Normalisation happens once and the result is stored
    # (§ 3 rule 1) — this is the storing.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, **{_group.field: label})

    # WHAT THE PERSON'S FORM SENT is theirs (`template.md` § 6.6
    # obligation 2): the form sends the fields that hold a value
    # (`form-schema.md` § 1.1), and the calculation's own name.  Every other
    # item is nobody's choice.
    sources = {k: "person" for k, v in params.items()
               if v is not None and v != ""}
    sources[_group.field] = "person"
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
    from ._shared import checked_periodicity
    struct, struct_notices = checked_periodicity(struct)
    try:
        # ``<label>.source.xyz`` -- the dotted segment is the reservation
        # (`job-contracts.md` § 6.3): every identity is validated
        # dot-free, so no engine output, which stems its files on an
        # identity, can ever take this name -- else a flat SIESTA run
        # whose label matched the structure's stem would overwrite its own
        # input via WriteCoorXmol.  The one call both
        # writers make -- `jobset init` too (`StructureCodec.source_files`).
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

    handover = {
        "schema":    TASK_HANDOVER_SCHEMA,
        # JSON has no comments, so the file carries a line that says what it
        # is.  It IS read by a person -- Task setup shows it in the editor --
        # and a file whose whole job is to be handed between two surfaces
        # should not need a document open beside it to be understood.
        # The SENDER is the calculation kind's tab (E-B9): both tabs post
        # through the same shared door (lib/task-handover.js).
        "_what":     f"A hand-over from the "
                     f"{'Spectrum' if calculation == 'vibration' else 'Structure-optimization'} "
                     f"tab, not a "
                     "description. It carries the parameters (in the .template.toml "
                     "beside it) plus what this calculation is OF. It is missing "
                     "`shape` and `stages` on purpose -- Task setup asks for those, "
                     "and on a successful save writes the real task.json and deletes "
                     "this file. Nothing runs from it. The structure it is OF is the "
                     "file named under `structure.files` in this same folder, written "
                     "by the server's one codec: `structure.source` names the .xyz, "
                     "which carries the coordinates and the cell, and the "
                     ".molstruct.json beside it carries the region labels and frozen "
                     "atoms.",
        "engine":    {"name": engine},
        "run":       {"name": typed,
                      "id": run_id(typed, formula),
                      "created": datetime.now().astimezone().isoformat(timespec="seconds")},
        # WHAT THIS IS OF -- by NAME, pointing at files in this same folder.
        # These names come from the structure that was sent, so they cannot
        # disagree with it (`molview.md` § 9.3a).
        "structure": {"source":  geometry_name,
                      "formula": formula,
                      "atoms":   len(getattr(struct, "elements", []) or [])},
        # No `shape`, no `stages` -- Task setup asks.  Stated rather than
        # omitted so a reader of the file knows it is waiting on them.
        "awaiting":  ["shape", "stages"],
    }
    # THE KIND RIDES THE HAND-OVER, every kind -- the receiving tab writes
    # it into task.json and proposes the kind's own ladder (for a vibration,
    # `relax` then `freq` on SIESTA unless the box says relaxed, `freq` alone
    # on PySCF) instead of the tier default.
    handover["calculation"] = calculation

    return jsonify({
        "ok":            True,
        "label":         label,
        # THE door (`template.template_filename`), not a literal suffix.
        "template_name": _template_filename(label),
        "template_text": template_text,
        "handover_name": TASK_HANDOVER_NAME,
        "handover_text": json.dumps(handover, indent=2) + "\n",
        # Each entry is a file as it would exist on disk, under the name it
        # would exist as -- nothing is left for the browser to work out.
        "structure_files": structure_files,
        "notices": struct_notices,
    })


@bp.route("/api/task-setup/prep", methods=["POST"])
def api_task_setup_prep():
    """Run `prep` for a stage -- the SAME function the terminal runs.

    **Why a browser may trigger this at all**, when
    `project-layout.md` § 2.2 says the deck cannot be finished in the
    browser: that section's argument is about WHOSE FACTS the deck is
    rendered from, not about which surface presses the button.  `prep`
    needs three inputs: two portable (the template, the description) and
    one the target machine's (`molbuilder.json`).  A named target's record
    supplies the machine half -- that is what `environments/<name>.json`
    IS -- so prepping FOR Sol FROM here is the case
    `preparing-for-another-machine.md` exists for, and prepping for THIS
    machine is the ordinary one.

    What this door does NOT do is submit.  `prep` writes files into the
    calculation and can be run again; `launch` spends a queue slot and
    refuses batch submission by design (one job per invocation, by hand).
    That line is the user's (2026-08-24) and it is where it is because
    the two verbs differ in what they cost to get wrong.

    ``plan: true`` is the entry's PREVIEW (`job-system.md` § 5.0): the plan,
    stopped before the save -- what it would write, the launch the header
    and the run script would carry, what the stage builds on -- or the
    refusal prep would give, with nothing saved, written or recorded.
    Nothing is prepped unseen -- the rule the launch door keeps
    (`submission.md` S4) -- and a Prep names the preview's plan
    (``plan_id``) and is refused when the plan it makes now differs.

    **Everything else is the entry's** (W55 B3, D15): the description, the
    stage -- a name in any case, or ``#N`` -- the machine, what the stage
    continues from, every refusal in its words.
    """
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "the body is a JSON object"}), 400
    # WORDS, as the page sends them: a field of another type is refused in
    # words, never a 500.
    for key in ("dest", "kind", "stage", "target", "from", "plan_id"):
        if body.get(key) is not None and not isinstance(body.get(key), str):
            return jsonify({"ok": False,
                            "error": f"`{key}` is a string"}), 400
    dest_raw = body.get("dest")
    kind = str(body.get("kind") or "").strip()
    stage = (body.get("stage") or "").strip() or None
    target = (body.get("target") or "").strip() or None
    # WHAT IT CONTINUES FROM is the entry's (plan W37, `job-system.md`
    # § 5.4): the page's **Continue from** choice is the CLI's two flags --
    # `from`, a run of this calculation named by its folder
    # (`01_coarse/run-0`), and `cold`.
    from_raw = body.get("from")
    from_attempt = (str(from_raw).strip() or None) if from_raw else None
    cold = bool(body.get("cold"))
    preview = bool(body.get("plan"))
    plan_id = (str(body.get("plan_id") or "").strip() or None)

    if kind not in ("run", "bench"):
        return jsonify({"ok": False,
                        "error": "kind must be 'run' or 'bench'"}), 400
    # NOTHING IS PREPPED UNSEEN (`job-system.md` § 5.0): a Prep names the
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
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.prep import prep_stage
    try:
        ans = prep_stage(dest, kind, stage, target=target,
                         allocation=Resources(),
                         from_attempt=from_attempt, cold=cold,
                         preview=preview, plan_id=plan_id)
    except PrepError as exc:
        # Refused, not repaired -- the reader's own words, as the terminal
        # gives them -- WITH what the entry had found by then: the preflight's
        # notes and what the inputs said (a bench's crossed-out cells, which
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
    said = ans.as_dict(dest)
    return jsonify({
        "ok": True, **said,
        "machine": ("(this machine)" if said.get("machine") == LOCAL_TARGET
                    else said.get("machine")),
    })


@bp.route("/api/task-setup/save", methods=["POST"])
def api_task_setup_save():
    """Write the description a person has been reading, and resolve a hand-over.

    **The BUFFER is the source.**  The editor is where a description is checked
    and corrected before it is written (`web/task-setup.md` § 9a), so this takes
    the text as edited -- never a re-serialisation of a parsed model, which
    would silently discard whatever was typed in the editor.

    **Refused rather than repaired.**  The text goes through the shipped reader
    (`task.read_task`), so a description that does not parse, names a field the
    schema does not know, or carries no stage is refused with the reason -- the
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
    if not dest_raw:
        return jsonify({"ok": False, "error": "no destination folder given"}), 400
    if not isinstance(text, str) or not text.strip():
        return jsonify({"ok": False, "error": "nothing to save"}), 400

    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    if not dest.is_dir():
        return jsonify({"ok": False, "error": f"not a directory: {dest_raw}"}), 400

    import tempfile
    # FILENAME comes from `task.py` too: the description's NAME is that
    # module's to spell, like its bytes (`task-description.md` § 6.4 --
    # one reader, so the two surfaces cannot produce different files).
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import read_task, write_task

    # Validate by READING it, in a scratch file, so nothing lands in the
    # calculation folder unless it is a description the rest of the system
    # can open.  `read_task` is the same door `prep` uses.
    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / TASK_FILENAME
        probe.write_text(text, encoding="utf-8")
        try:
            task = read_task(probe)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    # GATE ③ FIRES HERE TOO (G-1b): a description naming an unknown field,
    # a value outside its bounds, or a bench point that fits no item is
    # refused here rather than at prep, on the cluster.  Same function the CLI runs (`validation.task.preflight`),
    # so the two surfaces cannot disagree; the template beside the
    # description adds the sequence findings when it is already there.
    from molbuilder.template import find_template as _find_template
    from molbuilder.validation.task import (preflight as _task_preflight,
                                            config_class_for as _cfg_cls_for)
    try:
        _tpl_file = _find_template(dest, task.label)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    _pf = _task_preflight(
        task,
        template_text=(_tpl_file.read_text(encoding="utf-8")
                       if _tpl_file is not None else None))
    # THE CONFIG CLASS RIDES WITH THE FINDINGS.  Without it `_issues_to_json`
    # omits `workflow_group` and the page has no card to put a finding on
    # (`web/ui-contract.md` Rule 2).
    _pf_cfg = _cfg_cls_for(task)
    _pf_errs = [i for i in _pf if i.severity == "error"]
    if _pf_errs:
        return jsonify({
            "ok": False,
            "error": "the description fails its own preflight "
                     "(engines/stages.md § 6.6):\n  - "
                     + "\n  - ".join(i.message for i in _pf_errs),
            "findings": _issues_to_json(_pf, cfg=_pf_cfg),
        }), 400

    # ONE JOB PER FOLDER (`job-contracts.md` § 2.1 Rule 1).  A folder already
    # describing a DIFFERENT calculation is not a folder this save may land in:
    # the ids say they are different calculations, and overwriting one with the
    # other orphans every warm file and output already keyed to it.
    #
    # Compared by RUN ID, not by path: the id is `<label>_<formula>` and is the
    # one thing that says which calculation a folder is
    # (`run-identity.md` § 2.0a).  Re-saving the SAME calculation is the
    # ordinary case and must stay free.
    existing = dest / TASK_FILENAME
    if existing.is_file():
        try:
            prior = read_task(existing)
        except Exception:
            prior = None                      # unreadable: let the write fix it
        if prior is not None and prior.run.id != task.run.id:
            return jsonify({
                "ok": False,
                "error": f"this folder already describes a different "
                         f"calculation ({prior.run.id!r}); saving {task.run.id!r} "
                         f"here would orphan its results. One job per folder — "
                         f"pick or make another.",
            }), 409
        # THE SHAPE IS FIXED ONCE THE CALCULATION HAS PRODUCED
        # (`web/task-setup.md` § 4): a prepped stage -- a run or a benchmark,
        # the prep entry's own answer (`prep.prepped_stages`) -- lies where
        # the shape put it, and another shape would orphan every deck, output
        # and warm file there.  A redo is a rollback (`job-system.md` § 5.0).
        if prior is not None and prior.shape and task.shape != prior.shape:
            from molbuilder.jobset.commands import rollback
            from molbuilder.jobset.prep import prepped_stages
            done = prepped_stages(dest, prior)
            if done:
                return jsonify({
                    "ok": False,
                    "error": f"this calculation is {prior.shape}, and its "
                             f"shape is fixed once it has produced "
                             f"(task-setup.md § 4): {', '.join(done)} "
                             f"{'is' if len(done) == 1 else 'are'} prepped, "
                             f"laid out {prior.shape}, and a {task.shape} "
                             f"calculation would orphan every deck, output "
                             f"and warm file there.  To make it "
                             f"{task.shape}: " + rollback("its first prep",
                                                          base=dest),
                }), 409

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
        write_task(dest / TASK_FILENAME, task)      # atomic (persist.write_json)
    except OSError as exc:
        return jsonify({"ok": False, "error": f"could not write: {exc}"}), 500

    # The hand-over's REMOVAL is the browser's, through
    # `projects.deleteEntry` -- moving bytes is the content-blind layer's job
    # (`projects.md` § 1), and unlinking here would bypass its guard and the
    # sidebar re-list.  Reported so the caller knows whether to.
    return jsonify({
        "ok":            True,
        "saved":         kept.said(),
        "wrote":         TASK_FILENAME,
        "handover_name": TASK_HANDOVER_NAME,
        "handover_here": (dest / TASK_HANDOVER_NAME).is_file(),
        "stages":        [st.name for st in task.stages],
        # Gate ③'s non-refusing findings (sequence warnings and the
        # fingerprint note): the save proceeded, and the reader deserves
        # what the CLI would have echoed.
        "findings":      _issues_to_json(_pf, cfg=_pf_cfg),
    })


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

    Each item says whether the machine answers it: an ``allocation`` resolver
    means a description may never carry a value for it (`template.md` § 6.4),
    so the picker can show it as measurable-only rather than as a choice.
    """
    engine = str(request.args.get("engine") or "").lower()
    if not engine:
        return _unstated(engine=engine)
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400

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
        if "execution" not in (it.category or ()):
            continue
        out.append({
            "name":            it.name,
            "label":           it.label or it.name,
            "help":            it.help or "",
            "machine_answers": it.allocation,
            # THE VALUE SHAPE (user, 2026-08-20): the machine card births
            # a row at its value in force and offers an enum's choices or
            # a bool's two values from a dropdown -- it can only do either
            # by asking the catalogue.
            "type":            it.type or "",
            "choices":         list(it.choices) if it.choices else None,
            # THE KIND'S OWN DEFAULT (`template.recommended_for`), never the
            # general one -- what the hover calls *Recommended*.
            "default":         (_T.with_recommended(it, kind).default
                                if kind else it.default),
            # The unit rides too: the machine card's default/help text
            # reads m.unit, and max_memory_mb rendered unitless without
            # it (allocation items never pass through the columns door,
            # so this payload is their only source).
            "unit":            it.unit or None,
        })
    return jsonify({"ok": True, "engine": engine, "items": out})


def _column_items(engine: str, kind: str):
    """The catalogue items that may be a COLUMN of the stage table for this
    (engine, kind) -- **the one membership rule**, read by the columns route
    and by the presets route (`web/task-setup.md` § 5, § 9), so a preset can
    never offer to fill a column the table would refuse to add.

    `engines/stages.md` § 6.2: *"Any setting the description is allowed to
    hold may become a column. The ones it is not allowed to hold may not."*
    """
    from molbuilder import template as _T
    run_settings = _T.run_settings(engine)
    for it in _T.select(_T.catalogue(), engine=engine):
        # THE membership rule, asked of the item rather than restated here.
        # A RUN SETTING is the rung's run card's, never a column (plan § 5w
        # K5; `engines/stages.md` § 6.2): the machine's answers, and the ones
        # a person gives -- `use_gpu`, `restart`, the solver.
        if it.name in run_settings:
            continue
        # A column belongs to this folder's KIND (template.md § 6.3's
        # sibling rule); the tab passes its description's kind (P2).
        if it.calculations and kind not in it.calculations:
            continue
        # ...and a ROLE item is not a column at all for this kind: the rung
        # decides it, so a cell offering it would present a choice with one
        # correct answer, and a person who changed it would not be tuning
        # the run but stopping it being the run it is (template.md § 6.4's
        # third answerer, 2026-09-16).
        if kind in it.role:
            continue
        # ...nor is a value that binds EVERY rung of this kind (template.md
        # § 6.4's `shared`: "the value binds every rung and no stage
        # overrides it").  It is edited in the template, on the tab that
        # owns it; a column here would be a per-rung override `prep`
        # refuses by name.
        if kind in it.shared:
            continue
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
    if engine not in ("siesta", "pyscf"):
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400

    from molbuilder import template as _T
    out = []
    for it in _column_items(engine, _calc_kind):
        out.append({
            "name":    it.name,
            "label":   it.label or it.name,
            "help":    it.help or "",
            "unit":    it.unit or "",
            # THE KIND'S OWN DEFAULT (`template.recommended_for`).
            "default": _T.with_recommended(it, _calc_kind).default,
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


def _folder_template(folder, label) -> dict:
    """What the folder's template answers -- the payload, not the response.

    ``label`` is the
    calculation's -- its description's, else its hand-over's -- and ``None``
    for a folder that is neither, which has no template to show.
    """
    # THE door, not a glob (`template.find_template`): this tab and `prep`
    # read the same file.
    from molbuilder.template import (SOURCE_WORDS, find_template,
                                     read_template, select)
    if label is None:
        return {"ok": True, "name": None, "values": {}}
    try:
        found = find_template(folder, label)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if found is None:
        return {"ok": True, "name": None, "values": {}}

    try:
        tmpl = read_template(found.read_text())
    except Exception as exc:
        # A template that does not parse is the user's to fix, and saying which
        # file beats an empty table that looks like "nothing was sent".
        return {"ok": False, "name": found.name,
                "error": f"{found.name}: {exc}"}

    # Through `select` -- `engines/template.md` § 8.0 owns the rule.
    values = {it.name: it.value for it in select(tmpl) if it.is_set}
    # ...AND WHOSE EACH ONE IS, in the words every surface says it
    # (`template.SOURCE_WORDS`, `engines/template.md` § 6.6 obligation 2):
    # the hover's *"450.0 Ry -- you set this"*.  A file written before the
    # key existed says *not recorded*, never a guess.
    said = {it.name: SOURCE_WORDS[it.source or "unrecorded"]
            for it in select(tmpl) if it.is_set}
    return {"ok": True, "name": found.name, "values": values, "said": said}


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


@bp.route("/api/task-setup/bench-grid", methods=["POST"])
def api_task_setup_bench_grid():
    """The bench grid this description would produce, cell by cell, with
    the queues on the target that would take each one.

    **The same list the terminal prints, as data** -- served rather than
    recomputed, because a browser enumerating the grid a second way would
    be exactly the drifting second decider `generator.md` § 4.3a's rebuild
    removed.  `bench_inputs` is the one enumerator; this hands it the
    axes and collects its report.

    POST ``{dest, target?, bench}``.  ``bench`` is the axis map AS IT IS
    BEING EDITED -- the card's model, not what is saved -- so the list
    tracks the person's typing instead of the last write.

    Answers 200 with ``cells`` even when none survive: *nothing here fits*
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
    rows: list = []
    try:
        # THE SAME REPORT the prep's notes print, as data: the card refreshes
        # it on every keystroke, so it asks for the rows and not the notes.
        # One function, two renderings.
        bench_inputs(dest, target, bench_override=bench, report=rows)
    except Exception as exc:                      # noqa: BLE001
        # A grid where nothing survives raises, and its report is still the
        # answer worth showing -- the crossed-out rows say why.  The COUNT
        # is computed, never assumed: `bench_inputs` fills the report
        # before its last few refusals, so a raise can follow cells that
        # did survive, and writing 0 here would report them as struck.
        if rows:
            return jsonify({"ok": True, "cells": rows,
                            "kept": sum(1 for r in rows if not r["why"]),
                            "note": str(exc)})
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "cells": rows,
                    "kept": sum(1 for r in rows if not r["why"])})


@bp.route("/api/task-setup/prep-plan", methods=["POST"])
def api_task_setup_prep_plan():
    """What a `prep` would write, stage by stage — `task-setup.md` § 7.1.

    **The names come from the producer, never from the page.**  Flat and
    hierarchical name directories differently (§ 4), so a list composed in
    the browser would be a second answer free to disagree with the thing it
    describes.  `materialize.stage_home` is the one namer (decision 27) and
    `Shape.stage_dir` the one layout, and this asks both.

    The allocation each stage would carry is `task.allocation`,
    which is the same function the prep itself resolves through -- so the
    row cannot promise a wall the run will not ask for.

    POST ``{task, dest?}`` -- the description AS IT IS BEING EDITED, not
    what is saved, for the reason the grid door gives: the card's edits live
    in the browser's model until the person saves -- and the folder, whose
    files hold the stages' numbers (a stage that has files keeps its own,
    W38 F4).  Nothing here writes.
    """
    body = request.get_json(silent=True) or {}
    raw = body.get("task")
    if not isinstance(raw, dict):
        return jsonify({"ok": False,
                        "error": "task: must be the description object"}), 400
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.paths import Shape
    from molbuilder.runfiles import manifest
    from molbuilder.task import Task
    try:
        task = Task.from_dict(raw)
        shape = Shape.named(task.shape)
    except Exception as exc:                      # noqa: BLE001
        # A description mid-edit is often unreadable, and that is ordinary.
        # Its own words, so the card can say why rather than going blank.
        return jsonify({"ok": False, "error": str(exc)}), 400

    # ONE allocation for the calculation (`stages.md` § 6.8a), and the launch
    # shape the description CHOSE -- every machine-answered `bench` entry that
    # carries one point (`generator.md` § 4.3a).  Two reads because they are
    # two questions; neither is recomputed here.
    alloc = {"domain": task.allocation.domain, "time": task.allocation.time,
             "mem": task.allocation.mem}
    # The plan card reads the DESCRIPTION as posted, mid-edit, with no
    # machine resolved -- so it reports the condition as WRITTEN, without
    # asking the enumerator, which needs a target.  THE FOLDER, when the page
    # names it (`dest`), is where the stages' numbers are read: a stage that
    # has files keeps its number (`materialize.stage_home`, W38 F4), so the
    # plan names the folders prep will write, not the description's places.
    folder = None
    if body.get("dest"):
        try:
            folder = _resolve_within_roots(str(body["dest"]))
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status

    rows = []
    for st in task.stages:
        if st.enabled is False:
            continue
        token = stage_home(folder, task, st.name).token
        rows.append({"stage": st.name, "token": token,
                     "dir": shape.stage_dir(token),
                     "allocation": alloc,
                     # PER STAGE, because the condition is (§ 6.8d): the
                     # calculation's block with this rung's laid over it.
                     "chosen": task.run_condition(st.name),
                     # AND WHAT THIS RUNG WILL BE NAMED (user, 2026-09-07).
                     # Every name comes out of the grammar (`runfiles`
                     # § 2.2a) with THIS rung's token, so the card cannot
                     # show a spelling the writers do not use -- which is
                     # the failure that module exists for.  The engine
                     # filters it: a SIESTA run is not told about `.py`.
                     # ...spelled as THIS shape spells them: an attempt's
                     # `run.json`, a flat stage's first run's
                     # `<base>-run0.run.json` (plan D22, W57 decision 2).
                     "files": manifest(task.label, token,
                                       shape=shape.name, engine=task.engine,
                                       calculation=task.calculation)})
    bench = None
    if task.bench:
        # Every axis, with its points.  A row of length one is a DECISION and
        # measures one cell; the card says which is which (§ 6.2b).
        #
        # AND WHERE THE SWEEP LANDS, asked of `bench_container`, which is the
        # ONE spelling of that rule: the container is `bench_<NN>_<stage>`
        # flat, `<NN>_<stage>/bench` hierarchical.
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
        # calculation with no stages and `Task.from_dict` refuses one with
        # every stage disabled -- *"an all-disabled ladder is an empty one
        # spelled longer"* -- so there is no empty case to invent a
        # location for.
        from molbuilder.paths import bench_container
        bench = {"axes": {k: list(v) for k, v in task.bench.items()},
                 "allocation": alloc,
                 "rungs": [{"stage": r["stage"],
                            "dir": bench_container(shape, r["token"])}
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
    # And what `summarize run` writes there from results that exist -- the
    # transport record, a sweep's comparison -- whose own line says so.
    once = manifest(task.label, None, shape=shape.name, engine=task.engine,
                    when=("prep", "run", "summarize"),
                    calculation=task.calculation)
    return jsonify({"ok": True, "shape": shape.name, "stages": rows,
                    "bench": bench, "bundle": bundle, "once": once,
                    "warm": _warm_in_effect(task, folder)})


def _warm_in_effect(task, folder) -> "dict | None":
    """WHICH RESTART-FILE LIST THIS CALCULATION FOLLOWS -- the one door's
    answer (`warmfiles.warm_list`, `job-contracts.md` § 4.2a): its own copy
    beside ``task.json``, or molbuilder's for its engine -- and where a
    custom copy goes, which the card says (`web/task-setup.md` § 7.2; plan
    W36 ⑧: "provide information on the task setup web ui").  ``None`` for an
    engine with no list."""
    from molbuilder.warmfiles import FILENAME, WarmFilesError, warm_list
    try:
        in_effect = warm_list(str(task.engine), None, folder)
    except WarmFilesError:
        return None
    return {"path": in_effect.path, "own": in_effect.own,
            "copy_to": (str(pathlib.Path(folder) / FILENAME)
                        if folder is not None else None)}


def _folder_continue_from(dest) -> dict:
    """``{stage: choices}`` for every stage that continues from another by
    default -- what Task setup's **Continue from** offers
    (`continuation.continue_from_choices`); a stage that continues from
    nothing is absent.  Fail-soft per stage, like the folder's other parts."""
    from molbuilder.jobset.continuation import continue_from_choices
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import read_task
    try:
        task = read_task(dest / TASK_FILENAME)
    except Exception as exc:                      # noqa: BLE001
        return {"error": str(exc)}
    out = {}
    for st in task.stages:
        try:
            got = continue_from_choices(dest, task, st.name)
        except Exception as exc:                  # noqa: BLE001
            got = {"error": str(exc)}
        if got is not None:
            out[st.name] = got
    return out


def _folder_attempts(dest) -> dict:
    """How many attempts each stage has on disk -- the payload."""
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.paths import Shape, attempts_in
    from molbuilder.runfiles import find
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import read_task
    desc = dest / TASK_FILENAME
    if not desc.is_file():
        return {"ok": False, "error": f"no {TASK_FILENAME} here"}
    try:
        task = read_task(desc)
        shape = Shape.named(task.shape)
    except Exception as exc:                      # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    stages = {}
    for st in task.stages:
        token = stage_home(dest, task, st.name).token
        sd = shape.stage_dir(token)
        where = dest if sd == "." else dest / sd
        if shape.keeps_attempts_as_directories:
            # Across WHERE THIS RUNG'S ATTEMPTS ARE -- a bias scan's point
            # folders too, which the stage folder alone never showed (the
            # one door, plan § 5w K10; the M11 review's T-F13).
            from molbuilder.transport.stages import rung_containers
            n = sum(len(attempts_in(d))
                    for d, _v in rung_containers(dest, task, st.name))
        else:
            # FLAT TELLS ATTEMPTS APART BY THE FILENAME'S COUNTER, not by a
            # directory (§ 1.5a), so the count is how many run indices this
            # rung's files carry -- read through `runfiles`, which owns that
            # counter, rather than by matching `-run` in a listing.
            runs = {rf.run for _p, rf in find(where, task.label, stage=token)
                    if rf.run is not None}
            n = len(runs)
        stages[st.name] = {"token": token, "dir": sd, "attempts": n}
    return {"ok": True, "shape": task.shape, "stages": stages}


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

    def _read(name):
        f = folder / name
        if not f.is_file():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:                  # noqa: BLE001
            return {"error": f"{name}: {exc}"}

    def _described():
        """The description, through its one reader (`read_task`,
        `execution/architecture.md` § 3.2) -- the `Task` a molbuilder-written
        `task.json` holds, or the words its reader refused one with."""
        from molbuilder.task import read_task
        f = folder / TASK_FILENAME
        if not f.is_file():
            return None
        try:
            return read_task(f)
        except Exception as exc:                  # noqa: BLE001
            return {"error": f"{TASK_FILENAME}: {exc}"}

    # `task.json` and the hand-over are mutually exclusive by design: a save
    # writes the first and deletes the second (`task-setup.md` § 3), so the
    # page's MODE follows from which is here rather than from a flag it has
    # to keep.
    _task = _described()
    described = (_task.to_dict() if hasattr(_task, "to_dict") else _task)
    handover = None if described is not None else _read(TASK_HANDOVER_NAME)
    # THE LABEL THE TEMPLATE IS NAMED ON: the description's, else the
    # hand-over's name through the normaliser the hand-over named it with.
    if hasattr(_task, "label"):
        _label = _task.label
    elif isinstance(handover, dict) and (handover.get("run") or {}).get("name"):
        from molbuilder.identity import normalise_id
        _label = normalise_id(handover["run"]["name"])
    else:
        _label = None

    return jsonify({
        "ok": True,
        # THE SUBJECT.  Every folder-scoped answer says which folder it is
        # about; a consumer on another one discards it rather than painting.
        "dir": str(folder),
        "mode": ("description" if described is not None
                 else "handover" if handover is not None else "empty"),
        "description": described,
        "handover": handover,
        # WHICH FILES ARE HERE -- names only.  The "what gets written" card
        # asks *is the file this description names actually present*, which
        # is a fact about the folder and belongs in the folder's answer.
        "files": sorted(e.name for e in folder.iterdir() if e.is_file()),
        "template": _folder_template(folder, _label),
        "provenance": _folder_provenance(folder),
        # Only a described folder has stages, so only then is there anything
        # to count -- and `_folder_attempts` says so itself rather than this
        # door deciding for it.
        "attempts": (_folder_attempts(folder) if described is not None
                     else {"ok": True, "shape": None, "stages": {}}),
        # WHAT EACH STAGE CAN CONTINUE FROM -- its default, the runs of the
        # stage before it with what each was, and `--cold` where it is a
        # choice (`job-system.md` § 5.4, plan W37): the Continue-from choice
        # reads the folder's answer, so it shows what prep would take.
        "continue_from": (_folder_continue_from(folder)
                          if described is not None else {}),
        # WHY THIS DESCRIPTION HAS NO BENCH, or null -- the prep entry's own
        # answer (`prep_inputs.bench_refusal`), so the page offers the
        # Measure step exactly where `prep bench` would take it.
        "bench_refusal": _folder_bench_refusal(folder, described),
        # THE MACHINE IT IS SET TO -- its first prep's, read from its copy
        # of the record, or null before that (`configuration.md` M-3): the
        # page shows it fixed rather than offering a choice that can only be
        # refused (W55 B4, D12).
        "set_to": _folder_set_to(folder),
        # WHICH STAGES ARE PREPPED, as a run and as a benchmark -- each with
        # the prep entry's own sentence (`prep.prepped_already`), which the
        # page shows in place of a Prep it would refuse.
        "prepped": (_folder_prepped(folder) if described is not None
                    else {"run": {}, "bench": {}, "placed": {}}),
    })


def _folder_set_to(folder):
    """The machine the calculation is set to, or ``None`` before its first
    prep -- the name its copy of the record carries, in the tab's words:
    this machine is `(this machine)` there, as every answer here says it."""
    from molbuilder.scheduler.record import LOCAL_TARGET, calculation_machine
    name = calculation_machine(folder)
    return "(this machine)" if name == LOCAL_TARGET else name


def _folder_prepped(folder) -> dict:
    """``{"run": {stage: why}, "bench": {stage: why}}``: the stages prepped
    as each, and the prep entry's own sentence for each -- what a prep of it
    would answer, the way back in it (`job-system.md` § 5.0).  Fail-soft,
    like the folder's other parts."""
    from molbuilder.jobset.prep import prepped_already
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import read_task
    try:
        task = read_task(folder / TASK_FILENAME)
        out = {"run": {}, "bench": {}}
        for kind, said in out.items():
            for st in task.stages:
                why = prepped_already(folder, task, kind, st.name)
                if why:
                    said[st.name] = why
        # WHERE EACH PREPPED RUN WAS ADMITTED, as its job records it -- the
        # line both prep doors print (`job-system.md` § 6.0).
        from molbuilder.jobset.model import FILENAME as JOBSET_FILENAME
        from molbuilder.jobset.model import JobSet
        from molbuilder.jobset.placement import placement_line
        out["placed"] = {}
        if (folder / JOBSET_FILENAME).is_file():
            for job in JobSet.load(folder / JOBSET_FILENAME).jobs:
                if job.placement and job.name in out["run"]:
                    out["placed"][job.name] = placement_line(job.placement)
        return out
    except Exception as exc:                      # noqa: BLE001
        # THE SAME SHAPE, AND THE ERROR IN IT (`web-api.md`: one part
        # failing carries its own `error`).
        return {"error": str(exc), "run": {}, "bench": {}, "placed": {}}


@bp.route("/api/task-setup/commands", methods=["POST"])
def api_task_setup_commands():
    """**What a person types for one stage** -- its prep, as chosen (what it
    continues from, the machine), its launch, and a benchmark's verdict
    read: the lines the tab shows, composed by the terminal's own composer
    (`jobset/commands.stage_lines`), so a line on the page is one the
    terminal would print and `launch` would take (`job-system.md` § 5.3,
    *what molbuilder prints, you can type*; W55 B4).

    POST ``{dest, kind, stage, from?, cold?, target?}`` -> ``{ok, lines}``.
    A stage prepped already has no prep line -- prep would refuse it -- as
    `jobset status` prints it.  Writes nothing.
    """
    from molbuilder.jobset.commands import stage_lines
    from molbuilder.scheduler.record import LOCAL_TARGET
    body = request.get_json(silent=True) or {}
    dest_raw = str(body.get("dest") or "")
    kind = str(body.get("kind") or "run")
    stage = str(body.get("stage") or "")
    if not dest_raw or not stage or kind not in ("run", "bench"):
        return jsonify({"ok": False,
                        "error": "dest, kind (run|bench) and stage are "
                                 "required"}), 400
    try:
        dest = _resolve_within_roots(dest_raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    target = body.get("target") or None
    if target == "(this machine)":
        target = LOCAL_TARGET
    prepped = stage in _folder_prepped(dest).get(kind, {})
    return jsonify({"ok": True, "lines": stage_lines(
        kind, stage, base=dest, from_attempt=(body.get("from") or None),
        cold=bool(body.get("cold")), target=target, prepped=prepped)})


def _folder_bench_refusal(folder, described):
    """The folder answer's ``bench_refusal``: the entry's reason this
    description takes no bench, or ``None`` -- also when there is no
    readable description to ask, since then there is no Measure step."""
    if described is None:
        return None
    from molbuilder.jobset.prep_inputs import bench_refusal
    from molbuilder.task import FILENAME as TASK_FILENAME, read_task
    try:
        return bench_refusal(read_task(folder / TASK_FILENAME))
    except Exception:                                         # noqa: BLE001
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

    These are the SAME table `default_siesta_stages` builds the shipped ladder
    from, so a stage filled here and a stage of the default ladder cannot drift
    -- `engines/tuning.md` § 4 is the authority for what number each tier
    carries, and this serves it rather than restating it.

    **Offered per KIND, by the columns' own membership rule.** A tier is a
    set of values for named fields; it is offered for this folder's kind
    only when every one of those fields may be a column of the table
    (`_column_items`) -- `task-setup.md` § 9: *"a preset that half-applied
    would be worse than one that refused"*.  The relaxation tiers are
    columns of an optimization and of a vibration ladder's relax rung, and
    of NO transport rung (`stages` routes each rung its own items;
    `engines/transport.md` § 2a.7: a rung carries its role's profile), so a
    transport description gets an empty menu and its rows draw none.
    """
    engine = str(request.args.get("engine") or "").lower()
    kind = str(request.args.get("calculation") or "")
    if not engine or not kind:
        return _unstated(engine=engine, calculation=kind)
    out = []
    if engine == "siesta":
        from molbuilder.config.siesta import (SIESTA_STAGE_NAMES,
                                              SIESTA_STAGE_PRESETS)
        for tier in sorted(SIESTA_STAGE_PRESETS):
            out.append({"tier": tier,
                        "name": SIESTA_STAGE_NAMES[tier],
                        "values": dict(SIESTA_STAGE_PRESETS[tier])})
    elif engine == "pyscf":
        # Same source as the shipped PySCF ladder, for the same reason the
        # SIESTA arm above reads SIESTA's: a stage filled from a preset here
        # and a stage of the default ladder must not be able to disagree.
        # ``restart`` is dropped -- it is a rung's POSITION, not its tier
        # (`run-identity.md` § 4 rule 3), so it is not a value to fill a row
        # with.  Every tier is offered whatever the strategy enables; the
        # enable-mask is the ladder's business, not this menu's.
        from molbuilder.pyscf.stages import default_pyscf_stages
        for i, st in enumerate(default_pyscf_stages(), start=1):
            out.append({"tier": i, "name": st.name,
                        "values": {k: v for k, v in st.overrides.items()
                                   if k != "restart"}})
    else:
        return jsonify({"ok": False, "error": f"unknown engine {engine!r}"}), 400
    columns = {it.name for it in _column_items(engine, kind)}
    out = [ps for ps in out if set(ps["values"]) <= columns]
    return jsonify({"ok": True, "engine": engine, "calculation": kind,
                    "presets": out})
