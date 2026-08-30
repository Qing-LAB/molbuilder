"""Transport-calculation blueprint.

The ``/transport-calculation`` tab's server half.  The tab is the
composite's WHOLE describe surface (user ruling 2026-08-29: no
hand-over — nothing is awaiting, so the tab selects and decides):

    GET  /api/transport/schema            form-rendering schema for
                                          the TransportConfig dataclass
    GET  /api/transport/describe_attempt  the slot picker's describe
                                          seam: one line from a cited
                                          attempt's own .fdf, plus the
                                          server-spelled citation and
                                          the calculation's source file
    POST /api/transport/describe          the FINISHED task.json text —
                                          the web spelling of `jobset
                                          init --calculation transport`;
                                          the browser writes it where
                                          the user chose
    POST /api/transport/render            validate + render a device
                                          .fdf from a posted structure
                                          (the engine registry's
                                          validation surface; the
                                          composite renders through
                                          prep, never through this)

Mirrors the contract of Build's ``/api/build/schema/<engine>`` and
Spectra's ``/api/spectra/render``: the dataclass field metadata
is the single source of truth; the JS renders the form directly
from the returned schema; the render endpoint is thin and
dispatches all engine-specific logic through the registry.
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    config_from_params,
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
)

from molbuilder.config.transport import TransportConfig
from molbuilder.transport import get_engine, UnknownEngineError
from molbuilder.validation import validate as _validate


bp = Blueprint("transport", __name__)


def _fence(raw: str):
    """A tree-relative path → ``(citation, directory, root, refusal)``.

    ONE fence for both citation doors: contain the path to the projects
    tree, require a directory, and compute the citation string the rest
    of the composite speaks in.  ``refusal`` is a ready answer when
    either test fails and ``None`` when neither did.

    The two doors diverge only AFTER this, and deliberately: describing
    an unciteable directory answers 200 with the condition as its
    summary (that is how a person learns what a citation needs), while
    renaming one is simply a bad request.  Sharing the fence keeps that
    difference visible instead of hiding it inside two copies of the
    same three lines.
    """
    from pathlib import Path

    from molbuilder.projects import OutsideRoot, contain, projects_root
    root = Path(projects_root()).resolve()
    try:
        cite_dir = contain(root / raw, root)
    except OutsideRoot as exc:
        return None, None, root, (
            jsonify({"ok": False, "error": str(exc)}), 400)
    if not cite_dir.is_dir():
        return None, None, root, (
            jsonify({"ok": False,
                     "error": f"{raw} is not a directory"}), 404)
    return str(cite_dir.relative_to(root)), cite_dir, root, None


# ===================================================================== #
# /api/transport/describe_attempt  --  the slot picker's describe seam  #
# ===================================================================== #


@bp.route("/api/transport/describe_attempt", methods=["GET"])
def api_transport_describe_attempt() -> Any:
    """Classify a picked directory against the § 4.1b citation
    condition and describe what it provides — the `describe` seam the
    shared tree-picker feeds on when the Transport tab picks the
    junction slot (P7; reworked 2026-08-29, second user ruling: the
    condition is FILES, never layout).

    ``?path=`` is tree-relative.  The answer names the form the
    directory satisfies ("relaxation" | "structure"), or — when it
    satisfies neither — ``form: null`` with the refusal as the summary,
    naming exactly which file is missing (the condition stated).  For a
    relaxation it is honest about convergence (CONCLUDED / not / no
    record at all), and it says whether the electronic contract is the
    citation's ("cited") or the description's own ("open").
    ``structure`` carries the cited junction's labeled structure for
    the viewer (the /api/build/load ``{structure}`` envelope), so the
    tab shows the citation whatever the form — **and whether or not it
    composes**: a directory that classifies but cannot be built into a
    calculation still answers with its junction, and the refusal is
    appended to the summary above it.  ``fix`` is a word the tab can
    act on (today only ``"swap_electrodes"``), never prose to match.
    """
    from molbuilder.transport.compose import (ComposeError,
                                              classify_citation,
                                              compose_junction,
                                              labeled_citation_structure,
                                              recorded_contract_of)
    from molbuilder.transport.preflight import parse_fdf_params
    from molbuilder.transport.sort import (ORDER_INVERTED,
                                           electrode_orientation)

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    citation, cite_dir, root, refusal = _fence(raw)
    if refusal is not None:
        return refusal
    try:
        cited = classify_citation(cite_dir)
    except ComposeError as exc:
        # Not citable -- the refusal IS the answer (it names the
        # missing file and states the whole condition).
        return jsonify({"ok": True, "citation": None, "form": None,
                        "summary": str(exc)}), 200

    if cited.form == "structure":
        recorded = recorded_contract_of(cited)
        if recorded is not None:
            # 4.1b's third shade: the pair carries the finished run's
            # own contract, so the lane is CITED, same as a deck.
            contract = "cited"
            status = ("labeled structure · contract RECORDED from the "
                      f"{recorded.get('engine', '?')} deck "
                      f"({recorded.get('source', '?')})")
        else:
            contract = "open"
            status = "labeled structure (taken as given)"
        params_out = None
        concluded = None
    else:
        deck_text = cited.deck.read_text()
        p = parse_fdf_params(deck_text)
        bits = []
        if p.basis_size:
            bits.append(str(p.basis_size))
        if p.mesh_cutoff_ry:
            bits.append(f"{p.mesh_cutoff_ry:g} Ry")
        if p.xc:
            bits.append(str(p.xc))
        if p.kgrid:
            bits.append("k " + "x".join(str(k) for k in p.kgrid))
        if p.n_atoms:
            bits.append(f"{p.n_atoms} atoms")
        if cited.concluded is not None:
            state = f"CONCLUDED ({cited.concluded.strip()})"
        elif cited.has_record:
            state = ("NOT CONCLUDED -- still running, or force-stopped "
                     "(the two look identical on disk)")
        else:
            state = ("no run record -- the .XV is taken as the final "
                     "geometry (convergence unverified)")
        status = state + (" · " + " · ".join(bits) if bits else "")
        params_out = {
            "basis_size": p.basis_size,
            "mesh_cutoff_ry": p.mesh_cutoff_ry,
            "xc": p.xc,
            "kgrid": list(p.kgrid) if p.kgrid else None,
            "n_atoms": p.n_atoms,
        }
        contract = "cited"
        concluded = bool(cited.concluded)

    # TWO SEPARATE QUESTIONS, and the card needs both: *what is this
    # junction* (always answerable from the citation's own files) and
    # *can it be composed into a calculation* (a refusal, sometimes).
    # They used to be one call, so every reason a junction cannot be
    # BUILT -- labels missing, an electrode that moved, a mid-run
    # record, blocks that interleave -- also blanked the viewer, and
    # the refusal was read over an empty card instead of over the thing
    # it is about.
    #
    # The composition answers both when it succeeds (`relaxed` IS the
    # labeled citation structure), so the happy path reads the .XV once
    # and only the refusal path pays for a second look.
    structure_wire = None
    fix = None
    try:
        composed = compose_junction(citation, tree_root=root)
        rel_struct = (composed.relaxed
                      if composed.relaxed is not None
                      else composed.sorted.structure)
        structure_wire = rel_struct.to_dict()
        # THE CONVENTION IS CHECKED AND REPORTED, NEVER ENFORCED (user
        # ruling, 2026-08-29).  An inverted junction composes and runs
        # -- it biases the other end -- so the tab gets the observation
        # (with the numbers, for the meta line) plus `fix` as a WORD it
        # can act on without matching prose.
        for note in composed.sorted.notes:
            status = status + "  ⚠ " + note
        if (electrode_orientation(composed.sorted.structure)
                == ORDER_INVERTED):
            fix = "swap_electrodes"
    except Exception as exc:  # noqa: BLE001 -- surfaced, never fatal
        status = status + "  !! " + str(exc)
        # SHOW IT ANYWAY.  The labels may also be the wrong way round,
        # and the rename is still worth offering on a junction whose
        # refusal is about something else entirely.
        try:
            cited_struct, _src = labeled_citation_structure(cited)
            structure_wire = cited_struct.to_dict()
            if electrode_orientation(cited_struct) == ORDER_INVERTED:
                fix = "swap_electrodes"
        except (ComposeError, OSError, ValueError):
            # Labels that cannot be read leave nothing to draw and
            # nothing to offer.  NARROW on purpose: a blanket except
            # here would swallow a programming error into a silently
            # empty card.
            pass

    return jsonify({
        "ok": True,
        "citation": citation,
        "form": cited.form,
        "contract": contract,
        "concluded": concluded,
        "summary": status,
        "structure": structure_wire,
        "params": params_out,
        "fix": fix,
    })


@bp.route("/api/transport/swap_electrodes", methods=["POST"])
def swap_electrodes():
    """Swap ``L-electrode`` / ``R-electrode`` on a cited junction --
    the fix the person AGREED to after describe offered it.

    It edits their finished run's label block and nothing else (two
    arrays of indices in molbuilder's own metadata; no coordinate, no
    keyword, no result), which is why relabeling does not invalidate
    the relaxation.  Fixed at the source, so every later citation of
    that directory is right too.
    """
    from molbuilder.transport.compose import (ComposeError,
                                              resolve_citation,
                                              swap_electrode_labels)

    body = request.get_json(silent=True) or {}
    raw = str(body.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    citation, _cite_dir, root, refusal = _fence(raw)
    if refusal is not None:
        return refusal
    # RESOLVE AND CLASSIFY THROUGH THE ONE DOOR prep composes through
    # (`resolve_citation`), not a hand-rolled repeat of its three steps
    # with its own wording.  Unlike `describe_attempt` -- which must
    # answer 200 with the refusal as its summary, because describing an
    # unciteable directory is how a person LEARNS the condition -- a
    # rename asked of a directory that is not a citation is simply a
    # bad request.
    try:
        _dir, cited = resolve_citation(citation, root)
        changed = swap_electrode_labels(cited)
    except ComposeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({
        "ok": True,
        "changed": changed,
        "message": (f"Swapped L-electrode and R-electrode in {changed} "
                    f"-- labels only; no coordinate, keyword or result "
                    f"was touched."),
    })


# ===================================================================== #
# /api/transport/describe  --  the tab writes the DESCRIPTION itself   #
# ===================================================================== #


@bp.route("/api/transport/describe", methods=["POST"])
def api_transport_describe() -> Any:
    """Render the transport calculation's COMPLETE ``task.json`` text.

    There is no hand-over for the composite (user ruling 2026-08-29):
    the other kinds hand to Task setup because that tab owns questions
    they cannot answer — shape, stages, what varies.  Transport has
    none open: the five stages and the hierarchical shape are fixed by
    design, the identity derives from the citation, the knobs ride the
    stages' override bags.  So this door answers with the finished
    description, ONE file, and the browser writes it where the user
    chose through the content-blind file layer (`web/projects.md` § 1 —
    the same division of labour as every other tab's writes).

    Validation is the shipped codec's: the ``Task`` construction below
    is the same gate `read_task` and the CLI's ``jobset init`` run, and
    the citation resolves through the same door prep composes through.
    """
    from molbuilder.persist import json_text
    from molbuilder.projects import projects_root
    from molbuilder.task import FILENAME as TASK_FILENAME
    from molbuilder.task import Stage, Task, derive_run
    from molbuilder.transport.compose import ComposeError, resolve_citation
    from molbuilder.transport.stages import (CONTRACT_FIELDS,
                                             SEALED_ALWAYS,
                                             TRANSPORT_STAGES)

    body = request.get_json(silent=True) or {}
    engine = str(body.get("engine") or "siesta").lower()
    if engine != "siesta":
        return jsonify({"ok": False,
                        "error": "transport is SIESTA-first "
                                 "(TranSIESTA)"}), 400
    citation = str(body.get("junction") or "")
    if not citation:
        return jsonify({"ok": False,
                        "error": "no junction citation -- pick the "
                                 "relaxed junction's attempt first"}), 400
    bias_raw = body.get("bias") or [0.0]
    try:
        bias = tuple(float(v) for v in bias_raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False,
                        "error": f"bias must be a list of volts, "
                                 f"got {bias_raw!r}"}), 400
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        return jsonify({"ok": False,
                        "error": "overrides must be an object"}), 400
    typed = str(body.get("name") or "") or "transport"

    try:
        _, cited = resolve_citation(citation, projects_root())
    except ComposeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Refused HERE, not at prep on the cluster: an unknown knob or a
    # sealed one names itself while changing it is still free.  Same
    # sets, same conditions as config_for (4.1b: the contract fields
    # are the citation's ONLY when the citation carries a deck).
    import dataclasses as _dc
    _known = {f.name for f in _dc.fields(TransportConfig)}
    from molbuilder.transport.compose import recorded_contract_of
    _contract_sealed = (cited.form == "relaxation"
                        or recorded_contract_of(cited) is not None)
    for _name in overrides:
        if _name not in _known:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is not a transport "
                                     f"parameter"}), 400
        if _name in SEALED_ALWAYS:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is the description's "
                                     f"own field (identity, bias) -- "
                                     f"it is never an override"}), 400
        if _contract_sealed and _name in CONTRACT_FIELDS:
            return jsonify({"ok": False,
                            "error": f"{_name!r} is the citation's to "
                                     f"say (the electronic contract "
                                     f"arrives from the cited "
                                     f"relaxation's own deck) -- reset "
                                     f"it in the form; cite a "
                                     f"relaxation that ran with the "
                                     f"values you want, or cite a "
                                     f"plain .xyz+.molstruct pair, "
                                     f"whose contract fields are "
                                     f"open"}), 400
    try:
        task = Task(
            engine="siesta", shape="hierarchical",
            run=derive_run(typed, citation,
                           stage_names=TRANSPORT_STAGES),
            structure=None, calculation="transport",
            slots={"junction": citation}, bias=bias,
            # the stages.md 6.2 rule holds here too: an override names
            # a PROMOTED field, and `varies` is the promotion
            varies=tuple(sorted(overrides)),
            stages=tuple(Stage(name=n, enabled=True,
                               overrides=(dict(overrides)
                                          if n == "device" else {}))
                         for n in TRANSPORT_STAGES))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({
        "ok": True,
        "label": task.label,
        "files": [{"name": TASK_FILENAME,
                   "text": json_text(task.to_dict())}],
        "notices": [],
    })


# ===================================================================== #
# /api/transport/schema  --  form schema endpoint                       #
# ===================================================================== #


@bp.route("/api/transport/schema", methods=["GET"])
def api_transport_schema() -> Any:
    """Return the transport TAB's form schema: the transport-only knobs.

    The electronic contract (engine, basis, XC, mesh, temperature, the
    transverse k, the bias, the label) is the CITATION's to say — it
    arrives from the cited junction's own deck at prep, and the
    describe door refuses those fields BY NAME.  A form field the door
    is guaranteed to refuse is a trap, not a control (found rendered
    2026-08-29: ten sealed fields sat as editable inputs, and the bias
    was asked twice), so the sealed set is filtered HERE, from the same
    one constant the two refusing doors read.  What remains IS the
    override lane: Transmission / NEGF / Runtime knobs that ride the
    device stage's bag (stages.md § 6.2).  The bias is card 4's own
    input — a describe-level fact beside the citation, not a config
    override.

    Section order still follows ``TransportConfig._form_section_order``;
    sections the filter empties (System, Electrodes) are dropped whole.
    """
    from molbuilder.transport.stages import (CONTRACT_FIELDS,
                                             SEALED_ALWAYS)
    # ?contract=cited (default) hides the contract fields -- they are
    # the citation's deck's to say; ?contract=open offers them, for a
    # form-B citation (a labeled pair carries no deck; 4.1b).  The tab
    # passes what describe_attempt answered.
    hidden = set(SEALED_ALWAYS)
    if str(request.args.get("contract") or "cited") != "open":
        hidden |= CONTRACT_FIELDS
    schema = _dataclass_to_form_schema(TransportConfig, "t")
    kept = []
    for sec in schema.get("sections", []):
        fields_left = [f for f in sec.get("fields", [])
                       if f.get("name") not in hidden]
        if fields_left:
            sec = dict(sec)
            sec["fields"] = fields_left
            kept.append(sec)
    schema = dict(schema)
    schema["sections"] = kept
    response: Dict[str, Any] = {"ok": True, "schema": schema}
    return jsonify(response)


# ===================================================================== #
# /api/transport/render  --  dispatch via engine registry               #
# ===================================================================== #


def _transport_config_from_params(params: Dict[str, Any]) -> TransportConfig:
    """Build a :class:`TransportConfig` from a JSON form-values dict.

    Routes through ``config_from_params`` so per-field type coercion
    (``Sequence[float]`` for ``bias_voltages_v``, ``Tuple[int,int,int]``
    for ``k_mesh_transverse``, the standard numeric coercers) fires
    BEFORE the dataclass constructor sees the raw form values.

    Field-name lock-step with TransportConfig is the contract; if
    the form ever renames a field the JSON shape must follow.
    Unknown keys are silently dropped by ``config_from_params``
    (forward-compat with a UI that sends extras).
    """
    import typing as _typing
    hints = _typing.get_type_hints(TransportConfig)
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    return config_from_params(TransportConfig, clean, hints)


@bp.route("/api/transport/render", methods=["POST"])
def api_transport_render() -> Any:
    """Render the device script for the selected Transport engine.

    Body (JSON)::

      {
        "params":          {<TransportConfig field values>},
        "structure_path":  "/abs/path/to/relaxed.xyz",
      }

    Returns::

      {
        "ok":          True,
        "engine":      "transiesta",
        "script":      "<.fdf text>",
        "filename":    "<jobname>.fdf",
        "issues":      [{"severity": "warn", "message": "...", "where": "..."}],
        "errors_only": []
      }

    On preflight errors (``severity = "error"``) the endpoint
    returns ``ok = False`` with ``errors_only`` populated;
    ``script`` is not emitted (generating an incorrect .fdf would
    risk a silent runtime failure for the user).

    ``errors_only`` is the pre-filtered error-severity subset of
    ``issues`` — see the field-meaning comment block below the
    preflight-error branch for the full envelope shape.

    Engine dispatch goes through the registry — adding a new
    engine = drop ``molbuilder/transport/<engine>.py`` with an
    ``@register_engine`` decorator + import it in
    ``molbuilder/transport/__init__.py``.  This endpoint needs no
    change.
    """
    from .files import _PickerError
    # Pattern-B notice (regions_pattern_b_notice from _shared) is
    # NOT imported here: it fires when an engine doesn't consume
    # struct.regions (Build/Spectra are the consumers).  Transport
    # IS the consumer of region labels — they drive the entire
    # device/electrode separation — so the Pattern-B path doesn't
    # apply.  Documented for clarity (2026-06-10 post-review).

    body = request.get_json(silent=True) or {}
    params: Dict[str, Any] = body.get("params") or {}

    # THE STRUCTURE ARRIVES AS DATA, in the one envelope every structure door
    # takes (web-api.md § 1): the atoms as numbers with the labels and the cell
    # beside them, all read together by `molview.exportFile()`.
    #
    # A second place labels can arrive from is a place they can be dropped from
    # without anyone noticing (#41); the way to close that is to stop having a
    # second place, not to rank the two.
    if not isinstance(body.get("structure"), dict):
        # Said here rather than left to the shared helper, whose fallback is the
        # legacy `xyz` text field and whose complaint is therefore about `xyz` --
        # a field this route's only caller has never sent.
        return jsonify({
            "ok": False,
            "error": "no 'structure' provided (the region-labeled device)",
        }), 400
    from ._shared import struct_from_body
    try:
        struct = struct_from_body(body)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # PROVENANCE, NOT GEOMETRY.  `structure_path` says which file this came from
    # so a message can name it.  It is optional and nothing is read from it;
    # still checked against the picker roots when present, so a path the caller
    # invented cannot be echoed back into a response.
    structure_path_raw = (body.get("structure_path") or "").strip()
    if structure_path_raw:
        try:
            from .build import _resolve_path_within_roots
            _resolve_path_within_roots(structure_path_raw, require="file")
        except _PickerError as exc:
            return jsonify({"ok": False, "error": exc.message}), exc.status

    # The labels arrived WITH the structure and were applied by
    # `Structure.from_dict` -- the one deserialiser, which validates through the
    # same `__post_init__` a freshly built Structure runs.  Nothing to apply
    # here, and no second copy to rank against the first.
    from ._shared import periodicity_checked_for_emit
    struct = periodicity_checked_for_emit(struct)

    # Build the config.  Unknown-field protection + dataclass
    # validation surfaces a clean 400 for bad params instead of a
    # TypeError stack trace.
    try:
        cfg = _transport_config_from_params(params)
    except Exception as exc:    # noqa: BLE001
        return jsonify({
            "ok": False,
            "error": f"bad parameters: {exc}",
        }), 400

    # Dispatch via the registry.
    try:
        engine = get_engine(cfg.engine)
    except UnknownEngineError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # SINGLE validation gate (V1/V2, 2026-07): validate() runs
    # validate_geometry + _validate_config_metadata + the registered
    # engine validator.  TransportConfig is now registered (its validator
    # dispatches to the transport engine's preflight -- region-label
    # presence/unknown, device transport-axis kz=1, and cross-engine
    # chemistry checks), so there is no separate engine.preflight() pass
    # to hand-concatenate and forget.
    issues = list(_validate(struct, cfg))

    # What each field on this response means — written down here so
    # anyone reading the code later does not mistake the names for
    # each other:
    #
    #   error        — a short string for the status banner at the
    #                  top of the form, e.g. "preflight failed; see
    #                  issues".  Not present on a successful run.
    #                  Same field that /api/build/fdf, /api/build/
    #                  pyscf, and /api/spectra/render use.
    #
    #   issues       — the full list of things found while preparing
    #                  the script: errors, warnings, and notes
    #                  mixed together (each carries a severity).
    #                  The UI shows this as the colour-coded list
    #                  under the banner.
    #
    #   errors_only  — the same list as `issues`, with only the
    #                  error-severity items kept.  Always emitted as
    #                  a list, including [] on success.  The browser
    #                  does not read this today; it stays on the
    #                  wire so a future caller (a CI script, a
    #                  future "show only errors" button) can read
    #                  the blockers without doing the severity
    #                  filter on its own.
    #
    # Do not delete `errors_only` thinking it duplicates `error` or
    # `issues`.  It does not: `error` is one string, `issues` is the
    # mixed-severity list, `errors_only` is the pre-filtered
    # error-severity slice of `issues`.
    errors_only = [i for i in issues if i.severity == "error"]
    if errors_only:
        # Omit the `script` key entirely on preflight failure
        # (instead of returning `script: None`) — the JS detects
        # absence to drive the script-preview card visibility.
        # Mirrors /api/build/fdf, /api/build/pyscf,
        # /api/spectra/render.
        # web-api.md § 1.6 (b) scientific advisory: HTTP 200
        # explicit — the form's workflow cards (web-ui-coherence
        # Rule 2) render the findings inline.
        return jsonify({
            "ok":          False,
            "engine":      cfg.engine,
            "error":       "preflight failed; see issues",
            "issues":      _issues_to_json(issues, cfg=cfg),
            "errors_only": _issues_to_json(errors_only, cfg=cfg),
        }), 200

    # Emit the script.  Engine-side rendering is pure (no I/O); the
    # web layer writes the file separately via /api/files/write if
    # the JS path persists it.
    try:
        script = engine.render_script(struct, cfg)
    except NotImplementedError as exc:
        return jsonify({
            "ok":      False,
            "engine":  cfg.engine,
            "error":   str(exc),
        }), 501
    except Exception as exc:    # noqa: BLE001
        return jsonify({
            "ok":      False,
            "engine":  cfg.engine,
            "error":   f"render failed: {exc}",
        }), 500

    # Extension routing: SIESTA family → .fdf; PySCF family → .py.
    # Today's only registered engine is transiesta; the
    # if-chain shape leaves room for pyscf-negf to drop in.
    if cfg.engine == "transiesta":
        filename = f"{cfg.job_name}.fdf"
    else:
        filename = f"{cfg.job_name}.py"

    return jsonify({
        "ok":          True,
        "engine":      cfg.engine,
        "script":      script,
        "filename":    filename,
        "issues":      _issues_to_json(issues, cfg=cfg),
        # See the field-meaning comment near the preflight-error
        # branch above for what `errors_only` is for.
        "errors_only": [],
    })
