"""Transport-calculation blueprint.

Form-schema + script-render endpoints for the
``/transport-calculation`` tab.  Phase B.3 step 2 (2026-06-10):
the render endpoint dispatches via the
:mod:`molbuilder.transport` engine registry so a new engine
(``pyscf-negf``, ``inelastica``, ...) drops in without changes
here.

Routes:

    GET  /api/transport/schema       form-rendering schema for
                                     the TransportConfig dataclass
    POST /api/transport/render       render the device .fdf for the
                                     selected engine + structure;
                                     also runs preflight and
                                     returns the issues panel

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


# ===================================================================== #
# /api/transport/describe_attempt  --  the slot picker's describe seam  #
# ===================================================================== #


@bp.route("/api/transport/describe_attempt", methods=["GET"])
def api_transport_describe_attempt() -> Any:
    """One line about a cited attempt, FROM ITS OWN ``.fdf`` — the deck
    that actually ran is the truth about a result (user ruling
    2026-08-28), and this is the `describe` seam the shared tree-picker
    feeds on when the Transport tab picks the junction slot (P7).

    ``?path=`` is tree-relative (the same path language every citation
    speaks).  The answer is honest about the two states that matter:
    CONCLUDED or not (still running and force-stopped look identical on
    disk, and this endpoint never decides), and the electronic contract
    the composite would inherit.
    """
    from pathlib import Path

    from molbuilder.jobset.materialize import attempt_concluded
    from molbuilder.projects import OutsideRoot, contain, projects_root
    from molbuilder.transport.preflight import parse_fdf_params

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    root = Path(projects_root()).resolve()
    try:
        attempt = contain(root / raw, root)
    except OutsideRoot as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not attempt.is_dir():
        return jsonify({"ok": False,
                        "error": f"{raw} is not a directory"}), 404
    decks = sorted(attempt.glob("*.fdf"))
    if not decks:
        return jsonify({"ok": False,
                        "error": "no .fdf in this attempt -- it was "
                                 "never prepped, so it cannot be "
                                 "cited"}), 404
    deck = decks[0]
    concluded = attempt_concluded(attempt, deck.stem)
    p = parse_fdf_params(deck.read_text())
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
    status = (f"CONCLUDED ({concluded.strip()})" if concluded
              else "NOT CONCLUDED -- still running, or force-stopped "
                   "(the two look identical on disk)")
    return jsonify({
        "ok": True,
        "concluded": bool(concluded),
        "summary": status + (" · " + " · ".join(bits) if bits else ""),
        "deck": deck.name,
        "params": {
            "basis_size": p.basis_size,
            "mesh_cutoff_ry": p.mesh_cutoff_ry,
            "xc": p.xc,
            "kgrid": list(p.kgrid) if p.kgrid else None,
            "n_atoms": p.n_atoms,
        },
    })


# ===================================================================== #
# /api/transport/schema  --  form schema endpoint                       #
# ===================================================================== #


@bp.route("/api/transport/schema", methods=["GET"])
def api_transport_schema() -> Any:
    """Return the TransportConfig form schema.

    Section order is taken from ``TransportConfig._form_section_order``
    so the workflow-order (System → Geometry → Electrodes →
    Transmission → NEGF → Runtime) is stable independent of field-
    declaration order in the dataclass.

    Sidecar-driven defaults (electrode/bridge regions seeding the
    Geometry section) follow the Spectra pattern in a later
    iteration — engine implementations come first.  Today the
    endpoint returns the static schema only.
    """
    schema = _dataclass_to_form_schema(TransportConfig, "t")
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
