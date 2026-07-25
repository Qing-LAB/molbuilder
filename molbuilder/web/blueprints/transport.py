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
    apply_sidecar_if_possible,
)

from molbuilder.config.transport import TransportConfig
from molbuilder.transport import get_engine, UnknownEngineError
from molbuilder.validation import validate as _validate


bp = Blueprint("transport", __name__)


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
    structure_path_raw = (body.get("structure_path") or "").strip()
    if not structure_path_raw:
        return jsonify({
            "ok": False,
            "error": "structure_path is required (the relaxed "
                     "geometry XYZ)",
        }), 400

    # Path validation through the picker-root allowlist.
    try:
        from .build import _resolve_path_within_roots
        struct_path = _resolve_path_within_roots(
            structure_path_raw, require="file",
        )
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # Parse the XYZ into a Structure.
    try:
        from molbuilder.structure import Structure
        struct = Structure.from_xyz(struct_path.read_text())
    except (ValueError, IndexError, OSError) as exc:
        return jsonify({
            "ok": False,
            "error": f"could not parse structure file: {exc}",
        }), 400

    # 2026-06-14 contract update: prefer in-body labels (the
    # viewer-is-truth contract).  Transport's L-electrode /
    # R-electrode / bridge region labels travel in the POST body
    # from the Transport tab's in-memory state; only when neither
    # in-body key is present do we re-read the sidecar from disk.
    # See _shared.apply_labels_to_struct docstring.
    from ._shared import apply_labels_to_struct
    # ``body`` doesn't yet carry ``structure_path`` in the local
    # name scope here; the helper reads it via body.get for the
    # fallback path.
    sidecar_notice = apply_labels_to_struct(struct, body)

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
    # dispatches to the transport engine's preflight -- region / electrode
    # ordering / charge-neutrality / bias checks), so there is no separate
    # engine.preflight() pass to hand-concatenate and forget.
    issues = list(_validate(struct, cfg))
    if sidecar_notice:
        from molbuilder.issues import Issue
        # ``where="structure_path"`` matches Spectra's sidecar-load
        # notice (spectra.py:400) so the wire contract is consistent
        # across engines — the UI shows a single "[structure_path]"
        # tag for sidecar problems regardless of which engine flagged
        # them.  The notice text already explains the frozen-atoms
        # consequence in human-readable form.
        issues.append(Issue("warn", sidecar_notice, "structure_path"))

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
