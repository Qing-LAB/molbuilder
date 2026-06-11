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

from dataclasses import fields as _dataclass_fields
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ._shared import (
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
    apply_sidecar_if_possible,
)

from molbuilder.config.transport import TransportConfig
from molbuilder.transport import get_engine, UnknownEngineError


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

    Drops unknown keys (forward-compat with a UI that sends extras)
    and coerces None / missing values to the dataclass default.
    Field-name lock-step with TransportConfig is the contract; if
    the form ever renames a field the JSON shape must follow.
    """
    known = {f.name for f in _dataclass_fields(TransportConfig)}
    clean = {k: v for k, v in (params or {}).items()
             if k in known and v is not None}
    return TransportConfig(**clean)


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
        "ok":      True,
        "engine":  "transiesta",
        "script":  "<.fdf text>",
        "filename": "<jobname>.fdf",
        "issues":  [{"severity": "warn", "message": "...", "where": "..."}],
        "errors":  []
      }

    On preflight errors (``severity = "error"``) the endpoint
    returns ``ok = False`` with ``errors`` populated; ``script``
    is not emitted (generating an incorrect .fdf would risk a
    silent runtime failure for the user).

    Engine dispatch goes through the registry — adding a new
    engine = drop ``molbuilder/transport/<engine>.py`` with an
    ``@register_engine`` decorator + import it in
    ``molbuilder/transport/__init__.py``.  This endpoint needs no
    change.
    """
    from .files import _PickerError
    from ._shared import regions_pattern_b_notice  # noqa: F401  (future use)

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

    # Apply the .molstruct.json sidecar — this is where the
    # L-electrode / R-electrode / bridge region labels come from.
    # Without the sidecar, struct.regions is empty and preflight
    # will surface a clear error.
    sidecar_notice = apply_sidecar_if_possible(struct, str(struct_path))

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

    # Preflight: scientific + structural checks before emission.
    issues = engine.preflight(struct, cfg)
    if sidecar_notice:
        from molbuilder.issues import Issue
        issues.append(Issue("warn", sidecar_notice, "config.frozen_atoms"))

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        return jsonify({
            "ok":      False,
            "engine":  cfg.engine,
            "script":  None,
            "errors":  _issues_to_json(errors),
            "issues":  _issues_to_json(issues),
        }), 400

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
        "ok":       True,
        "engine":   cfg.engine,
        "script":   script,
        "filename": filename,
        "issues":   _issues_to_json(issues),
        "errors":   [],
    })
