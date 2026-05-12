"""Spectra blueprint -- harmonic frequencies + Raman + per-mode ES routes.

Routes (spec § 10):

    GET  /spectra                       the page (rendered template)
    GET  /api/build/schema/spectra      form-rendering schema (SpectraConfig)
    POST /api/spectra/render            render the runnable spectra.py script
    POST /api/spectra/load              parse an uploaded <job>.spectra.json
                                        (or read JSON body) into the typed
                                        SpectraResults shape

The form-schema endpoint name (``/api/build/schema/spectra``)
matches the existing schema-dispatch convention used by Build's
``/api/build/schema/<engine>`` -- a fourth ``engine`` value, not
a new namespace.  The Spectra-tab JS calls the same shared
schema-driven form renderer that Build uses; nothing duplicated.

Response shapes mirror Build's: every endpoint returns
``{"ok": True, ...}`` on success or ``{"ok": False, "error":
"<msg>"}`` on failure (plus an ``issues`` list when relevant).

The render path:

  1. Parse XYZ -> Structure (delegated to Structure.from_xyz).
  2. Build SpectraConfig from form params (config_from_params with
     SpectraConfig-specific Optional sentinels).
  3. Optionally load prior SpectraResults if the client supplied a
     ``prior_path`` so preflight can validate the selector against
     the L3-completed flag.
  4. Engine preflight -> List[Issue].  Errors block; warns pass
     through to the response.
  5. Engine render_script -> the runnable script as text.
  6. methods_text + bibliography_keys composed alongside so the
     UI's Methods-preview modal has the same prose without an
     extra round trip.

The load path:

  * Multipart: ``file=`` form-data field with the .spectra.json
    bytes.
  * JSON: ``{"path": "..."}`` to read from a local path (useful
    when the user runs the script and wants to load results from
    a known job dir without manually uploading).
  * JSON: ``{"json": {...}}`` to parse a dict in-memory (for
    programmatic callers; matches parse_spectra_json_dict).

Engine dispatch is via :func:`spectra.engine_base.get_engine` --
the v1 spec ships PySCF; SIESTA is reserved.  Adding a future
engine requires zero changes here: the registry handles
discovery, the engine's preflight + render_script + methods_fragment
hooks slot into the existing pipeline.
"""

from __future__ import annotations

import json
import typing
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from ._shared import (
    config_from_params as _config_from_params,
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
)

from molbuilder.config.spectra import SpectraConfig
from molbuilder.parsers.spectra_json import (
    SpectraJsonError,
    SpectraJsonFieldError,
    SpectraJsonMalformedError,
    SpectraJsonNotFoundError,
    SpectraJsonSchemaError,
    parse_spectra_json,
    parse_spectra_json_dict,
)
from molbuilder.spectra import (
    UnknownEngineError,
    get_engine,
    render_methods_md,
    extract_citation_keys,
)
from molbuilder.spectra.results import SpectraResults
from molbuilder.structure import Structure


bp = Blueprint("spectra", __name__)


# Fields where the form sends "" to mean "leave unset / inherit".
# Frequency window: empty string -> None (unconstrained side).
# Threads: empty string -> None (inherit OMP_NUM_THREADS).
# Dispersion: "none" handled by config_from_params already.
_NONE_SENTINELS = ("freq_min_cm1", "freq_max_cm1", "threads")


# ===================================================================== #
# Page route                                                            #
# ===================================================================== #


@bp.route("/spectra")
def spectra_page():
    """Render the Spectra tab.

    Heavy lifting (form schema, render, load) is done by the API
    routes below; this just hands back the template.  The JS does
    a /api/build/schema/spectra call on page load to populate the
    form.
    """
    return render_template("spectra.html")


# ===================================================================== #
# /api/build/schema/spectra  -- form schema endpoint                    #
# ===================================================================== #


@bp.route("/api/build/schema/spectra", methods=["GET"])
def api_spectra_schema():
    """Form-rendering schema for the Spectra panel.

    Mirrors the contract of Build's ``/api/build/schema/<engine>``
    (siesta / pyscf).  The Spectra-tab JS calls this once on page
    load and renders the form directly from the returned schema;
    no field declarations are duplicated in the HTML template.

    Section order is taken from ``SpectraConfig._form_section_order``
    so the workflow-order (System -> Method -> Frozen atoms ->
    Spectrum -> Electronic structure -> SCF -> Runtime) is stable
    independent of field declaration order in the dataclass.
    """
    return jsonify({
        "ok": True,
        "schema": _dataclass_to_form_schema(SpectraConfig, "s"),
    })


# ===================================================================== #
# /api/spectra/render  -- generate the script                           #
# ===================================================================== #


@bp.route("/api/spectra/render", methods=["POST"])
def api_spectra_render():
    """Render the runnable ``<job>.spectra.py`` for a Structure + params.

    Body (JSON)::

        {
          "xyz":         "<xyz text>",
          "params":      {<SpectraConfig dict>},
          "prior_path":  "<optional path to a prior .spectra.json>"
        }

    Returns on success::

        {
          "ok":                True,
          "script":            "<runnable Python source>",
          "methods_md":        "<Methods paragraph (markdown)>",
          "bibliography_keys": ["Sun2020", "Becke1993", ...],
          "job_name":          "<cfg.job_name>",
          "issues":            [{<Issue>}, ...]
        }

    On failure -- bad input, validator error, or engine render
    error -- returns ``{"ok": False, "error": "<msg>", "issues":
    [...]}`` with the appropriate HTTP code.

    The ``issues`` list comes from
    :meth:`PySCFSpectraEngine.preflight` (combines
    :func:`spectra.selection.validate_selection` with engine-
    specific scientific advisories).  Errors with
    ``severity == "error"`` block rendering; warnings pass
    through and the script is still generated.
    """
    body = request.get_json(silent=True) or {}
    xyz_text: Optional[str] = body.get("xyz")
    params: Dict[str, Any] = body.get("params") or {}
    prior_path: Optional[str] = body.get("prior_path")

    if not xyz_text:
        return jsonify({"ok": False, "error": "no xyz provided"}), 400

    # Parse XYZ.  Errors surface as 400 -- the form's previous step
    # would have populated this from Build; a parse failure here is
    # a wire-level fault.
    try:
        struct = Structure.from_xyz(xyz_text, title="from-browser")
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse xyz: {exc}"}), 400

    # Build SpectraConfig from form params.  Coercion failures
    # surface as an error-severity Issue rather than HTTP 400 so
    # the UI's validation panel renders them uniformly with
    # validator-side failures.
    try:
        cfg = _spectra_config_from_params(params)
    except Exception as exc:
        return jsonify({
            "ok":     False,
            "error":  f"bad parameters: {exc}",
            "issues": [{"severity": "error",
                        "message": f"bad parameters: {exc}",
                        "where":   "config"}],
        }), 400

    # Optionally load prior results so preflight can verify the
    # selector against the L3-completed flag.  A missing or
    # malformed prior is non-fatal -- preflight just doesn't get
    # the bonus check.  The user gets a warn-severity Issue so
    # they know the resume context was ignored.
    prior, prior_warn = _load_prior(prior_path)

    # Engine dispatch.
    try:
        engine = get_engine(cfg.engine)
    except UnknownEngineError as exc:
        return jsonify({
            "ok":     False,
            "error":  str(exc),
            "issues": [{"severity": "error",
                        "message": str(exc),
                        "where":   "config.engine"}],
        }), 400

    issues = list(engine.preflight(struct, cfg, prior=prior))
    if prior_warn is not None:
        issues.append(prior_warn)

    # Block render on any error-severity issue.
    if any(i.severity == "error" for i in issues):
        return jsonify({
            "ok":     False,
            "error":  "preflight failed; see issues",
            "issues": _issues_to_json(issues),
        }), 400

    # Render the script + Methods text + bibliography keys.
    try:
        script = engine.render_script(struct, cfg)
    except Exception as exc:
        # Unhandled render error -> 500 with the message + any
        # warns we accumulated so far so the user can see context.
        return jsonify({
            "ok":     False,
            "error":  f"render failed: {exc}",
            "issues": _issues_to_json(issues),
        }), 500

    methods_md = render_methods_md(
        cfg,
        engine=engine,
        struct=struct,
    )
    bib_keys = extract_citation_keys(methods_md)

    return jsonify({
        "ok":                True,
        "script":            script,
        "methods_md":        methods_md,
        "bibliography_keys": bib_keys,
        "job_name":          cfg.job_name,
        "issues":            _issues_to_json(issues),
    })


# ===================================================================== #
# /api/spectra/load  -- parse a .spectra.json into typed results        #
# ===================================================================== #


@bp.route("/api/spectra/load", methods=["POST"])
def api_spectra_load():
    """Parse a spectra.json (file upload / on-disk path / inline JSON)
    into the typed :class:`SpectraResults`, returned as a JSON-safe
    dict the UI consumes directly.

    Three input modes (mutually exclusive; pick whichever fits):

      1. ``file=``                multipart upload of the JSON bytes.
      2. ``{"path": "..."}``      read from a local path.  Useful for
                                  the post-run "I have results on
                                  disk" case where the user just
                                  enters the job dir.
      3. ``{"json": {...}}``      parse an in-memory dict (for
                                  programmatic callers, e.g. the
                                  live-watch poller that already has
                                  the JSON in hand).

    Returns on success::

        {"ok": True, "results": <SpectraResults.to_dict()>}

    On failure the response carries a structured error that maps
    1-to-1 to the parser's exception hierarchy so the UI can
    decide how to react:

      * SpectraJsonNotFoundError  -> 404
      * SpectraJsonSchemaError    -> 422 with ``expected`` / ``actual``
                                     so the UI can render an "update
                                     molbuilder" hint
      * SpectraJsonMalformedError -> 400
      * SpectraJsonFieldError     -> 400
    """
    # ---- multipart upload ---------------------------------------- #
    if request.files.get("file"):
        upload = request.files["file"]
        try:
            raw = upload.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            return _err_load(
                SpectraJsonMalformedError(
                    f"uploaded file is not UTF-8: {exc.reason} "
                    f"at byte {exc.start}"
                )
            )
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _err_load(SpectraJsonMalformedError(
                f"uploaded file is not valid JSON: {exc}"
            ))
        try:
            results = parse_spectra_json_dict(d)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return jsonify({"ok": True, "results": results.to_dict()})

    # ---- JSON body (path or inline JSON) ------------------------- #
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    inline = body.get("json")
    if path:
        try:
            results = parse_spectra_json(path)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return jsonify({"ok": True, "results": results.to_dict()})
    if inline is not None:
        if not isinstance(inline, dict):
            return _err_load(SpectraJsonMalformedError(
                f"'json' field must be an object, got {type(inline).__name__}"
            ))
        try:
            results = parse_spectra_json_dict(inline)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return jsonify({"ok": True, "results": results.to_dict()})

    return jsonify({
        "ok":    False,
        "error": (
            "no input -- send a multipart 'file' field, "
            "or JSON {'path': '<...>'} or {'json': {...}}"
        ),
    }), 400


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


_SPECTRA_HINTS = typing.get_type_hints(SpectraConfig)


def _spectra_config_from_params(params: Dict[str, Any]) -> SpectraConfig:
    """Build a SpectraConfig from a JSON params dict, with per-field
    type coercion that respects the Optional sentinels for the
    frequency-window bounds (empty string -> None) and the threads
    field (empty string -> inherit OMP_NUM_THREADS).
    """
    return _config_from_params(
        SpectraConfig, params, _SPECTRA_HINTS,
        none_sentinels=_NONE_SENTINELS,
    )


def _load_prior(prior_path: Optional[str]) -> Tuple[
        Optional[SpectraResults], Optional[Any]]:
    """Try to load prior results for the resume path; return
    ``(results, warn_issue_or_None)``.

    A missing file / wrong schema / corrupt JSON is non-fatal here
    -- preflight just doesn't get the L3-completed signal.  We
    surface the failure as a single warn-severity Issue so the
    user knows the resume context was ignored, rather than
    silently dropping it.
    """
    if not prior_path:
        return None, None
    from molbuilder.issues import Issue
    try:
        return parse_spectra_json(prior_path), None
    except SpectraJsonNotFoundError as exc:
        return None, Issue(
            severity="warn",
            message=(f"prior results path {prior_path!r} not found; "
                     f"running as a fresh job"),
            where="config.prior_path",
        )
    except SpectraJsonError as exc:
        return None, Issue(
            severity="warn",
            message=(f"prior results at {prior_path!r} could not be "
                     f"parsed ({exc}); running as a fresh job"),
            where="config.prior_path",
        )


def _err_load(exc: SpectraJsonError):
    """Map a parser exception class to the HTTP response shape.

    Returns (Flask response, status_code) -- the dispatcher decides
    HTTP code based on exception type so the UI can decide how to
    react without parsing the message string.  Schema mismatches
    carry their numeric versions so the UI can render an "update
    molbuilder" hint directly.
    """
    if isinstance(exc, SpectraJsonNotFoundError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "not_found",
        }), 404
    if isinstance(exc, SpectraJsonSchemaError):
        return jsonify({
            "ok":               False,
            "error":            str(exc),
            "kind":             "schema_mismatch",
            "expected_version": exc.expected,
            "actual_version":   exc.actual,
        }), 422
    if isinstance(exc, SpectraJsonMalformedError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "malformed",
        }), 400
    if isinstance(exc, SpectraJsonFieldError):
        return jsonify({
            "ok":    False,
            "error": str(exc),
            "kind":  "field",
        }), 400
    # Generic SpectraJsonError catch-all.
    return jsonify({
        "ok":    False,
        "error": str(exc),
        "kind":  "parse_error",
    }), 400


__all__ = ["bp"]
