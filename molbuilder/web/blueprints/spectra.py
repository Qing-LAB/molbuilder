"""Spectra blueprint -- harmonic frequencies + Raman + per-mode ES routes.

Routes (spec § 10):

    GET  /spectrum-calculation          the page (rendered template).  Renamed
                                        from /spectra in Phase B.5 (2026-06-07);
                                        legacy redirect deleted.
    GET  /api/build/schema/spectra      form-rendering schema (SpectraConfig).
                                        The ``spectra`` token in the schema URL
                                        is the engine name (alongside siesta,
                                        pyscf), unchanged by the page rename.
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
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from ._shared import (
    config_from_params as _config_from_params,
    dataclass_to_form_schema as _dataclass_to_form_schema,
    issues_to_json as _issues_to_json,
)
from .files import _PickerError, _resolve_within_roots

from molbuilder.config.spectra import SpectraConfig
from molbuilder.validation import validate as _validate
from molbuilder.sidecars import molstruct as _molstruct_json
from molbuilder.sidecars.spectra import (
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


@bp.route("/spectrum-calculation")
def spectrum_calculation_page():
    """Render the Spectrum calculation tab.

    Heavy lifting (form schema, render, load) is done by the API
    routes below; this just hands back the template.  The JS does
    a ``/api/build/schema/spectra`` call on page load to populate
    the form.
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

    Optional ``?structure_path=...`` query: when supplied AND a
    .molstruct.json sidecar sits next to the XYZ AND the sidecar
    carries a non-empty ``frozen_atoms`` set, the schema's
    ``frozen_indices`` field default is overridden with the
    comma-separated form of those indices.  This is the option-2
    "sidecar feeds defaults, form is authoritative" contract for
    the frozen-atom boundary condition: the sidecar's data appears
    pre-filled in the form so the user can SEE what's frozen
    before generating the script; editing the field overrides the
    pre-fill (form values are authoritative at render time).

    Sidecar load errors (missing structure, malformed sidecar,
    atom-count mismatch) are non-fatal here -- the endpoint
    returns the static schema with an additional ``notice`` field
    explaining why the pre-fill was skipped, so the user is
    informed without blocking the form.  The path is validated
    via ``files._resolve_within_roots`` so this endpoint inherits
    the same allow-list as /api/files/*.
    """
    schema = _dataclass_to_form_schema(SpectraConfig, "s")
    response: Dict[str, Any] = {"ok": True, "schema": schema}

    structure_path = request.args.get("structure_path") or ""
    if structure_path:
        notice = _seed_frozen_indices_from_sidecar(schema, structure_path)
        if notice:
            response["notice"] = notice
    return jsonify(response)


def _seed_frozen_indices_from_sidecar(
    schema: Dict[str, Any],
    structure_path: str,
) -> Optional[str]:
    """Override ``frozen_indices.default`` in the schema with the
    sidecar's frozen-atom indices for ``structure_path``.

    Returns ``None`` on success (default override applied OR
    nothing to override).  Returns a short, user-facing string
    explaining why the pre-fill was skipped (e.g. "sidecar atom
    count doesn't match the structure -- re-export from the Structure tab")
    when something went wrong but in a recoverable way.

    The function mutates ``schema`` in place; no return on success.
    """
    try:
        resolved = _resolve_within_roots(structure_path)
    except _PickerError as exc:
        return f"structure_path rejected: {exc.message}"
    if not resolved.exists():
        return f"structure_path not found: {resolved.name}"
    ext = resolved.suffix.lower()
    if ext not in (".xyz", ".pdb"):
        return (f"structure_path must be .xyz or .pdb "
                f"(got {resolved.suffix!r})")

    sidecar_path = _molstruct_json.sidecar_path_for(resolved)
    if not sidecar_path.exists():
        # No sidecar = nothing to seed, no notice needed.
        return None

    try:
        sidecar_data = _molstruct_json.load(sidecar_path)
    except _molstruct_json.MolstructJsonError as exc:
        return (f"sidecar at {sidecar_path.name} could not be read "
                f"({exc}); frozen-atom field not pre-filled")

    frozen = list(sidecar_data.get("frozen_atoms") or [])
    if not frozen:
        return None

    # Validate against the actual atom count so a stale sidecar
    # doesn't pre-fill indices that no longer point at real atoms.
    # ``read_text`` defaults to strict UTF-8; a binary or garbled
    # XYZ/PDB file raises UnicodeDecodeError (NOT OSError) -- catch
    # both so this code path never 500s the schema endpoint.
    try:
        text = resolved.read_text()
        struct = (Structure.from_xyz(text) if ext == ".xyz"
                  else Structure.from_pdb(text))
    except (ValueError, IndexError, OSError, UnicodeDecodeError) as exc:
        return (f"could not parse {resolved.name} to check sidecar "
                f"({exc}); frozen-atom field not pre-filled")
    if sidecar_data.get("n_atoms_total") != struct.n_atoms:
        return (f"sidecar atom count ({sidecar_data.get('n_atoms_total')}) "
                f"differs from structure ({struct.n_atoms}); re-export "
                f"the sidecar from the Structure tab.  Frozen-atom field not pre-filled.")

    # Find the frozen_indices field in the schema (the section is
    # "Frozen atoms"; the field name is "frozen_indices") and
    # override its default with a comma-separated string.  Match
    # the renderer's text-input format so the form value matches
    # what coerce_to_field_type expects on submit.
    formatted = ", ".join(str(i) for i in sorted(set(frozen)))
    for section in schema.get("sections", []):
        for field in section.get("fields", []):
            if field.get("name") == "frozen_indices":
                field["default"] = formatted
                return None
    # Schema topology unexpectedly missing the field.  This means
    # SpectraConfig was refactored without updating this code path
    # -- a programmer error, not a user-facing one.  Return a
    # notice so the user gets SOME signal (the form will silently
    # NOT pre-fill); the unit test
    # ``test_sidecar_frozen_atoms_prefills_field_default`` catches
    # the regression at CI time.
    return ("frozen_indices field unexpectedly missing from the form "
            "schema; pre-fill skipped (this is a programmer bug)")


# Sidecar application moved to web/blueprints/_shared.py
# (apply_sidecar_if_possible) so Build's /api/build/fdf +
# /api/build/pyscf can import from the shared module instead of
# reaching into the Spectra blueprint.  Backwards-compatible alias
# preserved here so any in-flight branch importing from this module
# keeps working.
from ._shared import apply_sidecar_if_possible as _apply_sidecar_if_possible


# ===================================================================== #
# /api/spectra/render  -- generate the script                           #
# ===================================================================== #


@bp.route("/api/spectra/render", methods=["POST"])
def api_spectra_render():
    """Render the runnable ``<job>.spectra.py`` for a Structure + params.

    Body (JSON)::

        {
          "structure_text": "<XYZ or PDB text content>",
          "params":         {<SpectraConfig dict>},
          "structure_path": "<optional path to the on-disk .xyz/.pdb>",
          "prior_path":     "<optional path to a prior .spectra.json>"
        }

    The ``structure_text`` field accepts either XYZ or PDB content;
    the server sniffs the format by the first line (integer ->
    XYZ; non-integer -> PDB).  Renamed 2026-05-22 from the
    misleading ``"xyz"`` field name.

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

    The ``structure_path`` (optional) lets the server read the
    .molstruct.json sidecar next to the on-disk XYZ and apply its
    ``frozen_atoms`` + ``regions`` to the Structure BEFORE preflight
    runs.  This is what makes stage-3 of the three-stage contract
    (design.md "Sidecar-driven boundary conditions") work end-to-end:
    Pattern A (divergence WARN) and Pattern B (unrecognized-label
    WARN) need ``struct.frozen_atoms`` / ``struct.regions`` to be
    populated -- which only happens when the sidecar is applied.
    If the structure_path is omitted (e.g. the user pasted raw XYZ
    text without a sidebar pick), the sidecar guards quietly stay
    inert; preflight still validates the inline form fields.
    """
    body = request.get_json(silent=True) or {}
    structure_text: Optional[str] = body.get("structure_text")
    params: Dict[str, Any] = body.get("params") or {}
    structure_path: Optional[str] = body.get("structure_path")
    prior_path: Optional[str] = body.get("prior_path")

    if not structure_text:
        return jsonify({"ok": False,
                        "error": "no 'structure_text' provided"}), 400

    # Parse the structure text.  Sniff XYZ vs PDB by the first
    # line (matches /api/build/load's auto-detect logic): an
    # integer = XYZ; anything else falls through to PDB.  Errors
    # surface as 400.
    try:
        first_line = (structure_text.lstrip().splitlines()[0]
                      if structure_text.strip() else "")
        if first_line.strip().isdigit():
            struct = Structure.from_xyz(structure_text, title="from-browser")
        else:
            # PDB / unknown: dispatch to from_pdb, which itself
            # raises if there's no ATOM/HETATM in the input.
            struct = Structure.from_pdb(structure_text, title="from-browser")
    except (ValueError, IndexError) as exc:
        return jsonify({"ok": False,
                        "error": f"could not parse structure text: {exc}"}), 400

    # 2026-06-14 contract update: prefer in-body labels (the
    # viewer-is-truth contract).  When the body carries
    # ``frozen_atoms`` / ``regions``, apply them directly with no
    # disk re-read; fall back to disk sidecar only when neither
    # in-body key is present (back-compat for the path-driven
    # flow).  See _shared.apply_labels_to_struct docstring.
    from ._shared import apply_labels_to_struct
    sidecar_notice: Optional[str] = apply_labels_to_struct(struct, body)
    from ._shared import apply_periodicity_from_body
    struct = apply_periodicity_from_body(struct, body)

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

    # SINGLE science gate (V1/V2, 2026-07): validate() runs
    # validate_geometry + _validate_config_metadata + the registered
    # SpectraConfig validator (the render-gate SCIENCE, via the engine's
    # render_checks).  The selector-availability check is preflight-only
    # UX (a top_n script is valid to emit; it resolves at run time), so it
    # is added HERE, not in the render gate.
    issues = (
        list(_validate(struct, cfg, prior=prior))
        + list(engine.selector_checks(struct, cfg, prior=prior))
    )
    if prior_warn is not None:
        issues.append(prior_warn)
    # Surface the sidecar-application failure (if any) so the user
    # sees that their sidecar's frozen / region data is NOT a
    # factor in this run -- no silent absorption.
    if sidecar_notice is not None:
        from molbuilder.issues import Issue
        issues.append(Issue(
            severity="warn",
            message=sidecar_notice,
            where="structure_path",
        ))

    # Block render on any error-severity issue.
    # web-api.md § 1.6 (b) scientific advisory: validator hard-fail
    # blocks emission, but the HTTP exchange succeeded — the server
    # ran the check and is reporting back.  HTTP 200 explicit so
    # the intent is visible at the call site.
    errors_only = [i for i in issues if i.severity == "error"]
    if errors_only:
        return jsonify({
            "ok":          False,
            "error":       "preflight failed; see issues",
            "issues":      _issues_to_json(issues, cfg=cfg),
            "errors_only": _issues_to_json(errors_only, cfg=cfg),
        }), 200

    # Render the script + Methods text + bibliography keys.
    try:
        script = engine.render_script(struct, cfg)
    except Exception as exc:
        # Unhandled render error -> 500 with the message + any
        # warns we accumulated so far so the user can see context.
        return jsonify({
            "ok":     False,
            "error":  f"render failed: {exc}",
            "issues": _issues_to_json(issues, cfg=cfg),
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
        "issues":            _issues_to_json(issues, cfg=cfg),
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
    except SpectraJsonNotFoundError:
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
