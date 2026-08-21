"""Spectra blueprint -- the page and the ARTIFACT reader.

Routes:

    GET  /spectrum-calculation          the page (rendered template).  Renamed
                                        from /spectra in Phase B.5 (2026-06-07);
                                        legacy redirect deleted.
    POST /api/spectra/load              parse an uploaded <job>.spectra.json
                                        (or read JSON body) into the typed
                                        SpectraResults shape

THE PRODUCER LEFT THIS FILE (spectra-migration plan P3, 2026-08-21):
``POST /api/spectra/render`` and ``GET /api/build/schema/spectra``
retired with the old generator.  The tab's compute half renders the
CATALOGUE's vibration form (``GET /api/build/schema/pyscf?calculation=
vibration``, build.py) and hands the description to Task setup
(``POST /api/task-setup/handover``); the deck is written by ``prep``
on the machine that runs it (docs/execution/script-preparation.md).

Response shapes mirror Build's: every endpoint returns
``{"ok": True, ...}`` on success or ``{"ok": False, "error":
"<msg>"}`` on failure.

The load path:

  * Multipart: ``file=`` form-data field with the .spectra.json
    bytes.
  * JSON: ``{"path": "..."}`` to read from a local path (useful
    when the user runs the deck and wants to load results from
    a known job dir without manually uploading).
  * JSON: ``{"json": {...}}`` to parse a dict in-memory (for
    programmatic callers; matches parse_spectra_json_dict).
"""
from __future__ import annotations

import json
import typing
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request


from molbuilder.sidecars.spectra import (
    SpectraJsonError,
    SpectraJsonFieldError,
    SpectraJsonMalformedError,
    SpectraJsonNotFoundError,
    SpectraJsonSchemaError,
    parse_spectra_json,
    parse_spectra_json_dict,
)
from molbuilder.spectra.results import SpectraResults, motion_share_by_element
from molbuilder.structure import Structure


bp = Blueprint("spectra", __name__)




# ===================================================================== #
# Page route                                                            #
# ===================================================================== #


@bp.route("/spectrum-calculation")
def spectrum_calculation_page():
    """Render the Spectrum calculation tab.

    Heavy lifting is done by API routes elsewhere; this just hands
    back the template.  On load the tab's JS renders the CATALOGUE's
    vibration form (``/api/build/schema/pyscf?calculation=vibration``,
    build.py) through the shared form renderer.
    """
    return render_template("spectra.html")


# Sidecar application moved to web/blueprints/_shared.py
# (apply_sidecar_if_possible) so Build's /api/build/fdf +
# /api/build/pyscf can import from the shared module instead of
# reaching into the Spectra blueprint.  Backwards-compatible alias
# preserved here so any in-flight branch importing from this module
# keeps working.


# ===================================================================== #
# /api/spectra/load  -- parse a .spectra.json into typed results        #
# ===================================================================== #


def _loaded(results):
    """The one success reply for /api/spectra/load — the typed results as a
    dict, plus what only the server can work out.

    WHY THE PAYLOAD IS NOT SIMPLY ``to_dict()``.  ``to_dict`` is the ON-DISK
    format: ``from_dict(to_dict(x)) == x``, byte-equal modulo float
    formatting, and the emitter writes it.  Derived facts must not leak into
    it or the file grows fields the schema never declared.  So this wraps it:
    the file's own fields, unchanged, plus a computed one.

    WHAT IS ADDED, and why it cannot be computed in the browser.
    ``motion_share_by_element`` says which atoms a mode belongs to -- 91%
    carbon for a ring stretch -- and that needs atomic masses, because
    hydrogen has the largest displacement in almost every mode of an organic
    molecule while carrying almost none of the motion.  The browser has no
    masses, the .spectra.json carries none, and shipping a periodic table
    into JavaScript to fix that would be a second copy of a table ASE already
    provides (``chemistry.atomic_mass``).  The server has it, so the server
    answers.

    COMPUTED AT LOAD, NOT STORED.  Every result already written -- including
    the ones on disk right now -- gains the field the moment it is opened; a
    schema bump would have left them showing nothing until re-run.  It costs
    one pass over each eigenvector.

    A mode whose shares cannot be worked out (an element ASE does not know)
    is served without the field rather than failing the load: the panel drops
    one clause, and the spectrum still opens.
    """
    payload = results.to_dict()
    elements = (payload.get("equilibrium") or {}).get("elements") or []
    # An empty free-atom list is "not recorded", not "no atom is free" -- a
    # result that never tracked the partition has one eigenvector row per atom,
    # which is what passing None means downstream.
    free = payload.get("free_atom_idxs") or None
    for mode in payload.get("modes") or []:
        rows = mode.get("eigenvector_canonical") or mode.get("eigenvector_display")
        if not rows or not elements:
            continue
        try:
            mode["motion_share_by_element"] = motion_share_by_element(
                elements, rows, free)
        except (ValueError, KeyError):
            pass
    return jsonify({"ok": True, "results": payload})


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
        return _loaded(results)

    # ---- JSON body (path or inline JSON) ------------------------- #
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    inline = body.get("json")
    if path:
        try:
            results = parse_spectra_json(path)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return _loaded(results)
    if inline is not None:
        if not isinstance(inline, dict):
            return _err_load(SpectraJsonMalformedError(
                f"'json' field must be an object, got {type(inline).__name__}"
            ))
        try:
            results = parse_spectra_json_dict(inline)
        except SpectraJsonError as exc:
            return _err_load(exc)
        return _loaded(results)

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
