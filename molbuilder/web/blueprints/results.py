"""Results blueprint -- the unified post-merge inspector page.

Routes:

    GET /results                            the results page (dispatch shell)
    GET /partials/trajectory-inspector      rendered trajectory inspector
                                            HTML, for in-place mount inside
                                            ``#inspector-host`` (consumed by
                                            ``lib/inspectors/trajectory.js``)
    GET /partials/spectra-inspector         rendered spectra inspector HTML
                                            (consumed by
                                            ``lib/inspectors/spectra.js``)
    GET /partials/selection-panel           rendered selection-panel HTML
                                            (consumed by /modify; later
                                            /spectra and any other tab that
                                            needs atom selection)

No ``/api/results/*`` endpoints in v1: the page reuses
``/api/watch/*`` and ``/api/spectra/*`` for trajectory and spectra
loading respectively, and ``/api/files/*`` for the source / structure
previews.

The blueprint is a clear seam for future ``/api/results/*`` endpoints
(e.g., "summarise this file's metadata for the dispatch label") --
they land here without touching the other tabs' blueprints.

Spec: ``docs/protocols/results-tab.md``.
"""
from __future__ import annotations

import re
from pathlib import Path

from flask import Blueprint, jsonify, make_response, render_template, request

bp = Blueprint("results", __name__)


# Stem charset for /api/results/bundle.  Same shape as the
# region-label charset (_SAFE_REGION_LABEL_RE in _shared.py) +
# the wrapper-basename charset in runwrap.py: ``[A-Za-z0-9._-]``,
# 1-64 chars.  Keeps shell-special / Unicode / whitespace out of
# every downstream consumer (filesystem rename, sidebar list view,
# the next tab's form fields).
_BUNDLE_STEM_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@bp.route("/results")
def results_page():
    """Render the results page.  All dispatch happens client-side
    via ``static/results/viewer.js`` subscribing to the projects-
    sidebar selection state."""
    return render_template("results.html")


# --------------------------------------------------------------------- #
#  Server-rendered partials                                             #
# --------------------------------------------------------------------- #
#
# Inspectors are mounted into a single ``#inspector-host`` element
# (registry-owned, see lib/inspectors/registry.js).  Small inspectors
# build their DOM via createElement; the trajectory inspector's DOM
# is large (~9.5 KB) and is the single source of truth shared with
# /watch via ``_trajectory_inspector.html``.  Rather than fork the
# markup into JS, the registry inspector fetches the partial here
# and assigns it to its host's innerHTML -- same-origin, autoescaped
# Jinja render, no user input, so safe.


@bp.route("/partials/trajectory-inspector")
def partial_trajectory_inspector():
    """Return the rendered trajectory inspector partial as HTML.

    Source: ``templates/_trajectory_inspector.html``.  Same partial
    is included server-side by ``watch.html``; this endpoint exists
    so ``/results`` can swap the inspector in client-side without
    duplicating the markup.

    Cache: ``private, max-age=300`` -- the partial is static
    content that only changes on template edits, but capping the
    cache at 5 minutes keeps an after-the-fact deploy from leaving
    stale clients running indefinitely.  No user data in the
    response so ``private`` is correct (the response can be cached
    only by the browser, not by intermediates).
    """
    html = render_template("_trajectory_inspector.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@bp.route("/partials/spectra-inspector")
def partial_spectra_inspector():
    """Return the rendered spectra inspector partial as HTML.

    Source: ``templates/_spectra_inspector.html``.  Same partial is
    included server-side by ``spectra.html`` (transitionally, until
    step 2.5 drops the inspect-side from /spectra); this endpoint
    exists so ``/results`` can swap the inspector in client-side
    without forking the markup.

    Cache + content-type semantics identical to the trajectory
    partial endpoint -- intentional, so the inspector wrappers
    share an HTTP contract.
    """
    html = render_template("_spectra_inspector.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@bp.route("/partials/selection-panel")
def partial_selection_panel():
    """Return the rendered selection-panel partial as HTML.

    Source: ``templates/_selection_panel.html``.  The panel is the L4
    UI of the atom-selection system (see
    ``molbuilder/web/blueprints/selection.py``); it's included by
    /modify (via fetch into the page's selection host) and will later
    be reused on /spectra to replace the three frozen-atoms inputs.

    Cache + content-type semantics identical to the inspector
    partials -- same HTTP contract for every partial endpoint.
    """
    html = render_template("_selection_panel.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


# --------------------------------------------------------------------- #
#  /api/results/bundle -- Step 3 PR-E (task #492)                       #
# --------------------------------------------------------------------- #
#
# Run-bundle handoff endpoint.  Reads a finished SIESTA/PySCF run
# dir, fuses final coords + in-body labels into a RunBundle, then
# materialises the pair to ``<target_dir>/<stem>.xyz`` +
# ``<target_dir>/<stem>.molstruct.json``.  The next tab's existing
# .xyz load path (which honors apply_sidecar_if_possible) picks the
# pair up unchanged.
#
# Contract: ``docs/protocols/bundle-contract.md`` § 5.2 + § 7
# (Storage and projects-sidebar integration).  This endpoint is
# the picker-roots security boundary called out in § 7.1.


@bp.route("/api/results/bundle", methods=["POST"])
def api_results_bundle():
    """POST /api/results/bundle

    Body::

        {
          "run_dir":    "<abs path to finished SIESTA/PySCF run dir>",
          "target_dir": "<abs path to where the bundle lands>",
          "stem":       "<basename for the .xyz + .molstruct.json pair>",
          "overwrite":  false                  // default
        }

    Responses (uniform envelope per design.md):

    Success::

        {
          "ok":                true,
          "xyz_path":          "...",
          "sidecar_path":      "...",
          "source_engine":     "siesta" | "pyscf",
          "final_coords_from": "xv" | "fdf-initial" | "py-opt" | "py-initial",
          "n_atoms":           N,
          "regions":           [<region label>, ...],
          "frozen_atoms_count":F,
          "notes":             [<diagnostic string>, ...]
        }

    Errors:

    * 400 ``missing/invalid <field>`` -- bad request shape.
    * 400 ``stem must be [A-Za-z0-9._-]+ (1-64 chars)`` -- charset rejection.
    * 400 -- BundleError (no script in run_dir, both engines present,
            atom-count mismatch, overwrite=False with existing target).
    * 400 -- _PickerError (path outside roots, ``..`` traversal).
    * 404 -- run_dir does not exist or is not a directory.
    """
    # Defer the imports until the handler runs so the blueprint
    # import-time graph stays thin (results.py is loaded at startup
    # by every page; script_bundle pulls in parsers + structure).
    from molbuilder.script_bundle import (
        BundleError, assemble_from_run_dir, write_bundle_as_handoff,
    )
    from .files import _resolve_within_roots, _PickerError

    body = request.get_json(silent=True) or {}
    run_dir    = (body.get("run_dir")    or "").strip()
    target_dir = (body.get("target_dir") or "").strip()
    stem       = (body.get("stem")       or "").strip()
    overwrite  = bool(body.get("overwrite", False))

    if not run_dir:
        return jsonify({"ok": False, "error": "missing 'run_dir'"}), 400
    if not target_dir:
        return jsonify({"ok": False, "error": "missing 'target_dir'"}), 400
    if not stem:
        return jsonify({"ok": False, "error": "missing 'stem'"}), 400
    if not _BUNDLE_STEM_RE.match(stem):
        return jsonify({
            "ok": False,
            "error": (
                "'stem' must be [A-Za-z0-9._-]+ (1-64 chars); got "
                f"{stem!r}.  No whitespace, no shell-special chars."
            ),
        }), 400

    # Picker-roots security boundary (bundle-contract.md § 7.1).
    # Both paths MUST resolve inside an allowed root; otherwise a
    # hostile client could read / write anywhere on the host.
    try:
        run_path    = _resolve_within_roots(run_dir)
        target_path = _resolve_within_roots(target_dir)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    if not run_path.is_dir():
        return jsonify({
            "ok": False,
            "error": (
                f"run_dir does not exist or is not a directory: "
                f"{run_path}.  Pick a finished SIESTA/PySCF run dir."
            ),
        }), 404
    # target_path is allowed to not exist; the materialiser creates
    # it (bundle-contract.md § 7.2).

    try:
        bundle = assemble_from_run_dir(run_path)
    except BundleError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    try:
        xyz_path, sidecar_path = write_bundle_as_handoff(
            bundle, target_path, stem=stem, overwrite=overwrite,
        )
    except BundleError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({
        "ok":                 True,
        "xyz_path":           str(xyz_path),
        "sidecar_path":       str(sidecar_path),
        "source_engine":      bundle.source_engine,
        "final_coords_from":  bundle.final_coords_from,
        "n_atoms":            len(bundle.structure.elements),
        "regions":            sorted(bundle.regions.keys()),
        "frozen_atoms_count": len(bundle.frozen_atoms),
        "notes":              list(bundle.notes),
    })
