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

For trajectory / spectra / preview LOADING the page reuses the other
tabs' endpoints -- ``/api/watch/*`` and ``/api/spectra/*`` for trajectory
and spectra, ``/api/files/*`` for the source / structure previews -- so
those are not re-exposed here.  ``/api/results/*`` is reserved for
results-only operations (``contract``, ``dir``), which stay
in this blueprint without touching the other tabs' blueprints.

Spec: ``docs/web/results.md``.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, make_response, render_template, request
from ._shared import err

bp = Blueprint("results", __name__)




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
# is large (~9.5 KB) and lives in ``_trajectory_inspector.html``.
# Rather than fork the
# markup into JS, the registry inspector fetches the partial here
# and assigns it to its host's innerHTML -- same-origin, autoescaped
# Jinja render, no user input, so safe.


@bp.route("/partials/trajectory-inspector")
def partial_trajectory_inspector():
    """Return the rendered trajectory inspector partial as HTML.

    Source: ``templates/_trajectory_inspector.html``.  This endpoint
    is how ``/results`` swaps the inspector in client-side.

    Cache: ``private, no-cache`` -- the browser asks before reusing its
    copy, as it does for every script and stylesheet (``app.py``,
    ``SEND_FILE_MAX_AGE_DEFAULT``): the partial and the scripts that
    query it change together, so a copy kept without asking would put an
    update's scripts under the last release's controls.  ``private``: only
    the browser keeps it, never an intermediary.
    """
    html = render_template("_trajectory_inspector.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, no-cache"
    return resp


@bp.route("/partials/spectra-inspector")
def partial_spectra_inspector():
    """Return the rendered spectra inspector partial as HTML.

    Source: ``templates/_spectra_inspector.html``.  ``/results`` swaps
    the inspector in client-side through this endpoint.

    Cache + content-type semantics identical to the trajectory
    partial endpoint -- intentional, so the inspector wrappers
    share an HTTP contract.
    """
    html = render_template("_spectra_inspector.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, no-cache"
    return resp


@bp.route("/api/results/dir", methods=["GET"])
def api_results_dir():
    """**What is in this run directory, and which file should open?**

    The HTTP surface over the run door's answer for the one folder,
    `runs.folder_answer` (`execution/architecture.md` § 3.2; `model/parse.md`
    § 5.0, § 5.5), served whole, and beside it the views the folder is read
    with, `runstatus.views_of` -- ``ladder``, ``points``, ``point``.  The
    route lists no field of either and decides no view, so a field a door
    adds reaches the page with nothing changed here (`web/results.md`
    § 0.2).

    **WHY THIS EXISTS.**  The browser is the one consumer that cannot import
    Python, so it is served the directory's own answer rather than deciding
    from the filename, in JavaScript, what the backend already owns.
    """
    from molbuilder.jobset.runstatus import views_of
    from molbuilder.runs import folder_answer
    from .files import _existing, _resolve_within_roots

    raw = str(request.args.get("path") or "")
    if not raw:
        return err("no path given", 400)
    p = _existing(_resolve_within_roots(raw))
    directory = p if p.is_dir() else p.parent
    # `run_dir`: the folder answered, as the server resolves it -- the
    # picker's spelling of the folder bound (`web-api.md`).
    return jsonify({"ok": True, "run_dir": str(directory),
                    **folder_answer(directory), **views_of(directory)})
