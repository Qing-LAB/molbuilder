"""Results blueprint -- the unified post-merge inspector page.

Routes:

    GET /results                            the results page (dispatch shell)
    GET /partials/trajectory-inspector      rendered trajectory inspector
                                            HTML, for in-place mount inside
                                            ``#inspector-host`` (consumed by
                                            ``lib/inspectors/trajectory.js``)
    GET /api/export                         a calculation's results as they
                                            stand, as NetCDF or as tables
                                            (`model/datafiles.md` § 5a)
    GET /api/export/offer                   what a calculation's export
                                            takes, which the page's one
                                            Export bar is drawn from
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


@bp.route("/api/export", methods=["GET"])
def api_export():
    """A CALCULATION'S RESULTS AS THEY STAND, for the person to keep
    (`model/datafiles.md` § 5a) -- ``?path=`` its folder or a file or folder
    below it (the result a viewer shows), ``format=`` ``netcdf`` or ``csv``,
    and what its kind's export asks, by name (`kind_table.export_asks`) --
    one attachment, composed on request by its kind's export
    (`kind_table.export_of`), nothing written into the calculation; what it
    holds without said in ``X-Molbuilder-Said``.  400 names what is
    refused."""
    from molbuilder.calculation import NotACalculation, at
    from molbuilder.kind_table import export_asks_of, export_of
    from molbuilder.ncdata import ExportError, export_format
    from ._shared import sent_export
    from .files import _existing, _resolve_within_roots
    path = _existing(_resolve_within_roots(str(request.args.get("path")
                                               or "")))
    try:
        calc = at(path)
        kind = calc.task.calculation
        fmt = export_format(str(request.args.get("format") or ""))
        got = export_of(kind).export(
            calc, fmt, folder=path if path.is_dir() else path.parent,
            **export_asks_of(kind, request.args, fmt=fmt))
    except (NotACalculation, ExportError, ValueError, OSError) as exc:
        return err(str(exc), 400)
    return sent_export(got)


@bp.route("/api/export/offer", methods=["GET"])
def api_export_offer():
    """WHAT A CALCULATION'S EXPORT TAKES (`model/datafiles.md` § 5a) --
    ``?path=`` its folder or a file or folder below it: ``offer`` the forms
    (`ncdata.EXPORT_FORMATS`, each with its words), its kind's asks' rows and
    the document that says how to read the export in Python, which the
    page's one Export bar is drawn from (`projects.exportBar`) -- ``null``
    for a path in no calculation, or in one whose kind exports nothing: an
    answer, not a refusal.  A description that does not read is refused
    with its way on, as the export is (400); a kind whose export does not
    resolve is the table's defect, said (`kind_table.export_of`, 500).
    ``offer.calculation`` -- the calculation's folder -- is what the page
    keeps one bar across."""
    from dataclasses import asdict
    from molbuilder.calculation import NotACalculation, at
    from molbuilder.kind_table import KindTableError, export_of, kind as kind_of
    from molbuilder.ncdata import EXPORT_FORMATS
    from .files import _existing, _resolve_within_roots
    path = _existing(_resolve_within_roots(str(request.args.get("path")
                                               or "")))
    try:
        calc = at(path)
    except NotACalculation:
        return jsonify({"ok": True, "offer": None})
    except ValueError as exc:
        return err(str(exc), 400)
    try:
        kind = kind_of(calc.task.calculation)
        if kind.export is None:
            return jsonify({"ok": True, "offer": None})
        export_of(calc.task.calculation)
    except KindTableError as exc:
        return err(str(exc), 500)
    return jsonify({"ok": True, "offer": {
        "calculation": str(calc.base),
        "forms": [{"name": name, "words": words}
                  for name, words in EXPORT_FORMATS.items()],
        "asks": [asdict(r) for r in kind.export_asks],
        "doc": kind.export_doc}})


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
