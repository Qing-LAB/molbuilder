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
                                            (consumed by /modify; later
                                            /spectra and any other tab that
                                            needs atom selection)

For trajectory / spectra / preview LOADING the page reuses the other
tabs' endpoints -- ``/api/watch/*`` and ``/api/spectra/*`` for trajectory
and spectra, ``/api/files/*`` for the source / structure previews -- so
those are not re-exposed here.  ``/api/results/*`` is reserved for
results-only operations (today: ``bundle``; a future "summarise this
file's metadata for the dispatch label" would land here too), which stay
in this blueprint without touching the other tabs' blueprints.

Spec: ``docs/web/results.md``.
"""
from __future__ import annotations

import re

from flask import Blueprint, jsonify, make_response, render_template, request

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

    Source: ``templates/_spectra_inspector.html``.  ``/results`` swaps
    the inspector in client-side through this endpoint; the
    server-side include that ``spectra.html`` carried transitionally
    left with step 2.5 (the standalone tab gates its inspect side on
    ``hasInspectSide`` and no longer embeds the partial).

    Cache + content-type semantics identical to the trajectory
    partial endpoint -- intentional, so the inspector wrappers
    share an HTTP contract.
    """
    html = render_template("_spectra_inspector.html")
    resp = make_response(html)
    resp.headers["Content-Type"]  = "text/html; charset=utf-8"
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


# --------------------------------------------------------------------- #
#  /api/results/bundle stood here (Step 3 PR-E, task #492) until        #
#  2026-08-29.  Calculation-to-calculation passing is RETIRED (user     #
#  ruling): a calculation that builds on a finished result CITES it --  #
#  the transport composite resolves its junction citation and prep      #
#  does the fuse (parse the .XV, overlay the labels, sort, gate:        #
#  transport/compose.py) -- rather than receiving a bundled copy.       #
# --------------------------------------------------------------------- #


@bp.route("/api/results/contract", methods=["GET"])
def api_results_contract():
    """The electronic contract recorded by the deck beside a structure.

    ``?path=`` names a structure file (or its directory), tree-relative;
    the answer is the ``info.calculation`` key of the block that
    directory answers for itself, or ``null`` when there is none (no
    deck, several decks, a deck stating nothing).  The Results tab's
    structure inspector calls this after a load and records the answer
    through the viewer's ``data.info`` door, so an export carries it
    (`plans/structure-info-plan.md` I5).

    It asks ``run_info_for_dir`` -- the ONE composer of "what does this
    directory say about itself" -- rather than the ``contract_of``
    extractor beneath it, so this door and the trajectory load door
    cannot come to disagree about what a directory records."""
    from pathlib import Path

    from flask import jsonify, request

    from molbuilder.parse.dirs.run_info import run_info_for_dir
    from .files import _PickerError, _resolve_within_roots

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    try:
        p = _resolve_within_roots(raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    directory = Path(p) if Path(p).is_dir() else Path(p).parent
    info = run_info_for_dir(directory) or {}
    return jsonify({"ok": True, "calculation": info.get("calculation")})
