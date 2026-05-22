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

from flask import Blueprint, make_response, render_template

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
