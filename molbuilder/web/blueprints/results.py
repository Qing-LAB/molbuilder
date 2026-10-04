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

    Source: ``templates/_trajectory_inspector.html``.  This endpoint
    is how ``/results`` swaps the inspector in client-side (the /watch
    page that included the partial server-side is gone).

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
    """What the run directory beside a structure records about it.

    ``?path=`` names a structure file (or its directory), tree-relative;
    the answer carries the two keys of the block that directory answers
    for itself -- ``calculation``, the electronic contract its deck
    states, and ``relaxation``, what the run did to the geometry it left
    (`model/parse.md` § 5b, § 5b.1) -- each ``null`` when there is nothing
    to say.  The Results tab's structure inspector calls this after a load
    and records each answer through the viewer's ``data.info`` door, so an
    export carries them (`archive/2026-09-01-structure-info-plan.md` I5).

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
    # THE RUN'S RECORD IS OF THE FILE THE RESULTS TAB OPENS in that folder
    # -- the run door's choice, handed down because `parse/` cannot ask
    # the door (`model/parse.md` § 5b.1).
    from molbuilder.runs import openable
    output, _trail = openable(directory)
    info = run_info_for_dir(directory, output=output) or {}
    return jsonify({"ok": True, "calculation": info.get("calculation"),
                    "relaxation": info.get("relaxation")})


@bp.route("/api/results/dir", methods=["GET"])
def api_results_dir():
    """**What is in this run directory, and which file should open?**

    The HTTP surface over the run door's directory answer,
    `runs.folder_answer` (`execution/architecture.md` § 3.2; `model/parse.md`
    § 5.0, § 5.5) -- and, for a calculation root, its ladder.

    **WHY THIS EXISTS.**  The browser is the one consumer that cannot import
    Python, and nothing served it the directory's own answer: the Results
    picker listed through `/api/files/list` (content-blind — `{name, kind,
    size, mtime}`) and then decided FOUR things the backend already owns,
    from the filename, in JavaScript.  Measured 2026-09-18 over 110 real run
    directories: *which file to open* differed from `openable_in` on **18 of
    96**; **13** files were offered that no parser can read and **155** that
    a parser handles were unreachable; the engine was guessed from the
    suffix; and the run-file grammar was re-implemented as a JS regex.

    **PER FILE the server says what the file IS** -- ``role``, ``label`` and
    ``stage``, read back with its run's label, ``parser`` from the registry
    (``null`` when nothing claims it — the browser must not offer it), and
    ``about``: the catalogue's line for a file molbuilder writes, or *not
    written by molbuilder* (`web/results.md` § 3b).  ``openable`` is the
    door's own pick, the run the folder speaks for.
    """
    from pathlib import Path

    from flask import jsonify, request

    from molbuilder import calcdirs
    from molbuilder.runs import folder_answer
    from .files import _PickerError, _resolve_within_roots

    raw = str(request.args.get("path") or "")
    if not raw:
        return jsonify({"ok": False, "error": "no path given"}), 400
    try:
        p = _resolve_within_roots(raw)
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status
    directory = Path(p) if Path(p).is_dir() else Path(p).parent
    if not directory.is_dir():
        return jsonify({"ok": False,
                        "error": f"{raw}: not a directory"}), 404

    # THE RUN DOOR ANSWERS what this folder is and holds -- its place, the run
    # it speaks for, that run's result, state and record, and what each file
    # is (`runs.folder_answer`).  It was `parse.dirs`' `JobDirParser`, read
    # through `parse_dir`, until 2026-10-04: below the floor that may know a
    # calculation (plan B11, B14).
    got = folder_answer(directory)
    attempts = list(got["attempts"])
    place = got["place"]
    ladder = None
    # A CALCULATION ROOT HAS A LADDER (`web/results.md` § 2.4; `plan.md`
    # § 5c.3 c-d): N rungs, each a run directory below it -- `jobset_status`'s
    # answer, the one the CLI's `status` verb prints, CONSUMED here, never
    # copied.  Its rows are the description's stages (`job-system.md` § 5.3),
    # the ones not prepped yet among them: a transport ladder is prepped rung
    # by rung, so the job-set grows while the description already names all
    # five (measured 2026-09-24: a root with the seed prepped answered a
    # one-rung ladder).  This route composed that itself until 2026-10-01,
    # and the CLI's `status` listed the prepped rungs alone.  `null` for a
    # container that is not the root.
    root = place["calculation"]
    if (place["role"] == calcdirs.CONTAINER and root is not None
            and Path(root).resolve() == directory.resolve()):
        from molbuilder.jobset.model import FILENAME as JOBSET_FILENAME
        from molbuilder.jobset.model import JobSet
        from molbuilder.jobset.runstatus import jobset_status
        try:
            jpath = directory / JOBSET_FILENAME
            got_status = jobset_status(
                JobSet.load(jpath) if jpath.is_file() else None, directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            attempts.append(f"the ladder could not be read: {exc}")
        else:
            # THE ONE WIRE FORM (`JobSetStatus.to_dict`), the next prep's
            # answer included -- a dict of this route's own dropped it (W52).
            ladder = got_status.to_dict()

    return jsonify({
        "ok":       True,
        "run_dir":  str(directory),
        "engine":   got["engine"],
        "openable": (Path(got["openable"]).name if got["openable"] else None),
        "attempts": attempts,
        # WHAT THIS DIRECTORY IS, and what it belongs to -- `null` when it
        # does not say, which the page shows as *read alone* rather than
        # hiding (§ 1.4a: absence narrows the answer, it does not refuse the
        # directory).
        "place":    place,
        # `null` where there is no run: a container, or a folder holding no
        # run of ours.
        "status":   got["status"],
        # WHAT RAN, WITH WHAT, AND HOW IT WENT (`model/parse.md` § 5d) --
        # the Run panel's one source (`web/results.md` § 3a); `null` where
        # `status` is.
        "record":   got["record"],
        # THE LADDER, for a calculation root (§ 2.4); `null` elsewhere.
        "ladder":   ladder,
        "files":    got["files"],
    })
