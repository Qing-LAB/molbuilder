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
    (`archive/2026-09-01-structure-info-plan.md` I5).

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


@bp.route("/api/results/dir", methods=["GET"])
def api_results_dir():
    """**What is in this run directory, and which file should open?**

    The HTTP surface over `parse.dirs`' front door — `JobDirParser` /
    `RunDirResult` — and the consumer it was built for and never got
    (`plans/plan.md` N9; `model/parse.md` § 5.0, § 5.5).

    **WHY THIS EXISTS.**  The browser is the one consumer that cannot import
    Python, and nothing served it the directory's own answer: the Results
    picker listed through `/api/files/list` (content-blind — `{name, kind,
    size, mtime}`) and then decided FOUR things the backend already owns,
    from the filename, in JavaScript.  Measured 2026-09-18 over 110 real run
    directories: *which file to open* differed from `openable_in` on **18 of
    96**; **13** files were offered that no parser can read and **155** that
    a parser handles were unreachable; the engine was guessed from the
    suffix; and the run-file grammar was re-implemented as a JS regex.  A
    spectrum run was shown a 1,373-byte progress stub instead of its
    spectrum because two mtimes landed in the same second and the
    tie-break picked by name.

    Every one of those is the same missing connection, so they are answered
    together, once, here.

    **PER FILE the server says what the file IS**, which is the half the
    browser cannot derive: ``role`` from the catalogue, ``label`` and
    ``stage`` — *which run this file is part of, and which rung* — read
    back with the label the deck states, ``parser`` from the registry
    (``null`` when nothing claims it — the browser must not offer it),
    and ``engine`` from the directory, not from the suffix.

    ``openable`` is the door's own pick, so the page defaults to the file the
    calculation produced rather than to whatever was written last.
    """
    from pathlib import Path

    from flask import jsonify, request

    from molbuilder.parse import detect
    from molbuilder.parse.dirs import (labels_in, openable_in, read_back,
                                       run_status)
    from molbuilder.parse.contract import engine_of
    from molbuilder.parse.errors import ParseError
    from molbuilder.runfiles import role_of
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

    # WHAT IS THIS DIRECTORY -- asked first, because it decides whether the
    # other questions apply at all (`model/parse.md` § 5.5,
    # `project-layout.md` § 1.4a).  A container holds runs; it is not one, and
    # asking `run_status` about one is how a `pseudos/` folder and a finished
    # stage both came back *running*.
    from molbuilder import calcdirs
    place = calcdirs.container_or_run(directory)
    root = calcdirs.root_of(directory)

    if place == calcdirs.CONTAINER:
        opened, attempts = None, [
            "this directory is a container, not a run -- its runs are the "
            "directories below it (project-layout.md § 1.4)"]
        st = None
    else:
        opened, attempts = openable_in(str(directory))
        # AND A DIRECTORY THAT SAYS NOTHING IS NOT ASKED TO INVENT ONE.
        # `run_status`'s four states are running/stale/finished/failed --
        # there is no *there is no run here*, so an absence comes back
        # `running, no result file yet`.  That is the same defect the
        # container branch above fixes, arriving by the other door:
        # measured 2026-09-19 on the regenerated tree, TEN of nineteen
        # directories reported *running* -- `scan/`, `structure/`,
        # `transport/` and the project root among them, none of which has
        # ever held a run.
        #
        # § 1.4a: absence NARROWS the answer.  An unmarked directory may
        # still be read alone -- its files listed, one of them opened --
        # and `openable_in` above does exactly that.  What it may not do is
        # assert a run state with nothing to ground it on, so the status is
        # claimed only where there is: the record says RUN, or the door
        # found this directory's product.  A molbuilder-made run is always
        # stamped, so it keeps its full status from the first branch even
        # before it writes anything.
        st = (run_status(directory)
              if (place == calcdirs.RUN or opened) else None)

    # THE LABEL, so each file can be read back EXACTLY.  `role_of` answers
    # WITHOUT one and therefore cannot answer an underscore role at all --
    # its own docstring says why: a role may contain `_` and so may a stage
    # name, and nothing tells them apart from the string alone.  So
    # `_initial.xyz`, `_optimized.xyz` and `_geom_optim.xyz` -- the very
    # files the browser has to fold into one run -- all came back
    # `role: null`, and the browser re-implemented the grammar as a regex to
    # compensate: `trajectory.js::absorbs` read `au_2_bdt_02_fine` as a label
    # of `au`.  The deck states the label; `labels_in` asks it.
    labels = labels_in(str(directory))

    files = []
    for entry in sorted(directory.iterdir(), key=lambda e: e.name):
        if not entry.is_file():
            continue
        rec = read_back(entry.name, labels)
        # THE REGISTRY DECIDES WHETHER IT CAN BE OPENED, never the suffix.
        # A refusal is an answer -- `parser: null` is what stops the page
        # offering a Slurm log, a 0-byte `.out`, or a `job-set.json` its own
        # inspector's route then refuses with a 400.
        try:
            kind = detect(str(entry))
            parser, opens = kind.name, getattr(
                getattr(kind, "output", None), "__name__", None)
        except (ParseError, OSError, ValueError, LookupError):
            parser, opens = None, None
        files.append({
            "name":   entry.name,
            "role":   rec.role if rec is not None else role_of(entry.name),
            # WHICH RUN THIS FILE IS PART OF, and which rung.  `label` is
            # what makes *these files are one run* answerable without string
            # surgery; both are `null` for a file no label claims -- a
            # foreign file, or a directory with no deck.
            "label":  rec.label if rec is not None else None,
            "stage":  rec.stage if rec is not None else None,
            "parser": parser,
            "opens":  opens,
            "size":   entry.stat().st_size,
            "mtime":  entry.stat().st_mtime,
        })

    return jsonify({
        "ok":       True,
        "run_dir":  str(directory),
        "engine":   engine_of(str(directory)),
        "openable": Path(opened).name if opened else None,
        "attempts": attempts,
        # WHAT THIS DIRECTORY IS, and what it belongs to -- `null` when it
        # does not say, which the page shows as *read alone* rather than
        # hiding (§ 1.4a: absence narrows the answer, it does not refuse the
        # directory).
        "place":    {"role": place,
                     "calculation": str(root) if root else None},
        # `null` for a container: it has no run state, and inventing one is
        # the defect this route now refuses to repeat.
        "status":   None if st is None else
                    {"state": st.state, "detail": st.detail,
                     "active_source": st.active_source,
                     "last_change_at": st.last_change_at,
                     "concluded": st.concluded},
        "files":    files,
    })
