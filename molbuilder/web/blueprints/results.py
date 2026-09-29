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
    info = run_info_for_dir(directory) or {}
    return jsonify({"ok": True, "calculation": info.get("calculation"),
                    "relaxation": info.get("relaxation")})


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

    from molbuilder.parse import detect, parse_dir
    from molbuilder.parse.dirs import labels_in, read_back
    from molbuilder.parse.errors import ParseError, UnknownFormatError
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

    # THE DOOR ANSWERS the run's questions -- engine, the file to open, the
    # run state and the record (`model/parse.md` § 5.0) -- and it owns the
    # rules for WHAT THIS DIRECTORY IS: a container has no run state but may
    # have a product (a transport ladder's I-V record at its root); a folder
    # nobody described is read alone, with a run state only where its product
    # was found.  They lived in this route until W35 P2 (2026-09-26), where
    # nothing below the web layer could apply them.  A folder the door does
    # not claim -- empty, or holding no file a run writes -- has no run to
    # describe, and is still listed below.
    try:
        got = parse_dir(directory)
    except UnknownFormatError as exc:
        got, attempts = None, [str(exc)]
    else:
        attempts = list(got.attempts)

    from molbuilder import calcdirs
    place = calcdirs.container_or_run(directory)
    root = calcdirs.root_of(directory)
    ladder = None
    # A CALCULATION ROOT HAS A LADDER (`web/results.md` § 2.4; `plan.md`
    # § 5c.3 c-d): N rungs, each a run directory below it.  THE RUNGS ARE THE
    # DESCRIPTION'S (`task.stages`, `stages.md` § 6.7: the ladder is read,
    # never inferred); each one's state is `jobset_status`'s reading -- the
    # one ladder door the CLI's `status` verb reads, CONSUMED here, never
    # copied.  A described rung the job-set does not hold yet is NOT_PREPPED
    # in the reader's own words: a transport ladder is prepped rung by rung,
    # so the job-set grows while the description already names all five
    # (measured 2026-09-24: a root with the seed prepped answered a one-rung
    # ladder).  `null` for a container that is not the root.
    if (place == calcdirs.CONTAINER and root is not None
            and Path(root).resolve() == directory.resolve()):
        from molbuilder.jobset.model import FILENAME as JOBSET_FILENAME
        from molbuilder.jobset.model import JobSet
        from molbuilder.jobset.runstatus import NOT_PREPPED, jobset_status
        from molbuilder.task import FILENAME as TASK_FILENAME, read_task
        try:
            _described = [s.name for s in
                          read_task(directory / TASK_FILENAME).stages]
            jpath = directory / JOBSET_FILENAME
            _known = ({s.name: s
                       for s in jobset_status(JobSet.load(jpath),
                                              directory).stages}
                      if jpath.is_file() else {})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            attempts.append(f"the ladder could not be read: {exc}")
        else:
            rows = []
            for name in _described:
                s = _known.get(name)
                rows.append(
                    {"name": name, "seq": s.seq, "state": s.state,
                     "detail": s.detail, "dir": s.dir,
                     "attempt": s.attempt} if s is not None else
                    {"name": name, "seq": None, "state": NOT_PREPPED[0],
                     "detail": NOT_PREPPED[1], "dir": None,
                     "attempt": None})
            _open = [r["name"] for r in rows if r["state"] != "finished"]
            ladder = {"complete": not _open,
                      "first_incomplete": _open[0] if _open else None,
                      "stages": rows}

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
        "engine":   got.engine if got is not None else "unknown",
        "openable": (Path(got.openable).name
                     if got is not None and got.openable else None),
        "attempts": attempts,
        # WHAT THIS DIRECTORY IS, and what it belongs to -- `null` when it
        # does not say, which the page shows as *read alone* rather than
        # hiding (§ 1.4a: absence narrows the answer, it does not refuse the
        # directory).
        "place":    {"role": place,
                     "calculation": str(root) if root else None},
        # `null` where there is no run: a container, or a folder whose run
        # the door could not ground (`RunDirResult`).
        "status":   got.status if got is not None else None,
        # WHAT RAN, WITH WHAT, AND HOW IT WENT (`model/parse.md` § 5d) --
        # the Run panel's one source (`web/results.md` § 3a); `null` where
        # `status` is.
        "record":   got.record if got is not None else None,
        # THE LADDER, for a calculation root (§ 2.4); `null` elsewhere.
        "ladder":   ladder,
        "files":    files,
    })
