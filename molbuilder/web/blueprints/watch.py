"""Trajectory-loader API endpoints (live-update polling + parser
auto-detection).  This module exposes ONLY the JSON API endpoints
consumed by the /results trajectory inspector (see
``lib/inspectors/trajectory.js`` + ``lib/trajectory/core.js``).

Routes (registered with no url_prefix; each carries its own full path):

    POST /api/watch/load       JSON {"path": "..."} or multipart upload
                                ``path`` may be either a single file or
                                a run directory (job-layout v1; see
                                ``docs/execution/job-contracts.md``).
    GET  /api/watch/data       poll for changes (mtime-based)

Flow on /results: the user picks a trajectory file in the Projects
sidebar; the registry mounts the trajectory inspector; the inspector
core POSTs to /api/watch/load with the absolute path, then polls
/api/watch/data every ~15 s while the mtime advances.  The directory
branch of /api/watch/load ASKS the run door's `runs.openable` -- *what
should a viewer load here* -- and does not restate its rule, which is
`model/parse.md` § 5.1-§ 5.2.

Format support is plugin-style: see ``molbuilder/parse/`` for the
registered parsers and the auto-detection registry
(``model/parse.md`` § 3).

State model: a single global "current file" dict guarded by a Lock.
This is intentional -- the trajectory inspector is single-user /
single-tab by design (one inspector mounted at a time; see
docs/design.md for the original rationale).
"""

from __future__ import annotations

import os
import sys
import tempfile
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request

from molbuilder.parse import (
    ParseError,
    detect as detect_parser,
)
from molbuilder.parse.contract import engine_of
from molbuilder.runs import openable, run_answer, run_of, view_of
from molbuilder.parse.engines._helpers import (
    trajectory_result_to_legacy_dict as trajectory_to_legacy_dict,
)


bp = Blueprint("watch", __name__)

# Single global "current file" state.  A single user / single tab is
# the expected usage so a plain dict + lock is enough; no need for
# sessions.
_lock = Lock()
_state: Dict[str, Any] = {
    "path":     None,
    "mtime":    None,
    "data":     None,
    "parser":   None,    # the TrajectoryParser class chosen for this file
    "uploaded": False,   # True when the active file was uploaded via
                         # the file-picker (one-shot, no live watching)
}

# Track the last temp file we created from a file-picker upload so
# we can clean it up when a new upload comes in.  An atexit hook
# also clears it on clean process exit (Ctrl-C of the dev server),
# so a workflow of "spin up dev server, drop one upload, Ctrl-C" does
# not leave a /tmp/molwatch_* file behind.  SIGKILL / power loss
# can't be caught; /tmp self-cleans on reboot.
import atexit as _atexit
_last_temp_upload: Optional[str] = None


@_atexit.register
def _cleanup_last_temp_upload() -> None:                # pragma: no cover
    global _last_temp_upload
    if _last_temp_upload:
        _remove_temp_quietly(_last_temp_upload)
        _last_temp_upload = None


def _parser_name(parser_cls_or_none) -> Optional[str]:
    """Return the stable ``.name`` identifier on a TrajectoryParser
    subclass (e.g., ``"siesta"``, ``"molwatch"``), or ``None`` when
    no parser is registered yet.  Used by ``_refresh_if_changed`` to
    detect concurrent-load swaps without relying on class-object
    identity (``is``) -- the name attribute is the documented
    stable identifier shared with engine_metadata + error messages.
    """
    return getattr(parser_cls_or_none, "name", None) \
        if parser_cls_or_none is not None else None


def _remove_temp_quietly(path: str) -> None:
    """Best-effort delete of a temp-upload file with smarter error
    handling than ``try / except OSError: pass``.

    File-already-gone is benign (the user may have raced an external
    sweep on /tmp).  Other OSErrors (permission denied, EBUSY) are
    NOT benign -- the temp file leaks and the operator should know.
    Log to stderr rather than swallow so degraded /tmp permissions
    surface in the server log instead of silently leaking files.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        # Race-with-something-else-deleting-it; benign.
        pass
    except OSError as exc:
        print(f"[watch] failed to remove temp upload {path!r}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _refuse_if_not_a_trajectory(parser_cls):
    """``None`` if this parser answers a trajectory, else a 400 body.

    `/api/watch/*` is the TRAJECTORY route: everything after detection
    reads ``.frames``, so a single-geometry file -- one the app itself
    writes and the parser reads perfectly -- is refused here by name
    rather than failing as a 500.

    Measured on ``<job>_optimized.xyz``, PySCF's final geometry.  It is
    normally ABSORBED into the run's ``.molwatch.log`` entry
    (`results.md` § 2.3) so the picker never offers it -- but absorption
    narrows the MENU, not what can be opened, and every other route to
    this one (a pasted path, `molbuilder watch parse`, a restored
    session) reaches it.

    A parser declares its own answer in ``output``, so this asks rather
    than guesses, and names the file's real kind in the refusal instead
    of failing at the first attribute that is missing.
    """
    from molbuilder.parse.types import answers_a_trajectory

    if answers_a_trajectory(parser_cls):
        return None
    kind = getattr(getattr(parser_cls, "output", None), "__name__",
                   "an unknown result")
    return {
        "ok": False,
        "error": (
            f"{parser_cls.label} is read by the {parser_cls.name!r} parser, "
            f"which answers a {kind} -- not a trajectory. This viewer shows "
            f"a run's frames over time. Open the run's .molwatch.log (or a "
            f"*_geom_optim.xyz) to see the trajectory this geometry came "
            f"from."),
    }


def _engine_of(search_dir, payload, parser_cls) -> str:
    """WHICH ENGINE PRODUCED THIS RUN -- not which parser read it.

    Two different facts, and `web-api.md` (the `/api/watch/*` row) states
    the rule: *"`format` names the ENGINE that ran; `label` names the
    PARSER that read the file."*  They coincide for an engine-native
    file -- a SIESTA `.out` is read by the parser called `siesta` -- and
    diverge for the canonical `.molwatch.log`, read by the parser called
    `molwatch` whatever wrote it.

    NOTHING IS COMPUTED HERE.  The engine is a property of the RUN
    DIRECTORY, declared when its deck was generated, and
    `running-a-job.md` § 4.2 owns the resolution order;
    `parse.contract.engine_of` is its one implementation.

    **`source_format` is the fallback, and only an upload reaches it.**
    A posted file has no run directory, so what the parser found is the
    best honest answer -- including the bare ``"molwatch"`` of a log
    with no ``# engine:`` header, which is the neutral case the client
    already branches on.  It is NOT an engine field in general
    (``siesta-mdnc``, ``pyscf-geom`` and ``siesta-xv`` all live in it),
    which is precisely why the declared engine is asked first: reading
    this one AS the engine is the substitution the rule above forbids,
    and it is the bug this signature exists to make impossible.
    """
    if search_dir:
        declared = engine_of(search_dir)
        if declared != "unknown":
            return declared
    return (payload or {}).get("source_format") or parser_cls.name


def _frame0_structure(
    data: "Optional[Dict[str, Any]]", meta: "Dict[str, Any]"
) -> "Optional[Dict[str, Any]]":
    """Frame 0 as a STRUCTURE ENVELOPE, with the run's metadata already on it
    -- ``meta``, the block the same load answers with (:func:`_run_metadata`),
    so the directory is read once per load, not once per consumer.

    `web-api.md` § 1: *"the browser sends what it holds; it never sends a
    document it wrote"*. The pieces are all here --
    the frames from the parsed logs, the labels from the run's input script,
    the box from its output logs -- so the assembly belongs here too.

    Returns ``None`` when there are no frames; never raises, because a run
    whose metadata cannot be recovered must still open.
    """
    frames = (data or {}).get("frames")
    if not frames:
        return None
    try:
        import json as _json
        from molbuilder.runs import RunView
        meta_json = meta.get("atom_metadata")
        view = RunView(atom_metadata=(_json.loads(meta_json)
                                      if meta_json else None),
                       periodicity=meta.get("periodicity"),
                       info=meta.get("info"))
        first = frames[0]
        return view.structure([a[0] for a in first],
                              [a[1:4] for a in first]).to_dict()
    except Exception:                                   # noqa: BLE001
        # A run that cannot be assembled still OPENS -- the frames are the
        # point, and the tab falls back to what it always had.
        return None


def _run_metadata(
    data: Optional[Dict[str, Any]], *,
    output: Optional[str] = None, traj: Any = None,
) -> Dict[str, Any]:
    """The metadata block EVERY ``/api/watch/load`` answer carries.

    One composer for every builder of a load answer: a builder spreads
    it, and the upload branch (which has no run directory) answers
    "nothing available" DELIBERATELY -- ``None`` in every field --
    rather than by omission.

    Omission means something else on this route.  The browser's APPLY
    rule is keep-on-``undefined`` (`web/trajectory.md` § 5.1), which is
    what lets a poll re-send the frames without re-sending the metadata.
    So the POLL omits this block on purpose and the LOAD always sends
    it; a poll that carried it would rewrite a run's metadata on every
    tick.

    The three fields are three sources, not one: the labels come from the
    run's input script, the cell from its output logs, and ``info`` from
    the deck's stated parameters.  They travel together because they
    answer one question -- *what does this run directory say about the
    structure it ran on?* -- and a new metadata category joins them as a
    KEY inside ``info`` (``parse.dirs.run_info``), not as a fourth field.
    """
    # THE VIEW OF THE RUN the opened file belongs to (`runs.view_of`): its
    # labels and box from its own deck, its record from `run_info` -- the
    # one view every door that builds a structure from a run asks.  No run
    # of ours, as for an upload: the output's own lattice and record alone.
    frames = (data or {}).get("frames")
    view = view_of(run_of(output) if output else None, output=output,
                   traj=traj, n_atoms=len(frames[0]) if frames else None,
                   lattice=(data or {}).get("lattice"))
    import json as _json
    return {
        # Per-atom metadata (region labels / frozen tags / annotation
        # channels) the Build tab embedded in the run's input script, as a
        # JSON string the viewer applies; None when the run carries no block.
        "atom_metadata": (_json.dumps(view.atom_metadata)
                          if view.atom_metadata else None),
        # The run's box (the cell from the output logs, the axis kinds
        # from the deck's ENGINE-OFFSET record, and the engine's origin,
        # 0).  The viewer passes it through verbatim -- guessing
        # periodicity in the browser is the one thing the Cell rules
        # refuse.
        "periodicity":   view.periodicity,
        # What the run says ABOUT itself (`info.calculation`,
        # `info.relaxation`).  Rides installMolecule in and exportFile out
        # (molview.md § 8.4a), so an export from a results view carries it.
        "info":          view.info,
    }


def _refresh_if_changed() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Re-parse the current file iff its mtime has advanced.

    Returns ``(state, None)`` on success or ``(None, error_message)`` on
    failure.  Cheap when the file is unchanged.

    Locking strategy: snapshot path/mtime/parser under the lock, then
    drop the lock during the actual parse.  After parsing we re-acquire
    and only commit the result if the active file hasn't changed under us
    (defensive against a /api/watch/load racing with a /api/watch/data
    poll).

    **Dropping the lock does NOT stop this blocking other requests**
    (measured 2026-09-03).  The
    parse is pure Python, so it holds the GIL: with a 25 MB ``.out`` it
    runs 4.7 s and every other request in the process drops to about 8%
    of full speed for the whole of it; 51 MB is 9.6 s.  Releasing the
    lock lets another request *enter* -- it does not let it *run*.

    The file is re-parsed WHOLE whenever its mtime advanced.  **This has
    never been a problem in practice** (user, 2026-09-03) -- a small-lab
    server does not see concurrent heavy requests -- so the numbers are
    recorded rather than acted on, and `web-api.md` § 1a says what to do
    if it ever does arise.
    """
    # ---- Snapshot under the lock --------------------------------
    with _lock:
        path = _state["path"]
        if not path:
            return None, "No file loaded yet."
        cached_mtime = _state["mtime"]
        parser_cls   = _state["parser"]

    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return None, str(exc)

    # ---- Cheap path: nothing changed ----------------------------
    if mtime == cached_mtime:
        with _lock:
            return dict(_state), None

    # ---- Parse OUTSIDE the lock ---------------------------------
    # Parsers return a Trajectory; the JS client consumes the legacy
    # molwatch v1 dict shape, so we adapt at the boundary.
    try:
        traj = parser_cls.parse(path)
        new_data = trajectory_to_legacy_dict(traj)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Parse error: {exc}"
    new_data["stop_reason"] = _stop_reason(traj)
    # THE RATE IS THE RUN'S STAMPED SCF ROWS' (`model/parse.md` § 2a P-T4,
    # § 5c): the SIESTA tee or a PySCF progress log, by the one rule the run
    # record reads them by, re-read with the output.  ``None`` -- not stated
    # -- for a run with none.
    # The timing log is THE RUN'S, at the output's own run index, found
    # through the run door (`runs.run_of`) -- read back with the run's label,
    # never one guessed off the folder's decks (plan B11).
    from molbuilder.parse.dirs.record import scf_timing_of
    _run = run_of(path)
    new_data["scf_timing"] = scf_timing_of(
        path, _run.file(".scf-timing.log", _run.run)
        if _run is not None else None) or None

    # ---- Re-acquire to commit (skip if a concurrent /api/load
    #      already swapped to a different file under us) ---------
    #
    # Parser comparison by ``.name``: ``is`` works today but is fragile to
    # future detection refactors.
    with _lock:
        if (_state["path"] == path
                and _parser_name(_state["parser"]) == parser_cls.name):
            _state["data"]  = new_data
            _state["mtime"] = mtime
        # THE PARSE ITSELF rides the answer, not the state: a load hands it
        # to the directory's metadata so the same file is not parsed again
        # for it (`_run_metadata`), and nothing holds it past the request.
        out = dict(_state)
        out["parsed"] = (path, traj)
        return out, None


def _stop_reason(traj) -> Optional[str]:
    """Why a stopped run stopped, in words: the cause the file's own parse
    carries -- the ending reader's (`model/parse.md` § 2b), so the file is
    not read a second time for it -- worded by the SIESTA family's table
    (`siesta_grammar.CAUSE_WORDS`).  ``None`` for a run that did not stop,
    or a format that names no cause."""
    if getattr(traj, "run_state", None) not in ("stopped", "out_of_memory"):
        return None
    from molbuilder.parse.engines.siesta_grammar import CAUSE_WORDS
    cause = getattr(traj, "cause", None)
    return CAUSE_WORDS.get(cause) if cause else None


@bp.route("/api/watch/load", methods=["POST"])
def api_load():
    """Two body shapes:

      * multipart/form-data with a single file field "file" -- file
        is saved to a temp file and parsed (one-shot, no live update);
      * application/json with {"path": "..."} -- server reads the
        absolute path off disk and polls it for live updates.

    The multipart branch is the file-picker fallback for users who
    don't want to type an absolute path.
    """
    # ---- multipart upload (file-picker mode) -----------------------
    if "file" in request.files:
        return _api_load_multipart(request.files["file"])

    # ---- JSON path (live-watch mode) -------------------------------
    body = request.get_json(silent=True) or {}
    raw_path = (body.get("path") or "").strip()
    if not raw_path:
        return jsonify({"ok": False, "error": "Empty path."}), 400
    # The JSON-path mode routes through the canonical
    # ``_resolve_within_roots`` helper like every other path-taking
    # endpoint, per web-api.md § 2.1: it constrains to the picker roots
    # (Capabilities.file_picker_roots()).
    from .files import _resolve_within_roots, _PickerError
    try:
        raw_path = str(_resolve_within_roots(raw_path))
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # If the user passed a directory, the run door picks the file to load
    # (`runs.openable`); a regular file is loaded as it is.
    resolved_from_dir: Optional[str] = None
    if os.path.isdir(raw_path):
        # A DIRECTORY RESOLVES TO ONE FILE.  The generated deck tells the
        # user to point Watch at the run directory and says what happens:
        # "the loader resolves it to <job>.molwatch.log".  That is this
        # chain, and it is the whole of it.
        #
        # STAGES ARE SEPARATE RUNS.  A ladder is separated by filename in a flat
        # directory or by directory name in a hierarchical one, and the
        # person picks one stage and judges it.  Stitching them into one
        # trajectory is not a view this project offers -- that is what the
        # bench summary is for, where comparison IS the question.
        path, attempts = openable(raw_path)
        if path is None:
            tried = "\n  ".join(attempts) if attempts else "(no candidates)"
            return jsonify({
                "ok": False,
                "error": (
                    f"Nothing here a viewer can open:\n"
                    f"  {raw_path}\n"
                    f"What was tried (docs/model/parse.md § 5.2):\n"
                    f"  {tried}\n"
                    f"Point the viewer at a run's folder -- one prep "
                    f"opened -- or at one of its files."
                ),
            }), 404
        resolved_from_dir = raw_path
    else:
        path = raw_path
        if not os.path.isfile(path):
            return jsonify({
                "ok": False,
                "error": f"File or directory not found: {path}",
            }), 404

    # Auto-detect parser before committing to the new path so an
    # unsupported file doesn't blank out a working one.
    try:
        parser_cls = detect_parser(path)
    except ParseError as exc:
        # ParseError, NOT just UnknownFormatError.  `detect()` also
        # raises AmbiguousFormatError when two parsers claim one
        # file, and that is its SIBLING, not its subclass
        # (`parse/errors.py`).  The message names the clashing
        # parsers, so a 400 carrying it is useful.
        return jsonify({"ok": False, "error": str(exc)}), 400

    refusal = _refuse_if_not_a_trajectory(parser_cls)
    if refusal is not None:
        return jsonify(refusal), 400

    with _lock:
        _state["path"]     = path
        _state["mtime"]    = None      # force a re-parse next time
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = False

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    run, state, err = _with_the_run(state)
    if err:
        return jsonify({"ok": False, "error": err}), 500
    # The run's metadata, ONCE per load, for the structure envelope and the
    # answer alike, with the file this load opened and its parse handed on:
    # the relaxation record is of the file on screen.  The parse rides only when it is of that file -- a load racing this
    # one can swap the state between the parse and here.
    _parsed = state.get("parsed")
    meta = _run_metadata(state["data"], output=path,
                         traj=(_parsed[1] if _parsed and _parsed[0] == path
                               else None))
    return jsonify({
        "ok":               True,
        "path":             state["path"],
        "resolved_from":    resolved_from_dir,
        "mtime":            state["mtime"],
        "format":           _engine_of(
            resolved_from_dir or os.path.dirname(path),
            state["data"], parser_cls),
        "label":            parser_cls.label,
        "data":             state["data"],
        "uploaded":         False,
        # HOW THE RUN THIS FILE BELONGS TO IS DOING -- the one door's answer,
        # which the viewer follows (`web/results.md` § 4.1).
        "run":              run,
        # FRAME 0 AS AN ENVELOPE -- what the viewer installs.  The parcels below
        # stay because the Cell page reads them directly.
        "structure":        _frame0_structure(state["data"], meta),
        **meta,
    })


def _api_load_multipart(uploaded_file):
    """Save the uploaded file to a tempdir, parse, and stash the temp
    path on _state.  Future /api/watch/data polls work like always but the
    mtime never advances (we don't write to the temp file again), so
    the data effectively snapshots at upload time.

    Old temp uploads are cleaned up when a new one comes in -- a
    process restart drops the rest.
    """
    global _last_temp_upload

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    # Keep the original suffix (.xyz / .out / .log) so the parser-
    # detection layer's content sniff isn't fooled by extension-less
    # names.  Sanitise the basename to dodge path-traversal in the
    # temp filename itself.
    #
    # mkstemp (R6) reserves a unique filename atomically, so two uploads
    # in the same second cannot overwrite each other while a parser is
    # reading the file.
    safe_name = os.path.basename(uploaded_file.filename) or "upload"
    safe_stem = os.path.splitext(safe_name)[0]
    safe_suffix = os.path.splitext(safe_name)[1] or ""
    try:
        # mkstemp returns (fd, path) with an atomically-reserved
        # unique filename.  Close the fd immediately and let
        # ``uploaded_file.save(path)`` reopen the path -- werkzeug's
        # FileStorage.save expects either a path string or a
        # writable binary stream.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f"molwatch_{safe_stem}_", suffix=safe_suffix,
        )
        os.close(tmp_fd)
        uploaded_file.save(tmp_path)
    except OSError as exc:
        return jsonify({"ok": False,
                        "error": f"Failed to write upload: {exc}"}), 500

    try:
        parser_cls = detect_parser(tmp_path)
    except ParseError as exc:
        # Don't keep an undetectable upload around.  ParseError covers
        # AmbiguousFormatError as well -- see the note at the JSON-path
        # branch above.
        _remove_temp_quietly(tmp_path)
        return jsonify({"ok": False, "error": str(exc)}), 400

    refusal = _refuse_if_not_a_trajectory(parser_cls)
    if refusal is not None:
        _remove_temp_quietly(tmp_path)
        return jsonify(refusal), 400

    with _lock:
        # Clean up any previous upload's temp file.
        if _last_temp_upload and _last_temp_upload != tmp_path:
            _remove_temp_quietly(_last_temp_upload)
        _last_temp_upload = tmp_path
        _state["path"]     = tmp_path
        _state["mtime"]    = None
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = True

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    meta = _run_metadata(state["data"])
    return jsonify({
        "ok":               True,
        "path":             tmp_path,
        "mtime":            state["mtime"],
        # An upload has no run directory to declare an engine, so this
        # is the one caller that reaches the `source_format`
        # fallback -- deliberately, and it is the only one.
        "format":           _engine_of(None, state["data"], parser_cls),
        "label":            parser_cls.label,
        "data":             state["data"],
        "uploaded":         True,
        "uploaded_filename": uploaded_file.filename,
        # An upload belongs to no run, so there is no run to follow --
        # said, in the field the path load answers (`web/results.md` § 4.1).
        "run":              None,
        # Frame 0 as an envelope, same as the path branch -- an upload has no
        # run directory, so it carries the geometry and nothing more.
        "structure":        _frame0_structure(state["data"], meta),
        # An upload is one file with no run directory behind it, so it
        # has nothing to say about itself -- and it SAYS so, in the same
        # fields the path builder answers.  One route, one response
        # shape: a reader learns what a load answers from one place, and
        # "nothing available" is a stated answer rather than a field a
        # caller has to notice is missing.
        **meta,
    })


@bp.route("/api/watch/data")
def api_data():
    """Return the parsed payload, or just an mtime if nothing changed."""
    client_mtime = request.args.get("mtime", type=float)
    state, err = _refresh_if_changed()
    if err:
        # web-api.md § 1, *Status codes* -- server fault: parse / IO error on a
        # user-selected trajectory file, the same 500 the sibling
        # /api/watch/load returns.  The JS poll-loop reads body.ok; external
        # consumers (curl / CI / monitoring) gating on HTTP status see the
        # actual failure.
        return jsonify({"ok": False, "error": err}), 500
    if client_mtime is not None and client_mtime == state["mtime"]:
        # NOTHING NEW FOR THIS VIEWER: how the run is doing is the news
        # (`web/results.md` § 4.1) -- and when it is over, the file's last
        # read comes after that answer, so a final write between the two
        # reaches the viewer with it.
        if state.get("uploaded"):
            return jsonify({"ok": True, "changed": False,
                            "mtime": state["mtime"], "run": None})
        run, state, err = _with_the_run(state)
        if err:
            return jsonify({"ok": False, "error": err}), 500
        if client_mtime == state["mtime"]:
            return jsonify({"ok": True, "changed": False,
                            "mtime": state["mtime"], "run": run})
        return jsonify(dict(_changed(state), run=run))
    # NEW CONTENT: the run was writing, so how it is doing is asked when the
    # file is quiet -- the field is left out, which keeps it.
    return jsonify(_changed(state))


def _with_the_run(state: Dict[str, Any]
                  ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any],
                             Optional[str]]:
    """``(run, state, error)`` -- how the run the watched file belongs to is
    doing (`runs.run_answer`, the one door), and, when it is no longer
    live, the file read again if it changed since ``state``: the run's end is
    read before the file's last read, so what a viewer stops on is the
    file's last (`web/results.md` § 4.1)."""
    run = run_answer(state["path"])
    if run is not None and run["live"]:
        return run, state, None
    again, err = _refresh_if_changed()
    if err or again is None or again["mtime"] == state["mtime"]:
        # Nothing new: the state in hand, with the parse it carries -- the
        # load hands that parse on (`_run_metadata`), and a refresh that
        # found nothing has none to hand.
        return run, state, err
    return run, again, None


def _changed(state: Dict[str, Any]) -> Dict[str, Any]:
    """A poll's answer carrying the file's new content."""
    parser_cls = state["parser"]
    return {
        "ok":       True,
        "changed":  True,
        "path":     state["path"],
        "mtime":    state["mtime"],
        # An UPLOAD has no run directory, and `os.path.dirname` of its
        # temp path is the system temp dir -- shared, and full of other
        # people's files, which must not decide the engine.  The load path passes None here for the
        # same reason; the poll must agree with it or one file gets two
        # answers.
        "format":   _engine_of(
            None if state.get("uploaded")
            else os.path.dirname(state["path"]),
            state["data"], parser_cls),
        "label":    parser_cls.label,
        "data":     state["data"],
        "uploaded": state.get("uploaded", False),
    }
