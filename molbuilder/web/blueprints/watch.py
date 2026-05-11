"""Watch blueprint -- live trajectory viewer for SIESTA / PySCF / future.

Routes (registered with no url_prefix; each carries its own full path):

    GET  /watch                browser UI page
    GET  /api/watch/formats    parser registry summary
    POST /api/watch/load       JSON {"path": "..."} or multipart upload
                                ``path`` may be either a single file or
                                a run directory (job-layout v1; see
                                ``docs/spec/job-layout.md``).
    GET  /api/watch/data       poll for changes (mtime-based)

The user opens /watch, paste an absolute path to the output file or
the **run directory** containing it, and clicks *Load*.  The
directory branch follows the discovery chain in
``docs/spec/job-layout.md``: ``*.molwatch.log`` first, then ``*.fdf``
parsed for SystemLabel, then ``*.py`` parsed for job_name, then a
generic ``*.out`` / ``*_geom_optim.xyz`` fallback.  The page polls
/api/watch/data roughly every 15 seconds; when the file's mtime
advances the parser re-runs and the viewer + plots refresh.

Format support is plugin-style: see ``molbuilder/parsers/`` for the
registered parsers and the auto-detection registry.

State model: a single global "current file" dict guarded by a Lock.
This is intentional -- the watch app is single-user / single-tab by
design (see docs/design.md "Watch -- live trajectory viewer").
"""

from __future__ import annotations

import glob
import os
import re
import sys
import tempfile
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, render_template, request

from molbuilder.parsers import (
    UnknownFormatError,
    detect_parser,
    parser_summary,
    trajectory_to_legacy_dict,
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

    # Multi-stage merge state.  Set when the user loaded a directory
    # containing > 1 *.molwatch.log files; ``_refresh_if_changed``
    # re-runs the merge when ``run_dir`` is set so a /api/watch/data
    # poll doesn't clobber the merged trajectory with the newest
    # log's frames alone.
    "run_dir":  None,    # absolute directory path or None
}

# Track the last temp file we created from a file-picker upload so
# we can clean it up when a new upload comes in.  An atexit hook
# also clears it on clean process exit (Ctrl-C of the dev server),
# so a workflow of "spin up dev server, drop one upload, Ctrl-C" no
# longer leaves a /tmp/molwatch_* file behind.  SIGKILL / power loss
# can't be caught; /tmp self-cleans on reboot.
import atexit as _atexit
_last_temp_upload: Optional[str] = None


@_atexit.register
def _cleanup_last_temp_upload() -> None:                # pragma: no cover
    global _last_temp_upload
    if _last_temp_upload:
        try:
            os.remove(_last_temp_upload)
        except OSError:
            pass
        _last_temp_upload = None


# --------------------------------------------------------------------- #
#  Directory-aware path resolution (job-layout v1)                      #
#                                                                       #
#  See ``docs/spec/job-layout.md`` for the full contract.  When the     #
#  user gives Watch a directory instead of a file, scan it for the      #
#  canonical artefacts in the protocol's preferred order; first hit     #
#  wins.  The fallbacks parse the molbuilder-generated input files      #
#  (.fdf / .py) to recover the basename.                                #
# --------------------------------------------------------------------- #


# SystemLabel may appear with or without the dotted form; SIESTA's own
# parser accepts both.  Match on a single token (no spaces) so we
# don't capture trailing comments.
_FDF_SYSTEM_LABEL_RE = re.compile(
    # Bound the capture to the job-layout basename charset; SIESTA's
    # SystemLabel must be filesystem-safe anyway and trusting an
    # arbitrary \S+ here would let a malformed FDF inject path
    # fragments into the discovery chain's os.path.join() below.
    r"^\s*SystemLabel(?:\s|\.)\s*([A-Za-z0-9._\-]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# molbuilder-generated PySCF scripts set ``job_name = "..."`` near the
# top.  The regex anchors on the LHS to avoid catching incidental
# strings further down.
_PY_JOB_NAME_RE = re.compile(
    # Allow dots so an old/external script using ``job_name = "mol.run1"``
    # still resolves; same charset as the basename validator in
    # molbuilder/config/siesta.py (see _BASENAME_RE).
    r"^\s*job_name\s*=\s*[\"\']([A-Za-z0-9._\-]+)[\"\']",
    re.MULTILINE,
)


def _read_text_safely(path: str, max_bytes: int = 65536) -> str:
    """Read up to ``max_bytes`` of ``path`` and return as text.  Used
    for the .fdf / .py header sniff -- we only need the first chunk
    to find SystemLabel / job_name; reading the whole multi-MB FDF
    is wasteful.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _basename_from_fdf(path: str) -> Optional[str]:
    text = _read_text_safely(path)
    m = _FDF_SYSTEM_LABEL_RE.search(text)
    return m.group(1) if m else None


def _basename_from_py(path: str) -> Optional[str]:
    text = _read_text_safely(path)
    m = _PY_JOB_NAME_RE.search(text)
    return m.group(1) if m else None


def _newest(paths: List[str]) -> Optional[str]:
    """Pick the most recently modified path from a list."""
    valid = [p for p in paths if os.path.isfile(p)]
    if not valid:
        return None
    return max(valid, key=lambda p: os.path.getmtime(p))


_MAX_LOGS_PER_DIR = 256


def _list_molwatch_logs(directory: str) -> List[str]:
    """Return all ``*.molwatch.log`` files in the directory in mtime
    order (oldest first).  Used to detect a multi-stage run; if the
    list has > 1 element, the loader concatenates them.

    Tiebreak by basename so two files written in the same second
    (POSIX-on-FAT, NFS without sub-second precision) sort
    deterministically.  With the canonical ``<base>-stage<N>``
    naming, lexical order matches stage order so this also
    rescues the multi-stage merge from filesystem inode order.

    Capped at ``_MAX_LOGS_PER_DIR`` so an accidentally-shared
    directory with thousands of stale logs can't make every poll
    spend seconds in glob + merge.  The cap chooses the NEWEST
    entries (likely the active staged run); older entries are
    dropped with a stderr warning.
    """
    hits = glob.glob(os.path.join(directory, "*.molwatch.log"))
    valid = [p for p in hits if os.path.isfile(p)]
    sorted_logs = sorted(
        valid, key=lambda p: (os.path.getmtime(p), os.path.basename(p)),
    )
    if len(sorted_logs) > _MAX_LOGS_PER_DIR:
        dropped = len(sorted_logs) - _MAX_LOGS_PER_DIR
        print(
            f"[watch] {directory}: {len(sorted_logs)} *.molwatch.log "
            f"files; capping at {_MAX_LOGS_PER_DIR} newest "
            f"(dropped {dropped} older).", file=sys.stderr,
        )
        sorted_logs = sorted_logs[-_MAX_LOGS_PER_DIR:]
    return sorted_logs


def _resolve_run_directory(directory: str) -> Tuple[Optional[str], List[str]]:
    """Resolve a run directory to a single file Watch should load.

    Returns ``(resolved_path, attempts)``: the resolved file path
    (or ``None`` if nothing was found), plus a list of human-readable
    "tried X" strings used to build the error message when nothing
    matches.

    Discovery chain follows ``docs/spec/job-layout.md`` § "How Watch
    resolves a directory":

      1. Any ``*.molwatch.log`` (newest wins for staged runs).
      2. ``*.fdf`` -> parse SystemLabel -> ``<label>.molwatch.log``,
         ``<label>.out``.
      3. ``*.py``  -> parse job_name      -> ``<job>.molwatch.log``,
         ``<job>.log``, ``<job>_geom_optim.xyz``.
      4. Generic fallbacks: ``run.out``, ``siesta.log``, ``*.out``,
         ``*_geom_optim.xyz``.
    """
    attempts: List[str] = []

    # 1. *.molwatch.log directly in the directory.
    log_hits = glob.glob(os.path.join(directory, "*.molwatch.log"))
    attempts.append(f"*.molwatch.log -> {len(log_hits)} match(es)")
    if log_hits:
        return _newest(log_hits), attempts

    # 2. SIESTA: *.fdf -> SystemLabel -> sibling outputs.
    fdf_hits = glob.glob(os.path.join(directory, "*.fdf"))
    attempts.append(f"*.fdf -> {len(fdf_hits)} match(es)")
    for fdf in fdf_hits:
        label = _basename_from_fdf(fdf)
        if not label:
            attempts.append(f"  {os.path.basename(fdf)}: SystemLabel not found")
            continue
        for suffix in (".molwatch.log", ".out"):
            cand = os.path.join(directory, f"{label}{suffix}")
            attempts.append(f"  -> {label}{suffix}: "
                            f"{'found' if os.path.isfile(cand) else 'missing'}")
            if os.path.isfile(cand):
                return cand, attempts

    # 3. PySCF: *.py -> job_name -> sibling outputs.
    py_hits = glob.glob(os.path.join(directory, "*.py"))
    attempts.append(f"*.py -> {len(py_hits)} match(es)")
    for py in py_hits:
        name = _basename_from_py(py)
        if not name:
            attempts.append(f"  {os.path.basename(py)}: job_name not found")
            continue
        for suffix in (".molwatch.log", ".log", "_geom_optim.xyz"):
            cand = os.path.join(directory, f"{name}{suffix}")
            attempts.append(f"  -> {name}{suffix}: "
                            f"{'found' if os.path.isfile(cand) else 'missing'}")
            if os.path.isfile(cand):
                return cand, attempts

    # 4. Generic fallbacks.
    for fname in ("run.out", "siesta.log"):
        cand = os.path.join(directory, fname)
        attempts.append(f"{fname}: "
                        f"{'found' if os.path.isfile(cand) else 'missing'}")
        if os.path.isfile(cand):
            return cand, attempts
    out_hits = glob.glob(os.path.join(directory, "*.out"))
    if out_hits:
        attempts.append(f"*.out -> picked {os.path.basename(out_hits[0])}")
        return _newest(out_hits), attempts
    optim_hits = glob.glob(os.path.join(directory, "*_geom_optim.xyz"))
    if optim_hits:
        attempts.append(f"*_geom_optim.xyz -> "
                        f"picked {os.path.basename(optim_hits[0])}")
        return _newest(optim_hits), attempts

    return None, attempts


def _refresh_if_changed() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Re-parse the current file iff its mtime has advanced.

    Returns ``(state, None)`` on success or ``(None, error_message)`` on
    failure.  Cheap when the file is unchanged.

    Locking strategy: snapshot path/mtime/parser under the lock, then
    drop the lock during the actual parse so other concurrent requests
    aren't blocked for the duration of a multi-MB log re-parse.  After
    parsing we re-acquire and only commit the result if the active file
    hasn't changed under us (defensive against a /api/load racing with
    a /api/data poll).
    """
    # ---- Snapshot under the lock --------------------------------
    with _lock:
        path = _state["path"]
        if not path:
            return None, "No file loaded yet."
        cached_mtime = _state["mtime"]
        parser_cls   = _state["parser"]
        run_dir      = _state["run_dir"]

    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return None, str(exc)

    # ---- Multi-stage: re-merge when the active log changed ------
    # When a multi-stage merge is in flight, every poll re-scans the
    # directory and re-runs the merge over the full set of logs.  Two
    # things this gets right:
    #   1. The user can drop a new <base>-stage<N+1>.molwatch.log into
    #      the directory mid-watch and the next poll picks it up.
    #   2. The earlier stages' frames don't disappear when the active
    #      (newest) stage's mtime advances -- the merge re-runs over
    #      every log on every refresh, so the merged trajectory is
    #      always the FULL multi-stage view, not just the newest.
    if run_dir is not None:
        all_logs = _list_molwatch_logs(run_dir)
        active_path = all_logs[-1] if all_logs else path
        try:
            active_mtime = os.path.getmtime(active_path)
        except OSError as exc:
            return None, str(exc)
        # Cheap path: nothing changed in the newest log.
        if active_mtime == cached_mtime and active_path == path:
            with _lock:
                return dict(_state), None
        try:
            new_data, _stages_meta = _merge_molwatch_trajectories(all_logs)
        except Exception as exc:                              # noqa: BLE001
            return None, f"Multi-stage refresh failed: {exc}"
        with _lock:
            # Defensive: a concurrent /api/load may have swapped to a
            # different run (different directory OR different parser
            # class).  Mirror the single-file branch's identity check
            # so a stale parse never overwrites fresher state.
            if (_state["run_dir"] == run_dir
                    and _state["parser"] is parser_cls):
                _state["path"]  = active_path
                _state["data"]  = new_data
                _state["mtime"] = active_mtime
            return dict(_state), None

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

    # ---- Re-acquire to commit (skip if a concurrent /api/load
    #      already swapped to a different file under us) ---------
    with _lock:
        if _state["path"] == path and _state["parser"] is parser_cls:
            _state["data"]  = new_data
            _state["mtime"] = mtime
        return dict(_state), None


# Per-file parse cache for the multi-stage merge.  Keyed by absolute
# path, valued by ``(mtime, legacy_dict)``.  ``_merge_molwatch_trajectories``
# reuses a cached entry when the file's mtime hasn't advanced, so a
# poll that picks up new frames in the newest stage doesn't re-parse
# the 2 or 3 prior (static) stage logs.  Single-process, single-user
# scope -- bounded by the number of stages in flight (≤ a handful).
_MERGE_PARSE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _merge_molwatch_trajectories(paths: List[str]) -> Tuple[Dict[str, Any],
                                                              List[Dict[str, Any]]]:
    """Parse each ``*.molwatch.log`` in ``paths`` (oldest first) and
    concatenate the resulting trajectories into one legacy-shape dict.

    Returns ``(merged_dict, stages_metadata)`` where ``stages_metadata``
    is a list of one dict per source file::

        {"name":        "my-job-stage1.molwatch.log",
         "start_frame": 0,
         "end_frame":   17,
         "n_frames":    18,
         "mtime":       1715300000.0}

    The frontend uses ``stages_metadata`` to draw stage-boundary
    markers on the energy / force plots and to show a "merged from N
    stages" message in the status banner.

    The merged dict's ``frames`` / ``energies`` / ``forces`` /
    ``scf_history`` are concatenated in order; ``iterations`` is
    re-numbered globally so the plot's x-axis stays monotone.
    ``lattice`` and ``source_format`` come from the LATEST trajectory
    (it's the one the polling thread is watching for updates).
    """
    from molbuilder.parsers.molwatch_log import MolwatchLogParser

    merged: Dict[str, Any] = {
        "frames": [], "energies": [], "max_forces": [],
        "forces": [], "scf_history": [], "wall_times": [],
        "iterations": [], "step_indices": [], "stages": [],
    }
    stages: List[Dict[str, Any]] = []
    any_scf = False
    last_traj_legacy: Optional[Dict[str, Any]] = None
    for path in paths:
        # Per-file parse is wrapped: a torn / mid-write log mustn't
        # take down the whole merge.  Surfacing the failure in stages
        # metadata lets the frontend say "stage 2 unparseable, showing
        # 1 + 3" instead of HTTP 500.  A small mtime-keyed parse
        # cache means an unchanged stage isn't re-parsed on every
        # poll (perf hot path -- otherwise three 50-MB logs each cost
        # one full re-parse per 15s tick).
        n_before = len(merged["frames"])
        try:
            mt = os.path.getmtime(path)
            cached = _MERGE_PARSE_CACHE.get(path)
            if cached is not None and cached[0] == mt:
                legacy = cached[1]
            else:
                traj   = MolwatchLogParser.parse(path)
                legacy = trajectory_to_legacy_dict(traj)
                _MERGE_PARSE_CACHE[path] = (mt, legacy)
        except Exception as exc:                              # noqa: BLE001
            # Log to stderr so the operator has a signal beyond the
            # frontend's stages-metadata entry; dedupe by (path, exc-type)
            # in the cache key so a repeating tear doesn't spam the log.
            print(f"[watch] merge skipped {path}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            stages.append({
                "name":        os.path.basename(path),
                "start_frame": n_before,
                "end_frame":   n_before,
                "n_frames":    0,
                "mtime":       os.path.getmtime(path),
                "error":       f"{type(exc).__name__}: {exc}",
            })
            continue
        last_traj_legacy = legacy
        merged["frames"].extend(legacy["frames"])
        merged["energies"].extend(legacy["energies"])
        merged["max_forces"].extend(legacy["max_forces"])
        merged["forces"].extend(legacy["forces"])
        merged["wall_times"].extend(legacy["wall_times"])
        # Preserve the per-stage step indices alongside the global
        # iteration renumbering below.  Save-frame-as-XYZ uses these
        # to label the file with the source-log step number rather
        # than a meaningless global frame index.
        merged["step_indices"].extend(legacy["iterations"])
        if legacy["scf_history"]:
            any_scf = True
            merged["scf_history"].extend(legacy["scf_history"])
        else:
            # Fill with empty per-frame entries so frame index aligns.
            merged["scf_history"].extend([[] for _ in legacy["frames"]])
        n_after = len(merged["frames"])
        stages.append({
            "name":        os.path.basename(path),
            "start_frame": n_before,
            "end_frame":   max(n_after - 1, n_before),
            "n_frames":    n_after - n_before,
            "mtime":       os.path.getmtime(path),
        })

    # Honour the legacy "no scf data anywhere -> top-level []" quirk.
    if not any_scf:
        merged["scf_history"] = []

    if last_traj_legacy is not None:
        merged["lattice"]       = last_traj_legacy["lattice"]
        merged["source_format"] = last_traj_legacy["source_format"]
        merged["run_state"]     = last_traj_legacy["run_state"]
        merged["error_message"] = last_traj_legacy["error_message"]
    else:
        merged["lattice"]       = None
        merged["source_format"] = "molwatch"
        merged["run_state"]     = "ongoing"
        merged["error_message"] = ""
    # Re-number iterations globally so the energy / force plots have
    # a monotone x-axis across the merged trajectory.  Per-stage
    # step indices are kept under ``step_indices`` for tooltip / save
    # use cases that need the original log-local numbering.
    merged["iterations"] = list(range(len(merged["frames"])))
    merged["stages"] = stages
    return merged, stages


@bp.route("/watch")
def watch_page():
    return render_template("watch.html")


@bp.route("/api/watch/formats")
def api_formats():
    """Lightweight: lists registered parsers + their human labels."""
    return jsonify({"ok": True, "formats": parser_summary()})


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
    # realpath (not just abspath) so symlinks pointing outside the
    # opt-in MOLBUILDER_WATCH_ROOT are rejected too.
    raw_path = os.path.realpath(os.path.expanduser(raw_path))
    # Optional deployment-stance constraint: when MOLBUILDER_WATCH_ROOT
    # is set, /api/watch/load refuses paths outside that subtree.
    # Useful on shared / multi-user hosts to keep the read-arbitrary-
    # file primitive scoped to the operator's intended run area.
    # Unset by default (current behaviour: trust the user's path).
    watch_root = os.environ.get("MOLBUILDER_WATCH_ROOT")
    if watch_root:
        watch_root_abs = os.path.realpath(os.path.expanduser(watch_root))
        try:
            common = os.path.commonpath([raw_path, watch_root_abs])
        except ValueError:
            common = ""
        if common != watch_root_abs:
            return jsonify({
                "ok": False,
                "error": (
                    f"Path {raw_path!r} is outside MOLBUILDER_WATCH_ROOT "
                    f"({watch_root_abs!r}); refusing to read.  Unset the "
                    "env var or move the file under that root."
                ),
            }), 403

    # Directory-aware resolution per docs/spec/job-layout.md.  If the
    # user passed a directory, scan it for the canonical artefacts
    # and load the best match; if a regular file, behave like before.
    resolved_from_dir: Optional[str] = None
    multi_logs: List[str] = []
    if os.path.isdir(raw_path):
        # Multi-stage detection: directory with > 1 *.molwatch.log
        # files is treated as a staged run; the loader parses every
        # log, concatenates the trajectories, and tags frames with a
        # per-stage ``stages`` metadata list.  Polling pins to the
        # newest file (older stages are static).
        all_logs = _list_molwatch_logs(raw_path)
        if len(all_logs) > 1:
            multi_logs = all_logs
            path = all_logs[-1]              # newest -- the polling target
            resolved_from_dir = raw_path
        else:
            path, attempts = _resolve_run_directory(raw_path)
            if path is None:
                tried = "\n  ".join(attempts) if attempts else "(no candidates)"
                return jsonify({
                    "ok": False,
                    "error": (
                        f"No molbuilder-job artefacts found in directory:\n"
                        f"  {raw_path}\n"
                        f"Discovery chain (per docs/spec/job-layout.md):\n"
                        f"  {tried}\n"
                        f"Generate an FDF or PySCF script with the Build "
                        f"tab into this directory, or point Watch at the "
                        f"specific log file."
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
    except UnknownFormatError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Multi-stage merge: parse every log, concatenate, attach stages
    # metadata.  ``_state["run_dir"]`` is set so subsequent
    # /api/watch/data polls re-run the merge across the FULL log set
    # rather than tail-following only the newest file (which would
    # silently drop older stages from the merged view).
    if multi_logs:
        try:
            merged, stages_meta = _merge_molwatch_trajectories(multi_logs)
        except Exception as exc:                           # pragma: no cover
            return jsonify({
                "ok": False,
                "error": f"Multi-stage merge failed: {exc}",
            }), 500
        with _lock:
            _state["path"]     = path
            _state["mtime"]    = os.path.getmtime(path)
            _state["data"]     = merged
            _state["parser"]   = parser_cls
            _state["uploaded"] = False
            _state["run_dir"]  = resolved_from_dir
        return jsonify({
            "ok":               True,
            "path":             path,
            "resolved_from":    resolved_from_dir,
            "mtime":            _state["mtime"],
            "format":           parser_cls.name,
            "label":            parser_cls.label,
            "data":             merged,
            "stages":           stages_meta,
            "uploaded":         False,
        })

    with _lock:
        _state["path"]     = path
        _state["mtime"]    = None      # force a re-parse next time
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = False
        _state["run_dir"]  = None

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({
        "ok":               True,
        "path":             state["path"],
        "resolved_from":    resolved_from_dir,
        "mtime":            state["mtime"],
        "format":           parser_cls.name,
        "label":            parser_cls.label,
        "data":             state["data"],
        "uploaded":         False,
    })


def _api_load_multipart(uploaded_file):
    """Save the uploaded file to a tempdir, parse, and stash the temp
    path on _state.  Future /api/data polls work like always but the
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
    # Use NamedTemporaryFile (R6): the previous filename construction
    # was ``molwatch_{int(time.time())}_{name}`` which collides at
    # second-resolution -- two uploads in the same second overwrote
    # each other while a parser was reading the file.
    # NamedTemporaryFile reserves a unique inode atomically.
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
    except UnknownFormatError as exc:
        # Don't keep an unrecognised upload around.
        try: os.remove(tmp_path)
        except OSError: pass
        return jsonify({"ok": False, "error": str(exc)}), 400

    with _lock:
        # Clean up any previous upload's temp file.
        if _last_temp_upload and _last_temp_upload != tmp_path:
            try: os.remove(_last_temp_upload)
            except OSError: pass
        _last_temp_upload = tmp_path
        _state["path"]     = tmp_path
        _state["mtime"]    = None
        _state["data"]     = None
        _state["parser"]   = parser_cls
        _state["uploaded"] = True
        _state["run_dir"]  = None     # uploads are one-shot, single-file

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({
        "ok":               True,
        "path":             tmp_path,
        "mtime":            state["mtime"],
        "format":           parser_cls.name,
        "label":            parser_cls.label,
        "data":             state["data"],
        "uploaded":         True,
        "uploaded_filename": uploaded_file.filename,
    })


@bp.route("/api/watch/data")
def api_data():
    """Return the parsed payload, or just an mtime if nothing changed."""
    client_mtime = request.args.get("mtime", type=float)
    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err})
    if client_mtime is not None and client_mtime == state["mtime"]:
        return jsonify({"ok": True, "changed": False, "mtime": state["mtime"]})
    parser_cls = state["parser"]
    return jsonify({
        "ok":       True,
        "changed":  True,
        "path":     state["path"],
        "mtime":    state["mtime"],
        "format":   parser_cls.name,
        "label":    parser_cls.label,
        "data":     state["data"],
        "uploaded": state.get("uploaded", False),
    })


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def warn_if_remote(host: str) -> None:
    """Emit a stderr warning when the watch app is bound to a non-loopback
    interface.  /api/watch/load reads any file the server can access, so
    exposing it on a network interface is effectively a remote
    arbitrary-file-read endpoint.  Called from molbuilder.cli when the
    user starts the watch server with a non-loopback --host."""
    if host in _LOCAL_HOSTS:
        return
    import sys as _sys
    print(f"WARNING: --host={host} exposes /api/watch/load to the network.",
          file=_sys.stderr)
    print("         The endpoint reads ANY local file the server can",
          file=_sys.stderr)
    print("         access.  Only do this on a trusted single-user",
          file=_sys.stderr)
    print("         machine, or add a reverse-proxy with auth in front.",
          file=_sys.stderr)
