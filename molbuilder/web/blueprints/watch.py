"""Trajectory-loader API endpoints (live-update polling + parser
auto-detection).  Filename + module name kept as ``watch`` for
git-history continuity; the page route (``/watch``) was retired
2026-05-19 -- this module now exposes ONLY the JSON API endpoints
consumed by the /results trajectory inspector (see
``lib/inspectors/trajectory.js`` + ``lib/trajectory/core.js``).

Routes (registered with no url_prefix; each carries its own full path):

    POST /api/watch/load       JSON {"path": "..."} or multipart upload
                                ``path`` may be either a single file or
                                a run directory (job-layout v1; see
                                ``docs/protocols/job-layout.md``).
    GET  /api/watch/data       poll for changes (mtime-based)

Flow on /results: the user picks a trajectory file in the Projects
sidebar; the registry mounts the trajectory inspector; the inspector
core POSTs to /api/watch/load with the absolute path, then polls
/api/watch/data every ~15 s while the mtime advances.  The directory
branch of /api/watch/load follows the discovery chain in
``docs/protocols/job-layout.md``: ``*.molwatch.log`` first, then
``*.fdf`` parsed for SystemLabel, then ``*.py`` parsed for job_name,
then a generic ``*.out`` / ``*_geom_optim.xyz`` fallback.

Format support is plugin-style: see ``molbuilder/parsers/`` for the
registered parsers and the auto-detection registry.

State model: a single global "current file" dict guarded by a Lock.
This is intentional -- the trajectory inspector is single-user /
single-tab by design (one inspector mounted at a time; see
docs/design.md for the original rationale).
"""

from __future__ import annotations

import glob
import os
import re
import sys
import tempfile
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

from molbuilder.parsers import (
    UnknownFormatError,
    detect_parser,
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

    # Per-iter SCF wall-time tracker.  See ``_attach_iter_walltime``
    # for the algorithm: file mtime is the clock source (engines like
    # SIESTA emit per-iter timing only at end-of-run, so the .out
    # itself has no usable per-iter timestamp mid-run, but mtime
    # advances every time the engine flushes a line).  Each entry:
    # ``{"mtime": float, "step_idx": int, "iters_in_step": int}``.
    "iter_walltime_samples": [],
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


# --------------------------------------------------------------------- #
#  Directory-aware path resolution (job-layout v1)                      #
#                                                                       #
#  See ``docs/protocols/job-layout.md`` for the full contract.  When the     #
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
    # Same charset as the basename validator in
    # molbuilder/config/siesta.py (see _BASENAME_RE).
    r"^\s*SystemLabel(?:\s|\.)\s*([A-Za-z0-9_\-]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# molbuilder-generated PySCF scripts set ``job_name = "..."`` near the
# top.  The regex anchors on the LHS to avoid catching incidental
# strings further down.
_PY_JOB_NAME_RE = re.compile(
    r"^\s*job_name\s*=\s*[\"\']([A-Za-z0-9_\-]+)[\"\']",
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

    Discovery chain follows ``docs/protocols/job-layout.md`` § "How Watch
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


_ITER_WALLTIME_BUFFER_CAP = 16


def _attach_iter_walltime(
    new_data: Dict[str, Any],
    mtime: float,
    samples: List[Dict[str, Any]],
) -> None:
    """Stamp ``wall_time_per_iter_s`` onto ``new_data`` using filesystem
    mtime as the clock source.

    Why mtime instead of ``Date.now()`` / ``time.time()``:

      * The .out file's mtime IS the moment the engine last flushed.
        For SIESTA the engine emits per-iter timing only at end-of-
        run (one diagnostic line at iter 1 of step 1, then a full
        ``>>> timer`` block at finalisation), so mid-run the .out
        itself carries no usable per-iter timestamp.  But mtime
        advances every time an SCF line is written, so the file's
        own metadata IS a per-iter clock -- no browser-clock
        deduction, just filesystem state.
      * Mtime is persistent across browser reloads: the user can
        refresh the Results tab and the per-iter number survives
        (modulo the server-side ring buffer, which is process-
        lifetime).
      * Engine-agnostic: works for SIESTA, PySCF, molwatch_log,
        anything that appends lines incrementally.

    Algorithm:

      1. Pull the latest non-empty SCF step from ``new_data``: that
         step's index and its current iter count.
      2. Look backwards through ``samples`` for the most recent
         entry **with the same ``step_idx``** -- step boundaries
         reset the iter counter to 1, so cross-step deltas would
         conflate inter-step bookkeeping (DM extrapolation, mesh
         rebuild) with single-iter SCF cost.
      3. Per-iter = ``(mtime_now - mtime_prev) / (iters_now -
         iters_prev)``, only when both deltas are positive.  When
         the iter delta is > 1 (poll cadence missed one or more
         iters), the division naturally averages.
      4. Append the new sample; cap the buffer at
         ``_ITER_WALLTIME_BUFFER_CAP`` (16) to bound memory.

    On success stamps:

        new_data["wall_time_per_iter_s"]      = float  # seconds/iter
        new_data["wall_time_per_iter_window"] = {
            "iters":   int,   # iters covered by this measurement
            "seconds": float, # wall-clock span
            "step_idx": int,  # which SCF step the measurement came from
        }

    When no valid pair is available (first poll, step just changed,
    no new iters since last poll), leaves ``new_data`` untouched
    and the JS falls back to the snapshot ladder.
    """
    history = new_data.get("scf_history") or []
    step_idx = -1
    iters_in_step = 0
    for i in range(len(history) - 1, -1, -1):
        step = history[i]
        if step:
            step_idx = i
            iters_in_step = len(step)
            break
    if step_idx < 0:
        return

    # Find the most recent same-step prior sample.  Walk backwards
    # and stop at the first step-mismatch -- older samples can't be
    # mixed with newer-step deltas without crossing the boundary.
    for j in range(len(samples) - 1, -1, -1):
        prev = samples[j]
        if prev["step_idx"] != step_idx:
            break
        di = iters_in_step - prev["iters_in_step"]
        dt = mtime - prev["mtime"]
        if di >= 1 and dt > 0.0:
            new_data["wall_time_per_iter_s"] = dt / di
            new_data["wall_time_per_iter_window"] = {
                "iters":    di,
                "seconds":  dt,
                "step_idx": step_idx,
            }
            break  # use the most-recent match; older samples are noisier

    samples.append({
        "mtime":         mtime,
        "step_idx":      step_idx,
        "iters_in_step": iters_in_step,
    })
    while len(samples) > _ITER_WALLTIME_BUFFER_CAP:
        del samples[0]


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
            #
            # Compare by ``parser.name`` (the documented stable
            # identifier on TrajectoryParser subclasses -- see
            # parsers/base.py) rather than ``is`` on the class
            # object.  ``is`` works today because detect_parser
            # returns module-level class refs, but it's fragile to
            # any future detection refactor (factory functions,
            # dynamic class registration).  The name attribute is
            # the same identifier the rest of the system uses
            # (engine_metadata, error messages, etc.).
            if (_state["run_dir"] == run_dir
                    and _parser_name(_state["parser"])
                        == parser_cls.name):
                samples = _state.setdefault("iter_walltime_samples", [])
                _attach_iter_walltime(new_data, active_mtime, samples)
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
    #
    # Parser comparison by ``.name`` -- see the multi-stage branch
    # above for the rationale (``is`` works today but is fragile
    # to future detection refactors).
    with _lock:
        if (_state["path"] == path
                and _parser_name(_state["parser"]) == parser_cls.name):
            samples = _state.setdefault("iter_walltime_samples", [])
            _attach_iter_walltime(new_data, mtime, samples)
            _state["data"]  = new_data
            _state["mtime"] = mtime
        return dict(_state), None


# Per-file parse cache for the multi-stage merge.  Keyed by absolute
# path, valued by ``(mtime, legacy_dict)``.  ``_merge_molwatch_trajectories``
# reuses a cached entry when the file's mtime hasn't advanced, so a
# poll that picks up new frames in the newest stage doesn't re-parse
# the 2 or 3 prior (static) stage logs.  Single-process, single-user
# scope -- bounded by the number of stages in flight (≤ a handful).
# Multi-stage parse cache.  Per docs/protocols/results-state-contract.md
# § 6 (PR 4):
#
#   * Keyed by absolute path.
#   * Value's cache-key tuple is (mtime, size).  1-second mtime
#     granularity systems (NFS, FAT) get torn-read defense for free
#     -- any size change invalidates.
#   * LRU bound: deleted files age out naturally as new entries push
#     them off the back.
#   * Insertion is gated on the freshness check at the cache-miss
#     site: a parsed snapshot is NOT cached if the read completed
#     less than ``_MERGE_PARSE_FRESHNESS_S`` after the file's mtime
#     (the file was still being written at parse time).
_MERGE_PARSE_CACHE: "OrderedDict[str, Tuple[Tuple[float, int], Dict[str, Any]]]" = OrderedDict()
_MERGE_PARSE_CACHE_LRU_BOUND = 64
# Don't cache snapshots taken less than this many seconds after the
# file's mtime.  200ms covers the typical "atomic-replace" window
# without being so long that warm runs lose cache benefit.  See
# results-state-contract.md § 6 cache contract.
_MERGE_PARSE_FRESHNESS_S = 0.200


def _merge_parse_cache_get(path: str, mtime: float, size: int
                            ) -> Optional[Dict[str, Any]]:
    """LRU-aware cache read.  Returns the cached legacy dict iff
    the stored (mtime, size) matches; on hit, moves the entry to
    the back (most-recently-used)."""
    entry = _MERGE_PARSE_CACHE.get(path)
    if entry is None:
        return None
    if entry[0] != (mtime, size):
        return None
    _MERGE_PARSE_CACHE.move_to_end(path)
    return entry[1]


def _merge_parse_cache_put(path: str, mtime: float, size: int,
                            legacy: Dict[str, Any]) -> None:
    """Store the parsed snapshot.  Evicts the oldest entry when
    the LRU bound is exceeded."""
    _MERGE_PARSE_CACHE[path] = ((mtime, size), legacy)
    _MERGE_PARSE_CACHE.move_to_end(path)
    while len(_MERGE_PARSE_CACHE) > _MERGE_PARSE_CACHE_LRU_BOUND:
        _MERGE_PARSE_CACHE.popitem(last=False)


def _merge_parse_cache_is_fresh(mtime: float) -> bool:
    """Return True iff a snapshot taken NOW would be safe to cache
    (the file's mtime is older than the freshness window, so it
    isn't mid-write).  Pure helper -- the caller decides whether
    to skip the put."""
    return (time.time() - mtime) >= _MERGE_PARSE_FRESHNESS_S


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
        # in_progress: per-frame bool array used by the JS
        # plottableFrames filter (results-state-contract.md § 4
        # Invariant 2).  Stays [] when no stage carries an
        # in-progress frame (the typical case for completed runs).
        "in_progress": [],
    }
    any_in_progress = False
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
            sz = os.path.getsize(path)
            legacy = _merge_parse_cache_get(path, mt, sz)
            if legacy is None:
                traj   = MolwatchLogParser.parse(path)
                legacy = trajectory_to_legacy_dict(traj)
                # PR 4 (results-state-contract § 6): 200 ms
                # freshness gate.  If the file's mtime is younger
                # than _MERGE_PARSE_FRESHNESS_S, the read may have
                # raced an in-flight write -- don't poison the
                # cache; the next tick will re-parse cheaply.
                if _merge_parse_cache_is_fresh(mt):
                    _merge_parse_cache_put(path, mt, sz, legacy)
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
        # Propagate per-frame in_progress flags.  The adapter
        # collapses to [] when no frame is in-progress, so we
        # expand back to per-frame bools here for the merge, then
        # collapse again at the bottom if no stage contributed any.
        stage_in_prog = legacy.get("in_progress") or []
        if stage_in_prog:
            any_in_progress = True
            merged["in_progress"].extend(stage_in_prog)
        else:
            # Stage carried no in_progress flags; pad with False so
            # the array stays aligned 1:1 with merged["frames"].
            merged["in_progress"].extend([False] * len(legacy["frames"]))
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
        # Runtime info: use the LATEST stage's bag.  Across a multi-
        # stage run the threading/GPU/host are constant in practice
        # (same physical machine + script), so the last stage's
        # values are representative.  If a future workflow does
        # cross-host stage handoff, this is where to merge.
        merged["runtime_info"]  = dict(last_traj_legacy.get("runtime_info") or {})
    else:
        merged["lattice"]       = None
        merged["source_format"] = "molwatch"
        merged["run_state"]     = "ongoing"
        merged["error_message"] = ""
        merged["runtime_info"]  = {}
    # Re-number iterations globally so the energy / force plots have
    # a monotone x-axis across the merged trajectory.  Per-stage
    # step indices are kept under ``step_indices`` for tooltip / save
    # use cases that need the original log-local numbering.
    merged["iterations"] = list(range(len(merged["frames"])))
    merged["stages"] = stages
    # Collapse in_progress back to top-level [] when no stage
    # contributed any -- matches the single-stage adapter contract.
    if not any_in_progress:
        merged["in_progress"] = []
    return merged, stages


# /watch page route removed 2026-05-19: the trajectory inspector is
# now served by /results via the registry adapter, which fetches
# _trajectory_inspector.html from GET /partials/trajectory-inspector
# (in web/blueprints/results.py).  This module retains the
# /api/watch/* endpoints below -- those are the canonical API for
# loading + polling a trajectory file and are consumed by the
# /results adapter.  KEEP the API; the page is gone.


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

    # Directory-aware resolution per docs/protocols/job-layout.md.  If the
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
                        f"Discovery chain (per docs/protocols/job-layout.md):\n"
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
            # Fresh load -> fresh iter-walltime tracking.  Carrying
            # samples from the previous file would either compute a
            # bogus delta (different run, different SCF index space)
            # or just be dead weight.
            _state["iter_walltime_samples"] = []
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
        # Same reasoning as the multi-stage branch: fresh load,
        # fresh samples.  See ``_attach_iter_walltime`` for why
        # cross-file deltas would be nonsense.
        _state["iter_walltime_samples"] = []

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
        _remove_temp_quietly(tmp_path)
        return jsonify({"ok": False, "error": str(exc)}), 400

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
        _state["run_dir"]  = None     # uploads are one-shot, single-file
        _state["iter_walltime_samples"] = []

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


# ``warn_if_remote()`` + ``_LOCAL_HOSTS`` (legacy helpers that printed a
# stderr warning when --host bound a non-loopback interface) were
# removed 2026-05-19 along with the ``molbuilder watch serve`` CLI
# subcommand: ``molbuilder serve`` is the canonical entry point now,
# and its ``_enforce_tls_for_remote_bind`` guard (in cli.py) already
# refuses a non-loopback bind without TLS or
# ``--allow-insecure-binding``.  The arbitrary-file-read concern
# documented here applies to /api/watch/* regardless of which CLI
# command started the server -- see docs/deployment.md for the
# recommended reverse-proxy + auth shape.
