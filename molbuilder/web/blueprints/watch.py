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
                                ``docs/execution/job-contracts.md``).
    GET  /api/watch/data       poll for changes (mtime-based)

Flow on /results: the user picks a trajectory file in the Projects
sidebar; the registry mounts the trajectory inspector; the inspector
core POSTs to /api/watch/load with the absolute path, then polls
/api/watch/data every ~15 s while the mtime advances.  The directory
branch of /api/watch/load follows the discovery chain in
``docs/execution/job-contracts.md``: ``*.molwatch.log`` first, then
``*.fdf`` parsed for SystemLabel, then ``*.py`` parsed for ``JOB``,
then a generic ``*.out`` / ``*_geom*_optim.xyz`` fallback.

Format support is plugin-style: see ``molbuilder/parse/`` for the
registered parsers and the auto-detection registry
(``model/parse.md`` § 3).

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
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

from molbuilder.parse import (
    ParseError,
    detect as detect_parser,
)
from molbuilder.parse.contract import engine_of
from molbuilder.parse.dirs.run_info import run_info_for_dir
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

    # Multi-stage merge state.  Set when the user loaded a directory
    # containing > 1 *.molwatch.log files; ``_refresh_if_changed``
    # poll doesn't clobber the merged trajectory with the newest
    # log's frames alone.

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
#  See ``docs/execution/job-contracts.md`` for the full contract.  When the     #
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
# molbuilder-generated PySCF scripts set ``JOB = "..."`` near the top
# (see pyscf/input.py:329).  The regex anchors on the LHS to avoid
# catching incidental strings further down.
_PY_JOB_NAME_RE = re.compile(
    r"^\s*JOB\s*=\s*[\"\']([A-Za-z0-9_\-]+)[\"\']",
    re.MULTILINE,
)


def _read_text_safely(path: str, max_bytes: int = 65536) -> str:
    """Read up to ``max_bytes`` of ``path`` and return as text.  Used
    for the .fdf / .py header sniff -- we only need the first chunk
    to find SystemLabel / ``JOB``; reading the whole multi-MB FDF
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


def _resolve_run_directory(directory: str) -> Tuple[Optional[str], List[str]]:
    """Resolve a run directory to a single file Watch should load.

    Returns ``(resolved_path, attempts)``: the resolved file path
    (or ``None`` if nothing was found), plus a list of human-readable
    "tried X" strings used to build the error message when nothing
    matches.

    Discovery chain follows ``docs/execution/job-contracts.md`` § "How Watch
    resolves a directory":

      1. Any ``*.molwatch.log`` (newest wins for staged runs).
      2. ``*.fdf`` -> parse SystemLabel -> ``<label>.molwatch.log``,
         ``<label>.out``.
      3. ``*.py``  -> parse ``JOB``       -> ``<job>.molwatch.log``,
         ``<job>.log``, ``<job>_geom_optim.xyz`` — and the deck
         FILENAME's stem tried the same way, because a staged deck is
         ``<job>_<token>.py`` and its stdout/molwatch siblings carry
         that token (`job-contracts.md` § 6.3) while ``JOB`` stays
         bare; then the rung-aware trajectory glob
         ``<job>_geom_*_optim.xyz``.
      4. Generic fallbacks: ``run.out``, ``siesta.log``, ``*.out``,
         ``*_geom*_optim.xyz`` (staged trajectories carry the rung
         token between ``_geom`` and ``_optim``).
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

    # 3. PySCF: *.py -> JOB -> sibling outputs.
    py_hits = glob.glob(os.path.join(directory, "*.py"))
    attempts.append(f"*.py -> {len(py_hits)} match(es)")
    for py in py_hits:
        name = _basename_from_py(py)
        if not name:
            attempts.append(f"  {os.path.basename(py)}: JOB not found")
            continue
        # A staged deck is ``<job>_<token>.py`` and its stdout / molwatch
        # siblings are stemmed on THAT (token included), while JOB
        # stays the bare ``<job>`` -- so the deck filename's stem is
        # tried alongside the parsed name (found 2026-08-19: every
        # staged spelling here was the unstaged one, and a staged run
        # without a molwatch seed resolved to nothing).
        py_stem = os.path.splitext(os.path.basename(py))[0]
        stems = [name] if py_stem == name else [name, py_stem]
        for stem in stems:
            for suffix in (".molwatch.log", ".log", "_geom_optim.xyz"):
                cand = os.path.join(directory, f"{stem}{suffix}")
                attempts.append(f"  -> {stem}{suffix}: "
                                f"{'found' if os.path.isfile(cand) else 'missing'}")
                if os.path.isfile(cand):
                    return cand, attempts
        rung_hits = glob.glob(
            os.path.join(directory, f"{name}_geom_*_optim.xyz"))
        attempts.append(f"  -> {name}_geom_*_optim.xyz: "
                        f"{len(rung_hits)} match(es)")
        if rung_hits:
            return _newest(rung_hits), attempts

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
    # ``*_geom*_optim.xyz``: the staged spelling carries the rung token
    # between ``_geom`` and ``_optim`` (``<job>_geom_<token>_optim.xyz``);
    # the tokenless glob that stood here matched only unstaged runs.
    optim_hits = glob.glob(os.path.join(directory, "*_geom*_optim.xyz"))
    if optim_hits:
        attempts.append(f"*_geom*_optim.xyz -> "
                        f"picked {os.path.basename(optim_hits[0])}")
        return _newest(optim_hits), attempts

    return None, attempts


def _atom_metadata_json(
    search_dir: Optional[str], data: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Recover the run's embedded per-atom metadata (region labels /
    frozen tags / annotation channels) as a JSON string, so the Results-tab
    MolView carries it despite loading *coordinates* from the output logs.

    All the real work -- finding the input script, parsing the
    ATOM-METADATA block, guarding the atom count -- lives in
    :func:`molbuilder.parse.dirs.atom_metadata.atom_metadata_json_for_run_dir`
    (the directory-scoped recovery helper).  This is pure results-adapter
    glue: it sources frame-0's atom count from the parsed trajectory and
    hands it in as the guard.  ``None`` when the run carries no block."""
    from molbuilder.parse.dirs.atom_metadata import (
        atom_metadata_json_for_run_dir,
    )
    frames = (data or {}).get("frames")
    n0 = len(frames[0]) if frames else None
    return atom_metadata_json_for_run_dir(search_dir, n0)


_ITER_WALLTIME_BUFFER_CAP = 16


def _run_periodicity_json(
    search_dir: Optional[str],
    data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The run's periodicity, composed ON THE SERVER (2026-08-20).

    Two facts from two places, merged where both are visible: the CELL from
    the run's own output logs (the frames' real box -- ``data["lattice"]``),
    and the AXIS KINDS / origin / vacuum from the run directory's
    ``.source`` pair when one exists (job-contracts § 6.3: the structure the
    calculation was prepped from, carrying the user's stated intent).

    Until this existed the browser composed ``{cell}`` alone, the axis
    kinds never reached the viewer, and an export from the Results tab
    stamped a lattice-bearing junction ``isolated`` on every axis -- the
    BDT-Au111 frame38 export that surfaced the hole.  A run made before the
    ``.source`` convention has no pair; the cell still travels, and the
    load door's own rule (a stated cell over never-stated axes derives
    periodic) covers the rest.

    Returns ``None`` when neither source knows anything; never raises -- a
    broken pair degrades to the lattice-only block rather than taking the
    load down.
    """
    out: Dict[str, Any] = {}
    lattice = (data or {}).get("lattice")
    if isinstance(lattice, list) and len(lattice) == 3:
        out["cell"] = lattice
    try:
        if search_dir:
            import glob as _glob
            pairs = sorted(_glob.glob(
                os.path.join(search_dir, "*.source.xyz")))
            if pairs:
                from pathlib import Path as _Path

                from molbuilder.workingcopy_structure import StructureCodec
                s = StructureCodec().read(_Path(pairs[0]))
                if s.axis_kind:
                    out["axis_kind"] = list(s.axis_kind)
                if s.cell_origin is not None:
                    out["cell_origin"] = [float(v) for v in s.cell_origin]
                if s.vacuum is not None:
                    out["vacuum"] = [float(v) for v in s.vacuum]
                if "cell" not in out and s.cell is not None:
                    out["cell"] = [[float(x) for x in row] for row in s.cell]
    except Exception:                        # noqa: BLE001
        pass
    return out or None


def _refuse_if_not_a_trajectory(parser_cls):
    """``None`` if this parser answers a trajectory, else a 400 body.

    `/api/watch/*` is the TRAJECTORY route: everything after detection
    reads ``.frames``.  It never checked what the detected parser
    actually produces, so handing it a single-geometry file raised
    ``AttributeError: 'StructureResult' object has no attribute
    'frames'`` -- a 500 for a file the app itself writes and the parser
    reads perfectly.

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
    from molbuilder.parse.types import TrajectoryResult

    out = getattr(parser_cls, "output", None)
    if out is not None and issubclass(out, TrajectoryResult):
        return None
    kind = getattr(out, "__name__", "an unknown result")
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
    PARSER that read the file."*  The route sent the parser's name for
    both until 2026-09-04.  They coincide for an engine-native file -- a
    SIESTA `.out` is read by the parser called `siesta` -- and diverge
    for the canonical `.molwatch.log`, read by the parser called
    `molwatch` whatever wrote it, so every molbuilder-generated run
    arrived as "molwatch" and the viewer's engine-specific SCF banner
    ("SIESTA DFT SCF progress / CG/MD step") fell through to its neutral
    branch.

    NOTHING IS COMPUTED HERE.  The engine is a property of the RUN
    DIRECTORY, declared when its deck was generated, and
    `running-a-job.md` § 4.2 owns the resolution order;
    `parse.contract.engine_of` is its one implementation.  It is asked
    about the same directory the neighbouring `_run_metadata` searches,
    so both facts this response carries about the run come from one
    place.

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


def _run_metadata(
    search_dir: Optional[str], data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """The metadata block EVERY ``/api/watch/load`` answer carries.

    Three builders answer this route -- multi-log, single-file, upload --
    and each used to compose the block itself: two of them from two
    DIFFERENT directory rules, and the upload one not at all.  What a
    load carried therefore depended on which branch had built it.  One
    composer, keyed by the directory to search, is the whole fix: a
    builder spreads it, and the upload branch (which has no run
    directory) answers "nothing available" DELIBERATELY -- ``None`` in
    every field -- rather than by omission.

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
    return {
        # Per-atom metadata (region labels / frozen tags / annotation
        # channels) the Build tab embedded in the run's input script:
        # coordinates come from the output logs, the labels from the
        # .fdf / .py.  A JSON string, applied downstream through
        # apply_to_structure; None when the run carries no block.
        "atom_metadata": _atom_metadata_json(search_dir, data),
        # The run's periodicity (the cell from the output logs, the axis
        # kinds from the run dir's .source pair).  The viewer passes it
        # through verbatim -- guessing periodicity in the browser is the
        # one thing the Cell rules refuse.
        "periodicity":   _run_periodicity_json(search_dir, data),
        # What the run says ABOUT itself: today the electronic contract
        # its deck records, as `info.calculation`.  Rides installMolecule
        # in and exportFile out (molview.md § 8.4a), so an export from a
        # results view carries the contract and a transport citation of
        # that pair seals its fields rather than leaving them open.
        "info":          run_info_for_dir(search_dir),
    }


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
    drop the lock during the actual parse.  After parsing we re-acquire
    and only commit the result if the active file hasn't changed under us
    (defensive against a /api/load racing with a /api/data poll).

    **Dropping the lock does NOT stop this blocking other requests, and
    the sentence here used to claim it did** (measured 2026-09-03).  The
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
    # 2026-06-18 security fix (audit B1): the JSON-path mode now
    # routes through the canonical ``_resolve_within_roots`` helper
    # like every other path-taking endpoint, per web-api.md § 2.1.
    # Pre-fix this site used ``os.path.realpath(expanduser(...))``
    # with an OPTIONAL ``MOLBUILDER_WATCH_ROOT`` gate that was unset
    # in the default deployment — a logged-in user could POST
    # ``{"path": "/etc/shadow"}`` and the parser would read it.
    # ``_resolve_within_roots`` constrains to picker roots
    # (Capabilities.file_picker_roots(); today: <cwd>/projects);
    # the per-endpoint MOLBUILDER_WATCH_ROOT env var is retired in
    # favour of the deployment-wide picker roots configuration.
    from .files import _resolve_within_roots, _PickerError
    try:
        raw_path = str(_resolve_within_roots(raw_path))
    except _PickerError as exc:
        return jsonify({"ok": False, "error": exc.message}), exc.status

    # Directory-aware resolution per docs/execution/job-contracts.md.  If the
    # user passed a directory, scan it for the canonical artefacts
    # and load the best match; if a regular file, behave like before.
    resolved_from_dir: Optional[str] = None
    if os.path.isdir(raw_path):
        # A DIRECTORY RESOLVES TO ONE FILE.  The generated deck tells the
        # user to point Watch at the run directory and says what happens:
        # "the loader resolves it to <job>.molwatch.log".  That is this
        # chain, and it is the whole of it.
        #
        # A "> 1 molwatch log means merge them" branch stood here from
        # 2026-05-10 until 2026-09-05 and is deleted, not moved: STAGES ARE
        # SEPARATE RUNS.  A ladder is separated by filename in a flat
        # directory or by directory name in a hierarchical one, and the
        # person picks one stage and judges it.  Stitching them into one
        # trajectory is not a view this project offers -- that is what the
        # bench summary is for, where comparison IS the question.
        path, attempts = _resolve_run_directory(raw_path)
        if path is None:
            tried = "\n  ".join(attempts) if attempts else "(no candidates)"
            return jsonify({
                "ok": False,
                "error": (
                    f"No molbuilder-job artefacts found in directory:\n"
                    f"  {raw_path}\n"
                    f"Discovery chain (per docs/execution/job-contracts.md):\n"
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
    except ParseError as exc:
        # ParseError, NOT just UnknownFormatError.  `detect()` also
        # raises AmbiguousFormatError when two parsers claim one
        # file, and that is its SIBLING, not its subclass
        # (`parse/errors.py`).  Catching only the one turned a
        # registry overlap into an unhandled exception -- an HTTP
        # 500 with an HTML body -- for a file both parsers could
        # read.  Measured on `<job>_optimized.xyz`, which PySCF
        # writes for every optimization.  The message names the
        # clashing parsers, so a 400 carrying it is useful where a
        # 500 was not.
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
        # Fresh load, fresh samples.  See ``_attach_iter_walltime`` for why
        # cross-file deltas would be nonsense.
        _state["iter_walltime_samples"] = []

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
    # Metadata search dir: the resolved run directory, else the parent of
    # the file we loaded (Watch was pointed straight at a log inside a run
    # dir).  The directory the resolved log sits in.
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
        **_run_metadata(resolved_from_dir or os.path.dirname(path),
                        state["data"]),
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
        _state["iter_walltime_samples"] = []

    state, err = _refresh_if_changed()
    if err:
        return jsonify({"ok": False, "error": err}), 500
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
        # An upload is one file with no run directory behind it, so it
        # has nothing to say about itself -- and it SAYS so, in the same
        # fields the other two builders answer.  One route, one response
        # shape: a reader learns what a load answers from one place, and
        # "nothing available" is a stated answer rather than a field a
        # caller has to notice is missing.
        **_run_metadata(None, state["data"]),
    })


@bp.route("/api/watch/data")
def api_data():
    """Return the parsed payload, or just an mtime if nothing changed."""
    client_mtime = request.args.get("mtime", type=float)
    state, err = _refresh_if_changed()
    if err:
        # web-api.md § 1, *Status codes* -- server fault: parse / IO error on a
        # user-selected trajectory file.  The sibling /api/watch/load
        # returns 500 on the same failure class (line 884); aligning
        # this site closes the inconsistency that motivated that rule's
        # codification.  JS poll-loop reads body.ok so its behaviour
        # is unchanged; external consumers (curl / CI / monitoring)
        # gating on HTTP status now see the actual failure.
        return jsonify({"ok": False, "error": err}), 500
    if client_mtime is not None and client_mtime == state["mtime"]:
        return jsonify({"ok": True, "changed": False, "mtime": state["mtime"]})
    parser_cls = state["parser"]
    return jsonify({
        "ok":       True,
        "changed":  True,
        "path":     state["path"],
        "mtime":    state["mtime"],
        # An UPLOAD has no run directory, and `os.path.dirname` of its
        # temp path is the system temp dir -- shared, and full of other
        # people's files.  Asking it produced an engine decided by
        # unrelated litter (and read every `*.py`/`*.fdf`/`*.run.sh` in
        # /tmp on every poll).  The load path passes None here for the
        # same reason; the poll must agree with it or one file gets two
        # answers.
        "format":   _engine_of(
            None if state.get("uploaded")
            else os.path.dirname(state["path"]),
            state["data"], parser_cls),
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
# command started the server -- see docs/ops/deployment.md for the
# recommended reverse-proxy + auth shape.
