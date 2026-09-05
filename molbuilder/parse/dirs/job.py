"""``run_status`` — how a run directory is doing.

``{state, detail, last_change_at, active_source}``, and nothing else.

**The status is the parsers' own answer.**  Every engine parser already
reports how its file ended -- ``run_state``, ``model/parse.md`` § 2b --
so this asks the registry for each result file (every ``.out``, and each
``*.molwatch.log`` whose footer concludes the run: the engine-neutral
end-of-run marker, and the only one a PySCF attempt has).  It never
opens an engine output directly.

Two things a parser cannot know are settled here, because they are not
in the file:

* **which file speaks for the directory** -- a folder holds one ``.out``
  per run index and one molwatch log per stage;
* **staleness** -- no ending marker and no growth is a dead job, not a
  slow one, and only the filesystem can tell those apart.

*(Until 2026-09-04 this module was a ``JobDirParser`` returning an
eleven-field ``JobResult``: job type, system label, geometry, plots,
progress, a source-file index, a per-stage input summary, diagnostics.
Measured across the tree, ten of the eleven had no reader anywhere, and
the eleventh -- this one -- was obtained by parsing every ``.out`` to
build plot data and then throwing the plots away.  1,414 lines produced
one field that was used.  The dead half is deleted rather than fixed;
four code-quality defects went with it, including a second
``LatticeConstant`` reader that disagreed with its sibling on units and
a second ``SystemLabel`` regex that returned a different answer.)*
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from molbuilder.identity import parse_stage_token


# Run-state detection is fully delegated to the engine trajectory
# parsers (detect().parse() -> traj.run_state); the end-of-run and
# failure markers live in engines/siesta.py + engines/pyscf.py, NOT
# here — the decoder never greps .out content itself (enforced by
# the engine parsers own it).  This cited
# `test_no_direct_out_grep_in_decoder` as the enforcing test until
# 2026-09-05; that lint was retired with the decoder on 2026-09-04 --
# its whole body was `assert src.count("read_text") < 8`.

# Default CG-step threshold for cg_step_milestone events.


# ---- helpers --------------------------------------------------------- #


def _detect_stage(filename: str) -> Optional[int]:
    """A file's stage ORDINAL, or ``None`` when it carries no stage token.

    The token is ``<NN>_<name>`` (``bdt_au_01_coarse.fdf``) and this returns
    the ``NN``.  Read through :func:`molbuilder.identity.parse_stage_token`,
    which is the one place the shape is written down -- the decoder used to
    carry its own ``-stage(N)`` regex, a second spelling of the emitter's
    convention that could and did drift from it.

    **Still an int, and deliberately so.**  Decision 27 kept the ordinal in
    the filename, so ordering by stage stays possible: this function is the
    first component of the active-file sort key in :func:`run_status`
    (stage first, mtime second), which is what makes a re-run of an earlier
    stage stop hijacking the run's reported state.

    *This cited ``_anchor_sort_key`` and "the ``stage`` field of the
    engine-input envelope" as the other downstream orderers until
    2026-09-05.  Neither exists: the envelope went with the run decoder on
    2026-09-04, and ``_anchor_sort_key`` has never been defined anywhere in
    the tree -- it appeared only in this sentence.*  Had the token carried
    the name alone, the anchor rule would have lost its sort key and the
    Results tab its notion of "the active stage"; that is the trap
    ``staged-runs-implementation-plan.md`` § 8d walked into and § 8e closed.
    """
    hit = parse_stage_token(filename)
    return hit[0] if hit else None


def _iso_z(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


# ---- file enumeration ------------------------------------------------ #


def _enumerate_files(run_dir: Path) -> Dict[str, List[Path]]:
    """Bucket relevant files in the dir by kind.

    Returns {"fdf": [...], "out": [...], "xv": [...], "struct_out": [...],
             "molstruct_json": [...], "ani": [...], "molwatch": [...]}.
    Paths sorted by name within each bucket.
    """
    by_kind: Dict[str, List[Path]] = {
        "fdf": [], "out": [], "xv": [], "struct_out": [],
        "molstruct_json": [], "ani": [], "molwatch": [],
    }
    for child in sorted(run_dir.iterdir()):
        if not child.is_file():
            continue
        name = child.name
        if name.endswith(".fdf"):
            by_kind["fdf"].append(child)
        elif name.endswith(".out"):
            by_kind["out"].append(child)
        elif name.endswith(".XV"):
            by_kind["xv"].append(child)
        elif name.endswith(".STRUCT_OUT"):
            by_kind["struct_out"].append(child)
        elif name.endswith(".molstruct.json"):
            by_kind["molstruct_json"].append(child)
        elif name.endswith(".ANI"):
            by_kind["ani"].append(child)
        elif name.endswith(".molwatch.log"):
            by_kind["molwatch"].append(child)
    return by_kind


# ---- plots from .out files ------------------------------------------- #


def _molwatch_conclusions(mw_paths: List[Path]) -> Dict[str, str]:
    """The CONCLUDED molwatch logs' run-states, by filename.

    A molwatch log is the engine-neutral end-of-run channel
    (``running-a-job.md`` § 4): its writer appends a conclusion footer
    when the run ends, so a log carrying one is a result file and its
    run-state counts.  One without a footer is a live view — a prep-time
    seed, or a run still going — and is deliberately NOT in the answer:
    feeding it into the state would let a stage's seed outvote its own
    ``.out``.  Fail-soft like the ``.out`` path: a log the registry
    cannot read simply contributes nothing.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError
    states: Dict[str, str] = {}
    for path in mw_paths:
        try:
            traj = detect(path).parse(path)
        except (ParseError, OSError, ValueError):
            continue
        if (traj.run_state or "") in ("ended", "stopped",
                                      "out_of_memory"):
            states[path.name] = traj.run_state
    return states


# ---- status + progress ---------------------------------------------- #


def run_status(run_dir) -> Dict[str, Any]:
    """How is this run doing?  ``{state, detail, last_change_at,
    active_source}``.

    **The status IS the parser's answer**, plus the two things no parser
    can know.  Every engine parser already reports how its file ended --
    ``run_state`` on the result, ``model/parse.md`` § 2b -- so this asks
    them and then settles the two questions a single file cannot:

    * **which file speaks for the directory.**  A folder holds one
      ``.out`` per run index and one molwatch log per stage; a parser
      sees one file and cannot pick.  Highest stage, newest mtime.
    * **staleness.**  A file with no ending marker is honestly
      "running" -- nothing IN it separates a slow DFT step from a job
      the scheduler killed.  Only the filesystem can, so the age check
      lives here (``_build_status``).

    Callers wanted exactly this and had to take it out of an
    eleven-field summary: ``decode_run_dir`` answered ``status`` plus
    ten fields with no reader anywhere, and reached the per-file
    run-states by building every PLOT and discarding them.
    """
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir)
    out_states = _out_conclusions(files["out"])
    mw_states = _molwatch_conclusions(files["molwatch"])
    return _build_status(
        files["out"] + [p for p in files["molwatch"] if p.name in mw_states],
        {**out_states, **mw_states})


def _out_conclusions(out_paths: List[Path]) -> Dict[str, str]:
    """Each ``.out``'s run-state, by filename, straight from its parser.

    Fail-soft, exactly as the molwatch sibling is: a file the registry
    cannot read contributes nothing rather than taking the walk down.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError
    states: Dict[str, str] = {}
    for path in out_paths:
        try:
            states[path.name] = detect(path).parse(path).run_state or "unknown"
        except (ParseError, OSError, ValueError):
            continue
    return states


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str]
                  ) -> Dict[str, Any]:
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``.out`` plus each concluded molwatch log
    (``running-a-job.md`` § 4)."""
    if not out_paths:
        return {
            "state":            "running",
            "detail":           "no result file yet",
            "last_change_at":   None,
            "active_source":    None,
        }
    # Active source = highest stage, latest mtime.
    sorted_outs = sorted(
        out_paths,
        key=lambda p: (_detect_stage(p.name) or 0, p.stat().st_mtime),
    )
    active = sorted_outs[-1]
    active_state = out_run_states.get(active.name, "unknown")

    state = "running"
    detail = "running"
    age_s = max(0.0, _wall_now() - active.stat().st_mtime)
    # THIS LAYER SETTLES WHAT CONTENT CANNOT (`model/parse.md` § 2b).  The
    # engine parser reports how the run ENDED from markers alone --
    # "running"|"ended"|"stopped"|"out_of_memory"|"unknown" -- and a file
    # with no ending marker is honestly "running": nothing IN it can tell
    # a slow DFT step from a job the scheduler killed.  Only the
    # filesystem can, so the age check lives here and nowhere else.
    #
    # Note what is NOT consulted: whether the SCF converged.  That is
    # P-S2's reported fact, carried beside this state for the reader to
    # show, never folded into it.
    if active_state == "ended":
        state, detail = "finished", "job_completed"
    elif active_state == "out_of_memory":
        state, detail = "failed", "out of memory"
    elif active_state == "stopped":
        state, detail = "failed", "stopped before its end -- see the .out"
    elif age_s > 60.0:
        # No ending marker and no growth: it is not running any more.
        state, detail = "stale", f"no file growth in {int(age_s)}s"

    return {
        "state":          state,
        "detail":         detail,
        "last_change_at": _iso_z(active.stat().st_mtime),
        "active_source":  active.name,
    }


def _wall_now() -> float:
    """Wall-clock now() in POSIX seconds.  Indirected so tests can
    monkeypatch without touching time.time globally."""
    import time
    return time.time()
