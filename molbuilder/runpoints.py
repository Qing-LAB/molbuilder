"""A RUN OF POINTS' INDEX AND RULES -- the one record of a run's points as
its job sees them, and the one rule each for whether a point ran, is done,
is still to run, what the run's state is and what a point's end says
(`execution/runs-of-points.md` § 2.3, § 4).

MODULE  runpoints (floor 1 + stdlib; travels beside the job in
        ``mb_monitor.pyz`` and in every bundle whose steps write the index,
        `runwrap.MONITOR_COMPANIONS`, `runwrap.VIBRATION_COMPANIONS`)
ROLE    the run's index -- one per run, at its root, named for its first
        try (``<label>_<NN>_<stage>-run0.points.json``, :func:`index_path`)
        -- written by launch (:func:`empty_index`, :func:`add_point`),
        stamped by each point's report steps (:func:`stamp_start`,
        :func:`stamp_end`, :func:`stamp_step_error`, :func:`stamp_finish`)
        and extended by a planner's after-step (:func:`add_point`).  Each
        point's entry says what it needs, what it must hold and whether its
        verdict counts, so the job judges every point with no molbuilder
        installed; this machine reads every point from its own files and
        asks the index only what launch decided (:func:`taken_from`).  The
        rules -- :func:`rows_of`, :func:`still_to_run`, :func:`close`,
        :func:`run_state`, :func:`said` -- and the words a kind's record
        names (:func:`part`) are one for every kind: no kind, phase or file
        of one is named here.
USED-BY `jobset.submit` (launch), `jobset.points`, `jobset.continuation`,
        `jobset.migrate`, `monitor` (the report steps, a member's line,
        `notify`), `spectra.siesta_vibration` (the vibration's own steps),
        `parse.sidecars.run_points`
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# TWO WAYS, because this module travels beside a job, where the package is
# not installed.
try:                                       # inside molbuilder
    from . import persist as _persist
    from . import runfiles as _rf
    from .calcdirs import RunPlace
    from .parse.engines._run_ending import holds, not_done_why, points_state
    from .runrecord import LaunchRecordError
    #: Where this module was imported from: the package, or the bundle
    #: beside a job -- where a part the index names is resolved too
    #: (:func:`part`).
    _IN_PACKAGE = True
except ImportError:                        # beside a job
    import persist as _persist  # type: ignore[no-redef]
    import runfiles as _rf  # type: ignore[no-redef]
    from _run_ending import (holds,  # type: ignore[no-redef]
                             not_done_why, points_state)
    from runrecord import LaunchRecordError  # type: ignore[no-redef]
    from calcdirs import RunPlace  # type: ignore[no-redef]
    _IN_PACKAGE = False

SCHEMA = "molbuilder/run-points@1"

#: WHEN THE RUN IS FINISHED (`runs-of-points.md` § 1.3; the kind table's
#: ``finished``): every point DONE, or every point RAN -- converged or not.
DONE, RAN = "done", "ran"

#: WHAT A RELAUNCH DOES with a point not done (`runs-of-points.md` § 1.2):
#: runs it AGAIN from its start, or CONTINUES it from its own last state.
AGAIN, CONTINUE = "again", "continue"

#: A POINT THE ENDED WALK NEVER REACHED, as every reader says it
#: (`runs-of-points.md` § 4.5; `continuation.never_reached`): failed.
NEVER_REACHED = "the walk ended before it reached this point"

#: THE INDEXES AN OLDER MOLBUILDER WROTE (2.42-2.54): a SIESTA vibration's
#: own file of its phases, and a transport run of points' earlier schema --
#: each refused by its reader naming `jobset migrate`, which rewrites it
#: into this one (`runs-of-points.md` § 3, op. 18).
OLDER_SCHEMAS = ("molbuilder/vibration-phases@1",
                 "molbuilder/transport-points@1")
MIGRATE = ("written by molbuilder 2.54 or before -- `molbuilder jobset "
           "migrate` rewrites it")


class OlderIndexError(LaunchRecordError):
    """A RUN'S INDEX AN OLDER MOLBUILDER WROTE (:data:`OLDER_SCHEMAS`, or
    its older file, :func:`older_index`), refused by its reader naming
    `jobset migrate`, which rewrites it -- the class a reader tells that
    way on by, never the refusal's words (`runstatus`: the next step's
    line)."""


def _older(path) -> OlderIndexError:
    """The refusal of the older index at ``path``, named by its run's
    folder and its own name -- where it is said, its stage is said beside
    it."""
    p = Path(path)
    return OlderIndexError(f"{p.parent.name}/{p.name}: {MIGRATE}")


# --------------------------------------------------------------------- #
#  The file -- where it lies, read, written                              #
# --------------------------------------------------------------------- #

def index_path(run_dir, names) -> Path:
    """Where the run's index lies: at the run's root, named for its first
    try, ``<label>_<NN>_<stage>-run0.points.json`` (`runfiles.ROLE_POINTS`)
    -- launch writes it there, and every try of every point reads it."""
    return Path(run_dir) / names.name(_rf.ROLE_POINTS, run=_rf.FIRST_TRY)


def older_index(run_dir, names) -> Path:
    """Where a SIESTA vibration's index of molbuilder 2.42-2.54 lies
    (`runfiles.ROLE_PHASES`) -- what `jobset migrate` rewrites."""
    return Path(run_dir) / names.name(_rf.ROLE_PHASES, run=_rf.FIRST_TRY)


def index_of(run_dir, names) -> Optional[Path]:
    """THE RUN'S INDEX, where it has one -- ``None`` for a run launch has
    not written one for (a run prep opened and nobody launched, a rung that
    runs once).  One an older molbuilder wrote is refused by name
    (:class:`OlderIndexError`)."""
    path = index_path(run_dir, names)
    if path.is_file():
        return path
    older = older_index(run_dir, names)
    if older.is_file():
        raise _older(older)
    return None


def read_index(path) -> Dict[str, Any]:
    """The index at ``path``, refused by name when it does not read or is
    not one of this schema -- an older molbuilder's naming `jobset migrate`
    (:class:`OlderIndexError`) -- as every record of a run is
    (`runrecord.LaunchRecordError`, a ``ValueError``): said where the run
    is shown, never the whole status."""
    p = Path(path)
    try:
        d = _persist.read_json(p)
        said = d.get("schema") if isinstance(d, dict) else None
        if said not in OLDER_SCHEMAS:
            _persist.check_schema(said or "", SCHEMA, label=p.name)
    except (OSError, ValueError) as exc:
        raise LaunchRecordError(f"{p}: {exc}") from exc
    if said in OLDER_SCHEMAS:
        raise _older(p)
    return d


def write_index(path, index: Dict[str, Any]) -> Path:
    """The index written whole, atomically (`persist.write_json`)."""
    return _persist.write_json(path, index)


def index_at(run_dir, names) -> Optional[Dict[str, Any]]:
    """THE RUN'S INDEX, READ (:func:`index_of`, :func:`read_index`) --
    ``None`` for a run launch has not written one for."""
    path = index_of(run_dir, names)
    return read_index(path) if path is not None else None


def taken_from(run_dir, names, rel: str) -> Optional[str]:
    """The point the point in ``rel`` was taken over from, as launch decided
    it and wrote it into the run's index -- ``None`` for a point the run
    walks itself, or a run with no index (`runs-of-points.md` § 5)."""
    index = index_at(run_dir, names)
    if index is None:
        return None
    return (entry_of(index, rel) or {}).get("taken_from") or None


# --------------------------------------------------------------------- #
#  Written by launch, extended by a planner                              #
# --------------------------------------------------------------------- #

def empty_index(*, stage: str, record, facts=None) -> Dict[str, Any]:
    """An index with no point yet -- what launch starts from: the stage, the
    kind table's record of the pair's run of points (``record``, a
    `kind_table.RunOfPoints`: what one point is called, when the run is
    finished, its words), copied since the job has no kind table, and the
    kind's facts about the run (`PointsOf.run_facts`)."""
    return {"schema": SCHEMA, "stage": stage, "unit": record.unit,
            "finished": record.finished, "count": record.count,
            "outcome": record.outcome, "facts": dict(facts or {}),
            "points": [], "finish": None}


def add_point(index: Dict[str, Any], *, rel: str, words: str, phase: str,
              round: str, needs: Sequence[str] = (),
              must_hold: Sequence[Tuple[Any, str]] = (),
              verdict_counts: bool = True,
              planned_by: Optional[str] = None,
              taken_from: Optional[str] = None,
              facts: Optional[Dict[str, Any]] = None,
              outcome: Optional[str] = None) -> Dict[str, Any]:
    """A point of the run, as launch or its planner knows it
    (`runs-of-points.md` § 2.3), appended in the LISTING ORDER: its folder
    below the run, its words, its phase and round, the points it needs --
    each already listed, so the listing order is the order points build on
    one another in -- what it must hold (``[(product, words)]``), whether
    its verdict counts, the point that planned it, the run it was taken
    over from, the kind's facts and, a point taken over, its outcome.  A
    folder listed twice is refused."""
    have = {e["rel"] for e in index.setdefault("points", [])}
    if rel in have:
        raise ValueError(f"the run's index lists {rel!r} already")
    odd = [n for n in needs if n not in have]
    if odd:
        raise ValueError(f"{rel!r} needs {', '.join(odd)}, which the run's "
                         f"index does not list before it")
    row = {"rel": rel, "words": words, "phase": phase, "round": round,
           "needs": list(needs),
           "must_hold": [[p, w] for p, w in must_hold],
           "verdict_counts": bool(verdict_counts),
           "planned_by": planned_by, "taken_from": taken_from,
           "started": None, "ended": None, "exit": None, "verdict": None,
           "step_error": None, "not_run": None, "outcome": outcome,
           "facts": dict(facts or {})}
    index["points"].append(row)
    return row


def entry_of(index: Dict[str, Any], rel: str) -> Optional[Dict[str, Any]]:
    """The index's entry for the point in ``rel``, or ``None``."""
    return next((e for e in index.get("points") or []
                 if e.get("rel") == rel), None)


def planned_by(index: Dict[str, Any], rels) -> List[Dict[str, Any]]:
    """Every entry a point in ``rels`` planned, in the listing order -- the
    points a launch again takes with their planner (`runs-of-points.md`
    § 3, op. 2)."""
    rels = set(rels)
    return [e for e in index.get("points") or []
            if e.get("planned_by") in rels]


def _entry(index: Dict[str, Any], rel: str) -> Dict[str, Any]:
    got = entry_of(index, rel)
    if got is None:
        raise ValueError(f"the run's index lists no point {rel!r}")
    return got


# --------------------------------------------------------------------- #
#  Stamped by the report steps (and a kind's own steps)                  #
# --------------------------------------------------------------------- #

def stamp_start(index: Dict[str, Any], rel: str, at: str) -> Dict[str, Any]:
    """The point's start -- kept where a try before it in the same run
    stamped one: a point retried in place keeps its first try's start."""
    row = _entry(index, rel)
    if row.get("started") is None:
        row["started"] = at
    return row


def stamp_end(index: Dict[str, Any], rel: str, at: str, exit_code: int,
              verdict) -> Dict[str, Any]:
    """The point's end, its engine's exit and its verdict -- what its output
    says it converged (``(of, converged)``, `_run_ending.verdict_of`),
    ``None`` where it says nothing."""
    row = _entry(index, rel)
    row.update(ended=at, exit=int(exit_code),
               verdict=({"of": verdict[0], "converged": bool(verdict[1])}
                        if verdict else None))
    return row


def stamp_step_error(index: Dict[str, Any], rel: str, what: str,
                     error: Optional[str] = None) -> Dict[str, Any]:
    """What the point's own step could not do -- kept where the step said
    it first, its own words over a report step's."""
    row = _entry(index, rel)
    if row.get("step_error") is None:
        row["step_error"] = {"what": what, "error": error}
    return row


def stamp_finish(index: Dict[str, Any], started: str, ended: str,
                 exit_code: int) -> Dict[str, Any]:
    """The run's finish step, once it has ended (`runs-of-points.md`
    § 2.3)."""
    index["finish"] = {"started": started, "ended": ended,
                       "exit": int(exit_code)}
    return index


def note(index: Dict[str, Any], rel: str, *, outcome: Optional[str] = None,
         **facts) -> Dict[str, Any]:
    """What a kind's own step writes of its point: its facts, and what it
    came to in its kind's words (``outcome``)."""
    row = _entry(index, rel)
    row.setdefault("facts", {}).update(facts)
    if outcome is not None:
        row["outcome"] = outcome
    return row


# --------------------------------------------------------------------- #
#  The rules (`runs-of-points.md` § 4)                                   #
# --------------------------------------------------------------------- #

def _step_words(err) -> Optional[str]:
    if not err:
        return None
    return (f"{err['what']}: {err['error']}" if err.get("error")
            else str(err["what"]))


def rows_of(index: Dict[str, Any], run_dir) -> List[Dict[str, Any]]:
    """EACH POINT THE INDEX LISTS, JUDGED (`runs-of-points.md` § 4.1, § 4.2)
    -- in the listing order, each row the point's facts and its judgement,
    one shape on both sides: ``ran`` -- taken over, or its engine exited 0
    with no failure of its own step; ``done`` -- it ran, and the one rule
    (`_run_ending.not_done_why`) says nothing against its verdict where it
    counts and the files it must hold; ``ended`` -- it ran, failed, was
    taken over, or the walk never ran it; ``state`` its run's, as the
    status door says it -- ``finished`` (it ran), ``failed``, ``running``
    (started), ``queued``;
    ``why`` it is not done."""
    place = RunPlace.of(run_dir)
    rows = []
    for e in index.get("points") or []:
        taken = e.get("taken_from") is not None
        never = e.get("not_run")
        stamped = e.get("ended") is not None
        ended = taken or never is not None or stamped
        missing = [w for p, w in e.get("must_hold") or []
                   if not holds(place.point(e["rel"]), p)]
        if never is not None:
            ran, why = False, never
        elif taken:
            ran = True
            why = not_done_why(None, counts=False, missing=missing)
        elif stamped:
            failed = "; ".join(w for w in (
                f"exit {e.get('exit')}" if int(e.get("exit") or 0) != 0
                else None, _step_words(e.get("step_error"))) if w)
            ran = not failed
            said = e.get("verdict")
            why = failed or not_done_why(
                (said["of"], bool(said["converged"])) if said else None,
                counts=bool(e["verdict_counts"]), missing=missing)
        else:
            ran, why = False, None
        rows.append({**(e.get("facts") or {}),
                     "rel": e["rel"], "words": e["words"],
                     "phase": e.get("phase"), "round": e.get("round"),
                     "needs": list(e.get("needs") or ()),
                     "planned_by": e.get("planned_by"),
                     "taken_from": e.get("taken_from"),
                     "outcome": e.get("outcome"),
                     "started": e.get("started"), "ended_at": e.get("ended"),
                     "exit": e.get("exit"), "step_error": e.get("step_error"),
                     "ended": ended, "ran": ran,
                     "done": ended and why is None,
                     # ITS RUN'S STATE IN THE STATUS DOOR'S WORDS: a
                     # launched point not ended runs once its start is
                     # stamped, else waits in the walk.
                     "state": (("running" if e.get("started") else "queued")
                               if not ended else
                               "finished" if ran else "failed"),
                     "why": why, "not_run": never})
    return rows


def _clear(rows) -> Dict[str, bool]:
    """For each point, whether nothing up its needs' chain stops it: a need
    that has ended RAN; one that has not, its own needs clear -- asked in
    the listing order, where every need comes before what needs it."""
    clear: Dict[str, bool] = {}
    for r in rows:
        clear[r["rel"]] = (r["ran"] if r["ended"]
                           else all(clear.get(n, True) for n in r["needs"]))
    return clear


def still_to_run(rows) -> List[Dict[str, Any]]:
    """THE POINTS THE WALK MAY STILL RUN (`runs-of-points.md` § 4.3): not
    ended, and every need up its chain that has ended ran -- a device
    voltage that fails ends the chain that starts from it, a relaxation
    that fails leaves the force constants unrun."""
    rows = list(rows)
    clear = _clear(rows)
    return [r for r in rows if not r["ended"]
            and all(clear.get(n, True) for n in r["needs"])]


def why_not_run(rows, rel: str) -> str:
    """WHY THE WALK DOES NOT RUN the point in ``rel`` (`runs-of-points.md`
    § 4.5): *it builds on <words>, which failed* -- the first need up its
    chain that failed -- or :data:`NEVER_REACHED`."""
    by_rel = {r["rel"]: r for r in rows}
    queue, seen = list(by_rel[rel]["needs"]), set()
    while queue:
        need = by_rel.get(queue.pop(0))
        if need is None or need["rel"] in seen:
            continue
        seen.add(need["rel"])
        if need["ended"]:
            if not need["ran"]:
                return f"it builds on {need['words']}, which failed"
        else:
            queue.extend(need["needs"])
    return NEVER_REACHED


def close(index: Dict[str, Any], rows) -> Dict[str, Any]:
    """THE WALK OVER -- no point left to run: each point it never ran
    stamped so, ``not_run`` its why (:func:`why_not_run`), so the run's
    count and state say it failed, as the status door says a point the
    ended walk never reached."""
    rows = list(rows)
    for r in rows:
        if not r["ended"]:
            _entry(index, r["rel"])["not_run"] = why_not_run(rows, r["rel"])
    return index


def not_done_words(row: Dict[str, Any]) -> str:
    """A point not done, in the words a failed part and a failed finish name
    it by (`run-reports.md` § 2.9): its words, and why."""
    return f"{row['words']}: not done -- {row['why'] or row['state']}"


def run_state(rows, index: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """THE RUN'S STATE, once no point is left to run (`runs-of-points.md`
    § 4.4) -- one rule, the job's and `status`'s: by the index's
    ``finished`` -- every point done, or every point ran
    (`_run_ending.points_state`) -- ``finished``, else ``failed``; a finish
    step that exited non-zero, failed.  Its words: where not every point is
    done, the first not done in the listing order, and how many when more
    than one is; *the finish exited N* -- ``None`` when nothing is
    wrong."""
    rows = list(rows)
    if index["finished"] == DONE:
        state = "finished" if all(r["done"] for r in rows) else "failed"
    else:
        state = points_state(r["state"] for r in rows)
    left = [r for r in rows if not r["done"]]
    why = (not_done_words(left[0]) + (
        f"; {len(left)} of {len(rows)} {index['unit']}s not done"
        if len(left) > 1 else "")) if left else None
    fin = (index.get("finish") or {}).get("exit")
    if fin not in (None, 0):
        state = "failed"
        why = "; ".join(w for w in (why, f"the finish exited {fin}") if w)
    return state, why


def reason(row: Dict[str, Any]) -> str:
    """A FAILED PART'S REASON, the key it is said once by (`run-reports.md`
    § 2.9): its exit where not 0, else what its own step could not do, else
    the rule's words, else its state."""
    if row.get("exit") is not None and int(row["exit"]) != 0:
        return f"exit {row['exit']}"
    err = row.get("step_error")
    if err:
        return str(err["what"])
    return row.get("why") or row.get("state") or "not done"


def round_ended(rows, rel: str) -> bool:
    """Whether the round of the point in ``rel`` has no point left to run --
    its last to end has ended (`runs-of-points.md` § 4.6)."""
    rows = list(rows)
    mine = next(r for r in rows if r["rel"] == rel)
    return not any(r["round"] == mine["round"] for r in still_to_run(rows))


def round_outcome(index: Dict[str, Any], rows, rel: str, run_dir, names
                  ) -> Optional[str]:
    """THE OUTCOME OF THE ROUND of the point in ``rel``, by the record's
    ``outcome`` part (`runs-of-points.md` § 2.1) -- ``None`` where it says
    none.  Its failure is the caller's to log: words that cannot be
    composed never keep the run from speaking."""
    rows = list(rows)
    mine = next(r for r in rows if r["rel"] == rel)
    return part(index["outcome"])(index, rows, mine["round"], Path(run_dir),
                                  names)


def said(rows, rel: str, outcome: Optional[str] = None
         ) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """WHAT THE END OF THE POINT IN ``rel`` SAYS (`runs-of-points.md`
    § 4.6) -- ``(event, detail, reason)``, or ``None`` for nothing:
    ``finish`` when no point is left to run (its detail ``outcome``, the
    outcome of this point's round; the caller closes the index, runs the
    finish step and puts the run's state before it); ``milestone`` when
    its round has no point left to run (``outcome``, else *<words>:
    done*); ``part_failed`` when it is not done (once per ``reason``)."""
    rows = list(rows)
    mine = next(r for r in rows if r["rel"] == rel)
    if not still_to_run(rows):
        return "finish", outcome, None
    if round_ended(rows, rel):
        return "milestone", outcome or (f"{mine['words']}: done"
                                        if mine["done"]
                                        else not_done_words(mine)), None
    if not mine["done"]:
        return "part_failed", not_done_words(mine), reason(mine)
    return None


def outcomes(index: Dict[str, Any], rows, round_: str, run_dir, names
             ) -> Optional[str]:
    """A ROUND'S OUTCOME AS ITS POINTS SAID THEIRS -- each one's
    ``outcome``, written by its own after-step, joined -- ``None`` where
    none said one.  The record's ``outcome`` part of a kind whose points
    say their own (`runs-of-points.md` § 2.1)."""
    said_ = [r["outcome"] for r in rows
             if r["round"] == round_ and r.get("outcome")]
    return "; ".join(said_) or None


def count(index: Dict[str, Any], rows) -> str:
    """The run's count in its kind's words -- the record's ``count`` part
    over ``rows`` and the index's facts (`runs-of-points.md` § 4.7)."""
    return part(index["count"])(rows, index.get("facts") or {})


# --------------------------------------------------------------------- #
#  A part, resolved on either side                                       #
# --------------------------------------------------------------------- #

def part(ref: str) -> Any:
    """THE OBJECT A RECORD NAMES -- ``module:NAME`` -- from where this
    module itself came (`runs-of-points.md` § 2.1): the package on this
    machine; beside a job, the module the bundle holds under its last name
    -- never a molbuilder the job's environment may hold, which need not be
    the bundle's.  A name that does not resolve is refused naming it."""
    module, sep, name = str(ref).partition(":")
    if not sep:
        raise ValueError(f"{ref!r} is no part: \"module:NAME\"")
    try:
        mod = importlib.import_module(
            module if _IN_PACKAGE else module.rsplit(".", 1)[-1])
    except ImportError as exc:
        raise ValueError(f"{ref!r} does not resolve: {exc}") from exc
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise ValueError(f"{ref!r} does not resolve: {exc}") from exc


__all__ = ["SCHEMA", "DONE", "RAN", "AGAIN", "CONTINUE", "NEVER_REACHED",
           "OLDER_SCHEMAS", "MIGRATE", "OlderIndexError", "index_path",
           "older_index",
           "index_of", "read_index", "index_at", "write_index",
           "taken_from",
           "empty_index", "add_point", "entry_of", "planned_by", "why_not_run",
           "stamp_start", "stamp_end", "stamp_step_error", "stamp_finish",
           "note", "rows_of", "still_to_run", "close", "not_done_words",
           "run_state", "reason", "round_ended", "round_outcome", "said",
           "outcomes", "count", "part"]
