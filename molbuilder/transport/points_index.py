"""A TRANSPORT RUN OF POINTS' INDEX -- every point of a device or
transmission run, what makes each done, and how each ended as its report
steps saw it (`engines/transport.md` § 2a.11; `execution/run-reports.md`
§ 2.8, § 2.9).

MODULE  transport.points_index (floor 1 + stdlib; travels in
        ``mb_monitor.pyz``, `runwrap.MONITOR_COMPANIONS`)
ROLE    the run's index -- one per run, at its root, named for its first
        try (``<label>_<NN>_<stage>-run0.points.json``, :func:`index_path`,
        :func:`write_index` / :func:`read_index`): launch writes
        every point of the run in the walk's order -- its folder, words,
        frame and voltage, the point it needs, whether it was taken over and
        from where, what it must hold to be done (`continuation.must_hold`)
        -- and what the run's points compute (the kind table's
        ``computes``), the leads' pair, the spin and the frames' weights;
        each point's report steps stamp its start, its end, its engine's exit
        and its verdict (:func:`stamp_start`, :func:`stamp_end`).  Its rows
        by the one done rule (:func:`rows_of`), the points still to run
        (:func:`still_to_run`), the count (:func:`summary`) and a round's
        outcome (:func:`round_words`) are what the report steps send, a
        member monitor's line says and `status` counts.  **What the job saw,
        never a second record of a point's state**: on this side every
        reader reads the points' own files (`continuation.done`); the
        index carries the same pieces for the job's side, which has no
        molbuilder.
USED-BY `jobset.submit`, `jobset.transport_points`, `transport.record`,
        `monitor`
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# TWO WAYS, because this module travels beside a job (`runwrap.
# MONITOR_COMPANIONS`), where the package is not installed.
try:                                       # inside molbuilder
    from ..persist import check_schema, read_json, write_bytes, json_text
    from ..runfiles import FIRST_TRY, ROLE_POINTS
    from ..parse.engines._run_ending import (holds, not_done_why,
                                              points_state)
    from ..parse.dirs.job import FINAL_STATES
except ImportError:                        # beside a job
    from persist import (check_schema, read_json,  # type: ignore[no-redef]
                         write_bytes, json_text)
    from runfiles import FIRST_TRY, ROLE_POINTS  # type: ignore[no-redef]
    from _run_ending import (holds,  # type: ignore[no-redef]
                             not_done_why, points_state)
    from job import FINAL_STATES  # type: ignore[no-redef]

SCHEMA = "molbuilder/transport-points@1"

#: WHAT A RUN'S POINTS COMPUTE, the kind table's words for the rung
#: (``computes``, `transport.stages.STAGE_FACT`): an SCF -- the device's
#: NEGF loop -- or a product of one, the transmission.  What a round's
#: milestone says of each point is read by this (:data:`_SAID`).
SCF, PRODUCT = "scf", "product"


# --------------------------------------------------------------------- #
#  The index -- written by launch, stamped by each point's report steps  #
# --------------------------------------------------------------------- #

def index_path(run_dir, names) -> Path:
    """Where the run's index lies: at the run's root, named for its first
    try, ``<label>_<NN>_<stage>-run0.points.json`` (`runfiles.ROLE_POINTS`)
    -- launch writes it there, and every point's report steps read it."""
    return Path(run_dir) / names.name(ROLE_POINTS, run=FIRST_TRY)


def empty_index(*, stage: str, computes: str, pair: str,
                leads: Sequence[str], spin: str,
                weights: Optional[Sequence[float]]) -> Dict[str, Any]:
    """An index with no point yet -- what launch starts from: the rung, what
    its points compute (:data:`SCF`, :data:`PRODUCT`), the leads' pair and
    each lead as their ``TS.Elecs`` blocks name them, the spin the decks
    state (``non-polarized`` or ``polarized``) and the frames' weights where
    the cited set states them, else ``None``."""
    return {"schema": SCHEMA, "stage": stage, "computes": computes,
            "pair": pair, "leads": list(leads), "spin": spin,
            "weights": list(weights) if weights is not None else None,
            "points": []}


def add_point(index: Dict[str, Any], *, rel: str, words: str,
              frame: Optional[int], frame_words: Optional[str],
              bias_v: float, needs: Optional[str], must_hold,
              verdict_counts: bool,
              window: Optional[Sequence[float]] = None,
              taken_from: Optional[str] = None) -> Dict[str, Any]:
    """A point of the run, as launch knows it: its folder below the run, its
    words and its frame's (`transport.stages.Point.words`,
    `structure.frame_words`), the voltage it runs at, the point it needs
    (``rel``; ``None`` where it needs none this run walks), what it must
    hold to be done, each with the words its absence is said in
    (`continuation.must_hold`), whether its verdict counts toward done, the
    energy grid its deck describes -- ``(from, to, points)``, a
    transmission point's, ``None`` for a device's (`energies.deck_window`)
    -- and the run it was taken over from."""
    row = {"rel": rel, "words": words, "frame": frame,
           "frame_words": frame_words, "bias_v": float(bias_v),
           "needs": needs,
           "must_hold": [[p, w] for p, w in must_hold],
           "verdict_counts": bool(verdict_counts),
           "window": list(window) if window is not None else None}
    if taken_from is not None:
        row["taken_from"] = taken_from
    index.setdefault("points", []).append(row)
    return row


def read_index(path) -> Dict[str, Any]:
    """The index at ``path``, refused by name when it is not one of this
    schema."""
    p = Path(path)
    try:
        d = read_json(p)
        check_schema(d.get("schema") if isinstance(d, dict) else "",
                     SCHEMA, label=p.name)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{p.name}: {exc}") from exc
    return d


def write_index(path, index: Dict[str, Any]) -> Path:
    """The index written whole, atomically (`persist.write_bytes`)."""
    p = Path(path)
    write_bytes(p, json_text(index).encode("utf-8"))
    return p


def entry_of(index: Dict[str, Any], rel: str) -> Optional[Dict[str, Any]]:
    """The index's entry for the point in ``rel``, or ``None``."""
    return next((r for r in index.get("points") or []
                 if r.get("rel") == rel), None)


def stamp_start(index: Dict[str, Any], rel: str, at: str) -> Dict[str, Any]:
    """The point's start, as its first report step saw it -- kept where a
    try before it in the same run stamped one: a point retried in place
    keeps its first try's start (`run-reports.md` § 2.8)."""
    row = _entry(index, rel)
    row.setdefault("started", at)
    return row


def stamp_end(index: Dict[str, Any], rel: str, at: str, exit_code: int,
              verdict) -> Dict[str, Any]:
    """The point's end, its engine's exit and its verdict -- what its output
    says it converged (`_run_ending.verdict_of`), ``None`` where it says
    nothing (TBtrans runs no SCF)."""
    row = _entry(index, rel)
    row.update(ended=at, exit=int(exit_code),
               verdict=({"of": verdict[0], "converged": bool(verdict[1])}
                        if verdict else None))
    return row


def _entry(index: Dict[str, Any], rel: str) -> Dict[str, Any]:
    row = entry_of(index, rel)
    if row is None:
        raise ValueError(f"the run's index lists no point {rel!r}")
    return row


# --------------------------------------------------------------------- #
#  The rows by the one done rule, and what the run does next             #
# --------------------------------------------------------------------- #

def rows_of(index: Dict[str, Any], run_dir) -> List[Dict[str, Any]]:
    """EACH POINT THE INDEX LISTS, BY THE ONE DONE RULE (`engines/transport.md`
    § 2a.11; `continuation.done` asks the same pieces): a point taken over
    was done when launch took it over, and is done holding what it must
    hold; a point this run walked is done when it ended with exit 0, its
    verdict did not say NO where it counts, and it holds what it must --
    the last two in the rule's own words (`_run_ending.not_done_why`).
    ``state`` is the point's RUN, apart from done, as the status door says
    it: ``finished`` -- ended with exit 0, converged or not, or taken over
    -- ``failed``, ended otherwise, or ``pending``, not ended.  Each row
    carries the point's words and frame's, its voltage, the point it needs,
    its start and end as the index stamped them, and the run it was taken
    over from -- ``phase`` the rung, the one kind of point the run holds,
    which the monitor's pace groups by."""
    run_dir = Path(run_dir)
    rows = []
    for row in index.get("points") or []:
        taken = row.get("taken_from") is not None
        never = row.get("not_run")
        ended = taken or never is not None or row.get("ended") is not None
        why = None
        if never is not None:
            why = never
        elif ended and not taken and row.get("exit") not in (0, "0"):
            why = f"exit {row.get('exit')}"
        if ended and why is None:
            said = None if taken else row.get("verdict")
            why = not_done_why(
                (said["of"], bool(said["converged"])) if said else None,
                counts=bool(row.get("verdict_counts", True)),
                missing=[w for p, w in row.get("must_hold") or []
                         if not holds(run_dir / row["rel"], p)])
        ok = ended and why is None
        ran = ((taken or row.get("exit") in (0, "0")) if ended
               and never is None else None)
        rows.append({"phase": index.get("stage"), "rel": row["rel"],
                     "words": row.get("words") or row["rel"],
                     "frame": row.get("frame"),
                     "frame_words": row.get("frame_words"),
                     "bias_v": float(row["bias_v"]),
                     "needs": row.get("needs"), "done": ok, "ended": ended,
                     "started": row.get("started"),
                     "ended_at": row.get("ended"),
                     "started_from": row.get("taken_from"),
                     "state": ("pending" if not ended else
                               "finished" if ran else "failed"),
                     "why": why, "not_run": never})
    return rows


def still_to_run(rows) -> List[Dict[str, Any]]:
    """THE POINTS THE WALK MAY STILL RUN -- the walk's own question
    (`engines/transport.md` § 2a.9, § 2a.11): not ended, and no point of
    the chain each needs -- a device point the one before it in its frame,
    and that one's -- ended without its run finishing: *a voltage that does
    not finish ends that frame's chain*, the walk going on with the other
    frames.  Finishing is the walk's word, the run script's exit 0 -- a
    point that finished unconverged hands its state on, though not done."""
    by_rel = {r["rel"]: r for r in rows}

    def runs(row) -> bool:
        need = by_rel.get(row.get("needs") or "")
        while need is not None:
            if need["ended"]:
                return need["state"] == "finished"
            need = by_rel.get(need.get("needs") or "")
        return True
    return [r for r in rows if not r["ended"] and runs(r)]


def summary(rows) -> str:
    """THE COUNT AND ITS AXES, in full every time (`engines/transport.md`
    § 2a.11): *k of N points done, m failed (F frames × V voltages)* -- a
    point that ended without being done counted failed, as a vibration's
    call is (`run-reports.md` § 2.9), where there are any.  One rule for
    the job's rows (:func:`rows_of`) and `status`'s
    (`continuation.point_rows`), each row's ``state`` its run's: ended is
    a state its run ended in (`job.FINAL_STATES`)."""
    rows = list(rows)
    n_done = sum(bool(r["done"]) for r in rows)
    failed = sum(1 for r in rows
                 if not r["done"] and r.get("state") in FINAL_STATES)
    n_frames = len({r.get("frame") for r in rows})
    n_volts = len({r["bias_v"] for r in rows})
    return (f"{n_done} of {len(rows)} points done"
            + (f", {failed} failed" if failed else "")
            + f" ({n_frames} frame{'' if n_frames == 1 else 's'} × "
              f"{n_volts} voltage{'' if n_volts == 1 else 's'})")


def round_of(rows, bias_v: float) -> List[Dict[str, Any]]:
    """THE ROUND at ``bias_v`` -- the voltage's point at every frame, in the
    walk's order (`engines/transport.md` § 2a.11, *What a run's points
    share*).  Launch wrote every voltage from one door, so a round's points
    carry one float."""
    return [r for r in rows if r["bias_v"] == float(bias_v)]


def not_done_words(row: Dict[str, Any]) -> str:
    """A point not done, in the words a failed part and a failed finish name
    it by (`run-reports.md` § 2.9): its words, and why by the one rule."""
    return f"{row['words']}: not done -- {row['why'] or row['state']}"


def run_state(rows) -> "tuple[str, Optional[str]]":
    """THE RUN'S STATE, once no point is left to run, and its words: by the
    status door's rule (`_run_ending.points_state`, `run-reports.md` § 2.9)
    -- finished when every point's run finished, converged or not -- and,
    where not every point is done, the first not done and how many when
    more than one is -- *…: not done -- exit 1; 3 of 12 points not done* --
    ``None`` when every point is done."""
    rows = list(rows)
    state = points_state(r["state"] for r in rows)
    # IN THE ORDER `status` LISTS THE POINTS (`stages.rung_points`): frame
    # by frame, each frame's voltages in the list's order.
    volts = {v: k for k, v in enumerate(dict.fromkeys(r["bias_v"]
                                                       for r in rows))}
    left = sorted((r for r in rows if not r["done"]),
                  key=lambda r: (r["frame"] if r["frame"] is not None
                                 else -1, volts[r["bias_v"]]))
    if not left:
        return state, None
    return state, not_done_words(left[0]) + (
        f"; {len(left)} of {len(rows)} points not done" if len(left) > 1
        else "")


# --------------------------------------------------------------------- #
#  What a round's milestone says of each point (`run-reports.md` § 2.9)  #
# --------------------------------------------------------------------- #

def round_words(index: Dict[str, Any], rows, bias_v: float, run_dir,
                names) -> str:
    """THE ROUND'S OUTCOME, as its milestone says it (`run-reports.md`
    § 2.9): the voltage, then each frame's point -- a device's converged in
    how many NEGF iterations, a transmission's conductance at E_F -- or not
    done and why, or not run and why; for a transmission, the frames'
    average where every frame's point gives one and the frames state their
    weights.  Each point's fact is read from its own files in ``run_dir``
    (:data:`_SAID`); a run of one frame says the voltage and its point
    alone."""
    mine = round_of(rows, bias_v)
    by_rel = {r["rel"]: r for r in rows}
    said = _SAID[index.get("computes") or SCF]
    head = f"{float(bias_v):g} V"
    facts = [(r, point_fact(index, r, run_dir, names) if r["done"]
              else None) for r in mine]
    return said.words(head, [(r, _state_words(r, by_rel), fact)
                             for r, fact in facts], index)


def point_fact(index: Dict[str, Any], row: Dict[str, Any], run_dir, names):
    """WHAT A ROUND SAYS OF ONE DONE POINT, read from its own files in
    ``run_dir`` by what the run's points compute (:data:`_SAID`): a device
    point's NEGF iterations; a transmission point's ``(G/G0, why none)``
    -- the number its round prints to four digits, here whole."""
    return _SAID[index.get("computes") or SCF].fact(row, Path(run_dir), names,
                                                    index)


def _state_words(row, by_rel) -> Optional[str]:
    """A point that is not done, in the round's words -- ``None`` for one
    that is: *not done -- why*, or, a point the walk never ran, *not run --*
    where its frame's chain ended (:func:`_never_run`)."""
    if row["done"]:
        return None
    if row.get("not_run") is not None or not row["ended"]:
        return f"not run -- {row.get('not_run') or _never_run(row, by_rel)}"
    return f"not done -- {row['why'] or row['state']}"


def _never_run(row, by_rel) -> str:
    """Why the walk does not run ``row``, in words: the point up its
    frame's chain that ended without its run finishing -- *a voltage that
    does not finish ends that frame's chain* (`engines/transport.md`
    § 2a.9) -- or, none ended so, the walk never reached it."""
    need = by_rel.get(row.get("needs") or "")
    while need is not None and not need["ended"]:
        need = by_rel.get(need.get("needs") or "")
    if need is not None and need["state"] != "finished":
        return (f"its frame's chain ended at {need['words']}, which did "
                f"not finish")
    return "the walk never reached it"


def close(index: Dict[str, Any], rows) -> Dict[str, Any]:
    """THE WALK OVER -- no point left to run (:func:`still_to_run`): each
    point it never ran stamped so in the index, ``not_run`` its why
    (:func:`_never_run`), so the run's count and state say it as the status
    door does a point the ended walk never reached: failed
    (`engines/transport.md` § 2a.11).  Asked by the step that sends the
    run's `finish`, before it."""
    by_rel = {r["rel"]: r for r in rows}
    for r in rows:
        if not r["ended"]:
            _entry(index, r["rel"])["not_run"] = _never_run(r, by_rel)
    return index


class _ScfSays:
    """A device point's fact: how many NEGF iterations its last SCF ran --
    the reading pass's count, the record's own (`transport.record`,
    ``negf.cycles``)."""

    @staticmethod
    def fact(row, run_dir: Path, names, index) -> Optional[int]:
        out = _newest_output(run_dir / row["rel"], names)
        if out is None:
            return None
        try:                               # inside molbuilder
            from ..parse.engines.siesta_reader import read_output
            from ..parse.engines.siesta_grammar import (PHASE_NEGF,
                                                        rows_of_phase)
        except ImportError:                # beside a job
            from siesta_reader import read_output  # type: ignore[no-redef]
            from siesta_grammar import (  # type: ignore[no-redef]
                PHASE_NEGF, rows_of_phase)
        steps = read_output(out).finish().get("steps") or []
        hist = (steps[-1].get("scf_history") or []) if steps else []
        return len(rows_of_phase(hist, PHASE_NEGF)) or None

    @staticmethod
    def words(head: str, points, index) -> str:
        def one(state, n):
            if state is not None:
                return state
            return (f"converged in {n} NEGF iterations" if n is not None
                    else "converged")
        if len(points) == 1 and points[0][0].get("frame_words") is None:
            r, state, n = points[0]
            return f"{head}: {one(state, n)}"
        return f"{head} -- " + "; ".join(
            f"{r.get('frame_words') or r['words']}: {one(state, n)}"
            for r, state, n in points)


class _ProductSays:
    """A transmission point's fact: its conductance at E_F, G/G0 = T(E_F)
    -- the record's own rule (`conductance.at_fermi`) on its k-averaged
    transmission per spin channel, a polarized point's two channels
    averaged, (T↑ + T↓)/2 (`engines/transport.md` § 2a.12) -- on the
    values the record reads: TBtrans's text files, the point's own and, for
    a point that shares energies with a finished point, its ``shared/``'s,
    joined by the record's one join (`energies.joined`, § 2a.16), each
    printed energy put on the point's grid, the energies its data file
    holds; ``(None, why)`` where it gives none."""

    @staticmethod
    def fact(row, run_dir: Path, names, index):
        try:                               # inside molbuilder
            from ..parse.engines.tbtrans import (PRINTED_HALF_DECIMAL_EV,
                                                 averaged_files,
                                                 read_averaged)
            from .conductance import (SPIN_CHANNELS, at_fermi,
                                      per_spin_channel)
            from .energies import grid, joined
        except ImportError:                # beside a job
            from tbtrans import (  # type: ignore[no-redef]
                PRINTED_HALF_DECIMAL_EV, averaged_files, read_averaged)
            from conductance import (  # type: ignore[no-redef]
                SPIN_CHANNELS, at_fermi, per_spin_channel)
            from energies import grid, joined  # type: ignore[no-redef]
        spin = index.get("spin") or "non-polarized"
        want = SPIN_CHANNELS[spin]
        on = grid(*row["window"]) if row.get("window") else None

        def read(folder: Path) -> Dict[str, Any]:
            # EACH CHANNEL'S T(E) IN ``folder``, its printed energies put on
            # the point's grid -- the data file's energies, which the text
            # files print to within half their last decimal -- so the join
            # matches energies as the record's does.
            found = averaged_files(folder, names.label, pair=index["pair"],
                                   leads=tuple(index["leads"]))
            got = {c: read_averaged(found[c]["transmission"]) for c in want
                   if "transmission" in (found.get(c) or {})}
            if not got:
                return {"energy_ev": []}
            printed = next(iter(got.values()))["energy_ev"]
            if any(g["energy_ev"] != printed for g in got.values()):
                raise ValueError("the two spin channels' energy grids differ")
            keep = list(range(len(printed)))
            energies = list(printed)
            if on is not None:
                at = [_on_grid(on, e, PRINTED_HALF_DECIMAL_EV)
                      for e in printed]
                keep = [k for k, x in enumerate(at) if x is not None]
                energies = [on[at[k]] for k in keep]
            return {"energy_ev": energies,
                    "transmission": {c: [g["columns"][0][k] for k in keep]
                                     for c, g in got.items()}}
        try:
            out = joined(run_dir / row["rel"], read,
                         window_of=lambda folder: None)
        except (OSError, ValueError) as exc:
            return None, str(exc)
        try:
            t = per_spin_channel(out.get("transmission") or {}, spin)
        except ValueError as exc:
            return None, str(exc)
        g = at_fermi(out["energy_ev"], t)
        return ((g, None) if g is not None else
                (None, "the window does not straddle E_F"))

    @staticmethod
    def words(head: str, points, index) -> str:
        def one(state, fact):
            if state is not None:
                return state
            g, why = fact
            return f"{g:.4g}" if g is not None else f"none ({why})"
        if len(points) == 1 and points[0][0].get("frame_words") is None:
            _r, state, fact = points[0]
            return (f"{head} -- G/G0 {one(state, fact)}" if state is None
                    else f"{head}: {state}")
        out = f"{head} -- G/G0: " + ", ".join(
            f"{r.get('frame_words') or r['words']} {one(state, fact)}"
            for r, state, fact in points)
        weights = index.get("weights")
        gs = [fact[0] if state is None else None
              for _r, state, fact in points]
        if (weights is not None and len(weights) == len(points)
                and all(g is not None for g in gs)):
            out += (f"; the frames' average "
                    f"{math.fsum(w * g for w, g in zip(weights, gs)):.4g}")
        return out


def _on_grid(on: Sequence[float], e: float, tol: float) -> Optional[int]:
    """The index of the energy of the grid ``on`` (ascending) that ``e``
    is, within ``tol``, or ``None``."""
    k = bisect.bisect_left(on, e)
    near = [j for j in (k - 1, k) if 0 <= j < len(on)]
    best = min(near, key=lambda j: abs(on[j] - e), default=None)
    return best if best is not None and abs(on[best] - e) <= tol else None


#: WHAT A ROUND SAYS OF EACH POINT, by what the run's points compute
#: (:data:`SCF`, :data:`PRODUCT`; the kind table's ``computes``).
_SAID = {SCF: _ScfSays, PRODUCT: _ProductSays}


def _newest_output(folder: Path, names) -> Optional[Path]:
    """The engine output of the point's newest try (`runrecord.newest_run`),
    or ``None`` where it printed none."""
    try:                                   # inside molbuilder
        from ..runrecord import newest_run
    except ImportError:                    # beside a job
        from runrecord import newest_run  # type: ignore[no-redef]
    n = newest_run(folder, names)
    if n is None:
        return None
    out = folder / names.name(".out", run=n)
    return out if out.is_file() else None


__all__ = ["SCHEMA", "SCF", "PRODUCT", "index_path", "empty_index",
           "add_point", "read_index", "write_index", "entry_of",
           "stamp_start", "stamp_end", "rows_of", "still_to_run", "summary",
           "round_of", "not_done_words", "run_state", "round_words",
           "point_fact", "close"]
