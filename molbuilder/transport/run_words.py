"""A TRANSPORT RUN OF POINTS' WORDS -- its count and what a round's
milestone says of each frame's point (`engines/transport.md` § 2a.11;
`execution/runs-of-points.md` § 2.1, § 4.6, § 4.7; `run-reports.md`
§ 2.9).

MODULE  transport.run_words (floor 1 + stdlib; travels in
        ``mb_monitor.pyz``, `runwrap.MONITOR_COMPANIONS`)
ROLE    the kind table's ``count`` and ``outcome`` parts of a transport
        rung's run of points: :func:`count_words` -- *k of N points done, m
        failed (F frames × V voltages)* -- over the job's rows
        (`runpoints.rows_of`) or `status`'s (`continuation.point_rows`), one
        shape; :func:`round_outcome` -- a device round's NEGF iterations per
        frame, a transmission round's G/G0 at E_F per frame and the frames'
        average -- read from each point's own files.  The index, its
        stamps and its rules are every run of points' (`runpoints`).
USED-BY the kind table (`transport.on.siesta.points`), through
        `runpoints.part` beside the job and `kind_table.part` here
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# TWO WAYS, because this module travels beside a job (`runwrap.
# MONITOR_COMPANIONS`), where the package is not installed.
try:                                       # inside molbuilder
    from .. import runrecord as _rr
    from ..calcdirs import RunPlace
    from ..parse.dirs.job import FINAL_STATES
    from ..runpoints import why_not_run
except ImportError:                        # beside a job
    import runrecord as _rr  # type: ignore[no-redef]
    from job import FINAL_STATES  # type: ignore[no-redef]
    from runpoints import why_not_run  # type: ignore[no-redef]
    from calcdirs import RunPlace  # type: ignore[no-redef]

#: WHAT A RUN'S POINTS COMPUTE, the kind table's words for the rung
#: (``computes``, `transport.stages.STAGE_FACT`), in the run's facts: an
#: SCF -- the device's NEGF loop -- or a product of one, the transmission.
#: What a round's milestone says of each point is read by this (:data:`_SAID`).
SCF, PRODUCT = "scf", "product"


def count_words(rows, facts=None) -> str:
    """THE COUNT AND ITS AXES, in full every time (`engines/transport.md`
    § 2a.11): *k of N points done, m failed (F frames × V voltages)* -- a
    point that ended without being done counted failed, as a vibration's
    call is (`run-reports.md` § 2.9), where there are any.  ``rows`` the
    job's (`runpoints.rows_of`) or `status`'s (`continuation.point_rows`),
    each row's ``state`` its run's: ended is a state its run ended in
    (`job.FINAL_STATES`)."""
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


def round_outcome(index: Dict[str, Any], rows, round_: str, run_dir,
                  names) -> Optional[str]:
    """THE ROUND'S OUTCOME, as its milestone says it (`run-reports.md`
    § 2.9): the voltage, then each frame's point -- a device's converged in
    how many NEGF iterations, a transmission's conductance at E_F -- or not
    done and why, or not run and why; for a transmission, the frames'
    average where every frame's point gives one and the frames state their
    weights.  Each point's fact is read from its own files in ``run_dir``
    (:data:`_SAID`); a run of one frame says the voltage and its point
    alone."""
    rows = list(rows)
    facts = index.get("facts") or {}
    mine = [r for r in rows if r["round"] == round_]
    said = _SAID[facts.get("computes") or SCF]
    out = [(r, _state_words(r, rows),
            point_fact(index, r, run_dir, names) if r["done"] else None)
           for r in mine]
    return said.words(round_, out, facts)


def point_fact(index: Dict[str, Any], row: Dict[str, Any], run_dir, names):
    """WHAT A ROUND SAYS OF ONE DONE POINT, read from its own files in
    ``run_dir`` by what the run's points compute (:data:`_SAID`): a device
    point's NEGF iterations; a transmission point's ``(G/G0, why none)``
    -- the number its round prints to four digits, here whole."""
    facts = index.get("facts") or {}
    return _SAID[facts.get("computes") or SCF].fact(row, Path(run_dir),
                                                    names, facts)


def _state_words(row, rows) -> Optional[str]:
    """A point that is not done, in the round's words -- ``None`` for one
    that is: *not done -- why*, or, one the walk never ran, *not run --*
    why (`runpoints.why_not_run`)."""
    if row["done"]:
        return None
    if row.get("not_run") is not None or not row["ended"]:
        return f"not run -- {row.get('not_run') or why_not_run(rows, row['rel'])}"
    return f"not done -- {row['why'] or row['state']}"


class _ScfSays:
    """A device point's fact: how many NEGF iterations its last SCF ran --
    the reading pass's count, the record's own (`transport.record`,
    ``negf.cycles``)."""

    @staticmethod
    def fact(row, run_dir: Path, names, facts) -> Optional[int]:
        out = _rr.newest_output(RunPlace.of(run_dir).point(row["rel"]),
                                names)
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
    def words(head: str, points, facts) -> str:
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
    def fact(row, run_dir: Path, names, facts):
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
        spin = facts.get("spin") or "non-polarized"
        want = SPIN_CHANNELS[spin]
        on = grid(*row["window"]) if row.get("window") else None

        def read(folder: Path) -> Dict[str, Any]:
            # EACH CHANNEL'S T(E) IN ``folder``, its printed energies put on
            # the point's grid -- the data file's energies, which the text
            # files print to within half their last decimal -- so the join
            # matches energies as the record's does.
            found = averaged_files(folder, names.label, pair=facts["pair"],
                                   leads=tuple(facts["leads"]))
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
            out = joined(RunPlace.of(run_dir).point(row["rel"]), read,
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
    def words(head: str, points, facts) -> str:
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
        weights = facts.get("weights")
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


__all__ = ["SCF", "PRODUCT", "count_words", "round_outcome", "point_fact"]
