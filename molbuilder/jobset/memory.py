"""THE MEMORY A STAGE NEEDS -- measured on the runs of that stage, bounded at
the ranks asked, shown beside the memory ask and never filling it
(`execution/job-request.md` § 3.1a).

A door: it reads, through the run door (`runs.run_of`, `runs.record_of`)
and the one lister of a stage's run folders (`runstatus.run_folders`), what
each run of a stage held -- its run record's one memory peak, as the monitor
measures it (`model/parse.md` § 5c.1), at the ranks its wrapper launched --
and answers, for the ranks an asker resolved, at most how much the stage
needs.  It refuses nothing; a run's state is asked of the status door, which
records an ending it learns of as it does for every reader.  A stage whose
runs' records do not read (`runrecord.LaunchRecordError` -- an index an
older molbuilder wrote among them) has its need said as not read, in the
reader's words: a check that cannot run says so.

**The bound** -- a finished run that held X at N ranks needs at most
X · max(1, M/N) at M ranks: an engine that gives every rank its own copy of
its matrices holds about M/N times as much, one that divides them holds less
a rank as ranks grow, and with fewer ranks neither holds more than it did.
A point's need is the smallest of its finished runs' bounds, a stage's the
largest of its points', a job's the largest of its stages'
(:func:`largest`).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: The record's paths the door reads of a run -- its peak and where the peak
#: was read, the ranks and threads its wrapper launched, the node it ran on
#: -- and nothing more (`runs.record_of`'s ``fields``): no row that reads the
#: engine's output is asked.
FIELDS = ("computation.memory.mem_peak_gb", "computation.memory.mem_peak_from",
          "computation.launch.ranks_asked", "computation.launch.threads",
          "computation.host.machine")

#: Where a peak was read, in words (`utilisation`'s ``mem_peak_from``):
#: under a scheduler the kernel's counter is the job's, since the job began.
_FROM_WORDS = {"kernel counter": "the kernel's counter, the job's",
               "largest sample": "the largest sample"}


@dataclass(frozen=True)
class Held:
    """What one run of a stage held -- a run, or one point of a run of
    points."""
    #: The run, as every record of this calculation spells it
    #: (`runrecord.RunRef`): its folder.
    run: str
    #: The point's key in its run (`stages.Point.rel`), ``None`` for a run
    #: with no point level.
    point: Optional[str]
    gb: float
    read_from: Optional[str]
    #: The ranks its wrapper launched; ``None`` for a run of one process
    #: (PySCF).
    ranks: Optional[int]
    threads: Optional[int]
    node: Optional[str]
    #: The status door's word for how it stands, and its detail.
    state: str
    detail: str = ""

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    def bound(self, ranks: Optional[int]) -> float:
        """At most what this run's stage needs at ``ranks`` -- X · max(1,
        M/N); what it held, for a run of one process or no ranks asked."""
        if self.ranks is None or ranks is None:
            return self.gb
        return self.gb * max(1.0, ranks / self.ranks)

    def words(self) -> str:
        """``held 12.31G at 16 ranks × 1 thread, 05_transmission/run-0 on
        qlabsrv (the largest sample; finished)``."""
        at = (f" at {self.ranks} rank{'s' if self.ranks != 1 else ''}"
              if self.ranks else "")
        if self.ranks and self.threads:
            at += f" × {self.threads} thread{'s' if self.threads != 1 else ''}"
        where = f" on {self.node}" if self.node else ""
        how = "; ".join(w for w in (_FROM_WORDS.get(self.read_from or "",
                                                    self.read_from),
                                    self.how_it_stands()) if w)
        return f"held {self.gb:.2f}G{at}, {self.run}{where} ({how})"

    def how_it_stands(self) -> str:
        if self.finished:
            return "finished"
        return f"{self.state}: {self.detail}" if self.detail else self.state

    def so_far(self) -> str:
        """A run that gives no bound, said: one still going holds what it
        holds so far; one that ended otherwise held it before it ended."""
        from ..parse.dirs.job import LIVE_STATES
        if self.state in LIVE_STATES:
            return (f"{self.run} holds {self.gb:.2f}G so far "
                    f"({self.how_it_stands()})")
        if self.state == "unreadable":
            return f"{self.run} held {self.gb:.2f}G ({self.how_it_stands()})"
        return (f"{self.run} held {self.gb:.2f}G before it ended "
                f"({self.how_it_stands()})")


@dataclass(frozen=True)
class Need:
    """The memory a stage needs at ``ranks`` -- or a job's, its stages'
    largest (:func:`largest`)."""
    #: The stage -- for a job, the stage whose need is the job's, or every
    #: stage of it when none is measured.
    stage: str
    #: The ranks it is said at -- the asker's, ``None`` when none is stated.
    ranks: Optional[int]
    #: At most this many GB, the scheduler's units; ``None`` when no finished
    #: run stated what it held.
    gb: Optional[float]
    #: The run whose bound it is.
    held: Optional[Held] = None
    #: The run that gives no bound and held the most, and how many give none.
    unfinished: Optional[Held] = None
    n_unfinished: int = 0
    #: A job's stages nothing has measured.
    unmeasured: Tuple[str, ...] = ()
    #: A job's need (:func:`largest`): its words name the stage it is.
    of_job: bool = False
    #: Why the stage's runs could not be read, in their reader's words --
    #: ``""`` when they were.
    unread: str = ""

    @property
    def line(self) -> str:
        """The door's words for it -- one sentence, which prep and Task setup
        print alike (`execution/job-request.md` § 3.1a)."""
        if self.unread:
            said = f"not read -- {self.unread}"
        elif self.held is None:
            said = (f"nothing measured -- no finished run of {self.stage} "
                    f"has stated its memory")
        elif self.ranks is None and self.held.ranks is not None:
            said = self.held.words()
        else:
            at = (f" at {self.ranks} rank{'s' if self.ranks != 1 else ''}"
                  if self.held.ranks and self.ranks else "")
            # AT MOST from a peak the kernel counted, which is the job's and
            # errs high; ABOUT from the largest sample, which may be under a
            # run's true peak (`execution/job-request.md` § 3.1a).
            bound = ("at most" if self.held.read_from == "kernel counter"
                     else "about")
            said = f"{bound} {_up(self.gb)}G{at} -- {self.held.words()}"
        if self.of_job and (self.held is not None or self.unread):
            said = f"{self.stage}: {said}"
        if self.unfinished is not None:
            more = (f" (and {self.n_unfinished - 1} more giving no bound)"
                    if self.n_unfinished > 1 else "")
            said += f"; {self.unfinished.so_far()}{more}"
        if self.unmeasured:
            said += f"; nothing measured of {', '.join(self.unmeasured)}"
        return said

    def to_wire(self) -> dict:
        return {"stage": self.stage, "line": self.line,
                "gb": _up(self.gb) if self.gb is not None else None,
                "ranks": self.ranks,
                "run": self.held.run if self.held is not None else None}


def _up(gb: Optional[float]) -> Optional[float]:
    """A figure said in tenths of a GB, rounded up -- the side that asks for
    no less than a run needs."""
    return None if gb is None else math.ceil(round(gb * 10, 6)) / 10


def _ranks(asked) -> Optional[int]:
    """The ranks an asker resolved, or ``None`` -- a value the setting door
    refuses (a card still holding what was typed: ``"0"``, ``"8a"``) states no
    ranks, and the measurement alone is said."""
    if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
        return None
    return asked


def held_by(calc, stage: str) -> List[Held]:
    """What every run of ``stage`` held -- each run, each point of a run of
    points -- read through the run door, those whose record states a memory
    peak."""
    from ..runrecord import RunRef
    from ..runs import record_of, run_of, state_of
    from .naming import stage_home
    from .runstatus import run_folders
    base, task = Path(calc.base), calc.task
    home = stage_home(base, task, stage)
    if home.dir is None or not home.dir.is_dir():
        return []
    out: List[Held] = []
    for _n, folder, pt in run_folders(base, task, home.dir, home.name):
        run = run_of(folder)
        if run is None:
            continue
        st = state_of(run)
        rec = record_of(run, status=st, fields=FIELDS) or {}
        comp = rec.get("computation") or {}
        mem = comp.get("memory") or {}
        if mem.get("mem_peak_gb") is None:
            continue
        launch = comp.get("launch") or {}
        ref = RunRef.at(base, folder)
        out.append(Held(
            run=str(ref), point=(str(pt.rel) if pt is not None else None),
            gb=float(mem["mem_peak_gb"]), read_from=mem.get("mem_peak_from"),
            ranks=(int(launch["ranks_asked"]) if launch.get("ranks_asked")
                   else None),
            threads=(int(launch["threads"]) if launch.get("threads")
                     else None),
            node=((comp.get("host") or {}).get("machine") or {}).get("node"),
            state=st.state, detail=st.detail or ""))
    return out


def need(calc, stage: str, ranks) -> Need:
    """The memory ``stage`` needs at ``ranks`` -- the ranks the asker
    resolved: prep's, the job's own; Task setup's, the card as edited -- or,
    with none stated, the measurement alone.  Nothing measured for a stage
    ``calc`` does not hold, or for no calculation (``None``)."""
    from ..identity import stage_key
    from ..runrecord import LaunchRecordError
    ranks = _ranks(ranks)
    if calc is None or not any(stage_key(s.name) == stage_key(stage)
                               for s in calc.task.stages):
        return Need(stage, ranks, None)
    try:
        helds = held_by(calc, stage)
    except LaunchRecordError as exc:
        # A RUN'S RECORDS THAT DO NOT READ: what its stage needs is not
        # known, and the reader's words say why.
        return Need(stage, ranks, None, unread=str(exc))
    best: dict = {}
    for h in helds:
        if not h.finished:
            continue
        # A POINT'S NEED: the smallest of its finished runs' bounds -- every
        # one of them holds.
        b = h.bound(ranks)
        if h.point not in best or b < best[h.point][0]:
            best[h.point] = (b, h)
    unfinished = [h for h in helds if not h.finished]
    worst = max(unfinished, key=lambda h: h.gb) if unfinished else None
    if not best:
        return Need(stage, ranks, None, None, worst, len(unfinished))
    # A STAGE'S NEED: the largest of its points'.
    gb, held = max(best.values(), key=lambda bh: bh[0])
    return Need(stage, ranks, gb, held, worst, len(unfinished))


def largest(needs: Sequence[Need]) -> Need:
    """A JOB'S NEED -- the largest of its stages' (a group's members, a
    sequence's groups), naming the stages nothing has measured; when none is
    measured, that, of every stage.  A stage whose runs do not read leaves
    the largest unknown: its need, not read, is the job's."""
    unread = [n for n in needs if n.unread]
    if unread:
        return dataclasses.replace(unread[0], of_job=True)
    measured = [n for n in needs if n.gb is not None]
    unmeasured = tuple(n.stage for n in needs if n.gb is None)
    if not measured:
        return Need(", ".join(n.stage for n in needs),
                    needs[0].ranks if needs else None, None, of_job=True)
    top = max(measured, key=lambda n: n.gb)
    return dataclasses.replace(top, unmeasured=unmeasured, of_job=True)


__all__ = ["FIELDS", "Held", "Need", "held_by", "need", "largest"]
