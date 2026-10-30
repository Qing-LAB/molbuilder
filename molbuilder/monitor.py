"""Background job monitor + notifier hooks.

The front end of the job-monitor/watcher + notifier surface
(`execution/running-a-job.md` § 4.1, `execution/job-contracts.md` — the
monitor's section).  A lightweight, periodically-waking process that
**reads** a running job's artifacts -- every engine's -- through the
framework's own readers, which travel beside it
(`execution/run-reports.md` § 2.3), appends a status line to its
``.monitor.log``, samples utilisation into its ``.util.csv``, and
notifies -- rarely, and only when the calculation's own policy says to.

**It never runs inside molbuilder.**  A verbatim copy of this file ships
beside the job inside ``mb_monitor.pyz`` -- one file, holding it as
``mb_monitor.py`` with the framework modules it reads through
(`runwrap.MONITOR_BUNDLE`) -- and runs with the JOB's own python from
the working directory: molbuilder is deliberately never pip-installed into
any env, so ``import molbuilder`` fails on the job's python.  It is
stdlib-only because it is the instrument that reports a failure: it must
load when the env's own packages do not, so a broken numpy is reported by
it, never hidden by it.

**WHICH python it gets is the ENV's**: the activated env's own,
``$CONDA_PREFIX/bin/python3`` -- never the first ``python3`` on PATH
(`running-a-job.md` § 2).  Every recipe pins ``_PYTHON_SPEC``, so the
version parsing this file is one the registry names.

The run-wrapper backgrounds it at low OS priority (``nice -n 19``) so it
never competes with the compute ranks on the same node: it sleeps almost
all of the time and does a few ms of tail-reads per wake (default 10 s),
far below any benchmark's measurement noise.

**Looking and telling are separate.**  It looks often, because
``util.csv`` is the diagnostic record.  It tells rarely, because a message
per wake is a message every ten seconds for the length of a run -- which
is what a notifier registered on every wake would receive.  When to
tell is the calculation's, carried from `task.json`'s ``notify`` block:
``--notify-on-scf``, ``--notify-every-hours``, and a run ending, always.
So is WHICH CHANNELS, by name -- ``--notify-channels``, ``*`` for every
channel this machine has, absent or ``""`` for none (`run-reports.md` 3.0).

CLI (also available as ``molbuilder monitor``)::

    nice -n 19 "$_mb_py" mb_monitor.pyz \\
        --label job --stage 01_coarse --run 0 --util --cores 8 \\
        --notify-every-hours 6 --watch-pid $$ &

**Where** a name POINTS is never here and never in the description: the
addresses and their credentials are the user's own file,
:func:`default_notify_path`, mode 0600 on the machine that runs the job.
A description carries names; this reads what they mean.  ``MB_NOTIFY_URL`` (with ``MB_NOTIFY_KEY`` for our
own listener) overrides it for a one-off.  A notifier can
also be registered programmatically::

    from molbuilder import monitor
    monitor.register_notifier(lambda st, ev: my_push(st.as_text()))

Design notes:
- **stdlib-only hot path** -- no heavy imports in the loop.
- Each tick feeds each live output's parser only what the file GAINED since
  the last wake -- one reader per file, kept between wakes -- so a large
  output costs its growth, never its length; the timing log is read by the
  timing instrument's own reader.
- It stops when ``--watch-pid`` goes away (the job wrapper's PID) or when
  the wrapper stops it: SIGTERM at the job's end -- the scheduler's own
  SIGTERM at a walltime or a cancel reads the same -- and SIGUSR1 when the
  run is retried in place, as its next try, which is not an ending.  Output markers are
  not consulted: they can appear before a run is actually over, which would
  end the sampling early (`job-contracts.md`, the monitor's section).
- The seconds per iteration are the timing instrument's, one phase at a
  time (`scf_timing_rows.timing_of`, `model/parse.md` § 5c): the SIESTA
  family's tee, a PySCF run's stamped progress log -- SIESTA's own per-scf
  time is not trusted.
- **It judges no stall** (`run-reports.md` § 2): a step can take hours, and
  nothing in the output tells a slow one from a stuck one.  The loop wakes
  often (default 10 s) but logs a ``[STATUS]`` line only when the SCF
  iteration count / geometry move / energy / state changed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Sequence,
                    Tuple)


# --------------------------------------------------------------------- #
#  What it reads, and through what                                      #
# --------------------------------------------------------------------- #
#
# THE MONITOR HAS NO READER OF ITS OWN (`execution/run-reports.md` § 2.3).
# Every fact below is read by the reader the rest of molbuilder reads that
# file with -- `run_status`, the Results tab's own status door; the ONE
# SIESTA parser's and the ONE molwatch parser's reading passes, fed as the
# output grows; the timing instrument; `runfiles` for every name; and what a
# report may carry, from the one declaration of it -- and those
# modules travel beside the job (`runwrap.MONITOR_COMPANIONS`): imported from
# the package here, from the copies beside the job there, as `config_dir`
# always has been.
try:                                        # inside molbuilder
    from .parse.engines import _run_ending as _ending
    from . import report_cards as _cards
    from . import report_fields as _fields
    from .runtime_info import JOB_STARTED_VAR
    from . import runfiles as _rf
    from . import runrecord as _rr
    from .parse.dirs.job import LIVE_STATES, MONITOR_ENDED, run_status
    from .parse.engines.molwatch_reader import MolwatchReader
    from .parse.engines.siesta_reader import SiestaReader
    from .parse.engines.tbtrans import loop_words
    from .parse.instruments.scf_timing_rows import timing_of
    from .siesta import vibration_phases as _vp
    from .trajectory_log import keys as _K
    from .transport import points_index as _pi
    from .wrapper_log import LOG_CLOCK, LOG_LINE
except ImportError:                         # beside the job
    import _run_ending as _ending
    import report_cards as _cards
    import report_fields as _fields
    from runtime_info import JOB_STARTED_VAR
    import runfiles as _rf
    import runrecord as _rr
    from job import LIVE_STATES, MONITOR_ENDED, run_status
    from molwatch_reader import MolwatchReader
    from siesta_reader import SiestaReader
    from tbtrans import loop_words
    from scf_timing_rows import timing_of
    import vibration_phases as _vp
    import keys as _K
    import points_index as _pi
    from wrapper_log import LOG_CLOCK, LOG_LINE


@dataclass
class JobStatus:
    """One snapshot of the watched run, in the framework's words.

    ``state`` is ``running`` while the watched PID lives, and at the end
    `run_status`'s verdict with its ``detail`` (`run-reports.md` § 2.3).  The
    rest is the live reading: each field present only when the run's files
    state it, ``None`` otherwise -- a reader must be able to tell *not
    stated* from zero.  Each residual, the step and the finished steps are
    the output's own parser's (`LIVE_READERS`); the monitor displays them.
    """
    state: str = "running"
    elapsed_s: float = 0.0
    detail: Optional[str] = None
    #: The SCF phase the latest row belongs to (SIESTA family): ``periodic``,
    #: or TranSIESTA's ``negf``.
    phase: Optional[str] = None
    #: SCF iterations run so far in that phase -- a SIESTA run's timing rows,
    #: a PySCF run's SCF history across its finished steps.
    n_iters: Optional[int] = None
    #: The latest row's own iteration number, as the output printed it.
    cycle: Optional[int] = None
    #: The timing instrument's seconds per iteration, that phase's own.
    per_iter_s: Optional[float] = None
    energy: Optional[float] = None
    geom_step: Optional[int] = None
    #: What a step of this run IS, in the engine's own words (``Broyden opt.
    #: move``, ``FC step``); ``None`` when the output names none.
    step_kind: Optional[str] = None
    #: How many steps have finished -- each one an SCF that reached its
    #: criterion (`run-reports.md` § 2.2).  SIESTA begins step N once N are
    #: done; a PySCF block is written when its step ends.
    steps_done: Optional[int] = None
    #: Whether a step of this run moves the geometry, as the reading pass
    #: states it from its grammar (`run-reports.md` § 2.2): a relaxation's
    #: move does, a force-constant displacement does not.
    step_moves: Optional[bool] = None
    #: ``{name: (value, tolerance or None, unit)}`` -- dDmax, dHmax, and a
    #: NEGF loop's dQ, each beside the criterion the run states for it.
    residuals: Dict[str, Tuple[float, Optional[float], str]] = field(
        default_factory=dict)
    #: The largest force component on the free atoms -- those the run does
    #: not hold -- and its
    #: tolerance; and every atom's, only when it is the larger -- a held atom
    #: carries it (`run-reports.md` § 2.7).
    max_force: Optional[float] = None
    max_force_tol: Optional[float] = None
    max_force_held: Optional[float] = None
    #: A TranSIESTA run that computes no forces: the sentence said in their
    #: place, the reading pass's (`model/parse.md` § 4a).
    forces_not_computed: Optional[str] = None
    #: TBtrans's account of its loop, as the output's reading pass states it
    #: (`model/parse.md` § 5d.5) -- not a report field: the summary line
    #: says it in words (`tbtrans.loop_words`).
    tbtrans: Optional[Dict[str, Any]] = None
    #: WHERE THE RUN IS, in the reading pass's words (`model/parse.md`
    #: § 2b, ``doing``: the row in flight's phase and iteration, or a
    #: TranSIESTA device's set-up until its first SCF row) -- the summary
    #: line says it as given; ``None`` when the output states neither.
    doing: Optional[str] = None
    #: The pass underway's share done, in per cent, and the time TBtrans
    #: says it has left, as its latest progress line states them
    #: (`run-reports.md` § 4.1a); ``None`` when no pass is underway.
    done_pct: Optional[float] = None
    eta_s: Optional[float] = None
    #: At the end: each SCF phase's convergence, and the process's goodbye.
    converged: Dict[str, Optional[bool]] = field(default_factory=dict)
    #: At the end of a relaxation: whether its geometry converged, as the
    #: output states it (`_run_ending.RunEnding.relaxed`); ``None`` for a run
    #: that relaxes nothing.
    relaxed: Optional[bool] = None
    exit: Optional[str] = None
    #: A job sent with a wall: the wall, and the time it has left, counted
    #: from the job's start (`run-reports.md` § 2.1b); ``None`` for a run
    #: whose record states no wall -- a run here.
    wall_s: Optional[float] = None
    wall_left_s: Optional[float] = None
    #: THE PART OF THE RUN UNDERWAY (`run-reports.md` § 2.8), the reading
    #: pass's: a row of the output's parts table by name -- ``doing`` its
    #: sentence -- ``None`` while the SCF speaks and once the run has ended;
    #: when this monitor first read it begin; and the last SCF's verdict,
    #: which an ended SCF's facts are said with.
    part: Optional[str] = None
    part_since: Optional[float] = None
    scf_converged: Optional[bool] = None
    #: A MEMBER'S RUN (§ 2.8): the run's count in its own words (its
    #: index's reader, :data:`RUN_INDEXES`), this point's -- ``call_unit``
    #: what one is called in the kind's words, a vibration's *call*, a
    #: transport sweep's *point* -- and this point's clock, ``elapsed_s``
    #: being the run's.
    run_words: Optional[str] = None
    call_words: Optional[str] = None
    call_unit: Optional[str] = None
    call_elapsed_s: Optional[float] = None
    #: THE PACE (§ 2.8): what a unit of the run's work is, in words -- an
    #: engine's step, a run of points' call -- how many are finished and,
    #: where the run states how many there are, left; the mean a finished
    #: one took; and the time the left need at it.
    unit_words: Optional[str] = None
    units_done: Optional[int] = None
    units_left: Optional[int] = None
    unit_s: Optional[float] = None
    left_at_pace_s: Optional[float] = None

    def as_text(self) -> str:
        """The summary line every channel shows, in the order of
        `run-reports.md` § 2.8: the state; a member's run and call; the part
        underway -- since when, for one that prints nothing until it ends;
        the SCF's facts, as they stand or, once another part is underway,
        as ended; the energy, the force and the step; TBtrans's loop; the
        pace; at the end how each phase converged and the exit; the clocks;
        the wall."""
        span = _fields.span_words
        head = self.state + (f" ({self.detail})" if self.detail else "")
        bits = [head]
        if self.run_words:
            bits.append(f"the run: {self.run_words}")
        if self.call_words:
            bits.append(f"this {self.call_unit or 'call'}: "
                        f"{self.call_words}")
        # WHERE THE RUN IS, in the reading pass's one sentence
        # (`model/parse.md` § 2b, ``doing``) -- a part of the output's parts
        # table with when this monitor saw it begin; not repeated when the
        # head's detail, the run door's, already says it.
        if self.doing and not (self.detail and self.doing in self.detail):
            bits.append(self.doing + (
                "" if self.part is None or self.part_since is None else
                " (since " + time.strftime(
                    "%H:%M", time.localtime(self.part_since)) + ")"))
        residuals = [f"{name} {value:.3g}{(' ' + unit) if unit else ''}"
                     + (f" (tol {tol:g})" if tol is not None else "")
                     for name, (value, tol, unit) in self.residuals.items()]
        # A PART'S FACTS ENDED once another part is underway (§ 2.8), each
        # field by its own declaration; the SCF's sentence -- its
        # iterations, their time, its residuals -- by its iterations'.
        ended = {f.name: f.ended(self.part) for f in _fields.FIELDS}
        scf_ended = ended["n_iters"]
        if scf_ended and (self.n_iters or residuals):
            # AN SCF THAT HAS ENDED, said as ended (§ 2.8) -- its facts
            # together, never its last row as the row in flight.
            said = {True: "converged", False: "did not converge"}.get(
                self.scf_converged, "ended")
            took = (f" ({span(self.n_iters * self.per_iter_s)})"
                    if self.n_iters and self.per_iter_s is not None else "")
            bits.append(", ".join(
                [f"its SCF {said}"
                 + (f" in {self.n_iters} iterations{took}"
                    if self.n_iters else "")] + residuals))
        elif self.n_iters:
            bits.append(f"{self.n_iters} SCF rows")
        if self.energy is not None:
            bits.append(f"E {self.energy:.6f} eV")
        if not scf_ended:
            bits += residuals
        if self.forces_not_computed:
            bits.append(self.forces_not_computed)
        elif self.max_force is not None:
            bits.append(f"max force {self.max_force:.4g} eV/Ang"
                        + (f" (tol {self.max_force_tol:g})"
                           if self.max_force_tol is not None else "")
                        + (f"; {self.max_force_held:.4g} eV/Ang on the held "
                           f"atoms" if self.max_force_held is not None
                           else ""))
        if self.geom_step is not None:
            bits.append(f"{self.step_kind or 'step'} {self.geom_step}")
        if self.per_iter_s is not None and not ended["per_iter_s"]:
            bits.append(f"{self.per_iter_s:.2f} s/iter")
        loop = loop_words(self.tbtrans, over=self.state not in LIVE_STATES)
        if loop:
            bits.append(loop)
        if self.unit_s is not None and self.units_done:
            bits.append(f"each {self.unit_words or 'step'} takes "
                        f"{span(self.unit_s)} ({self.units_done} done)"
                        + (f" -- the {self.units_left} left need about "
                           f"{span(self.left_at_pace_s)} at that pace"
                           if self.units_left and self.left_at_pace_s
                           is not None else ""))
        if self.converged:
            bits.append("converged: " + ", ".join(
                f"{ph} {'yes' if ok else 'no' if ok is False else '?'}"
                for ph, ok in self.converged.items()))
        if self.relaxed is not None:
            bits.append("geometry relaxed" if self.relaxed
                        else "geometry did not converge within its moves")
        if self.exit:
            bits.append(f"exit {self.exit.split(' at ')[0]}")
        bits.append(f"the run {span(self.elapsed_s)}"
                    + (f" · this {self.call_unit or 'call'} "
                       f"{span(self.call_elapsed_s)}"
                       if self.call_elapsed_s is not None else ""))
        if self.wall_s is not None and self.wall_left_s is not None:
            bits.append(f"wall {span(self.wall_s)}: "
                        f"{span(max(0.0, self.wall_left_s))} left")
        return " | ".join(bits)


# NO COMPLETION MARKERS DECIDE ANYTHING HERE.  `job-contracts.md` states the
# rule: the monitor "follows the launcher's PID -- so it knows authoritatively
# when the run ended, rather than guessing from output markers":
# `siesta: Final energy` prints BEFORE a run is over, so a loop that read it
# could return while the job was still holding CPUs and GPUs.  HOW the
# run ended is reported the way the Results tab reads it -- `run_status`,
# asked once the PID has said it is over (`run-reports.md` § 2.2).

#: ROLE -> the reading pass of that file's ONE parser -- chosen by what the
#: file IS, never by the engine, as `_run_ending.READERS` chooses how it
#: ENDED.  A SIESTA-family `.out` is its own live channel; a PySCF run's is
#: its progress log (its stdout is block-buffered and says nothing live).
#: Each is built for the rung's stage: a staged progress log keys its targets
#: by it.
LIVE_READERS: Dict[str, Callable[[Optional[str]], Any]] = {
    ".out":          lambda stage: SiestaReader(),
    ".molwatch.log": lambda stage: MolwatchReader(stage=stage),
}

#: What a live reading must state to say where the run IS -- an SCF row, a
#: step, TBtrans's loop (TBtrans states no SCF), or the words of where it
#: is (`model/parse.md` § 2b, ``doing``: a device's set-up until its first
#: SCF row).
_POSITION = ("cycle", "step", "energy", "tbtrans", "doing")


#: How much of a file's beginning, and of what was last read, identifies it
#: between wakes.
_HEAD_BYTES = 256
_TAIL_BYTES = 64


class _Growth:
    """One output as it grows, fed to its reader a line at a time -- the
    monitor never reads a growing output whole.  A last line still being
    written waits for its newline.

    **A file rewritten in place is read afresh**, and it is told by its
    CONTENT, not its size: a PySCF deck truncates the progress log prep
    seeded and writes its own header, and by the next wake the new file is
    longer than the old -- read on from the old offset, the header and its
    convergence targets were skipped (measured on a real water run,
    2026-09-26: its force had no tolerance beside it).  So each wake checks
    that the file still begins as it did and still holds, just before the
    offset, what was last read there."""

    def __init__(self, reader):
        self.reader = reader
        self.offset = 0
        self.partial = ""
        self.head = b""
        self.tail = b""

    def _same_file(self, fh) -> bool:
        if fh.read(len(self.head)) != self.head:
            return False
        fh.seek(self.offset - len(self.tail))
        return fh.read(len(self.tail)) == self.tail

    def feed_from(self, path: Path, make) -> None:
        try:
            with path.open("rb") as fh:
                if self.offset and not self._same_file(fh):
                    self.reader, self.offset, self.partial = make(), 0, ""
                    self.head = self.tail = b""
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return
        if self.offset < _HEAD_BYTES:
            self.head = (self.head + data)[:_HEAD_BYTES]
        self.offset += len(data)
        self.tail = (self.tail + data)[-_TAIL_BYTES:]
        text = self.partial + data.decode("utf-8", "replace")
        lines = text.split("\n")
        self.partial = lines.pop()
        for line in lines:
            self.reader.feed(line.rstrip("\r"))


@dataclass
class WatchedRun:
    """The run this monitor watches, as the wrapper names it -- its stage's
    names (`runfiles.RunNames`) and its try's number (``--run N``: 0 from
    launch, the next from a warm retry) -- in the directory the job runs
    in.

    Every file is named and found through `runfiles` (`run-reports.md`
    § 2.3), the way `project-layout.md` § 4.5 has every caller ask for our
    names; the wrapper passes no paths.
    """
    names: "_rf.RunNames"
    run: int
    directory: Path = field(default_factory=lambda: Path("."))
    #: THE RUN'S FOLDER, for a member of a run of points (`run-reports.md`
    #: § 2.1b; `engines/vibration.md` § 5.11 § 7a): where the run's one
    #: launch record and its notification record lie -- ``None`` for a run
    #: in its own folder, which holds them itself.
    launched_in: Optional[Path] = None
    #: Each live output, and the reader fed from it.
    _growth: Dict[Path, "_Growth"] = field(default_factory=dict)
    #: The part of the run last read underway, and when this monitor first
    #: read it (`run-reports.md` § 2.8).
    _part_seen: Optional[Tuple[Optional[str], float]] = None
    #: A member's run, read once per version of its index: ``(the index's
    #: mtime, the index, its rows by the one done rule)``.
    _run_cache: Optional[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]] = None

    @property
    def label(self) -> str:
        return self.names.label

    @property
    def stage(self) -> str:
        return self.names.stage

    @property
    def stem(self) -> str:
        """``<label>_<stage>`` -- the rung's files all begin with it."""
        return self.names.stem

    def path(self, role: str) -> Path:
        """This run's file in ``role``, by its stage's names."""
        return self.directory / self.names.name(role, self.run)

    def files(self) -> Dict[str, Path]:
        """This run's files by role: its own try's, and those the folder
        holds for every try -- a progress log carries no try's number
        (`runfiles.RunNames`)."""
        out: Dict[str, Path] = {}
        for p, rf in _rf.find(self.directory, self.label, stage=self.stage):
            if rf.run is None or rf.run == self.run:
                out[rf.role] = p
        return out

    def read(self, start_epoch: float, now_epoch: float) -> JobStatus:
        """Where the run is now, as its files' own parsers read them -- each
        fed what the file gained since the last wake.  Never raises on a
        missing or locked file, and never judges whether the run is over:
        that is the watched PID's answer alone."""
        st = JobStatus(elapsed_s=max(0.0, now_epoch - start_epoch))
        files = self.files()
        # EVERY CHANNEL THAT STATES WHERE THE RUN IS, and the freshest of
        # them speaks.  A channel that states nothing is not a candidate: a
        # SIESTA run's progress log is the prep's seed, whose mtime says
        # nothing about the run (a copied tree reorders it).
        heard = []
        for role, path in files.items():
            make = LIVE_READERS.get(role)
            if make is None:
                continue
            grow = self._growth.get(path)
            if grow is None:
                grow = self._growth[path] = _Growth(make(self.stage))
            grow.feed_from(path, lambda: make(self.stage))
            state = grow.reader.now()
            if any(k in state for k in _POSITION):
                heard.append((_mtime(path), state))
        if heard:
            self._apply(st, max(heard, key=lambda h: h[0])[1])
        # THE RUN'S STAMPED SCF ROWS, by the one rule (`model/parse.md`
        # § 5c): the SIESTA family's tee, else a PySCF run's progress log --
        # whose rows the deck stamps itself.
        timing = files.get(".scf-timing.log") or files.get(".molwatch.log")
        if timing is not None:
            m = timing_of(timing)
            st.n_iters = m.get("rows") or st.n_iters
            st.per_iter_s = m.get("s_per_iter")
        # WHEN THE PART UNDERWAY BEGAN, as this monitor first read it -- within
        # one wake of its opening line (`run-reports.md` § 2.8).
        if self._part_seen is None or self._part_seen[0] != st.part:
            self._part_seen = (st.part, now_epoch)
        if st.part is not None:
            st.part_since = self._part_seen[1]
        if self.launched_in is not None:
            self._read_run(st, now_epoch)
        return st

    def _read_run(self, st: JobStatus, now_epoch: float) -> None:
        """A MEMBER'S RUN (`run-reports.md` § 2.8; `engines/vibration.md`
        § 5.11 § 7a; `engines/transport.md` § 2a.11): its count and this
        call, in the run's own words, from its index through its kind's one
        reader and one done rule (:data:`RUN_INDEXES`); the run's
        clock from its first call this run computed, this call's from this
        monitor's start; and the pace of the phase underway's calls, once
        one has finished.  An index that does not read -- being rewritten,
        or none -- leaves the line without them.  The index is the run's,
        in its folder and named for its first try, whichever try this call
        is in -- a call retried in place reads the run it belongs to."""
        found = _run_index_of(self.launched_in, self.names)
        if found is None:
            return
        reader, idx = found
        try:
            version = idx.stat().st_mtime_ns
        except OSError:
            return
        if self._run_cache is None or self._run_cache[0] != version:
            try:
                index = reader.read(idx)
                rows = reader.rows(index, self.launched_in, self.label)
            except (OSError, ValueError, KeyError, TypeError):
                return
            self._run_cache = (version, index, rows)
        _version, index, rows = self._run_cache
        try:
            rel = Path(os.path.relpath(self.directory,
                                       self.launched_in)).as_posix()
        except ValueError:
            rel = None
        me = next((r for r in rows if r["rel"] == rel), None)
        # THIS CALL IS RUNNING -- its monitor is its witness; the index
        # says it pending until its record step writes its end.
        st.run_words = reader.summary(
            [dict(r, state="running") if r is me and not r["ended"] else r
             for r in rows], index)
        if me is None:
            return
        st.call_words = me["words"]
        st.call_unit = reader.unit
        st.call_elapsed_s = st.elapsed_s
        clock = _run_clock(rows, now_epoch)
        if clock is not None:
            st.elapsed_s = clock
        same = [r for r in rows if r["phase"] == me["phase"]]
        if len(same) < 2:
            return
        # THE PACE, over the calls this run computed -- as its clock is --
        # never one taken over from an earlier run's allocation.
        took = []
        for r in same:
            if r["done"] and not r.get("started_from"):
                a = _rr.stamp_time(r.get("started"))
                b = _rr.stamp_time(r.get("ended_at"))
                if a is not None and b is not None:
                    took.append((b - a).total_seconds())
        st.unit_words = reader.unit_words(me)
        st.units_done = sum(1 for r in same
                            if r["done"] and not r.get("started_from"))
        # LEFT: what the walk will still run of this phase, by the kind's
        # own rule.
        st.units_left = sum(1 for r in reader.left(index, rows)
                            if r["phase"] == me["phase"])
        if took:
            st.unit_s = sum(took) / len(took)
            st.left_at_pace_s = st.unit_s * st.units_left

    @staticmethod
    def _apply(st: JobStatus, state: Dict[str, Any]) -> None:
        """A parser's live reading onto the report -- as the parser states
        it: the residuals come with the criteria the run states for them."""
        st.phase = state.get("phase")
        st.cycle = state.get("cycle")
        st.energy = state.get("energy")
        if "step" in state:
            st.geom_step = state["step"]
            st.step_kind = state.get("step_kind")
        st.steps_done = state.get("steps_done")
        st.step_moves = state.get("step_moves")
        if "scf_rows" in state:
            st.n_iters = state["scf_rows"]
        st.residuals.update(state.get("residuals") or {})
        st.forces_not_computed = state.get("forces_not_computed")
        if state.get("max_force") is not None:
            st.max_force = state["max_force"]
            st.max_force_tol = (state.get("targets") or {}).get(
                _K.MAX_FORCE_TOL)
            every = state.get("max_force_all")
            # THE HELD ATOMS' FORCE, said only when it is the larger.
            st.max_force_held = (every if every is not None
                                 and every > st.max_force else None)
        st.tbtrans = state.get("tbtrans")
        st.doing = state.get("doing")
        st.part = state.get("part")
        st.scf_converged = state.get("scf_converged")
        st.done_pct = state.get("done_pct")
        st.eta_s = state.get("eta_s")

    @property
    def run_folder(self) -> Path:
        """Where the run's records lie: the run's folder for a member of a
        run of points, else this one."""
        return self.launched_in if self.launched_in is not None else self.directory

    def launch(self) -> Optional[Dict[str, Any]]:
        """This run's launch record (`runrecord.launch_record`) -- ``None``
        while it is not written: the sender writes it as the scheduler
        answers, and a job can start first.  One that does not read is
        :class:`LaunchRecordError`.  A member of a run of points reads the
        run's, in the run's folder."""
        return _rr.launch_record(self.run_folder)

    def notified(self) -> List[Dict[str, Any]]:
        """Every event sent for the run so far (`runrecord.read_notified`)."""
        return _rr.read_notified(self.run_folder)

    def note(self, event: str, at: float,
             reason: Optional[str] = None) -> None:
        """``event`` sent for the run at ``at`` -- a failed part's
        ``reason`` with it -- into its notification record
        (`runrecord.note_notified`, `run-reports.md` § 2.5)."""
        _rr.note_notified(self.run_folder, event, at, reason=reason)

    def conclude(self, st: JobStatus) -> JobStatus:
        """How the run ended, as the Results tab reads it: `run_status` over
        this try's own files (`run-reports.md` § 2.3) -- the try this monitor
        watches, never one a warm retry started since -- its state and
        detail, each phase's convergence from the ending of the file that
        speaks, and the process's goodbye.  Asked after the closing record
        is written: a forced stop leaves no other word, and `run_status`
        reads it there."""
        try:
            rs = run_status(self.directory, self.stem, run=self.run)
        except Exception:                               # noqa: BLE001
            return st           # an unreadable directory says nothing
        st.state, st.detail = rs.state, rs.detail
        ending = rs.endings.get(rs.active_source) if rs.active_source else None
        if ending is not None and ending.phases:
            st.converged = dict(ending.phases)
        if ending is not None:
            st.relaxed = ending.relaxed
        st.exit = rs.concluded
        return st


@dataclass(frozen=True)
class RunIndex:
    """ONE KIND OF RUN OF POINTS' INDEX, as the monitor reads it
    (`run-reports.md` § 2.8): where the run keeps it, its reader, its rows
    by the kind's one done rule -- each with ``phase``, ``rel``, ``words``,
    ``done``, ``ended``, ``started``, ``ended_at``, ``started_from``,
    ``state`` -- the count in the kind's words, and what one of its points
    is called and a unit of its pace."""
    path: Callable
    read: Callable
    rows: Callable
    summary: Callable
    #: what one point of the run is called -- *call*, *point* -- its
    #: index module's own word (``UNIT``)
    unit: str
    #: ``row -> words``: a unit of the pace, for a point's row
    unit_words: Callable
    #: ``(index, rows) -> rows``: the points the walk will still run, by
    #: the kind's own rule (its index module's ``still_to_run``)
    left: Callable


#: THE RUNS OF POINTS THE MONITOR READS, each by its own index's reader,
#: which travels in this bundle: a SIESTA vibration's run in phases
#: (`siesta.vibration_phases`) and a transport sweep's run of points
#: (`transport.points_index`).  A run's index is the first whose file its
#: folder holds (:func:`_run_index_of`).
RUN_INDEXES: Tuple[RunIndex, ...] = (
    RunIndex(path=_vp.index_path, read=_vp.read_index, rows=_vp.rows_of,
             summary=lambda rows, index: _vp.summary(
                 rows, index.get("phases") or ()),
             unit=_vp.UNIT,
             unit_words=lambda row: f"{row['phase']} {_vp.UNIT}",
             # A CALL THE WALK WILL STILL RUN -- never one whose phase needs
             # a call that did not end done (`engines/vibration.md` § 5.11
             # § 5).
             left=_vp.still_to_run),
    RunIndex(path=_pi.index_path, read=_pi.read_index,
             rows=lambda index, run_dir, label: _pi.rows_of(index, run_dir),
             summary=lambda rows, index: _pi.summary(rows),
             unit=_pi.UNIT, unit_words=lambda row: _pi.UNIT,
             # A POINT THE WALK WILL STILL RUN -- never one after a point of
             # its frame that ended without being done (§ 2a.9).
             left=lambda index, rows: _pi.still_to_run(rows)),
)


def _run_index_of(run_dir: Path, names: "_rf.RunNames"
                  ) -> Optional[Tuple[RunIndex, Path]]:
    """THE RUN'S INDEX AND ITS READER (:data:`RUN_INDEXES`): the first
    whose file the run's folder holds -- named for the run's first try,
    whichever try a point is in -- or ``None``."""
    for reader in RUN_INDEXES:
        path = reader.path(run_dir, names)
        if path.is_file():
            return reader, path
    return None


def _run_clock(rows: List[Dict[str, Any]], now_epoch: float
               ) -> Optional[float]:
    """A RUN OF POINTS' CLOCK (`run-reports.md` § 2.8): since its first call
    this run computed started -- a call taken over from an earlier run is
    not this run's time; ``None`` before one has started."""
    starts = [t for t in (_rr.stamp_time(r["started"]) for r in rows
                          if r.get("started") and not r.get("started_from"))
              if t is not None]
    return (max(0.0, now_epoch - min(starts).timestamp()) if starts
            else None)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------- #
#  Utilization sampling (cpu% / mem / GPU sm% / VRAM)                    #
# --------------------------------------------------------------------- #
#
# The SAME monitor loop that watches SCF progress also samples machine
# utilization, so a post-run plot answers "were we GPU-bound or host/CPU-
# bound?" (sustained GPU sm% high => GPU-bound; low while cpu% pegged =>
# host-bound).  To keep the file small it is CHANGE-GATED like the status
# log: a row is written only when some metric moved >= ``change_frac``
# from the last logged row (or a keepalive elapsed).  With timestamps the
# sparse series still plots cleanly.  Stdlib only: ``/proc`` + nvidia-smi.


def _cgroup_paths() -> Dict[str, str]:
    """``{controller: path}`` from ``/proc/self/cgroup``, both generations.

    **The path must come from here.**  Reading ``/sys/fs/cgroup/cpu.stat``
    directly lands on the ROOT cgroup and silently answers for the whole
    node -- the very defect this reader exists to end, in a new spelling.

    v2 writes one ``0::/path`` line, registered under the key ``""``.  v1
    writes one line per hierarchy, ``id:controllers:/path``, controllers
    comma-separated -- so ``cpu,cpuacct`` is registered under its joined
    spelling (which is also the mount directory) AND under each name.
    """
    out: Dict[str, str] = {}
    try:
        with open(_PROC_CGROUP, encoding="ascii") as fh:
            for line in fh:
                bits = line.rstrip("\n").split(":", 2)
                if len(bits) != 3:
                    continue
                ctrls, path = bits[1], bits[2]
                if not ctrls:                      # v2: "0::/path"
                    out[""] = path
                    continue
                out[ctrls] = path                  # the mount-dir spelling
                for c in ctrls.split(","):
                    out.setdefault(c, path)
    except OSError:
        pass
    return out


#: ``memory.limit_in_bytes`` reads ``2**63 - 4096`` when nothing is
#: enforced.  MEASURED on ASU Sol 2026-08-26: the *task* cgroup carries
#: exactly that while the *job* cgroup one level up carries the real ask.
#: A reader that takes the sentinel for a limit reports 0% of an
#: astronomical number, so it is recognised as NO LIMIT STATED.
_NO_LIMIT = 1 << 62

#: cgroup v1's mount layout is ``/sys/fs/cgroup/<controller>/<path>``; v2's
#: is ``/sys/fs/cgroup/<path>``.  Both spellings of the cpu hierarchy's
#: directory are tried because sites mount it either way.
#:
#: These two are NAMED rather than inlined so a test can point them at a
#: fixture.  Every machine this code will ever run on has exactly one
#: answer for each, so a test that cannot supply its own is a test that can
#: only check the machine it happens to be running on -- and the whole
#: point here is reading a layout (SLURM cgroup v1) that this workstation
#: does not have.
_CG = "/sys/fs/cgroup"
_PROC_CGROUP = "/proc/self/cgroup"


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, encoding="ascii") as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _job_cgroup(path: str) -> str:
    """The JOB cgroup for a step/task path.

    SLURM nests ``<job>/step_N/task_M`` and enforces memory on the JOB.
    Measured on Sol: the task level answers with the no-limit sentinel
    while the job level answers ``8589934592`` for ``--mem=8G``.
    """
    cut = path.find("/step_")
    return path[:cut] if cut > 0 else path


def _read_cpu_used_ns() -> Optional[Tuple[int, str]]:
    """``(cpu-nanoseconds this job has consumed, which rung answered)``.

    THE NUMERATOR.  ``/proc/stat``'s aggregate line counts every process on
    the node, including other people's jobs, so it is the last resort and
    labels itself ``node`` -- a caller must be able to see when the number
    it is showing is not the job's.
    """
    cg = _cgroup_paths()
    p = cg.get("cpuacct") or cg.get("cpu")
    if p:                                          # cgroup v1
        for d in ("cpu,cpuacct", "cpuacct"):
            ns = _read_int(_CG + "/" + d + p + "/cpuacct.usage")
            if ns is not None:
                return ns, "cgroup-v1"
    v2 = cg.get("")
    if v2 is not None:                             # cgroup v2
        try:
            with open(_CG + v2 + "/cpu.stat", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("usage_usec"):
                        return int(line.split()[1]) * 1000, "cgroup-v2"
        except (OSError, ValueError, IndexError):
            pass
    node = _read_node_busy_ns()
    if node is not None:
        return node, "node"                        # NOT this job's
    return None


def _read_node_busy_ns() -> Optional[int]:
    """Node-wide busy time in nanoseconds, from ``/proc/stat``.

    The fallback rung, kept honest: converted to the same unit as the
    cgroup readings so one subtraction serves all three, and always
    reported under the label ``node`` so nobody mistakes it for the job.
    """
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            parts = fh.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        hz = os.sysconf("SC_CLK_TCK") or 100
        return int((sum(vals) - idle) * (1000000000 // hz))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _alloc_cores() -> Tuple[int, str]:
    """``(cores this job may use, which rung answered)``.

    THE DENOMINATOR, and the whole of the rule: *a run reports how well it
    used WHAT IT WAS GIVEN.*  Cores it did not ask for are unpredictable and
    are not its business -- and a fraction taken over them makes a job that
    is saturating its own allocation look starved, which argues for a bigger
    machine, which is the queue this practice exists to stay out of.

    The affinity mask answers first: measured on Sol, a ``-c 4`` job reports
    exactly 4 through it, it needs no cgroup path, and it works on both
    generations.
    """
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n, "affinity"
    except (AttributeError, OSError):
        pass
    # WHAT THE SCHEDULER GRANTED, in molbuilder's names -- the job's header
    # says them once for everything it starts (`scheduler/kinds.py`).
    for var in ("MB_CPUS_ON_NODE", "MB_CPUS_PER_TASK"):
        try:
            n = int(os.environ.get(var, ""))
            if n > 0:
                return n, var
        except ValueError:
            pass
    return (os.cpu_count() or 1), "node"


def _read_mem_used_gb() -> Optional[Tuple[float, str]]:
    """``(GB this job's cgroup holds, which rung)``, else the node's.

    ``MemTotal - MemAvailable`` is every process on the machine -- on a
    shared node, other people's jobs as much as this one's -- so it is the
    last resort and labels itself ``node``.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + p + "/memory.usage_in_bytes")
        if b is not None:
            return round(b / 1073741824.0, 2), "cgroup-v1"
    v2 = cg.get("")
    if v2 is not None:
        b = _read_int(_CG + v2 + "/memory.current")
        if b is not None:
            return round(b / 1073741824.0, 2), "cgroup-v2"
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])           # kB
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return round((info.get("MemTotal", 0) - avail) / 1048576.0, 2), "node"
    except (OSError, ValueError, IndexError):
        return None


def _read_mem_peak_gb() -> Optional[float]:
    """The kernel's OWN running peak, when it keeps one.

    v1's ``memory.max_usage_in_bytes`` is a counter the kernel maintains, so
    it is a true peak rather than the largest value a 10-second sampler
    happened to catch.  Measured on Sol it read ABOVE ``usage_in_bytes``,
    which is what proves it is not a copy of current.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + p + "/memory.max_usage_in_bytes")
        if b is not None:
            return round(b / 1073741824.0, 2)
    v2 = cg.get("")
    if v2 is not None:                             # newer v2 kernels only
        b = _read_int(_CG + v2 + "/memory.peak")
        if b is not None:
            return round(b / 1073741824.0, 2)
    return None


def _read_mem_limit_gb() -> Optional[float]:
    """The limit the kernel ENFORCES, or ``None`` when nothing is stated.

    Read from the JOB cgroup, never the task one.  ``None`` for the no-limit
    sentinel: `scheduler.md` R3 -- *an unstated limit never bars* -- and a
    sentinel is an absence wearing a number.
    """
    cg = _cgroup_paths()
    p = cg.get("memory")
    if p:
        b = _read_int(_CG + "/memory" + _job_cgroup(p)
                      + "/memory.limit_in_bytes")
        if b is not None:
            return None if b >= _NO_LIMIT else round(b / 1073741824.0, 2)
    v2 = cg.get("")
    if v2 is not None:
        try:
            with open(_CG + _job_cgroup(v2) + "/memory.max",
                      encoding="ascii") as fh:
                raw = fh.read().strip()
            return None if raw == "max" else round(int(raw) / 1073741824.0, 2)
        except (OSError, ValueError):
            pass
    return None


# --------------------------------------------------------------------- #
#  A run started DIRECTLY has no job cgroup: its job is its process tree  #
# --------------------------------------------------------------------- #
#
# Under a scheduler the job is its cgroup, and the readers above read it.
# A run started directly (`running-a-job.md` § 5.4, ``--mode direct``) has
# none: `/proc/self/cgroup` then names the cgroup the LAUNCHING session sits
# in -- a login scope, the web server's service -- and every other process
# there.  MEASURED 2026-09-26 on a real 2-rank H2 relaxation: the monitor
# reported "cpu mean=4% of 40 core(s) [affinity]; cpu time [cgroup-v2]" --
# the session's CPU over the whole node, which is § 2.1a's defect exactly.
# So there the job is the watched process and its descendants, and the cores
# it holds are the ones it was launched on.

_PROC = "/proc"


def _in_scheduler_job() -> bool:
    """Is this run inside a scheduler's job?  Then the job is its cgroup.
    The scheduler's id for it, in molbuilder's name (`scheduler/kinds.py`),
    says so."""
    return bool(os.environ.get("MB_JOB_ID"))


def _proc_stat(pid: str) -> Optional[List[str]]:
    """``/proc/<pid>/stat``'s fields after the command name (which may hold
    spaces and parentheses, so the split is at the LAST ``)``)."""
    try:
        with open(f"{_PROC}/{pid}/stat", "rb") as fh:
            raw = fh.read().decode("ascii", "replace")
    except OSError:
        return None
    return raw[raw.rfind(")") + 2:].split()


def _tree(root: int, skip: int) -> List[int]:
    """The watched process and every descendant -- the job, when it has no
    cgroup of its own -- minus ``skip`` (this monitor, a child of the
    wrapper it watches) and whatever it started."""
    kids: Dict[int, List[int]] = {}
    try:
        names = os.listdir(_PROC)
    except OSError:
        return []
    for name in names:
        if not name.isdigit():
            continue
        f = _proc_stat(name)
        if f is None or len(f) < 2:
            continue
        try:
            kids.setdefault(int(f[1]), []).append(int(name))
        except ValueError:
            continue
    out: List[int] = []
    todo = [root]
    while todo:
        pid = todo.pop()
        if pid == skip or pid in out:
            continue
        out.append(pid)
        todo.extend(kids.get(pid, ()))
    return out


def _read_tree_cpu_ns(root: int, skip: int) -> Optional[int]:
    """CPU-nanoseconds the tree has consumed: each live process's own time
    and its reaped children's (``utime + stime + cutime + cstime``), so a
    rank that has exited still counts once, in whoever waited for it.

    ``None`` -- no reading -- when a process listed a moment ago is gone: it
    may have been reaped between its parent's read and its own, and then its
    time is in neither, to land whole in the next reading as a spike.  A
    missed reading only lengthens the next interval."""
    hz = os.sysconf("SC_CLK_TCK") or 100
    total = 0
    for pid in _tree(root, skip):
        f = _proc_stat(str(pid))
        if f is None or len(f) < 15:
            return None
        try:
            total += sum(int(x) for x in f[11:15])
        except ValueError:
            return None
    return total * (1000000000 // hz)


def _read_tree_mem_gb(root: int, skip: int) -> Optional[Tuple[float, str]]:
    """``(GB the tree holds, which reading)``: each process's PROPORTIONAL
    set size where the kernel reports one (``smaps_rollup``'s ``Pss``) --
    MPI ranks map the same libraries and shared segments, and summing their
    RSS counts those pages once per rank -- else its RSS."""
    total_kb = 0
    how = "Pss"
    seen = False
    for pid in _tree(root, skip):
        kb = None
        try:
            with open(f"{_PROC}/{pid}/smaps_rollup", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        kb = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            kb = None
        if kb is None:
            try:
                with open(f"{_PROC}/{pid}/status", encoding="ascii") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            how = "RSS"
                            break
            except (OSError, ValueError, IndexError):
                kb = None
        if kb is not None:
            total_kb += kb
            seen = True
    return (round(total_kb / 1048576.0, 2), how) if seen else None


@dataclass(frozen=True)
class _Basis:
    """What this run's percentages are fractions OF and whose time and
    memory they count -- resolved ONCE, at start: a basis that changed
    mid-series would silently change what every number means (§ 2.1a)."""
    cores: int
    cores_from: str
    #: The job's process-tree root when it has no cgroup of its own.
    tree: Optional[int] = None

    def cpu_ns(self) -> Optional[Tuple[int, str]]:
        if self.tree is None:
            return _read_cpu_used_ns()
        ns = _read_tree_cpu_ns(self.tree, os.getpid())
        return (ns, "process tree") if ns is not None else None

    def mem_gb(self) -> Optional[Tuple[float, str]]:
        if self.tree is None:
            return _read_mem_used_gb()
        got = _read_tree_mem_gb(self.tree, os.getpid())
        return (got[0], f"process tree {got[1]}") if got else None


def _basis(watch_pid: int = 0, cores: Optional[int] = None) -> _Basis:
    """The job this run's numbers are about.  Under a scheduler -- or with
    no process to watch -- its cgroup and its allocation; started directly,
    the watched process tree and the cores it was launched on (``cores``,
    the wrapper's ranks x threads), else its affinity, labelled as such."""
    if watch_pid > 0 and not _in_scheduler_job():
        if cores and cores > 0:
            return _Basis(int(cores), "launched on", tree=watch_pid)
        n, src = _alloc_cores()
        return _Basis(n, src, tree=watch_pid)
    n, src = _alloc_cores()
    return _Basis(n, src)


_GPU_QUERY = ["nvidia-smi",
              "--query-gpu=index,utilization.gpu,utilization.memory,"
              "memory.used",
              "--format=csv,noheader,nounits"]


def _gpu_models() -> List[str]:
    """The distinct device models this job can see, via ``nvidia-smi -L``.

    ``nvidia-smi -L`` prints one line per visible device::

        GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-8f...)

    and the model is the text between the index and the UUID.  Distinct
    models only, first-seen order: the **count** is deliberately not
    reported anywhere, because inside a scheduled job the device cgroup
    shows only what the job was granted -- a count would be the
    allocation's, not the node's (`scheduler.md` R12).  Empty list when no
    device is visible or the tool is absent, which are the same honest
    answer: this run could not touch one.
    """
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           text=True, timeout=5)
        if r.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []
    models: List[str] = []
    for line in r.stdout.splitlines():
        m = re.match(r"GPU \d+:\s*(.+?)\s*\(UUID", line)
        if m and m.group(1) not in models:
            models.append(m.group(1))
    return models


def _gpu_present() -> bool:
    return bool(_gpu_models())


def machine_identity() -> Dict[str, str]:
    """What kind of node is under this run -- `scheduler.md` **R12**.

    NOT the allocation.  Every source here reads the NODE and ignores what
    this job was given, and each choice is load-bearing:

    * ``cores`` is ``os.cpu_count()`` -- the online processors, which a
      cgroup does not shrink.  ``_alloc_cores()`` reads the affinity mask,
      which the scheduler DOES shrink; use it here and a rank-scaling sweep
      (48 vs 64 vs 128 ranks on identical nodes) would report a different
      machine per trial, breaking the comparison this record exists for.
    * ``mem_gb`` is ``/proc/meminfo`` MemTotal -- the node's, where the
      cgroup limit is the allocation's.
    * ``gpu`` is the device **model** (:func:`_gpu_models`), never a count.

    ``node`` is the one exception: the host NAME is provenance (which box,
    for tracing a bad one), not identity -- SLURM spreads a sweep over
    whatever boxes are free, so readers compare the other fields and only
    report the name (`scheduler.md` R11).
    """
    ident: Dict[str, str] = {}
    try:
        ident["node"] = os.uname().nodename[:128]
    except (AttributeError, OSError):
        ident["node"] = "?"
    ident["cores"] = str(os.cpu_count() or 0)
    mem = ""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    mem = f"{int(line.split()[1]) / 1048576.0:.1f}"
                    break
    except (OSError, ValueError, IndexError):
        pass
    ident["mem_gb"] = mem or "?"
    ident["gpu"] = ", ".join(_gpu_models()) or "none"
    return ident


def machine_line() -> str:
    """:func:`machine_identity` as the ``[MACHINE]`` line's payload.

    ``gpu`` comes LAST because device models contain spaces: a reader
    splits the three fixed ``key=value`` fields and takes the rest of the
    line as the model list.
    """
    m = machine_identity()
    return (f"node={m['node']} cores={m['cores']} "
            f"mem_gb={m['mem_gb']} gpu={m['gpu']}")


def _sample_gpus() -> List[Tuple[int, float, float, float]]:
    """Per-GPU ``(index, sm%, mem_util%, vram_gb)`` via nvidia-smi
    (empty list on any failure -- best-effort)."""
    try:
        r = subprocess.run(_GPU_QUERY, capture_output=True, text=True,
                          timeout=10)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 4:
                out.append((int(f[0]), float(f[1]), float(f[2]),
                            round(float(f[3]) / 1024.0, 2)))   # MiB -> GB
        return out
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _metric_moved(new, old, frac: float) -> bool:
    """True if ``new`` differs from ``old`` by >= ``frac`` (relative, with
    a floor of 1 so near-zero values still register a real jump)."""
    if old is None or new is None:
        return new is not None
    return abs(new - old) >= max(abs(old), 1.0) * frac


@dataclass
class UtilSample:
    epoch: float
    cpu_pct: Optional[float]
    mem_gb: Optional[float]
    gpus: List[Tuple[int, float, float, float]] = field(default_factory=list)

    def changed_from(self, other: "UtilSample", frac: float) -> bool:
        if other is None:
            return True
        if _metric_moved(self.cpu_pct, other.cpu_pct, frac):
            return True
        if _metric_moved(self.mem_gb, other.mem_gb, frac):
            return True
        og = {g[0]: g for g in other.gpus}
        for idx, sm, mu, vram in self.gpus:
            o = og.get(idx)
            if o is None or _metric_moved(sm, o[1], frac) \
                    or _metric_moved(vram, o[3], frac):
                return True
        return False


@dataclass
class UtilAccum:
    """Running cpu% + per-GPU sm% stats for the end-of-run verdict.
    Flat memory (sum/min/max/count), so it is safe for arbitrarily long
    runs -- no per-sample list."""
    cpu_sum: float = 0.0
    cpu_n: int = 0
    cpu_min: float = 1e9
    cpu_max: float = -1e9
    gpu_sum: Dict[int, float] = field(default_factory=dict)
    gpu_n: Dict[int, int] = field(default_factory=dict)
    gpu_min: Dict[int, float] = field(default_factory=dict)
    gpu_max: Dict[int, float] = field(default_factory=dict)

    def add(self, s: UtilSample) -> None:
        if s.cpu_pct is not None:
            self.cpu_sum += s.cpu_pct
            self.cpu_n += 1
            self.cpu_min = min(self.cpu_min, s.cpu_pct)
            self.cpu_max = max(self.cpu_max, s.cpu_pct)
        for idx, sm, _mu, _vram in s.gpus:
            self.gpu_sum[idx] = self.gpu_sum.get(idx, 0.0) + sm
            self.gpu_n[idx] = self.gpu_n.get(idx, 0) + 1
            self.gpu_min[idx] = min(self.gpu_min.get(idx, 1e9), sm)
            self.gpu_max[idx] = max(self.gpu_max.get(idx, -1e9), sm)

    def summary(self) -> str:
        bits = []
        if self.cpu_n:
            bits.append(f"cpu mean={self.cpu_sum / self.cpu_n:.0f}% "
                        f"({self.cpu_min:.0f}-{self.cpu_max:.0f})")
        gpu_means = []
        for idx in sorted(self.gpu_n):
            m = self.gpu_sum[idx] / self.gpu_n[idx]
            gpu_means.append(m)
            bits.append(f"gpu{idx} sm mean={m:.0f}% "
                        f"({self.gpu_min[idx]:.0f}-{self.gpu_max[idx]:.0f})")
        verdict = ""
        if gpu_means:
            mx = max(gpu_means)
            if mx >= 85:
                verdict = " -> GPU-bound (GPU saturated)"
            elif mx <= 60:
                verdict = " -> host/CPU-bound (GPU starved)"
            else:
                verdict = " -> mixed (GPU not saturated)"
        return "; ".join(bits) + verdict


def _util_csv_header(ngpu: int) -> str:
    cols = ["epoch", "iso", "cpu_pct", "mem_gb"]
    for i in range(ngpu):
        cols += [f"gpu{i}_sm", f"gpu{i}_memutil", f"gpu{i}_vram_gb"]
    return ",".join(cols)


def _util_csv_row(s: UtilSample, ngpu: int) -> str:
    def _f(v):
        return "" if v is None else f"{v:g}"
    cells = [f"{s.epoch:.0f}", _iso(s.epoch), _f(s.cpu_pct), _f(s.mem_gb)]
    by_idx = {g[0]: g for g in s.gpus}
    for i in range(ngpu):
        g = by_idx.get(i)
        if g is None:
            cells += ["", "", ""]
        else:
            cells += [_f(g[1]), _f(g[2]), _f(g[3])]
    return ",".join(cells)


# --------------------------------------------------------------------- #
#  Notifier hooks                                                       #
# --------------------------------------------------------------------- #

# A notifier is ``fn(status, event)`` where ``event`` is one of
# "start" | "scf_converged" | "periodic" | "wall_near" | "milestone" |
# "part_failed" | "finish" (`run-reports.md` § 2).
# Register as many as you like; they are
# called in registration order and individually guarded (one failing
# hook never breaks the loop or the other hooks).
Notifier = Callable[[JobStatus, str], None]

_NOTIFIERS: List[Notifier] = []


def register_notifier(fn: Notifier) -> None:
    """Add a notifier hook.  This is the connection point for a real
    push (webhook/email/molwatch)."""
    _NOTIFIERS.append(fn)


def clear_notifiers() -> None:
    """Drop all registered notifiers (used by tests)."""
    _NOTIFIERS.clear()


def _fire(status: JobStatus, event: str) -> None:
    for fn in list(_NOTIFIERS):
        try:
            fn(status, event)
        except Exception as exc:                       # noqa: BLE001
            # A notifier must never break monitoring.
            print(f"[monitor] notifier {getattr(fn, '__name__', fn)!r} "
                  f"raised: {exc!r}", flush=True)


#: How long a POST may take before the monitor gives up on it.  Short on
#: purpose: this runs beside compute ranks, the server may be down, and a
#: notification is never worth costing the run anything.  Each hook is also
#: individually guarded, so a dead endpoint cannot break the loop either.
NOTIFY_TIMEOUT_S = 2.0

#: The override that reports to one address instead of the channels file,
#: and the key a molbuilder listener there needs (`run-reports.md` § 3).
NOTIFY_URL_VAR = "MB_NOTIFY_URL"
NOTIFY_KEY_VAR = "MB_NOTIFY_KEY"

#: The destination file's name inside molbuilder's config directory.  NOT in
#: the calculation's description: `task.json` travels -- to a cluster, into a
#: citation's composed copy, to whoever is handed the calculation -- and a token must
#: not travel with it.  The policy is in the description; the secret is here.
NOTIFY_FILENAME = "notify"
#: The operator's signing keys for those reports.  Named beside the file they
#: sign for, because this module owns that exchange's format (A11).
NOTIFY_KEYS_FILENAME = "notify_keys"


def _secrets_dir():
    """Where every credential lives, from the one module that defines it.

    Imported two ways because this file runs two ways: inside the package on
    a login node, and from `mb_monitor.pyz` on a compute node with no
    molbuilder installed, where `config_dir.py` travels with it in that one
    file (`runwrap.MONITOR_BUNDLE`), so the SAME LINES answer on both
    machines.

    Restating the rule here -- joining ``config_dir() / "secrets"`` -- would
    be another copy of it, which review exists to refuse
    (`process/code-audit.md` § 1c): *"a comment is not a mechanism"*.

    Raises `ModuleNotFoundError` when the companion is absent, which
    `load_channels` catches: absent is reports-off, never a dead monitor.
    """
    try:
        from .config_dir import secrets_dir     # inside the package
    except ImportError:                          # in the shipped bundle
        from config_dir import secrets_dir
    return secrets_dir()


def default_notify_path() -> Path:
    """Where the destination file lives: ``<config dir>/secrets/notify``.

    Honouring ``XDG_CONFIG_HOME`` -- which :func:`_secrets_dir` does, because
    `config_dir.py` does -- is load-bearing HERE in particular.  On an HPC
    login node ``$HOME`` is NFS-mounted and often snapshotted, and
    ``XDG_CONFIG_HOME=/scratch/$USER`` is how a person keeps a token off it.
    This file is read on a compute node; a path hardcoded to ``$HOME`` would
    give them no way to.
    """
    return _secrets_dir() / NOTIFY_FILENAME


def notify_keys_path():
    """The operator's run-report signing keys.

    Beside :func:`default_notify_path` because this module owns the format of
    the exchange they belong to (A11).
    """
    return _secrets_dir() / NOTIFY_KEYS_FILENAME


# ── The server's key file, whole (A11: this module owns the format) ──────────
#
# IT CARRIES ITS OWN ROUTE (user, 2026-08-31: *"why would you have to repeat
# the notify_route twice, with one in the file and one in the molbuilder.json
# which has its own pointer to that same file"*).
#
# One place for one fact (`run-reports.md` § 4.3): `notify-token` cannot read
# molbuilder.json, so a route kept there would let a second key generate a
# NEW segment and move the route out from under everyone already set up --
# silently, because a notifier swallows failures.
#
# With the route in here the command reads what it already issued, the server
# reads the same file, and `molbuilder.json` needs nothing at all.

#: One URL segment: letters, digits, '-' or '_'.  A value with a slash in it
#: would silently mean a different path than the one written down, and the
#: destination file's url is built from this -- so the two ends would disagree
#: about where reports go.  The rule belongs with the file that carries the
#: value.
_ROUTE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def read_notify_keys():
    """``(route, {user: key})`` from the operator's 0600 file.

    ``(None, {})`` when the file is absent, unreadable, malformed, or carries
    no route -- and that is the safe reading in every case: no route means the
    listener is not registered, and no keys means it accepts nothing.  A
    misconfiguration here removes a capability; it never grants one.
    """
    p = notify_keys_path()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    route = obj.get("route")
    keys = obj.get("keys")
    if not isinstance(route, str):
        return None, {}
    route = route.strip().strip("/")      # whitespace first, then the slashes
    if not is_route_segment(route):
        return None, {}
    if not isinstance(keys, dict):
        return None, {}
    return route, {str(k): str(v) for k, v in keys.items()
                   if isinstance(v, str) and v}


def notify_keys_document(route, keys):
    """The file's bytes, from the two things it holds.

    One writer, so the shape cannot drift from :func:`read_notify_keys`.
    """
    return json.dumps({"route": route, "keys": dict(keys)}, indent=2) + "\n"



@dataclass(frozen=True)
class NotifyPolicy:
    """WHEN to speak -- the two settable occasions, as one value.

    They are one thing everywhere else: born together in `task.Notify`,
    carried together on `jobset.Resources`, consumed together here.  Passing
    them as two loose arguments is the shape `architecture.md` § 3.1's rule
    A8 forbids, and for a measured reason -- a caller re-assembling an
    object the callee should have been handed is how a third field gets
    forgotten.

    Not imported from `task.Notify`: this module ships to a compute node as
    a standalone file with no molbuilder importable (see the module
    docstring).  The wire between them is the CLI flag pair.
    """
    #: WHAT each report carries beyond the name (`stages.md` § 6.9).
    #: `None` is every field this monitor could determine; `()` is the
    #: summary line alone.
    report: Optional[Tuple[str, ...]] = None

    on_scf: bool = False
    every_hours: float = 0.0
    #: WHICH channels, by name.  ``None`` means every channel this machine
    #: has (``--notify-channels *``, :func:`_channels_from_flag`).
    #: An EMPTY tuple is the opposite and is not the same state: reports off
    #: for this calculation on a machine where they are set up.  The two
    #: spellings exist because they are two intentions (`run-reports.md`
    #: 3.0); collapsing them would send a report to a channel the person
    #: had just unticked.
    channels: Optional[Tuple[str, ...]] = None


def _notify_say(msg: str, log: Optional[Path] = None) -> None:
    """Say something about notification setup, where it can be READ.

    The wrapper backgrounds this process with its stdout at ``/dev/null``
    (its stderr goes to the session log, which carries its start --
    `run-reports.md` § 2.6), so anything printed goes nowhere: a misconfigured channel would produce no
    notifications and no explanation, which is the worst of both.  ``log`` is
    the monitor log the user actually opens; printing is the fallback for a
    caller that has no log yet -- an interactive `molbuilder monitor`, or a
    test.

    **One writer**: one function written twice is two places for a fix to
    miss.
    """
    line = f"[{_iso(time.time())}] [NOTIFY] {msg}"
    if log is not None:
        _append(Path(log), line)
    else:
        print(f"[monitor] {msg}", flush=True)


def is_route_segment(route) -> bool:
    """Is this a usable run-report route segment?  One home for the rule.

    **It has to be asked at BOTH ends.**  :func:`read_notify_keys` applies
    `_ROUTE_RE` on the way IN and returns ``(None, {})`` for anything failing
    it, so a bad route written unchecked leaves the server with no route and
    NO KEYS AT ALL -- every key already issued stops working, and silently,
    because a notifier swallows failures by design.
    `auth_setup.issue_notify_key` asks it on the way OUT.

    This module ships to a compute node as a standalone stdlib-only file,
    so it owns the rules for the exchange it defines and nobody restates
    them.
    """
    return bool(isinstance(route, str) and _ROUTE_RE.fullmatch(route))


#: A run-report user id: it becomes a log FILENAME on the listener, so it is
#: what is safe as one -- no path separator, nothing a shell reads.
_USER_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")


def is_notify_user(user) -> bool:
    """Is this a usable run-report user id?  One home for the rule, asked at
    both ends: when a key is issued (`auth_setup.issue_notify_key` -- a key
    that authenticates and then cannot be recorded is worse than none) and
    when a report arrives (`web/blueprints/notify.py`, which writes the log
    file it names)."""
    return bool(isinstance(user, str) and _USER_RE.fullmatch(user))


#: What molbuilder calls itself on the wire.  **Not decoration**: Discord's
#: edge answers a default ``Python-urllib/3.x`` with ``403`` and Cloudflare
#: code 1010 -- before the request reaches Discord, so a live webhook and a
#: deleted one look identical and the status says nothing about either
#: (`run-reports.md` § 4.1b).
USER_AGENT = "molbuilder (https://github.com/qqing/molbuilder, 1.0)"

def _card(report: Dict[str, Any],
          items: "Optional[Tuple[str, ...]]",
          receiver: str,
          layout: "Optional[_cards.Layout]") -> Dict[str, Any]:
    """The report as a CHAT CARD, before either vocabulary is chosen.

    Discord embeds, Slack attachments and the This machine tab's card are
    the same picture -- a coloured bar, a title, a summary, a grid of
    fields -- so the picture is built once here and each sender only
    spells it (`run-reports.md` § 4.1b).  Building it twice is how two
    channels of the same event stop looking alike.  HOW it is laid out for
    ``receiver`` -- the fields' order for the report's event, side by side
    or not, how many at most, whether the summary shows, the bar's colour
    -- is the layout file's (`run-reports.md` § 4.1c), read here when the
    caller did not read it once already.
    """
    lay = layout if layout is not None else _cards.load()
    rc = lay.receiver(receiver)
    state = str(report.get("state") or "")
    # THE NAME IS ALWAYS IN THE TITLE.  A report you cannot attribute to a
    # job is a notification you have to go and look up, which is the thing
    # a notification exists to save you (`stages.md` § 6.9).
    run = str(report.get("run") or "molbuilder")
    job = str(report.get("job") or "")
    title = run + (f" · {job}" if job else "") + (f" — {state}" if state else "")
    # `items` IS A CEILING, NEVER A FLOOR (`stages.md` § 6.9): the layout's
    # order, narrowed by it, then by what the run stated, then cut at the
    # receiver's limit (`report_cards.Layout.fields`).
    text = str(report.get("text") or "") or "(no summary)"
    return {"title": title[:256],
            "text": text if rc.summary else "",
            "color": lay.color(state),
            "fields": lay.fields(receiver, report, items),
            "inline": rc.inline}


def card(report: Dict[str, Any],
         items: "Optional[Tuple[str, ...]]" = None,
         receiver: str = "web",
         layout: "Optional[_cards.Layout]" = None) -> Dict[str, Any]:
    """The chat card's picture for ``report`` -- what Slack and Discord are
    sent (:func:`_card`) -- laid out for ``receiver``, the This machine
    tab's by default (`this-machine.md` § 4.1).  A page drawing many reads
    the layout once and passes it."""
    return _card(report, items, receiver, layout)


#: The three wire formats.  `run-reports.md` § 4.1b.
_KINDS = ("molbuilder", "slack", "discord")


def channel_kind(dest: Dict[str, Any]) -> str:
    """Which wire format this channel wants -- DECLARED, host as the default.

    `run-reports.md` § 4.1b.  The file may say ``"kind"``; absent, the URL's
    host supplies one.  Declared wins, so a webhook reached through a proxy
    or a relay is still expressible -- a rule discovered from a string stops
    holding the moment the string changes.
    """
    said = str(dest.get("kind") or "").strip().lower()
    if said in _KINDS:
        return said
    host = urllib.parse.urlparse(str(dest.get("url") or "")).hostname or ""
    host = host.lower()
    if host == "hooks.slack.com":
        return "slack"
    if host in ("discord.com", "www.discord.com", "discordapp.com",
                "ptb.discord.com", "canary.discord.com"):
        return "discord"
    return "molbuilder"


#: The schemes a channel's address may take (`run-reports.md` § 3).
_SCHEMES = ("https", "http")


def address_refusal(url, kind: Optional[str] = None) -> Optional[str]:
    """Why ``url`` cannot be a channel's address, or ``None`` -- THE ONE
    RULE, asked by the This machine tab's save and by :func:`load_channels`
    (`run-reports.md` § 3): it reads as ``scheme://host/…`` with no space or
    control character; its scheme is ``https``, or ``http`` for a molbuilder
    listener on a host you trust; and a channel whose address IS its
    credential -- Slack, Discord (:func:`channel_kind`, ``kind`` as
    declared) -- is ``https``, since over ``http`` the credential would travel
    unencrypted.  **The sentence never quotes the address**: for those two
    kinds it is the secret."""
    url = str(url or "")
    if not url:
        return "the address is empty"
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        return "the address holds a space or a control character"
    try:
        parts = urllib.parse.urlsplit(url)
        parts.port                       # a port that is no number raises
    except ValueError:
        return "the address does not read as a web address (scheme://host/…)"
    if parts.scheme.lower() not in _SCHEMES or not parts.hostname:
        return ("the address must start with https:// -- or http:// for a "
                "molbuilder listener on a host you trust")
    which = channel_kind({"url": url, "kind": kind})
    if which in ("slack", "discord") and parts.scheme.lower() != "https":
        return (f"a {which} address is its credential, so it must start "
                f"with https:// -- over http:// it would travel unencrypted")
    return None


def delivery_failure(exc: BaseException) -> str:
    """What a failed send says -- THE ONE COMPOSER, for the monitor's log
    and the This machine tab's Test (`run-reports.md` § 3): what failed,
    from the exception's structured parts -- an HTTP status, the operating
    system's own words, a TLS reason -- and **never the exception's text**,
    which quotes the address it was sent to, for Slack and Discord the
    credential."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"the service answered HTTP {exc.code}"
    reason = (exc.reason if isinstance(exc, urllib.error.URLError)
              else exc)
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "it timed out"
    if isinstance(reason, ssl.SSLError):
        return ("the TLS handshake failed"
                + (f" ({reason.reason})" if getattr(reason, "reason", None)
                   else ""))
    if isinstance(reason, socket.gaierror):
        return "the host could not be found"
    if isinstance(reason, ConnectionRefusedError):
        return "the connection was refused"
    if isinstance(reason, OSError) and reason.strerror:
        return f"the connection failed: {reason.strerror}"
    return f"it could not be sent ({type(exc).__name__})"


def webhook_request(dest: Dict[str, Any],
                    report: Dict[str, Any],
                    items: "Optional[Tuple[str, ...]]" = None,
                    layout: "Optional[_cards.Layout]" = None
                    ) -> "tuple[bytes, Dict[str, str]]":
    """THE ONE PRODUCER of a webhook's ``(body, headers)`` pair.

    The report is one thing; the ENVELOPE it travels in is the
    destination's, and the three destinations do not read the same one
    (`run-reports.md` § 4.1b).  Every sender calls this -- the monitor's
    notifier and the setup page's *Test* button both: built separately, the
    button that exists to prove the path could pass while the path failed.
    ``layout`` is the card's layout read once (`report_cards.load`); absent,
    it is read here.
    """
    kind = channel_kind(dest)
    head = {"Content-Type": "application/json", "User-Agent": USER_AGENT}

    if kind in ("slack", "discord"):
        card = _card(report, items, kind, layout)

        if kind == "slack":
            # `attachments` rather than a bare `text`: same picture as the
            # Discord embed, in Slack's words.  `text` at the top level is
            # the notification/fallback line, which is what a phone shows.
            # The field count is the layout's, capped at Slack's own ten
            # when it is read (`report_cards.WIRE_LIMIT`).
            att = {"color": "#%06X" % card["color"],
                   "title": card["title"],
                   "text": card["text"],
                   "fallback": card["title"]}
            if card["fields"]:
                att["fields"] = [{"title": n, "value": v,
                                  "short": card["inline"]}
                                 for n, v in card["fields"]]
            return json.dumps({"text": card["title"],
                               "attachments": [att]}).encode(), head

        # DISCORD IGNORES `text` ENTIRELY.  Execute Webhook wants at least one
        # of content / embeds / components / file / poll, and a body with none
        # of them is refused 400.  The field count is the layout's, capped at
        # Discord's own twenty-five when it is read.
        embed = {"title": card["title"],
                 "description": card["text"][:4096],
                 "color": card["color"]}
        if card["fields"]:
            embed["fields"] = [{"name": n, "value": v,
                                "inline": card["inline"]}
                               for n, v in card["fields"]]
        return json.dumps({"embeds": [embed]}).encode(), head

    # A molbuilder listener: the `run-reports.md` § 4.1a record, whole,
    # signed.
    body = json.dumps(report).encode()
    head.update(dest.get("headers") or {})
    if dest.get("key"):
        # The timestamp is signed WITH the body, not sent beside it --
        # otherwise it could be rewritten freely.
        ts = "%d" % int(time.time())
        head[TIMESTAMP_HEADER] = ts
        head[SIGNATURE_HEADER] = sign_report(dest["key"], ts, body)
    return body, head


def load_channels(log: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """The named webhook channels, or ``{}`` when none are set up.

    The file is ``{"channels": {name: {"url": ..., "key"?: ...,
    "kind"?: ..., "headers"?: ...}}}``.  ``kind`` is
    ``"molbuilder" | "slack" | "discord"`` and says which WIRE FORMAT the
    destination reads; absent, it is read off the URL's host
    (`run-reports.md` § 4.1b).  Two shapes of channel, one mechanism -- Slack and
    Discord put the credential IN the url, because a third party handed
    nothing but a URL has nowhere else to keep one; a molbuilder listener
    takes a plain url and a ``key`` that **signs the body and never
    travels**.  Either way the file is the user's own, mode 0600, and
    nothing here is created on anybody's behalf.

    **Absent is not an error.**  No file means no notifier, and the run
    proceeds exactly as it does with the feature switched off.  A malformed
    file is not an error either: this is a monitor, and refusing to watch a
    job because a notification could not be configured would be the tail
    wagging the dog.  It says so and carries on.

    **One bad channel does not cost the others.**  A file with three
    channels and a typo in the second reports on two.  Refusing the file
    whole would turn one mistake into total silence, which is the failure
    this whole area keeps producing.

    **It says so in the LOG**, not on stdout.  The wrapper backgrounds this
    process with its stdout at ``/dev/null`` (its stderr goes to the session
    log, which carries its start), so anything printed goes nowhere:
    a misconfigured channel would produce no notifications and no
    explanation, which is the worst of both.  ``log`` is the monitor log the
    user actually reads.  Printing is the fallback for a caller that has no
    log yet -- an interactive `molbuilder monitor`, or a test.
    """
    def _say(msg: str) -> None:
        _notify_say(msg, log)

    try:
        # ONE RESOLVER, no `path=`.  For Slack and Discord the URL in this
        # file IS the credential, so it is reached the way every other
        # credential in this tree is: through the door that owns it.  A second
        # way to name a file with one home is how a file gets written where
        # nothing reads it.
        p = default_notify_path()
        try:
            from .config_dir import is_channel_name   # inside the package
        except ImportError:                            # in the shipped bundle
            from config_dir import is_channel_name
    except ImportError:
        # The shipped monitor could not find `config_dir.py` beside it, so
        # WHERE the channels live cannot be answered.  Absent is off
        # (`run-reports.md` § 3): a monitor that cannot report must still
        # MONITOR -- dying here would cost every status line, the util
        # series and the [MACHINE] record, silently.
        _say("no config_dir.py beside the monitor -- reports off")
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        # NO CHANNELS FILE ON THIS MACHINE is reports off -- and SAID, once,
        # for the same reason a named channel absent here is said
        # (`run-reports.md` § 3.0): a calculation that asked for every
        # channel on this machine and found none sends nothing, which is
        # indistinguishable from working unless the log says so.  Channels
        # are set up on the machine that RUNS the job (its This machine
        # tab, or the file copied there), not where it was prepared.
        _say(f"no channels set up on this machine ({p}): nothing is sent -- "
             f"set them up on the machine that runs the job (its This "
             f"machine tab), where the monitor reads them")
        return {}
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        _say(f"{p}: not valid JSON ({exc}); not notifying")
        return {}
    if not isinstance(obj, dict):
        _say(f"{p}: needs a JSON object; not notifying")
        return {}
    chans = obj.get("channels")
    if not isinstance(chans, dict):
        # NAMED: the single-destination file is not read.  Saying which is the whole point: `{"url": ...}` is a valid
        # JSON object, so a silent skip here is indistinguishable from never
        # having set anything up -- the exact failure `run-reports.md` 3.1
        # exists to stop.
        if isinstance(obj.get("url"), str):
            _say(f"{p}: this is the old single-destination file. It is now a "
                 f"map of named channels -- re-save it on the This machine "
                 f"tab, or wrap it as "
                 f'{{"channels": {{"<name>": {{...}}}}}}. Not notifying')
        else:
            _say(f"{p}: needs a 'channels' object; not notifying")
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in chans.items():
        name = str(name)
        if not is_channel_name(name):
            _say(f"{p}: channel name {name!r} is not letters, digits, '-' "
                 f"or '_'; skipping it")
            continue
        if not isinstance(spec, dict) or not isinstance(spec.get("url"), str) \
                or not spec["url"]:
            _say(f"{p}: channel {name!r} needs an object with a 'url' "
                 f"string; skipping it")
            continue
        headers = spec.get("headers") or {}
        if not isinstance(headers, dict):
            _say(f"{p}: channel {name!r}: 'headers' must be an object; "
                 f"ignoring them")
            headers = {}
        key = spec.get("key")
        if key is not None and not (isinstance(key, str) and key):
            _say(f"{p}: channel {name!r}: 'key' must be a non-empty string; "
                 f"skipping it")
            continue
        kind = spec.get("kind")
        if kind is not None and str(kind).strip().lower() not in _KINDS:
            # NAMED AND WRONG is not the same as absent: the host would
            # supply a default here, and a typo'd `"kind": "discrod"` would
            # silently take it -- sending a Slack-shaped body to Discord and
            # getting a 400 nobody could trace back to a spelling.
            _say(f"{p}: channel {name!r}: 'kind' must be one of "
                 f"{', '.join(sorted(_KINDS))}; reading it off the URL "
                 f"instead")
            kind = None
        why = address_refusal(spec["url"], kind)
        if why:
            # THE ADDRESS SAID BY ITS FAULT, NEVER QUOTED: for Slack and
            # Discord it is the credential (`run-reports.md` § 3).
            _say(f"{p}: channel {name!r}: {why}; skipping it")
            continue
        out[name] = {"url": spec["url"], "key": key, "kind": kind,
                     "headers": {str(k): str(v) for k, v in headers.items()}}
    return out


def _report_from_flag(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """``--notify-report`` as :class:`NotifyPolicy` carries it.

    Absent is ``None`` -- every field this monitor can determine.  ``""`` is
    ``()`` -- the summary line with no grid.  The two are different answers
    and the default of ``None`` is what keeps them apart
    (`stages.md` § 6.9).

    **An unknown name is dropped, not fatal.**  The wrapper already refused
    one at `prep`; a monitor that died here would cost a running job its
    whole report over a field it could simply not show.
    """
    if value is None:
        return None
    return tuple(n for n in (p.strip() for p in value.split(","))
                 if n in _fields.NAMES)


def _channels_from_flag(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """``--notify-channels`` as :class:`NotifyPolicy` carries it.

    The flag ABSENT is nothing set up, so nothing is sent: ``()``
    (user, 2026-09-26: *"when no set up for notification that means no
    notification"*).  ``*`` (``config_dir.ALL_CHANNELS``) is every channel
    this machine has -- ``None`` in the policy -- and ``""`` is none.  **The
    reason this is a function**: those spellings are a character apart on a
    command line and opposite in meaning.
    """
    if value is None:
        return ()
    if value.strip() == "*":
        return None
    return tuple(n for n in (part.strip() for part in value.split(",")) if n)


def channels_for(wanted: Optional[Sequence[str]],
                 available: Dict[str, Dict[str, Any]]
                 ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """``(chosen, missing)`` -- **the one door** for `run-reports.md` 3.0.

    ``wanted`` is the description's ``notify.channels``: ``None`` for every
    channel this machine has, a list for those and only those, an empty list
    for none at all.  Written once here rather than at each caller, because
    a rule with three cases and two of them spelled almost the same is the
    shape that gets restated slightly wrong.

    ``missing`` is what the description named and this machine does not
    have.  Returned rather than logged, so the caller decides where it goes
    -- and it must go SOMEWHERE: a channel that resolves to nothing is
    silent by design, and this is the only moment anything knows.
    """
    if wanted is None:
        return dict(available), []
    chosen, missing = {}, []
    for name in wanted:
        if name in available:
            chosen[name] = available[name]
        else:
            missing.append(name)
    return chosen, missing


#: The two headers a signed report carries, named once for both ends (the
#: listener reads them by these names, `web/blueprints/notify.py`).
TIMESTAMP_HEADER = "X-Molbuilder-Timestamp"
SIGNATURE_HEADER = "X-Molbuilder-Signature"


def sign_report(key: str, timestamp: str, body: bytes) -> str:
    """The signature a report carries, and the one the listener checks it
    against: **one definition**.  This file ships to a compute node as a
    standalone stdlib-only script, and the server imports this same function
    (`web/blueprints/notify.py`), so the two ends cannot come to disagree.

    The timestamp is inside the signed material rather than beside it:
    signing the body alone would leave the timestamp free to be rewritten.

    Why a signature rather than a bearer token: a token is on the wire every time, so one capture yields a
    credential good forever and for any body.  This key never leaves the
    cluster, and what travels is valid for one exact body.
    """
    msg = timestamp.encode("ascii", "replace") + b"." + body
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def make_webhook_notifier(url: str, *,
                          key: Optional[str] = None,
                          headers: Optional[Dict[str, str]] = None,
                          ident: Optional[Dict[str, str]] = None,
                          kind: Optional[str] = None,
                          report: Optional[Sequence[str]] = None,
                          layout: "Optional[_cards.Layout]" = None,
                          channel: Optional[str] = None,
                          log: Optional[Path] = None
                          ) -> Notifier:
    """A stdlib webhook notifier: POSTs ``event`` + the status summary to
    ``url`` as JSON.  Best-effort, short timeout, never raises out: a send
    that fails costs the run nothing and is SAID in the monitor log ``log``
    -- the channel's name and what failed (:func:`delivery_failure`), never
    the address (`run-reports.md` § 3).  ``channel`` is the name it was
    chosen by; ``None`` is ``MB_NOTIFY_URL``'s.  ``layout`` is the card's
    layout the monitor read at its start (`run-reports.md` § 4.1c).

    JSON rather than form encoding because both destinations want it: Slack
    and Discord read a JSON body, and a private endpoint that appends to a
    record log wants structure rather than one flattened string.

    ``key`` signs the body for a molbuilder listener.  ``headers`` stays for
    a third party that has no other way to be told who is calling -- Slack
    and Discord put the credential in the URL itself, so they need neither.

    **Both are keyword-only, deliberately**: two optional parameters of
    different types in a row invite a call site to pass one where the other
    goes -- binding silently, and failing only later inside the signing.
    """
    def _hook(status: JobStatus, event: str) -> None:
        record = {
            # WHO AND WHERE, first, so a line is self-contained: run label,
            # scheduler job id, host.  `sent_at` is the SENDER's clock --
            # the listener stamps its own arrival separately, and when they
            # disagree that is itself worth seeing.
            **(ident or {}),
            "sent_at":    round(time.time(), 3),
            "event":      event,
            "text":       status.as_text(),
            # The one-line summary.  A chat card uses it as its description;
            # our own listener parses the fields beside it.  It is NOT the
            # whole Discord body -- Discord ignores a bare `text`
            # (`run-reports.md` § 4.1b).
            "state":      status.state,
            # THE PART UNDERWAY, which a field of an ended part is said by
            # (`report_fields.ReportField.ended`, § 2.8).
            "part":       status.part,
            # WHAT A MEMBER'S POINT IS CALLED, which its clock's label says
            # (`report_cards`, `run-reports.md` § 2.8).
            "call_unit":  status.call_unit,
            # EVERY FIELD THE DECLARATION NAMES, by its own name -- which is
            # the status's attribute (`report_fields`, `stages.md` § 6.9).
            **{f.name: getattr(status, f.name) for f in _fields.FIELDS},
            "elapsed_s":  round(status.elapsed_s, 1),
        }
        # THE ENVELOPE IS THE DESTINATION'S, and one producer decides it --
        # the User-Agent among it, without which Discord's edge answers 403
        # before the request arrives (`run-reports.md` § 4.1b).
        body, head = webhook_request(
            {"url": url, "key": key, "headers": headers, "kind": kind},
            record, tuple(report) if report is not None else None, layout)
        req = urllib.request.Request(
            url, data=body, method="POST", headers=head)
        try:
            urllib.request.urlopen(req, timeout=NOTIFY_TIMEOUT_S).close()
        except Exception as exc:                       # noqa: BLE001
            who = f"channel {channel!r}" if channel else NOTIFY_URL_VAR
            _notify_say(f"{who}: {event} not delivered -- "
                        f"{delivery_failure(exc)}", log)
    _hook.__name__ = "webhook_notifier"
    return _hook


def run_identity(watched: Optional["WatchedRun"] = None) -> Dict[str, str]:
    """Who this report is ABOUT — gathered once, sent on every line.

    **A report with no identity is a result you cannot use**: with two jobs
    running, or two clusters, a line that does not say which calculation,
    on which machine, or when it was sent is indistinguishable.  These
    reports are a record of COMPUTATION, not of molbuilder's
    own health, so every line has to stand on its own -- somebody will
    parse this file a year from now with no session to ask.

    ``run`` is the rung's **stem** -- the label and the stage token, *"the
    stem of every file"* (`run-identity.md`) -- as `runfiles` composes it
    for the run the wrapper named (`project-layout.md` § 4.5).  The try's
    number is left out: the stem names the calculation, not the try.

    Everything is best-effort: an identity that cannot be gathered must
    never stop a run reporting.  A missing field is simply absent, which a
    reader can tell from a wrong one.
    """
    ident: Dict[str, str] = {}
    if watched is not None:
        try:
            ident["run"] = watched.stem[:200]
        except Exception:                               # noqa: BLE001
            pass                     # an identity never stops a report
    # THE SCHEDULER'S ID FOR THE JOB, in molbuilder's name -- set only
    # inside one (`scheduler/kinds.py`).
    v = os.environ.get("MB_JOB_ID")
    if v:
        ident["job"] = str(v)[:64]
    try:
        ident["host"] = os.uname().nodename[:128]
    except (AttributeError, OSError):
        pass
    return ident


def _install_env_notifiers(log: Optional[Path] = None,
                           ident: Optional[Dict[str, str]] = None,
                           channels: Optional[Sequence[str]] = None,
                           report: Optional[Sequence[str]] = None,
                           layout: "Optional[_cards.Layout]" = None) -> None:
    """Register the user's notifiers, if they configured any.  ``layout``
    is the card's layout, read once at the monitor's start
    (`run-reports.md` § 4.1c), handed to every notifier.

    ``MB_NOTIFY_URL`` wins when set -- an explicit environment override is
    how you test a destination once without editing a file.  It names no
    channel and needs none: one URL, used directly, and ``channels`` is not
    consulted.  Pair it with ``MB_NOTIFY_KEY`` for a molbuilder listener; a
    third party that keeps its credential in the URL (Slack, Discord) needs
    no key.  Otherwise the standing configuration at
    :func:`default_notify_path` is used.  Neither present means no notifier
    at all.

    ``channels`` is the description's own selection, straight from
    :class:`NotifyPolicy`; :func:`channels_for` owns what its three states
    mean.  **One notifier per chosen channel**, so a run can reach a Slack
    and a listener at once.

    **Registers once per process.**  :data:`_NOTIFIERS` is module state and
    ``run_monitor`` calls this every time, so without the guard a second
    call in one process adds a second copy of every webhook and every
    event is POSTed twice.  The shipped monitor runs one job per
    process and would never have shown it; anything embedding this would.
    """
    if any(getattr(fn, "__name__", "") == "webhook_notifier"
           for fn in _NOTIFIERS):
        return
    url = os.environ.get(NOTIFY_URL_VAR)
    if url:
        # THE SAME RULE AS A CHANNEL'S ADDRESS (:func:`address_refusal`),
        # said by its fault and never quoted.
        why = address_refusal(url)
        if why:
            _notify_say(f"{NOTIFY_URL_VAR}: {why}; nothing is sent", log)
            return
        # ``MB_NOTIFY_KEY`` rides with it, because without a key the
        # override could not reach OUR OWN listener at all: an unsigned
        # report is refused there with a 404 -- said in the log as a
        # failed delivery, never the address.
        register_notifier(make_webhook_notifier(
            url, key=os.environ.get(NOTIFY_KEY_VAR) or None, ident=ident,
            report=report, layout=layout, log=log))
        return
    if channels is not None and len(channels) == 0:
        # NONE ASKED FOR (`notify.channels = []`, `run-reports.md` § 3.0):
        # reports off for this calculation -- nothing to look up, nothing
        # to say about a machine with no channels file.
        return
    chosen, missing = channels_for(channels, load_channels(log=log))
    if missing:
        # NAMED IN THE DESCRIPTION, ABSENT HERE -- the travelling case, and
        # the run is not wrong, so this cannot be an error.  It must not be
        # silent either: a channel that resolves to nothing sends nothing,
        # which is indistinguishable from a channel that is working.
        _notify_say(f"this calculation asks for {', '.join(sorted(missing))}"
                    f" -- not set up on this machine, so nothing is sent "
                    f"there", log)
    for name in sorted(chosen):
        dest = chosen[name]
        register_notifier(make_webhook_notifier(
            dest["url"], key=dest.get("key"), headers=dest["headers"],
            ident=ident, kind=dest.get("kind"), report=report,
            layout=layout, channel=name, log=log))


# --------------------------------------------------------------------- #
#  The monitor loop                                                     #
# --------------------------------------------------------------------- #


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True   # not watching a pid
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, not ours
    except OSError:
        return True


def _append(log_path: Path, line: str) -> None:
    try:
        with Path(log_path).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _progressed(curr: JobStatus, prev: JobStatus) -> bool:
    """Did real work advance between two ticks?  True iff the SCF row count,
    the row's own iteration, the phase or the step moved -- or TBtrans's
    loop did: a progress line, a pass begun or completed -- or TranSIESTA's
    set-up did, until its first SCF row."""
    if (curr.n_iters or 0) > (prev.n_iters or 0):
        return True
    if (curr.phase, curr.cycle) != (prev.phase, prev.cycle):
        return curr.cycle is not None
    if curr.tbtrans != prev.tbtrans:
        return curr.tbtrans is not None
    if curr.part != prev.part:
        # A PART OF THE OUTPUT BEGAN -- its opening line read, the SCF's
        # last row unchanged (`run-reports.md` § 2.8).
        return curr.part is not None
    if curr.cycle is None and curr.doing != prev.doing:
        # TRANSIESTA'S SET-UP MOVED, before the first SCF row: a lead's
        # cell checked, its surface Green's function begun, sized, written
        # or found, the charge distribution (`run-reports.md` § 2.2).
        return curr.doing is not None
    return (curr.geom_step or 0) > (prev.geom_step or 0)


#: The variable a job's header sets once, when the job starts, for
#: everything it runs (`runwrap.render_sbatch`): the epoch second each run's
#: monitor counts the job's wall down from (`run-reports.md` § 2.1b) -- by
#: its one name (`runtime_info.JOB_STARTED_VAR`), imported above.


def run_monitor(watched: "WatchedRun", *,
                interval: float = 10.0,
                watch_pid: int = 0,
                start_epoch: Optional[float] = None,
                max_ticks: Optional[int] = None,
                util: bool = False,
                cores: Optional[int] = None,
                gpu: bool = False,
                util_change_frac: float = 0.10,
                util_keepalive_s: float = 300.0,
                sampler: Optional[Callable[[], "UtilSample"]] = None,
                notify: "NotifyPolicy" = NotifyPolicy(),
                job_started: Optional[float] = None,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.time,
                member: bool = False) -> JobStatus:
    """Periodically read + log + notify until the watched job ends.

    ``member``: this run is one call of a run of points, which reports as
    one run (`run-reports.md` § 2.4; `engines/vibration.md` § 5.11 § 7a) --
    it sends no `start` and no `finish` of its own, those are the run's
    Python steps', and its N-hour clock counts from the last event the
    run's notification record holds (``watched.launched_in``).  Every
    monitor writes each event it sends into its run's record.

    ``watched`` names the run; its log and its utilisation CSV (with
    ``util``) are its own files, named through `runfiles`.  Every reading is
    the framework's (:class:`WatchedRun`, `run-reports.md` § 2.3).
    ``cores`` is what a run started directly was launched on and ``gpu``
    whether it uses a GPU -- the wrapper's to say (:func:`_basis`,
    :func:`_make_default_sampler`).

    Returns the final :class:`JobStatus`.  ``max_ticks`` bounds the loop
    (tests pass a small value); ``sleep``/``clock`` are injectable for
    deterministic testing.

    ``job_started`` is when the job began -- its header's
    :data:`JOB_STARTED_VAR` -- from which a job sent with a wall counts
    down (`run-reports.md` § 2.1b): for every run in it, a walk's later
    member and a warm retry too.

    **How often it LOOKS and how often it TELLS you are different numbers.**
    It wakes every ``interval`` seconds and writes a ``[STATUS]`` line
    whenever the job advanced -- that is the record, and it stays dense.
    Notifying is separate and rare, set by ``notify``
    (`execution/run-reports.md` § 2).

    Wakes every ``interval`` seconds but is QUIET when nothing changed: a
    ``[STATUS]`` line is written only when the job advanced (SCF iteration,
    geometry move, or TBtrans's loop) or its energy/state
    changed.  **It judges no stall**
    (`run-reports.md` § 2): a step can take hours, and nothing in the output
    tells a slow one from a stuck one.
    """
    log = watched.path(".monitor.log")
    util_path: Optional[Path] = watched.path(".util.csv") if util else None
    start = clock() if start_epoch is None else start_epoch

    st0 = watched.read(start, clock())
    # FIRST line, before [MONITOR] start: the machine is known now, and a
    # run killed with its allocation still says where it died -- writing it
    # in the terminal block (as [UTIL-BASIS] is) would lose exactly the
    # runs whose machine most needs explaining (`scheduler.md` R12).  One
    # clock() for both lines: they describe the same moment, and the fake
    # clocks tests inject are counted.
    t0 = _iso(clock())
    _append(log, f"[{t0}] [MACHINE] {machine_line()}")
    _append(log, f"[{t0}] [MONITOR] start "
                 f"(interval={interval:.0f}s watch_pid={watch_pid}) "
                 f"{st0.as_text()}")
    # THE CARD'S LAYOUT, READ ONCE, HERE (`run-reports.md` § 4.1c): this
    # machine's copy or the shipped file.  One that does not read is
    # refused with its reason in this log, and the monitor stops here --
    # it is the person's own file, and nothing is guessed at.
    try:
        layout = _cards.load()
    except _cards.LayoutError as exc:
        _append(log, f"[{t0}] [MONITOR] the card's layout is refused -- {exc}")
        raise
    # The channels AFTER the first two lines: what they say about themselves
    # (a channel not set up here, a malformed file) follows the machine.
    _install_env_notifiers(log, run_identity(watched), notify.channels,
                           notify.report, layout)

    def _said(st: JobStatus, event: str, now: float) -> None:
        # ONE EVENT SENT, and written into the run's notification record --
        # never breaking the monitor over the record.  EVERY MESSAGE
        # RESTARTS THE N-HOUR CLOCK (`run-reports.md` § 2).
        nonlocal last_notify
        _fire(st, event)
        last_notify = now
        try:
            watched.note(event, now)
        except Exception as exc:                       # noqa: BLE001
            _append(log, f"[{_iso(now)}] [NOTIFY] not recorded: {exc}")

    if not member:
        _said(st0, "start", start)
    # Policy state (`run-reports.md` § 2).  ``last_notify`` starts at the
    # job's start, so
    # the first periodic message lands one full period in -- not immediately,
    # which would make "every 6 hours" mean "now, then every 6 hours".  A
    # member's counts from the last event sent for the RUN, its record.
    last_notify = start
    warned_before = False
    if member:
        try:
            sent = watched.notified()
            if sent:
                last_notify = max(float(e.get("at") or start) for e in sent)
            # THE WALL WARNED OF ONCE FOR THE RUN (§ 5.11 § 7a): a
            # `wall_near` its record holds is not sent again.
            warned_before = any(e.get("event") == "wall_near" for e in sent)
        except Exception as exc:                       # noqa: BLE001
            _append(log, f"[{t0}] [NOTIFY] the run's record not read: {exc}")
    notify_period_s = max(0.0, notify.every_hours) * 3600.0

    # THE TIME THE JOB HAS LEFT (`run-reports.md` § 2.1b): its wall, from
    # the run's launch record, read on each wake until it is there -- the
    # sender writes it as the scheduler answers, and a job can start first
    # -- counted from the job's start.  Warned of once, when it falls below
    # the longest step the run has taken.
    wall: Optional[float] = None
    wall_settled = False
    step_began = start
    longest_s: Optional[float] = None
    steps_timed = 0
    step_times: List[float] = []
    wall_warned = warned_before

    def _count_down(st: JobStatus, now: float) -> None:
        nonlocal wall, wall_settled
        if not wall_settled:
            try:
                record = watched.launch()
                if record is not None:
                    wall_settled = True
                    wall = _rr.sent_wall_s(record)
            except _rr.LaunchRecordError as exc:
                wall_settled = True
                _append(log, f"[{_iso(now)}] [WALL] not counted: {exc}")
            else:
                if wall is not None:
                    if job_started is None:
                        _append(log, f"[{_iso(now)}] [WALL] not counted: "
                                     f"the job's header states no start "
                                     f"({JOB_STARTED_VAR}) -- it was "
                                     f"written before it did")
                        wall = None
        if wall is not None:
            st.wall_s = wall
            st.wall_left_s = wall - (now - job_started)

    # --- utilization sampling setup (same loop, separate change-gated
    # output file; `run-reports.md` § 2.1).  ``sampler`` is injectable for
    # tests. ---
    # WHAT THE JOB HOLDS, once (§ 2.1a): its cgroup and allocation under a
    # scheduler; started directly, the watched process tree and the cores
    # it was launched on.
    basis = _basis(watch_pid, cores)
    _sample = (sampler if sampler is not None
               else _make_default_sampler(clock, basis, gpu))
    util_accum = UtilAccum()
    util_prev: Optional[UtilSample] = None
    util_ngpu = 0
    util_last_log = start
    if util_path is not None:
        first = _sample()
        util_ngpu = len(first.gpus)
        try:
            Path(util_path).write_text(
                _util_csv_header(util_ngpu) + "\n", encoding="utf-8")
        except OSError:
            util_path = None
        if util_path is not None:
            _append(util_path, _util_csv_row(first, util_ngpu))
            util_accum.add(first)
            util_prev = first

    def _util_tick(now: float, *, force: bool = False,
                   count: bool = True) -> None:
        nonlocal util_prev, util_last_log
        if util_path is None:
            return
        s = _sample()
        if count:
            util_accum.add(s)
        if (force or util_prev is None
                or s.changed_from(util_prev, util_change_frac)
                or now - util_last_log >= util_keepalive_s):
            _append(util_path, _util_csv_row(s, util_ngpu))
            util_prev = s
            util_last_log = now

    _count_down(st0, start)
    prev = st0
    ticks = 0
    while True:
        sleep(interval)
        ticks += 1
        now = clock()
        # STOPPED IS OVER -- for this process.  The wrapper stops it when the
        # job ends (its EXIT trap) and when it retries the run in place,
        # both while the watched pid is still alive.  Asked BEFORE sampling:
        # a sample taken after the stop is not the run's.
        alive = _pid_alive(watch_pid) and _STOPPED_BY is None
        if alive:
            _util_tick(now)
        st = watched.read(start, now)
        _count_down(st, now)
        # A STEP'S LENGTH, from one step finishing to the next (§ 2.2), and
        # the pace of the run's steps where its index states no other unit
        # of work (`run-reports.md` § 2.8).
        if (st.steps_done is not None
                and st.steps_done > (prev.steps_done or 0)):
            took = now - step_began
            step_times.append(took)
            longest_s = took if longest_s is None else max(longest_s, took)
            step_began, steps_timed = now, st.steps_done
        if st.unit_s is None and step_times:
            st.unit_words = st.step_kind or "step"
            st.units_done = st.steps_done
            st.unit_s = sum(step_times) / len(step_times)

        if not alive:
            # WHY IT STOPPED comes first: the closing record is evidence
            # `run_status` reads -- a forced stop leaves no other word
            # (`run-reports.md` § 2.4) -- and HOW IT ENDED, asked next, is
            # the Results tab's own reading of the run's files.
            retry = _STOPPED_BY == _STOP_RETRY
            if retry:
                # NOT AN ENDING: the wrapper re-execs itself in this pid for
                # the run's next try, which starts its own monitor
                # (`run-reports.md` § 2 -- "it ended" is the watched pid
                # going, and across an exec it does not go).
                _append(log, f"[{_iso(now)}] [MONITOR] stopped: the wrapper "
                             f"is retrying this run in place; its next "
                             f"try starts its own monitor")
            else:
                _append(log, f"[{_iso(now)}] {MONITOR_ENDED} "
                             + (f"(stopped by {_STOPPED_BY}); "
                                if _STOPPED_BY else
                                f"(watched pid {watch_pid} gone); ")
                             + "final notify + exit")
            st = watched.conclude(st)
            _append(log, f"[{_iso(now)}] [STATUS] {st.as_text()}")
            if util_path is not None:
                _append(log, f"[{_iso(now)}] [UTIL-SUMMARY] "
                             f"{util_accum.summary()}")
                _append(log, f"[{_iso(now)}] [UTIL-BASIS] "
                             f"{measurement_provenance(basis)}")
            if not retry and not member:
                _said(st, "finish", now)
            # The series' end, LAST: a GPU sample can take seconds and the
            # wrapper waits for this process only so long, so the closing
            # lines and the `finish` message go out before it.
            _util_tick(now, force=True, count=False)
            return st

        if (_progressed(st, prev) or st.energy != prev.energy
                or st.state != prev.state):
            _append(log, f"[{_iso(now)}] [STATUS] {st.as_text()}")

        # THE ONE WARNING (§ 2.1b): a run of points whose units left need
        # more than the wall leaves at the measured pace (§ 2.8, plan § 1af);
        # else the step underway that may not finish before the wall.
        if (not wall_warned and st.wall_left_s is not None
                and st.left_at_pace_s is not None and st.units_left
                and st.wall_left_s < st.left_at_pace_s):
            wall_warned = True
            _append(log, f"[{_iso(now)}] [WALL] "
                         f"{_fields.span_words(max(0.0, st.wall_left_s))} "
                         f"left; {st.units_left} {st.unit_words}s left at "
                         f"{_fields.span_words(st.unit_s)} each "
                         f"({st.units_done} done) need about "
                         f"{_fields.span_words(st.left_at_pace_s)} -- the "
                         f"run may not finish before the wall")
            _said(st, "wall_near", now)
        elif (not wall_warned and st.wall_left_s is not None
                and st.left_at_pace_s is None
                and longest_s is not None and st.wall_left_s < longest_s):
            wall_warned = True
            steps = ("the one step so far" if steps_timed == 1 else
                     f"the longest of {steps_timed} steps so far")
            _append(log, f"[{_iso(now)}] [WALL] "
                         f"{_fields.span_words(max(0.0, st.wall_left_s))} "
                         f"left; {steps} took "
                         f"{_fields.span_words(longest_s)} -- the step "
                         f"underway may not finish before the wall")
            _said(st, "wall_near", now)

        # --- the two settable triggers (`run-reports.md` § 2) ------------
        #
        # A STEP FINISHING means its SCF reached its criterion
        # (`run-reports.md` § 2.2): SIESTA begins step N once N are done
        # (`Begin <CG|Broyden|FIRE> opt. move = N`), and a PySCF block is
        # written when its step ends -- the parser counts them
        # (`JobStatus.steps_done`).  Read that way rather than by scanning
        # for a convergence phrase.  ONLY A STEP THAT MOVES THE GEOMETRY is
        # a message, as the reading pass states it (`JobStatus.step_moves`):
        # a force-constant displacement's count and pace are the periodic
        # report's.  A single point states no step, so nothing fires and
        # the finish message is the whole report.
        if (notify.on_scf and st.step_moves and st.steps_done is not None
                and st.steps_done > (prev.steps_done or 0)):
            _said(st, "scf_converged", now)
        elif notify_period_s > 0 and now - last_notify >= notify_period_s:
            # `elif`: a step and a period landing on the same wake is one
            # thing worth saying, not two.  The step is the more informative
            # of them, so it wins and resets the clock.  A MEMBER asks its
            # run's record first: the run's Python steps speak too -- a
            # milestone its call's record step sent while this monitor
            # still watched (`run-reports.md` § 2.9) -- and the N hours
            # count from the run's last word.
            if member:
                try:
                    sent = watched.notified()
                    if sent:
                        last_notify = max(last_notify, max(
                            float(e.get("at") or 0.0) for e in sent))
                except Exception as exc:               # noqa: BLE001
                    _append(log, f"[{_iso(now)}] [NOTIFY] the run's record "
                                 f"not read: {exc}")
            if now - last_notify >= notify_period_s:
                _said(st, "periodic", now)

        prev = st
        if max_ticks is not None and ticks >= max_ticks:
            return st


def measurement_provenance(basis: Optional[_Basis] = None) -> str:
    """One line naming what every percentage in this run is a fraction OF.

    **A percentage whose denominator is invisible is how the Au-BDT-Au sweep
    went wrong**: 48 ranks on a 128-core node capped the node-wide reading at
    37.5%, and 32.2% read as idleness rather than as a job at 86% of its own
    allocation.  With two cgroup generations in play a number that does not
    say where it came from cannot be checked at all, so the rung that
    answered is part of the measurement, not a debug aid.
    """
    b = basis if basis is not None else _basis()
    # A PROCESS TREE IS ITS OWN SOURCE.  One read at the end that met a
    # process exiting -- the wrapper's `sleep`s while it waits for this
    # monitor -- says nothing about what every sample was read from; a PySCF
    # water run closed on "cpu time [unavailable]" (2026-09-26).
    if b.tree is not None:
        cpu_from = "process tree"
    else:
        cpu = b.cpu_ns()
        cpu_from = cpu[1] if cpu else "unavailable"
    mem = b.mem_gb()
    bits = [f"cpu% of {b.cores} core(s) [{b.cores_from}]",
            f"cpu time [{cpu_from}]",
            f"mem [{mem[1] if mem else 'unavailable'}]"]
    # The kernel's peak and limit are a CGROUP's: a process tree has neither.
    peak = _read_mem_peak_gb() if b.tree is None else None
    if peak is not None:
        bits.append(f"peak {peak:g} GB (kernel counter)")
    lim = _read_mem_limit_gb() if b.tree is None else None
    bits.append(f"limit {lim:g} GB" if lim is not None
                else "limit not stated")
    return "; ".join(bits)


#: The shortest interval a CPU rate is taken over.  A process tree's counters
#: tick in 1/``SC_CLK_TCK`` s (10 ms), and the monitor's first sample follows
#: the sampler's baseline by microseconds: one tick over those read as
#: 262144% on a real 2-rank run (2026-09-26).  Shorter than this, a sample
#: states no rate and the baseline waits for the next.
_RATE_MIN_S = 1.0


def _make_default_sampler(clock: Callable[[], float],
                          basis: Optional[_Basis] = None,
                          gpu: bool = False) -> Callable[[], "UtilSample"]:
    """A stateful sampler closure.

    cpu% is ``delta cpu-time / (delta wall-time x cores held)``, so it holds
    the previous cumulative reading and the wall clock that went with it.
    **The denominator is what the job holds** (:func:`_basis`), resolved once
    up front: it cannot change during a run, and re-reading it per tick
    would let a percentage silently change meaning mid-series.

    **GPUs are sampled only for a run that uses one** (``gpu``, the wrapper's
    to say): a CPU run on a GPU node holds no GPU, and what the node's GPUs
    are doing is somebody else's -- sampled, a CPU-only relaxation would
    close on *"GPU starved"*.  Presence is probed
    once (no per-tick ``nvidia-smi -L``).
    """
    b = basis if basis is not None else _basis()
    cores = b.cores
    first = b.cpu_ns()
    state = {"ns": first[0] if first else None, "t": clock()}
    gpu_on = gpu and _gpu_present()

    def _s() -> "UtilSample":
        now = clock()
        cur = b.cpu_ns()
        cpu_pct = None
        d_t = now - state["t"]
        if d_t >= _RATE_MIN_S:
            if cur is not None and state["ns"] is not None:
                d_ns = cur[0] - state["ns"]
                if cores > 0 and d_ns >= 0:
                    cpu_pct = round(100.0 * d_ns / (d_t * 1e9 * cores), 1)
            # THE TWO BASES MOVE TOGETHER: a missed reading leaves both, so
            # the next rate spans two intervals of CPU over two of wall time.
            if cur is not None:
                state["ns"] = cur[0]
                state["t"] = now
        mem = b.mem_gb()
        return UtilSample(epoch=now, cpu_pct=cpu_pct,
                          mem_gb=mem[0] if mem is not None else None,
                          gpus=_sample_gpus() if gpu_on else [])
    return _s


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch))


# --------------------------------------------------------------------- #
#  Default log notifier (the PoC stub)                                  #
# --------------------------------------------------------------------- #


def make_log_notifier(log: Path) -> Notifier:
    """The PoC notifier: records into the monitor log what *would* be
    pushed.  Swap/extend with a real channel via :func:`register_notifier`."""
    def _hook(status: JobStatus, event: str) -> None:
        # Every event a notifier can see.  `start` is here because the log
        # is also the record of what the monitor did; a real destination
        # gets the same set, and it is the POLICY upstream -- not this
        # list -- that decides which of them ever occur.
        _append(Path(log),
                f"[{_iso(time.time())}] [NOTIFY] (stub) {event}: "
                f"{status.as_text()}")
    _hook.__name__ = "log_notifier"
    return _hook


# --------------------------------------------------------------------- #
#  Standalone entry (stdlib only -- runs WITHOUT the molbuilder package) #
# --------------------------------------------------------------------- #
#
# CRITICAL: shipped, this module needs ONLY the stdlib (os/re/signal/time/
# urllib/dataclasses/pathlib/typing) and the framework modules that travel
# with it (`runwrap.MONITOR_COMPANIONS`) -- no molbuilder package, no numpy.
# That is what lets the run-wrapper SHIP this file -- as ``mb_monitor.py``
# inside ``mb_monitor.pyz`` -- next to the job and run it with the JOB's own
# python, from the working directory, with no install and no repo on PATH,
# whatever else that env's packages can or cannot import.


#: Why this process was stopped, or None: ``"SIGTERM"`` -- the job ended (the
#: wrapper's EXIT trap, or the scheduler's walltime or cancel) -- or
#: :data:`_STOP_RETRY`, SIGUSR1, the wrapper retrying the run in place.
#: Set by PLAIN ASSIGNMENT in the handler: a handler that took a lock could
#: meet the loop holding it and hang the monitor for good.  Only `main`
#: installs the handlers: a library call of `run_monitor` gets no
#: process-wide signal behaviour.  SIGUSR1 is chosen because nothing else
#: sends it to a job's processes; wrong when a scheduler is set to (Slurm's
#: ``--signal=USR1``), which would close the monitor quietly.
_STOPPED_BY: Optional[str] = None
_STOP_RETRY = "the wrapper's retry"


def _on_stop_signal(signum, _frame) -> None:
    global _STOPPED_BY
    _STOPPED_BY = (_STOP_RETRY if signum == getattr(signal, "SIGUSR1", None)
                   else "SIGTERM")


def _sleep_until_stopped(seconds: float) -> None:
    """The loop's sleep in `main`: short slices, so a stop is acted on within
    a fifth of a second instead of at the end of the interval."""
    end = time.monotonic() + seconds
    while _STOPPED_BY is None:
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.2, left))


def main(argv=None) -> int:
    """The entry of the SHIPPED bundle, ``mb_monitor.pyz``
    (`runwrap.MONITOR_BUNDLE`): argparse for the monitor, and ``ending ...``
    handed to `_run_ending`'s door, which travels in the same file -- the
    wrapper asks how a run ended with ``mb_monitor.pyz ending OUTPUT ...``.

    ONE command line: ``molbuilder monitor`` hands its arguments here, so
    the monitor a person starts from the package is the one the job runs.
    Zero third-party deps, so it runs in any python.  Self-lowers priority
    via ``os.nice`` and installs the default log notifier.
    """
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["ending"]:
        return _ending.main(args[1:])
    if args[:1] == [POINT_VERB]:
        # A TRANSPORT POINT'S REPORT STEPS: its start and end into the run's
        # index, and the run's events (`engines/transport.md` § 2a.11).
        return _point_main(args[1:])
    if args[:1] == ["notify"]:
        # A RUN OF POINTS REPORTS AS ONE RUN: its start, milestones, failed
        # parts and finish, sent by the run's own python through this verb
        # (`engines/vibration.md` § 5.11 § 7a; `run-reports.md` § 2.4,
        # § 2.9).
        return _notify_main(args[1:])
    import argparse
    p = argparse.ArgumentParser(
        prog="mb_monitor",
        description="molbuilder background job-monitor + notifier hooks "
                    "(reads the run through the framework's readers, which "
                    "travel in this same file; execution/run-reports.md "
                    "§ 2.3).  `ending OUTPUT ...` asks how a run ended.")
    # WHICH RUN, never a path: every file is named through `runfiles`
    # (`run-reports.md` § 2.3), from the identity the wrapper was rendered for.
    p.add_argument("--label", required=True,
                   help="the run's label -- the stem every file begins with")
    p.add_argument("--stage", required=True,
                   help="the stage token (e.g. 01_coarse)")
    p.add_argument("--dir", default=".", dest="directory",
                   help="the directory the run is in (default: here, where "
                        "the wrapper starts it)")
    p.add_argument("--run", type=int, required=True, dest="try_n",
                   help="the try's number (-runN): 0 from launch, the "
                        "next from a warm retry")
    p.add_argument("--interval", type=float, default=10.0,
                   help="seconds between wakes (default 10; this is the "
                        "utilization sample rate -- status lines stay "
                        "change-gated, so a fast rate does not spam)")
    p.add_argument("--util", action="store_true", dest="util",
                   help="append change-gated cpu%%/mem/GPU-sm%%/VRAM samples "
                        "to the run's .util.csv")
    # WHAT THE JOB HOLDS, from the wrapper that launched it (§ 2.1a): the
    # cores a run started directly was launched on -- under a scheduler the
    # allocation answers instead -- and whether it uses a GPU at all.
    p.add_argument("--cores", type=int, default=None, dest="cores",
                   help="the cores the run was launched on (ranks x "
                        "threads); the denominator of cpu%% for a run "
                        "started directly")
    p.add_argument("--gpu", action="store_true", dest="gpu",
                   help="the run uses a GPU: sample and judge it")
    p.add_argument("--util-keepalive", type=float, default=300.0,
                   dest="util_keepalive_s",
                   help="even with no >10%% change, write a util row at "
                        "least this often (seconds, default 300) so the "
                        "plotted series has anchor points")
    p.add_argument("--watch-pid", type=int, default=0, dest="watch_pid",
                   help="stop when this PID disappears; 0 = until done")
    p.add_argument("--nice", type=int, default=19, dest="nice_level",
                   help="self-lower OS priority by this much (default 19)")
    # WHEN to tell someone -- the calculation's own policy, carried here from
    # `task.json`'s `notify` block by the wrapper.  Neither flag says WHERE:
    # the destination is the user's file on this machine (NOTIFY_FILENAME,
    # `run-reports.md` § 3).
    p.add_argument("--notify-on-scf", action="store_true",
                   dest="notify_on_scf",
                   help="notify when a step finishes (its SCF reached its "
                        "criterion)")
    p.add_argument("--notify-every-hours", type=float, default=0.0,
                   dest="notify_every_hours",
                   help="notify every N hours; 0 = never (default)")
    # WHICH channels -- names only, never an address and never a key.  `*`
    # is every channel this machine has; a description that names none
    # renders no flag, or `--notify-channels ""`, and both mean none at
    # all (`run-reports.md` 3.0, `_channels_from_flag`).
    p.add_argument("--notify-channels", type=str, default=None,
                   dest="notify_channels",
                   help="comma-separated channel names to report to; "
                        "'*' for every channel; omit it, or pass '', "
                        "for none")
    # WHAT each report carries.  Same two-state shape as the channels above:
    # absent is every field this monitor can determine, `""` is the summary
    # line alone (`stages.md` 6.9).  The calculation's own name is always
    # sent and is not one of these.
    p.add_argument("--notify-report", type=str, default=None,
                   dest="notify_report",
                   help="comma-separated report fields ("
                        + ", ".join(_fields.NAMES)
                        + "); omit for all of them, pass '' for none")
    # A MEMBER OF A RUN OF POINTS (`run-reports.md` § 2.4): the run's
    # folder holds its launch record and its notification record; the
    # run's start and finish are its Python steps'.
    p.add_argument("--member", action="store_true", dest="member",
                   help="this run is one call of a run of points: no start "
                        "or finish of its own, its clock the run's")
    p.add_argument("--launched-in", default=None, dest="launched_in",
                   help="the run's folder, from here -- where its launch "
                        "record and notification record lie")
    a = p.parse_args(args)
    watched = WatchedRun(names=_rf.RunNames(a.label, a.stage),
                         run=a.try_n, directory=Path(a.directory),
                         launched_in=(Path(a.directory) / a.launched_in
                                      if a.launched_in else None))
    signal.signal(signal.SIGTERM, _on_stop_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_stop_signal)
    try:
        os.nice(max(0, a.nice_level))
    except (OSError, AttributeError):
        pass
    register_notifier(make_log_notifier(watched.path(".monitor.log")))
    started = os.environ.get(JOB_STARTED_VAR, "")
    run_monitor(watched,
                job_started=float(started) if started.isdigit() else None,
                interval=a.interval, watch_pid=a.watch_pid,
                sleep=_sleep_until_stopped,
                util=a.util, cores=a.cores, gpu=a.gpu,
                util_keepalive_s=a.util_keepalive_s,
                notify=NotifyPolicy(
                    on_scf=a.notify_on_scf,
                    every_hours=a.notify_every_hours,
                    channels=_channels_from_flag(a.notify_channels),
                    report=_report_from_flag(a.notify_report)),
                member=a.member)
    return 0


#: THE POLICY'S CHANNELS AND REPORT, in the environment of a run of
#: points' calls (`runwrap._monitor_block` exports them from the monitor's
#: own flags), so the run's own python can send the run's events -- its
#: start, milestones, failed parts and finish -- through
#: :func:`_notify_main` with the policy the description stated.
NOTIFY_CHANNELS_VAR = "MB_NOTIFY_CHANNELS"
NOTIFY_REPORT_VAR = "MB_NOTIFY_REPORT"


#: A TRANSPORT POINT'S TWO REPORT STEPS (`engines/transport.md` § 2a.11):
#: the verb and its two words, as the point's run script renders them
#: (`runwrap._report_block`).
POINT_VERB, POINT_STARTED, POINT_ENDED = "point", "started", "ended"


def _point_say(level: str, message: str) -> None:
    """One line in the point's session log, in the wrapper's own format."""
    import sys
    sys.stderr.write(LOG_LINE % (time.strftime(LOG_CLOCK), level, message)
                     + "\n")
    sys.stderr.flush()


def _point_main(argv) -> int:
    """``mb_monitor.pyz point started|ended --label L --stage S --run N
    --launched-in UP [--exit RC]`` -- ONE OF A TRANSPORT POINT'S TWO REPORT
    STEPS, run in the point's folder by its run script, ``N`` the point's
    try and ``UP`` the way up to its run's folder (`engines/transport.md`
    § 2a.11; `run-reports.md` § 2.4, § 2.9).

    ``started``, before the engine: the point's start into the run's index
    (`points_index.stamp_start`; a try retried in place keeps the first's)
    and the run's `start`, sent once whichever point asks first.
    ``ended``, after it, success or failure: its end, its engine's exit
    and its verdict into the index (`points_index.stamp_end`, the verdict
    by the status door's own rule, `_run_ending.verdict_of`), then what the
    run's rule calls for -- with no point left to run, the `finish`: the
    run's state by the status door's rule over the index
    (`points_index.run_state`), the first point not done and how many, and
    the outcome of this point's round; else, none of this point's round
    left to run, the
    round's `milestone`; else, this point not done, a `part_failed`, once
    per reason (`points_index`).  **It never fails its point**: what it
    cannot do is said in the point's session log, and its exit is only
    logged by the run script."""
    import argparse
    p = argparse.ArgumentParser(prog="mb_monitor point")
    p.add_argument("verb", choices=(POINT_STARTED, POINT_ENDED))
    p.add_argument("--label", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--run", type=int, required=True, dest="try_n",
                   help="the point's try, as its run script was given it")
    p.add_argument("--launched-in", required=True, dest="launched_in",
                   help="the run's folder, from the point's")
    p.add_argument("--exit", type=int, default=None, dest="exit_code",
                   help="the engine's exit (point ended)")
    a = p.parse_args(argv)
    here = Path.cwd().resolve()
    run_dir = (here / a.launched_in).resolve()
    names = _rf.RunNames(a.label, a.stage)
    rel = Path(os.path.relpath(here, run_dir)).as_posix()
    path = _pi.index_path(run_dir, names)
    try:
        index = _pi.read_index(path)
        if a.verb == POINT_STARTED:
            _pi.stamp_start(index, rel, _rr.stamp())
        else:
            if a.exit_code is None:
                raise ValueError("point ended is told the engine's exit "
                                 "(--exit)")
            out = here / names.name(".out", run=a.try_n)
            verdict = None
            if out.is_file():
                try:
                    verdict = _ending.verdict_of([_ending.ending_of(out)])
                except Exception as exc:               # noqa: BLE001
                    _point_say("WARN", f"report step: {out.name}'s verdict "
                                       f"not read -- {exc}")
            _pi.stamp_end(index, rel, _rr.stamp(), a.exit_code, verdict)
        _pi.write_index(path, index)
    except (OSError, ValueError) as exc:
        _point_say("WARN", f"report step: the run's index not stamped -- "
                           f"{exc}")
        return 1
    said = ["--label", a.label, "--stage", a.stage, "--dir", str(run_dir)]
    if a.verb == POINT_STARTED:
        return _notify_main(["start"] + said)
    rows = _pi.rows_of(index, run_dir)
    mine = next(r for r in rows if r["rel"] == rel)
    left = _pi.still_to_run(rows)
    round_left = [r for r in left if r["bias_v"] == mine["bias_v"]]
    outcome = None
    if not round_left:
        # THE ROUND'S OUTCOME, read from its points' own files; words that
        # cannot be composed are logged, never the reason a run is silent.
        try:
            outcome = _pi.round_words(index, rows, mine["bias_v"], run_dir,
                                      names)
        except Exception as exc:                       # noqa: BLE001
            _point_say("WARN", f"report step: the round at "
                               f"{mine['bias_v']:g} V not put in words -- "
                               f"{type(exc).__name__}: {exc}")
    if not left:
        # THE WALK IS OVER: a point it never ran is said failed, as the
        # status door says it (`points_index.close`), in the index the
        # finish's count is read from.
        try:
            _pi.write_index(path, _pi.close(index, rows))
            rows = _pi.rows_of(index, run_dir)
        except (OSError, ValueError) as exc:
            _point_say("WARN", f"report step: the points never run not "
                               f"stamped -- {exc}")
        state, why = _pi.run_state(rows)
        detail = "; ".join(w for w in (why, outcome) if w) or None
        return _notify_main(["finish"] + said + ["--state", state]
                            + (["--detail", detail] if detail else []))
    if not round_left:
        return _notify_main(["milestone"] + said
                            + (["--detail", outcome] if outcome else []))
    if not mine["done"]:
        # SAID ONCE PER REASON (`run-reports.md` § 2.9): the `notify` verb
        # sends no second failure for a reason the run already reported.
        return _notify_main(["part_failed"] + said + [
            "--detail", _pi.not_done_words(mine),
            "--reason", mine["why"] or mine["state"]])
    return 0


def _notify_main(argv) -> int:
    """``mb_monitor.pyz notify start|milestone|part_failed|finish --label L
    --stage S --dir RUN [--state S --detail D --reason R]`` -- ONE event for
    a
    run of points, sent by its Python steps -- a vibration's record steps,
    a transport point's report steps (:func:`_point_main`) -- through the
    channels and card the monitor uses, with the run's identity, and
    written into the run's notification record (`engines/vibration.md`
    § 5.11 § 7a; `engines/transport.md` § 2a.11; `run-reports.md` § 2.4,
    § 2.9).  Every event says the run's count -- the start's, its points
    before any has ended -- and, once a point has started, its clock, read
    from its index by its kind's reader (:data:`RUN_INDEXES`), which the
    step sending it has just written.  The channels and
    report fields are the policy's, from the environment the call's run
    script set (:data:`NOTIFY_CHANNELS_VAR`, :data:`NOTIFY_REPORT_VAR`);
    none set up, nothing is sent -- the record holds what was said all the
    same, whether or not a channel took it."""
    import argparse
    p = argparse.ArgumentParser(prog="mb_monitor notify")
    p.add_argument("event",
                   choices=("start", "milestone", "part_failed", "finish"))
    p.add_argument("--label", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--dir", default=".", dest="directory",
                   help="the run's folder")
    p.add_argument("--state", default="running",
                   help="the run's state: a finish's by the one rule over "
                        "the run's index; running until then")
    p.add_argument("--detail", default=None)
    p.add_argument("--reason", default=None,
                   help="a failed part's why: a part_failed whose reason "
                        "the run has already reported is not sent again")
    a = p.parse_args(argv)
    # THE RUN'S FOLDER HOLDS NO TRY OF ITS OWN -- its calls' or points'
    # folders do -- so its log is named for the first.
    watched = WatchedRun(names=_rf.RunNames(a.label, a.stage),
                         run=_rf.FIRST_TRY, directory=Path(a.directory))
    log = watched.path(".monitor.log")
    now = time.time()
    try:
        sent = watched.notified()
    except Exception as exc:                           # noqa: BLE001
        # A RECORD THAT DOES NOT READ silences nothing: the event is sent,
        # and the log says why it may be a second time.
        _append(log, f"[{_iso(now)}] [NOTIFY] the run's record not read: "
                     f"{exc}")
        sent = []
    # THE RUN STARTS ONCE: a start the record already holds is not sent
    # again -- whichever call's plan step asks (§ 5.11 § 7a).
    if a.event == "start" and any(e.get("event") == "start" for e in sent):
        return 0
    # A FAILED PART SPEAKS ONCE PER REASON (`run-reports.md` § 2.9): a run
    # whose calls fail alike -- its wall reached, a file missing -- says it
    # once, and the finish counts the calls not done; never as many
    # messages as calls.
    if a.event == "part_failed" and a.reason is not None and any(
            e.get("event") == "part_failed" and e.get("reason") == a.reason
            for e in sent):
        return 0
    register_notifier(make_log_notifier(log))
    try:
        layout = _cards.load()
    except _cards.LayoutError as exc:
        _append(log, f"[{_iso(now)}] [NOTIFY] the card's layout is refused "
                     f"-- {exc}")
        return 1
    _install_env_notifiers(
        log, run_identity(watched),
        _channels_from_flag(os.environ.get(NOTIFY_CHANNELS_VAR)),
        _report_from_flag(os.environ.get(NOTIFY_REPORT_VAR)), layout)
    st = JobStatus(state=a.state, detail=a.detail)
    # THE RUN'S COUNT AND CLOCK, from its index (§ 2.4, § 2.8), which
    # the step sending the event has just written -- the start's, its
    # points before any has ended (plan § 1af).
    try:
        found = _run_index_of(watched.directory, watched.names)
        if found is None:
            raise ValueError("the run's folder holds no index of its "
                             "points")
        reader, idx = found
        index = reader.read(idx)
        rows = reader.rows(index, watched.directory, watched.label)
        st.run_words = reader.summary(rows, index)
        clock = _run_clock(rows, now)
    except Exception as exc:                       # noqa: BLE001
        # THE MESSAGE GOES OUT WITHOUT THEM: an index that does not
        # read is said in the log, never the reason a run is silent.
        _append(log, f"[{_iso(now)}] [NOTIFY] the run's index not "
                     f"read: {exc}")
        clock = None
    if clock is not None:
        st.elapsed_s = clock
    _fire(st, a.event)
    try:
        watched.note(a.event, now, reason=a.reason)
    except Exception as exc:                           # noqa: BLE001
        _append(log, f"[{_iso(now)}] [NOTIFY] not recorded: {exc}")
    return 0


__all__ = [
    "JobStatus",
    "WatchedRun",
    "LIVE_READERS",
    "run_monitor",
    "register_notifier",
    "clear_notifiers",
    "make_webhook_notifier",
    "make_log_notifier",
    "load_channels",
    "channels_for",
    "NotifyPolicy",
    "webhook_request",
    "channel_kind",
    "NOTIFY_FILENAME",
    "default_notify_path",
    "is_route_segment",
    "NOTIFY_TIMEOUT_S",
    "Notifier",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
