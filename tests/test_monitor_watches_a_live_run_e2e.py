"""The monitor watching a real run (`execution/run-reports.md` § 2), and the
files it ships with reading one with molbuilder absent.

The run is made in this pass, on the road, once for every module that reads
it: a vibration's `relax` stage, an H2 with one atom held, run with the real
SIESTA (`support/real_runs.py`).
The monitor is then shown that run as it grew -- a copy of the run's own
folder, its output written up to a line and grown on, under its own names
(`process/testing.md` § 6) -- with the clock and the wakes the test's own, so
its policy is asked of a real output stream.  Where the run's lines fall, and
what it states at each, is read off the run's own output here: the SCF rows,
the moves SIESTA began, the ``Max`` force it printed, the criteria its
``redata:`` lines state.

The run is made, never saved (user, 2026-10-06: "when a test need siesta's
output why is it not part of a e2e test?").
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from molbuilder import monitor
from molbuilder.monitor import NotifyPolicy
from molbuilder.runfiles import RunNames

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

_STEM = "H2_01_relax"
_OUT = f"{_STEM}-run0.out"


@pytest.fixture(scope="module")
def run(real_h2_relaxed_for_vibration):
    """The relaxation's run folder, made on the road."""
    return real_h2_relaxed_for_vibration / "01_relax" / "run-0"


@pytest.fixture(autouse=True)
def _clean_notifiers():
    monitor.clear_notifiers()
    yield
    monitor.clear_notifiers()


def _fake_clock(values):
    """Successive ``values``, holding the last when drained."""
    it = iter(values)
    box = {"last": values[0] if values else 0.0}

    def _c():
        try:
            box["last"] = next(it)
        except StopIteration:
            pass
        return box["last"]
    return _c


class _Output:
    """The run's own output, and where its lines fall: the SCF rows, and the
    geometry moves SIESTA began (``Begin <kind> opt. move = N``)."""

    def __init__(self, run: Path):
        self.lines = (run / _OUT).read_text(
            errors="replace").splitlines(keepends=True)
        self.rows = [i for i, ln in enumerate(self.lines)
                     if ln.lstrip().startswith("scf:")]
        self.moves = [i for i, ln in enumerate(self.lines)
                      if re.match(r"\s*Begin \w+ opt\. move", ln)]

    def rows_of(self, k: int):
        """The SCF rows of move ``k``."""
        end = (self.moves[k + 1] if k + 1 < len(self.moves)
               else len(self.lines))
        return [r for r in self.rows if self.moves[k] < r < end]

    def through(self, line: int) -> int:
        """The ``upto`` that writes line ``line`` and everything before it."""
        return line + 1

    def end_of_move(self, k: int) -> int:
        return self.through(self.rows_of(k)[-1])

    def last_free_force(self, upto: int) -> float:
        """The largest force on a free atom, as SIESTA last printed it before
        ``upto`` -- with atoms held, its ``Max <v> constrained`` line: what
        a relaxation's tolerance is judged against (`run-reports.md` § 2)."""
        found = [float(m.group(1)) for ln in self.lines[:upto]
                 for m in [re.match(r"\s*Max\s+(\S+)\s+constrained", ln)]
                 if m]
        assert found, "no constrained Max force printed before that line"
        return found[-1]

    def criterion(self, name: str) -> float:
        """A criterion as the run's own ``redata:`` line states it --
        ``name`` a pattern for SIESTA's words before ``tolerance``."""
        for ln in self.lines:
            m = re.match(rf"\s*redata:\s*(?:{name})\s+tolerance\b[^=]*=\s*(\S+)",
                         ln)
            if m:
                return float(m.group(1))
        raise AssertionError(f"no redata line states the {name} tolerance")


def _replay(run: Path, tmp_path: Path, upto: int):
    """The run's folder with its output written to ``upto``, and a
    ``grow(to)`` that writes on, as SIESTA would.  What the monitor itself
    writes -- its log and its samples -- and the timing tee, which grows with
    the output, are left for the replay to write."""
    there = tmp_path / "run-0"
    shutil.copytree(run, there, ignore=shutil.ignore_patterns(
        _OUT, "*.monitor.log", "*.util.csv", "*.scf-timing.log"))
    lines = _Output(run).lines
    out = there / _OUT
    out.write_text("".join(lines[:upto]))
    at = {"n": upto}

    def grow(to: int) -> None:
        with out.open("a") as fh:
            fh.write("".join(lines[at["n"]:to]))
        at["n"] = to
    return there, monitor.WatchedRun(
        names=RunNames.of("H2", "01_relax", "hierarchical"), run=0,
        directory=there), grow


def _statuses(there: Path):
    return [ln for ln in (there / f"{_STEM}-run0.monitor.log")
            .read_text().splitlines() if "[STATUS]" in ln]


def _whole(run: Path) -> int:
    return len(_Output(run).lines)


# --------------------------------------------------------------------- #
#  The PID says WHEN; the framework says HOW (`run-reports.md` § 2.3)    #
# --------------------------------------------------------------------- #

def test_a_completion_marker_does_not_stop_the_sampling(run, tmp_path):
    """The loop keeps going while the watched PID lives, whatever the output
    says: the whole run is on disk, ``>> End of run`` included, and this
    process -- certainly alive -- is the watched PID (`job-contracts.md`: the
    monitor follows the launcher's PID rather than guessing from output
    markers)."""
    _, watched, _grow = _replay(run, tmp_path, _whole(run))
    slept = []
    final = monitor.run_monitor(
        watched, interval=1, watch_pid=os.getpid(), max_ticks=3,
        sleep=lambda s: slept.append(s),
        clock=_fake_clock([1000.0, 1001.0, 1002.0, 1003.0, 1004.0]))
    assert len(slept) == 3, f"the loop ran {len(slept)} time(s)"
    assert final.state == "running", "the watched PID is alive"


def test_when_the_pid_goes_the_verdict_is_run_status_s(run, tmp_path):
    """Asked once the PID has gone, HOW it ended is the Results tab's own
    reading -- `run_status` over the rung's files -- with each phase's
    convergence, the process's goodbye, and whether the geometry converged
    as the output states it."""
    from molbuilder.parse.dirs import run_status
    there, watched, _grow = _replay(run, tmp_path, _whole(run))
    final = monitor.run_monitor(
        watched, interval=1, watch_pid=999_999_999,
        sleep=lambda s: None, clock=_fake_clock([0.0, 0.0, 1.0]))
    rs = run_status(there, _STEM)
    assert (final.state, final.detail) == (rs.state, rs.detail) == (
        "finished", "job_completed")
    assert final.converged == {"periodic": True}
    assert final.exit and final.exit.startswith("rc=0")
    closing = _statuses(there)[-1]
    assert "finished (job_completed)" in closing
    assert "converged: periodic yes" in closing
    assert final.relaxed is True and "geometry relaxed" in closing, closing
    assert "job ended" in (there / f"{_STEM}-run0.monitor.log").read_text()


# --------------------------------------------------------------------- #
#  Looking often, logging on change                                      #
# --------------------------------------------------------------------- #

def test_each_advance_is_a_status_line_stating_where_the_run_is(run,
                                                                tmp_path):
    """A progressing run logs a [STATUS] line on each advance, saying what is
    going on: the phase and iteration, each residual beside the criterion the
    run states for it, the step SIESTA began in its own words and number,
    and the force against its tolerance."""
    o = _Output(run)
    assert len(o.moves) >= 3, "the relaxation made fewer than three moves"
    first = o.rows_of(0)
    k = min(7, len(first))
    there, watched, grow = _replay(run, tmp_path, o.through(first[0]))
    chunks = iter([o.through(first[k - 1]), o.end_of_move(1),
                   o.end_of_move(2)])
    monitor.run_monitor(watched, interval=1, watch_pid=0,
                        sleep=lambda _: grow(next(chunks)), max_ticks=3,
                        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0]))
    lines = _statuses(there)
    assert len(lines) == 3, lines
    assert f"periodic SCF iteration {k}" in lines[0], lines[0]
    dm = o.criterion("DM")
    h = o.criterion("H|Hamiltonian")
    assert "dDmax" in lines[0] and f"(tol {dm:g})" in lines[0], lines[0]
    assert "dHmax" in lines[0] and f"(tol {h:g})" in lines[0], lines[0]
    assert re.search(r"\w+ opt\. move 2\b", lines[-1]), lines[-1]
    force = o.last_free_force(o.end_of_move(2))
    tol = o.criterion("Force")
    assert f"max force {force:.4g} eV/Ang (tol {tol:g})" in lines[-1], (
        lines[-1])


def test_a_run_that_does_not_advance_writes_and_sends_nothing(run, tmp_path):
    """A live run that does not advance -- for over an hour here, as a heavy
    SCF step does -- adds no status line and tells no channel anything: the
    monitor judges no stall (`run-reports.md` § 2)."""
    o = _Output(run)
    there, watched, _grow = _replay(run, tmp_path,
                                    o.through(o.rows_of(0)[1]))
    seen = []
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    monitor.run_monitor(watched, interval=1, watch_pid=0,
                        sleep=lambda s: None, max_ticks=4,
                        clock=_fake_clock([0.0, 0.0, 0.0, 1200.0, 2400.0,
                                           3600.0, 4800.0]))
    text = (there / f"{_STEM}-run0.monitor.log").read_text()
    assert "[STATUS]" not in text, text
    assert [e for e in seen if e not in ("start", "finish")] == [], seen


def test_util_sampling_is_change_gated_and_summarised(run, tmp_path):
    """The run's `.util.csv` gets a row only when a metric moves >= 10 % (or a
    keepalive passes), and a [UTIL-SUMMARY] verdict lands at the end.  The
    samples are scripted -- the SAMPLER is the monitor's own, and what it
    reads is the machine, not a file."""
    there, watched, _grow = _replay(run, tmp_path, _whole(run))
    seq = [
        monitor.UtilSample(0.0, 50.0, 100.0, [(0, 95.0, 40.0, 20.0)]),
        monitor.UtilSample(0.0, 51.0, 100.0, [(0, 94.0, 40.0, 20.0)]),
        monitor.UtilSample(0.0, 52.0, 100.0, [(0, 60.0, 40.0, 20.0)]),
        monitor.UtilSample(0.0, 52.0, 100.0, [(0, 60.0, 40.0, 20.0)]),
    ]
    it = iter(seq)
    last = {"s": seq[0]}

    def _sampler():
        try:
            last["s"] = next(it)
        except StopIteration:
            pass
        return last["s"]

    monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                        sleep=lambda s: None, max_ticks=4, util=True,
                        util_keepalive_s=1e9, sampler=_sampler,
                        clock=_fake_clock([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
    rows = (there / f"{_STEM}-run0.util.csv").read_text().splitlines()
    assert rows[0].startswith("epoch,iso,cpu_pct,mem_gb,gpu0_sm")
    assert len(rows) >= 2
    log = (there / f"{_STEM}-run0.monitor.log").read_text()
    assert "[UTIL-SUMMARY]" in log and "gpu0 sm mean=" in log


def test_a_failing_notifier_does_not_break_the_loop(run, tmp_path):
    """Ends on the WATCHED PID, the only stop signal."""
    _, watched, _grow = _replay(run, tmp_path, _whole(run))

    def _boom(st, ev):
        raise RuntimeError("notifier blew up")

    seen = []
    monitor.register_notifier(_boom)
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    final = monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                                sleep=lambda s: None,
                                clock=_fake_clock([0.0, 0.0, 1.0]))
    assert final.state == "finished"
    assert "finish" in seen, "the second notifier ran despite the first"


# --------------------------------------------------------------------- #
#  Who is TOLD: the policy (`run-reports.md` § 2)                         #
# --------------------------------------------------------------------- #

def _events(run, tmp_path, *, upto, chunks=(), notify_on_scf=False,
            notify_every_hours=0.0, **kw):
    """The monitor over the replayed run -- ``chunks`` are where the output
    has grown to at each wake -- and the events its notifiers saw."""
    _, watched, grow = _replay(run, tmp_path, upto)
    it = iter(chunks)

    def _sleep(_):
        to = next(it, None)
        if to is not None:
            grow(to)

    seen = []
    monitor.register_notifier(lambda st, ev: seen.append(ev))
    monitor.run_monitor(watched, interval=1,
                        notify=NotifyPolicy(on_scf=notify_on_scf,
                                            every_hours=notify_every_hours),
                        sleep=_sleep, **kw)
    return seen


def _rows_of_the_first_move(o: _Output):
    """Each wake adds one more SCF row of the first move."""
    return [o.through(r) for r in o.rows_of(0)[1:]]


def _the_next_moves(o: _Output):
    """Each wake completes one more move's SCF, up to three."""
    assert len(o.moves) >= 3, "the relaxation made fewer than three moves"
    return [o.end_of_move(k) for k in range(1, min(4, len(o.moves)))]


def test_an_advancing_job_notifies_nothing_by_default(run, tmp_path):
    """THE REGRESSION.  With no policy set, a job actively progressing
    produces no notification beyond its start and end."""
    o = _Output(run)
    seen = _events(run, tmp_path, upto=o.through(o.rows_of(0)[0]),
                   chunks=_rows_of_the_first_move(o), watch_pid=0,
                   max_ticks=6, clock=_fake_clock([0, 1, 2, 3, 4, 5, 6, 7]))
    assert [e for e in seen if e not in ("start", "finish")] == [], seen


def test_one_message_per_geometry_step(run, tmp_path):
    """A geometry step advancing means the previous SCF reached its
    criterion -- SIESTA begins the next move -- so each move begun is one
    message."""
    o = _Output(run)
    moves = _the_next_moves(o)
    seen = _events(run, tmp_path, upto=o.end_of_move(0), chunks=moves,
                   watch_pid=0, max_ticks=len(moves), notify_on_scf=True,
                   clock=_fake_clock(list(range(len(moves) + 3))))
    assert seen.count("scf_converged") == len(moves), seen


def test_periodic_counts_hours_not_wakes(run, tmp_path):
    """Eight wakes an hour apart, a two-hour period: four messages at most,
    not eight.  The wake interval and the reporting period are independent."""
    o = _Output(run)
    seen = _events(run, tmp_path, upto=o.through(o.rows_of(0)[0]),
                   chunks=_rows_of_the_first_move(o), watch_pid=0,
                   max_ticks=8, notify_every_hours=2,
                   clock=_fake_clock([i * 3600.0 for i in range(0, 9)]))
    assert 3 <= seen.count("periodic") <= 4, seen


def test_the_first_period_is_a_full_period_in(run, tmp_path):
    """"Every 6 hours" must not mean "now, and then every 6 hours": the clock
    starts at the job's start, so nothing is due on the first wake."""
    o = _Output(run)
    seen = _events(run, tmp_path, upto=o.through(o.rows_of(0)[0]),
                   watch_pid=0, max_ticks=2, notify_every_hours=6,
                   clock=_fake_clock([0.0, 60.0, 120.0]))
    assert "periodic" not in seen, seen


def test_a_step_and_a_period_on_one_wake_is_one_message(run, tmp_path):
    """Both triggers coming due together is one thing worth saying: the step
    is the more informative, so it wins and resets the clock."""
    o = _Output(run)
    moves = _the_next_moves(o)
    seen = _events(run, tmp_path, upto=o.end_of_move(0), chunks=moves,
                   watch_pid=0, max_ticks=len(moves), notify_on_scf=True,
                   notify_every_hours=1,
                   clock=_fake_clock([i * 3600.0
                                      for i in range(len(moves) + 2)]))
    assert "periodic" not in seen, f"the step already said it; {seen}"
    assert seen.count("scf_converged") == len(moves), seen


def test_a_relaxation_with_the_trigger_OFF_reports_no_steps(run, tmp_path):
    """The steps genuinely advance and the trigger is off, so a message
    would be one the person did not ask for."""
    o = _Output(run)
    moves = _the_next_moves(o)
    seen = _events(run, tmp_path, upto=o.end_of_move(0), chunks=moves,
                   watch_pid=0, max_ticks=len(moves), notify_on_scf=False,
                   clock=_fake_clock(list(range(len(moves) + 3))))
    assert "scf_converged" not in seen, seen


# --------------------------------------------------------------------- #
#  Where it ran (`run-reports.md` § 2.5)                                  #
# --------------------------------------------------------------------- #

def _clock():
    it = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    last = [0.0]

    def clock():
        last[0] = next(it, last[0])
        return last[0]
    return clock


def test_the_machine_is_the_logs_first_line(run, tmp_path):
    """Written at START, not in the terminal block: a monitor killed with
    its allocation must still have said where it died."""
    _, watched, _grow = _replay(run, tmp_path, _whole(run))
    monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                        sleep=lambda s: None, clock=_clock())
    first = watched.path(".monitor.log").read_text(
        encoding="utf-8").splitlines()[0]
    assert "[MACHINE]" in first, first
    assert f"node={os.uname().nodename[:128]}" in first


def test_a_monitor_missing_its_companion_still_monitors(run, tmp_path,
                                                        monkeypatch):
    """`config_dir.py` travels beside the shipped monitor; when a staging
    defect loses it, WHERE reports go cannot be answered -- and the answer is
    *reports off*, said in the log, never startup death."""
    def _no_companion():
        raise ModuleNotFoundError("config_dir")
    monkeypatch.setattr(monitor, "_secrets_dir", _no_companion)
    monkeypatch.delenv("MOLBUILDER_NOTIFY_FILE", raising=False)
    _, watched, _grow = _replay(run, tmp_path, _whole(run))
    monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                        sleep=lambda s: None, clock=_clock())
    text = watched.path(".monitor.log").read_text(encoding="utf-8")
    assert "[MACHINE]" in text and "[MONITOR]" in text, text
    assert "reports off" in text, text


# --------------------------------------------------------------------- #
#  The shipped files, beside a job, with molbuilder absent (§ 2.3)       #
# --------------------------------------------------------------------- #

def test_every_file_that_ships_beside_a_job_reads_the_run_without_molbuilder(
        run, tmp_path):
    """`runwrap.MONITOR_COMPANIONS` travels to the machine that runs the job
    and executes under the job's own python, where molbuilder is not
    installed: staged into a copy of the run with the bundle beside it, the
    shipped monitor imports, reads the run through the SHIPPED readers, and
    answers how it ended as the package's own `run_status` does."""
    from molbuilder.parse.dirs import run_status
    from molbuilder.runwrap import MONITOR_BUNDLE, monitor_bundle
    there, _watched, _grow = _replay(run, tmp_path, _whole(run))
    (there / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    control = subprocess.run([sys.executable, "-c", "import molbuilder"],
                             cwd=there, env=env, capture_output=True,
                             text=True, timeout=120)
    assert control.returncode != 0, (
        "molbuilder is importable from the run folder, so this cannot "
        "reproduce a compute node")
    probe = (
        f"import sys; sys.path.insert(0, {MONITOR_BUNDLE!r})\n"
        "import json, os, mb_monitor as M\n"
        "assert M.default_notify_path().name == 'notify'\n"
        "P = sys.modules[M.SiestaReader.__module__]\n"
        "where = {n: getattr(o, '__module__', getattr(o, '__name__', ''))\n"
        "         for n, o in (('status', M.run_status),\n"
        "                      ('siesta', M.SiestaReader),\n"
        "                      ('molwatch', M.MolwatchReader),\n"
        "                      ('grammar', P._G), ('rules', P.compile_rules),\n"
        "                      ('names', M._rf))}\n"
        "where['from'] = os.path.basename(os.path.dirname(M.__file__))\n"
        f"w = M.WatchedRun(names=M._rf.RunNames.of('H2', '01_relax', "
        f"'hierarchical'), run=0)\n"
        "st = w.conclude(w.read(0.0, 1.0))\n"
        "print(json.dumps({'where': where, 'state': st.state,\n"
        "                  'detail': st.detail, 'converged': st.converged,\n"
        "                  'energy': st.energy, 'text': st.as_text()}))\n")
    done = subprocess.run([sys.executable, "-c", probe], cwd=there, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"{done.stdout}{done.stderr}"
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["where"] == {"status": "job", "siesta": "siesta_reader",
                            "molwatch": "molwatch_reader",
                            "grammar": "siesta_grammar",
                            "rules": "_section_rules",
                            "names": "runfiles",
                            "from": MONITOR_BUNDLE}, got["where"]
    here = run_status(run, _STEM)
    assert (got["state"], got["detail"]) == (here.state, here.detail), got
    speaker = here.endings[here.active_source]
    assert got["converged"] == dict(speaker.phases.items()), got
    assert got["energy"] is not None, got["text"]
