"""THE REAL RUNS AN END-TO-END PASS MAKES, EACH ONCE (`process/testing.md`
§ 0, tier 3; plan § 5y).

A test that reads what an engine wrote is an end-to-end test, made with the
engine (user, 2026-10-08: *"when something reads the output of \\"siesta\\"
output that's an end to end by definition"*), and one pass makes each run
once: every module that reads a run asks for it here (*"it should be
integrated/merged with existing e2e tests so one run of e2e would produce all
information"*).

Each run is made down the road -- `jobset init`, `prep`,
`launch` -- with the real SIESTA (`_road.live_siesta`), writes down what
every command it typed answered, and leaves the process environment as it
found it before any test reads the run: a later test in the same worker
meets the suite's tripwire, never an engine.  A test reads the run's files
and the answers written down; it types no command that needs the engine's
environment.

Each is the session fixture of its name in `tests/conftest.py`:

* :func:`layered` (``real_h2_layered``) -- an H2 optimization, layered
  folders, the `publishable` ladder: `coarse` launched, a dry run of
  launching it again, launched again warm and then cold, `medium` prepared
  on it and sent by a launch that names no stage, left in the background,
  and the jobs launched from here asked twice once it ended; and the run
  script's own dry runs, on a copy of `coarse`'s prepared folder;
* :func:`bench` (``real_h2_bench``) -- a two-trial benchmark of the same
  H2's `coarse`: one trial launched by name, then the rest by a launch of
  the benchmark under a per-trial bound;
* :func:`relaxed_for_vibration` (``real_relaxed_for_vibration``) -- the
  vibration kind's `relax` stage of the road's CO2, listed O O C with the
  carbon held, so its decks are in the engine's held-first order
  (`support.road.CO2`; plan T-CO2).

A run with one thing changed is a copy of it beside it (:func:`copy_of`):
its next stage prepared and launched (:func:`freq_launched_on`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pytest


@dataclass
class Said:
    """What one command answered: its exit code and its output."""
    exit_code: int
    output: str


@dataclass
class RealRun:
    """A calculation made on the road with the engine, and what every
    command typed on it answered, in the order typed (``said``, by a name
    the fixture gives each step).  ``files_before`` / ``files_after`` hold
    the calculation's files around the one step that must write nothing;
    ``jobs`` what `molbuilder jobs status` answered, by when it was asked
    (`job-system.md` § 6.0a)."""
    tree: Path
    bundle: Path
    said: Dict[str, Said] = field(default_factory=dict)
    jobs: Dict[str, object] = field(default_factory=dict)
    files_before: Dict[str, tuple] = field(default_factory=dict)
    files_after: Dict[str, tuple] = field(default_factory=dict)

    def attempt(self, stage: str, n: int = 0) -> Path:
        """Attempt ``n`` of ``stage`` -- its folder, by molbuilder's own
        doors (`jobset.materialize.stage_home`, `paths.attempt_dir`): the
        handle every reader of the run takes, asking the run door for the
        rest (`runs.run_of`)."""
        from molbuilder.jobset.materialize import stage_home
        from molbuilder.paths import attempt_dir
        from molbuilder.runfiles import TASK_FILE
        from molbuilder.task import read_task
        task = read_task(self.bundle / TASK_FILE)
        return attempt_dir(stage_home(self.bundle, task, stage).dir, n)


def _engine_here() -> bool:
    from _road import conda_hook, env_available
    return conda_hook().is_file() and env_available("molbuilder-siesta")


@contextmanager
def _on_the_road(tmp_path_factory, name: str):
    """A projects tree of its own, the one door pointed at it, and the real
    SIESTA on the road -- while the runs are made; the process environment
    as it was, after."""
    with _road_at(tmp_path_factory.mktemp(name), tmp_path_factory) as tree:
        yield tree


@contextmanager
def _road_at(where: Path, tmp_path_factory):
    """The projects tree under ``where`` -- a new one, or the one a run was
    made in, for a command typed on a copy of it -- the one door pointed at
    it, and the real SIESTA on the road; the process environment as it was,
    after."""
    from _pytest.monkeypatch import MonkeyPatch

    from _road import live_siesta
    from conftest import _point_the_one_door_at
    if not _engine_here():
        pytest.skip("needs the molbuilder-siesta env + a detectable conda hook")
    mp = MonkeyPatch()
    try:
        tree = _point_the_one_door_at(where, mp.setenv)
        with live_siesta(tree, tmp_path_factory):
            yield tree
    finally:
        mp.undo()


def _typed(run: RealRun, step: str, *words, input=None) -> Said:
    """``molbuilder jobset <words> --bundle <the calculation>``, written down
    under ``step``."""
    from support.road import jobset
    r = jobset(*words, "--bundle", run.bundle, input=input)
    said = Said(r.exit_code, r.output)
    run.said[step] = said
    return said


def _taken(run: RealRun, step: str, *words) -> Said:
    said = _typed(run, step, *words)
    assert said.exit_code == 0, f"{step}: {said.output}"
    return said


def _launched_here(run: RealRun) -> int:
    """The pid of the run here a launch left in the background -- its
    ledger's ``launched`` line (`job-system.md` § 6.0)."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    lines = [json.loads(x) for x in
             (run.bundle / LEDGER_FILE).read_text().splitlines() if x.strip()]
    return next(e["pid"] for e in reversed(lines)
                if e["decision"] == "launched" and e.get("background"))


def _until(holds, *, within: float, what: str) -> None:
    """``holds()``, within ``within`` seconds -- a run's own files read as
    it goes."""
    import time
    end = time.monotonic() + within
    while not holds():
        assert time.monotonic() < end, f"{what}: not within {within} s"
        time.sleep(0.5)


def _stopped_by_its_line(run: RealRun, step: str, launched: Said) -> None:
    """A run here a launch left in the background, stopped as that launch
    said to stop it -- its *to stop it:* line, typed at a terminal and its
    question answered yes, written down under ``step`` -- then waited for
    until no process of its group is left: the run script, the engine and
    the monitor, whose closing record is the stop's last word
    (`job-system.md` § 6.0, § 6.0a)."""
    import shlex
    import time
    line = next(ln.split("to stop it:", 1)[1].strip()
                for ln in launched.output.splitlines() if "to stop it:" in ln)
    words = shlex.split(line)
    assert words[0] == "molbuilder", line
    said = _at_a_terminal(words[1:], "y")
    run.said[step] = said
    assert said.exit_code == 0, f"{step}: {said.output}"
    pid = _launched_here(run)
    _left_to_end(run)
    end = time.monotonic() + 120
    while True:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        assert time.monotonic() < end, f"{line}: its processes went on"
        time.sleep(0.5)


def _left_to_end(run: RealRun) -> None:
    """A run here the launch left in the background, waited for to its end:
    the process its ledger's ``launched`` line names (`job-system.md`
    § 6.0) -- a child of this process, which the launch started.  What it
    was as the launch returned is kept in ``run.left``: its pid, its session
    (``sid``), and whether it was still this process's to wait for
    (``waited_here``) -- a launch that waited for it would have reaped it."""
    pid = _launched_here(run)
    run.left = {"pid": pid, "sid": None, "waited_here": False}
    try:
        run.left["sid"] = os.getsid(pid)
        os.waitpid(pid, 0)
        run.left["waited_here"] = True
    except (ChildProcessError, ProcessLookupError):
        pass        # reaped already -- by the launch, or `subprocess`'s own
                    # cleanup of an ended child: it ended


def _at_a_terminal(words, answer: str) -> Said:
    """``molbuilder <words>`` at a terminal, its question answered
    ``answer`` as a person types it: the real command in a process of its
    own, a pseudo-terminal its input (`job-system.md` § 6.0), started from
    the repository's root as every test that runs the command in its own
    process starts it -- the package is run from its source, never
    installed -- the road's settings reaching it in its environment."""
    import pty
    import sys
    lead, follow = pty.openpty()
    try:
        os.write(lead, f"{answer}\n".encode())
        done = subprocess.run(
            [sys.executable, "-m", "molbuilder.cli", *map(str, words)],
            stdin=follow, capture_output=True, text=True, timeout=3600,
            cwd=Path(__file__).resolve().parents[2])
    finally:
        os.close(follow)
        os.close(lead)
    return Said(done.returncode, done.stdout + done.stderr)


def _answered(run: RealRun, step: str, answer: str, *words) -> Said:
    """``molbuilder jobset <words> --bundle <the calculation>`` at a
    terminal, its question answered ``answer`` (:func:`_at_a_terminal`);
    written down under ``step``; it must succeed."""
    said = _at_a_terminal(["jobset", *words, "--bundle", run.bundle], answer)
    run.said[step] = said
    assert said.exit_code == 0, f"{step}: {said.output}"
    return said


def _described(tree: Path, where: str, *, structure: dict, name: str,
               calculation: str, stage_strategy: str = "",
               **blocks) -> Path:
    """`jobset init` of the road's ``structure`` (`support.road`, written
    where `init` takes it), SIESTA, layered folders, the ``calculation``'s
    own ladder or ``stage_strategy``; then the description's run card and
    benchmark as given -- each block stated, or left out.  THE ONE MAKER of
    a calculation on the road with the engine."""
    from molbuilder.workingcopy_structure import StructureCodec
    from support.road import jobset, road_structure
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    source = f"P/structure/{name.lower()}.xyz"
    StructureCodec().write(road_structure(structure), tree / source)
    r = jobset("init", "--structure", source, "--bundle", where,
               "--engine", "siesta", "--calculation", calculation,
               "--shape", "hierarchical", "--name", name,
               "--psml-lib", "pseudopotential",
               *(("--stage-strategy", stage_strategy) if stage_strategy
                 else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / where
    task = json.loads((bundle / "task.json").read_text())
    for block, value in blocks.items():
        if value is None:
            task.pop(block, None)
        else:
            task[block] = value
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def _files(bundle: Path) -> Dict[str, tuple]:
    """Every file under the calculation, by its size and write time."""
    return {str(p.relative_to(bundle)): (p.stat().st_size,
                                         p.stat().st_mtime_ns)
            for p in sorted(bundle.rglob("*")) if p.is_file()}


def _script_dry_runs(run: RealRun, prepared: Path, scratch: Path) -> None:
    """The run script's own dry run (`running-a-job.md` § 3.5), on a copy of
    a prepared folder so no run another test reads gains a log: as prepared,
    and with ``-np 2`` typed on it.  The rank and thread counts the shell
    may carry are scrubbed, so what resolves is the script's own chain."""
    there = scratch / prepared.name
    shutil.copytree(prepared, there)
    script = next(there.glob("*.run.sh"))
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "MB_NP",
                        "SLURM_NTASKS", "SLURM_JOB_ID", "PBS_NP",
                        "MOLBUILDER_USE_MPS")}
    env.update(MB_LAUNCHED_BY="manual")
    for step, extra in (("script dry run", ()),
                        ("script dry run -np 2", ("-np", "2"))):
        done = subprocess.run(["bash", str(script), "--run", "0",
                               "--dry-run", *extra],
                              cwd=there, capture_output=True, text=True,
                              timeout=120, env=env)
        run.said[step] = Said(done.returncode, done.stdout + done.stderr)


def _h2_optimization(tree: Path, where: str, **blocks) -> Path:
    """The road's H2, a SIESTA optimization on the `publishable` ladder
    (:func:`_described`)."""
    from support.road import H2
    return _described(tree, where, structure=H2, name="H2",
                      calculation="optimization",
                      stage_strategy="publishable", **blocks)


def layered(tmp_path_factory) -> RealRun:
    """Run A of plan § 5y -- see the module's note."""
    with _on_the_road(tmp_path_factory, "layered") as tree:
        bundle = _h2_optimization(tree, "P/opt/H2",
                                  execution={"mpi_np": 1, "omp_threads": 1})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep coarse", "prep", "task", "--group", "1",
               "--target", "this")
        _script_dry_runs(run, bundle / "01_coarse" / "run-0",
                         tmp_path_factory.mktemp("dry-run"))
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
        run.files_before = _files(bundle)
        _taken(run, "dry run of launching coarse again", "launch", "task",
               "--stage", "coarse", "--mode", "direct", "--dry-run", "--yes")
        run.files_after = _files(bundle)
        _taken(run, "launch coarse again", "launch", "task", "--stage",
               "coarse", "--mode", "direct", "--foreground", "--yes")
        # ASKED AT A TERMINAL, AND ANSWERED THE OTHER WAY: the foreground.
        _answered(run, "launch coarse again cold", "f", "launch", "task",
                  "--stage", "coarse", "--mode", "direct", "--cold")
        _taken(run, "prep medium", "prep", "task", "--group", "2",
               "--target", "this")
        # A RUN HERE, LEFT IN THE BACKGROUND -- a launch's default
        # (`job-system.md` § 6.0) -- and waited for.
        _taken(run, "launch naming no stage", "launch", "task", "--mode",
               "direct", "--yes")
        _left_to_end(run)
        # ...AND THE JOBS LAUNCHED FROM HERE, asked once it ended, and again
        # (`job-system.md` § 6.0a).
        from support.road import molbuilder
        run.jobs["after it ended"] = molbuilder("jobs", "status")
        run.jobs["again"] = molbuilder("jobs", "status")
        _taken(run, "status", "status")
    return run


def bench(tmp_path_factory) -> RealRun:
    """The benchmark of plan § 5y -- see the module's note."""
    with _on_the_road(tmp_path_factory, "bench") as tree:
        bundle = _h2_optimization(tree, "P/opt/H2bench", execution=None,
                                  bench={"mpi_np": [1, 2],
                                         "omp_threads": [1]})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep bench", "prep", "bench", "coarse",
               "--target", "this")
        _taken(run, "launch one trial", "launch", "bench", "coarse",
               "G0K1C1", "--mode", "direct", "--foreground", "--yes")
        _taken(run, "launch the benchmark", "launch", "bench", "coarse",
               "--mode", "direct", "--foreground", "--yes", "--trial-timeout", "5")
    return run


def co2_vibration(tree: Path, where: str = "P/freq/CO2") -> Path:
    """The road's CO2 (`support.road.CO2`) described as a SIESTA vibration
    in ``tree`` at ``where``: the kind's own ladder (`relax` then `freq`), one
    rank, one thread (:func:`_described`) -- THE ONE MAKER of it, for the
    shared relaxation and for a test's own calculation alike."""
    from support.road import CO2
    return _described(tree, where, structure=CO2, name="CO2",
                      calculation="vibration",
                      execution={"mpi_np": 1, "omp_threads": 1})


def relaxed_for_vibration(tmp_path_factory) -> RealRun:
    """The vibration kind's `relax` stage of the road's CO2, run: the kind's
    own ladder (`relax` then `freq`), one rank, one thread, `relax` prepared
    and launched."""
    from molbuilder.runs import run_answer
    with _on_the_road(tmp_path_factory, "vibration") as tree:
        run = RealRun(tree=tree, bundle=co2_vibration(tree))
        _taken(run, "prep relax", "prep", "task", "--group", "1",
               "--target", "this")
        _taken(run, "launch relax", "launch", "task", "--stage", "relax",
               "--mode", "direct", "--foreground", "--yes")
    # IT REACHED ITS END, by the one status door: every reader of the run
    # reads a relaxation that finished.
    said = run_answer(run.attempt("relax"))
    assert (said["state"], said["detail"]) == ("finished", "job_completed"), said
    return run


#: The copies whose change succeeded, this session (:func:`copy_of`).
_MADE: Dict[Path, RealRun] = {}


def copy_of(run: RealRun, name: str, alter) -> RealRun:
    """THE SHARED RUN WITH ONE THING CHANGED: a copy of the calculation
    beside it, ``<bundle>-<name>``, altered once by ``alter(copy)`` -- made
    once a session, the run's own files left as they are, and the same copy,
    with what was typed on it, every time it is asked for.  A copy counts as
    made once its change succeeded; one a failed change left behind is made
    again."""
    bundle = run.bundle.parent / f"{run.bundle.name}-{name}"
    if bundle not in _MADE:
        copy = RealRun(tree=run.tree, bundle=bundle)
        shutil.rmtree(bundle, ignore_errors=True)
        shutil.copytree(run.bundle, bundle)
        alter(copy)
        _MADE[bundle] = copy
    return _MADE[bundle]


def freq_launched_on(run: RealRun, tmp_path_factory) -> RealRun:
    """A copy of the calculation (:func:`copy_of`) with its `freq` stage
    prepared and launched on the road, `summarize` asked of it, and `freq`
    launched again -- what each answered written down, ``said["prep freq"]``,
    ``said["launch freq"]``, ``said["summarize"]``,
    ``said["launch freq again"]``."""
    def launched(copy):
        with _road_at(copy.tree.parent, tmp_path_factory):
            _taken(copy, "prep freq", "prep", "task", "--group", "2",
                   "--target", "this")
            _taken(copy, "launch freq", "launch", "task", "--stage", "freq",
                   "--mode", "direct", "--foreground", "--yes")
            _typed(copy, "summarize", "summarize", "task")
            _taken(copy, "launch freq again", "launch", "task", "--stage",
                   "freq", "--mode", "direct", "--foreground", "--yes")
    return copy_of(run, "freq-launched", launched)
