"""THE REAL RUNS AN END-TO-END PASS MAKES, EACH ONCE (`process/testing.md`
§ 0, tier 3; plan § 5y).

A test that reads what an engine wrote is an end-to-end test, made with the
engine (user, 2026-10-08: *"when something reads the output of \\"siesta\\"
output that's an end to end by definition"*), and one pass makes each run
once: every module that reads a run asks for it here (*"it should be
integrated/merged with existing e2e tests so one run of e2e would produce all
information"*).

Each run is made down the road -- `molbuilder describe`, `prep`,
`launch` -- with the real SIESTA (`_road.live_siesta`), writes down what
every command it typed answered, and leaves the process environment as it
found it before any test reads the run: a later test in the same worker
meets the suite's tripwire, never an engine.  A test reads the run's files
and the answers written down; it types no command that needs the engine's
environment.

Each is the session fixture of its name in `tests/conftest.py`:

* :func:`layered` (``real_h2_layered``) -- an H2 optimization on the
  `publishable` ladder: `coarse` launched, a dry run of
  launching it again, launched again warm and then cold, `medium` prepared
  on it and sent by a launch that names no stage, left in the background,
  and the jobs launched from here asked twice once it ended; and the run
  script's own dry runs, on a copy of `coarse`'s prepared folder;
* :func:`bench` (``real_h2_bench``) -- a two-trial benchmark of the same
  H2's `coarse`: one trial launched by name, then the rest by a launch of
  the benchmark under a per-trial bound;
* :func:`vibration` (``real_co2_vibration``) -- the vibration kind's one
  run in phases of the road's CO2, listed O O C with the carbon held, so its
  decks are in the engine's held-first order (`support.road.CO2`; plan
  T-CO2): its `relax` call, its `fc` call and the two stretches' five
  optical calls (`engines/vibration.md` § 5.11), waited on;
* :func:`gpu` (``real_h2_gpu``) -- the same H2 asking this machine's GPU,
  one rank: its run script's dry runs, then the run, waited on;
* :func:`pyscf` (``real_h2_pyscf``) -- an H2 optimization on PySCF,
  stretched so it moves: `coarse` launched and waited on.

A run with one thing changed is a copy of it beside it (:func:`copy_of`).
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
from support.road import _all_written, molbuilder


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
    the calculation's files around the one step that must write nothing,
    ``relaunched`` them once a stage was launched again, warm and cold;
    ``jobs`` what `molbuilder jobs status` answered, by when it was asked
    (`job-system.md` § 6.0a)."""
    tree: Path
    bundle: Path
    said: Dict[str, Said] = field(default_factory=dict)
    jobs: Dict[str, object] = field(default_factory=dict)
    files_before: Dict[str, tuple] = field(default_factory=dict)
    files_after: Dict[str, tuple] = field(default_factory=dict)
    relaunched: Dict[str, tuple] = field(default_factory=dict)

    def attempt(self, stage: str, n: int = 0) -> Path:
        """Run ``n`` of ``stage`` -- its folder, by molbuilder's own
        doors (`jobset.naming.stage_home`, `paths.attempt_dir`): the
        handle every reader of the run takes, asking the run door for the
        rest (`runs.run_of`)."""
        from molbuilder.jobset.naming import stage_home
        from molbuilder.paths import attempt_dir
        from molbuilder.task import of as task_of
        task = task_of(self.bundle)
        return attempt_dir(stage_home(self.bundle, task, stage).dir, n)

    def call(self, stage: str, phase: str, n: int = 0) -> Path:
        """The SIESTA call of ``phase`` in run ``n`` of ``stage`` -- a
        run in phases' point, its folder below the run as the kind names it
        (`siesta.vibration_phases.Call.rel`): the relaxation's, the force
        constants'."""
        from molbuilder.siesta.vibration_phases import Call
        return self.attempt(stage, n) / Call(phase, phase).rel


#: Each engine's environment, and the road a run of it is made on.
_ENGINES = {"siesta": "molbuilder-siesta", "pyscf": "molbuilder-pySCF"}


def stage_status(bundle, stage: str):
    """THE STAGE AS `jobset status` READS IT -- the one status door every
    surface asks (`jobset.runstatus.jobset_status`: the status verb, Task
    setup's ladder), asked of the calculation's folder: its newest run
    (``attempt``) and every run (``attempts``), its launch record, what the
    run gathered (``gathered``, ``<file> <- <run>`` each) and what it is a
    copy of (``copied_from``), and each point -- ``frame``, ``bias_v``,
    ``folder`` below the stage, ``state``, ``done``, ``started_from``,
    ``took``, ``copied_from``.  A test reads a run's records here, or, for a
    run older than the newest, through the Results tab's folder door
    (`runs.folder_answer`) -- never from its marker files."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.runstatus import jobset_status
    return next(s for s in jobset_status(open_calculation(bundle)).stages
                if s.ref.name == stage)


def point_row(status, bias_v: float, frame=None) -> dict:
    """``status``'s point at ``bias_v`` on ``frame`` -- the frame matched as
    the status door states it, ``None`` for one structure."""
    return next(p for p in status.points
                if p["bias_v"] == bias_v and p["frame"] == frame)


def _engine_here(engine: str = "siesta") -> bool:
    from _road import conda_hook, env_available
    return conda_hook().is_file() and env_available(_ENGINES[engine])


@contextmanager
def _on_the_road(tmp_path_factory, name: str, *, engine: str = "siesta"):
    """A projects tree of its own, the one door pointed at it, and the real
    ``engine`` on the road -- while the runs are made; the process
    environment as it was, after."""
    with _road_at(tmp_path_factory.mktemp(name), tmp_path_factory,
                  engine=engine) as tree:
        yield tree


@contextmanager
def _road_at(where: Path, tmp_path_factory, *, engine: str = "siesta"):
    """The projects tree under ``where`` -- a new one, or the one a run was
    made in, for a command typed on a copy of it -- the one door pointed at
    it, and the real ``engine`` on the road (`_road.live_siesta`,
    `_road.live_pyscf`); the process environment as it was, after."""
    from _pytest.monkeypatch import MonkeyPatch

    from _road import live_pyscf, live_siesta
    from conftest import _point_the_one_door_at
    if not _engine_here(engine):
        pytest.skip(f"needs the {_ENGINES[engine]} env + a detectable conda "
                    f"hook")
    mp = MonkeyPatch()
    try:
        tree = _point_the_one_door_at(where, mp.setenv)
        live = live_siesta if engine == "siesta" else live_pyscf
        with live(tree, tmp_path_factory):
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


def notified(run: RealRun, stage: str, n: int = 0) -> list:
    """The events run ``n`` of ``stage`` sent, by its record in the run's
    folder (`runrecord.read_notified`) -- a run of points is one run in its
    folder, one record for its points (`run-reports.md` § 2.4)."""
    from molbuilder.runrecord import read_notified
    return [e["event"] for e in read_notified(run.attempt(stage, n))]


def told(run: RealRun, event: str, stage: str, n: int = 0) -> list:
    """WHAT RUN ``n`` OF ``stage`` SAID AT ``event``, in order: the summary
    lines its ``notify`` verb wrote into the run's own monitor log, the
    record every event leaves whether or not a channel is set up
    (`run-reports.md` § 2.5) -- named as the run's files are
    (`monitor.WatchedRun`)."""
    import re
    from molbuilder.jobset.naming import stage_home
    from molbuilder.monitor import WatchedRun
    from molbuilder.runfiles import FIRST_TRY
    from molbuilder.task import of as task_of
    task = task_of(run.bundle)
    names = stage_home(run.bundle, task, stage).names
    log = WatchedRun(names=names, run=FIRST_TRY,
                     directory=run.attempt(stage, n)).path(".monitor.log")
    said = re.compile(r"\[NOTIFY\] \(stub\) (\w+): (.*)$")
    return [m.group(2) for m in map(said.search,
                                   log.read_text(encoding="utf-8")
                                   .splitlines())
            if m and m.group(1) == event]


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
               engine: str = "siesta", **blocks) -> Path:
    """`molbuilder describe` of the road's ``structure`` (`support.road`,
    written where `describe` takes it), on ``engine``, the
    ``calculation``'s own ladder or ``stage_strategy``; then the job
    request's blocks as given -- its run card, its allocation, its
    benchmark: each stated, or left out.  THE ONE MAKER of a calculation on
    the road with the engine."""
    from molbuilder.workingcopy_structure import StructureCodec
    from support.road import road_structure
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    source = f"P/structure/{name.lower()}.xyz"
    StructureCodec().write(road_structure(structure), tree / source)
    r = molbuilder("describe", "--structure", source, "--bundle", where,
               "--engine", engine, "--calculation", calculation,
               "--name", name,
               *(("--psml-lib", "pseudopotential") if engine == "siesta"
                 else ()),
               *(("--stage-strategy", stage_strategy) if stage_strategy
                 else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / where
    from support.road import read_request, write_request
    request = read_request(bundle)
    for block, value in blocks.items():
        if value is None:
            request.pop(block, None)
        else:
            request[block] = value
    write_request(bundle, request)
    return bundle


def _script_dry_runs(run: RealRun, prepared: Path, scratch: Path,
                     **given: str) -> None:
    """The run script's own dry run (`running-a-job.md` § 3.4), on a copy of
    a prepared folder so no run another test reads gains a log: as prepared,
    and with ``-np 2`` typed on it.  The rank and thread counts the shell
    may carry are scrubbed, so what resolves is the script's own chain;
    ``given`` is what the job is given there -- its GPUs, as a scheduler
    gives them (``CUDA_VISIBLE_DEVICES``)."""
    there = scratch / prepared.name
    shutil.copytree(prepared, there)
    script = next(there.glob("*.run.sh"))
    for step, extra in (("script dry run", ()),
                        ("script dry run -np 2", ("-np", "2"))):
        done = subprocess.run(["bash", str(script), "--run", "0",
                               "--dry-run", *extra],
                              cwd=there, capture_output=True, text=True,
                              timeout=120, env=_scrubbed(**given))
        run.said[step] = Said(done.returncode, done.stdout + done.stderr)


#: What a shell may carry into a run: the counts a person sets, which the run
#: script's chains read; what a scheduler tells a job in molbuilder's names,
#: which they read too; in SLURM's own names (`scheduler.md` § 6a), which a
#: header maps; and the GPUs a job is given -- so none reaches a dry run but
#: what a step sets.
_CARRIED = ("OMP_NUM_THREADS", "MB_NP", "MOLBUILDER_USE_MPS", "MB_JOB_ID",
            "MB_NTASKS", "MB_CPUS_PER_TASK", "MB_CPUS_ON_NODE", "MB_JOB_GPUS",
            "MB_JOB_NODELIST", "SLURM_JOB_ID", "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_GPUS",
            "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")


def _scrubbed(**more) -> Dict[str, str]:
    """This process's environment without what :data:`_CARRIED` names, the
    launch claim made by hand, and ``more``."""
    env = {k: v for k, v in os.environ.items() if k not in _CARRIED}
    return {**env, "MB_LAUNCHED_BY": "manual", **more}


def _header_under_a_slurm_grant(run: RealRun, tree: Path,
                                scratch: Path) -> None:
    """THE HEADER PREP WRITES FOR A SLURM MACHINE, started as SLURM starts
    it -- ``bash <its .sbatch> --run 0 --dry-run``, 2 ranks granted
    (``SLURM_NTASKS``, the variable `scheduler.md` § 6a names) -- on a copy
    of a sibling H2 prepared for ``sol``: this machine's record as a SLURM
    machine with one queue, entering its environments the same way, so the
    run script activates here.  Said under ``header dry run under a SLURM
    grant of 2 ranks``; the named record is taken away after, so no later
    prep is asked which machine it means."""
    import dataclasses
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import (Domain, environments_dir,
                                      machine_scope_path,
                                      named_environment_path,
                                      read_environment, write_environment)
    from support.machines import SLURM
    sol = named_environment_path("sol")
    ensure_private_dir(environments_dir())
    write_environment(dataclasses.replace(
        read_environment(machine_scope_path()), scheduler=SLURM,
        domains=[Domain.from_row({
            "name": "cpu", "partition": "cpu", "qos": "public",
            "max_time": "1-00:00:00", "max_cores": 64,
            "node_types": [{"cores": 64, "nodes": 1}]})]), sol)
    try:
        bundle = _h2_optimization(
            tree, "P/opt/H2-on-sol",
            execution={"mpi_np": 1, "omp_threads": 1},
            allocation={"domain": "cpu", "time": "01:00:00", "mem": "1G"})
        on_sol = RealRun(tree=tree, bundle=bundle)
        _taken(on_sol, "prep coarse for sol", "prep", "task", "--group",
               "1", "--target", "sol")
    finally:
        sol.unlink()
    there = scratch / "on-sol"
    shutil.copytree(on_sol.attempt("coarse"), there)
    header = next(there.glob("*.sbatch"))
    done = subprocess.run(["bash", header.name, "--run", "0", "--dry-run"],
                          cwd=there, capture_output=True, text=True,
                          timeout=120, env=_scrubbed(SLURM_NTASKS="2"))
    run.said["header dry run under a SLURM grant of 2 ranks"] = Said(
        done.returncode, done.stdout + done.stderr)


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
        _script_dry_runs(run, run.attempt("coarse"),
                         tmp_path_factory.mktemp("dry-run"))
        _header_under_a_slurm_grant(run, tree,
                                    tmp_path_factory.mktemp("header-dry-run"))
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
        run.files_before = _all_written(bundle)
        _taken(run, "dry run of launching coarse again", "launch", "task",
               "--stage", "coarse", "--mode", "direct", "--dry-run", "--yes")
        run.files_after = _all_written(bundle)
        _taken(run, "launch coarse again", "launch", "task", "--stage",
               "coarse", "--mode", "direct", "--foreground", "--yes")
        # ASKED AT A TERMINAL, AND ANSWERED THE OTHER WAY: the foreground.
        _answered(run, "launch coarse again cold", "f", "launch", "task",
                  "--stage", "coarse", "--mode", "direct", "--cold")
        run.relaunched = _all_written(bundle)
        _taken(run, "prep medium", "prep", "task", "--group", "2",
               "--target", "this")
        # A RUN HERE, LEFT IN THE BACKGROUND -- a launch's default
        # (`job-system.md` § 6.0) -- and waited for.
        _taken(run, "launch naming no stage", "launch", "task", "--mode",
               "direct", "--yes")
        _left_to_end(run)
        # ...AND THE JOBS LAUNCHED FROM HERE, asked once it ended, and again
        # (`job-system.md` § 6.0a).
        run.jobs["after it ended"] = molbuilder("jobs", "status")
        run.jobs["again"] = molbuilder("jobs", "status")
        _taken(run, "status", "status")
    return run


def gpu(tmp_path_factory) -> RealRun:
    """A SIESTA relaxation on this machine's GPU (`execution/gpu.md`): the
    road's H2 asking one GPU, one rank -- this machine's record stating one
    GPU and how its shell enters an environment, the run entering the
    source-built `molbuilder-siesta-gpu`.  Its run script's own dry run as
    prepared, and with ``-np 2`` -- each given GPU 0, as a job asking one GPU
    is given one -- then the run, waited on."""
    with _on_the_road(tmp_path_factory, "gpu") as tree:
        from conftest import write_machine_record
        from molbuilder.scheduler import Topology
        from _road import conda_hook
        write_machine_record(
            topology=Topology(sockets=2, cores_per_socket=8, gpus_per_node=1),
            env_init={"activation": "conda activate",
                      "preamble": f"source {conda_hook()}"})
        bundle = _h2_optimization(tree, "P/opt/H2gpu", execution={
            "mpi_np": 1, "omp_threads": 1, "use_gpu": True, "gpu_count": 1})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep coarse", "prep", "task", "--group", "1",
               "--target", "this")
        _script_dry_runs(run, run.attempt("coarse"),
                         tmp_path_factory.mktemp("gpu-dry-run"),
                         CUDA_VISIBLE_DEVICES="0")
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
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
               "G0N1C1", "--mode", "direct", "--foreground", "--yes")
        _taken(run, "launch the benchmark", "launch", "bench", "coarse",
               "--mode", "direct", "--foreground", "--yes", "--trial-timeout", "5")
    return run


#: THE ROAD CO2'S OWN SETTINGS, set by the person: SIESTA's own mixing
#: weight and a 150 Ry mesh -- every CO2 run 2.4 times faster than at the
#: catalogue's (0.02, 300 Ry), its four modes and the bends' degeneracy
#: kept (measured 2026-10-23: one force-constant run 216 s -> 89 s).  The
#: road tests the mechanism; no test holds these frequencies to a
#: literature value.
CO2_SETTINGS = {"mixing_weight": 0.25, "mesh_cutoff": 150.0}

#: THE OPTICAL PHASE ASKED FOR, on the shared run: Compute Raman, for the
#: modes above 1000 cm-1 -- the two stretches, modes 3 and 4 of four, 1-based
#: (`spectra.selection.select_modes`); the bends sit near 620 cm-1 at this
#: mesh -- the one whose oxygens move apart and the one whose oxygens move
#: together, so the Raman view can be held to CO2's symmetry; so the run's
#: optical calls are five, the equilibrium and each mode's two sides; the
#: optical mesh the one point an isolated molecule's box needs.
CO2_OPTICAL = {"compute_raman": True, "raman_modes": "range",
               "raman_freq_min_cm1": 1000.0, "optical_mesh": [1, 1, 1]}


def co2_vibration(tree: Path, where: str = "P/freq/CO2", *,
                  optical: bool = False) -> Path:
    """The road's CO2 (`support.road.CO2`) described as a SIESTA vibration
    in ``tree`` at ``where``: the kind's one stage, `freq`, one rank, one
    thread (:func:`_described`), its template carrying :data:`CO2_SETTINGS`
    as the person's (`template.with_values_set`) -- and, ``optical``,
    :data:`CO2_OPTICAL` -- THE ONE MAKER of it, for the shared run and for
    a test's own calculation alike."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.template import with_values_set
    from molbuilder.template_file import template_path
    from support.road import CO2
    bundle = _described(tree, where, structure=CO2, name="CO2",
                        calculation="vibration",
                        execution={"mpi_np": 1, "omp_threads": 1})
    path = template_path(bundle, "CO2")
    path.write_text(with_values_set(
        path.read_text(), SiestaConfig, calculation="vibration",
        values={**CO2_SETTINGS, **(CO2_OPTICAL if optical else {})},
        source="person"))
    return bundle


def vibration(tmp_path_factory) -> RealRun:
    """The vibration kind's one run in phases of the road's CO2
    (`engines/vibration.md` § 5.11): the one stage `freq`, one rank, one
    thread, its optical phase asked for the two stretches
    (:data:`CO2_OPTICAL`), prepared and launched here, waited on -- its
    `relax` call, its `fc` call and the five optical calls, the equilibrium
    and each stretch's two sides, each a point of the run."""
    with _on_the_road(tmp_path_factory, "vibration") as tree:
        run = RealRun(tree=tree, bundle=co2_vibration(tree, optical=True))
        _taken(run, "prep freq", "prep", "task", "--group", "1",
               "--target", "this")
        _taken(run, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--foreground", "--yes")
    # IT REACHED ITS END, by the one status door: every reader of the run
    # reads a run that finished, every call of it done.
    status = stage_status(run.bundle, "freq")
    assert status.state == "finished", (status.state, status.detail)
    assert all(p["done"] for p in status.points), status.points
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


def pyscf(tmp_path_factory) -> RealRun:
    """An H2 optimization on PySCF -- its run `01_coarse/run-0/`
    (:meth:`RealRun.attempt`) -- one thread, the bond stretched to 0.80 A so the optimization
    takes steps: `coarse` prepared, launched and waited on.  Each module that
    reads a finished PySCF run reads this one (`process/testing.md` tier 3)."""
    from support.road import H2
    stretched = {**H2, "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.80]]}
    with _on_the_road(tmp_path_factory, "pyscf", engine="pyscf") as tree:
        bundle = _described(tree, "P/opt/H2", structure=stretched, name="H2",
                            calculation="optimization", engine="pyscf",
                            execution={"threads": 1})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep coarse", "prep", "task", "--group", "1",
               "--target", "this")
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
    return run
