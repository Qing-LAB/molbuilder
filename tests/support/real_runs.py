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
  (`support.road.CO2`; plan T-CO2);
* :func:`gpu` (``real_h2_gpu``) -- the same H2 asking this machine's GPU,
  one rank: its run script's dry runs, then the run, waited on;
* :func:`pyscf` (``real_h2_pyscf``) -- an H2 optimization on PySCF, flat,
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
from support.road import molbuilder


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
        doors (`jobset.naming.stage_home`, `paths.attempt_dir`): the
        handle every reader of the run takes, asking the run door for the
        rest (`runs.run_of`)."""
        from molbuilder.jobset.naming import stage_home
        from molbuilder.paths import attempt_dir
        from molbuilder.runfiles import TASK_FILE
        from molbuilder.task import of as task_of
        task = task_of(self.bundle)
        return attempt_dir(stage_home(self.bundle, task, stage).dir, n)


#: Each engine's environment, and the road a run of it is made on.
_ENGINES = {"siesta": "molbuilder-siesta", "pyscf": "molbuilder-pySCF"}


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
               engine: str = "siesta", shape: str = "hierarchical",
               **blocks) -> Path:
    """`molbuilder describe` of the road's ``structure`` (`support.road`,
    written where `describe` takes it), on ``engine`` in ``shape``, the
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
               "--shape", shape, "--name", name,
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


def _files(bundle: Path) -> Dict[str, tuple]:
    """Every file under the calculation, by its size and write time."""
    return {str(p.relative_to(bundle)): (p.stat().st_size,
                                         p.stat().st_mtime_ns)
            for p in sorted(bundle.rglob("*")) if p.is_file()}


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
    shutil.copytree(bundle / "01_coarse" / "run-0", there)
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
        _script_dry_runs(run, bundle / "01_coarse" / "run-0",
                         tmp_path_factory.mktemp("dry-run"))
        _header_under_a_slurm_grant(run, tree,
                                    tmp_path_factory.mktemp("header-dry-run"))
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
        _script_dry_runs(run, bundle / "01_coarse" / "run-0",
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
               "G0K1C1", "--mode", "direct", "--foreground", "--yes")
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


def co2_vibration(tree: Path, where: str = "P/freq/CO2") -> Path:
    """The road's CO2 (`support.road.CO2`) described as a SIESTA vibration
    in ``tree`` at ``where``: the kind's own ladder (`relax` then `freq`), one
    rank, one thread (:func:`_described`), its template carrying
    :data:`CO2_SETTINGS` as the person's (`template.with_values_set`) --
    THE ONE MAKER of it, for the shared relaxation and for a test's own
    calculation alike."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.template import with_values_set
    from molbuilder.template_file import template_path
    from support.road import CO2
    bundle = _described(tree, where, structure=CO2, name="CO2",
                        calculation="vibration",
                        execution={"mpi_np": 1, "omp_threads": 1})
    path = template_path(bundle, "CO2")
    path.write_text(with_values_set(path.read_text(), SiestaConfig,
                                    calculation="vibration",
                                    values=CO2_SETTINGS, source="person"))
    return bundle


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


def pyscf(tmp_path_factory) -> RealRun:
    """An H2 optimization on PySCF, flat -- the run is the calculation's own
    folder -- one thread, the bond stretched to 0.80 A so the optimization
    takes steps: `coarse` prepared, launched and waited on.  Each module that
    reads a finished PySCF run reads this one (`process/testing.md` tier 3)."""
    from support.road import H2
    stretched = {**H2, "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.80]]}
    with _on_the_road(tmp_path_factory, "pyscf", engine="pyscf") as tree:
        bundle = _described(tree, "P/opt/H2", structure=stretched, name="H2",
                            calculation="optimization", engine="pyscf",
                            shape="flat", execution={"threads": 1})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep coarse", "prep", "task", "--group", "1",
               "--target", "this")
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
    return run
