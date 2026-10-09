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
  on it and sent by a launch that names no stage; and the run script's own
  dry runs, on a copy of `coarse`'s prepared folder;
* :func:`bench` (``real_h2_bench``) -- a two-trial benchmark of the same
  H2's `coarse`: one trial launched by name, then the rest by a launch of
  the benchmark under a per-trial bound;
* :func:`relaxed_for_vibration` (``real_h2_relaxed_for_vibration``) -- the
  vibration kind's `relax` stage of a held H2
  (`_road.h2_relaxed_for_vibration`).
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

import numpy as np
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
    the calculation's files around the one step that must write nothing."""
    tree: Path
    bundle: Path
    said: Dict[str, Said] = field(default_factory=dict)
    files_before: Dict[str, tuple] = field(default_factory=dict)
    files_after: Dict[str, tuple] = field(default_factory=dict)


def _engine_here() -> bool:
    from _road import conda_hook, env_available
    return conda_hook().is_file() and env_available("molbuilder-siesta")


@contextmanager
def _on_the_road(tmp_path_factory, name: str):
    """A projects tree of its own, the one door pointed at it, and the real
    SIESTA on the road -- while the runs are made; the process environment
    as it was, after."""
    from _pytest.monkeypatch import MonkeyPatch

    from _road import live_siesta
    from conftest import _point_the_one_door_at
    if not _engine_here():
        pytest.skip("needs the molbuilder-siesta env + a detectable conda hook")
    mp = MonkeyPatch()
    try:
        tree = _point_the_one_door_at(tmp_path_factory.mktemp(name), mp.setenv)
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


def _write_h2(tree: Path) -> str:
    """The road's H2, its first atom held, in a 10 Å box isolated on every
    axis: the path `init` takes."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([10.0] * 3), axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")
    return "P/structure/h2.xyz"


def _described(tree: Path, where: str, **blocks) -> Path:
    """`jobset init` of the H2, SIESTA, layered folders, the `publishable`
    ladder; then the description's run card and benchmark as given -- each
    block stated, or left out."""
    from support.road import jobset
    r = jobset("init", "--structure", _write_h2(tree), "--bundle", where,
               "--engine", "siesta", "--calculation", "optimization",
               "--shape", "hierarchical", "--name", "H2",
               "--psml-lib", "pseudopotential",
               "--stage-strategy", "publishable")
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


def layered(tmp_path_factory) -> RealRun:
    """Run A of plan § 5y -- see the module's note."""
    with _on_the_road(tmp_path_factory, "layered") as tree:
        bundle = _described(tree, "P/opt/H2",
                            execution={"mpi_np": 1, "omp_threads": 1})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep coarse", "prep", "task", "--stage", "coarse",
               "--target", "this")
        _script_dry_runs(run, bundle / "01_coarse" / "run-0",
                         tmp_path_factory.mktemp("dry-run"))
        _taken(run, "launch coarse", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--yes")
        run.files_before = _files(bundle)
        _taken(run, "dry run of launching coarse again", "launch", "task",
               "--stage", "coarse", "--mode", "direct", "--dry-run", "--yes")
        run.files_after = _files(bundle)
        _taken(run, "launch coarse again", "launch", "task", "--stage",
               "coarse", "--mode", "direct", "--yes")
        _taken(run, "launch coarse again cold", "launch", "task", "--stage",
               "coarse", "--mode", "direct", "--yes", "--cold")
        _taken(run, "prep medium", "prep", "task", "--stage", "medium",
               "--target", "this")
        _taken(run, "launch naming no stage", "launch", "task", "--mode",
               "direct", "--yes")
        _taken(run, "status", "status")
    return run


def bench(tmp_path_factory) -> RealRun:
    """The benchmark of plan § 5y -- see the module's note."""
    with _on_the_road(tmp_path_factory, "bench") as tree:
        bundle = _described(tree, "P/opt/H2bench", execution=None,
                            bench={"mpi_np": [1, 2], "omp_threads": [1]})
        run = RealRun(tree=tree, bundle=bundle)
        _taken(run, "prep bench", "prep", "bench", "coarse",
               "--target", "this")
        _taken(run, "launch one trial", "launch", "bench", "coarse",
               "G0K1C1", "--mode", "direct", "--yes")
        _taken(run, "launch the benchmark", "launch", "bench", "coarse",
               "--mode", "direct", "--yes", "--trial-timeout", "5")
    return run


def relaxed_for_vibration(tmp_path_factory):
    """The vibration kind's `relax` stage of a held H2, run: the bundle."""
    from _road import h2_relaxed_for_vibration
    with _on_the_road(tmp_path_factory, "vibration") as tree:
        bundle = h2_relaxed_for_vibration(tree)
    return bundle
