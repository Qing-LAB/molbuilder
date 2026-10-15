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
* :func:`relaxed_for_vibration` (``real_relaxed_for_vibration``) -- the
  vibration kind's `relax` stage of the road's CO2, listed O O C with the
  carbon held, so its decks are in the engine's held-first order
  (`_road.co2_relaxed_for_vibration`; plan T-CO2).

A run with one thing changed is a copy of it beside it (:func:`copy_of`): its
decks as an older molbuilder wrote them, a damaged order record, its held set
grown and its next stage prepared on the road.
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
        _taken(run, "prep coarse", "prep", "task", "--group", "1",
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
        _taken(run, "prep medium", "prep", "task", "--group", "2",
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


def relaxed_for_vibration(tmp_path_factory) -> RealRun:
    """The vibration kind's `relax` stage of the road's CO2, run."""
    from _road import co2_relaxed_for_vibration
    from molbuilder.runs import run_answer
    with _on_the_road(tmp_path_factory, "vibration") as tree:
        run = RealRun(tree=tree, bundle=co2_relaxed_for_vibration(tree))
    # IT REACHED ITS END, by the one status door: every reader of the run
    # reads a relaxation that finished.
    said = run_answer(run.attempt("relax"))
    assert (said["state"], said["detail"]) == ("finished", "job_completed"), said
    return run


#: The copies whose change succeeded, this session (:func:`copy_of`).
_MADE: set = set()


def copy_of(run: RealRun, name: str, alter) -> RealRun:
    """THE SHARED RUN WITH ONE THING CHANGED: a copy of the calculation
    beside it, ``<bundle>-<name>``, altered once by ``alter(copy)`` -- made
    once a session, the run's own files left as they are.  A copy counts as
    made once its change succeeded; one a failed change left behind is made
    again.  Every test that reads the run with one change asks it."""
    copy = RealRun(tree=run.tree,
                   bundle=run.bundle.parent / f"{run.bundle.name}-{name}")
    if copy.bundle not in _MADE:
        shutil.rmtree(copy.bundle, ignore_errors=True)
        shutil.copytree(run.bundle, copy.bundle)
        alter(copy)
        _MADE.add(copy.bundle)
    return copy


def relax_decks(run: RealRun):
    """The decks of the `relax` stage's run -- the one it read and the
    stage's own copy (`runs.Run.deck`, `stage_deck`)."""
    from molbuilder.runs import run_of
    ran = run_of(run.attempt("relax"))
    return sorted({ran.deck, ran.stage_deck})


def with_atom_metadata(deck: Path, change) -> None:
    """``deck``'s ATOM-METADATA block read by its reader
    (`script_emit.read_atom_metadata_block`), ``change(payload)`` applied,
    and written back between the block's own fences (`deck_record`) -- as
    an older molbuilder stated it, or as a damaged deck does, the way the pin
    table rewrites a sidecar."""
    from molbuilder.deck_record import (BLOCK_ATOM_METADATA, begin_marker,
                                        end_marker)
    from molbuilder.script_emit import read_atom_metadata_block
    text = deck.read_text()
    payload = read_atom_metadata_block(text)
    change(payload)
    head, rest = text.split(begin_marker(BLOCK_ATOM_METADATA), 1)
    _old, tail = rest.split(end_marker(BLOCK_ATOM_METADATA), 1)
    deck.write_text(head + "\n".join([
        begin_marker(BLOCK_ATOM_METADATA),
        f"# format: molstruct-json/v{payload['schema_version']}",
        f"# {json.dumps(payload)}",
        end_marker(BLOCK_ATOM_METADATA)]) + tail)


#: WHAT A DECK FROM BEFORE 2.10.0 IS SAID TO BE (`model/overview.md` § 2.2a,
#: I9) -- the version :func:`as_prepared_by_2_9_6` stamps, the release the
#: contract names, and its way out.
PREPARED_BY_2_9_6 = ("prepared by molbuilder 2.9.6",
                     "before 2.10.0 recorded the order a deck's atoms are in",
                     "prepare the stage again")


def as_prepared_by_2_9_6(run: RealRun) -> RealRun:
    """A copy of the calculation (:func:`copy_of`), every deck of its `relax`
    stage as molbuilder 2.9.6 wrote it -- that version's stamp, and its
    ATOM-METADATA block at schema 13, with no order record
    (`model/overview.md` § 2.2a, I9), as the pin table rewrites a sidecar as
    an older molbuilder wrote it."""
    import re

    def as_2_9_6_wrote_it(copy):
        for deck in relax_decks(copy):
            with_atom_metadata(deck, lambda block: (
                block.pop("atom_order"), block.update(schema_version=13)))
            deck.write_text(re.sub(r"(?m)^(#[ \t]*generator-version[ \t]+)\S+",
                                   r"\g<1>2.9.6", deck.read_text()))
    return copy_of(run, "before-2.10.0", as_2_9_6_wrote_it)


#: THE HELD SET GROWN after `relax` ran (`engines/vibration.md` § 2.3a): the
#: carbon and the first oxygen, where `relax` held the carbon alone.
HELD_GROWN = [0, 2]


def held_grown(run: RealRun, tmp_path_factory) -> RealRun:
    """A copy of the calculation (:func:`copy_of`), its structure's held set
    grown (:data:`HELD_GROWN`) as a person would on the Molbuilder tab, and
    its `freq` stage prepared on the road -- the force-constant prep built
    from a relaxation that held other atoms (plan V1.41)."""
    from molbuilder.runfiles import TASK_FILE
    from molbuilder.task import read_task
    from molbuilder.workingcopy_structure import StructureCodec

    def grow_and_prepare(copy):
        task = read_task(copy.bundle / TASK_FILE)
        source = copy.bundle / Path(task.structure.source).name
        mine = StructureCodec().load(source)
        StructureCodec().write(
            mine.replace(regions={**mine.regions,
                                  "frozen_atoms": list(HELD_GROWN)}),
            source)
        with _road_at(copy.tree.parent, tmp_path_factory):
            _taken(copy, "prep freq", "prep", "task", "--group", "2",
                   "--target", "this")
    return copy_of(run, "held-grown", grow_and_prepare)

