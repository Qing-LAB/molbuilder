"""A structure saved out of the Results tab reloads where the engine had it
(plan § 5q.4, T3): the engine's coordinates WITH the engine's origin -- a
stated offset of 0 -- "so its next treatment applies nothing and the box
stays where the engine had it" (`model/structure-periodicity.md` § 6.0).

§ 6.0 asks it of "every door that makes a structure from an engine's output
... and every save of a run's output", and the Results tab reaches a run's
output two ways, which take the frame by different routes: the TRAJECTORY
viewer -- a SIESTA run's cell from its own output, a PySCF run's from the cell
its deck recorded -- and the STRUCTURE preview, which opens an engine's own
geometry file (SIESTA's ``<label>.xyz``, written with no sidecar) through the
codec.  One case per route.

The runs are made on the road -- ``jobset init`` -> ``prep task`` -> ``launch
run --mode direct`` -- and saved as a person saves them
(`support/results_export.py`).  Each saved pair is read back through the
codec every load goes through and compared with the engine's own statement of
its frame: the output as the Results door parses it (``parse.detect``), or
the record in the deck it ran.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, env_bin
from support.results_export import save_to_project, wait_for_viewer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONDA_SH = conda_hook()

pytestmark = pytest.mark.engine


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _described(tree, name, engine, shape):
    """``jobset init`` of an H2 optimisation, activated through the conda
    hook; the bundle it made."""
    r = _jobset("init", "--structure", f"P/structure/{name}.xyz",
                "--bundle", f"P/opt/{name}", "--engine", engine,
                "--shape", shape, "--calculation", "optimization",
                "--name", "H2", *(("--psml-lib", "pseudopotential")
                                  if engine == "siesta" else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "opt" / name
    # How a shell enters conda HERE is this machine's record's to say -- and
    # the run states its shape (`architecture.md` § 5.2).
    from conftest import write_machine_record
    write_machine_record(env_init={
        "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {**task.get("execution", {}),
                         **({"mpi_np": 1, "omp_threads": 1}
                            if engine == "siesta" else {"threads": 1})}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def _ran(bundle, run_dir, *prep_args):
    """``prep task`` then ``launch task --mode direct``, and the run FINISHED --
    asked of the run's own door, because a direct launch whose job failed
    still exits 0 (plan W40)."""
    from molbuilder.parse.dirs import run_status
    r = _jobset("prep", "task", "--stage", "coarse", "--bundle", bundle,
                "--target", "this", *prep_args)
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
                "--mode", "direct", "--yes")
    st = run_status(run_dir, "H2_01_coarse")    # the run's stem
    assert st.state == "finished", (st.state, st.detail, r.output[-2000:])


@pytest.fixture(scope="module")
def road(isolated_projects_root_module, tmp_path_factory):
    """The projects tree, this module's config and machine record, and the
    folder a person runs ``jobset`` from."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    tree = isolated_projects_root_module
    mp = pytest.MonkeyPatch()
    try:
        # The suite's config rule, which its autouse fixture applies per test
        # and so not to a module's fixture (`conftest`).
        mp.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
        mp.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
        from conftest import write_machine_record
        write_machine_record()
        mp.chdir(tree.parent)
        for name, bond in (("siesta", 0.741), ("pyscf", 0.80)):
            StructureCodec().write(
                Structure(elements=["H", "H"],
                          positions=np.array([[5.0, 5.0, 5.0],
                                              [5.0, 5.0, 5.0 + bond]])),
                tree / "P" / "structure" / f"{name}.xyz")
        yield tree, mp
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def siesta_run(road):
    """The attempt directory of a finished SIESTA H2 relaxation."""
    if not (CONDA_SH.is_file() and env_available("molbuilder-siesta")):
        pytest.skip("needs the molbuilder-siesta env + a detectable conda hook")
    tree, _mp = road
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml",
                tree / "pseudopotential" / "H.psml")
    bundle = _described(tree, "siesta", "siesta", "hierarchical")
    run_dir = bundle / "01_coarse" / "run-0"
    # The SIESTA binary on PATH for THIS run only: left in place it reaches
    # the next run too, whose wrapper then finds this env's python.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", f"{env_bin('molbuilder-siesta')}{os.pathsep}"
                          f"{os.environ['PATH']}")
        _ran(bundle, run_dir, "--np", 1)
    return run_dir


@pytest.fixture(scope="module")
def pyscf_run(road):
    """The folder of a finished PySCF H2 optimisation -- flat, so the run is
    the calculation's own folder."""
    if not (CONDA_SH.is_file() and env_available("molbuilder-pySCF")):
        pytest.skip("needs the molbuilder-pySCF env + a detectable conda hook")
    tree, _mp = road
    bundle = _described(tree, "pyscf", "pyscf", "flat")
    _ran(bundle, bundle)
    return bundle


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, flask_server, monkeypatch, folder, pick):
    """The Results tab on ``folder`` with ``pick`` chosen from its list --
    the folder set the way a returning tab finds it."""
    from molbuilder import diagnostics
    root = next(p for p in Path(folder).parents if p.name == "projects")
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "road"),))
    page.add_init_script(
        "try { sessionStorage.setItem('molbuilder.current_dir', "
        f"{json.dumps(str(folder))}); }} catch (_) {{}}")
    page.goto(f"{flask_server}/results")
    page.wait_for_function(
        "(want) => [...document.querySelectorAll("
        "  '#results-file-picker-select option')].some(o => o.value === want)",
        arg=str(pick), timeout=30000)
    page.select_option("#results-file-picker-select", value=str(pick))
    return root


def _as_the_door_reads(output):
    """The frames and the cell of ``output``, through the parser the
    Results door picks for it."""
    from molbuilder.parse import detect
    traj = detect(Path(output)).parse(Path(output))
    frames = [f for f in traj.frames if f.structure is not None]
    assert frames, f"{output} holds no geometry"
    return frames, traj.lattice


def _the_decks_cell(run_dir):
    """The cell the deck in ``run_dir`` recorded placing the atoms in."""
    from molbuilder.runfiles import find_by_role
    from molbuilder.deck_record import extract_engine_offset
    decks = find_by_role(Path(run_dir), ".py") + find_by_role(Path(run_dir),
                                                              ".fdf")
    records = [extract_engine_offset(d.read_text()) for d in decks]
    records = [r for r in records if r]
    assert records, f"no deck in {run_dir} records its placement"
    return records[0]["cell"]


def _holds_the_engines_frame(saved, cell, positions):
    from molbuilder.workingcopy_structure import StructureCodec
    back = StructureCodec().read(saved)
    assert back.engine_offset is not None, (
        "the saved pair states no origin, so its next treatment re-derives "
        "one and moves the box off where the engine had it")
    assert np.allclose(back.engine_offset, 0.0), back.engine_offset
    assert back.cell is not None, "the saved pair carries no cell"
    assert np.allclose(back.cell, cell, atol=1e-5), (back.cell, cell)
    assert np.allclose(back.positions, positions, atol=1e-5), (
        "not the engine's own coordinates")


def test_a_siesta_trajectory_saves_the_box_its_output_states(
        siesta_run, page, flask_server, monkeypatch):
    """The trajectory viewer on a SIESTA relaxation: the cell is the one the
    output states, the origin a stated 0 -- and the name offered is the
    output's own (`web/molview.md`: open ``runs/mine.xyz`` and an Export
    writes ``mine.xyz``), with the frame it holds.

    MUTATIONS THIS MUST FAIL AGAINST: the one composer not stating the
    engine's origin (`runs.Declared.frame`);
    the viewer naming its install after the parser's label, which offered
    " .log_frame6" and saved a hidden ``.log_frame6.xyz`` until 2026-09-27.
    """
    out = siesta_run / "H2_01_coarse-run0.out"
    frames, cell = _as_the_door_reads(out)
    assert len(frames) > 1, "a relaxation with one frame asks no frame range"
    root = _open(page, flask_server, monkeypatch, siesta_run, out)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    offered = save_to_project(page, "siesta-trajectory", frames=len(frames))
    assert offered == f"{out.stem}_frame{len(frames)}", offered
    _holds_the_engines_frame(root / "siesta-trajectory.xyz", cell,
                             frames[-1].structure.positions)


def test_the_engines_own_geometry_file_saves_the_box_the_run_had(
        siesta_run, page, flask_server, monkeypatch):
    """SIESTA's ``<label>.xyz`` -- its final coordinates, written with no
    sidecar -- opened in the structure preview and saved: the pair carries
    the box the run had and a stated 0, as the output states it.

    MUTATION THIS MUST FAIL AGAINST: the codec reading an engine's own file
    with no frame (the no-sidecar branch of `StructureCodec.load`), which
    saved it with no cell and no origin until 2026-09-27.
    """
    from molbuilder.structure import Structure
    own = siesta_run / "H2.xyz"
    _frames, cell = _as_the_door_reads(siesta_run / "H2_01_coarse-run0.out")
    positions = Structure.from_xyz(own.read_text()).positions
    root = _open(page, flask_server, monkeypatch, siesta_run, own)
    wait_for_viewer(page, atoms=2, frames=1)
    save_to_project(page, "siesta-own", frames=1)
    _holds_the_engines_frame(root / "siesta-own.xyz", cell, positions)


def test_a_pyscf_trajectory_saves_the_box_its_deck_recorded(
        pyscf_run, page, flask_server, monkeypatch):
    """The trajectory viewer on a PySCF optimisation, whose output states no
    cell: the box is the one its deck recorded placing the atoms in.

    MUTATION THIS MUST FAIL AGAINST: the one composer ignoring the deck
    record's cell when the output carries none.
    """
    log = next(pyscf_run.glob("*.molwatch.log"))
    frames, _no_cell = _as_the_door_reads(log)
    assert len(frames) > 1, "an optimisation with one frame asks no range"
    cell = _the_decks_cell(pyscf_run)
    root = _open(page, flask_server, monkeypatch, pyscf_run, log)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    save_to_project(page, "pyscf-trajectory", frames=len(frames))
    _holds_the_engines_frame(root / "pyscf-trajectory.xyz", cell,
                             frames[-1].structure.positions)
