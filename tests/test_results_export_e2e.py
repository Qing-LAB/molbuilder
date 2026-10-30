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

The runs are the pass's own, made once on the road -- ``molbuilder describe``
-> ``prep task`` -> ``launch task --mode direct`` (`support/real_runs.py`) --
and saved as a person saves them (`support/results_export.py`).  Each saved pair is read back through the
codec every load goes through and compared with the engine's own statement of
its frame: the output as the Results door parses it (``parse.detect``), or
the record in the deck it ran.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from support.results_export import save_to_project, wait_for_viewer

pytestmark = pytest.mark.engine


@pytest.fixture(scope="module")
def siesta_run(real_h2_layered):
    """The run folder of a finished SIESTA H2 relaxation -- the
    shared one's first `coarse` run (`support.real_runs.layered`), made once
    a pass (`process/testing.md` tier 3)."""
    return real_h2_layered.attempt("coarse")


@pytest.fixture(scope="module")
def pyscf_run(real_h2_pyscf):
    """The run folder of a finished PySCF H2 optimisation -- its `coarse`
    stage's first run (`support.real_runs.pyscf`)."""
    return real_h2_pyscf.attempt("coarse")


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
    """The cell the run's deck (`runs.Run.deck`) recorded placing the atoms
    in."""
    from molbuilder.deck_record import extract_engine_offset
    from molbuilder.runs import run_of
    deck = run_of(run_dir).deck
    record = extract_engine_offset(deck.read_text()) if deck else None
    assert record, f"no deck in {run_dir} records its placement"
    return record["cell"]


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
    from molbuilder.runs import run_of
    out = run_of(siesta_run).stdout
    frames, cell = _as_the_door_reads(out)
    assert len(frames) > 1, "a relaxation with one frame asks no frame range"
    root = _open(page, flask_server, monkeypatch, siesta_run, out)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    offered = save_to_project(page, "siesta-trajectory", frames=len(frames))
    assert offered == f"{out.stem}_frame{len(frames) - 1}", offered
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
    from molbuilder.runs import run_of
    from molbuilder.structure import Structure
    run = run_of(siesta_run)
    own = run.on_label(".xyz")
    _frames, cell = _as_the_door_reads(run.stdout)
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
    from molbuilder.runs import run_of
    log = run_of(pyscf_run).file(".molwatch.log")
    assert log is not None, sorted(p.name for p in pyscf_run.iterdir())
    frames, _no_cell = _as_the_door_reads(log)
    assert len(frames) > 1, "an optimisation with one frame asks no range"
    cell = _the_decks_cell(pyscf_run)
    root = _open(page, flask_server, monkeypatch, pyscf_run, log)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    save_to_project(page, "pyscf-trajectory", frames=len(frames))
    _holds_the_engines_frame(root / "pyscf-trajectory.xyz", cell,
                             frames[-1].structure.positions)
