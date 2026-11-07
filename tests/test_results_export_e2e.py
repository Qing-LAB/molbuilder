"""A structure saved out of the Results tab reloads where the engine had it
(`model/structure-periodicity.md` § 6.0): *"a structure saved from an
engine's output carries the engine's coordinates with the engine's origin,
(0,0,0), set together and recorded as the engine's"* -- so the saved pair
draws its box where the engine had it, and a new calculation made from it is
centred.

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

The PySCF trajectory open also reads the run's heading in the picker
(`web/presenters.md` § 2): *"the engine's words and the calculation's noun,
both served with the folder answer"* -- *PySCF optimization*; a SIESTA
`.out`'s, *SIESTA optimization*, is a row of `tests/data/presenters.toml`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from support.results_export import save_to_project, wait_for_viewer

pytestmark = pytest.mark.engine


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, flask_server, monkeypatch, run, pick):
    """The Results tab on ``run``'s first `coarse` run (a
    `support.real_runs.RealRun`, made once a pass, `process/testing.md`
    tier 3) with ``pick`` chosen from its list -- the tab bound to the
    folder as a person binds it, by the sidebar's rows and Reload
    (`support.sidebar.results_on`), in the projects tree the run was made
    in, the one door pointed at it: that tree."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.sidebar import results_on
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(run.tree))
    results_on(page, flask_server, run.attempt("coarse"))
    page.wait_for_function(
        "(want) => [...document.querySelectorAll("
        "  '#results-file-picker-select option')].some(o => o.value === want)",
        arg=str(pick), timeout=30000)
    page.select_option("#results-file-picker-select", value=str(pick))
    return run.tree


def _heading(page, pick):
    """The heading ``pick`` sits under in the picker -- its group's label."""
    return page.eval_on_selector(
        "#results-file-picker-select",
        "(s, want) => { const o = [...s.options].find((o) => o.value === want);"
        "  return o && o.parentElement.tagName === 'OPTGROUP'"
        "    ? o.parentElement.label : null; }",
        str(pick))


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
    """The saved pair, read back through the codec: the engine's cell and
    coordinates, its origin a stated 0 recorded as the ENGINE's (§ 6.0) --
    recorded as the person's, *Entering a new calculation* would keep the
    0 instead of centring it."""
    from molbuilder.workingcopy_structure import StructureCodec
    back = StructureCodec().load(saved)
    assert back.engine_offset is not None, (
        "the saved pair states no origin, so its next treatment re-derives "
        "one and moves the box off where the engine had it")
    assert np.allclose(back.engine_offset, 0.0), back.engine_offset
    assert back.offset_by == "engine", (
        f"the saved origin is recorded as {back.offset_by!r}'s, not the "
        "engine's, so a new calculation made from it is not centred")
    assert back.cell is not None, "the saved pair carries no cell"
    assert np.allclose(back.cell, cell, atol=1e-5), (back.cell, cell)
    assert np.allclose(back.positions, positions, atol=1e-5), (
        "not the engine's own coordinates")


def test_a_siesta_trajectory_saves_the_box_its_output_states(
        real_h2_layered, page, flask_server, monkeypatch):
    """The trajectory viewer on a SIESTA relaxation -- the shared H2's first
    `coarse` run (`support.real_runs.layered`): the cell is the one the
    output states, the origin a stated 0 -- and the name offered is the
    output's own (`web/molview.md`: open ``runs/mine.xyz`` and an Export
    writes ``mine.xyz``), with the frame it holds.

    MUTATIONS THIS MUST FAIL AGAINST: the one composer not stating the
    engine's origin (`runs.Declared.frame`), or stating it as the person's;
    the viewer naming its install after the parser's label, which offered
    " .log_frame6" and saved a hidden ``.log_frame6.xyz`` until 2026-09-27.
    """
    from molbuilder.runs import run_of
    out = run_of(real_h2_layered.attempt("coarse")).stdout
    frames, cell = _as_the_door_reads(out)
    assert len(frames) > 1, "a relaxation with one frame asks no frame range"
    root = _open(page, flask_server, monkeypatch, real_h2_layered, out)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    offered = save_to_project(page, "siesta-trajectory", frames=len(frames))
    assert offered == f"{out.stem}_frame{len(frames) - 1}", offered
    _holds_the_engines_frame(root / "siesta-trajectory.xyz", cell,
                             frames[-1].structure.positions)


def test_the_engines_own_geometry_file_saves_the_box_the_run_had(
        real_h2_layered, page, flask_server, monkeypatch):
    """SIESTA's ``<label>.xyz`` -- its final coordinates, written with no
    sidecar -- opened in the structure preview and saved: the pair carries
    the box the run had and a stated 0, as the output states it.

    MUTATION THIS MUST FAIL AGAINST: the codec reading an engine's own file
    with no frame (the no-sidecar branch of `StructureCodec.load`), which
    saved it with no cell and no origin until 2026-09-27.
    """
    from molbuilder.runs import run_of
    from molbuilder.structure import Structure
    run = run_of(real_h2_layered.attempt("coarse"))
    own = run.on_label(".xyz")
    _frames, cell = _as_the_door_reads(run.stdout)
    positions = Structure.from_xyz(own.read_text()).positions
    root = _open(page, flask_server, monkeypatch, real_h2_layered, own)
    wait_for_viewer(page, atoms=2, frames=1)
    save_to_project(page, "siesta-own", frames=1)
    _holds_the_engines_frame(root / "siesta-own.xyz", cell, positions)


def test_a_pyscf_trajectory_saves_the_box_its_deck_recorded(
        real_h2_pyscf, page, flask_server, monkeypatch):
    """The trajectory viewer on a PySCF optimisation -- the shared H2's
    first `coarse` run (`support.real_runs.pyscf`) -- whose output states no
    cell: the box is the one its deck recorded placing the atoms in.

    MUTATIONS THIS MUST FAIL AGAINST: the one composer ignoring the deck
    record's cell when the output carries none; the heading not the engine's
    words and the calculation's noun, *PySCF optimization*
    (`web/presenters.md` § 2).
    """
    from molbuilder.runs import run_of
    folder = real_h2_pyscf.attempt("coarse")
    log = run_of(folder).file(".molwatch.log")
    assert log is not None, sorted(p.name for p in folder.iterdir())
    frames, _no_cell = _as_the_door_reads(log)
    assert len(frames) > 1, "an optimisation with one frame asks no range"
    cell = _the_decks_cell(folder)
    root = _open(page, flask_server, monkeypatch, real_h2_pyscf, log)
    assert _heading(page, log) == "PySCF optimization", _heading(page, log)
    wait_for_viewer(page, atoms=2, frames=len(frames))
    save_to_project(page, "pyscf-trajectory", frames=len(frames))
    _holds_the_engines_frame(root / "pyscf-trajectory.xyz", cell,
                             frames[-1].structure.positions)
