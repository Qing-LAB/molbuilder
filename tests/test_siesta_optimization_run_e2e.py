"""A SIESTA optimization of ours, run with the real SIESTA, and what each
reader says about it (`process/testing.md` § 6: a test about what a run ended
as, produced or shows makes that run with the engine, through the road).

The road's H2 optimization (`support.real_runs.layered`, ``real_h2_layered``:
``molbuilder describe`` on the shipped `publishable` ladder -> ``prep task
--group 1`` -> ``launch task --stage coarse --mode direct``), an H2 in a
10 Å box, isolated on every axis, its first atom held: the coarse stage's
first run in its own folder, ``01_coarse/run-0/``, SIESTA's own files --
named by ``SystemLabel`` -- among ours, as it ran (its stage launched again
opens the next run, `project-layout.md` § 1.5).  And a calculation of this
module's own whose relaxation is allowed one move, so its first run ends out
of moves -- finished, not converged -- and its stage launched again opens
``run-1/`` warm, continuing from it (`running-a-job.md` § 3.5;
`project-layout.md` § 1.6.1: to continue is to launch again).

Every expectation is what the run was given -- the structure written here,
the deck prep wrote, the template, the env's pinned build -- or a rule of the
contract applied to the run's own files; never a number a reader printed.
The readers are tested on this run, made with the engine (user, 2026-10-06:
"when a test need siesta's output why is it not part of a e2e test?"), and
every file of ours is asked of the run door (`runs.run_of`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, live_siesta, set_template_value
from support.road import H2, road_structure
from support.road import molbuilder, one_rank

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: The structure both calculations hold: the road's H2 (`support.road.H2`),
#: in a 10 Å box isolated on every axis, its first atom held -- its box's
#: edge and its atoms.
_BOX = H2["cell"][0]
_H2 = H2["positions"]


@pytest.fixture(scope="module")
def capped(isolated_projects_root_module, tmp_path_factory):
    """The capped calculation's coarse stage, in a projects tree of its own:
    ONE MOVE ALLOWED -- its first run, ``run-0``, ends out of moves -- and
    the stage launched again, which opens ``run-1`` warm from it
    (`running-a-job.md` § 3.5): ``(run-0, run-1)``."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    from molbuilder.workingcopy_structure import StructureCodec
    from support.road import group_words, jobset
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory):
        (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
        StructureCodec().write(road_structure(H2),
                               tree / "P" / "structure" / "h2.xyz")
        bundle = tree / "P" / "opt" / "H2cap"
        r = molbuilder("describe", "--structure", "P/structure/h2.xyz",
                   "--bundle", "P/opt/H2cap", "--engine", "siesta",
                   "--calculation", "optimization",
                   "--name", "H2", "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        one_rank(bundle)
        set_template_value(bundle / "H2.template.toml", "relax_steps", "1")
        r = jobset("prep", "task", *group_words(bundle, "coarse"),
                   "--bundle", bundle, "--target", "this")
        assert r.exit_code == 0, r.output
        home = stage_home(bundle, task_of(bundle), "coarse").dir
        # A RUN OUT OF MOVES FINISHED (`running-a-job.md` § 3.5), so the
        # launch that waited on it exits 0 (`job-system.md` § 6.0, *A run
        # here, waited on*: "exits 0 when every member finished").
        launched = jobset("launch", "task", "--stage", "coarse", "--bundle",
                          bundle, "--mode", "direct", "--foreground", "--yes")
        assert launched.exit_code == 0, launched.output
        first = latest_attempt(home)
        # ...AND LAUNCHED AGAIN: the next run, warm from the one that ran
        # out of moves.
        again = jobset("launch", "task", "--stage", "coarse", "--bundle",
                       bundle, "--mode", "direct", "--foreground", "--yes")
        second = latest_attempt(home)
        assert second is not None and second.name == "run-1", again.output
        yield first, second


@pytest.fixture(scope="module")
def opt(real_h2_layered):
    """The road's H2 optimization's coarse stage's first run,
    ``01_coarse/run-0/`` (:meth:`support.real_runs.RealRun.attempt`)."""
    folder = real_h2_layered.attempt("coarse")
    from molbuilder.runs import run_of
    assert ">> End of run" in run_of(folder).stdout.read_text(
        errors="replace"), "the coarse run did not reach its end"
    return folder


def _run(folder: Path):
    """The run in ``folder``, through the run door."""
    from molbuilder.runs import run_of
    run = run_of(folder)
    assert run is not None, folder
    return run


def _out(folder: Path) -> Path:
    return _run(folder).stdout


def _deck(folder: Path) -> Path:
    return _run(folder).deck


def _deck_value(deck: Path, key: str) -> str:
    """The value the deck states for ``key`` -- what SIESTA was told."""
    for ln in deck.read_text().splitlines():
        words = ln.split()
        if words and words[0] == key:
            return words[1]
    raise AssertionError(f"{deck.name} states no {key}")


def _template_value(bundle: Path, name: str):
    from molbuilder.template import one, read_template
    return one(read_template((bundle / "H2.template.toml").read_text()),
               name).value


# --------------------------------------------------------------------- #
#  The run's folder, as the Results tab asks it (`model/parse.md` § 5)    #
# --------------------------------------------------------------------- #

def test_the_run_folder_answers_for_its_run(opt):
    """The run's folder is a run (`project-layout.md` § 1.4a), it opens the
    run's own engine output before the progress log prep seeded beside it
    (§ 5.2, § 5.5), and every name is read back with the run's label:
    SIESTA's files, named by ``SystemLabel`` -- its history among them --
    are the engine's, the deck and the output the stage's own, named on its
    stem (`job-contracts.md` § 2.2, § 2.2a)."""
    from molbuilder.runs import folder_answer
    got = folder_answer(opt)
    assert got["place"]["role"] == "run", got["place"]
    assert Path(got["openable"]).name == "H2_01_coarse.out", (
        got["attempts"])
    assert got["status"]["state"] == "finished", got["status"]
    assert Path(got["record"]["deck"]["path"]).name == "H2_01_coarse.fdf", (
        got["record"]["deck"])
    files = {f["name"]: f for f in got["files"]}
    engines = [p.name for p in opt.glob("fdf.*.log")] + [
        "H2.XV", "H2.xyz", "H2.MD.nc"]
    for name in engines:
        assert (files[name]["role"], files[name]["about"]) == (
            None, {"ours": False}), files[name]
    for name, role in (("H2_01_coarse.fdf", ".fdf"),
                       ("H2_01_coarse.out", ".out")):
        assert (files[name]["role"], files[name]["stage"],
                files[name]["about"]["ours"]) == (role, "01_coarse", True), (
            files[name])


def test_the_run_reads_its_own_deck(opt, monkeypatch):
    """The Results load of the run shows atom 0 held in the isolated 10 Å box
    this module wrote; the run reads its own deck (user, 2026-10-04: *"make
    sure that it does make each run sees its own .fdf"*); and SIESTA's own
    ``H2.xyz``, opened as a structure, carries the same box at the engine's
    origin (`model/structure-periodicity.md` § 6.0)."""
    from molbuilder import diagnostics
    from molbuilder.web.app import create_app
    from molbuilder.workingcopy_structure import StructureCodec
    root = next(p for p in opt.parents if p.name == "projects")
    bundle = next(p for p in opt.parents if (p / "task.json").is_file())
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "projects"),))
    d = create_app(config={}).test_client().post(
        "/api/watch/load", json={"path": str(opt)}).get_json()
    assert d["ok"] is True, d
    held = json.loads(d["atom_metadata"])["regions"]["frozen_atoms"]
    assert held == [0], d["atom_metadata"]
    box = d["periodicity"]
    assert box["axis_kind"] == ["isolated"] * 3, box
    assert box["engine_offset"] == [0.0, 0.0, 0.0], box
    np.testing.assert_allclose(box["cell"], np.eye(3) * _BOX, atol=1e-4)
    # ITS OWN OUTPUT NAMES ITS COMPANIONS AND ITS HELD ATOMS (§ 5.3).
    rt = d["data"]["runtime_info"]
    assert rt.get("mdnc_source") == "H2.MD.nc", rt
    assert rt.get("frozen_atoms") == [0], rt
    # ITS OWN DECK STATES ITS CONTRACT.
    calc = d["info"]["calculation"]
    assert calc["source"] == "H2_01_coarse.fdf", calc
    assert calc["contract"]["basis_size"] == _template_value(
        bundle, "basis_size"), calc

    own = StructureCodec().load(_run(opt).on_label(".xyz"))
    assert own.cell is not None, "the run's box never reached its H2.xyz"
    np.testing.assert_allclose(np.asarray(own.cell), np.eye(3) * _BOX,
                               atol=1e-4)
    np.testing.assert_allclose(own.engine_offset, np.zeros(3))


# --------------------------------------------------------------------- #
#  The run's .XV (`parse/coords.py`, `molbuilder xv2xyz`)                 #
# --------------------------------------------------------------------- #

def _xv(folder: Path) -> Path:
    """SIESTA's own ``<label>.XV`` in the run's folder, asked of the run
    door (`Run.on_label`)."""
    got = _run(folder).on_label(".XV")
    assert got is not None, sorted(p.name for p in folder.iterdir())
    return got


def _xv_as_written(xv: Path):
    """The ``.XV`` read as SIESTA writes it -- three cell rows, the atom
    count, one row per atom ``species Z x y z vx vy vz`` -- in Bohr."""
    rows = xv.read_text().split("\n")
    cell = np.array([[float(w) for w in rows[i].split()[:3]]
                     for i in range(3)])
    n = int(rows[3].split()[0])
    atoms = [rows[4 + i].split() for i in range(n)]
    return cell, [int(a[1]) for a in atoms], np.array(
        [[float(w) for w in a[2:5]] for a in atoms])


def test_the_xv_reader_reads_what_siesta_wrote(opt):
    """Elements keyed off Z, coordinates and the cell converted Bohr -> Å,
    and a file that states a lattice yields a structure that has one -- its
    axis kinds `Structure`'s own (`model/structure-periodicity.md` § 2)."""
    from molbuilder.constants import BOHR_ANGSTROM
    from molbuilder.parse.coords import read_xv, read_xv_cell, \
        read_xv_with_cell
    xv = _xv(opt)
    cell_bohr, zs, pos_bohr = _xv_as_written(xv)
    assert zs == [1, 1]
    s = read_xv(xv)
    assert s.elements == ["H", "H"]
    np.testing.assert_allclose(s.positions, pos_bohr * BOHR_ANGSTROM,
                               rtol=1e-9)
    np.testing.assert_allclose(read_xv_cell(xv), cell_bohr * BOHR_ANGSTROM,
                               rtol=1e-9)
    # ...and that cell is the run's 10 Å box
    np.testing.assert_allclose(read_xv_cell(xv), np.eye(3) * _BOX, atol=1e-4)
    struct, cell = read_xv_with_cell(xv)
    assert struct.cell is not None
    np.testing.assert_allclose(np.asarray(struct.cell), cell, atol=1e-9)
    assert struct.axis_kind == ("periodic", "periodic", "periodic")


def test_xv2xyz_writes_the_pair_and_the_next_deck_takes_its_frame(opt,
                                                                  tmp_path):
    """A bare ``.XV`` becomes the PAIR, so the cell has somewhere to go, and
    what prep reopens -- through the one read door, `StructureCodec.load` --
    is the run's frame: the cell, and the coordinates at the engine's origin
    -- a stated offset of 0 (`model/structure-periodicity.md` § 6.0)."""
    from click.testing import CliRunner

    from molbuilder import cli
    from molbuilder.cell import to_engine
    from molbuilder.parse.coords import read_xv
    from molbuilder.workingcopy_structure import StructureCodec
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(_xv(opt)),
                                       str(out)])
    assert res.exit_code == 0, res.output
    assert out.is_file() and (tmp_path / "h2.molstruct.json").is_file()
    assert out.read_text().splitlines()[0].strip() == "2"
    s = StructureCodec().load(out)
    np.testing.assert_allclose(np.asarray(s.cell), np.eye(3) * _BOX,
                               atol=1e-4)
    assert s.axis_kind == ("periodic", "periodic", "periodic")
    frame = to_engine(s)
    assert frame.stated, "the pair lost the .XV's origin"
    np.testing.assert_allclose(frame.positions,
                               read_xv(_xv(opt)).positions, atol=1e-6)


def test_xv2xyz_from_run_takes_what_the_run_declared(opt, tmp_path):
    """``--from-run``: the atom the run's deck holds and the axes its
    structure stated -- isolated, the 10 Å box written here -- on the
    ``.XV``'s own cell at the engine's origin; the run's deck is the one the
    run door names (`runs.run_of`, `runs.declared`; plan D19)."""
    from click.testing import CliRunner

    from molbuilder import cli
    from molbuilder.workingcopy_structure import StructureCodec
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(_xv(opt)),
                                       str(out), "--from-run"])
    assert res.exit_code == 0, res.output
    assert "H2_01_coarse.fdf" in res.output, res.output
    got = StructureCodec().load(out)
    assert got.frozen_atoms == [0]
    assert got.axis_kind == ("isolated", "isolated", "isolated")
    np.testing.assert_allclose(np.asarray(got.cell), np.eye(3) * _BOX,
                               atol=1e-4)
    np.testing.assert_allclose(got.engine_offset, np.zeros(3))


def test_xv2xyz_from_run_refuses_a_xv_no_run_of_ours_holds(opt, tmp_path):
    """The flag says a run is there: the run's ``.XV`` taken out to a folder
    no calculation marks has nothing declaring its atoms, so it is refused
    by name -- never written at the defaults as if a run had said so."""
    import shutil

    from click.testing import CliRunner

    from molbuilder import cli
    away = tmp_path / "H2.XV"
    shutil.copy(_xv(opt), away)
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(away), str(out),
                                       "--from-run"])
    assert res.exit_code != 0
    assert "no run of ours holds" in res.output, res.output
    assert not out.exists()


# --------------------------------------------------------------------- #
#  The output and its companions (`model/parse.md` § 3, § 5c, § 5d)      #
# --------------------------------------------------------------------- #

def test_the_registry_reads_the_output_as_siestas(opt):
    """The output SIESTA wrote is claimed by exactly one parser, SIESTA's,
    and read as a trajectory with frames (`model/parse.md` § 3: `detect`
    returns the one parser whose `can_parse` is true -- none, or two, is
    refused; registration order is not precedence).

    THE SILENT FAILURE: a real SIESTA output claimed by no parser or by
    two -- a new parser's `can_parse` matching it -- so nothing opens the
    run's own output; only SIESTA's real output can show it."""
    from molbuilder.parse import TrajectoryResult, detect, parse
    from molbuilder.parse.engines import SiestaOutFileParser
    assert detect(_out(opt)) is SiestaOutFileParser
    res = parse(_out(opt))
    assert isinstance(res, TrajectoryResult)
    assert (res.result_kind, res.source_format, res.parser_name) == (
        "trajectory", "siesta", "siesta")
    assert res.frames


def test_the_convergence_targets_are_the_decks_read_from_its_echo(opt):
    """What the run was asked to reach, from SIESTA's own echo of the deck --
    never a default reproduced: the force tolerance and the SCF cap the
    coarse deck states."""
    from molbuilder.parse.engines.siesta import SiestaParser
    deck = _deck(opt)
    ct = SiestaParser.parse(str(_out(opt))).runtime_info.get(
        "convergence_targets")
    assert ct is not None and ct.get("source") == "siesta_input_echo", ct
    assert ct.get("max_force_tol_eV_per_A") == pytest.approx(
        float(_deck_value(deck, "MD.MaxForceTol"))), ct
    assert ct.get("max_scf_iter") == int(_deck_value(deck,
                                                     "MaxSCFIterations")), ct


def test_the_run_states_the_build_it_ran_and_its_solver(opt):
    """The header SIESTA printed and the solver it resolved
    (`siesta_grammar.read_build_line`, ``read_diag_line``): the version and
    the MPI build the env's recipe pins (`envs/recipes.py`,
    ``siesta=5.4.2=mpi_openmpi_*``), and no token SIESTA does not write."""
    from molbuilder.envs.recipes import _SIESTA
    from molbuilder.parse.engines.siesta import SiestaParser
    spec = next(p.spec for p in _SIESTA.conda_packages
                if p.spec.startswith("siesta="))
    _, version, build = spec.split("=")
    traj = SiestaParser.parse(str(_out(opt)))
    got = traj.runtime_info["siesta_build"]
    assert got["version"].startswith(version), got
    assert build.startswith("mpi") and "MPI" in got["parallelisations"], got
    assert traj.runtime_info["siesta_diag"].get("algorithm"), traj.runtime_info
    assert [w for w in traj.parse_warnings
            if w.category == "runtime_info"] == []


def test_a_relaxations_step_boundaries_are_not_iterations(opt):
    """The run's SCF-timing log, ``<epoch> <iscf> <row>`` a line: the delta
    into a row whose iteration is 1 spans the previous SCF's end, the forces,
    the move and the next step's setup, and the phase's first delta can
    carry warm-up -- neither is an iteration (`scf_timing_rows`,
    `model/parse.md` § 5c).  And the run's record states those figures, read
    from this file alone (§ 5c, *A run's timing file is chosen by one
    door*; 2.64.0).

    MUTATION THIS MUST FAIL AGAINST: keep the deltas into iteration 1; the
    record timed by prep's seeded progress log (no rate, zero iterations)
    instead of the run's own `.scf-timing.log`."""
    from molbuilder.parse.instruments.scf_timing_rows import \
        scf_timing_metrics
    text = _run(opt).file(".scf-timing.log").read_text()
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    starts = sum(1 for r in rows if r[1] == "1")
    assert starts >= 2, "a relaxation of one SCF shows no boundary"
    epochs = [float(r[0]) for r in rows]
    within = [b - a for a, b, r in zip(epochs, epochs[1:], rows[1:])
              if r[1] != "1"][1:]
    m = scf_timing_metrics(text)
    assert m["rows"] == len(rows), m
    assert m["iters_measured"] == len(within) == len(rows) - starts - 1, m
    # stated to the millisecond, the precision it is reported at
    assert m["s_per_iter"] == pytest.approx(sum(within) / len(within),
                                            abs=5e-4), m
    # THE RUN'S RECORD TIMES IT BY THIS FILE ALONE (2.64.0; `model/parse.md`
    # § 5c: a run whose output the catalogue cuts a tee from "is timed by
    # that tee alone ... since its progress log is prep's seed"; § 5d.1b,
    # the `instruments` row): the figures the Results tab shows are this
    # file's -- never the seed's, which read as zero iterations.
    from molbuilder.runs import record_of
    timed = record_of(_run(opt))["computation"]["time"]
    assert (timed.get("s_per_iter"), timed.get("iters_measured")) == (
        m["s_per_iter"], m["iters_measured"]), timed


def test_the_engines_own_log_says_what_it_used_and_what_nobody_set(opt):
    """SIESTA's ``fdf.<stamp>.log`` (`model/parse.md` § 5d.2-5d.3): every
    key SIESTA read, with the value it read -- a key the deck carries reads
    as set, any other as SIESTA's default; a block is found by the deck's
    own spelling, fdf's label rule."""
    from molbuilder.parse import detect
    from molbuilder.parse.fdf import fdf_key
    deck = _deck(opt)
    stated, blocks = set(), set()
    for ln in deck.read_text().splitlines():
        words = ln.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].lower() == "%block":
            blocks.add(fdf_key(words[1]))
        elif not words[0].startswith("%"):
            stated.add(fdf_key(words[0]))
    [log] = list(opt.glob("fdf.*.log"))
    res = detect(log).parse(log)
    assert res.result_kind == "engine-params"
    one_reading = {k: v for k, v in res.params.items() if "default" in v}
    assert one_reading, res.params
    for key, p in one_reading.items():
        assert p["default"] is (key not in stated), (key, p)
    assert any(p["default"] for p in one_reading.values()), (
        "SIESTA read no key the deck left to it")
    mesh = res.params[fdf_key("MeshCutoff")]
    assert (mesh["number"], mesh["unit"], mesh["default"]) == (
        float(_deck_value(deck, "MeshCutoff")), "Ry", False), mesh
    for name in blocks:
        assert name in res.blocks, (name, sorted(res.blocks))
    # A BLOCK'S ROWS AND NOTHING ELSE: `LatticeConstant`, looked up while
    # SIESTA reads `LatticeVectors`, is a setting read, never a row of it.
    assert len(res.blocks[fdf_key("LatticeVectors")]) == 3, res.blocks
    assert fdf_key("LatticeConstant") in res.params


# --------------------------------------------------------------------- #
#  SIESTA's history, each run's own (`job-contracts.md` § 2.2;           #
#  `model/parse.md` § 5a)                                                #
# --------------------------------------------------------------------- #

def _md_nc(folder: Path) -> Path:
    """SIESTA's own ``<label>.MD.nc`` in the run's folder, asked of the run
    door (`Run.on_label`)."""
    got = _run(folder).on_label(".MD.nc")
    assert got is not None, sorted(p.name for p in folder.iterdir())
    return got


def test_the_history_is_found_by_the_output_s_own_name(opt):
    """The output, ``H2_01_coarse.out``, prints the ``SystemLabel`` it ran
    under, and the run's history is SIESTA's ``H2.MD.nc``, named by that
    label -- which is how the output finds its own, read back with the label
    it prints as the reading pass read it (`siesta_mdnc.sibling_md_nc`,
    § 4a, § 5.3) -- and the output's frames are upgraded from it.

    MUTATION THIS MUST FAIL AGAINST: the history not named from the label
    the output prints -- the trajectory keeps the output's four-decimal
    coordinates."""
    from molbuilder.parse import parse
    res = parse(str(_out(opt)))
    assert res.runtime_info.get("mdnc_source") == _md_nc(opt).name, (
        res.runtime_info)
    assert res.runtime_info.get("mdnc_coords_upgraded", 0) >= 1, (
        res.runtime_info)


def test_the_history_is_claimed_by_its_reader_and_routed_there(opt):
    """Its reader claims the ``.MD.nc`` and declines the output beside it --
    the other half of the same run, whose parser reads its run state and its
    SCF -- and the registry routes the history to it."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import SiestaMdNcFileParser
    assert SiestaMdNcFileParser.can_parse(_md_nc(opt))
    assert not SiestaMdNcFileParser.can_parse(_out(opt))
    res = parse(str(_md_nc(opt)))
    assert (res.parser_name, res.source_format) == ("siesta-mdnc",
                                                    "siesta-mdnc")


def test_the_history_says_only_what_it_holds(opt):
    """No run state -- a complete-looking history may belong to a live
    run -- no forces and no SCF; coordinates converted from Bohr by the
    file's own attribute (an H2 bond, Å)."""
    from molbuilder.parse import parse
    res = parse(str(_md_nc(opt)))
    assert res.run_state == "unknown"
    assert res.frames
    assert all(f.forces is None and f.scf_history is None
               for f in res.frames)
    for f in res.frames:
        bond = float(np.linalg.norm(f.structure.positions[1]
                                    - f.structure.positions[0]))
        assert 0.5 < bond < 1.3, f"bond {bond} Å -- the unit looks wrong"


def test_a_frames_geometry_and_energy_belong_to_one_step(opt):
    """A history row holds the geometry about to be tried beside the energy
    of the one just evaluated; the reader pairs each energy with its own
    geometry -- checked against the output of the same run, to the output's
    printing precision -- and the last geometry, not yet evaluated, has no
    energy.  The per-step series move with the energy."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    nc = parse(str(_md_nc(opt)))
    out = parse(str(_out(opt)))
    mapping = align_to_reference(out.frames, nc.frames)
    compared = 0
    for i, j in enumerate(mapping):
        if j is None:
            continue
        e_out, e_nc = out.frames[i].energy, nc.frames[j].energy
        if e_out is None or e_nc is None:
            continue
        compared += 1
        assert abs(e_out - e_nc) < 5e-4, (i, j, e_out, e_nc)
    assert compared >= 2, f"only {compared} frame(s) compared"
    assert nc.frames[-1].energy is None
    eks = nc.runtime_info.get("mdnc_eks_eV")
    assert eks is not None and len(eks) == len(nc.frames)
    assert all((f.energy is None) == (e is None)
               for f, e in zip(nc.frames, eks))


def test_the_output_keeps_its_frames_and_the_history_matches_in_order(opt):
    """The output defines the frame list and opens with the input geometry,
    which the history never stores; the history only upgrades the frames it
    matches, in its own order, each row once (`model/parse.md` § 5a, *The
    SIESTA case*: the primary file owns the frame list; no row for the
    input geometry; a row matched by coordinates)."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    out = parse(str(_out(opt)))
    nc = parse(str(_md_nc(opt)))
    mapping = align_to_reference(out.frames, nc.frames)
    assert len(mapping) == len(out.frames)
    assert mapping[0] is None
    # THE INPUT GEOMETRY, in the engine's frame: the atoms this module wrote,
    # placed in the box by the deck (`model/structure-periodicity.md` § 6.0)
    first = np.asarray(out.frames[0].structure.positions)
    np.testing.assert_allclose(first - first[0],
                               np.array(_H2) - np.array(_H2[0]), atol=1e-4)
    assert out.frames[0].energy is not None
    matched = [j for j in mapping if j is not None]
    assert len(matched) >= 2 and matched == list(range(len(matched))), mapping


# --------------------------------------------------------------------- #
#  A run out of moves, and launched again (`running-a-job.md` § 3.5)     #
# --------------------------------------------------------------------- #

def test_a_step_a_run_is_inside_or_stopped_inside_is_no_step(real_h2_layered):
    """`runs.StepsTable`: a run's output cut inside its first SCF, as a running
    or a killed run leaves it, and the same with SIESTA's own stop.  The silent
    failure: the energy of an SCF row read as the run's final energy, about
    1000 eV off on a real 52-atom relaxation stopped inside a move."""
    from molbuilder.parse import detect
    from molbuilder.parse.engines.siesta_grammar import FATAL_MARKERS
    from molbuilder.postprocess import RunStepsError, read_steps
    from molbuilder.runs import run_of
    from support.real_runs import copy_of

    def cut_with(tail):
        def cut(copy):
            out = run_of(copy.attempt("coarse")).stdout
            lines = out.read_text(errors="replace").splitlines(keepends=True)
            first = next(k for k, line in enumerate(lines)
                         if line.lstrip().startswith("scf:"))
            out.write_text("".join(lines[:first + 1]) + tail)
        return cut

    fatal = next(m for m, state in FATAL_MARKERS if state == "stopped")
    for name, tail, flag in (("in-flight", "", "in_progress"),
                             ("stopped", f"\n{fatal}\n", "cut_short")):
        run = copy_of(real_h2_layered, f"cut-{name}", cut_with(tail))
        folder = run.attempt("coarse")
        out = run_of(folder).stdout
        assert getattr(detect(out).parse(str(out)).frames[-1], flag)
        with pytest.raises(RunStepsError,
                           match="holds no step its run finished"):
            read_steps(folder)


def test_a_relaxation_out_of_moves_ends_finished_not_converged(capped):
    """One move allowed: SIESTA exits 0 printing ``Final (unrelaxed)``
    coordinates, and the run reads *finished, not converged* -- its
    convergence column ``geometry NO`` -- never an SCF stop
    (`running-a-job.md` § 3.5, § 4.2).

    THE SILENT FAILURES: a relaxation out of moves read as converged, or as
    failed by an SCF it did not stop on."""
    from molbuilder.parse.engines._run_ending import ending_of
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
    from molbuilder.runs import state_of
    first, _ = capped
    run = _run(first)
    end = ending_of(run.stdout)
    assert end.relaxed is False and end.cause != SCF_NOT_CONV_MARKER, end
    st = state_of(run)
    assert (st.state, st.converged_words) == ("finished", "geometry NO"), st


def test_a_relaxation_out_of_moves_launched_again_continues_warm(capped):
    """To continue is to launch again (`project-layout.md` § 1.6.1;
    `running-a-job.md` § 3.5): the run that ran out of moves holds its one
    start -- its session log, its banner saying it started clean, its
    conclusion marker -- and the stage launched again opens ``run-1``, which
    records ``run-0`` as the run it continues from, holds the restart file
    it saved, and says in its own banner that SIESTA will load it
    (`job-contracts.md` § 4.4).

    THE SILENT FAILURES: a launch again that starts cold, its relaxation
    begun again from the structure; a new run that names no run it
    continues, or another.  MUTATION THIS MUST FAIL AGAINST: launch again
    copying none of the run's restart files."""
    from molbuilder.runrecord import RunRef, launch_record, read_continued_from
    first, again = capped
    for folder in (first, again):
        run = _run(folder)
        assert run.session_log is not None and run.file(".concluded"), (
            sorted(p.name for p in folder.iterdir()))
    started = _run(first).session_log.read_text(errors="replace")
    assert "CLEAN START (no restart file here)" in started, started
    assert read_continued_from(again) == RunRef.read("01_coarse/run-0")
    assert launch_record(again)["continued_from"] == "01_coarse/run-0"
    assert (again / "H2.XV").is_file(), sorted(
        p.name for p in again.iterdir())
    mode = next((ln for ln in _run(again).session_log.read_text(
        errors="replace").splitlines() if "CONTINUES (engine will load" in ln),
        None)
    assert mode is not None and "H2.XV" in mode, mode
