"""A flat calculation of ours, run with the real SIESTA, and what each reader
says about it (`process/testing.md` § 6: a test about what a run ended as,
produced or shows makes that run with the engine, through the road).

``jobset init`` (flat, the shipped `publishable` ladder) -> ``prep task
coarse`` -> ``launch task --stage coarse --mode direct`` -> ``prep task --stage medium``, on an
H2 in a 10 Å box, isolated on every axis, its first atom held: the coarse
stage's run beside the medium stage's deck in one folder, SIESTA's own files
-- named by ``SystemLabel`` -- among ours.  And a second calculation whose
relaxation is allowed one move, so it ends out of moves -- its run card set
to start over when launched again, and launched again.

What molbuilder does with a flat folder's runs is read here too (plan
§ 5y): the next stage's prep leaves the run folder's own files as they
were, and every name the Task setup card gives a stage is a file there;
launched again, a stage runs again in its folder as its run 1, taking
nothing and overwriting nothing of run 0's, and the folder speaks for its
newest run index though an older run's output is the newer file.

Every expectation is what the run was given -- the structure written here,
the deck prep wrote, the template, the env's pinned build -- or a rule of the
contract applied to the run's own files; never a number a reader printed.
The readers are tested on this run, made with the engine (user, 2026-10-06:
"when a test need siesta's output why is it not part of a e2e test?").
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, live_siesta, set_template_value

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: The structure, as this module writes it: H2 in a 10 Å box, isolated on
#: every axis, the first atom held.
_BOX = 10.0
_H2 = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]


def _write_h2(tree: Path, name: str) -> None:
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    StructureCodec().write(
        Structure(elements=["H", "H"], positions=np.array(_H2),
                  regions={"frozen_atoms": [0]},
                  cell=np.diag([_BOX] * 3), axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / name)


def _one_rank(bundle: Path, **card) -> None:
    """The run card: one rank, one thread -- an H2 needs no more -- and
    what else ``card`` states."""
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {**task.get("execution", {}), "mpi_np": 1,
                         "omp_threads": 1, **card}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))


def _as_written(path: Path):
    """A file's bytes and its write time to the nanosecond."""
    return path.read_bytes(), path.stat().st_mtime_ns


@pytest.fixture(scope="module")
def road(isolated_projects_root_module, tmp_path_factory):
    """The projects tree, with the flat calculation run and the capped one
    run beside it, and what their later steps answered and left:
    ``{"flat": <folder>, "capped": <folder>, ...}``."""
    from support.road import jobset
    tree = isolated_projects_root_module
    with live_siesta(tree, tmp_path_factory):
        _write_h2(tree, "h2.xyz")

        flat = tree / "P" / "opt" / "H2flat"
        r = jobset("init", "--structure", "P/structure/h2.xyz",
                   "--bundle", "P/opt/H2flat", "--engine", "siesta",
                   "--calculation", "optimization", "--shape", "flat",
                   "--name", "H2", "--psml-lib", "pseudopotential",
                   "--stage-strategy", "publishable")
        assert r.exit_code == 0, r.output
        _one_rank(flat)
        r = jobset("prep", "task", "--stage", "coarse", "--bundle", flat,
                   "--target", "this")
        assert r.exit_code == 0, r.output
        r = jobset("launch", "task", "--stage", "coarse", "--bundle", flat,
                   "--mode", "direct", "--yes")
        assert r.exit_code == 0, r.output
        assert ">> End of run" in (flat / "H2_01_coarse-run0.out").read_text(
            errors="replace"), "the coarse run did not reach its end"
        kept = ("H2_01_coarse.fdf", "H2_01_coarse.run.sh")
        kept_before = {n: _as_written(flat / n) for n in kept}
        r = jobset("prep", "task", "--stage", "medium", "--bundle", flat,
                   "--target", "this")
        assert r.exit_code == 0, r.output
        kept_after = {n: _as_written(flat / n) for n in kept}

        # ONE MOVE ALLOWED, AND NO RETRY: the relaxation ends out of moves
        # (`running-a-job.md` § 3.5), and that run is the one asked about.
        capped = tree / "P" / "opt" / "H2cap"
        r = jobset("init", "--structure", "P/structure/h2.xyz",
                   "--bundle", "P/opt/H2cap", "--engine", "siesta",
                   "--calculation", "optimization", "--shape", "flat",
                   "--name", "H2", "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        _one_rank(capped, restart="clean")
        set_template_value(capped / "H2.template.toml", "relax_steps", "1")
        set_template_value(capped / "H2.template.toml", "continue_retries",
                           "0")
        r = jobset("prep", "task", "--stage", "coarse", "--bundle", capped,
                   "--target", "this")
        assert r.exit_code == 0, r.output
        jobset("launch", "task", "--stage", "coarse", "--bundle", capped,
               "--mode", "direct", "--yes")
        assert (capped / "H2_01_coarse-run0.out").is_file(), \
            sorted(p.name for p in capped.iterdir())
        # LAUNCHED AGAIN, as its run 1 -- and run 0's own files as they were
        # (`project-layout.md` § 1.5a: re-running never overwrites).
        run0 = {p.name: p.read_bytes()
                for p in capped.glob("H2_01_coarse-run0*") if p.is_file()}
        again = jobset("launch", "task", "--stage", "coarse", "--bundle",
                       capped, "--mode", "direct", "--yes")
        assert (capped / "H2_01_coarse-run1.out").is_file(), again.output
        run0_after = {n: (capped / n).read_bytes()
                      if (capped / n).is_file() else None for n in run0}
        # AN EARLIER RUN'S OUTPUT THE NEWER FILE, as a copy or a restore
        # leaves a folder's times -- which reorder files, never runs.
        later = time.time() + 120
        os.utime(capped / "H2_01_coarse-run0.out", (later, later))
        yield {"flat": flat, "capped": capped, "again": again.output,
               "run0": run0, "run0_after": run0_after,
               "kept_before": kept_before, "kept_after": kept_after}


@pytest.fixture(scope="module")
def flat(road):
    return road["flat"]


def _out(flat: Path) -> Path:
    return flat / "H2_01_coarse-run0.out"


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
#  The folder, as the Results tab asks it (`model/parse.md` § 5)          #
# --------------------------------------------------------------------- #

def test_the_folder_answers_for_the_stage_that_ran(flat):
    """Two stages' decks lie in the one folder; the folder speaks for the
    stage that ran (§ 5.1), the calculation decides what opens -- an
    optimization's engine output, the run's own ``-run0.out``, before the
    progress log prep seeded beside it (§ 5.2, § 5.5) -- and every name is
    read back with the run's label: SIESTA's files, named by ``SystemLabel``,
    are the engine's, the decks their stages' own (`job-contracts.md`
    § 2.2a)."""
    from molbuilder.runs import folder_answer
    got = folder_answer(flat)
    assert got["place"]["role"] == "run", got["place"]
    assert Path(got["openable"]).name == "H2_01_coarse-run0.out", (
        got["attempts"])
    assert got["status"]["state"] == "finished", got["status"]
    assert got["record"]["deck"]["path"] == "H2_01_coarse.fdf", (
        got["record"]["deck"])
    files = {f["name"]: f for f in got["files"]}
    engines = [p.name for p in flat.glob("fdf.*.log")] + [
        "H2.XV", "H2.xyz", "H2.MD.nc"]
    for name in engines:
        assert (files[name]["role"], files[name]["about"]) == (
            None, {"ours": False}), files[name]
    for name, stage in (("H2_01_coarse.fdf", "01_coarse"),
                        ("H2_02_medium.fdf", "02_medium")):
        assert (files[name]["role"], files[name]["stage"],
                files[name]["about"]["ours"]) == (".fdf", stage, True), (
            files[name])


def test_each_run_reads_its_own_deck_and_nothing_else(flat, monkeypatch):
    """The Results load of the run shows atom 0 held in the isolated 10 Å box
    this module wrote; each of the folder's two stages reads its own deck
    (user, 2026-10-04: *"make sure that it does make each run sees its own
    .fdf"*); and SIESTA's own ``H2.xyz``, opened as a structure, carries the
    same box at the engine's origin (`model/structure-periodicity.md`
    § 6.0)."""
    from molbuilder import diagnostics
    from molbuilder.runs import declared, run_of
    from molbuilder.web.app import create_app
    from molbuilder.workingcopy_structure import StructureCodec
    root = next(p for p in flat.parents if p.name == "projects")
    monkeypatch.setattr(type(diagnostics.get_capabilities()),
                        "file_picker_roots",
                        lambda self: ((root.resolve(), "projects"),))
    d = create_app(config={}).test_client().post(
        "/api/watch/load", json={"path": str(flat)}).get_json()
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
    # ITS OWN DECK STATES ITS CONTRACT, though the folder holds two decks.
    calc = d["info"]["calculation"]
    assert calc["source"] == "H2_01_coarse.fdf", calc
    assert calc["contract"]["basis_size"] == _template_value(
        flat, "basis_size"), calc

    assert (declared(run_of(_out(flat))).deck.name == "H2_01_coarse.fdf")
    assert (declared(run_of(flat, stage="02_medium")).deck.name
            == "H2_02_medium.fdf")

    own = StructureCodec().load(flat / "H2.xyz")
    assert own.cell is not None, "the run's box never reached its H2.xyz"
    np.testing.assert_allclose(np.asarray(own.cell), np.eye(3) * _BOX,
                               atol=1e-4)
    np.testing.assert_allclose(own.engine_offset, np.zeros(3))


# --------------------------------------------------------------------- #
#  The next stage, and a flat stage launched again (job-system.md § 5.4)  #
# --------------------------------------------------------------------- #

def test_the_next_stages_prep_keeps_the_run_folders_own_files(road):
    """The earlier stage's deck and run script are the run folder's own,
    which ran: `medium`'s prep writes neither again."""
    assert road["kept_after"] == road["kept_before"]


def test_every_name_the_card_gives_a_stage_is_on_disk(road):
    """A launched stage's names for its prep, its launch and its run, and
    the stage prepared after it -- in the flat folder (`job-contracts.md`
    § 2.2; the layered one: `test_the_road_on_real_runs_e2e.py`)."""
    from support.road import _road_card_written
    _road_card_written([{"stage": "coarse", "moments": ["prep", "launch", "run"]},
                        {"stage": "medium", "moments": ["prep"]}],
                       road["flat"])


def test_launched_again_with_restart_clean_it_runs_again_taking_nothing(road):
    """The run card says start over: launched again, the stage runs in its
    folder as its run 1, carrying nothing from run 0."""
    assert ("launched again in the same folder, as its run 1, where its "
            "files are: it starts over") in road["again"], road["again"]
    assert not (road["capped"] / "H2_01_coarse-run1.continued-from").exists()


def test_a_run_launched_again_overwrites_nothing_of_the_run_before(road):
    """Every file run 0 wrote under its index is as it was after run 1, and
    run 1 wrote its own (`project-layout.md` § 1.5a)."""
    capped = road["capped"]
    assert road["run0_after"] == road["run0"], sorted(
        n for n in road["run0"] if road["run0_after"][n] != road["run0"][n])
    for suffix in (".out", ".monitor.log"):
        assert (capped / f"H2_01_coarse-run0{suffix}").is_file(), suffix
        assert (capped / f"H2_01_coarse-run1{suffix}").is_file(), suffix


def test_the_folder_speaks_for_its_newest_run_index(road):
    """Run 0's output is the newer file -- as a copy or a restore leaves a
    folder's times -- and the folder still speaks for run 1 (`model/parse.md`
    § 5.1: a file's time decides nothing)."""
    from molbuilder.runs import folder_answer
    got = folder_answer(road["capped"])
    assert got["status"]["active_source"] == "H2_01_coarse-run1.out", (
        got["status"])
    assert Path(got["openable"]).name == "H2_01_coarse-run1.out", (
        got["attempts"])


# --------------------------------------------------------------------- #
#  The run's .XV (`parse/coords.py`, `molbuilder xv2xyz`)                 #
# --------------------------------------------------------------------- #

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


def test_the_xv_reader_reads_what_siesta_wrote(flat):
    """Elements keyed off Z, coordinates and the cell converted Bohr -> Å,
    and a file that states a lattice yields a structure that has one -- its
    axis kinds `Structure`'s own (`model/structure-periodicity.md` § 2)."""
    from molbuilder.constants import BOHR_ANGSTROM
    from molbuilder.parse.coords import read_xv, read_xv_cell, \
        read_xv_with_cell
    xv = flat / "H2.XV"
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


def test_xv2xyz_writes_the_pair_and_the_next_deck_takes_its_frame(flat,
                                                                  tmp_path):
    """A bare ``.XV`` becomes the PAIR, so the cell has somewhere to go, and
    what the deck generator reopens is the run's frame: the cell, and the
    coordinates at the engine's origin -- a stated offset of 0
    (`model/structure-periodicity.md` § 6.0)."""
    from click.testing import CliRunner

    from molbuilder import cli
    from molbuilder.cell import to_engine
    from molbuilder.parse.coords import read_xv
    from molbuilder.siesta.input import _struct_from_file
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(flat / "H2.XV"),
                                       str(out)])
    assert res.exit_code == 0, res.output
    assert out.is_file() and (tmp_path / "h2.molstruct.json").is_file()
    assert out.read_text().splitlines()[0].strip() == "2"
    s, cell = _struct_from_file(str(out))
    np.testing.assert_allclose(np.asarray(cell), np.eye(3) * _BOX, atol=1e-4)
    assert s.axis_kind == ("periodic", "periodic", "periodic")
    frame = to_engine(s)
    assert frame.stated, "the pair lost the .XV's origin"
    np.testing.assert_allclose(frame.positions,
                               read_xv(flat / "H2.XV").positions, atol=1e-6)


def test_xv2xyz_from_run_takes_what_the_run_declared(flat, tmp_path):
    """``--from-run``: the atom the run's deck holds and the axes its
    structure stated -- isolated, the 10 Å box written here -- on the
    ``.XV``'s own cell at the engine's origin; the run's deck is the one the
    run door names, of the folder's two (`runs.run_of`, `runs.declared`;
    plan D19)."""
    from click.testing import CliRunner

    from molbuilder import cli
    from molbuilder.workingcopy_structure import StructureCodec
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(flat / "H2.XV"),
                                       str(out), "--from-run"])
    assert res.exit_code == 0, res.output
    assert "H2_01_coarse.fdf" in res.output, res.output
    got = StructureCodec().load(out)
    assert got.frozen_atoms == [0]
    assert got.axis_kind == ("isolated", "isolated", "isolated")
    np.testing.assert_allclose(np.asarray(got.cell), np.eye(3) * _BOX,
                               atol=1e-4)
    np.testing.assert_allclose(got.engine_offset, np.zeros(3))


def test_xv2xyz_from_run_refuses_a_xv_no_run_of_ours_holds(flat, tmp_path):
    """The flag says a run is there: the run's ``.XV`` taken out to a folder
    no calculation marks has nothing declaring its atoms, so it is refused
    by name -- never written at the defaults as if a run had said so."""
    import shutil

    from click.testing import CliRunner

    from molbuilder import cli
    away = tmp_path / "H2.XV"
    shutil.copy(flat / "H2.XV", away)
    out = tmp_path / "h2.xyz"
    res = CliRunner().invoke(cli.cli, ["xv2xyz", str(away), str(out),
                                       "--from-run"])
    assert res.exit_code != 0
    assert "no run of ours holds" in res.output, res.output
    assert not out.exists()


# --------------------------------------------------------------------- #
#  The output and its companions (`model/parse.md` § 2a, § 5.3, § 6)      #
# --------------------------------------------------------------------- #

def test_the_registry_reads_the_output_as_siestas(flat):
    from molbuilder.parse import TrajectoryResult, detect, parse
    from molbuilder.parse.engines import SiestaOutFileParser
    assert detect(_out(flat)) is SiestaOutFileParser
    res = parse(_out(flat))
    assert isinstance(res, TrajectoryResult)
    assert (res.result_kind, res.source_format, res.parser_name) == (
        "trajectory", "siesta", "siesta")
    assert res.frames


def test_the_convergence_targets_are_the_decks_read_from_its_echo(flat):
    """What the run was asked to reach, from SIESTA's own echo of the deck --
    never a default reproduced: the force tolerance and the SCF cap the
    coarse deck states."""
    from molbuilder.parse.engines.siesta import SiestaParser
    deck = flat / "H2_01_coarse.fdf"
    ct = SiestaParser.parse(str(_out(flat))).runtime_info.get(
        "convergence_targets")
    assert ct is not None and ct.get("source") == "siesta_input_echo", ct
    assert ct.get("max_force_tol_eV_per_A") == pytest.approx(
        float(_deck_value(deck, "MD.MaxForceTol"))), ct
    assert ct.get("max_scf_iter") == int(_deck_value(deck,
                                                     "MaxSCFIterations")), ct


def test_the_run_states_the_build_it_ran_and_its_solver(flat):
    """The header SIESTA printed and the solver it resolved
    (`siesta_grammar.read_build_line`, ``read_diag_line``): the version and
    the MPI build the env's recipe pins (`envs/recipes.py`,
    ``siesta=5.4.2=mpi_openmpi_*``), and no token SIESTA does not write."""
    from molbuilder.envs.recipes import _SIESTA
    from molbuilder.parse.engines.siesta import SiestaParser
    spec = next(p.spec for p in _SIESTA.conda_packages
                if p.spec.startswith("siesta="))
    _, version, build = spec.split("=")
    traj = SiestaParser.parse(str(_out(flat)))
    got = traj.runtime_info["siesta_build"]
    assert got["version"].startswith(version), got
    assert build.startswith("mpi") and "MPI" in got["parallelisations"], got
    assert traj.runtime_info["siesta_diag"].get("algorithm"), traj.runtime_info
    assert [w for w in traj.parse_warnings
            if w.category == "runtime_info"] == []


def test_a_relaxations_step_boundaries_are_not_iterations(flat):
    """The run's SCF-timing log, ``<epoch> <iscf> <row>`` a line: the delta
    into a row whose iteration is 1 spans the previous SCF's end, the forces,
    the move and the next step's setup, and the phase's first delta can
    carry warm-up -- neither is an iteration (`scf_timing_rows`,
    `model/parse.md` § 5c).

    MUTATION THIS MUST FAIL AGAINST: keep the deltas into iteration 1."""
    from molbuilder.parse.instruments.scf_timing_rows import \
        scf_timing_metrics
    text = (flat / "H2_01_coarse-run0.scf-timing.log").read_text()
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


def test_the_engines_own_log_says_what_it_used_and_what_nobody_set(flat):
    """SIESTA's ``fdf.<stamp>.log`` (`model/parse.md` § 5d.2-5d.3): every
    key SIESTA read, with the value it read -- a key the deck carries reads
    as set, any other as SIESTA's default; a block is found by the deck's
    own spelling, fdf's label rule."""
    from molbuilder.parse import detect
    from molbuilder.parse.fdf import _norm
    deck = flat / "H2_01_coarse.fdf"
    stated, blocks = set(), set()
    for ln in deck.read_text().splitlines():
        words = ln.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].lower() == "%block":
            blocks.add(_norm(words[1]))
        elif not words[0].startswith("%"):
            stated.add(_norm(words[0]))
    [log] = list(flat.glob("fdf.*.log"))
    res = detect(log).parse(log)
    assert res.result_kind == "engine-params"
    one_reading = {k: v for k, v in res.params.items() if "default" in v}
    assert one_reading, res.params
    for key, p in one_reading.items():
        assert p["default"] is (key not in stated), (key, p)
    assert any(p["default"] for p in one_reading.values()), (
        "SIESTA read no key the deck left to it")
    mesh = res.params[_norm("MeshCutoff")]
    assert (mesh["number"], mesh["unit"], mesh["default"]) == (
        float(_deck_value(deck, "MeshCutoff")), "Ry", False), mesh
    for name in blocks:
        assert name in res.blocks, (name, sorted(res.blocks))


# --------------------------------------------------------------------- #
#  The MD history SIESTA names by its label (`model/parse.md` § 6)        #
# --------------------------------------------------------------------- #

def _md_nc(flat: Path) -> Path:
    return flat / "H2.MD.nc"


def test_the_history_is_found_by_the_label_the_output_prints(flat):
    """The output carries the stage and the run index, ``H2_01_coarse-run0
    .out``; SIESTA names the history ``H2.MD.nc`` and prints the label --
    which is how the output finds its own (`siesta_mdnc.sibling_md_nc`,
    § 5.3).

    MUTATION THIS MUST FAIL AGAINST: the label line not read -- the
    trajectory falls back to the output's four-decimal coordinates."""
    from molbuilder.parse import parse
    res = parse(str(_out(flat)))
    assert res.runtime_info.get("mdnc_source") == "H2.MD.nc", res.runtime_info
    assert res.runtime_info.get("mdnc_coords_upgraded", 0) >= 1, (
        res.runtime_info)


def test_the_history_is_claimed_by_its_reader_and_routed_there(flat):
    """Its reader claims the ``.MD.nc`` and declines the output beside it --
    the other half of the same run, whose parser reads its run state and its
    SCF -- and the registry routes the history to it."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import SiestaMdNcFileParser
    assert SiestaMdNcFileParser.can_parse(_md_nc(flat))
    assert not SiestaMdNcFileParser.can_parse(_out(flat))
    res = parse(str(_md_nc(flat)))
    assert (res.parser_name, res.source_format) == ("siesta-mdnc",
                                                    "siesta-mdnc")


def test_the_history_says_only_what_it_holds(flat):
    """No run state -- a restart appends to the file, so a complete-looking
    history may belong to a live run -- no forces and no SCF; coordinates
    converted from Bohr by the file's own attribute (an H2 bond, Å)."""
    from molbuilder.parse import parse
    res = parse(str(_md_nc(flat)))
    assert res.run_state == "unknown"
    assert res.frames
    assert all(f.forces is None and f.scf_history is None
               for f in res.frames)
    for f in res.frames:
        bond = float(np.linalg.norm(f.structure.positions[1]
                                    - f.structure.positions[0]))
        assert 0.5 < bond < 1.3, f"bond {bond} Å -- the unit looks wrong"


def test_a_frames_geometry_and_energy_belong_to_one_step(flat):
    """A history row holds the geometry about to be tried beside the energy
    of the one just evaluated; the reader pairs each energy with its own
    geometry -- checked against the output of the same run, to the output's
    printing precision -- and the last geometry, not yet evaluated, has no
    energy.  The per-step series move with the energy."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    nc = parse(str(_md_nc(flat)))
    out = parse(str(_out(flat)))
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


def test_the_output_keeps_its_frames_and_the_history_matches_in_order(flat):
    """The output defines the frame list and opens with the input geometry,
    which the history never stores; the history only upgrades the frames it
    matches, in its own order, each row once (§ 6)."""
    from molbuilder.parse import parse
    from molbuilder.parse.engines.siesta_mdnc import align_to_reference
    out = parse(str(_out(flat)))
    nc = parse(str(_md_nc(flat)))
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


def test_the_history_reads_the_same_without_netCDF4(flat, monkeypatch):
    """Two backends, interchangeable: netCDF4 when present, scipy.io -- a
    hard dependency -- otherwise."""
    import builtins
    import sys

    from molbuilder.parse.engines.siesta_mdnc import SiestaMdNcFileParser
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "netCDF4":
            raise ImportError("simulated: netCDF4 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "netCDF4", raising=False)
    viascipy = SiestaMdNcFileParser.parse(_md_nc(flat))
    monkeypatch.undo()
    native = SiestaMdNcFileParser.parse(_md_nc(flat))
    assert len(viascipy.frames) == len(native.frames)
    for a, b in zip(viascipy.frames, native.frames):
        assert a.structure.elements == b.structure.elements
        np.testing.assert_allclose(a.structure.positions,
                                   b.structure.positions)
        assert (a.energy is None) == (b.energy is None)
        if a.energy is not None:
            assert abs(a.energy - b.energy) < 1e-12


# --------------------------------------------------------------------- #
#  How a run ended -- what the wrapper's retries ask (§ 3.5)             #
# --------------------------------------------------------------------- #

def test_a_relaxation_out_of_moves_is_capped_and_no_scf_stop(road):
    """One move allowed: SIESTA exits 0 printing ``Final (unrelaxed)``
    coordinates -- the ending the geometry retry asks for, on the zero-exit
    path -- and not an SCF stop (`running-a-job.md` § 3.5)."""
    from molbuilder.parse.engines._run_ending import QUESTIONS, ending_of
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
    end = ending_of(road["capped"] / "H2_01_coarse-run0.out")
    assert QUESTIONS["relaxation-capped"](end), end
    assert not QUESTIONS["stopped-by"](end, SCF_NOT_CONV_MARKER), end
