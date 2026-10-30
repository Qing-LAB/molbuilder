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
module's own whose relaxation is allowed one move and one warm retry, so it
ends out of moves and its run script runs it again in place: its run holds
two tries, ``-run0`` and ``-run1`` (`running-a-job.md` § 3.5;
`project-layout.md` § 1.6.1).

Every expectation is what the run was given -- the structure written here,
the deck prep wrote, the template, the env's pinned build -- or a rule of the
contract applied to the run's own files; never a number a reader printed.
The readers are tested on this run, made with the engine (user, 2026-10-06:
"when a test need siesta's output why is it not part of a e2e test?"), and
every file of ours is asked of the run door (`runs.run_of`).
"""
from __future__ import annotations

import json
import os
import time
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
    """The capped calculation's coarse run, in a projects tree of its own:
    ONE MOVE ALLOWED, AND ONE WARM RETRY -- the relaxation ends out of
    moves, and the run script runs it again in place as its next try
    (`running-a-job.md` § 3.5): two tries in one run folder."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runs import run_of
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
        # ...its retry budget the run card's, a run setting.
        one_rank(bundle, continue_retries=1)
        set_template_value(bundle / "H2.template.toml", "relax_steps", "1")
        r = jobset("prep", "task", *group_words(bundle, "coarse"),
                   "--bundle", bundle, "--target", "this")
        assert r.exit_code == 0, r.output
        jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--foreground", "--yes")
        folder = latest_attempt(stage_home(bundle, task_of(bundle),
                                           "coarse").dir)
        assert folder is not None, sorted(p.name for p in bundle.rglob("*"))
        outs = run_of(folder).outputs
        assert [p.name for p in outs] == ["H2_01_coarse-run1.out",
                                          "H2_01_coarse-run0.out"], \
            sorted(p.name for p in folder.iterdir())
        # AN EARLIER TRY'S OUTPUT THE NEWER FILE, as a copy or a restore
        # leaves a folder's times -- which reorder files, never tries.
        later = time.time() + 120
        os.utime(outs[-1], (later, later))
        yield folder


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
    SIESTA's files, named by ``SystemLabel``, are the engine's, the deck and
    the run's history its stage's own (`job-contracts.md` § 2.2a)."""
    from molbuilder.runs import folder_answer
    got = folder_answer(opt)
    assert got["place"]["role"] == "run", got["place"]
    assert Path(got["openable"]).name == "H2_01_coarse-run0.out", (
        got["attempts"])
    assert got["status"]["state"] == "finished", got["status"]
    assert got["record"]["deck"]["path"] == "H2_01_coarse.fdf", (
        got["record"]["deck"])
    files = {f["name"]: f for f in got["files"]}
    engines = [p.name for p in opt.glob("fdf.*.log")] + ["H2.XV", "H2.xyz"]
    for name in engines:
        assert (files[name]["role"], files[name]["about"]) == (
            None, {"ours": False}), files[name]
    for name, role in (("H2_01_coarse.fdf", ".fdf"),
                       ("H2_01_coarse-run0.MD.nc", ".MD.nc")):
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
    assert rt.get("mdnc_source") == "H2_01_coarse-run0.MD.nc", rt
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
#  The output and its companions (`model/parse.md` § 2a, § 5.3, § 6)      #
# --------------------------------------------------------------------- #

def test_the_registry_reads_the_output_as_siestas(opt):
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
    `model/parse.md` § 5c).

    MUTATION THIS MUST FAIL AGAINST: keep the deltas into iteration 1."""
    from molbuilder.parse.instruments.scf_timing_rows import \
        scf_timing_metrics
    text = _run(opt).file(".scf-timing.log", 0).read_text()
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


# --------------------------------------------------------------------- #
#  SIESTA's history, each run's own (`job-contracts.md` § 2.2;           #
#  `model/parse.md` § 5a)                                                #
# --------------------------------------------------------------------- #

def _md_nc(folder: Path) -> Path:
    return _run(folder).file(".MD.nc", 0)


def test_the_history_is_found_by_the_output_s_own_name(opt):
    """The output carries the stage and the try's number, ``H2_01_coarse-run0
    .out``, and the run's history the same, ``H2_01_coarse-run0.MD.nc`` --
    which is how the output finds its own, read back with the label it prints
    as the reading pass read it (`siesta_mdnc.sibling_md_nc`, § 4a, § 5.3)
    -- and the output's frames are upgraded from it.

    MUTATION THIS MUST FAIL AGAINST: the run's file not composed from the
    output's name -- the trajectory keeps the output's four-decimal
    coordinates."""
    from molbuilder.parse import parse
    res = parse(str(_out(opt)))
    assert res.runtime_info.get("mdnc_source") == _md_nc(opt).name, (
        res.runtime_info)
    assert res.runtime_info.get("mdnc_coords_upgraded", 0) >= 1, (
        res.runtime_info)


def test_every_try_s_history_is_its_own(capped):
    """Each try writes its steps into a history of its own, whatever came
    before it in the folder: the capped run's first try, and the warm retry
    the run script ran in the same folder after it -- each try's rows are
    its own output's steps, and SIESTA's shared name is not left behind
    (`job-contracts.md` § 2.2; a stage's runs across folders,
    `test_the_road_on_real_runs_e2e.py`).

    MUTATION THIS MUST FAIL AGAINST: the run script laying no pointer --
    SIESTA appends both tries to ``H2.MD.nc`` and neither has a history of
    its own."""
    from support.road import history_is_the_runs_own
    outs = _run(capped).outputs
    assert [p.name for p in outs] == ["H2_01_coarse-run1.out",
                                      "H2_01_coarse-run0.out"], outs
    for out in outs:
        history_is_the_runs_own(out, "H2")


def test_the_run_folder_speaks_for_its_newest_try(capped):
    """The first try's output is the newer file -- as a copy or a restore
    leaves a folder's times -- and the run's folder still speaks for its
    newest try, by the number every file of it carries (`model/parse.md`
    § 5.1: a file's time decides nothing; `project-layout.md` § 1.6.1)."""
    from molbuilder.runs import folder_answer
    got = folder_answer(capped)
    assert got["status"]["active_source"] == "H2_01_coarse-run1.out", (
        got["status"])
    assert Path(got["openable"]).name == "H2_01_coarse-run1.out", (
        got["attempts"])


def test_a_warm_retry_continues_and_only_the_last_try_concludes(capped):
    """`running-a-job.md` § 3.5: out of moves, the run script runs itself
    again in the same run folder, with ``--continue`` and the next try's
    number -- each try with a session log of its own, the first saying it
    started clean and the retry that it resumes warm, in the banner's own
    words (`job-contracts.md` § 4.4) -- and only the last try writes the
    conclusion marker (`project-layout.md` § 1.6.3).  (Each try's output
    carrying its own number is the fixture's own check.)

    THE SILENT FAILURES: a retry that starts cold, its relaxation begun
    again from the structure; a first try concluding the run before its
    retry ran, so a reader takes the run as ended mid-retry.  MUTATION THIS
    MUST FAIL AGAINST: the retry run again without ``--continue``."""
    from molbuilder.wrapper_log import log_of_run
    run = _run(capped)
    assert run.file(".concluded", 0) is None, sorted(
        p.name for p in capped.iterdir())
    assert run.file(".concluded", 1) is not None, sorted(
        p.name for p in capped.iterdir())
    logs = [log_of_run(capped, run.label, n, run.stage) for n in (0, 1)]
    assert None not in logs, logs
    first, retry = (log.read_text(errors="replace") for log in logs)
    assert "initial-run (clean state)" in first, logs[0].name
    assert "WARM-RESUME (--continue; engine will load" in retry, logs[1].name


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
    matches, in its own order, each row once (§ 6)."""
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
#  How a run ended -- what the wrapper's retries ask (§ 3.5)             #
# --------------------------------------------------------------------- #

def test_a_relaxation_out_of_moves_is_capped_and_no_scf_stop(capped):
    """One move allowed: SIESTA exits 0 printing ``Final (unrelaxed)``
    coordinates -- the ending the geometry retry asks for, on the zero-exit
    path -- and not an SCF stop (`running-a-job.md` § 3.5)."""
    from molbuilder.parse.engines._run_ending import QUESTIONS, ending_of
    from molbuilder.parse.engines.siesta_grammar import SCF_NOT_CONV_MARKER
    end = ending_of(_run(capped).file(".out", 0))
    assert QUESTIONS["relaxation-capped"](end), end
    assert not QUESTIONS["stopped-by"](end, SCF_NOT_CONV_MARKER), end
