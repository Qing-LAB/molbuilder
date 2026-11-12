"""What a person's script reads of a finished PySCF relaxation, and what `jobset
export` saves of it (`engines/optimization-postprocessing.md`; `process/
testing.md` tier 3): the steps the run's own writer logged, on a real run of
water with its oxygen held (`support.real_runs.held_water_pyscf`).
"""
from __future__ import annotations

import numpy as np
import pytest

from _road import conda_hook, env_available
from molbuilder.runfiles import ROLE_MOLWATCH, ROLE_PYSCF_LOG
from support.road import molbuilder

#: The run's facts geomeTRIC's trajectory takes from its progress log
#: (`model/parse.md` § 5a's table): the convergence targets, the SCF's
#: criteria and what each SCF row measures, the atoms the run holds.
LOG_FACTS = ("frozen_atoms", "convergence_targets", "scf_criteria",
             "scf_residuals")

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-pySCF env + a detectable conda hook"),
]


def test_a_person_s_script_reads_the_steps_a_run_logged(real_held_water_pyscf):
    """The input geometry the log opens with is no step: the first step stands
    from the person's structure by the offset the deck states.  The largest
    force is asked over every atom and over the atoms the run moved, which a
    held atom feeling the largest force tells apart; the elements and the atoms
    held are the deck's, in the person's order.

    THE SILENT FAILURE: an atom's row read as another's, or the largest force a
    person reads taken over the wrong atoms.  (That a convergence is judged on
    the atoms moved is `test_siesta_relax_run_e2e.py`'s, where the held atom ends
    above its tolerance, which this run's does not.)"""
    from molbuilder.postprocess import read_steps, read_structure
    run = real_held_water_pyscf
    steps = read_steps(run.attempt("coarse"))
    mine = read_structure(next(run.bundle.glob("*.source.xyz")))[0]
    assert steps.atom_index_map.listed_by == "user_index"
    assert list(steps.elements) == list(mine.elements)
    assert list(steps.held_atoms) == [0]
    forces = steps.forces_ev_per_ang
    assert steps.positions_ang.shape == (len(steps.energy_ev), 3, 3)
    assert np.allclose(steps.positions_ang[0],
                       mine.positions + np.array(steps.applied_offset_ang))
    assert np.allclose(steps.max_force_ev_per_ang,
                       np.abs(forces).max(axis=(1, 2)))
    assert np.allclose(steps.max_force_free_ev_per_ang,
                       np.abs(forces[:, [1, 2]]).max(axis=(1, 2)))
    assert steps.max_force_ev_per_ang[0] > steps.max_force_free_ev_per_ang[0], (
        "the held oxygen must feel the largest force, or this cannot tell "
        "the two figures apart")
    assert steps.relaxation["max_force_free_ev_ang"] == pytest.approx(
        steps.max_force_free_ev_per_ang[-1])
    assert steps.final_energy_ev == pytest.approx(steps.energy_ev[-1])
    assert np.allclose(steps.last.positions, steps.positions_ang[-1])


def test_a_pair_saved_beside_a_run_s_files_is_not_read_as_its_steps(
        real_held_water_pyscf):
    """`runs.StepsTable`: a pair is in the person's order already, and crossing
    it again would stand every atom on another's place."""
    from molbuilder.postprocess import RunStepsError, read_steps
    from molbuilder.workingcopy_structure import StructureCodec
    run = real_held_water_pyscf.attempt("coarse")
    steps = read_steps(run)
    pair = run / "saved.xyz"
    StructureCodec().write(steps.last, pair)
    try:
        with pytest.raises(RunStepsError,
                           match="no file its run writes its steps"):
            read_steps(pair)
    finally:
        pair.unlink()
        pair.with_suffix(".molstruct.json").unlink()


def test_the_export_holds_the_steps_and_says_why_a_run_holds_none(
        real_held_water_pyscf, monkeypatch):
    """`engines/optimization-postprocessing.md` § 6: every row is what
    `read_steps` hands back, the map on the file, nothing written into the
    calculation -- and a run with no step stands on the axis at its fill, saying
    why, never read as a rung that was not asked.  Refused inside the
    calculation's folder."""
    from conftest import point_the_one_door_at
    from molbuilder.postprocess import (list_runs, read_optimization_data,
                                        read_steps)
    from molbuilder.runs import run_of
    run = real_held_water_pyscf
    bundle = run.bundle
    point_the_one_door_at(run.tree.parent, monkeypatch.setenv)
    inside = molbuilder("jobset", "export", "--format", "netcdf", "--to",
                        bundle / "exported", "--bundle", bundle)
    assert inside.exit_code != 0 and "is inside the calculation" in inside.output
    assert not (bundle / "exported").exists()
    to = bundle.parent / "exported"
    for form in ("netcdf", "csv"):
        said = molbuilder("jobset", "export", "--format", form, "--to", to,
                          "--bundle", bundle)
        assert said.exit_code == 0, said.output
    assert not list(bundle.rglob("*.optimization*"))
    finished, unstarted = list_runs(bundle)
    assert finished == run.attempt("coarse")
    assert unstarted == run.attempt("medium")
    steps = read_steps(finished)
    label = run_of(finished).label
    assert (to / f"{label}.optimization-csv" / "step_atom_xyz.csv").is_file()
    data = read_optimization_data(to / f"{label}.optimization.nc")
    assert data["atom_index_map"].listed_by == "user_index"
    got = {k: v["data"] for k, v in data["variables"].items()}
    assert list(got["run"]) == [p.relative_to(bundle).as_posix()
                                for p in (finished, unstarted)]
    assert list(got["rung_of_run"]) == ["coarse", "medium"]
    assert list(got["run_of_step"]) == [got["run"][0]] * len(steps.energy_ev)
    assert "holds no step its run finished" in got["said"][1]
    assert np.isnan(got["final_energy_ev"][1])
    assert got["force_within_tolerance"][1] is None
    assert list(got["element"]) == list(steps.elements)
    assert np.allclose(got["position_ang"], steps.positions_ang)
    assert np.allclose(got["force_ev_per_ang"], steps.forces_ev_per_ang)
    assert np.allclose(got["energy_ev"], steps.energy_ev)
    assert np.allclose(got["max_force_free_ev_per_ang"],
                       steps.max_force_free_ev_per_ang)
    assert np.allclose(got["applied_offset_ang"][0], steps.applied_offset_ang)
    assert [i for i, h in enumerate(got["held"][0]) if h] == [0]
    assert got["force_within_tolerance"][0] is steps.relaxation["converged"]


def test_the_trajectory_beside_a_run_s_progress_log_and_the_atom_index_door_read_it(
        real_held_water_pyscf):
    """`model/parse.md` § 5a: geomeTRIC's trajectory states no targets, no
    atoms held, no SCF criteria and no ending; the run's progress log beside
    it does, and the trajectory carries them as the log's own parser states
    them -- the plots' threshold lines, the held atoms and the run's ending
    the same whichever of the two files a person opens -- and it says which
    files it took them from (rule 4): the progress log, and PySCF's own log,
    counting the steps it gave an SCF history.  The atom index door answers the
    log by the map its header states (`model/overview.md` § 2.2b), not by
    its run's deck.

    THE SILENT FAILURE: the two files of one run telling a person two
    things -- the held oxygen moving in one, the run still going in one, a
    threshold line missing from one; a number on the trajectory's plots no
    file can be traced to; the log's map passed over for its run's deck."""
    from molbuilder.atom_index import atom_index_map
    from molbuilder.parse import parse
    from molbuilder.runfiles import ROLE_GEOM_TRAJ
    from molbuilder.runs import run_of
    run = run_of(real_held_water_pyscf.attempt("coarse"))
    traj, log = run.file(ROLE_GEOM_TRAJ), run.file(ROLE_MOLWATCH)
    of_traj, of_log = parse(traj), parse(log)
    assert of_traj.source_format == "pyscf"
    assert of_log.runtime_info["frozen_atoms"] == [0]
    assert of_log.runtime_info["convergence_targets"]
    assert of_log.run_state == "ended"
    for fact in LOG_FACTS:
        assert of_traj.runtime_info.get(fact) == of_log.runtime_info[fact], (
            f"{traj.name} and {log.name} state {fact} two ways")
    assert ((of_traj.run_state, of_traj.error_message)
            == (of_log.run_state, of_log.error_message))
    said = of_traj.runtime_info
    # The count is the observable the key names: the steps carrying an SCF
    # history, which none does without PySCF's log.
    stepped = sum(1 for f in of_traj.frames if f.scf_history is not None)
    assert (said["molwatch_source"], said["pyscf_log_source"]) == (
        log.name, run.file(ROLE_PYSCF_LOG).name), said
    assert said["pyscf_log_scf_upgraded"] == stepped > 0, (said, stepped)
    assert atom_index_map(log).stated_by == str(log)


@pytest.mark.parametrize("role, keys, alone", [
    (ROLE_MOLWATCH, "molwatch_",
     lambda r: (r.run_state, r.error_message) == ("unknown", None)
     and not any(k in r.runtime_info for k in LOG_FACTS)),
    (ROLE_PYSCF_LOG, "pyscf_log_",
     lambda r: all(f.scf_history is None for f in r.frames)),
], ids=["the progress log", "PySCF's own log"])
def test_a_file_beside_the_trajectory_that_does_not_read_is_said(
        real_held_water_pyscf, tmp_path, role, keys, alone):
    """`model/parse.md` § 5a: a companion that does not read leaves the
    trajectory read as itself alone (rule 2), and the trajectory says which
    file gave it nothing, and why (rule 4).  The run's own files, replayed
    in a folder of their own (`support.replay`), the companion made
    unreadable there -- a real read failure, as a disk or a permission
    gives one.

    THE SILENT FAILURE: the plots' threshold lines, the held atoms, the
    run's ending or the SCF chart gone with no word of which file or why."""
    import errno
    import os
    if os.geteuid() == 0:
        pytest.skip("root reads a file whose mode lets no one read it")
    from molbuilder.parse import parse
    from molbuilder.runfiles import ROLE_GEOM_TRAJ
    from molbuilder.runs import run_of
    from support.replay import output_lines, replay
    run = real_held_water_pyscf.attempt("coarse")
    ran = run_of(run)
    there, _watched, _grow = replay(run, tmp_path, len(output_lines(run)))
    companion = there / ran.file(role).name
    companion.chmod(0)
    got = parse(there / ran.file(ROLE_GEOM_TRAJ).name)
    said = {k: v for k, v in got.runtime_info.items() if k.startswith(keys)}
    assert list(said) == [keys + "error"], said
    why = said[keys + "error"]
    assert (why.startswith(companion.name)
            and os.strerror(errno.EACCES) in why), why
    assert alone(got), (got.run_state, got.error_message, got.runtime_info)
