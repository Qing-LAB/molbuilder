"""What a person's script reads of a finished PySCF relaxation, and what `jobset
export` saves of it (`engines/optimization-postprocessing.md`; `process/
testing.md` tier 3): the steps the run's own writer logged, on a real run of
water with its oxygen held (`support.real_runs.held_water_pyscf`).
"""
from __future__ import annotations

import numpy as np
import pytest

from _road import conda_hook, env_available
from support.road import molbuilder

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
    the same whichever of the two files a person opens.  The atom index door
    answers the log by the map its header states (`model/overview.md`
    § 2.2b), not by its run's deck.

    THE SILENT FAILURE: the two files of one run telling a person two
    things -- the held oxygen moving in one, the run still going in one, a
    threshold line missing from one; the log's map passed over for its
    run's deck."""
    from molbuilder.atom_index import atom_index_map
    from molbuilder.parse import parse
    from molbuilder.runfiles import ROLE_GEOM_TRAJ, ROLE_MOLWATCH
    from molbuilder.runs import run_of
    run = run_of(real_held_water_pyscf.attempt("coarse"))
    traj, log = run.file(ROLE_GEOM_TRAJ), run.file(ROLE_MOLWATCH)
    of_traj, of_log = parse(traj), parse(log)
    assert of_traj.source_format == "pyscf"
    assert of_log.runtime_info["frozen_atoms"] == [0]
    assert of_log.runtime_info["convergence_targets"]
    assert of_log.run_state == "ended"
    for fact in ("frozen_atoms", "convergence_targets", "scf_criteria",
                 "scf_residuals"):
        assert of_traj.runtime_info.get(fact) == of_log.runtime_info[fact], (
            f"{traj.name} and {log.name} state {fact} two ways")
    assert ((of_traj.run_state, of_traj.error_message)
            == (of_log.run_state, of_log.error_message))
    assert atom_index_map(log).stated_by == str(log)
