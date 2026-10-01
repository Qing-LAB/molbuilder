"""A PySCF relaxation's outcome is geomeTRIC's own, read — never assumed
(`engines/pyscf.md` § 3; plan § 5w K6).

geomeTRIC raises at its step cap with its criteria unmet, PySCF's driver
catches it, and only ``geometric_solver.kernel`` returns the flag.  Both decks
called ``optimize``, which drops it, and wired the policy to
``assert_convergence`` — the guard on one step's SCF — so a rung that ran out of
steps was recorded converged and handed on under every policy (the M11
review).  Both now relax through the one spliced ``relax_policy.relax``.

Driven through ``jobset init`` -> ``prep run`` -> ``launch run --mode
direct`` on H2 started at 0.95 Å, far from its minimum, with the rung's own
values in ``task.json``'s stage overrides, as the Task setup table saves them:
one or two geometry steps cannot relax it.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from _road import conda_hook, env_available

CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-pySCF env + a detectable conda hook"),
]


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


#: H2 at 0.95 Å, far from its minimum -- the relaxation's subject.
_H2 = (["H", "H"], [[5.0, 5.0, 5.0], [5.0, 5.0, 5.95]])


def _run(tmp_path, monkeypatch, *, calculation, overrides, template=None,
         molecule=_H2, info=None):
    """``molecule``, its one stage carrying ``overrides`` and its template
    ``template``'s values -- written by the hand-over's own writer
    (`template_with_values`), since the electronic state binds every rung
    and is the template's -- prepped and launched directly: the bundle, the
    stage's state as the ladder reads it, and what the run itself printed."""
    from conftest import write_machine_record
    from molbuilder.jobset.model import FILENAME, JobSet
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.task import read_task, write_task
    from molbuilder.workingcopy_structure import StructureCodec

    write_machine_record()
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    struct = Structure(elements=list(molecule[0]),
                       positions=np.array(molecule[1], dtype=float))
    for key, value in (info or {}).items():
        struct.set_info(key, value)
    StructureCodec().write(struct, tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", f"P/{calculation}/H2", "--engine", "pyscf",
                "--shape", "flat", "--name", "H2",
                "--calculation", calculation)
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / "H2"
    if template:
        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.template import (config_from_template,
                                         template_path, template_with_values)
        path = template_path(bundle, "H2")
        cfg = dataclasses.replace(
            config_from_template(path.read_text(), PySCFConfig), **template)
        path.write_text(template_with_values(cfg, engine="pyscf",
                                             calculation=calculation))
    task = read_task(bundle / "task.json")
    stage = dataclasses.replace(task.stages[0], overrides=dict(overrides))
    write_task(bundle / "task.json", dataclasses.replace(
        task, stages=(stage,),
        varies=tuple(dict.fromkeys((*task.varies, *overrides)))))
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": f"source {CONDA_SH}"}}))
    r = _jobset("prep", "run", stage.name, "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output
    _jobset("launch", "run", stage.name, "--bundle", bundle,
            "--mode", "direct", "--yes")
    row = next(iter(jobset_status(JobSet.load(bundle / FILENAME),
                                  bundle).stages))
    said = "\n".join(p.read_text(errors="replace")
                     for p in sorted(bundle.glob("*-run0.pyscf.log")))
    assert said, sorted(p.name for p in bundle.iterdir())
    return bundle, row.state, said


def test_halt_stops_the_rung_before_its_geometry_is_written(tmp_path,
                                                            monkeypatch):
    """One step cannot relax H2 from 0.95 Å: ``halt`` stops the job, saying
    the budget and the policy, and no ``_optimized.xyz`` is written — so no
    later rung can start from a geometry nobody accepted.

    And the run's live log says it stopped, not that it ended: the product's
    own reader of that log reads ``stopped``.

    MUTATION THIS MUST FAIL AGAINST: ``relax`` taking the geometry without
    asking geomeTRIC (``optimize``, or ``return mol, True`` after the first
    call) — the decks before 2026-09-29, which exited 0 and wrote the
    unconverged geometry for the next rung; and a stop raised as a
    ``SystemExit``, which no ``excepthook`` sees, so the log closed as a
    clean end (the K6 review, R1).
    """
    from pathlib import Path

    from molbuilder.parse.registry import parse

    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="optimization",
        overrides={"geom_max_steps": 1, "on_nonconvergence": "halt"})
    assert state == "failed", said[-3000:]
    assert ("did not meet geomeTRIC's criteria in 1 steps "
            "(on_nonconvergence = halt)") in said, said[-3000:]
    assert not list(bundle.glob("*_optimized.xyz")), (
        "a halted rung left a relaxed geometry for the next rung to read")
    log = parse(Path(next(bundle.glob("*.molwatch.log"))))
    assert log.run_state == "stopped", log.run_state


def test_continue_reenters_from_the_geometry_reached(tmp_path, monkeypatch):
    """One step a batch and two batches more, from 1.6 Å: CERTAIN not to relax
    H2, by geomeTRIC's own rule -- its trust radius starts at 0.1 Å
    (``params.py``) and every re-entry starts it afresh, so no step moves the
    bond more than 0.2 Å, and three steps cannot cover the 0.86 Å to its
    minimum.  So the run says each re-entry, each batch starts where the last
    one stopped and never at the input geometry, and at the end of the
    budget it stops as ``halt`` does, naming the whole budget and writing no
    relaxed geometry.

    MUTATION THIS MUST FAIL AGAINST: a re-entry from the input geometry (no
    ``reset`` to the geometry reached) -- the live log returns to 1.6 Å; a
    ``continue`` that never re-enters (the old loop retried only a failed
    SCF); and a rung that takes the geometry without asking.
    """
    from pathlib import Path

    from molbuilder.parse.registry import parse

    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="optimization",
        molecule=(["H", "H"], [[5.0, 5.0, 5.0], [5.0, 5.0, 6.6]]),
        overrides={"geom_max_steps": 1, "on_nonconvergence": "continue",
                   "geom_continue_retries": 2})
    assert state == "failed", said[-3000:]
    assert said.count("continuing from the geometry it reached") == 2, (
        said[-3000:])
    assert ("did not meet geomeTRIC's criteria in 3 steps "
            "(on_nonconvergence = continue)") in said, said[-3000:]
    assert not list(bundle.glob("*_optimized.xyz"))
    # The live log keeps every step of every batch.  A re-entry evaluates
    # the geometry it starts from, so it shows as a frame repeating the one
    # before it; after the first move nothing is back at the input's bond.
    frames = [fr for fr in parse(Path(next(bundle.glob("*.molwatch.log"))))
              .frames if fr.energy is not None]
    bonds = [round(float(np.linalg.norm(
        np.asarray(fr.structure.positions)[0]
        - np.asarray(fr.structure.positions)[1])), 4) for fr in frames]
    assert bonds[0] == 1.6, bonds
    reentries = [i for i in range(1, len(bonds)) if bonds[i] == bonds[i - 1]]
    assert len(reentries) == 2, bonds
    assert 1.6 not in bonds[1:], (
        f"a batch started again at the input geometry: {bonds}")


def test_proceed_keeps_the_geometry_and_the_result_says_so(tmp_path,
                                                          monkeypatch):
    """A vibration whose relaxation runs out of steps under ``proceed`` goes
    on to the Hessian at the geometry it reached, and its result says so: the
    warning names what happened, and ``relaxation.converged`` is the judged
    force against ``geom_gmax`` — false here, one step from 0.95 Å.

    MUTATION THIS MUST FAIL AGAINST: ``proceed`` recording the relaxation as
    converged, or recording no verdict (``converged: null``, the deck before
    2026-09-29), or taking the geometry without the warning.
    """
    from molbuilder.parse.registry import parse

    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="vibration",
        overrides={"geom_max_steps": 1, "on_nonconvergence": "proceed"})
    assert state == "finished", said[-3000:]
    rx = parse(next(bundle.glob("*.spectra.json"))).payload["relaxation"]
    assert "did not meet geomeTRIC's criteria in 1 steps" in (
        rx.get("warning") or ""), rx
    assert rx["converged"] is False, rx
    assert rx["max_force_eh_bohr"] is not None, rx


def _a_gpu_is_visible() -> bool:
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return False
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return r.returncode == 0 and "GPU" in r.stdout


def test_an_open_shell_run_on_the_gpu_says_its_stability_was_not_checked(
        tmp_path, monkeypatch):
    """gpu4pyscf's GPU classes DECLARE no stability analysis (``stability =
    NotImplemented``), and the open-shell check runs after the GPU promotion:
    the deck asks the mean field, says NOT CHECKED and why, and the run goes
    on (`engines/pyscf.md` § 7.3).  Triplet O2, UHF/STO-3G, on the GPU.

    MUTATION THIS MUST FAIL AGAINST: calling ``mf.stability()`` without
    asking -- the deck before 2026-09-29, where calling ``NotImplemented``
    raised a TypeError and every open-shell GPU run died before its first
    step.
    """
    if not _a_gpu_is_visible():
        pytest.skip("no NVIDIA GPU visible here")
    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="optimization",
        molecule=(["O", "O"], [[5.0, 5.0, 5.0], [5.0, 5.0, 6.21]]),
        template={"method": "HF", "basis": "sto-3g",
                  "spin_treatment": "unrestricted", "unpaired_electrons": 2},
        overrides={"use_gpu": True, "geom_max_steps": 1,
                   "on_nonconvergence": "proceed"})
    assert "GPU acceleration ON" in said, said[-3000:]
    assert ("stability: NOT CHECKED -- this mean field declares no stability "
            "analysis") in said, said[-3000:]
    assert state == "finished", said[-3000:]


def test_the_runs_own_pair_carries_no_record_of_the_inputs_run(tmp_path,
                                                              monkeypatch):
    """The input arrives as a SIESTA run's export -- its relaxation record
    and its level of theory in ``info`` -- and a PySCF relaxation writes its
    own ``_optimized.xyz`` pair: that pair is this run's geometry, so the
    input's run records stay behind (`engines/pyscf.md` § 2).

    MUTATION THIS MUST FAIL AGAINST: the input's ``info`` copied onto the
    run's pair -- the deck before 2026-09-29, which handed a PySCF-relaxed
    geometry a SIESTA run's tolerance, force and level of theory (PS-C10).
    """
    from molbuilder.workingcopy_structure import StructureCodec

    siesta_run = {
        "relaxation": {"engine": "siesta", "source": "elsewhere.out",
                       "n_steps": 7, "force_tolerance_ev_ang": 0.04,
                       "max_force_ev_ang": 0.03,
                       "max_force_free_ev_ang": 0.03, "held_atom_idxs": [],
                       "held_atom_keys": [], "converged": True,
                       "run_state": "finished",
                       "geometry_sha256": "sha256:" + "0" * 64},
        "calculation": {"engine": "siesta", "source": "elsewhere.fdf",
                        "contract": {"net_charge": 0,
                                     "spin_treatment": "restricted",
                                     "unpaired_electrons": 0}}}
    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="optimization",
        overrides={"geom_max_steps": 1, "on_nonconvergence": "proceed"},
        info=siesta_run)
    assert state == "finished", said[-3000:]
    written = StructureCodec().read(next(bundle.glob("*_optimized.xyz")))
    assert not {"relaxation", "calculation"} & set(written.info or {}), (
        written.info)


def test_a_structure_stated_relaxed_is_measured_not_relaxed(tmp_path,
                                                           monkeypatch):
    """A PySCF vibration whose structure is stated relaxed runs no relaxation
    -- the phase is complete by assertion, no step is taken -- and the gradient
    check measures the statement: H2 at 0.95 Å is not stationary, so the
    result's verdict is false and its warning carries the one remedy for a
    structure stated relaxed on this engine (`engines/vibration.md` § 4.3,
    § 5.5).

    MUTATION THIS MUST FAIL AGAINST: the relaxation escaping its
    ``if not ALREADY_RELAXED`` guard (the rule the retired text test pinned),
    a gradient check that records no verdict (before 2026-09-29), and a
    remedy worded apart from the one text.
    """
    from molbuilder.parse.registry import parse
    from molbuilder.spectra.vibrational_analysis import nonstationary_remedy

    bundle, state, said = _run(
        tmp_path, monkeypatch, calculation="vibration",
        template={"already_relaxed": True}, overrides={})
    assert state == "finished", said[-3000:]
    doc = parse(next(bundle.glob("*.spectra.json"))).payload
    rx = doc["relaxation"]
    assert (rx["enabled"], rx["n_steps"]) == (False, 0), rx
    assert doc["phase_relaxation"] == "complete", doc["phase_relaxation"]
    assert rx["converged"] is False, rx
    assert nonstationary_remedy(None, "pyscf") in (rx.get("warning") or ""), rx
