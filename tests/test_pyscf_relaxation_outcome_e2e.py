"""A PySCF relaxation's outcome is geomeTRIC's own, read — never assumed
(`engines/pyscf.md` § 3; the K6 review) -- and a failed run's ending in PySCF's
own words (`execution/running-a-job.md` § 3.5, § 4.2).

geomeTRIC raises at its step cap with its criteria unmet, PySCF's driver
catches it, and only ``geometric_solver.kernel`` returns the flag.  Both decks
called ``optimize``, which drops it, and wired the policy to
``assert_convergence`` — the guard on one step's SCF — so a rung that ran out of
steps was recorded converged and handed on under every policy (the M11
review).  Both now relax through the one spliced ``relax_policy.relax``.

Driven through ``molbuilder describe`` -> ``prep task`` -> ``launch task --mode
direct`` on H2 started at 0.95 Å, far from its minimum, with the rung's own
values in ``task.json``'s stage overrides, as the Task setup table saves them:
one or two geometry steps cannot relax it.
"""
from __future__ import annotations

import numpy as np
import pytest

from _road import conda_hook, env_available
from support.road import group_words, jobset

CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-pySCF env + a detectable conda hook"),
]


#: H2 at 0.95 Å, far from its minimum -- the relaxation's subject.
_H2 = (["H", "H"], [[5.0, 5.0, 5.0], [5.0, 5.0, 5.95]])


def _run(tmp_path_factory, name: str, *, calculation, overrides,
         template=None, molecule=_H2, execution=None):
    """``molecule``, described on the road in a tree of its own, ``name``,
    with the real PySCF (`support.real_runs.on_the_road`, asked from a
    module's fixture: a test's own PATH holds the suite's stand-in conda,
    where no engine is found; `support.road.described`), its template
    ``template``'s values set as a person sets them
    (`support.road.set_by_the_person`), since the electronic state binds
    every rung and is the template's, and its one stage carrying
    ``overrides`` and its own run card ``execution`` (`job-request.md` § 2:
    a run setting lives in the job request alone), saved through Task
    setup's Save -- prepared and launched directly: the stage's run
    (`runs.run_of`), the stage's state as the ladder reads it, and what the
    run itself printed."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.runs import run_of
    from molbuilder.structure import Structure
    from molbuilder.task import of as task_of
    from support.real_runs import on_the_road
    from support.road import (described, description_with_overrides,
                              read_request, save_both, set_by_the_person,
                              structure_saved)
    struct = Structure(elements=list(molecule[0]),
                       positions=np.array(molecule[1], dtype=float))
    with on_the_road(tmp_path_factory, name, engine="pyscf") as tree:
        bundle = described(tree, structure=structure_saved(tree, struct),
                           name="H2", calculation=calculation, engine="pyscf")
        if template:
            set_by_the_person(bundle, template)
        task = task_of(bundle)
        stage = task.stages[0].name
        # A CASE'S OWN RUN CARD, its stage's, laid over the calculation's
        # field by field (`job-request.md` § 3.2) -- the road's, one thread,
        # which `described` wrote (`support.road.RUN_CARD`).
        request = read_request(bundle)
        if execution:
            request["stages"][stage]["execution"] = dict(execution)
        saved = save_both(bundle, description_with_overrides(
            bundle, {stage: overrides}), request)
        assert saved.status_code == 200, saved.get_json()
        r = jobset("prep", "task", *group_words(bundle, stage),
                   "--bundle", bundle,
                   "--target", "this")
        assert r.exit_code == 0, r.output
        jobset("launch", "task", "--stage", stage, "--bundle", bundle,
               "--mode", "direct", "--foreground", "--yes")
        row = next(iter(jobset_status(open_calculation(bundle)).stages))
        ran_in = latest_attempt(stage_home(bundle, task, stage).dir)
    assert ran_in is not None, sorted(p.name for p in bundle.rglob("*"))
    run = run_of(ran_in)
    log = run.file(".pyscf.log")
    assert log is not None, sorted(p.name for p in ran_in.iterdir())
    return run, row.state, log.read_text(errors="replace")


def _the_engines_words_kept(run, words: str) -> str:
    """A failed PySCF run's ending keeps PySCF's own words -- its exception
    line -- on every surface that says how it ended (2.65.1; 2.64.0 left it
    *untested before F9*), and the run script said it after the failure:

    * the status door -- the ladder row `jobset status` and the Results tab
      show, and the folder answer's ``status`` -- `running-a-job.md` § 4.2:
      *failed -- "stopped before its end: <the line that stopped it>"*, *"The
      detail quotes the line"*; for PySCF that line is *"the exception
      line"* (`model/parse.md` § 2b, ``error_message``);
    * the run's record, whose ``verdict.detail`` is *"`run_status`'s words
      for the directory"* (`model/parse.md` § 5d.1a);
    * the session log: *"After a failure the run script says ... How it
      ended, asked of the framework's reader, never grepped, whatever the
      engine"* (`running-a-job.md` § 3.5, 1; *a PySCF run's now too*,
      2.63.0) -- the engine's own output goes to its ``.pyscf.log``, so the
      words reach the session log only by that line.

    THE SILENT FAILURE: a failed PySCF run that says only *concluded (rc=1
    ...)* -- the person told it failed and not why -- or a run script that
    asks a SIESTA run how it ended and never a PySCF one.  Returns the
    session log's text."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.runs import folder_answer
    row = next(iter(jobset_status(open_calculation(run.place.root)).stages))
    answer = folder_answer(run.folder)
    for surface, detail in (
            ("jobset status", row.detail),
            ("the folder answer", answer["status"]["detail"]),
            ("the record", answer["record"]["verdict"]["detail"])):
        assert (detail or "").startswith("stopped before its end: ") \
            and words in detail, (surface, detail)
    assert run.session_log is not None, sorted(
        p.name for p in run.folder.iterdir())
    session = run.session_log.read_text(errors="replace")
    assert words in session, session[-3000:]
    return session


@pytest.fixture(scope="module")
def halted(tmp_path_factory):
    """H2 from 0.95 Å, one step allowed under ``halt`` (:func:`_run`)."""
    return _run(tmp_path_factory, "halted", calculation="optimization",
                overrides={"geom_max_steps": 1, "on_nonconvergence": "halt"})


def test_halt_stops_the_rung_and_keeps_the_geometry_it_reached(halted):
    """One step cannot relax H2 from 0.95 Å: ``halt`` stops the job, an
    error -- so no later rung builds on it -- saying the budget and the
    policy, and the geometry it reached is kept in ``_optimized.xyz``, as
    SIESTA keeps its ``.XV`` (user, 2026-10-06), so the rung launched again
    continues from it -- which the message says.

    And the run's live log says it stopped, not that it ended: the product's
    own reader of that log reads ``stopped``.

    MUTATION THIS MUST FAIL AGAINST: ``relax`` taking the geometry without
    asking geomeTRIC (``optimize``, or ``return mol, True`` after the first
    call) — the decks before 2026-09-29, which exited 0 and handed the
    unconverged geometry to the next rung; the geometry reached not kept;
    and a stop raised as a
    ``SystemExit``, which no ``excepthook`` sees, so the log closed as a
    clean end (the K6 review, R1).
    """
    from molbuilder.parse.registry import parse

    run, state, said = halted
    assert state == "failed", said[-3000:]
    assert ("did not meet geomeTRIC's criteria in 1 steps "
            "(on_nonconvergence = halt)") in said, said[-3000:]
    assert "launch this stage again to continue from it" in said, (
        said[-3000:])
    assert run.on_label("_optimized.xyz") is not None, (
        "a halted rung kept no geometry to continue from")
    log = parse(run.file(".molwatch.log"))
    assert log.run_state == "stopped", log.run_state
    # WHAT A PAGE SAYS OF THE GEOMETRY IT REACHED (`engines/vibration.md`
    # § 2.2): the target it had, and that it did not reach it -- "relaxed on
    # PySCF against a target of …, not reaching it", where a reached one
    # reads "relaxed on PySCF to …".  THE SILENT FAILURE: *relaxed to* of a
    # run that ran out of steps.
    from molbuilder.parse.contract import relaxation_of_output
    from molbuilder.validation.sidecar import check_relaxation_record
    rec = relaxation_of_output(run.file(".molwatch.log"), log, engine="pyscf")
    assert rec["converged"] is False, rec
    reached = [fr for fr in log.frames if fr.structure is not None][-1].structure
    reached.apply_info_dict({"relaxation": rec})
    words = " ".join(i.message for i in check_relaxation_record(
        reached, calculation="vibration", engine="pyscf",
        already_relaxed=False, force_tolerance_ev_ang=None))
    assert "against a target of" in words and "not reaching it" in words, (
        words)
    # ITS OWN WORDS on every surface (:func:`_the_engines_words_kept`) --
    # and what a launch again will do, said by the run script: this stage
    # continues (`running-a-job.md` § 3.5, 3: *"a stage that continues |
    # launch the stage again to continue warm from the files this run
    # saved"*).  THE SILENT FAILURE: a halted rung told to start over.
    session = _the_engines_words_kept(
        run, "did not meet geomeTRIC's criteria in 1 steps "
             "(on_nonconvergence = halt)")
    assert ("launch the stage again to continue warm from the files this "
            "run saved") in session, session[-3000:]


@pytest.fixture(scope="module")
def continued(tmp_path_factory):
    """H2 from 1.6 Å, one step a batch and two batches more under
    ``continue`` (:func:`_run`)."""
    return _run(tmp_path_factory, "continued", calculation="optimization",
                molecule=(["H", "H"], [[5.0, 5.0, 5.0], [5.0, 5.0, 6.6]]),
                overrides={"geom_max_steps": 1,
                           "on_nonconvergence": "continue",
                           "geom_continue_retries": 2})


def test_continue_reenters_from_the_geometry_reached(continued):
    """One step a batch and two batches more, from 1.6 Å: CERTAIN not to relax
    H2, by geomeTRIC's own rule -- its trust radius starts at 0.1 Å
    (``params.py``) and every re-entry starts it afresh, so no step moves the
    bond more than 0.2 Å, and three steps cannot cover the 0.86 Å to its
    minimum.  So the run says each re-entry, each batch starts where the last
    one stopped and never at the input geometry, and at the end of the
    budget it stops as ``halt`` does, naming the whole budget and keeping the
    geometry it reached.

    MUTATION THIS MUST FAIL AGAINST: a re-entry from the input geometry (no
    ``reset`` to the geometry reached) -- the live log returns to 1.6 Å; a
    ``continue`` that never re-enters (the old loop retried only a failed
    SCF); and a rung that takes the geometry without asking.
    """
    from molbuilder.parse.registry import parse

    run, state, said = continued
    assert state == "failed", said[-3000:]
    assert said.count("continuing from the geometry it reached") == 2, (
        said[-3000:])
    assert ("did not meet geomeTRIC's criteria in 3 steps "
            "(on_nonconvergence = continue)") in said, said[-3000:]
    # The whole budget's stop, in PySCF's words on every surface
    # (:func:`_the_engines_words_kept`).
    _the_engines_words_kept(run, "did not meet geomeTRIC's criteria in 3 "
                                 "steps (on_nonconvergence = continue)")
    kept = run.on_label("_optimized.xyz").read_text().split("\n")
    z = [float(ln.split()[3]) for ln in kept[2:4]]
    assert round(abs(z[1] - z[0]), 4) != 1.6, (
        f"the geometry kept is the input's, not the one reached: {kept}")
    # The live log keeps every step of every batch.  A re-entry evaluates
    # the geometry it starts from, so it shows as a frame repeating the one
    # before it; after the first move nothing is back at the input's bond.
    frames = [fr for fr in parse(run.file(".molwatch.log"))
              .frames if fr.energy is not None]
    bonds = [round(float(np.linalg.norm(
        np.asarray(fr.structure.positions)[0]
        - np.asarray(fr.structure.positions)[1])), 4) for fr in frames]
    assert bonds[0] == 1.6, bonds
    reentries = [i for i in range(1, len(bonds)) if bonds[i] == bonds[i - 1]]
    assert len(reentries) == 2, bonds
    assert 1.6 not in bonds[1:], (
        f"a batch started again at the input geometry: {bonds}")


@pytest.fixture(scope="module")
def proceeded(tmp_path_factory):
    """H2's vibration from 0.95 Å, its relaxation one step allowed under
    ``proceed`` (:func:`_run`)."""
    return _run(tmp_path_factory, "proceeded", calculation="vibration",
                overrides={"geom_max_steps": 1,
                           "on_nonconvergence": "proceed"})


def test_proceed_keeps_the_geometry_and_the_result_says_so(proceeded):
    """A vibration whose relaxation runs out of steps under ``proceed`` goes
    on to the Hessian at the geometry it reached, and its result says so: the
    warning names what happened, and ``relaxation.converged`` is the judged
    force against ``geom_gmax`` — false here, one step from 0.95 Å.

    MUTATION THIS MUST FAIL AGAINST: ``proceed`` recording the relaxation as
    converged, or recording no verdict (``converged: null``, the deck before
    2026-09-29), or taking the geometry without the warning.
    """
    from molbuilder.parse.registry import parse

    run, state, said = proceeded
    assert state == "finished", said[-3000:]
    rx = parse(run.on_label(".spectra.json")).payload["relaxation"]
    assert "did not meet geomeTRIC's criteria in 1 steps" in (
        rx.get("warning") or ""), rx
    assert rx["converged"] is False, rx
    assert rx["max_force_eh_bohr"] is not None, rx
    # A FINISHED RUN'S FILE IS WHOLE (`running-a-job.md` § 3.6): its engine
    # ended on its own, so one dated at or after the monitor closed is not named
    # as a file a kill may have caught.  THE SILENT FAILURE: every warm start
    # from a finished PySCF run warned of a `.chk` that is whole.  ITS WRITE
    # TIME IS THE ONE THING SET BY HAND, to now, after the monitor closed: a
    # run's last write falls in its monitor's closing second only as the
    # clock happens to fall, which no run reaches on demand -- the file and
    # its run are the road's own.
    import os
    from molbuilder.jobset.materialize import written_as_ended
    chk = run.on_label(".chk")
    os.utime(chk)
    assert written_as_ended(run.folder, run.label, [chk.name]) == {}


@pytest.fixture(scope="module")
def open_shell_on_the_gpu(tmp_path_factory):
    """Triplet O2, UHF/STO-3G, on the GPU (:func:`_run`) -- where an NVIDIA
    GPU is visible."""
    from _road import gpu_visible
    if not gpu_visible():
        pytest.skip("no NVIDIA GPU visible here")
    return _run(
        tmp_path_factory, "open-shell-gpu", calculation="optimization",
        molecule=(["O", "O"], [[5.0, 5.0, 5.0], [5.0, 5.0, 6.21]]),
        template={"method": "HF", "basis": "sto-3g",
                  "spin_treatment": "unrestricted", "unpaired_electrons": 2},
        overrides={"geom_max_steps": 1, "on_nonconvergence": "proceed"},
        # A GPU run states how many (`execution/gpu.md` G5: no default).
        execution={"use_gpu": True, "gpu_count": 1})


def test_an_open_shell_run_on_the_gpu_says_its_stability_was_not_checked(
        open_shell_on_the_gpu):
    """gpu4pyscf's GPU classes DECLARE no stability analysis (``stability =
    NotImplemented``), and the open-shell check runs after the GPU promotion:
    the deck asks the mean field, says NOT CHECKED and why, and the run goes
    on (`engines/pyscf.md` § 7.3).  Triplet O2, UHF/STO-3G, on the GPU.

    MUTATION THIS MUST FAIL AGAINST: calling ``mf.stability()`` without
    asking -- the deck before 2026-09-29, where calling ``NotImplemented``
    raised a TypeError and every open-shell GPU run died before its first
    step.
    """
    run, state, said = open_shell_on_the_gpu
    assert "GPU acceleration ON" in said, said[-3000:]
    assert ("stability: NOT CHECKED -- this mean field declares no stability "
            "analysis") in said, said[-3000:]
    assert state == "finished", said[-3000:]


@pytest.fixture(scope="module")
def stated_relaxed(tmp_path_factory):
    """H2's vibration at 0.95 Å, its structure stated relaxed
    (:func:`_run`)."""
    return _run(tmp_path_factory, "stated-relaxed", calculation="vibration",
                template={"already_relaxed": True}, overrides={})


def test_a_structure_stated_relaxed_is_measured_not_relaxed(stated_relaxed):
    """A PySCF vibration whose structure is stated relaxed runs no relaxation
    -- the phase is `not requested` from the first write, on both engines
    (`engines/vibration.md` § 4.9), no step is taken -- and the gradient
    check measures the statement: H2 at 0.95 Å is not stationary, so the
    result's verdict is false and its warning carries the one remedy for a
    structure stated relaxed on this engine (`engines/vibration.md` § 4.3,
    § 5.5).

    MUTATION THIS MUST FAIL AGAINST: the relaxation escaping its
    ``if not ALREADY_RELAXED`` guard, a gradient check that records no verdict (before 2026-09-29), and a
    remedy worded apart from the one text.
    """
    from molbuilder.parse.registry import parse
    from molbuilder.spectra.results import PHASE_NOT_REQUESTED
    from molbuilder.spectra.vibrational_analysis import nonstationary_remedy

    run, state, said = stated_relaxed
    assert state == "finished", said[-3000:]
    doc = parse(run.on_label(".spectra.json")).payload
    rx = doc["relaxation"]
    assert (rx["enabled"], rx["n_steps"]) == (False, 0), rx
    assert doc["phase_relaxation"] == PHASE_NOT_REQUESTED, \
        doc["phase_relaxation"]
    assert rx["converged"] is False, rx
    assert nonstationary_remedy(None) in (rx.get("warning") or ""), rx


@pytest.fixture(scope="module")
def relaxation_past_its_wall(tmp_path_factory):
    """H2 from 0.95 Å under ``proceed``, its run card's wall 1 s
    (:func:`_run`)."""
    return _run(tmp_path_factory, "relaxation-walled",
                calculation="optimization",
                overrides={"geom_max_steps": 50,
                           "on_nonconvergence": "proceed"},
                execution={"time": "00:00:01"})


def test_the_wall_stops_the_relaxation_and_keeps_the_geometry(
        relaxation_past_its_wall):
    """`engines/pyscf.md` § 3.9, `running-a-job.md` § 3.6: the deck keeps the
    deadline its environment states -- the job's wall, its start and the
    slack, as the run script exports them -- and the first step past it
    stops geomeTRIC: the geometry reached stays in ``_optimized.xyz``, the
    deck says why, the log's footer says ``relaxed: no`` and the run ends
    normally, so the run door reads `finished`, not converged.

    The wall is the run card's (`running-a-job.md` § 3.6, *one run, here*),
    1 s, less than its slack: the deadline is past when the deck starts, so
    the first step is the last.

    MUTATION THIS MUST FAIL AGAINST: the card's wall not told to a run here
    (`submit._go`), the deadline not read from the environment, or the
    relaxation running to its step budget (H2 from 0.95 Å converges in a few
    steps, and the run would read converged)."""
    from molbuilder.parse.engines._run_ending import ending_of
    run, state, said = relaxation_past_its_wall
    assert state == "finished", said[-3000:]
    assert "stopped before the time limit" in said, said[-3000:]
    assert run.on_label("_optimized.xyz") is not None, sorted(
        p.name for p in run.folder.iterdir())
    # THE LOG'S FOOTER, read by the ending reader: the geometry verdict is
    # the deck's own word (`# relaxed: no`) -- and the run's, through the
    # status door every surface asks (`model/parse.md` § 2b).
    log = run.file(".molwatch.log")
    assert ending_of(log).relaxed is False, log.read_text()[-600:]
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.runstatus import jobset_status
    row = next(iter(jobset_status(open_calculation(run.place.root)).stages))
    assert row.converged == "geometry NO", row
    # ...AND NO SCF PAST THE WALL: the run ends with the geometry reached.
    assert "Final energy" not in said, said[-3000:]


@pytest.fixture(scope="module")
def vibration_past_its_wall(tmp_path_factory):
    """H2's vibration, frequencies alone, under ``halt``, its run card's
    wall 1 s (:func:`_run`)."""
    return _run(tmp_path_factory, "vibration-walled",
                calculation="vibration",
                overrides={"geom_max_steps": 50, "on_nonconvergence": "halt"},
                template={"compute_raman": False, "compute_ir": False},
                execution={"time": "00:00:01"})


def test_the_wall_stops_a_vibration_before_its_spectrum(
        vibration_past_its_wall):
    """`engines/pyscf.md` § 3.9: a vibration past its wall computes no
    spectrum -- the run ends failed, saying it stopped before the wall
    before the spectrum was written, and a launch runs it again.  The
    deadline here is past when the deck starts, frequencies alone (an
    analytic Hessian, no displaced SCF), so the deck's first SCF is where it
    stops; the check after its relaxation (the wall passing during it) is
    the same door, `relax_policy.stop_if_past_wall`.  The wall is the run
    card's, 1 s, less than its slack: past when the deck starts.

    THE SILENT FAILURE: a spectrum written at a geometry the wall stopped,
    the run reading finished (the 2026-10-23 review).

    MUTATION THIS MUST FAIL AGAINST: the vibration deck asking the wall
    nowhere (`stop_if_past_wall` answering nothing)."""
    run, state, said = vibration_past_its_wall
    assert state == "failed", said[-3000:]
    assert ("stopped before the time limit (the job's time limit less its slack) "
            "before the spectrum was written") in said, said[-3000:]
    # ITS OWN WORDS on every surface (:func:`_the_engines_words_kept`) --
    # and what a launch again will do: a PySCF vibration does not resume
    # (`running-a-job.md` § 3.5, 3: *"its kind does not resume (a PySCF
    # vibration) | a launch again starts this run over from its deck,
    # nothing of it copied in — this kind of run does not resume"*).  THE
    # SILENT FAILURE: a run told it continues warm from files no launch
    # copies into the next run.
    session = _the_engines_words_kept(
        run, "stopped before the time limit (the job's time limit less its slack) "
             "before the spectrum was written")
    for words in ("a launch again starts this run over from its deck, "
                  "nothing of it copied in", "this kind of run does not "
                                             "resume"):
        assert words in session, (words, session[-3000:])
