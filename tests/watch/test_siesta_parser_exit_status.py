"""SIESTA parser: detect non-convergence / error exit status.

Lands the 2026-05-29 directive: "the parser should be more
intelligent to detect the finishing/exit status as well".  Previous
behaviour was binary -- ">> End of run" -> finished, else ongoing,
no error path.  Now:

  * Fatal markers (``siesta: ERROR`` / ``propor: ERROR`` /
    ``Stopping Program from Node`` / ``siesta died``) flip run_state
    to "error" + capture the line as ``error_message``.
  * SCF block convergence is tracked.  If the run ends WITHOUT
    ``>> End of run`` AND the last SCF block did NOT converge, the
    parser sets run_state="stopped" with a synthetic message naming
    the failure mode.
  * ``>> End of run`` does NOT downgrade a prior "error" -- once
    error_message is set, it persists.
"""
from __future__ import annotations

import tempfile

import pytest

from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.parse.engines.siesta import SiestaParser


def _parse(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
        fh.write(text)
        path = fh.name
    return SiestaParser.parse(path)


# Minimal one-frame trajectory + one SCF block.  Each test mutates
# the END of the file to inject either a clean exit, a fatal marker,
# or a non-convergence path.
PROLOGUE = (
    "Welcome to SIESTA\n"
    "outcoor: Atomic coordinates (Ang):\n"
    "   1.00    2.00    3.00   1       1  C\n"
    "\n"
    "   iscf  Eharris(eV)  E_KS(eV)  FreeEng(eV)  dDmax  Ef(eV) dHmax(eV)\n"
    "   scf:    1   -100.0       -100.0       -100.0   0.001  -1.0   0.5\n"
    "siesta: E_KS(eV) =        -100.1234\n"
)


# ---- Clean exit baselines -------------------------------------------------


def test_clean_exit_with_end_of_run_marks_finished():
    """Baseline: ">> End of run" present + no error -> finished."""
    text = PROLOGUE + "SCF Convergence by DM+H criterion\n>> End of run: x\n"
    traj = _parse(text)
    assert traj.run_state == "ended"
    assert traj.error_message is None


def test_truncated_without_markers_stays_ongoing():
    """Baseline: no End-of-run, no error, no SCF non-convergence ->
    ongoing.  Honest default for "file is mid-stream / SIESTA still
    running"."""
    text = PROLOGUE + "SCF Convergence by DM+H criterion\n"
    traj = _parse(text)
    assert traj.run_state == "running"
    assert traj.error_message is None


# ---- Fatal markers --------------------------------------------------------


def test_siesta_error_marker_sets_error():
    text = PROLOGUE + "siesta: ERROR  bad input\n"
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert traj.error_message is not None
    assert "siesta: ERROR" in traj.error_message


def test_siesta_error_case_insensitive():
    """Uppercase variant still recognised."""
    text = PROLOGUE + "SIESTA: ERROR  bad input\n"
    traj = _parse(text)
    assert traj.run_state == "stopped"


def test_propor_error_marker_sets_error():
    """The propor: ERROR: IMAX = 0 failure the wrapper already
    grep-detects -- now the parser surfaces it too."""
    text = PROLOGUE + "propor: ERROR: IMAX = 0\n"
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert "propor: ERROR" in traj.error_message


def test_stopping_program_from_node_sets_error():
    text = PROLOGUE + " * Stopping Program from Node:    0\n"
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert "Stopping Program from Node" in traj.error_message


def test_siesta_died_sets_error():
    text = PROLOGUE + "siesta died: pseudopotential read failure\n"
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert "siesta died" in traj.error_message


def test_first_fatal_marker_wins():
    """If two fatal markers appear in sequence, the FIRST captures
    error_message -- subsequent crashes are usually cascade effects
    of the original cause."""
    text = (
        PROLOGUE
        + "propor: ERROR: IMAX = 0\n"     # first
        + "siesta: ERROR follow-up\n"     # second; should NOT overwrite
    )
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert "propor: ERROR" in traj.error_message
    assert "ERROR follow-up" not in traj.error_message


# ---- End-of-run does NOT paper over a prior error ------------------------


def test_end_of_run_does_not_clear_prior_error():
    """If a fatal marker fires AND then End-of-run also somehow
    appears, the badge stays Error.  Defensible default: a successful
    finish would not have emitted the fatal marker; if both appear
    the run hit a real fault."""
    text = (
        PROLOGUE
        + "propor: ERROR: IMAX = 0\n"
        + ">> End of run: x\n"
    )
    traj = _parse(text)
    assert traj.run_state == "stopped"
    assert "propor: ERROR" in traj.error_message


# ---- Strict SCF-not-converged detection ----------------------------------


def test_scf_did_not_converge_is_REPORTED_not_judged():
    """Rewritten 2026-08-25 for `model/parse.md` § 2b.

    This asserted the "strict policy" -- non-convergence without an
    End-of-run marker IS an error.  P-S2 retired that: the science does
    not decide how the process ended.  What the parser must still do is
    NOTICE, and say so as a fact.

    The ending stays ``running`` because nothing in the text can tell a
    truncated run from a slow one; ``parse/dirs/job.py`` settles that by
    file age (P-S1)."""
    text = PROLOGUE + "SCF did NOT converge after 200 iterations\n"
    traj = _parse(text)
    assert traj.scf_converged is False, "the non-convergence went unnoticed"
    assert traj.run_state == "running", (
        f"the parser guessed {traj.run_state!r} from content that cannot "
        f"support it")


def test_scf_not_conv_constant_form_also_detected():
    """SIESTA's internal SCF_NOT_CONV constant appears verbatim in
    some builds.  Detected as the same condition -- reported, not
    judged (§ 2b P-S2)."""
    text = PROLOGUE + "SCF_NOT_CONV\n"
    traj = _parse(text)
    assert traj.scf_converged is False


def test_scf_not_converged_followed_by_convergence_clears():
    """Mid-relax non-convergence followed by a later converged SCF
    block must NOT flip the run to error.  The 'last block' is what
    counts (strict policy)."""
    text = (
        "Welcome to SIESTA\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.00    2.00    3.00   1       1  C\n"
        "\n"
        # First SCF block: did NOT converge
        "   iscf  Eharris(eV)  E_KS(eV)  FreeEng(eV)  dDmax  Ef(eV) dHmax(eV)\n"
        "   scf:    1   -100.0       -100.0       -100.0   0.001  -1.0   0.5\n"
        "SCF did NOT converge after 200 iterations\n"
        # Second SCF block: converged
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.01    2.00    3.00   1       1  C\n"
        "\n"
        "   iscf  Eharris(eV)  E_KS(eV)  FreeEng(eV)  dDmax  Ef(eV) dHmax(eV)\n"
        "   scf:    1   -100.1       -100.1       -100.1   0.001  -1.0   0.3\n"
        "SCF Convergence by DM+H criterion\n"
        ">> End of run: x\n"
    )
    traj = _parse(text)
    # Even though the first SCF non-converged, the last one
    # converged + End-of-run fired -> finished.
    assert traj.run_state == "ended"
    assert traj.error_message is None


def test_scf_not_converged_with_end_of_run_stays_finished():
    """Edge case: SIESTA emits 'did NOT converge' but then
    eventually finishes (perhaps with SCF.MustConverge .false.).
    End-of-run wins -- the run completed."""
    text = PROLOGUE + (
        "SCF did NOT converge after 200 iterations\n"
        ">> End of run: x\n"
    )
    traj = _parse(text)
    assert traj.run_state == "ended"
    assert traj.error_message is None


def test_scf_converged_matcher_requires_by_keyword():
    """Defensive: the success matcher ``contains_ci("scf
    convergence by")`` would NOT match a hypothetical diagnostic
    line ``SCF Convergence check failed: SCF did NOT converge``.
    The "by" keyword is the canonical SIESTA success-phrase signal;
    requiring it prevents an accidental success-flag flip on a line
    that mentions both phrases.  Caught in the 2026-05-29 holistic
    review."""
    # The SCF-converged matcher does NOT fire on this line.
    text = PROLOGUE + (
        "SCF Convergence check failed -- SCF did NOT converge\n"
    )
    traj = _parse(text)
    # The "SCF did NOT converge" matcher SHOULD fire.  If the SUCCESS
    # matcher fired instead (the defect this guards), `scf_converged`
    # would read True and the reader would report a convergence that
    # never happened.
    assert traj.scf_converged is False


def test_json_emit_carries_run_state_and_error_message():
    """The Trajectory -> JSON shape (trajectory_to_legacy_dict) MUST
    carry both run_state AND error_message so the /results badge
    can render 'Error' with the cause line.  Caught here so a
    future refactor of the wire-shape can't silently drop the field."""
    text = PROLOGUE + "propor: ERROR: IMAX = 0\n"
    traj = SiestaParser.parse(_write_temp(text))
    payload = trajectory_to_legacy_dict(traj)
    assert payload["run_state"] == "stopped"
    assert payload["error_message"] is not None
    assert "propor: ERROR" in payload["error_message"]


def _write_temp(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
        fh.write(text)
        return fh.name


def test_real_world_hemec_stage3_pattern():
    """Pin the EXACT real-world wording observed 2026-05-30 on
    projects/hemeC-dithiol/optimization/gasrun1 stage 3:

      SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).
      Geom step, scf iteration, dDmax, dHmax: ...
      ABNORMAL_TERMINATION
      Stopping Program from Node: 0
      ABNORMAL_TERMINATION
      Stopping Program from Node: 6
      ...

    The badge must say Error, AND error_message must carry the
    informative ROOT CAUSE line (SCF_NOT_CONV: SCF did not
    converge ...) NOT the cascade ("Stopping Program from Node:
    0").  First-fatal-wins; the rule order is what guarantees this.
    """
    text = PROLOGUE + (
        "SCF_NOT_CONV: SCF did not converge in maximum number of "
        "steps (required).\n"
        "Geom step, scf iteration, dDmax, dHmax:    0   500      "
        "0.000008     0.000223\n"
        "ABNORMAL_TERMINATION\n"
        "Stopping Program from Node:    0\n"
        "ABNORMAL_TERMINATION\n"
        "Stopping Program from Node:    6\n"
        "ABNORMAL_TERMINATION\n"
        "Stopping Program from Node:    1\n"
    )
    traj = _parse(text)
    assert traj.run_state == "stopped"
    # Root cause, not cascade.
    assert traj.error_message is not None
    assert "SCF_NOT_CONV" in traj.error_message
    assert "did not converge" in traj.error_message
    assert "(required)" in traj.error_message
    # And explicitly NOT the cascade message.
    assert "Stopping Program" not in traj.error_message
    assert "ABNORMAL_TERMINATION" not in traj.error_message


def test_abnormal_termination_alone_is_fatal():
    """If a SIESTA build emits ABNORMAL_TERMINATION without a
    preceding SCF_NOT_CONV (some crash modes do this), it still
    marks the run as error."""
    text = PROLOGUE + (
        "ABNORMAL_TERMINATION\n"
        "Stopping Program from Node:    0\n"
    )
    traj = _parse(text)
    assert traj.run_state == "stopped"
    # First-fatal-wins: ABNORMAL_TERMINATION is the first error
    # marker so it captures error_message.
    assert "ABNORMAL_TERMINATION" in traj.error_message


def test_scf_not_conv_capture_overrides_softer_form():
    """Defensive: a line containing BOTH "SCF_NOT_CONV:" AND
    "SCF did not converge" -- which is exactly what real SIESTA
    emits -- must dispatch to the FATAL handler, not the soft
    flag.  Pin the rule-order guarantee."""
    text = PROLOGUE + (
        "SCF_NOT_CONV: SCF did not converge "
        "in maximum number of steps (required).\n"
    )
    # On its own the line proves nothing about the ending (§ 2b P-S2)...
    traj = _parse(text)
    assert traj.scf_converged is False
    assert traj.run_state == "running"
    # ...but when something DOES prove the run died, the held line is
    # what surfaces -- not the cascade marker, and not a synthetic
    # fallback.  That is the rule-order guarantee this test exists for.
    traj = _parse(text + "ABNORMAL_TERMINATION\n"
                         "Stopping Program from Node:    0\n")
    assert traj.run_state == "stopped"
    assert "SCF_NOT_CONV" in traj.error_message
    assert "run truncated" not in traj.error_message


def test_scf_did_not_converge_case_insensitive():
    """Capitalisation variants of the SIESTA phrase still detected."""
    for variant in (
        "SCF did NOT converge after 200 iterations",
        "scf did not converge after 200 iterations",
        "Scf Did Not Converge after 200 iterations",
    ):
        text = PROLOGUE + variant + "\n"
        traj = _parse(text)
        assert traj.scf_converged is False, f"variant={variant!r}"
