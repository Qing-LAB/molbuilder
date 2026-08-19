"""SIESTA warm-retry (``continue_retries``) — wrapper contract + ground truth.

The retry feature re-execs the wrapper with ``--continue`` when a SIESTA run
failed in a RETRIABLE way.  Two detection points, each grounded in the
project's frozen real-output fixtures (``tests/watch/fixtures/siesta_frozen/``):

  * SCF abort — with ``SCF.MustConverge`` (SIESTA's default, and molbuilder
    emits no override) an unconverged SCF stops the run NON-zero with
    ``SCF_NOT_CONV:`` in the ``.out`` (then ABNORMAL_TERMINATION + MPI
    abort).  So the SCF retry lives in the wrapper's non-zero-exit branch,
    and warm ``--continue`` resumes from the banked ``.DM`` with a fresh
    iteration budget.
  * Geometry cap — a relaxation that exhausts its MD step budget unconverged
    exits 0 and prints ``outcoor: Final (unrelaxed) atomic coordinates``
    (a converged relax prints ``Relaxed atomic coordinates``).  So the
    geometry retry lives in the zero-exit path.

2026-07-29: replaces the first-cut implementation whose zero-exit-only check
could never fire (SCF aborts exit non-zero; its geometry marker string does
not occur in SIESTA output), and whose ``exec "$0"`` PATH-searched a bare
relative name — exit 127 under the canonical ``bash <base>.run.sh``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.jobset.model import Resources
from molbuilder.runwrap import render_run_wrapper

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "tests" / "watch" / "fixtures" / "siesta_frozen"

# The exact idioms the wrapper renders (tested both as source text and,
# below, behaviourally).  If these change in runwrap.py, change them here
# in the same commit.
SELF_LINE = '_mb_self="$(readlink -f -- "$0" 2>/dev/null || echo "$0")"'
EXEC_LINE = 'exec bash "$_mb_self" --continue'
SCF_MARKER = "SCF_NOT_CONV"
GEOM_MARKER = "outcoor: Final (unrelaxed) atomic coordinates"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Minimal config so the generator will emit (mirrors test_runwrap_v2)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preamble": "module load mamba",
                              "activation": "source activate"},
    }))
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary="/usr/bin/conda"))
    yield tmp_path
    set_capabilities(None)


def _render(**kw) -> str:
    """The wrapper's text for a four-rank SIESTA job.

    ``kw`` are allocation fields: the door takes the object whole (A8), so
    they are set on it rather than passed beside it."""
    return render_run_wrapper(Path("/x/JOB.fdf"),
                              resources=Resources(mpi_np=4, **kw))


# --------------------------------------------------------------------- #
#  Rendered-wrapper contract                                            #
# --------------------------------------------------------------------- #


class TestRenderedContract:

    def test_no_retry_machinery_without_continue_retries(self, sandbox):
        text = _render()
        assert "_mb_warm_retry" not in text
        assert SCF_MARKER not in text
        assert GEOM_MARKER not in text

    def test_scf_retry_lives_in_the_nonzero_exit_branch(self, sandbox):
        """The retriable SCF failure exits non-zero (SCF.MustConverge
        default) — the check must run BEFORE ``exit "$_siesta_exit"``."""
        text = _render(continue_retries=2)
        assert "_mb_warm_retry() {" in text
        i_scf = text.index(f'grep -aq "{SCF_MARKER}"')
        i_exit = text.index('exit "$_siesta_exit"')
        assert i_scf < i_exit, (
            "SCF_NOT_CONV retry must precede the non-zero-exit exit — "
            "SIESTA aborts non-zero on SCF non-convergence")

    def test_geometry_retry_uses_the_real_siesta_marker(self, sandbox):
        """The zero-exit retry greps the marker SIESTA actually prints
        (fixture-verified), not an invented phrase."""
        text = _render(continue_retries=2)
        assert GEOM_MARKER in text
        assert "Geometry step did NOT converge" not in text
        # ... and it sits AFTER the non-zero-exit branch closes.
        assert text.index('exit "$_siesta_exit"') < text.index(GEOM_MARKER)

    def test_reexec_is_bash_on_an_absolute_self_path(self, sandbox):
        """``exec "$0"`` PATH-searches a bare name under ``bash x.run.sh``
        (exit 127); the wrapper must exec via bash + captured abs path."""
        text = _render(continue_retries=2)
        assert SELF_LINE in text
        assert EXEC_LINE in text
        # No bare self-exec anywhere outside the explanatory comment.
        for line in text.splitlines():
            if 'exec "$0"' in line:
                assert line.lstrip().startswith("#")

    def test_retry_preserves_original_args_minus_continuation_flags(
            self, sandbox):
        text = _render(continue_retries=2)
        assert '_mb_orig_args=(${@:+"$@"})' in text
        # --force / --cold are filtered: they would reset the run-index /
        # move aside the warm-start files the retry depends on.
        assert "--continue|-c|--force|-f|--cold|--from-scratch) ;;" in text

    def test_monitor_is_killed_before_reexec(self, sandbox):
        """exec skips the EXIT trap; without the kill each retry would
        stack another mb_monitor.py appending to the same util CSV."""
        text = _render(continue_retries=2)
        fn = text[text.index("_mb_warm_retry() {"):text.index(EXEC_LINE)]
        assert 'kill "$_monitor_pid"' in fn


# --------------------------------------------------------------------- #
#  Marker ground truth — frozen real SIESTA output                       #
# --------------------------------------------------------------------- #


class TestMarkerGroundTruth:
    """If SIESTA's wording ever drifts, these fail first — and they pin
    that the wrapper's grep strings match REAL output, not lore."""

    def test_scf_abort_fixtures_carry_the_scf_marker(self):
        for name in ("hemeC-stage1-scf_not_conv-5fr.out",
                     "hemeC-stage3-scf_not_conv-1fr.out"):
            out = (FROZEN / name).read_text(errors="replace")
            assert SCF_MARKER in out, name
            # The abort cascade the parser documents (siesta.py): the
            # non-zero exit is what routes this to the wrapper's
            # non-zero branch.
            assert "ABNORMAL_TERMINATION" in out, name

    def test_geometry_cap_fixture_carries_the_geom_marker(self):
        out = (FROZEN / "hemeC-stage2-run3-finished-42fr.out").read_text(
            errors="replace")
        assert GEOM_MARKER in out
        # ...and is NOT an SCF abort (exit 0 path).
        assert SCF_MARKER not in out

    def test_scf_abort_fixtures_do_not_false_trigger_geometry(self):
        for name in ("hemeC-stage1-scf_not_conv-5fr.out",
                     "hemeC-stage3-scf_not_conv-1fr.out"):
            out = (FROZEN / name).read_text(errors="replace")
            assert GEOM_MARKER not in out, name


# --------------------------------------------------------------------- #
#  Behavioural regression — the exit-127 re-exec bug                     #
# --------------------------------------------------------------------- #


class TestReexecMechanics:

    def test_self_reexec_survives_bare_bash_invocation(self, tmp_path):
        """Run the EXACT self-path + exec idiom the wrapper renders, as a
        standalone script invoked the canonical way (``bash x.run.sh``
        with a bare relative name, from the script's dir AND from a
        parent dir).  The first-cut ``exec "$0"`` died here with 127."""
        sub = tmp_path / "sub"
        sub.mkdir()
        scr = sub / "t.run.sh"
        scr.write_text(
            "#!/bin/bash\nset -u\n"
            + SELF_LINE + "\n"
            + 'if [ "${MB_RETRY_N:-0}" -ge 1 ]; then\n'
            + '    echo "SECOND_RUN_OK argv=$*"\n    exit 0\nfi\n'
            + "export MB_RETRY_N=1\n"
            + EXEC_LINE + "\n")
        # From the script's own directory (bare name — the 127 trap):
        r1 = subprocess.run(["bash", "t.run.sh"], cwd=sub,
                            capture_output=True, text=True, timeout=30)
        assert r1.returncode == 0, r1.stderr
        assert "SECOND_RUN_OK" in r1.stdout
        assert "--continue" in r1.stdout      # the retry flag arrived
        # From a parent dir via a relative path (the sbatch shape):
        r2 = subprocess.run(["bash", "sub/t.run.sh"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
        assert r2.returncode == 0, r2.stderr
        assert "SECOND_RUN_OK" in r2.stdout


# --------------------------------------------------------------------- #
#  /api/run/install-wrapper — continue_retries validation                #
# --------------------------------------------------------------------- #


class TestInstallWrapperEndpointValidation:

    @pytest.fixture
    def web(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        monkeypatch.chdir(tmp_path)
        script = tmp_path / "projects" / "P" / "opt" / "S"
        script.mkdir(parents=True)
        (script / "JOB.fdf").write_text("SystemLabel JOB\n")
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        from molbuilder.web.app import create_app
        client = create_app(config={}).test_client()
        yield client, str(script / "JOB.fdf")
        set_capabilities(None)

    def test_out_of_range_continue_retries_is_a_400(self, web):
        client, script_path = web
        r = client.post("/api/run/install-wrapper", json={
            "script_path": script_path, "continue_retries": 1000000})
        assert r.status_code == 400
        assert "between 1 and 5" in r.get_json()["error"]

    def test_non_numeric_continue_retries_is_a_400_not_500(self, web):
        client, script_path = web
        r = client.post("/api/run/install-wrapper", json={
            "script_path": script_path, "continue_retries": "lots"})
        assert r.status_code == 400
        assert r.get_json()["ok"] is False

    def test_unconfigured_activation_is_a_400_with_the_fix_text(self, web):
        """No script_generation.activation configured -> the generator
        refuses.  That is an OPERATOR-config error whose message IS the
        fix; it must come back as a 400 carrying it, not a generic 500
        'wrapper write failed'.  (The qlabsrv missing-.run.sh regression:
        a pre-v2 molbuilder.json without the section made every wrapper
        install fail this way.)"""
        client, script_path = web       # sandbox has no molbuilder.json
        r = client.post("/api/run/install-wrapper",
                        json={"script_path": script_path})
        assert r.status_code == 400
        err = r.get_json()["error"]
        assert "script_generation" in err and "activation" in err
