"""The launch-door gate + config provenance (user decisions, 2026-08-12).

Contract: `job-contracts.md` § 2.6 (the Launch-door gate row: one launch
door; `submit` sets ``MB_LAUNCHED_BY``, a bare call warns or refuses,
``manual`` is the logged override) · the provenance allowlist in
`runtime_config` (paths + execution-relevant values only — the machine file
also carries auth/tls, which must never reach a terminal or a shipped log).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from molbuilder.runtime_config import config_provenance, format_provenance
from molbuilder.runwrap import write_run_wrapper
from molbuilder.jobset.model import Resources


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Isolated cwd + HOME so nothing here reads the developer's config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()


def _wrapper(tmp_path):
    (tmp_path / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    (tmp_path / "JOB.fdf").write_text("SystemLabel JOB\n")
    # A PROBED MACHINE.  Since 2026-09-02 a rank count is read from a record
    # and nowhere else -- no probe of the running box, no fallback
    # (`running-a-job.md` § 3.1).  A wrapper cannot be rendered on an
    # unprobed machine, so a fixture that renders one probes first,
    # exactly as a person does:  molbuilder jobset probe --write
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (tmp_path / "environment.json").write_text(
        _Env(scheduler="slurm",
             topology=_Topo(sockets=2, cores_per_socket=32)).to_json()
        + "\n")
    return write_run_wrapper(tmp_path / "JOB.fdf", resources=Resources(), env="e")


# ---- the gate, in the emitted text and under execution ---------------- #

def test_the_wrapper_carries_the_gate_block(tmp_path):
    text = _wrapper(tmp_path).read_text()
    assert "# --- Launch-door gate" in text
    assert "MB_LAUNCHED_BY" in text


def test_a_bare_noninteractive_call_refuses_with_the_fix(tmp_path):
    """No claim + no terminal -> exit 2 before ANY work, naming both the
    right door and the deliberate override.  This is the `nohup`, cron and
    hand-`sbatch` case."""
    sh = _wrapper(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "MB_LAUNCHED_BY"}
    cp = subprocess.run(["bash", str(sh)], cwd=str(tmp_path),
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True, env=env)
    assert cp.returncode == 2
    assert "jobset launch" in cp.stderr
    assert "MB_LAUNCHED_BY=manual" in cp.stderr


def test_a_claimed_call_passes_the_gate(tmp_path):
    """With the claim set the gate says nothing -- the script proceeds (and
    fails LATER here for lack of an engine, which is not the gate's exit
    code and carries none of its message)."""
    sh = _wrapper(tmp_path)
    cp = subprocess.run(["bash", str(sh)], cwd=str(tmp_path),
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True,
                        env={**os.environ, "MB_LAUNCHED_BY": "manual"})
    assert "was called directly" not in cp.stderr
    assert cp.returncode != 2 or "Launch-door" not in cp.stderr


# ---- config provenance ------------------------------------------------ #

def test_provenance_names_each_values_source(tmp_path, machine_config):
    machine_config(
        {"execution": {"mode": "direct"},
         "script_generation": {"activation": "conda activate"}})
    bundle = tmp_path / "calc"
    bundle.mkdir()
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "source activate"}}))
    prov = config_provenance(project_dir=bundle)
    assert prov["effective"]["execution.mode"] == {
        "value": "direct", "from": "machine"}
    # the project file wins where both speak
    assert prov["effective"]["script_generation.activation"] == {
        "value": "source activate", "from": "project"}
    scopes = {s["scope"]: s for s in prov["sources"]}
    assert scopes["machine"]["found"] and scopes["machine"]["via"] == "config-dir"
    # `project`, ONE name for this scope (2026-08-23).  It answered to
    # "bundle" here and "project" in the registry -- and `bundle` already
    # names the portable prepped directory the JobSet framework's
    # --bundle points at.
    assert scopes["project"]["found"]


def test_provenance_never_carries_secret_material(tmp_path, machine_config):
    """The allowlist is the guarantee: `auth` and `tls` live in the same
    machine file and must never reach the formatted output, which lands in
    terminals, STAGE-PLAN.md and shipped logs.

    (`secret_key_file` was a third such section until 2026-08-31; the session
    key now has one home and the config cannot name it at all.)"""
    machine_config(
        {"execution": {"mode": "direct"},
         "tls": {"key": "PEMKEYMATERIAL"}})
    text = format_provenance(config_provenance(project_dir=None))
    assert "PEMKEYMATERIAL" not in text
    assert "tls" not in text
    assert "secret" not in text.lower()
    assert "execution.mode = 'direct'" in text


# ---- U10 (2026-08-12): the gate's four repaired edges ----------------- #

def test_the_refusal_verdict_lands_in_the_runwrap_log_too(tmp_path):
    """The log tee opens BEFORE the gate, so a refusal is a fact on
    disk -- the runwrap log alone answers what happened, even when
    nothing ran."""
    sh = _wrapper(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "MB_LAUNCHED_BY"}
    cp = subprocess.run(["bash", str(sh)], cwd=str(tmp_path),
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True, env=env)
    assert cp.returncode == 2
    assert "launched-by: NONE" in cp.stdout        # the job's own output
    logs = list(tmp_path.glob("*.runwrap-*.log"))
    assert logs, "no runwrap log written for the refused launch"
    assert "launched-by: NONE" in logs[0].read_text()


def test_help_is_answerable_without_a_claim_or_an_environment(tmp_path):
    """-h/--help is scanned BEFORE the gate and runs NONE of the
    bootstrap: asking what a script does never meets a permission
    prompt, and a broken activation cannot kill the answer."""
    sh = _wrapper(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "MB_LAUNCHED_BY"}
    cp = subprocess.run(["bash", str(sh), "-h"], cwd=str(tmp_path),
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True, env=env)
    assert cp.returncode == 0, cp.stderr
    assert "Usage:" in cp.stdout
    assert "was called directly" not in cp.stderr   # gate never spoke
    # `conda activate` does not exist in this bare shell: reaching the
    # usage proves the bootstrap was skipped, not survived.


def test_the_manual_yes_is_exported_for_the_warm_retry_exec(tmp_path):
    """The y-branch EXPORTS the claim: the warm-retry re-execs this very
    wrapper, and an unexported answer would re-prompt (or refuse, stdin
    long gone) mid-retry.  Rendered-text pin plus a child-visibility
    check driven through the same case arm."""
    sh = _wrapper(tmp_path)
    text = sh.read_text()
    assert "export MB_LAUNCHED_BY=manual" in text
    # EOF on the prompt must fall to the refusal, not die under set -e
    assert 'read -r _mb_answer || true' in text
    # and the interactive decline prints the verdict line too
    ix = text.index("Aborted.")
    assert "launched-by: NONE" in text[ix:ix + 300]
