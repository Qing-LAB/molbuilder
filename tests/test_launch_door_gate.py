"""The launch-door gate + config provenance (user decisions, 2026-08-12).

Contract: `job-contracts.md` § 2.6 (the Launch-door gate row: one launch
door; `submit` sets ``MB_LAUNCHED_BY``, a bare call warns or refuses,
``manual`` is the logged override) · the provenance allowlist in
`runtime_config` (paths + execution-relevant values only — the machine file
also carries auth/tls, which must never reach a terminal or a shipped log).
"""
from __future__ import annotations

import os
import subprocess


from molbuilder.runtime_config import config_provenance, format_provenance
from molbuilder.runwrap import write_run_wrapper
from molbuilder.jobset.model import Resources
import pytest
from molbuilder.runfiles import RunNames


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """An isolated cwd; conftest isolates the config root, so nothing here
    reads the developer's config."""
    monkeypatch.chdir(tmp_path)


def _wrapper(tmp_path):
    names = RunNames.of("JOB", "01_coarse", "hierarchical")
    deck = tmp_path / names.name(".fdf")
    deck.write_text("SystemLabel JOB\n")
    # A PROBED MACHINE, its record saying how a shell enters an environment
    # there -- the activation the generator reads (`configuration.md` § 4); a
    # wrapper is not rendered without one.  Its shape is STATED, as every
    # run's is (`architecture.md` § 5.2).
    from molbuilder.scheduler import Environment as _Env, Topology as _Topo
    (tmp_path / "environment.json").write_text(
        _Env(scheduler="slurm",
             topology=_Topo(sockets=2, cores_per_socket=32),
             env_init={"activation": "conda activate",
                                "preamble": "true"}).to_json()
        + "\n")
    return write_run_wrapper(deck, env="e", names=names,
                             resources=Resources(mpi_np=2, cpus_per_task=1),
                             emit_sbatch=False)


# ---- the gate, in the emitted text and under execution ---------------- #


def test_a_bare_noninteractive_call_refuses_with_the_fix(tmp_path):
    """No claim + no terminal -> exit 2 before ANY work, naming both the
    right door and the deliberate override.  This is the `nohup`, cron and
    hand-`sbatch` case."""
    sh = _wrapper(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "MB_LAUNCHED_BY"}
    cp = subprocess.run(["bash", str(sh), "--run", "0"], cwd=str(tmp_path),
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True, env=env)
    assert cp.returncode == 2
    assert "jobset launch" in cp.stderr
    assert "MB_LAUNCHED_BY=manual" in cp.stderr


# ---- config provenance ------------------------------------------------ #

def test_provenance_names_each_values_source(tmp_path, machine_config):
    machine_config(
        {"paths": {"projects": "/srv/projects"},
         "launch": {"mode": "direct"}})
    bundle = tmp_path / "calc"
    bundle.mkdir()
    prov = config_provenance(project_dir=bundle)
    assert prov["effective"]["paths.projects"] == {
        "value": "/srv/projects", "from": "machine"}
    assert prov["effective"]["launch.mode"] == {
        "value": "direct", "from": "machine"}
    scopes = {s["scope"]: s for s in prov["sources"]}
    assert scopes["machine"]["found"]


def test_provenance_never_carries_secret_material(tmp_path, machine_config):
    """The allowlist is the guarantee: `auth` and `tls` live in the same
    machine file and must never reach the formatted output, which lands in
    terminals, STAGE-PLAN.md and shipped logs."""
    machine_config(
        {"launch": {"mode": "direct"},
         "tls": {"key": "PEMKEYMATERIAL"}})
    text = format_provenance(config_provenance(project_dir=None))
    assert "PEMKEYMATERIAL" not in text
    assert "tls" not in text
    assert "secret" not in text.lower()
    assert "launch.mode = 'direct'" in text
