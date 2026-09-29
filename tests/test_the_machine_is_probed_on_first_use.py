"""A command asks the machine what it has only when it needs to know
(plan W36 ⑥, M2c) -- and what it is told reaches the recipes.

Two probes, each resolved on its first use:

* the machine snapshot -- the conda env listing (`diagnostics.
  get_capabilities`, 0.76 s measured on 2026-09-29), taken by `cli.main`
  before it parsed anything until that day;
* the CUDA version the GPU recipes target -- one ``nvidia-smi``
  (`recipes.builtin_recipes`), run at import of the recipes module until
  that day, which every command loaded through `cli.py`'s ``envs`` group.

Asserted through the command line itself, in a subprocess, because a probe
at IMPORT is exactly what an in-process test cannot see: the suite imported
the module long before.  Each probe is a stub that logs its call; the
driver stub reports CUDA 12, so an answer that reaches a recipe is told
apart from the project's default, 13 (`ops/env-framework.md` § 3).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def molbuilder(tmp_path):
    """``molbuilder(*args, **env) -> (run, calls)``: the CLI in a subprocess,
    with the env manager (recorded in molbuilder.json) and ``nvidia-smi`` as
    stubs that log each call; ``STUB_DRIVER_FAILS=1`` makes the driver's
    probe fail.  ``MOLBUILDER_CUDA_VERSION`` is unset unless passed."""
    log = tmp_path / "calls.log"
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (
            ("conda", "echo '{\"envs\": [], \"envs_details\": {}}'"),
            ("nvidia-smi", '[ -n "$STUB_DRIVER_FAILS" ] && exit 1\n'
                           "echo 'CUDA Version: 12.2'")):
        stub = stubs / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{log}"\n'
                        f"{body}\n")
        stub.chmod(0o755)
    config = tmp_path / "config"
    config.mkdir(mode=0o700)
    (config / "molbuilder.json").write_text(
        json.dumps({"envs": {"manager": str(stubs / "conda")}}))
    (config / "molbuilder.json").chmod(0o600)
    base = {**os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "MOLBUILDER_CONFIG_DIR": str(config),
            "HOME": str(tmp_path)}
    base.pop("MOLBUILDER_CUDA_VERSION", None)

    def run(*args, **env):
        log.write_text("")
        done = subprocess.run([sys.executable, "-m", "molbuilder", *args],
                              cwd=REPO, env={**base, **env},
                              capture_output=True, text=True, timeout=120)
        return done, log.read_text().splitlines()
    return run


def test_a_command_that_needs_neither_probe_runs_none(molbuilder):
    """``--help``, a group's ``--help`` (the root and ``envs`` callbacks
    run), and a real verb that reads neither the snapshot nor a recipe."""
    for args in (["--help"], ["envs", "list", "--help"],
                 ["jobset", "machines"]):
        run, calls = molbuilder(*args)
        assert run.returncode == 0, (args, run.stderr)
        assert calls == [], f"{' '.join(args)} asked the machine: {calls}"


def test_a_recipe_reader_runs_both(molbuilder):
    run, calls = molbuilder("envs", "list")
    assert run.returncode == 0, run.stderr
    assert any(c.startswith("conda env list") for c in calls), calls
    assert any(c.startswith("nvidia-smi") for c in calls), calls


def test_the_gpu_recipes_carry_the_cuda_version_each_tier_answers(molbuilder):
    """The three tiers of `_resolve_cuda_version`, read off what the install
    would run: the driver's major, the variable over it, and the project's
    default when the driver answers nothing (`ops/env-framework.md` § 3)."""
    run, _ = molbuilder("envs", "install", "molbuilder-pySCF", "--dry-run")
    assert "'cupy-cuda12x[ctk]'" in run.stdout, run.stdout
    assert "gpu4pyscf-cuda12x" in run.stdout, run.stdout
    run, _ = molbuilder("envs", "install", "molbuilder-siesta-gpu",
                        "--dry-run")
    assert "'cuda-version=12.*'" in run.stdout, run.stdout
    assert "NVIDIA driver supporting CUDA runtime 12.x" in run.stdout

    run, _ = molbuilder("envs", "install", "molbuilder-pySCF", "--dry-run",
                        MOLBUILDER_CUDA_VERSION="11.8")
    assert "'cupy-cuda11x[ctk]'" in run.stdout, run.stdout

    run, _ = molbuilder("envs", "install", "molbuilder-pySCF", "--dry-run",
                        STUB_DRIVER_FAILS="1")
    assert "'cupy-cuda13x[ctk]'" in run.stdout, run.stdout
