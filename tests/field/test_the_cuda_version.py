"""FIELD TEST -- the GPU recipes carry the CUDA major this machine's driver says
it supports, and the variable over it (`ops/env-framework.md` § 3,
`_resolve_cuda_version`).

What a driver answers is read where it answers: this runs on the target, against
its real ``nvidia-smi``, and is skipped where there is none.  In a subprocess,
because the recipes are read once in a process.

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest


def _dry_run(recipe, **env):
    base = {k: v for k, v in os.environ.items()
            if k != "MOLBUILDER_CUDA_VERSION"}
    done = subprocess.run(
        [sys.executable, "-m", "molbuilder", "envs", "install", recipe,
         "--dry-run"], env={**base, **env}, capture_output=True, text=True,
        timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def driver_major():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on this machine")
    said = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                          timeout=5).stdout
    found = re.search(r"CUDA Version:\s*(\d+)", said)
    if found is None:
        pytest.skip("this driver states no CUDA version")
    return found.group(1)


def test_the_recipes_carry_the_drivers_major(driver_major):
    assert f"cupy-cuda{driver_major}x" in _dry_run("molbuilder-pySCF")
    assert f"cuda-version={driver_major}." in _dry_run("molbuilder-siesta-gpu")


def test_the_variable_is_over_the_driver(driver_major):
    other = "11" if driver_major != "11" else "12"
    said = _dry_run("molbuilder-pySCF", MOLBUILDER_CUDA_VERSION=f"{other}.8")
    assert f"cupy-cuda{other}x" in said, said
