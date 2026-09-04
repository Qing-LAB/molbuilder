"""Does the recipe still resolve to the toolchain it claims?

WHY THIS FILE IS SEPARATE (and slow, and network-bound):

    Every other recipe test is a string assertion over
    ``recipe.conda_packages``.  Those are fast, deterministic, and
    structurally blind to the failure that motivated this module: in
    2026-09 the recipe's declared specs did not change at all, and the
    package the solver installed for them did.  conda-forge moved the
    default ``sysroot_linux-64`` from 2.17 to 2.39, the build began
    emitting binaries that would not start on RHEL-class hosts, and no
    static assertion could have noticed, because the recipe text was
    still exactly what its author intended.

    This is the only test in the suite that runs a real solve and
    compares the RESULT against what the recipe intends.  It is the one
    that catches upstream drift.

    It is marked ``slow`` and skips itself cleanly without conda or
    without a network, so pre-commit and offline runs are unaffected.
    Run it nightly, not per-commit::

        pytest tests/test_envs_solve_drift.py -m slow
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from molbuilder.envs.abi import parse_version
from molbuilder.envs.recipes import recipe_by_name


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SOLVE_TIMEOUT = 600


def _env_manager():
    """The same resolution order scripts/install-env.sh uses, minus the
    disk walk -- if conda is not on PATH or in the standard env vars,
    this test simply does not apply here."""
    for candidate in ("mamba", "conda"):
        found = shutil.which(candidate)
        if found:
            return found
    for var in ("MAMBA_EXE", "CONDA_EXE"):
        found = os.environ.get(var)
        if found and os.access(found, os.X_OK):
            return found
    return None


def _dry_run_solve(manager, specs):
    """Return ``{package_name: version_tuple}`` for a --dry-run solve.

    Skips (never fails) on anything that is not a solver verdict:
    no network, a dead mirror, a timeout.  A red test here must mean
    "the recipe resolves to something it should not", nothing else.
    """
    argv = [manager, "create", "--dry-run", "--json",
            "-n", "_molbuilder_drift_probe",
            "-c", "conda-forge", "--override-channels", *specs]
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=_SOLVE_TIMEOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        pytest.skip(f"could not run a solve here: {exc}")
    try:
        payload = json.loads(done.stdout)
    except ValueError:
        pytest.skip(f"solver returned no JSON (rc={done.returncode}): "
                    f"{done.stderr[-400:]}")
    if not payload.get("success", True) and "actions" not in payload:
        pytest.skip(f"solve did not complete: {payload.get('message', '')[:400]}")
    resolved = {}
    for record in (payload.get("actions") or {}).get("LINK", []):
        version = parse_version(record.get("version"))
        if version:
            resolved[record.get("name")] = version
    if not resolved:
        pytest.skip("solve produced no LINK actions to inspect")
    return resolved


@pytest.fixture(scope="module")
def solved():
    manager = _env_manager()
    if manager is None:
        pytest.skip("no conda/mamba available to solve with")
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    return _dry_run_solve(manager, recipe.conda_packages)


def _declared_pin(name):
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    for spec in recipe.conda_packages:
        if spec.split("=")[0] == name and "=" in spec:
            return parse_version(spec.split("=", 1)[1])
    return None


def test_the_solve_honours_the_sysroot_pin(solved):
    """The regression, tested where it actually lives: in the solver's
    answer rather than in the recipe's question."""
    pin = _declared_pin("sysroot_linux-64")
    assert pin is not None, "recipe stopped pinning sysroot_linux-64"
    resolved = solved.get("sysroot_linux-64")
    if resolved is None:
        pytest.skip("solve did not install sysroot_linux-64")
    assert resolved[:len(pin)] == pin, (
        f"recipe pins sysroot_linux-64={'.'.join(map(str, pin))} but the "
        f"solver returned {'.'.join(map(str, resolved))}.  Binaries built "
        f"against a sysroot newer than a target host's glibc link cleanly "
        f"and then fail to start."
    )


def test_the_solve_honours_the_gcc_pin(solved):
    """Same class of drift, different pin.  gcc has been stable, which
    is exactly why nobody thought to check its neighbour."""
    pin = _declared_pin("gcc_linux-64")
    resolved = solved.get("gcc_linux-64")
    if pin is None or resolved is None:
        pytest.skip("gcc_linux-64 not pinned or not in the solve")
    assert resolved[:len(pin)] == pin


def test_kernel_headers_follows_the_sysroot_down(solved):
    """The recipe deliberately does not pin kernel-headers, on the
    grounds that sysroot pins it exactly.  If that stops being true the
    recipe needs a second pin -- and this is how we find out."""
    resolved = solved.get("kernel-headers_linux-64")
    if resolved is None:
        pytest.skip("solve did not install kernel-headers_linux-64")
    assert resolved < (4, 0, 0), (
        f"sysroot 2.17 used to pin kernel-headers to 3.10.x; the solve "
        f"now returns {'.'.join(map(str, resolved))}.  The 'one decision, "
        f"one place' assumption in recipes.py no longer holds."
    )
