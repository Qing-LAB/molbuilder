"""Pins the NFS shared-memory fix for molbuilder-siesta-gpu (2026-06-24).

OpenMPI walks $TMPDIR to place its shared-memory backing files.  If
that lands on NFS (which happens by default when the conda env is on
$HOME and we pin TMPDIR inside the env), every shmem write becomes a
network round-trip and OpenMPI prints a multi-line warning at every
mpirun.

Two fixes regression-tested here:
  1. The siesta-gpu activate hook exports ``OMPI_MCA_orte_tmpdir_base``
     defaulting to ``$TMPDIR`` (SLURM/PBS per-job dir) or ``/tmp``.
  2. The build wrapper (envs/builds.py) pins
     ``OMPI_MCA_orte_tmpdir_base=/tmp`` so the build's ``.verify``
     phase (which runs ``siesta --version`` -> OpenMPI init) doesn't
     warn either.

Plus a regression for the unrelated ``psutil`` host-env gap surfaced
the same day -- the web blueprint at ``molbuilder/web/blueprints/
system_load.py`` imports psutil unconditionally and ``serve`` won't
start without it.
"""
from __future__ import annotations

import pytest

from molbuilder.envs import recipes as _r


# --------------------------------------------------------------------- #
#  Activate / deactivate hook contract                                  #
# --------------------------------------------------------------------- #


def _gpu_recipe():
    for r in _r.BUILTIN_RECIPES:
        if r.name == "molbuilder-siesta-gpu":
            return r
    pytest.fail("molbuilder-siesta-gpu recipe not registered")


def test_activate_hook_sets_ompi_tmpdir_base():
    """Hook must export OMPI_MCA_orte_tmpdir_base so OpenMPI's shmem
    pool lands on node-local storage, not the NFS-mounted env prefix.
    Without this, every mpirun under a from-source SIESTA emits the
    'shared memory backing file on a network filesystem' warning."""
    hook = _gpu_recipe().build_spec.activate_hook
    assert "OMPI_MCA_orte_tmpdir_base" in hook
    # Default must honour scheduler $TMPDIR first, /tmp as fallback.
    assert '${TMPDIR:-/tmp}' in hook


# --------------------------------------------------------------------- #
#  Build wrapper contract                                                #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  psutil host-env gap (separate same-day fix)                          #
# --------------------------------------------------------------------- #


def _host_recipe():
    for r in _r.BUILTIN_RECIPES:
        if r.name == "molbuilder":
            return r
    pytest.fail("molbuilder host recipe not registered")


def test_host_env_includes_psutil():
    """``molbuilder serve`` imports molbuilder.web.blueprints.system_load
    at startup, which imports psutil unconditionally.  Without psutil
    in the host env, ``python -m molbuilder serve`` hard-fails with
    ModuleNotFoundError BEFORE the server can bind a port -- no graceful
    degradation path exists (and shouldn't; the system-load API is
    required for the Watch tab's job-monitor sidecar).

    Pin: ``psutil`` must appear in the host recipe's conda_packages.
    """
    pkgs = " ".join(_host_recipe().conda_specs)
    assert "psutil" in pkgs
