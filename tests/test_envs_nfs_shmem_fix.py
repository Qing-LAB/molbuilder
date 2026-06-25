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


def test_activate_hook_respects_user_set_OMPI_tmpdir():
    """Hook must NOT trample a user / scheduler-set value -- only set
    when unset.  If a user already pointed OMPI_MCA_orte_tmpdir_base
    at /scratch/$USER, the hook leaves it alone."""
    hook = _gpu_recipe().build_spec.activate_hook
    assert 'if [ -z "${OMPI_MCA_orte_tmpdir_base:-}" ]; then' in hook


def test_deactivate_hook_unsets_only_what_we_set():
    """Symmetry: deactivate hook unsets OMPI_MCA_orte_tmpdir_base
    only when WE set it (sentinel ``_MOLBUILDER_SIESTA_GPU_SET_OMPI_
    TMPDIR=1`` is the gate).  Without the sentinel check, deactivate
    would clobber a user-set value on every ``conda deactivate``."""
    deactivate = _gpu_recipe().build_spec.deactivate_hook
    assert "_MOLBUILDER_SIESTA_GPU_SET_OMPI_TMPDIR" in deactivate
    assert "unset OMPI_MCA_orte_tmpdir_base" in deactivate


# --------------------------------------------------------------------- #
#  Build wrapper contract                                                #
# --------------------------------------------------------------------- #


def test_build_wrapper_pins_ompi_tmpdir_to_local_storage():
    """The build wrapper in envs/builds.py exports
    OMPI_MCA_orte_tmpdir_base=/tmp so the .verify-phase MPI init
    (siesta --version) doesn't warn about NFS shmem.  Without this
    fix every fresh install of molbuilder-siesta-gpu spams two
    multi-line warnings during the verify phase.
    """
    import molbuilder.envs.builds as _b
    from pathlib import Path
    src = Path(_b.__file__).read_text()
    assert "OMPI_MCA_orte_tmpdir_base=/tmp" in src


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
    pkgs = " ".join(_host_recipe().conda_packages)
    assert "psutil" in pkgs


def test_host_env_psutil_version_matches_pyproject_floor():
    """The host recipe and pyproject.toml's runtime deps must agree
    on the psutil version floor -- drift would let conda install
    psutil<5.9 while the import-time check (if we ever add one)
    rejects it.  Today there's no version check; this is here so
    a future floor bump in pyproject can't silently leave the host
    recipe behind."""
    from pathlib import Path
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml not at repo root?"
    text = pyproject.read_text()
    # pyproject.toml has e.g.  "psutil>=5.9"  in the runtime deps list.
    assert "psutil>=5.9" in text
    pkgs = " ".join(_host_recipe().conda_packages)
    assert "psutil>=5.9" in pkgs
