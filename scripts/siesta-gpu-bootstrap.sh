#!/usr/bin/env bash
#
# scripts/siesta-gpu-bootstrap.sh
# ------------------------------
# Convenience wrapper for first-time install of `molbuilder-siesta-gpu`.
#
# This is a thin shim over `bash scripts/install-env.sh
# molbuilder-siesta-gpu`.  It exists because users coming to GPU SIESTA
# for the first time will look for a script with that name in
# scripts/.  All the real work happens in:
#
#   molbuilder/envs/recipes.py    (declarative BuildSpec)
#   molbuilder/envs/builds.py     (executor: clone+cmake+install+sentinels)
#   molbuilder/envs/install.py    (chains conda create -> pip -> build_spec)
#
# Usage:
#   bash scripts/siesta-gpu-bootstrap.sh
#   bash scripts/siesta-gpu-bootstrap.sh --dry-run     # plan only
#
# After first install completes, use scripts/siesta-gpu-rebuild.sh to
# wipe + rebuild components without re-creating the conda env.
#
# Pre-flight (Python layer enforces these and surfaces actionable errors):
#   * NVIDIA driver supporting CUDA >= 12.4
#   * CUDA toolkit (nvcc) reachable via $CUDA_HOME / /usr/local/cuda / PATH
#   * ~30 GB free disk under $CONDA_PREFIX
#   * Internet access for git clone (ELPA, ELSI, SIESTA)
#
# See docs/engines/siesta-gpu.md for the full engineering reference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Explain what's about to happen at the bash layer (before the python
# layer does its own detailed preflight + confirmation prompt).  This
# is the line a user lands on after typing the script name; the Python
# layer below will surface detected toolchain + ask for explicit
# confirmation.
cat <<'EOF'
========================================================================
  molbuilder-siesta-gpu  --  source-build env (SIESTA + ELPA, GPU)
========================================================================
This wrapper hands off to:
  python -m molbuilder envs install molbuilder-siesta-gpu

Architecture (2-component, per SIESTA 5.4 INSTALL.md):
  - ELPA built externally (CUDA-enabled eigensolver).
  - SIESTA cloned with --recurse-submodules: External/ holds
    libfdf, libpsml, xmlf90, libgridxc, ELSI, libxc as submodules;
    SIESTA cmake compiles them on the fly.
  - Everything else (gcc, MPI, BLAS, ScaLAPACK, NetCDF, HDF5, FFTW,
    libxc, CUDA toolkit) comes from conda-forge packages.

What the Python layer will do, in order:
  1. probe host: NVIDIA driver + nvidia-smi + CUDA (env) + disk + git
  2. show you a preflight report; ask for confirmation
  3. conda create + pip + extra steps (~5-10 min)
  4. clone + cmake build + install ELPA, then SIESTA (~30-45 min)
  5. write activate.d hook so the env's binaries land on PATH at
     the next `conda activate`

IMPORTANT -- existing env / artifact dir handling:

  If the conda env "molbuilder-siesta-gpu" already exists OR the
  artifact dir $CONDA_PREFIX/opt/siesta-gpu-stack/ has content from
  a prior failed install:

    DEFAULT (no flags)
        Install resumes from valid sentinels; phases that succeeded
        in the prior run are skipped.  Conda create reports "env
        already exists; skipping create".

    --clean
        REMOVES the entire conda env via `conda env remove --all -y`
        (every package goes -- gcc, cmake, openmpi, cuda toolkit) AND
        wipes the artifact dir.  Then runs a fresh install.  Use
        this when you want a guaranteed-clean state (after a failed
        install, a recipe upgrade, or just to be sure).  Destructive:
        explicit confirmation required.

    --rebuild=all
        Keeps the conda env; wipes per-component install + build dirs
        (sources stay so re-clone is skipped).  Useful when only the
        source-build state needs to be reset.

    --rebuild=elpa  /  --rebuild=siesta
        Wipes the named component + anything downstream of it.

You can pass:
  --dry-run        print every command + cost estimate; do not run.
  --yes / -y       skip confirmation prompts (for CI / unattended).
  --skip-network-check
                   skip the per-component `git ls-remote` reachability
                   probe (for firewalled hosts where clone works but
                   ls-remote is blocked).

Pre-flight (the Python layer enforces these and prints actionable
messages on failure):
  * NVIDIA driver supporting CUDA >= 12.4
  * CUDA toolkit (nvcc) at $CUDA_HOME / /usr/local/cuda / PATH
  * ~30 GB free disk under $CONDA_PREFIX
  * git can reach gitlab.mpcdf.mpg.de + github.com + gitlab.com

See docs/engines/siesta-gpu.md for the engineering reference.
========================================================================
EOF
echo

exec bash "${SCRIPT_DIR}/install-env.sh" "$@" molbuilder-siesta-gpu
