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

IMPORTANT -- artifact directory should be clean:
  If $CONDA_PREFIX/opt/siesta-gpu-stack/ already exists from a prior
  failed install or an older recipe version, the preflight will warn
  you about stale directories and tell you to pass `--rebuild=all`
  to wipe everything and start fresh.  Resuming from sentinels on
  partial state usually works, but starting clean is the safe call.

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
