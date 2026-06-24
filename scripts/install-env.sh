#!/usr/bin/env bash
#
# scripts/install-env.sh -- thin wrapper around `molbuilder envs install`.
#
# Why this exists: a user setting up molbuilder on a fresh machine does
# NOT yet have the host env created; the bootstrap step ("create the
# host env from the README's conda block") is the first thing they need
# to do.  Once the host env exists, every other env is installed by
# delegating to `molbuilder envs install`.  This script automates that
# delegation so users have a single entry point:
#
#     bash scripts/install-env.sh <recipe-name>
#     bash scripts/install-env.sh --list
#     bash scripts/install-env.sh --check <recipe-name>
#
# What the script does NOT do:
#   - Install conda itself (out of scope; conda is a hard precondition).
#   - Install OS-level CUDA / NVIDIA drivers (system layer; see the
#     2026-06-14 design discussion).
#   - Pick env names; recipe names are the canonical names from
#     molbuilder/envs/recipes.py.  `molbuilder.json` overrides apply
#     automatically inside the python layer.
#
# Exit codes mirror `molbuilder envs install` (0 = OK; 1 = step failed;
# 2 = usage error / no conda).
set -euo pipefail

HOST_ENV="${MOLBUILDER_HOST_ENV:-molbuilder}"

usage() {
    cat <<EOF
Usage: bash scripts/install-env.sh [--list | --check] <recipe-name>
       bash scripts/install-env.sh --list           # show all recipes
       bash scripts/install-env.sh --doctor         # full health report
       bash scripts/install-env.sh --bootstrap [--yes]
                                                    # install EVERY conda-only
                                                    # recipe + run doctor; the
                                                    # one-command first-run path
                                                    # for HPC / fresh-machine
                                                    # deployment.  Use --yes
                                                    # for non-interactive (CI,
                                                    # batch jobs).  Append
                                                    # --include-source-builds
                                                    # to also build SIESTA-GPU
                                                    # (~45 min, opt-in).
       bash scripts/install-env.sh --dry-run <recipe-name>
       bash scripts/install-env.sh --rebuild=<comp> <recipe-name>
                                                    # source-build recipes only
                                                    # (e.g. molbuilder-siesta-gpu)
       bash scripts/install-env.sh --clean <recipe-name>
                                                    # WIPE artifact dir
                                                    # (source-build recipes only;
                                                    # confirmation required)

Recipe names are the canonical defaults from
molbuilder/envs/recipes.py (e.g. molbuilder-siesta).
Per-machine overrides from molbuilder.json apply automatically.

Environment variables:
  MOLBUILDER_HOST_ENV   conda env that has molbuilder's host stack
                        installed (default: molbuilder).  The script
                        dispatches into this env to invoke the CLI.
  MOLBUILDER_BUILD_JOBS for source-build recipes: cap build concurrency
                        (default: min(nproc, 8)).
  MOLBUILDER_CUDA_CC    for siesta-gpu: force compute capability
                        (e.g. 8.0) when nvidia-smi is unavailable.
EOF
}

# Conda-compatible env manager autodetect (mamba > micromamba > conda).
# Mamba / micromamba are drop-in replacements for ``conda create``,
# ``conda run``, ``conda env list`` and other commands this script + the
# Python layer issue.  Mamba is ~5-10x faster on HPC clusters with slow
# filesystems; micromamba is the only option on systems where the user
# lacks admin rights to install Miniconda.  All three accept the same
# CLI surface for our use case, so detection is transparent downstream.
ENV_MGR=""
detect_env_mgr() {
    if [[ -n "${ENV_MGR}" ]]; then
        return 0
    fi
    for candidate in mamba micromamba conda; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            ENV_MGR="${candidate}"
            return 0
        fi
    done
    # Fall back to env vars set by mamba's / conda's activation hooks.
    if [[ -n "${MAMBA_EXE:-}" && -x "${MAMBA_EXE}" ]]; then
        ENV_MGR="${MAMBA_EXE}"
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        ENV_MGR="${CONDA_EXE}"
        return 0
    fi
    return 1
}

require_conda() {
    if ! detect_env_mgr; then
        echo "Error: no conda-compatible env manager found on PATH." >&2
        echo "Looked for: mamba, micromamba, conda.  Looked at:" >&2
        echo "  \$MAMBA_EXE = ${MAMBA_EXE:-(unset)}" >&2
        echo "  \$CONDA_EXE = ${CONDA_EXE:-(unset)}" >&2
        echo "Install one (Miniconda / Miniforge / micromamba); see" >&2
        echo "docs/README_install.md." >&2
        exit 2
    fi
    # First-run banner: which manager we'll use.  Only print once.
    if [[ -z "${_ENV_MGR_BANNERED:-}" ]]; then
        echo "[molbuilder] env manager: ${ENV_MGR}" >&2
        _ENV_MGR_BANNERED=1
    fi
}

require_host_env() {
    if ! "${ENV_MGR}" env list 2>/dev/null \
            | awk 'NR>2 {print $1}' \
            | grep -qx "${HOST_ENV}"; then
        echo "Error: host env '${HOST_ENV}' does not exist." >&2
        echo "" >&2
        echo "Create it with the conda block from" >&2
        echo "  docs/README_install.md § 'Host env (required)'" >&2
        echo "or set MOLBUILDER_HOST_ENV to point at an existing env" >&2
        echo "that has molbuilder's host-side packages." >&2
        exit 2
    fi
}

dispatch() {
    # ``--no-capture-output`` so the user sees the inner process's
    # progress in real time.  Mamba / micromamba accept the same flag.
    "${ENV_MGR}" run -n "${HOST_ENV}" --no-capture-output \
        python -m molbuilder envs "$@"
}

case "${1:-}" in
    ""|-h|--help)
        usage
        exit 0
        ;;
    --list)
        require_conda
        require_host_env
        dispatch list
        ;;
    --doctor)
        require_conda
        require_host_env
        dispatch doctor
        ;;
    --bootstrap)
        # --bootstrap [extra args...]: install every conda-only recipe
        # in one pass + run doctor at the end.  Extra args (--yes,
        # --include-source-builds, --dry-run, --no-skip-existing) are
        # forwarded to the python layer.  Designed for HPC first-run
        # deploy where the user wants one command to get every env set
        # up.  Idempotent (already-present envs are skipped by default).
        shift
        require_conda
        require_host_env
        dispatch bootstrap "$@"
        ;;
    --check)
        if [[ $# -lt 2 ]]; then
            echo "Error: --check requires a recipe name." >&2
            usage
            exit 2
        fi
        require_conda
        require_host_env
        dispatch install --check "$2"
        ;;
    --dry-run)
        if [[ $# -lt 2 ]]; then
            echo "Error: --dry-run requires a recipe name." >&2
            usage
            exit 2
        fi
        require_conda
        require_host_env
        dispatch install --dry-run "$2"
        ;;
    --rebuild=*)
        # --rebuild=<comp> <recipe-name>: for source-build recipes only
        # (e.g. siesta-gpu).  Component must be one of the recipe's
        # build_spec.components, or `all` to wipe everything.
        REBUILD_ARG="${1#--rebuild=}"
        if [[ -z "$REBUILD_ARG" || $# -lt 2 ]]; then
            echo "Error: --rebuild=<component> requires a recipe name." >&2
            usage
            exit 2
        fi
        require_conda
        require_host_env
        dispatch install --rebuild "$REBUILD_ARG" "$2"
        ;;
    --clean)
        # --clean <recipe-name>: WIPE the artifact directory under
        # $CONDA_PREFIX/opt/<artifact_subdir>/ before installing.
        # Destructive; the Python layer asks for confirmation.
        if [[ $# -lt 2 ]]; then
            echo "Error: --clean requires a recipe name." >&2
            usage
            exit 2
        fi
        require_conda
        require_host_env
        dispatch install --clean "$2"
        ;;
    *)
        require_conda
        require_host_env
        dispatch install "$1"
        ;;
esac
