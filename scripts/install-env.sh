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
       bash scripts/install-env.sh --dry-run <recipe-name>
       bash scripts/install-env.sh --rebuild=<comp> <recipe-name>
                                                    # source-build recipes only
                                                    # (e.g. molbuilder-siesta-gpu)

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

require_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda CLI not found on PATH." >&2
        echo "Install conda first; see docs/README_install.md." >&2
        exit 2
    fi
}

require_host_env() {
    if ! conda env list 2>/dev/null \
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
    # `--no-capture-output` so the user sees conda's progress in real time.
    conda run -n "${HOST_ENV}" --no-capture-output \
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
    *)
        require_conda
        require_host_env
        dispatch install "$1"
        ;;
esac
