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

# Package list for the host env -- KEEP IN SYNC with
# molbuilder/envs/recipes.py::_HOST.conda_packages +
# _HOST.pip_packages.  Drift-guard test at
# tests/test_envs_readme_consistency.py asserts the two lists
# match.  Inlined here so the bash script can create the host env
# on a truly fresh machine -- it cannot read the Python recipe
# without Python, and we cannot dispatch into the host env until
# the host env exists.  Chicken-and-egg: this is the egg.
HOST_CONDA_PACKAGES=(
    python=3.12 pip
    numpy ase sisl
    rdkit openbabel biopython
    flask click plotly
    authlib python-cas
    pytest pyflakes
    numactl
)
HOST_PIP_PACKAGES=(
    PeptideBuilder pubchempy
)

host_env_exists() {
    "${ENV_MGR}" env list 2>/dev/null \
            | awk 'NR>2 {print $1}' \
            | grep -qx "${HOST_ENV}"
}

create_host_env() {
    # Create the host env using the env manager that
    # detect_env_mgr already picked.  --yes so we don't block on
    # the conda confirmation prompt; HOST_CONDA_PACKAGES is the
    # single source of truth for what lands inside.
    echo "[molbuilder] creating host env '${HOST_ENV}' (~2 min)" >&2
    "${ENV_MGR}" create -n "${HOST_ENV}" -c conda-forge --yes \
        "${HOST_CONDA_PACKAGES[@]}"
    # Pip-installable packages that aren't on conda-forge.  Run
    # via the freshly-created env's python so we don't depend on a
    # global pip.
    if [[ ${#HOST_PIP_PACKAGES[@]} -gt 0 ]]; then
        echo "[molbuilder] pip-installing host-env extras: ${HOST_PIP_PACKAGES[*]}" >&2
        "${ENV_MGR}" run -n "${HOST_ENV}" --no-capture-output \
            python -m pip install "${HOST_PIP_PACKAGES[@]}"
    fi
    echo "[molbuilder] host env '${HOST_ENV}' ready" >&2
}

ensure_host_env() {
    # Bootstrap-friendly: create the host env if missing instead
    # of erroring out.  Used by --bootstrap so a truly fresh
    # machine works in one command.  ``require_host_env`` below
    # is the strict variant used by per-recipe subcommands where
    # the user is expected to have set up the host env already.
    if ! host_env_exists; then
        create_host_env
    fi
}

require_host_env() {
    # Strict: error out if the host env doesn't exist.  Used by
    # the per-recipe subcommands (``--list``, ``--doctor``,
    # ``--check``, ``--dry-run``, etc.) where dispatching into a
    # missing host env would be a confusing error rather than the
    # user's intent.  The --bootstrap path uses ``ensure_host_env``
    # which auto-creates.
    if ! host_env_exists; then
        echo "Error: host env '${HOST_ENV}' does not exist." >&2
        echo "" >&2
        echo "Run the bootstrap to create it + every backend env:" >&2
        echo "  bash scripts/install-env.sh --bootstrap --yes" >&2
        echo "" >&2
        echo "Or, to create JUST the host env (without backends):" >&2
        echo "  ${ENV_MGR} create -n ${HOST_ENV} -c conda-forge --yes \\" >&2
        echo "      ${HOST_CONDA_PACKAGES[*]}" >&2
        echo "  ${ENV_MGR} run -n ${HOST_ENV} python -m pip install \\" >&2
        echo "      ${HOST_PIP_PACKAGES[*]}" >&2
        echo "" >&2
        echo "Or set MOLBUILDER_HOST_ENV to point at an existing env" >&2
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
        # --bootstrap [extra args...]: full base-system -> every-env
        # path.  Creates the host env if missing (auto, no manual
        # conda block), then dispatches into it to install every
        # backend recipe and run doctor.  Extra args (--yes,
        # --include-source-builds, --dry-run, --no-skip-existing) are
        # forwarded to the python layer.  Idempotent (already-present
        # envs are skipped by default).
        #
        # 2026-06-24 UX fix: pre-fix this called ``require_host_env``
        # which ERRORED OUT when the host env was missing and pointed
        # the user at a doc-section conda-block.  That contradicted
        # the whole bootstrap promise -- a bootstrap creates
        # everything from nothing.  ``ensure_host_env`` now creates
        # the host env if missing (using HOST_CONDA_PACKAGES +
        # HOST_PIP_PACKAGES above) before dispatching the Python
        # layer.
        shift
        require_conda
        ensure_host_env
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
