#!/usr/bin/env bash
#
# scripts/install-env.sh -- thin bootstrap shim for `molbuilder envs`.
#
# DESIGN INVARIANT (read this before editing):
#
#   This script does ONE thing -- solve the chicken-and-egg of
#   "you can't run `molbuilder envs ...` until the molbuilder host
#   env exists."  Everything else lives in the Python CLI
#   (``molbuilder envs <subcommand>``).
#
#   The script's role:
#     1. Detect conda / mamba on PATH.
#     2. Ensure the host env exists (auto-create from the inlined
#        HOST_*_PACKAGES list if missing).
#     3. Forward "$@" verbatim to `python -m molbuilder envs ...`
#        inside the host env.
#
#   Everything else -- recipe-name validation, --rebuild component
#   lookup, --check / --dry-run semantics, the elsi→siesta alias,
#   per-component preflight, doctor reports -- lives in Python where
#   the Recipe / BuildSpec data lives.  See docs/design.md
#   (2026-06-24 decision-log: thin-shim rewrite) for the architectural
#   rationale and the drift-bug history that motivated it.
#
# BASE-SYSTEM ASSUMPTION:
#
#   ONLY `conda` or `mamba` is on PATH.  No python, no pip, no
#   pre-activated env.  The script bootstraps everything on top of
#   that.  This matches HPC / fresh-cluster reality where the OS
#   image carries a conda installer and nothing else.
#
# COMMAND SHAPE (1:1 with `molbuilder envs <subcommand>`):
#
#     # First-time install on a fresh machine:
#     bash scripts/install-env.sh bootstrap --yes
#     bash scripts/install-env.sh bootstrap --include-source-builds --yes
#
#     # Post-bootstrap operations (same as `molbuilder envs ...`):
#     bash scripts/install-env.sh list
#     bash scripts/install-env.sh doctor
#     bash scripts/install-env.sh install molbuilder-siesta-gpu --yes
#     bash scripts/install-env.sh install molbuilder-siesta-gpu --rebuild=siesta --yes
#     bash scripts/install-env.sh install molbuilder-siesta-gpu --clean --yes
#     bash scripts/install-env.sh install molbuilder-siesta --check
#     bash scripts/install-env.sh install molbuilder-siesta --dry-run
#
#   Equivalently, after running ``bootstrap`` once:
#
#     conda activate molbuilder
#     molbuilder envs list
#     molbuilder envs install molbuilder-siesta-gpu --rebuild=siesta --yes
#     ...
#
#   The shim and the activated-env path produce identical results.
#
# EXIT CODES:
#   0 = OK
#   1 = step failed inside the Python layer
#   2 = usage error / no conda manager / host env missing for non-bootstrap

set -euo pipefail

# ---- repo root resolution ------------------------------------------------
#
# molbuilder is NOT pip-installed (intentional convention -- the
# host env's site-packages does not contain ``molbuilder``).  So
# ``python -m molbuilder`` needs the repo on PYTHONPATH.  Resolving
# the repo root from the script's own location lets us call the
# shim from any CWD; the user's CWD is preserved so ./molbuilder.json
# overrides (which are read from CWD) still apply.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

HOST_ENV="${MOLBUILDER_HOST_ENV:-molbuilder}"

# ---- usage ---------------------------------------------------------------

usage() {
    cat <<'EOF'
molbuilder env installer -- thin shim over `molbuilder envs ...`.
Base-system requirement: conda OR mamba on PATH.

==============================================================
FIRST TIME on a fresh machine?  Run this:

    bash scripts/install-env.sh bootstrap --yes

That creates the host env, installs every conda-only backend
(SIESTA / PySCF / MDtools / tests), and runs a health check.
Idempotent: re-running skips envs already present.

Want GPU SIESTA too (~45 min extra)?  Add the opt-in flag:

    bash scripts/install-env.sh bootstrap --yes \
        --include-source-builds
==============================================================

Post-bootstrap subcommands (forwarded verbatim to the Python CLI):

  bash scripts/install-env.sh list
      Show every recipe + whether the env exists.

  bash scripts/install-env.sh doctor
      Health report: which envs exist, which verify, which need help.

  bash scripts/install-env.sh install <recipe> [flags]
      Install (or repair) one recipe.  Flags forwarded to Python:
        --dry-run                print the install plan; do not run
        --check                  report present/verify; do not install
        --rebuild=<component>    source-build recipes only (e.g.
                                 --rebuild=siesta, --rebuild=elpa,
                                 --rebuild=all on molbuilder-siesta-gpu)
        --clean                  wipe conda env + artifact dir, then
                                 reinstall (source-build recipes only)
        --yes / -y               proceed without confirmation prompts
        --skip-network-check     for firewalled hosts

  bash scripts/install-env.sh advise <recipe>
      Recommend mpi_np / omp / mps for the recipe + this host.

  bash scripts/install-env.sh validate <recipe>
      Run post-install correctness probes.

  bash scripts/install-env.sh bootstrap [flags]
      Install every conda-only recipe + run doctor.  Flags:
        --include-source-builds  also build GPU SIESTA (~45 min)
        --no-skip-existing       re-run install on envs already present
        --dry-run                print the plan; do not install
        --yes / -y               non-interactive (CI / HPC batch)

After bootstrap, an equivalent shortcut is:

    conda activate molbuilder
    molbuilder envs <subcommand>          # identical behavior

Environment variables:
  MOLBUILDER_HOST_ENV            host env name (default: molbuilder).
  MOLBUILDER_HOST_ENV_CHANNELS   comma-separated channels to pass when
                                 creating the host env, overriding
                                 .condarc.  Leave unset to respect
                                 ~/.condarc (recommended on HPC nodes
                                 with site-configured mirrors).
                                 Falls back to "conda-forge" only when
                                 .condarc has NO channels configured.
  MOLBUILDER_DEBUG_CHANNELS      when set, echo the raw output of
                                 ``<mgr> config --get channels`` so you
                                 can verify how the bootstrap probed
                                 your .condarc.  Useful when a host-env
                                 create silently fell back to
                                 ``-c conda-forge``.
  MOLBUILDER_BUILD_JOBS          source-build recipes: cap build
                                 concurrency (default: min(nproc, 8)).
  MOLBUILDER_CUDA_CC             siesta-gpu: force CUDA compute
                                 capability (e.g. 8.0) when nvidia-smi
                                 is unavailable.

.condarc / pip config:
  The host-env create step respects ~/.condarc by default -- if you
  have channels configured (likely via Miniforge defaults or
  ``conda config --add channels ...``), no ``-c`` flag is added.
  pip's user config (~/.pip/pip.conf) is always respected (the env
  manager doesn't intercept pip).  See "Environment variables" above
  for explicit overrides.
EOF
}

# ---- env-manager autodetect (mamba > conda) ------------------------------

ENV_MGR=""
detect_env_mgr() {
    [[ -n "${ENV_MGR}" ]] && return 0
    for candidate in mamba conda; do
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
        cat >&2 <<EOF
Error: no conda-compatible env manager found on PATH.
Looked for: mamba, conda.  Looked at:
  \$MAMBA_EXE = ${MAMBA_EXE:-(unset)}
  \$CONDA_EXE = ${CONDA_EXE:-(unset)}
Install one (Miniforge is the recommended distribution; it ships
mamba + conda-forge as defaults).  See docs/README_install.md.
EOF
        exit 2
    fi
    if [[ -z "${_ENV_MGR_BANNERED:-}" ]]; then
        echo "[molbuilder] env manager: ${ENV_MGR}" >&2
        _ENV_MGR_BANNERED=1
    fi
}

# ---- host-env package list (chicken-and-egg solver) ----------------------
#
# This list MUST match molbuilder/envs/recipes.py::_HOST.conda_packages +
# _HOST.pip_packages.  A drift-guard test
# (tests/test_envs_readme_consistency.py) asserts the two lists
# stay in sync.  The bash duplication is intentional: bash cannot
# read the Python recipe without first having Python, and Python
# isn't available until the host env exists.

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
    # Parse ``<mgr> env list`` robustly across conda / mamba (2 header
    # lines starting with ``#``) and micromamba (3 header lines: a
    # column header + a ─-rule separator).  Filter:
    #   * skip blank lines (NF == 0)
    #   * skip column-header / separator lines
    #   * print column 1 (the env name)
    "${ENV_MGR}" env list 2>/dev/null \
            | awk 'NF && $1 !~ /^(#|─|Name$)/ {print $1}' \
            | grep -qx "${HOST_ENV}"
}

resolve_host_env_channels() {
    # Respect the user's ~/.condarc.  On HPC nodes the user (or the
    # site admin) typically has channels + channel_priority configured
    # already -- often pointing at an internal mirror or pinning a
    # specific channel order for license/security reasons.  Our pre-
    # 2026-06-25 behavior always prepended ``-c conda-forge`` to the
    # create command, which overrode that order silently.
    #
    # Resolution order (first hit wins):
    #   1. MOLBUILDER_HOST_ENV_CHANNELS env var (explicit user
    #      override; comma-separated, e.g. "site-mirror,conda-forge").
    #   2. ~/.condarc-configured channels (probed via
    #      ``<mgr> config --get channels``).  If the user has any
    #      channel listed, we pass NO ``-c`` flag and let .condarc
    #      drive the resolve.
    #   3. Fallback: ``-c conda-forge`` (covers bare-conda installs
    #      with no .condarc, where the default ``defaults`` channel
    #      doesn't carry our scientific stack).
    HOST_ENV_CHANNEL_ARGS=()
    if [[ -n "${MOLBUILDER_HOST_ENV_CHANNELS:-}" ]]; then
        local _chs
        IFS=',' read -r -a _chs <<< "${MOLBUILDER_HOST_ENV_CHANNELS}"
        for _ch in "${_chs[@]}"; do
            HOST_ENV_CHANNEL_ARGS+=(-c "${_ch}")
        done
        echo "[molbuilder] channels from MOLBUILDER_HOST_ENV_CHANNELS:" \
             "${_chs[*]}" >&2
        return 0
    fi
    local _raw _configured
    _raw="$("${ENV_MGR}" config --get channels 2>/dev/null || true)"
    if [[ -n "${MOLBUILDER_DEBUG_CHANNELS:-}" ]]; then
        # Surface the raw probe output so HPC users can verify what
        # ``<mgr> config --get channels`` actually emitted -- mamba's
        # and conda's formats have varied across versions, and a
        # silently-wrong probe would fall back to ``-c conda-forge``
        # (the safe default) without telling the user their
        # ``.condarc`` was misread.
        echo "[molbuilder] DEBUG: raw '<${ENV_MGR}> config --get channels' output:" >&2
        if [[ -z "${_raw}" ]]; then
            echo "[molbuilder] DEBUG: (empty)" >&2
        else
            printf '[molbuilder] DEBUG: | %s\n' "${_raw}" >&2 \
                || true  # printf can fail on weird input; don't abort
        fi
    fi
    _configured="$(echo "${_raw}" \
                  | awk '/--add channels/ {print $NF}' \
                  | tr -d "'\"" \
                  | tr '\n' ' ')"
    if [[ -n "${_configured// /}" ]]; then
        echo "[molbuilder] respecting user's .condarc channels:" \
             "${_configured}" >&2
        # Leave HOST_ENV_CHANNEL_ARGS empty -- let .condarc + the env
        # manager's channel_priority setting handle ordering.
        return 0
    fi
    echo "[molbuilder] no channels configured in .condarc;" \
         "falling back to -c conda-forge" >&2
    HOST_ENV_CHANNEL_ARGS=(-c conda-forge)
}

create_host_env() {
    resolve_host_env_channels
    echo "[molbuilder] creating host env '${HOST_ENV}' (~2 min)" >&2
    "${ENV_MGR}" create -n "${HOST_ENV}" "${HOST_ENV_CHANNEL_ARGS[@]}" \
        --yes "${HOST_CONDA_PACKAGES[@]}"
    if [[ ${#HOST_PIP_PACKAGES[@]} -gt 0 ]]; then
        # pip's own config (~/.pip/pip.conf, ~/.config/pip/pip.conf)
        # is respected automatically -- the env manager doesn't get
        # in pip's way here, so internal PyPI mirrors / index-url
        # overrides set in the user's pip config apply transparently.
        echo "[molbuilder] pip-installing host-env extras: ${HOST_PIP_PACKAGES[*]}" >&2
        "${ENV_MGR}" run -n "${HOST_ENV}" --no-capture-output \
            python -m pip install "${HOST_PIP_PACKAGES[@]}"
    fi
    echo "[molbuilder] host env '${HOST_ENV}' ready" >&2
}

# ---- dispatch (1:1 with `molbuilder envs ...`) ---------------------------
#
# Sets PYTHONPATH=$REPO_ROOT so ``python -m molbuilder`` can find
# the package regardless of the user's CWD.  mamba / conda ``run``
# inherits the env var.  See REPO_ROOT comment above.

dispatch() {
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    MOLBUILDER_REPO_ROOT="${REPO_ROOT}" \
        "${ENV_MGR}" run -n "${HOST_ENV}" --no-capture-output \
        python -m molbuilder envs "$@"
}

# ---- main ----------------------------------------------------------------

# State machine, encoded in this dispatcher:
#
#                       ┌─────────────────────────────┐
#                       │   ENTRY (any subcommand)    │
#                       └────────────┬────────────────┘
#                                    │
#                          require_conda  ── fails → exit 2
#                                    │
#               ┌────────────────────┴────────────────────┐
#               │                                         │
#         subcmd = "bootstrap"                  any other subcmd
#               │                                         │
#       host env missing? ───── yes ──→ create        host env missing?
#               │ no                       │              │ yes
#               ▼                          ▼              ▼
#                              dispatch "$@"     error: run bootstrap first
#                              (PYTHONPATH set)
#
# The only auto-creation point is bootstrap.  Every other subcommand
# expects the host env to already exist -- if it doesn't, the user's
# next step is bootstrap, not "copy this conda block."

if [[ $# -eq 0 ]]; then
    cat <<'EOF' >&2
No subcommand given.  First-time install on a fresh machine?

    bash scripts/install-env.sh bootstrap --yes

Full usage below.
EOF
    echo "" >&2
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help|help)
        usage
        exit 0
        ;;
    bootstrap)
        # The one auto-create path.  Any flags after ``bootstrap``
        # (--yes, --include-source-builds, --dry-run, ...) forward
        # verbatim to the Python ``cmd_bootstrap`` handler.
        require_conda
        if ! host_env_exists; then
            create_host_env
        fi
        dispatch "$@"
        ;;
    *)
        # Every other subcommand needs the host env up.  No
        # auto-create here -- if the host env is missing the user
        # wants ``bootstrap``, not a silently-created skeleton env
        # that then errors halfway through their non-bootstrap
        # invocation.
        require_conda
        if ! host_env_exists; then
            cat >&2 <<EOF
Error: host env '${HOST_ENV}' does not exist.

Run this first:
    bash scripts/install-env.sh bootstrap --yes

Or, to override the host env name:
    MOLBUILDER_HOST_ENV=<name> bash scripts/install-env.sh bootstrap --yes
EOF
            exit 2
        fi
        dispatch "$@"
        ;;
esac
