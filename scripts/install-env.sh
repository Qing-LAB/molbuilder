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
#     1. Locate conda / mamba (PATH, MAMBA_EXE/CONDA_EXE, CONDA_PREFIX
#        walk, common install prefixes on disk).  Report what it
#        found and ask the user to confirm (unless --yes is passed).
#     2. On ``bootstrap``: ensure the host env exists (auto-create
#        from the inlined HOST_*_PACKAGES list if missing).
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
#   The system provides ``conda`` OR ``mamba``.  No other tools
#   required.  The script does not install conda for you -- that
#   is the OS / cluster admin's job (Miniforge / Miniconda).
#
#   Detection is robust: PATH first, then activation-hook env vars
#   (MAMBA_EXE / CONDA_EXE), then a walk up CONDA_PREFIX (when an
#   env is active), then a search of common install prefixes on
#   disk (handles the very common case where ``conda`` is a shell
#   function loaded in interactive shells but the binary in
#   ``~/miniconda3/bin/`` isn't on PATH for sub-shells / scripts).
#   If all four fail, the script tells the user where it looked
#   and exits 2 -- it does NOT silently install anything.
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
#   This shim is the canonical entry point.  molbuilder is not
#   pip-installed into the host env, so after ``conda activate
#   molbuilder`` the bare ``molbuilder`` command does NOT exist on
#   PATH -- you'd need ``python -m molbuilder ...`` with PYTHONPATH
#   pointed at the repo root.  Sticking with the shim avoids that
#   footgun.
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

# AUTO_YES is set by the main case below when any arg is ``--yes``
# / ``-y``.  The env-manager confirmation prompt and any other
# interactive prompt the script owns skip when this is set.  CI /
# HPC batch always passes --yes anyway.
AUTO_YES=0
ORIGINAL_ARGS=("$@")

# ---- usage ---------------------------------------------------------------

usage() {
    cat <<'EOF'
molbuilder env installer -- thin shim over `molbuilder envs ...`.
Base-system requirement: conda OR mamba installed on the system
(the script probes PATH, MAMBA_EXE / CONDA_EXE, CONDA_PREFIX, and
common install prefixes -- you don't need to set anything if it's
in a standard location).  Asks you to confirm the detected env
manager unless you pass --yes / -y.

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

Note: this shim is the canonical entry point.  molbuilder is not
pip-installed into the host env (intentional convention), so after
``conda activate molbuilder`` the bare ``molbuilder`` command does
NOT exist on PATH -- you'd need ``python -m molbuilder ...`` with
PYTHONPATH set to the repo root.  Sticking with the shim avoids
that footgun.

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
# Common install-prefix list, searched in order.  Covers Miniforge
# (recommended), Mambaforge (legacy), Miniconda, Anaconda, and a
# few generic /opt layouts seen on HPC.  Picked up here so the
# script Just Works without the user having to set MAMBA_EXE /
# CONDA_EXE by hand -- many setups install conda as a SHELL FUNCTION
# via .bashrc (so ``command -v conda`` only sees it in interactive
# shells) while the underlying ``~/miniconda3/bin/conda`` binary
# is NOT on PATH for sub-shells / non-interactive scripts.
_CONDA_PREFIX_CANDIDATES=(
    "$HOME/miniforge3"
    "$HOME/miniforge"
    "$HOME/mambaforge"
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "$HOME/conda"
    "/opt/miniforge3"
    "/opt/miniforge"
    "/opt/mambaforge"
    "/opt/miniconda3"
    "/opt/anaconda3"
    "/opt/conda"
    "/usr/local/miniforge3"
    "/usr/local/miniconda3"
)

_CONDA_PROBE_LOG=()

_record_probe() {
    _CONDA_PROBE_LOG+=("$1")
}

detect_env_mgr() {
    [[ -n "${ENV_MGR}" ]] && return 0
    # 1. Already on PATH? -- works when the user runs the script
    #    from inside a shell where conda init has injected PATH.
    #    Use ``type -P`` rather than ``command -v``: -P matches only
    #    executable FILES on PATH, never shell functions or aliases.
    #    Defends against the case where the user has ``conda`` /
    #    ``mamba`` defined as exported shell functions but no binary
    #    on PATH -- ``command -v`` would return the function body
    #    string and downstream calls would fail with bizarre errors.
    local _p
    for candidate in mamba conda; do
        _p="$(type -P "${candidate}" 2>/dev/null || true)"
        if [[ -n "${_p}" && -x "${_p}" ]]; then
            ENV_MGR="${_p}"
            _record_probe "PATH: found ${candidate} at ${ENV_MGR}"
            return 0
        fi
        _record_probe "PATH: ${candidate} not found"
    done
    # 2. Activation-hook env vars (set by ``conda activate`` or by
    #    sourcing conda.sh).  Available even when conda is exposed
    #    only as a shell function in the parent shell.
    for var in MAMBA_EXE CONDA_EXE; do
        local v="${!var:-}"
        if [[ -n "${v}" && -x "${v}" ]]; then
            ENV_MGR="${v}"
            _record_probe "env: \$${var} -> ${v}"
            return 0
        fi
        _record_probe "env: \$${var} = ${v:-(unset)} (not executable)"
    done
    # 3. CONDA_PREFIX -- set when an env (any env, including base)
    #    is currently activated.  Walk up to find the install root
    #    (CONDA_PREFIX of a non-base env is ``<root>/envs/<name>``;
    #    of base it's ``<root>``).
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        # Bounded walk so a deeply-nested CONDA_PREFIX can't run forever.
        # 6 levels is enough to cover ``<root>/envs/<name>/<subdir>...``
        # plus whatever the user nests above the install root.
        local p="${CONDA_PREFIX}"
        local _i
        for _i in 1 2 3 4 5 6; do
            for mgr in mamba conda; do
                if [[ -x "${p}/bin/${mgr}" ]]; then
                    ENV_MGR="${p}/bin/${mgr}"
                    _record_probe "CONDA_PREFIX walk: found ${ENV_MGR}"
                    return 0
                fi
            done
            [[ "${p}" == "/" ]] && break
            p="$(dirname "${p}")"
        done
        _record_probe "CONDA_PREFIX=${CONDA_PREFIX}: no mamba/conda found above"
    fi
    # 4. Probe common install locations on disk.  This catches the
    #    very common "``conda`` is a shell function in .bashrc but
    #    ``~/miniconda3/bin/`` isn't on PATH for sub-shells" setup.
    for prefix in "${_CONDA_PREFIX_CANDIDATES[@]}"; do
        for mgr in mamba conda; do
            if [[ -x "${prefix}/bin/${mgr}" ]]; then
                ENV_MGR="${prefix}/bin/${mgr}"
                _record_probe "disk: found ${ENV_MGR}"
                return 0
            fi
        done
    done
    _record_probe "disk: checked ${#_CONDA_PREFIX_CANDIDATES[@]} common prefixes, none had a mamba or conda binary"
    return 1
}

require_conda() {
    if ! detect_env_mgr; then
        cat >&2 <<EOF
Error: no conda-compatible env manager found.

molbuilder expects ``conda`` or ``mamba`` to be installed on the
system already.  Installing one is the OS / cluster admin's job,
not this script's.

What the script checked, in order:

EOF
        for line in "${_CONDA_PROBE_LOG[@]}"; do
            echo "  - ${line}" >&2
        done
        cat >&2 <<EOF

If conda/mamba lives somewhere not in that list, point the script
at it explicitly:

    CONDA_EXE=/path/to/conda bash ${SCRIPT_DIR}/install-env.sh ${ORIGINAL_ARGS[*]:-bootstrap --yes}

If conda/mamba is not installed at all, install Miniforge,
Miniconda, or your site's conda distribution by hand first, then
re-run.  See docs/README_install.md for links.
EOF
        exit 2
    fi
    if [[ -z "${_ENV_MGR_BANNERED:-}" ]]; then
        echo "[molbuilder] env manager detected: ${ENV_MGR}" >&2
        if [[ "${AUTO_YES}" -eq 0 ]]; then
            # User-facing confirmation.  Lets the user catch the case
            # where the script picked a different mamba/conda than
            # they intended (e.g. an old Anaconda install lingering
            # in ~/anaconda3 alongside a newer Miniforge they meant
            # to use).  --yes / -y in ORIGINAL_ARGS skips this prompt.
            # ``read ... </dev/tty 2>/dev/null`` -- suppress bash's
            # own redirect error ("/dev/tty: No such device or
            # address") when there's no TTY (nohup, container without
            # a tty alloc, CI runner).  Our own handler prints a
            # clearer message right after.
            read -r -p "[molbuilder] use this env manager? [Y/n] " _ans </dev/tty 2>/dev/null \
                || { echo "[molbuilder] no TTY for confirmation; pass --yes to skip." >&2; exit 2; }
            case "${_ans}" in
                ""|y|Y|yes|YES|Yes) ;;
                *)
                    echo "[molbuilder] aborted by user." >&2
                    echo "[molbuilder] tip: set CONDA_EXE or MAMBA_EXE to pin the env manager:" >&2
                    echo "    CONDA_EXE=/path/to/conda bash $0 ${ORIGINAL_ARGS[*]:-bootstrap --yes}" >&2
                    exit 0
                    ;;
            esac
        fi
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

_resolve_env_python() {
    # Resolve the env's python binary.  Three strategies, in order:
    #
    #   1. Parse ``<mgr> env list`` -- the same registry call
    #      ``host_env_exists`` already uses.  Picks up the named env
    #      regardless of where conda stores it (custom ``envs_dirs``,
    #      mamba ``--prefix`` installs, etc.).
    #   2. Parse ``<mgr> info --json``'s ``envs`` array (FULL env
    #      paths, not the ``envs_dirs`` search list).  Catches cases
    #      where ``env list`` output got mangled by mamba 2.x's
    #      different table layout.
    #   3. Derive from ENV_MGR's own path.  Handles both the
    #      ``<root>/bin/conda`` layout (Miniforge, Miniconda) AND
    #      the ``<root>/condabin/conda`` layout (which is what
    #      ``conda init`` adds to PATH on most systems).
    #
    # ``<root>/condabin`` is the load-bearing detail that the prior
    # version missed: ``${ENV_MGR%/bin/*}`` only strips ``/bin/X``,
    # so a condabin/conda ENV_MGR fell through unchanged and the
    # fallback python path was junk.
    local _name="$1"
    local _prefix
    # Strategy 1: env list.  Last column is the prefix path; name is
    # column 1 (active env's ``*`` marker is column 2 -- awk's $1
    # still gets the name).
    _prefix="$("${ENV_MGR}" env list 2>/dev/null \
        | awk -v name="${_name}" 'NF && $1 == name {print $NF}' \
        | head -n 1)"
    if [[ -n "${_prefix}" && -x "${_prefix}/bin/python" ]]; then
        echo "${_prefix}/bin/python"
        return 0
    fi
    # Strategy 2: info --json -> envs array (full paths).  Match by
    # basename so a customised env layout still resolves.
    _prefix="$("${ENV_MGR}" info --json 2>/dev/null \
        | awk '/"envs":/{flag=1;next} flag && /]/{flag=0} flag' \
        | tr -d '",' \
        | awk -v name="${_name}" '{
            n = split($1, parts, "/");
            if (parts[n] == name) { print $1; exit }
        }')"
    if [[ -n "${_prefix}" && -x "${_prefix}/bin/python" ]]; then
        echo "${_prefix}/bin/python"
        return 0
    fi
    # Strategy 3: derive from ENV_MGR's path AND probe mamba's
    # default envs_dir (``$HOME/.conda/envs``).  Most users never
    # change ``envs_dirs`` in .condarc, but mamba's default differs
    # from conda's:
    #   * conda init  -> envs live at ``<conda root>/envs/<name>``
    #   * mamba init  -> envs live at ``~/.conda/envs/<name>``
    # If the user runs ``mamba create -n molbuilder`` after mamba
    # init, the env lands at ``~/.conda/envs/molbuilder`` even
    # though the env manager binary is at ``<conda root>/bin/mamba``.
    # Both layouts checked.  Strategy 1 (env list) catches both
    # already; this is just a fallback for the case where env list
    # output got mangled or the registry got out of sync.
    local _root="${ENV_MGR%/condabin/*}"
    if [[ "${_root}" == "${ENV_MGR}" ]]; then
        _root="${ENV_MGR%/bin/*}"
    fi
    local _candidate
    for _candidate in \
        "${_root}/envs/${_name}/bin/python" \
        "${HOME}/.conda/envs/${_name}/bin/python" \
    ; do
        if [[ -x "${_candidate}" ]]; then
            echo "${_candidate}"
            return 0
        fi
    done
    return 1
}

create_host_env() {
    resolve_host_env_channels
    echo "[molbuilder] creating host env '${HOST_ENV}' (~2 min)" >&2
    "${ENV_MGR}" create -n "${HOST_ENV}" "${HOST_ENV_CHANNEL_ARGS[@]}" \
        --yes "${HOST_CONDA_PACKAGES[@]}"
    if [[ ${#HOST_PIP_PACKAGES[@]} -gt 0 ]]; then
        # Call the env's python directly instead of ``mamba run``.
        # mamba 1.x's ``run`` implementation generates a shell stub
        # that uses ``exec -- "$@"`` which fails on shells where
        # ``exec`` doesn't accept ``--`` (most bash, dash); the user
        # sees ``line 5: exec: --: invalid option`` and the pip step
        # aborts.  Direct invocation bypasses the buggy stub and
        # writes to the env's own site-packages.  pip's user config
        # (~/.pip/pip.conf, ~/.config/pip/pip.conf) is respected
        # transparently -- internal PyPI mirrors / index-url
        # overrides apply unchanged.
        local _py
        if ! _py="$(_resolve_env_python "${HOST_ENV}")"; then
            echo "Error: cannot find python in env '${HOST_ENV}' after create." >&2
            echo "Probed three ways, none worked:" >&2
            echo "  1. ${ENV_MGR} env list  (match by name)" >&2
            echo "  2. ${ENV_MGR} info --json -> envs[]" >&2
            echo "  3. <ENV_MGR's install root>/envs/${HOST_ENV}/bin/python" >&2
            echo "Run '${ENV_MGR} env list' yourself to see what env paths" >&2
            echo "are registered; if molbuilder is there, file an issue." >&2
            exit 1
        fi
        echo "[molbuilder] pip-installing host-env extras: ${HOST_PIP_PACKAGES[*]}" >&2
        echo "[molbuilder] using ${_py}" >&2
        # ``|| true`` plus an explicit-rc check so a pip failure on
        # OPTIONAL extras doesn't kill the bootstrap.  Both
        # PeptideBuilder and pubchempy are UI-only conveniences;
        # the env is usable without them.
        if "${_py}" -m pip install "${HOST_PIP_PACKAGES[@]}"; then
            : # ok
        else
            echo "" >&2
            echo "[molbuilder] WARNING: pip install failed for the optional" >&2
            echo "[molbuilder] extras (${HOST_PIP_PACKAGES[*]}).  The host env" >&2
            echo "[molbuilder] is usable; you'll lose only the UI's peptide-" >&2
            echo "[molbuilder] builder + PubChem-lookup features until you" >&2
            echo "[molbuilder] install them manually:" >&2
            echo "[molbuilder]   ${_py} -m pip install ${HOST_PIP_PACKAGES[*]}" >&2
            echo "[molbuilder] continuing bootstrap..." >&2
        fi
    fi
    echo "[molbuilder] host env '${HOST_ENV}' ready" >&2
}

# ---- dispatch (1:1 with `molbuilder envs ...`) ---------------------------
#
# Bypasses ``mamba run`` and calls the host env's python directly.
# Why: mamba 1.x's ``run`` generates a shell stub that uses
# ``exec -- "$@"`` -- bash's exec builtin rejects ``--`` with
# ``line 5: exec: --: invalid option`` and the whole dispatch dies.
# Calling the env's python directly avoids the buggy stub and works
# the same way as a ``conda activate molbuilder && python ...``
# without needing activate.d hooks (which only set up shell PATH /
# LD_LIBRARY_PATH that this python invocation doesn't need).
#
# PYTHONPATH=$REPO_ROOT so ``python -m molbuilder`` finds the package
# regardless of the user's CWD (molbuilder is intentionally not
# pip-installed into the host env).

dispatch() {
    local _py
    if ! _py="$(_resolve_env_python "${HOST_ENV}")"; then
        echo "Error: cannot find python in env '${HOST_ENV}'." >&2
        echo "Probed three ways, none worked:" >&2
        echo "  1. ${ENV_MGR} env list  (match by name)" >&2
        echo "  2. ${ENV_MGR} info --json -> envs[]" >&2
        echo "  3. <ENV_MGR's install root>/envs/${HOST_ENV}/bin/python" >&2
        echo "Confirm the env is healthy with '${ENV_MGR} env list' --" >&2
        echo "if molbuilder is listed but the path's python is missing," >&2
        echo "remove + reinstall the env:" >&2
        echo "    ${ENV_MGR} env remove -n ${HOST_ENV} -y" >&2
        echo "    bash ${SCRIPT_DIR}/install-env.sh bootstrap --yes" >&2
        exit 1
    fi
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    MOLBUILDER_REPO_ROOT="${REPO_ROOT}" \
        "${_py}" -m molbuilder envs "$@"
}

# ---- main ----------------------------------------------------------------

# State machine, encoded in this dispatcher:
#
#                       ┌─────────────────────────────┐
#                       │   ENTRY (any subcommand)    │
#                       └────────────┬────────────────┘
#                                    │
#                          require_conda
#                          (probe → confirm unless --yes;
#                           fail → exit 2 with probe log)
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

# Scan args for --yes / -y so the env-manager confirm in require_conda
# (and any other prompt the shim owns) skips when CI / HPC batch is
# already passing the flag.  We don't strip it -- it forwards verbatim
# to the Python layer too.
for _a in "$@"; do
    case "${_a}" in
        --yes|-y) AUTO_YES=1; break ;;
    esac
done

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
