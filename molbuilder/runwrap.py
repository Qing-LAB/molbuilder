"""Shell-wrapper emission for ``molbuilder run``.

Each generated script (``.fdf`` or ``.py``) gets a sibling
``<basename>.run.sh`` that activates the right conda env and executes
the tool.  The user runs the ``.sh`` manually (foreground / background
/ cluster scheduler -- their call); molbuilder does **not** manage
processes.

The wrapper is intentionally small and human-readable:

* A user can read it to understand what command they're about to run.
* They can edit it to add custom flags (MPI options, env vars, ulimit).
* They can copy chunks into SLURM / PBS / GNU parallel scripts.

The wrapper is regenerated freshly each time ``molbuilder run`` runs
(it's per-invocation output, not state); edits between regenerations
are lost.

Testing hook: tests inject a synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities`.  Production call
sites pass only the script path + optional ``env`` / ``mpi_np``
overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .diagnostics import EXTENSION_TO_CATEGORY, get_capabilities


class WrapperError(Exception):
    """Wrapper cannot be generated -- unsupported extension, no env
    routing, missing script file, ..."""


def render_run_wrapper(script_path: Path, *,
                        env: Optional[str] = None,
                        mpi_np: Optional[int] = None,
                        omp_threads: Optional[int] = None,
                        max_memory_mb: Optional[int] = None) -> str:
    """Return the bash text for a wrapper running ``script_path``.

    Routing by file extension:

    * ``.fdf``  → SIESTA.  Uses ``mpirun -np <N>`` when ``mpi_np`` is
                  given and ≥ 2; redirects stdout to ``<basename>.out``
                  (the convention from job-layout v1).
    * ``.py``   → PySCF.  Runs ``python <script>``; the inlined
                  ``_MolwatchEmitter`` handles its own log files.

    Args:
      script_path: the ``.fdf`` or ``.py`` to wrap.
      env: override the routed env name for this invocation.  Default
        is whatever ``Capabilities.env_for_category(<category>)`` returns.
      mpi_np: SIESTA MPI rank count.  Ignored for ``.py`` scripts.
    """
    script_path = Path(script_path)
    suffix = script_path.suffix.lower()
    category = EXTENSION_TO_CATEGORY.get(suffix)
    if category is None:
        raise WrapperError(
            f"`{script_path.name}`: unsupported script extension "
            f"`{suffix}`.  Supported: "
            f"{', '.join(sorted(EXTENSION_TO_CATEGORY))}."
        )

    caps = get_capabilities()
    target_env = env if env is not None else caps.env_for_category(category)
    if target_env is None:
        raise WrapperError(
            f"category `{category}`: no env name registered.  Pass "
            f"env=... explicitly or add a default to "
            f"molbuilder.diagnostics.DEFAULT_ENV_NAMES."
        )

    basename = script_path.stem
    script_name = script_path.name

    # Pre-command env exports.  SIESTA: shared anti-oversubscription
    # recipe with PySCF / spectra (see molbuilder/runtime_info.py).
    # OMP threads per rank = user-set ``omp_threads``, or auto-detect
    # physical cores // mpi_np (so a 20-core host with 4 MPI ranks
    # gets 5 OMP threads per rank by default -- a sane hybrid).
    # BLAS always pinned to 1 so OMP * BLAS doesn't multiply.
    env_prefix = ""
    if category == "siesta":
        # Resolve MPI rank count.  SIESTA is fundamentally an MPI
        # code; even single-host execution is launched via mpirun.
        # When the user leaves mpi_np blank we default to ALL physical
        # cores -- that matches user expectation ("the wrapper should
        # use MPI") instead of silently emitting a bare ``siesta``
        # invocation that ignores all but one core.
        from .runtime_info import physical_core_count
        phys = physical_core_count()
        if mpi_np is None or int(mpi_np) < 1:
            resolved_mpi = max(1, phys)
            mpi_source   = f"auto: physical_cores ({phys})"
        else:
            resolved_mpi = int(mpi_np)
            mpi_source   = "user-set"

        # OMP threads.  SIESTA mainline is mostly NOT OMP-aware;
        # pure MPI + OMP=1 is the standard SIESTA recipe.  User can
        # explicitly request hybrid by setting omp_threads > 1 (only
        # meaningful with an OMP-compiled SIESTA build).
        if omp_threads is None:
            resolved_omp = 1
            omp_source   = "default; SIESTA isn't reliably OMP-aware"
        else:
            resolved_omp = max(1, int(omp_threads))
            omp_source   = "user-set"

        # NOTE: the actual exec line is computed at RUN time by the
        # probe block below, NOT here -- the wrapper picks ``mpirun
        # -np N siesta`` vs bare ``siesta`` based on what ``siesta
        # --version`` reports for the currently-installed binary.
        # ``inner`` becomes the run-time-resolved shell expression
        # below (post-probe); we still derive a static ``description``
        # for the file-header comment.
        inner = f"$_launch_cmd {script_name} > {basename}.out"
        description = f"SIESTA run, {resolved_mpi} ranks intended"

        env_prefix = (
            f"# Thread / BLAS pinning.\n"
            f"#   * OMP_NUM_THREADS ({omp_source}): SIESTA mainline is\n"
            f"#     mostly not OMP-aware, so pure MPI with OMP=1 is the\n"
            f"#     standard recipe.  Bump only with an OMP-compiled\n"
            f"#     SIESTA build (hybrid MPI+OMP).\n"
            f"#   * BLAS pinned to 1 per rank so OMP * BLAS doesn't\n"
            f"#     oversubscribe.\n"
            f"export OMP_NUM_THREADS={resolved_omp}\n"
            f"export MKL_NUM_THREADS=1\n"
            f"export OPENBLAS_NUM_THREADS=1\n"
        )
        if max_memory_mb is not None and int(max_memory_mb) > 0:
            kb = int(max_memory_mb) * 1024
            env_prefix += (
                f"# Memory cap (cfg.max_memory_mb): {max_memory_mb} MB\n"
                f"ulimit -v {kb} || true  # soft cap; ignored if shell can't set it\n"
            )
        env_prefix += "\n"

        # Runtime SIESTA build probe + launcher selection.
        #
        # ``siesta --version`` (5.x +) self-reports the parallelisation
        # the binary was compiled with.  Example for a typical conda-
        # forge build:
        #
        #   Version         : 5.4.2
        #   Parallelisations: MPI
        #
        # We parse the ``Parallelisations:`` line and pick the launcher
        # accordingly:
        #
        #   MPI present      ->  mpirun -np <N> siesta   (always)
        #   OMP present      ->  bare siesta             (OMP env vars take effect)
        #   both             ->  mpirun -np <N> siesta   (hybrid)
        #   probe failed     ->  mpirun -np <N> siesta   (safe default for
        #                                                 MPI-compiled binaries)
        #   serial build     ->  bare siesta
        #
        # The probe runs ONCE per wrapper invocation and prints what
        # it found before exec, so the user sees the actual build
        # capability + the launcher choice in the log.  This adapts
        # automatically if you rebuild SIESTA with different flags --
        # no need to regenerate the wrapper.
        env_prefix += (
            f"# --- Probe SIESTA build at runtime ---\n"
            f'_siesta_bin_path="$(command -v siesta || echo \"\")"\n'
            f'if [ -z "$_siesta_bin_path" ]; then\n'
            f"    echo \"ERROR: 'siesta' not on PATH after activating "
            f"'{target_env}'.  Is SIESTA installed in this env?\" >&2\n"
            f"    exit 1\n"
            f"fi\n"
            f'_siesta_version_out="$(siesta --version 2>/dev/null || true)"\n'
            f'_siesta_ver="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Version/ {{print $2; exit}}')\"\n"
            f'_siesta_par="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Parallelisations/ {{print $2; exit}}')\"\n"
            f"# Decide launcher from probe.  Default to mpirun (safe\n"
            f"# for any MPI-compiled binary) when the probe can't\n"
            f"# tell us anything.\n"
            f'_has_mpi=0; _has_omp=0\n'
            f'case " $_siesta_par " in *MPI*) _has_mpi=1 ;; esac\n'
            f'case " $_siesta_par " in *OMP*|*OpenMP*) _has_omp=1 ;; esac\n'
            f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="mpirun -np {resolved_mpi} siesta"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ({resolved_mpi} ranks x {resolved_omp} OMP threads)"\n'
            f'    else\n'
            f'        _launch_note="pure MPI ({resolved_mpi} ranks; OMP setting irrelevant to this binary)"\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="OMP-only build ({resolved_omp} threads)"\n'
            f'elif [ -z "$_siesta_par" ]; then\n'
            f'    _launch_cmd="mpirun -np {resolved_mpi} siesta"\n'
            f'    _launch_note="MPI fallback (probe inconclusive; safe default for MPI-compiled SIESTA)"\n'
            f'else\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="serial build (no parallelisation compiled in)"\n'
            f"fi\n"
            f"\n"
        )

        # Human-readable banner printed at run time so the user sees
        # the rank count / threading / cwd / command + BUILD probe
        # results before SIESTA spends 30 seconds reading the .fdf.
        env_prefix += (
            f'echo "===== molbuilder SIESTA run-wrapper ====="\n'
            f'echo "  Date          : $(date -Iseconds)"\n'
            f'echo "  Host          : $(hostname)"\n'
            f'echo "  Cwd           : $(pwd)"\n'
            f'echo "  Conda env     : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'echo "  SIESTA binary : $_siesta_bin_path"\n'
            f'echo "  SIESTA version: ${{_siesta_ver:-unknown}}"\n'
            f'echo "  Build paral.  : ${{_siesta_par:-unknown}}"\n'
            f'echo "  Launch mode   : $_launch_note"\n'
            f'echo "  Threading     : OMP_NUM_THREADS={resolved_omp}, '
            f'OPENBLAS=1, MKL=1"\n'
            f'echo "  Command       : $_launch_cmd {script_name} > {basename}.out"\n'
            f'echo "  Stdout        : {basename}.out (live; tail -f to follow)"\n'
            f'echo "========================================="\n'
            f"\n"
        )
    else:                                          # pyscf
        inner = f"python {script_name}"
        description = "PySCF run"
        # PySCF: the inline ``runtime_info`` block in the emitted
        # .py sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 1 via
        # ``os.environ.setdefault`` BEFORE numpy import.  We don't
        # set them in the wrapper too -- doing so would override
        # the env-respect (the script honors a pre-export) AND
        # mask the in-script auto-detect that picks physical cores.

        # Same human-readable banner pattern as SIESTA -- the script
        # itself logs its own runtime info but the wrapper covers
        # the "did it even start" window before Python imports.
        env_prefix = (
            f'echo "===== molbuilder PySCF run-wrapper ====="\n'
            f'echo "  Date    : $(date -Iseconds)"\n'
            f'echo "  Host    : $(hostname)"\n'
            f'echo "  Cwd     : $(pwd)"\n'
            f'echo "  Conda   : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'echo "  Command : python {script_name}"\n'
            f'echo "  Logs    : see <basename>.molwatch.log (script writes its own)"\n'
            f'echo "========================================"\n'
            f"\n"
        )

    # Conda env activation block.  Three paths so the wrapper Just
    # Works in the common cases:
    #
    # 1. Already in the right env (CONDA_DEFAULT_ENV == target_env):
    #    skip activation, run directly.  Lets the user activate
    #    interactively + invoke the wrapper without double-init.
    # 2. conda on PATH: source the conda.sh hook + activate.  Full
    #    env-setup (PATH, LD_LIBRARY_PATH, env-specific hooks like
    #    CUDA bootstraps) -- more robust than `conda run` for MPI
    #    launchers that can mishandle the `--no-capture-output`
    #    pipe redirection.
    # 3. conda not on PATH: print a clear error message naming the
    #    target env + how to install conda; exit 1.
    #
    # This is the "hybrid" pattern: catches the common cases, gives
    # a real error message instead of a cryptic "command not found".
    env_activation = (
        f"# --- Activate conda env ({target_env}) ----------------------\n"
        f'if [ "${{CONDA_DEFAULT_ENV:-}}" = "{target_env}" ]; then\n'
        f"    : # already in the target env -- nothing to do\n"
        f"elif command -v conda >/dev/null 2>&1; then\n"
        f'    _conda_base="$(conda info --base 2>/dev/null)"\n'
        f'    if [ -z "$_conda_base" ] || [ ! -f "$_conda_base/etc/profile.d/conda.sh" ]; then\n'
        f"        echo \"ERROR: conda is on PATH but conda.sh not found; \"\\\n"
        f"             \"reinstall conda or set CONDA_PREFIX manually.\" >&2\n"
        f"        exit 1\n"
        f"    fi\n"
        f'    # shellcheck disable=SC1091\n'
        f'    source "$_conda_base/etc/profile.d/conda.sh"\n'
        f"    conda activate {target_env}\n"
        f"else\n"
        f"    echo \"ERROR: conda not on PATH; this wrapper needs the \"\\\n"
        f"         \"'{target_env}' env activated.  Either:\" >&2\n"
        f"    echo \"  * install Miniconda + create the env: \"\\\n"
        f"         \"see docs/README_install.md\" >&2\n"
        f"    echo \"  * or pre-activate it: \"\\\n"
        f"         \"conda activate {target_env} && bash $0\" >&2\n"
        f"    exit 1\n"
        f"fi\n"
        f"\n"
    )

    return (
        f"#!/usr/bin/env bash\n"
        f"#\n"
        f"# molbuilder run-wrapper -- {description}\n"
        f"# Script: {script_name}\n"
        f"# Target env: {target_env}\n"
        f"#\n"
        f"# Generated by `molbuilder run`.  Edit freely; molbuilder will\n"
        f"# not regenerate this file unless `molbuilder run` is invoked\n"
        f"# again on the same script.  Run directly:\n"
        f"#\n"
        f"#     bash {basename}.run.sh        # foreground\n"
        f"#     nohup ./{basename}.run.sh &   # background, detached\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"cd \"$(dirname \"$0\")\"\n"
        f"\n"
        f"{env_activation}"
        f"{env_prefix}"
        f"exec {inner}\n"
    )


def write_run_wrapper(script_path: Path, *,
                       env: Optional[str] = None,
                       mpi_np: Optional[int] = None,
                       omp_threads: Optional[int] = None,
                       max_memory_mb: Optional[int] = None) -> Path:
    """Render + write ``<basename>.run.sh`` next to ``script_path``.

    Returns the wrapper's path.  Sets executable bit (0o755) so the
    user can ``./my-job.run.sh`` directly.  Overwrites any existing
    wrapper.
    """
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise WrapperError(f"script not found: {script_path}")
    text = render_run_wrapper(
        script_path,
        env=env, mpi_np=mpi_np,
        omp_threads=omp_threads,
        max_memory_mb=max_memory_mb,
    )
    # Use stem + ".run.sh" rather than ``.with_suffix(".run.sh")``: the
    # latter REPLACES only the last suffix, so ``job.spectra.py`` would
    # become ``job.run.sh`` and lose the "spectra" tag.  We want
    # ``job.spectra.run.sh``.
    wrapper_path = script_path.parent / (script_path.stem + ".run.sh")
    wrapper_path.write_text(text)
    wrapper_path.chmod(0o755)
    return wrapper_path


__all__ = [
    "WrapperError",
    "render_run_wrapper",
    "write_run_wrapper",
]
