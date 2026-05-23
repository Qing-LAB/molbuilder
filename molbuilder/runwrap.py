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
        # Resolve OMP thread count.  None -> auto: physical_cores //
        # mpi_np (MPI case) or physical_cores (single-process).  An
        # explicit value wins.
        from .runtime_info import physical_core_count
        phys = physical_core_count()
        if omp_threads is None:
            mpi_n = max(1, mpi_np or 1)
            resolved_omp = max(1, phys // mpi_n)
            omp_source   = f"auto: physical_cores ({phys}) // mpi_np ({mpi_n})"
        else:
            resolved_omp = int(omp_threads)
            omp_source   = "user-set"

        if mpi_np is not None and mpi_np >= 2:
            inner = (f"mpirun -np {mpi_np} siesta {script_name} "
                      f"> {basename}.out")
            description = f"SIESTA-MPI run on {mpi_np} ranks"
        else:
            inner = f"siesta {script_name} > {basename}.out"
            description = "SIESTA (single-process)"

        # Anti-oversubscription guards: BLAS=1, OMP=resolved.  Same
        # recipe as PySCF / spectra emit inline -- canonical recipe.
        env_prefix = (
            f"# Thread / BLAS pinning ({omp_source}).  BLAS pinned to 1\n"
            f"# per rank so OMP * BLAS doesn't oversubscribe (canonical\n"
            f"# recipe shared with /spectra + Build PySCF, see\n"
            f"# molbuilder/runtime_info.py).  Edit freely.\n"
            f"export OMP_NUM_THREADS={resolved_omp}\n"
            f"export MKL_NUM_THREADS=1\n"
            f"export OPENBLAS_NUM_THREADS=1\n"
        )
        if max_memory_mb is not None and int(max_memory_mb) > 0:
            # Many BLAS / MPI libs honour an explicit memory cap via
            # env vars (OMP_STACKSIZE etc.).  We record the cap in
            # the wrapper as a comment + emit ulimit -v as a soft
            # cap so a runaway process doesn't OOM the host.
            kb = int(max_memory_mb) * 1024
            env_prefix += (
                f"# Memory cap (cfg.max_memory_mb): {max_memory_mb} MB\n"
                f"ulimit -v {kb} || true  # soft cap; ignored if shell can't set it\n"
            )
        env_prefix += "\n"
    else:                                          # pyscf
        inner = f"python {script_name}"
        description = "PySCF run"
        # PySCF: the inline ``runtime_info`` block in the emitted
        # .py sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 1 via
        # ``os.environ.setdefault`` BEFORE numpy import.  We don't
        # set them in the wrapper too -- doing so would override
        # the env-respect (the script honors a pre-export) AND
        # mask the in-script auto-detect that picks physical cores.

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
        f"{env_prefix}"
        f"exec conda run -n {target_env} --no-capture-output \\\n"
        f"    {inner}\n"
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
