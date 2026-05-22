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
                        mpi_np: Optional[int] = None) -> str:
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

    # Pre-command env exports.  Empty by default; SIESTA-MPI adds
    # BLAS/OpenMP thread pinning so each MPI rank doesn't spawn its
    # own pool of threads on top of the rank count (M×N
    # oversubscription on a host with M cores and N MPI ranks).
    env_prefix = ""
    if category == "siesta":
        if mpi_np is not None and mpi_np >= 2:
            inner = (f"mpirun -np {mpi_np} siesta {script_name} "
                      f"> {basename}.out")
            description = f"SIESTA-MPI run on {mpi_np} ranks"
            # Pin BLAS / OpenMP thread count to 1 per process so the
            # only parallelism is MPI.  SIESTA links MKL (via
            # mkl-spblas) or OpenBLAS depending on the build; both
            # honour their own NUM_THREADS env var, OMP_NUM_THREADS
            # is the OpenMP fallback.  Without these exports each of
            # the N MPI ranks would spawn N more threads (default =
            # cpu count), so an 8-rank job on a 32-core host
            # produces 8x32=256 threads competing for 32 cores --
            # classic "SIESTA hits 100% CPU on every core but runs
            # slower than 1-rank" pathology.  Users who DO want
            # hybrid MPI+OpenMP (rare) can edit the wrapper after
            # generation.
            env_prefix = (
                "# Thread pinning: one BLAS/OpenMP thread per MPI rank.\n"
                "# Drop these exports if you want hybrid MPI + OpenMP.\n"
                "export OMP_NUM_THREADS=1\n"
                "export MKL_NUM_THREADS=1\n"
                "export OPENBLAS_NUM_THREADS=1\n"
                "\n"
            )
        else:
            inner = f"siesta {script_name} > {basename}.out"
            description = "SIESTA (single-process)"
            # Single-process: do NOT pin threads.  Without MPI the
            # user wants BLAS / OpenMP threading -- it's the ONLY
            # parallelism available.
    else:                                          # pyscf
        inner = f"python {script_name}"
        description = "PySCF run"
        # PySCF uses BLAS threading deliberately; no pinning.

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
                       mpi_np: Optional[int] = None) -> Path:
    """Render + write ``<basename>.run.sh`` next to ``script_path``.

    Returns the wrapper's path.  Sets executable bit (0o755) so the
    user can ``./my-job.run.sh`` directly.  Overwrites any existing
    wrapper.
    """
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise WrapperError(f"script not found: {script_path}")
    text = render_run_wrapper(script_path, env=env, mpi_np=mpi_np)
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
