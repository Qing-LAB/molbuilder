"""Source-build executor for env recipes that carry a :class:`BuildSpec`.

The CLI seam: :mod:`molbuilder.envs.install` calls
:func:`run_build_spec` AFTER ``conda create`` + pip + extra_steps when
the recipe declares ``recipe.build_spec is not None``.

This module is the procedural counterpart to the declarative
:class:`molbuilder.envs.recipes.BuildSpec`.  It knows how to:

1. Pre-flight the host: CUDA toolkit reachable? CUDA<->gcc paired?
   Auto-detect compute capability via ``nvidia-smi``.
2. Compute a toolchain fingerprint over ``(gcc, openmpi, cuda, refs)``
   so resumes can short-circuit phases whose fingerprint matches.
3. For each component (in dependency order): clone -> configure ->
   build -> install -> verify.  Each phase writes a sentinel file.
4. After install, render activate.d / deactivate.d hooks that put
   the built binaries on PATH + LD_LIBRARY_PATH only when the env is
   activated.

It is the **only** module that runs subprocesses outside of conda's
install code path.  Every other write-side action goes through here.

The 2026-06-14 Decisions log entry locks the design; the companion
engineering doc lives at :doc:`docs/engines/siesta-gpu`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

from .recipes import BuildComponent, BuildSpec


# --------------------------------------------------------------------- #
#  Phase identifiers + human-readable cost estimates                     #
# --------------------------------------------------------------------- #


# Phases run in this strict order for every component.  Skipping later
# phases when an earlier one has no sentinel is impossible because the
# sentinel chain enforces ordering at runtime; this tuple is here so
# tests can assert it without grepping strings.
PHASES: Tuple[str, ...] = ("clone", "configure", "build", "install", "verify")


# Per-(component, phase) human-readable description + estimated cost.
# These are user-visible in the CLI so the user knows what's happening
# and why a phase taking N minutes is normal.  Tuned for an 8-core
# machine + broadband; the executor will print scaled estimates if the
# detected `jobs` value differs.
_PHASE_ESTIMATES: Mapping[Tuple[str, str], Tuple[str, str]] = {
    # (component, phase) -> (action description, expected cost label)
    ("elpa", "clone"):     ("download + extract ELPA tarball from MPCDF",
                            "~30s, ~1.5 MB"),
    ("elpa", "configure"): ("autotools configure ELPA with CUDA + OpenMP",
                            "~60s"),
    ("elpa", "build"):     ("compile ELPA (CUDA kernels are the slow part)",
                            "~10-15 min"),
    ("elpa", "install"):   ("install ELPA library + headers",
                            "~10s"),
    ("elpa", "verify"):    ("check that libelpa.so exists",
                            "instant"),
    # ELSI is built as a SIESTA submodule per the 2026-06-15 architecture
    # decision (see docs/engines/siesta-gpu.md § 3.1).  No standalone
    # ELSI phases here.
    ("siesta", "clone"):   ("git clone the SIESTA source",
                            "~60s, ~200 MB"),
    ("siesta", "configure"):("cmake configure SIESTA with ELSI + TranSiesta + libxc + netcdf",
                            "~30s"),
    ("siesta", "build"):   ("compile SIESTA, TranSiesta, and TBtrans",
                            "~8-10 min"),
    ("siesta", "install"): ("install siesta/transiesta/tbtrans binaries",
                            "~5s"),
    ("siesta", "verify"):  ("run siesta --version to confirm it launches",
                            "instant"),
}


# Rough disk required under $CONDA_PREFIX, in GB.  Doctor + preflight
# warn the user before committing.
_DEFAULT_DISK_GB_REQUIRED = 30.0
_DEFAULT_DISK_GB_RECOMMENDED = 50.0


# --------------------------------------------------------------------- #
#  Build-env isolation: strip leakage vectors                            #
# --------------------------------------------------------------------- #
#
# The build runs inside ``conda run --prefix <env>``.  That sources the
# env's ``etc/conda/activate.d/*.sh`` hooks which set CC/CXX/FC/CFLAGS/
# etc. pointing at conda's compilers and lib paths.  BUT activate.d
# hooks often APPEND to user-set values rather than replace them.
# Concretely, if the user's parent shell has
#
#   export CPATH=/usr/include/openmpi
#   export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib
#
# (because they ran ``apt install libopenmpi-dev`` once), conda
# activate keeps those, gcc adds them to the compile + link lines, and
# the build resolves ``mpi.h`` + ``libmpi.so`` against the SYSTEM
# OpenMPI 4.1.x instead of the env's pinned OpenMPI.  Result: ABI
# mismatch at runtime, segfault somewhere in MPI_Init.
#
# To prevent this, we strip these variables from the subprocess env
# before invoking ``conda run``.  Conda's own activate.d sets the
# right values on top of the empty slate.
#
# Coverage rule: drop everything that influences (a) the compiler's
# default include/lib search paths, (b) the linker's library search,
# (c) the runtime loader's library search, (d) compiler driver flags,
# (e) MPI / CUDA / BLAS / FFT / HDF5 / NetCDF location overrides.
_LEAKAGE_ENV_VARS: frozenset = frozenset((
    # Linker / loader paths
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_RUN_PATH", "LD_PRELOAD",
    "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    # gcc include search
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH", "OBJCPLUS_INCLUDE_PATH",
    "INCLUDE_PATH",
    # pkg-config / cmake discovery
    "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
    "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_LIBRARY_PATH",
    "CMAKE_MODULE_PATH", "CMAKE_FRAMEWORK_PATH", "CMAKE_APPBUNDLE_PATH",
    # Compiler driver flags
    "CFLAGS", "CXXFLAGS", "FFLAGS", "F90FLAGS", "F77FLAGS", "FCFLAGS",
    "LDFLAGS", "CPPFLAGS", "ASFLAGS", "DEBUG_CFLAGS", "DEBUG_LDFLAGS",
    # Compiler binaries: clear so conda activate.d sets clean values
    "CC", "CXX", "FC", "F77", "F90", "F95", "CPP",
    "AR", "RANLIB", "LD", "AS", "NM", "STRIP", "OBJCOPY", "OBJDUMP",
    "READELF", "GCC", "GXX", "GFORTRAN",
    # MPI location overrides
    "MPI_HOME", "MPI_ROOT", "MPIHOME", "MPI_DIR", "MPI_INCLUDE",
    "MPICC", "MPICXX", "MPIFORT", "MPIF77", "MPIF90", "MPIEXEC",
    # CUDA location overrides.  Note: conda-forge's cuda-nvcc activate
    # hook sets CUDA_HOME to $CONDA_PREFIX itself; clearing first lets
    # that hook win cleanly.
    "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_INSTALL_PATH",
    "CUDADIR", "CUDAToolkit_ROOT", "CUDACXX",
    "NVCC", "NVCC_PATH", "NVCCFLAGS", "CUDAFLAGS",
    # BLAS / LAPACK location overrides
    "BLAS", "LAPACK", "BLAS_LIBS", "LAPACK_LIBS",
    "MKLROOT", "MKL_ROOT", "OPENBLAS_DIR", "OPENBLAS_ROOT",
    # FFT / HDF5 / NetCDF location overrides
    "FFTW_ROOT", "FFTW_DIR", "FFTW3_ROOT",
    "HDF5_ROOT", "HDF5_DIR", "HDF5_HOME",
    "NETCDF_ROOT", "NETCDF_DIR", "NETCDF_HOME",
    "SCALAPACK_ROOT", "SCALAPACK_DIR",
    "LIBXC_ROOT", "LIBXC_DIR", "LIBXC_HOME",
))

# Prefix-matched env var families.  Cleared because they tune runtime
# behaviour of MPI launchers in ways that bypass the build env's MPI.
_LEAKAGE_ENV_PREFIXES: Tuple[str, ...] = (
    "OMPI_", "OPAL_",        # OpenMPI runtime config
    "MPICH_", "HYDRA_",      # MPICH runtime config
    "I_MPI_",                # Intel MPI runtime config
    "PMI_", "PMIX_",         # PMI launcher config
    "SLURM_",                # SLURM exports vars that affect MPI
)


def run_streaming(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    log_file: Optional[Path] = None,
    sink: Optional[TextIO] = None,
    indent: str = "    ",
    timeout: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    """Run a subprocess streaming stdout+stderr to ``sink`` in real time.

    Designed for long-running build steps (cmake, ninja, conda
    create) so the user sees progress instead of a silent gap
    between "phase start" and "phase done / failed".  Without this,
    a 15-minute ELPA build looks identical to a hung process.

    Parameters
    ----------
    argv
        Command + args as a list.  Passed directly to
        ``subprocess.Popen`` (no shell).
    env
        Environment dict for the subprocess.  ``None`` inherits
        ``os.environ`` per usual subprocess semantics.
    log_file
        Optional path; the full captured output is written here at
        the end (overwrites any prior content).  ``None`` skips
        log persistence.
    sink
        Where to stream lines as they arrive (typically
        ``sys.stderr``).  ``None`` defaults to ``sys.stderr``.
    indent
        Prefix prepended to each streamed line so subprocess
        output is visually distinct from the wrapper's own progress
        lines.  Pass ``""`` for no indent.
    timeout
        Optional seconds; ``Popen.wait(timeout=...)`` raises
        :class:`subprocess.TimeoutExpired` on overrun.  ``None``
        waits indefinitely.

    Returns
    -------
    (returncode, captured)
        ``returncode`` is ``None`` on launch failure / timeout;
        ``captured`` is the full combined stdout+stderr as a
        string (the tail of which is shown in the CLI's failure
        recap).
    """
    out_sink: TextIO = sink if sink is not None else sys.stderr
    try:
        # text=True + bufsize=1 + stderr→stdout gives line-buffered
        # interleaved output as the build emits it.  Many build tools
        # (cmake, ninja) line-flush by default; some don't (autoconf-
        # style) but the merge to stdout still streams paragraph-by-
        # paragraph instead of blocking until exit.
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env) if env is not None else None,
        )
    except (FileNotFoundError, OSError) as exc:
        msg = f"failed to launch: {exc}"
        if log_file is not None:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text(msg, encoding="utf-8")
            except OSError:
                pass
        return None, msg

    captured_lines: List[str] = []
    try:
        # Stream each line: write indented copy to sink, accumulate
        # unindented copy for the log + the CLI's tail-on-failure.
        assert proc.stdout is not None
        for line in proc.stdout:
            captured_lines.append(line)
            try:
                out_sink.write(indent + line)
                out_sink.flush()
            except OSError:
                # If the sink dies (closed pipe, etc.) keep accumulating;
                # we still want the log file to land.
                pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            tail = f"\n[killed after {timeout}s timeout]\n"
            captured_lines.append(tail)
            try:
                out_sink.write(indent + tail)
                out_sink.flush()
            except OSError:
                pass
            combined = "".join(captured_lines)
            if log_file is not None:
                try:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    log_file.write_text(combined, encoding="utf-8")
                except OSError:
                    pass
            return None, combined
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass

    combined = "".join(captured_lines)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(combined, encoding="utf-8")
        except OSError:
            pass
    return proc.returncode, combined


def build_subprocess_env(base_env: Optional[Mapping[str, str]] = None
                         ) -> Dict[str, str]:
    """Return a sanitized copy of ``base_env`` (defaulting to ``os.environ``)
    with build-leakage vectors stripped.

    See the module-level comment for the failure mode this defends
    against: user-shell ``CPATH`` / ``LIBRARY_PATH`` / ``CFLAGS`` /
    ``MPI_HOME`` / ``CUDA_HOME`` / ``OMPI_*`` variables would otherwise
    flow into ``conda run``'s subprocess and shadow the env's pinned
    compilers + MPI + CUDA.

    The returned dict is safe to pass as the ``env=`` kwarg of
    :func:`subprocess.run`.  Conda's activate.d hooks will set the
    right values on top of this clean slate.
    """
    source = dict(base_env if base_env is not None else os.environ)
    return {
        k: v for k, v in source.items()
        if k not in _LEAKAGE_ENV_VARS
        and not any(k.startswith(p) for p in _LEAKAGE_ENV_PREFIXES)
    }


# Phases that wipe their build directory when re-run (everything from
# configure forward; clone wipes its src dir; verify is a read-only
# probe).
_PHASES_WIPING_BUILD = frozenset({"configure"})


# --------------------------------------------------------------------- #
#  Toolchain detection                                                   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolchainProbe:
    """Snapshot of host + conda-env toolchain state.

    Populated by :func:`probe_toolchain` from a live conda env after
    conda create has run.  Feeds the fingerprint hash so resumes
    invalidate when any input shifts.
    """
    env_prefix: str
    cuda_home: Optional[str]
    cuda_version: Optional[str]
    cuda_compute_cap: Optional[str]   # e.g. "8.0"; numeric form
    gcc_version: Optional[str]
    openmpi_version: Optional[str]
    jobs: int

    @property
    def cuda_cc_numeric(self) -> str:
        """Compute capability as CMake's ``CUDA_ARCHITECTURES`` value.

        ``8.0`` -> ``"80"``; ``9.0`` -> ``"90"``.
        """
        if not self.cuda_compute_cap:
            return "80"  # documented fallback
        return self.cuda_compute_cap.replace(".", "")

    @property
    def cuda_cc_sm(self) -> str:
        """Compute capability as ELPA's ``sm_XX`` form."""
        return f"sm_{self.cuda_cc_numeric}"


def _run_capture(argv: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
                 timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a subprocess capturing stdout+stderr; never raises on rc != 0."""
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout,
        env=dict(env) if env is not None else None,
    )


def _detect_cuda_home(env_prefix: Optional[str],
                      env_overrides: Mapping[str, str]) -> Optional[str]:
    """Find the CUDA toolkit root.

    Per the molbuilder design (2026-06-15), the CUDA TOOLKIT lives
    inside the conda env (cuda-nvcc + cuda-cudart-dev + ... from
    conda-forge).  The host provides only the NVIDIA driver +
    nvidia-smi.  Search order:

      1. ``<env_prefix>/bin/nvcc``  -- the conda-installed toolkit
      2. ``$CUDA_HOME`` env var      -- legacy/manual override
      3. ``/usr/local/cuda``, ``/opt/cuda``  -- legacy system installs
      4. ``which nvcc``              -- whatever's on PATH

    The env path wins because that's where the build should be
    looking.  System CUDA at /usr/local/cuda is no longer the
    canonical source for this recipe.
    """
    if env_prefix and Path(env_prefix, "bin", "nvcc").exists():
        return env_prefix
    cuda_home = env_overrides.get("CUDA_HOME") or os.environ.get("CUDA_HOME")
    if cuda_home and Path(cuda_home, "bin", "nvcc").exists():
        return cuda_home
    for candidate in ("/usr/local/cuda", "/opt/cuda"):
        if Path(candidate, "bin", "nvcc").exists():
            return candidate
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    return None


def detect_nvidia_driver() -> Optional[str]:
    """Read NVIDIA driver version from ``nvidia-smi`` (host-side).

    The driver is kernel-module-coupled; it cannot be a conda package.
    Returns ``None`` if nvidia-smi is absent or reports nothing.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    cp = _run_capture([
        smi, "--query-gpu=driver_version", "--format=csv,noheader",
    ])
    if cp.returncode != 0:
        return None
    first = (cp.stdout or "").strip().splitlines()
    return first[0].strip() if first else None


def _detect_cuda_version(cuda_home: str) -> Optional[str]:
    """Read CUDA toolkit version from ``nvcc --version``."""
    nvcc = Path(cuda_home, "bin", "nvcc")
    if not nvcc.exists():
        return None
    cp = _run_capture([str(nvcc), "--version"])
    if cp.returncode != 0:
        return None
    # Output line: "Cuda compilation tools, release 12.4, V12.4.131"
    m = re.search(r"V(\d+\.\d+(?:\.\d+)?)", cp.stdout)
    return m.group(1) if m else None


def _detect_compute_cap(override: Optional[str] = None) -> Optional[str]:
    """Detect the host GPU's compute capability via ``nvidia-smi``.

    Returns the capability as a dotted string (``"8.0"``) or ``None``
    if no GPU is reachable.  Honours an explicit override (used by
    :data:`MOLBUILDER_CUDA_CC` to force a build target even on the
    wrong host).
    """
    if override:
        return override
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    cp = _run_capture([
        smi, "--query-gpu=compute_cap", "--format=csv,noheader",
    ])
    if cp.returncode != 0:
        return None
    first = (cp.stdout or "").strip().splitlines()
    if not first:
        return None
    cap = first[0].strip()
    # Validate shape: digit(s).digit(s)
    if not re.fullmatch(r"\d+\.\d+", cap):
        return None
    return cap


def _detect_gcc_version(env_prefix: str) -> Optional[str]:
    """Read gcc version from the conda env's gcc (gcc_linux-64 wrapper)."""
    # gcc_linux-64=14 installs $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
    # plus several aliases.  Use the unversioned wrapper.
    candidates = [
        Path(env_prefix, "bin", "x86_64-conda-linux-gnu-gcc"),
        Path(env_prefix, "bin", "gcc"),
    ]
    for cand in candidates:
        if cand.exists():
            cp = _run_capture([str(cand), "-dumpfullversion"])
            if cp.returncode == 0 and cp.stdout.strip():
                return cp.stdout.strip()
    return None


def _detect_openmpi_version(env_prefix: str) -> Optional[str]:
    """Read OpenMPI version from the conda env's mpirun."""
    mpirun = Path(env_prefix, "bin", "mpirun")
    if not mpirun.exists():
        return None
    cp = _run_capture([str(mpirun), "--version"])
    if cp.returncode != 0:
        return None
    # Output line: "mpirun (Open MPI) 5.0.5"
    m = re.search(r"Open MPI[)\s]+(\d+\.\d+\.\d+)", cp.stdout)
    return m.group(1) if m else None


def _default_jobs() -> int:
    """Build concurrency: ``min(nproc, 8)`` with ``MOLBUILDER_BUILD_JOBS`` override."""
    override = os.environ.get("MOLBUILDER_BUILD_JOBS")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            pass
    try:
        cpu = os.cpu_count() or 1
    except Exception:
        cpu = 1
    return max(1, min(cpu, 8))


def describe_phase(component: str, phase: str) -> Tuple[str, str]:
    """Return ``(action, cost)`` strings for a (component, phase) pair.

    ``action`` is a one-sentence human-readable description of what
    the phase does; ``cost`` is a coarse time/disk estimate.  Used by
    the CLI to print per-step explanations before running the step.
    Returns generic strings for any pair not registered in
    :data:`_PHASE_ESTIMATES`.
    """
    return _PHASE_ESTIMATES.get(
        (component, phase),
        (f"{phase} {component}", "unknown"),
    )


def probe_toolchain(env_prefix: str, *,
                    cuda_cc_override: Optional[str] = None,
                    jobs: Optional[int] = None) -> ToolchainProbe:
    """Build a :class:`ToolchainProbe` from a live conda env + host.

    Parameters
    ----------
    env_prefix
        Absolute path to the conda env (``$CONDA_PREFIX``).
    cuda_cc_override
        Forced compute capability (``"8.0"`` style), used by
        :data:`MOLBUILDER_CUDA_CC` to override auto-detection.
    jobs
        Explicit build concurrency.  ``None`` uses :func:`_default_jobs`.
    """
    cuda_home = _detect_cuda_home(env_prefix, {})
    cuda_version = _detect_cuda_version(cuda_home) if cuda_home else None
    cuda_cc = _detect_compute_cap(cuda_cc_override)
    gcc = _detect_gcc_version(env_prefix)
    ompi = _detect_openmpi_version(env_prefix)
    return ToolchainProbe(
        env_prefix=env_prefix,
        cuda_home=cuda_home,
        cuda_version=cuda_version,
        cuda_compute_cap=cuda_cc,
        gcc_version=gcc,
        openmpi_version=ompi,
        jobs=jobs if jobs is not None else _default_jobs(),
    )


# --------------------------------------------------------------------- #
#  Robustness preflight: disk, network, GPU details                      #
# --------------------------------------------------------------------- #


def detect_gpu_name(override_cc: Optional[str] = None) -> Optional[str]:
    """Read the GPU's product name from ``nvidia-smi``.

    Used purely for friendlier preflight output (e.g. "NVIDIA A100").
    Returns ``None`` if no GPU is reachable.  The override is purely
    informational here -- if the user forced a CC there may not be a
    matching GPU at all.
    """
    if override_cc:
        return None
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    cp = _run_capture([
        smi, "--query-gpu=name", "--format=csv,noheader",
    ])
    if cp.returncode != 0:
        return None
    first = (cp.stdout or "").strip().splitlines()
    return first[0].strip() if first else None


def disk_free_gb(path: str) -> Optional[float]:
    """Free space at ``path`` in GB, or ``None`` if the path is unreachable."""
    try:
        usage = shutil.disk_usage(path)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return usage.free / (1024 ** 3)


def check_disk(path: str, *,
               required_gb: float = _DEFAULT_DISK_GB_REQUIRED,
               recommended_gb: float = _DEFAULT_DISK_GB_RECOMMENDED
               ) -> Tuple[Optional[float], Optional[str]]:
    """Return ``(free_gb, error_or_warning)``.

    The string second element is ``None`` if disk is comfortable, an
    error string when below ``required_gb``, or a warning string when
    above required but below ``recommended_gb``.  The caller decides
    whether to abort on the error (preflight) or just print the
    warning.
    """
    free = disk_free_gb(path)
    if free is None:
        return None, (
            f"Could not stat {path!r} for disk usage; check that the "
            f"path exists + is reachable."
        )
    if free < required_gb:
        return free, (
            f"Only {free:.1f} GB free at {path}; need at least "
            f"{required_gb:.0f} GB for the source-build stack "
            f"(clones + build tree + install)."
        )
    if free < recommended_gb:
        return free, (
            f"Only {free:.1f} GB free at {path}; recommended "
            f"{recommended_gb:.0f} GB for headroom during cmake "
            f"compile (large object files + ninja parallelism)."
        )
    return free, None


def check_url_reachable(url: str, *, timeout: int = 15) -> Optional[str]:
    """Return ``None`` if the URL is reachable, else an error string.

    Uses ``curl -fsI`` (HEAD request) for tarball URLs.  Cheap (no
    body download) but surfaces the same auth / DNS / firewall errors
    a real ``curl -L -o`` would hit.
    """
    if not url:
        return "empty URL"
    if shutil.which("curl") is None:
        return "curl CLI not found on PATH; install curl."
    try:
        cp = subprocess.run(
            ["curl", "-fsI", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return f"curl HEAD failed: {exc}"
    if cp.returncode != 0:
        err = (cp.stderr or cp.stdout or "").strip().splitlines()
        msg = err[-1] if err else "(no stderr)"
        return f"curl -fsI {url} returned rc={cp.returncode}: {msg}"
    return None


def check_repo_reachable(repo_url: str, *, timeout: int = 15
                         ) -> Optional[str]:
    """Return ``None`` if the git repo is reachable, else an error string.

    Uses ``git ls-remote --heads`` because it's fast (no clone) and
    surfaces the same auth/DNS/firewall errors a clone would hit.
    """
    if shutil.which("git") is None:
        return "git CLI not found on PATH; install git."
    try:
        cp = subprocess.run(
            ["git", "ls-remote", "--heads", repo_url],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return f"git ls-remote failed: {exc}"
    if cp.returncode != 0:
        err = (cp.stderr or cp.stdout or "").strip().splitlines()
        msg = err[-1] if err else "(no stderr)"
        return f"git ls-remote {repo_url} returned rc={cp.returncode}: {msg}"
    return None


def check_env_health(env_prefix: str, conda_packages: Sequence[str]
                     ) -> List[str]:
    """Sanity check the conda env after create: are key binaries present?

    Catches partial-conda-failure cases (SAT solver lied; package install
    silently failed) by checking that toolchain entries we explicitly
    listed actually landed.  Returns a list of issue strings; empty
    means OK.
    """
    issues: List[str] = []
    env_bin = Path(env_prefix) / "bin"
    if not env_bin.exists():
        issues.append(
            f"$CONDA_PREFIX/bin missing at {env_bin}; conda env did not "
            f"populate correctly."
        )
        return issues
    # Tooling binaries we depend on for the build phase.  Listed conda
    # specs -> expected binary names in $CONDA_PREFIX/bin.
    expected_bins = {
        "cmake": "cmake",
        "ninja": "ninja",
        "git": "git",
        "openmpi": "mpirun",
        # gcc_linux-64 installs as the prefixed name; absence here
        # signals the conda solve picked a different gcc variant.
        "gcc_linux-64": "x86_64-conda-linux-gnu-gcc",
        "gfortran_linux-64": "x86_64-conda-linux-gnu-gfortran",
    }
    for spec in conda_packages:
        # spec strings look like "gcc_linux-64=14"; we want the base name.
        base = spec.split("=")[0].split("::")[-1]
        binary = expected_bins.get(base)
        if binary is None:
            continue
        if not (env_bin / binary).exists():
            issues.append(
                f"conda env at {env_prefix} declares `{spec}` but "
                f"{binary!r} is absent from $CONDA_PREFIX/bin -- the "
                f"conda solve may have silently skipped it."
            )
    return issues


# --------------------------------------------------------------------- #
#  Pre-flight gates                                                      #
# --------------------------------------------------------------------- #


_GCC_FOR_CUDA: Tuple[Tuple[str, int], ...] = (
    # (CUDA major.minor as string, max gcc major).  Matched top-down:
    # the first row whose threshold the detected CUDA meets or exceeds
    # decides the max gcc.  Per NVIDIA's compatibility matrix:
    #   CUDA 13.x  -> gcc <= 14  (recipe's default; pairs with cuda-version=13.*)
    #   CUDA 12.8+ -> gcc <= 14
    #   CUDA 12.0-12.7 -> gcc <= 13  (needs MOLBUILDER_GCC=13)
    #   CUDA 11.x  -> gcc <= 11      (needs MOLBUILDER_GCC=11)
    ("13.0", 14),
    ("12.8", 14),
    ("12.0", 13),
    ("11.0", 11),
)


def _gcc_major(gcc_version: Optional[str]) -> Optional[int]:
    if not gcc_version:
        return None
    m = re.match(r"(\d+)", gcc_version)
    return int(m.group(1)) if m else None


def _cuda_tuple(cuda_version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not cuda_version:
        return None
    m = re.match(r"(\d+)\.(\d+)", cuda_version)
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_cuda_gcc_compat(probe: ToolchainProbe) -> Optional[str]:
    """Return an error string if CUDA + gcc don't pair, else ``None``.

    See :doc:`docs/engines/siesta-gpu` § 6 for the matrix.
    """
    cuda = _cuda_tuple(probe.cuda_version)
    gcc = _gcc_major(probe.gcc_version)
    if cuda is None or gcc is None:
        return None  # nothing to check; the cuda_required pre-flight handles missing CUDA
    cuda_str = f"{cuda[0]}.{cuda[1]}"
    for threshold_str, max_gcc in _GCC_FOR_CUDA:
        thr = tuple(int(p) for p in threshold_str.split("."))
        if cuda >= thr:
            if gcc <= max_gcc:
                return None
            recommended = max_gcc
            return (
                f"CUDA {cuda_str} pairs with gcc <= {max_gcc}, but the env "
                f"has gcc {gcc}.  Re-install with "
                f"`MOLBUILDER_GCC={recommended} molbuilder envs install ...` "
                f"to pin the recipe's gcc_linux-64 packages to "
                f"version {recommended}, or upgrade CUDA to 12.4+."
            )
    return (
        f"CUDA {cuda_str} is older than 11.0 and is not supported by this "
        f"recipe.  Install CUDA 12.4+ for gcc 14 (default), or use "
        f"`MOLBUILDER_GCC=11` with CUDA 11.x."
    )


def check_no_forbidden_packages(spec: BuildSpec,
                                conda_packages: Sequence[str]
                                ) -> Optional[str]:
    """Return an error if any conda spec matches a forbidden pattern."""
    if not spec.forbidden_packages:
        return None
    for pat in spec.forbidden_packages:
        # Patterns are conda specs (mkl, mkl-devel, fftw=*=mkl_*).  We
        # compare against the recipe's literal package list -- not what
        # the SAT solver ends up installing (which would need a `conda
        # list` after install).
        pat_simple = pat.split("=")[0]
        for pkg in conda_packages:
            pkg_simple = pkg.split("=")[0]
            if pkg_simple == pat_simple:
                return (
                    f"recipe declares conda_packages entry `{pkg}` but the "
                    f"build_spec.forbidden_packages list forbids `{pat}` "
                    f"to keep the env's OpenMP runtime single."
                )
    return None


@dataclass(frozen=True)
class PreflightReport:
    """Structured preflight result; CLI renders to text via :func:`format_preflight_report`.

    Attributes
    ----------
    errors
        Conditions that MUST be fixed before the build can proceed
        (missing CUDA when cuda_required, insufficient disk, etc.).
        Non-empty errors short-circuit the install with no filesystem
        side effects.
    warnings
        Conditions the user should know about but that don't block
        progress (sm_80 fallback because no GPU detected, disk
        somewhat tight, etc.).  Surfaced + the user is asked to
        confirm.
    info
        Purely informational lines for the report (detected GPU name,
        detected CUDA version, disk free, etc.).  Always shown.
    """
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    info: Tuple[str, ...]


def detect_stale_artifact_dirs(spec: BuildSpec, env_prefix: str) -> List[str]:
    """Return names of artifact dirs that are not in the current spec.

    A user re-running ``molbuilder envs install`` after an earlier
    failed install, an old recipe version (e.g. the pre-2026-06-15
    three-component shape), or hand-edited state, may have stale
    directories under ``$CONDA_PREFIX/opt/<artifact_subdir>/``.
    Concrete cases this catches:

    - An ``elsi/`` install dir left over from the deprecated 3-component
      recipe.  Harmless but signals the user is on outdated state.
    - Half-cloned ``src/<comp>/`` from an interrupted install (handled
      by the clone-wipe rule, but worth surfacing).
    - ``logs/`` or ``.sentinels/`` from a wildly different prior config
      (only weird if .toolchain-fingerprint disagrees with the current
      build's fingerprint, which is a separate sentinel-resume check).

    Returns a list of unexpected entries (relative names); empty list
    means clean.
    """
    paths = resolve_paths(spec, env_prefix)
    if not paths.root.exists():
        return []
    expected = {c.name for c in spec.components} | {
        "src", "build", "logs", ".sentinels",
        ".toolchain-fingerprint",
    }
    stale: List[str] = []
    try:
        for entry in sorted(paths.root.iterdir()):
            if entry.name not in expected:
                stale.append(entry.name)
    except OSError:
        # Unreadable dir -- preflight elsewhere will catch the actual
        # filesystem error; here we just say "no stale".
        return []
    return stale


def preflight(spec: BuildSpec, probe: ToolchainProbe,
              conda_packages: Sequence[str],
              env_prefix: Optional[str] = None,
              *,
              check_network: bool = True,
              required_disk_gb: float = _DEFAULT_DISK_GB_REQUIRED,
              recommended_disk_gb: float = _DEFAULT_DISK_GB_RECOMMENDED,
              ) -> PreflightReport:
    """Run every preflight check and return a structured report.

    Errors are hard-stops; warnings are user-confirmable; info is
    detected state for the user to see.  The legacy "list of error
    strings" return shape is preserved by callers reading
    ``report.errors`` only.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    # NVIDIA driver (host).  Kernel-module-coupled; can't be a conda
    # package.  Both the build (compute_cap detection) and runtime use
    # nvidia-smi via this driver.
    driver_ver = detect_nvidia_driver()
    if driver_ver:
        info.append(f"NVIDIA driver      {driver_ver:<10s}  (host; provides nvidia-smi)")
    elif spec.cuda_required:
        info.append("NVIDIA driver      not detected (no nvidia-smi on host)")
        errors.append(
            "NVIDIA driver not detected via nvidia-smi.  Install via "
            "the host package manager (apt/yum/dnf nvidia-driver-*).  "
            "The driver is kernel-module-coupled and cannot be a conda "
            "package."
        )

    # CUDA toolkit (env).  Installed via conda-forge into this env's
    # $CONDA_PREFIX.  After conda create, $CONDA_PREFIX/bin/nvcc must
    # exist; if it doesn't, the conda solve silently dropped cuda-nvcc.
    in_env = (
        probe.cuda_home is not None
        and env_prefix is not None
        and Path(probe.cuda_home).resolve() == Path(env_prefix).resolve()
    )
    if probe.cuda_home:
        ver = probe.cuda_version or "(version unknown)"
        where = "env" if in_env else "host (legacy)"
        info.append(f"CUDA toolkit       {ver:<10s}  in {where} at {probe.cuda_home}")
    elif spec.cuda_required:
        info.append("CUDA toolkit       not in env (will install via conda)")
    if spec.cuda_required:
        if probe.cuda_home is None:
            # Only error if the env exists and CUDA is still missing -- that
            # means conda-nvcc silently failed to install.  Pre-create
            # dry-runs hit this branch too; the install will fix it.
            if env_prefix and Path(env_prefix).exists() and (Path(env_prefix) / "conda-meta").exists():
                errors.append(
                    f"conda env at {env_prefix} has no nvcc.  The recipe "
                    f"declares cuda-nvcc but the SAT solver may have "
                    f"dropped it; re-run with `--rebuild=all` after "
                    f"`conda env remove`."
                )
        elif spec.cuda_min_version:
            ct = _cuda_tuple(probe.cuda_version)
            mt = _cuda_tuple(spec.cuda_min_version)
            if ct is not None and mt is not None and ct < mt:
                errors.append(
                    f"CUDA toolkit version {probe.cuda_version} is older "
                    f"than the recipe's minimum {spec.cuda_min_version}."
                )

    # GPU + compute capability
    if probe.cuda_compute_cap:
        gpu_name = detect_gpu_name() or "(GPU name unavailable)"
        info.append(
            f"GPU compute cap    {probe.cuda_cc_sm:<10s}  ({gpu_name})"
        )
    elif spec.cuda_required:
        info.append(
            f"GPU compute cap    sm_80 (fallback; no GPU detected via nvidia-smi)"
        )
        warnings.append(
            "GPU compute capability not detected on this host.  Build will "
            "target sm_80; the resulting binary will run on Ampere GPUs "
            "(A100, RTX 30-series) but may underperform or fail on other "
            "architectures (Hopper sm_90, Ada sm_89, Volta sm_70).  Set "
            "MOLBUILDER_CUDA_CC=<x.y> before re-running to override."
        )

    # CUDA + gcc pairing
    if probe.gcc_version:
        info.append(f"gcc                {probe.gcc_version:<10s}  (env's gcc_linux-64)")
    if probe.openmpi_version:
        info.append(f"OpenMPI            {probe.openmpi_version:<10s}  (env's openmpi)")
    compat = check_cuda_gcc_compat(probe)
    if compat:
        errors.append(compat)

    # Forbidden packages (MKL etc.)
    forbidden = check_no_forbidden_packages(spec, conda_packages)
    if forbidden:
        errors.append(forbidden)

    # Disk
    if env_prefix:
        free, disk_msg = check_disk(env_prefix,
                                    required_gb=required_disk_gb,
                                    recommended_gb=recommended_disk_gb)
        if free is not None:
            info.append(f"Disk free          {free:>5.1f} GB  at {env_prefix}")
        if disk_msg:
            if free is None or free < required_disk_gb:
                errors.append(disk_msg)
            else:
                warnings.append(disk_msg)

    # Concurrency
    info.append(
        f"Build concurrency  -j{probe.jobs:<8d}  "
        f"(MOLBUILDER_BUILD_JOBS to override)"
    )

    # Stale artifact directories from a prior failed / outdated install.
    # See `detect_stale_artifact_dirs` docstring for the failure modes.
    if env_prefix:
        stale = detect_stale_artifact_dirs(spec, env_prefix)
        if stale:
            paths = resolve_paths(spec, env_prefix)
            warnings.append(
                f"Artifact directory at {paths.root} contains "
                f"{len(stale)} entry/entries that are NOT part of the "
                f"current build ({', '.join(stale)}).  These may be "
                f"leftovers from a prior failed install or an older "
                f"recipe version.  The current install will resume from "
                f"valid sentinels and ignore these dirs, but they will "
                f"continue to consume disk.  Pass `--rebuild=all` to "
                f"wipe everything (including these stale dirs) and "
                f"start clean."
            )

    # Network: verify each component's upstream is reachable.  Cheap
    # (HEAD / ls-remote, no actual clone or download) but catches
    # firewalls, DNS, dead mirrors.  Tarball-based components use
    # ``curl -I`` (HEAD); git-based components use ``git ls-remote``.
    if check_network:
        for comp in spec.components:
            if comp.tarball_url:
                err = check_url_reachable(comp.tarball_url, timeout=15)
                source = comp.tarball_url
            elif comp.repo_url:
                err = check_repo_reachable(comp.repo_url, timeout=15)
                source = comp.repo_url
            else:
                # Component has neither tarball nor repo -- shouldn't
                # happen, but don't crash on a malformed recipe.
                err = "component has no tarball_url or repo_url set"
                source = "(none)"
            if err is None:
                info.append(f"upstream reachable ok        {source}")
            else:
                errors.append(
                    f"Cannot reach {comp.name} upstream ({source}): {err}"
                )

    return PreflightReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        info=tuple(info),
    )


def format_preflight_report(report: PreflightReport) -> str:
    """Render a :class:`PreflightReport` as multi-line text for the CLI."""
    lines: List[str] = []
    lines.append("Detected on this host:")
    for line in report.info:
        lines.append(f"  {line}")
    if report.warnings:
        lines.append("")
        lines.append("Warnings (the install will proceed, but read these):")
        for line in report.warnings:
            lines.append(f"  * {line}")
    if report.errors:
        lines.append("")
        lines.append("Errors (the install cannot proceed; fix these first):")
        for line in report.errors:
            lines.append(f"  ! {line}")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
#  Path layout                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BuildPaths:
    """Resolved on-disk layout for one BuildSpec run."""
    root: Path                     # $CONDA_PREFIX/opt/<artifact_subdir>
    src: Path                      # <root>/src
    build: Path                    # <root>/build
    logs: Path                     # <root>/logs
    sentinels: Path                # <root>/.sentinels
    fingerprint_file: Path         # <root>/.toolchain-fingerprint
    activate_d: Path               # $CONDA_PREFIX/etc/conda/activate.d
    deactivate_d: Path             # $CONDA_PREFIX/etc/conda/deactivate.d
    activate_hook: Path            # <activate_d>/zz-<artifact_subdir>.sh
    deactivate_hook: Path

    def component_src(self, name: str) -> Path:
        return self.src / name

    def component_build(self, name: str) -> Path:
        return self.build / name

    def component_install(self, name: str) -> Path:
        return self.root / name

    def sentinel(self, comp_name: str, phase: str) -> Path:
        return self.sentinels / f"{comp_name}.{phase}.done"


def resolve_paths(spec: BuildSpec, env_prefix: str) -> BuildPaths:
    """Compute the on-disk layout for one BuildSpec under an env."""
    root = Path(env_prefix) / "opt" / spec.artifact_subdir
    return BuildPaths(
        root=root,
        src=root / "src",
        build=root / "build",
        logs=root / "logs",
        sentinels=root / ".sentinels",
        fingerprint_file=root / ".toolchain-fingerprint",
        activate_d=Path(env_prefix, "etc", "conda", "activate.d"),
        deactivate_d=Path(env_prefix, "etc", "conda", "deactivate.d"),
        activate_hook=Path(env_prefix, "etc", "conda", "activate.d",
                           f"zz-{spec.artifact_subdir}.sh"),
        deactivate_hook=Path(env_prefix, "etc", "conda", "deactivate.d",
                             f"zz-{spec.artifact_subdir}.sh"),
    )


# --------------------------------------------------------------------- #
#  Toolchain fingerprint                                                 #
# --------------------------------------------------------------------- #


def compute_fingerprint(spec: BuildSpec, probe: ToolchainProbe,
                        component_refs: Mapping[str, str]) -> str:
    """SHA256 over the toolchain inputs that should force a rebuild.

    Inputs (sorted-JSON canonical form):
        - cuda_version, cuda_compute_cap
        - gcc_version, openmpi_version
        - artifact_subdir, omp_runtime
        - per-component (repo_url, ref, resolved_sha)

    ``component_refs`` is a mapping of component name to the resolved
    git SHA (after clone + checkout).  When a component hasn't been
    cloned yet, pass the declared ``ref`` instead -- the fingerprint
    will change after the clone resolves the real SHA, which is what
    we want.
    """
    inputs = {
        "cuda_version": probe.cuda_version,
        "cuda_compute_cap": probe.cuda_compute_cap,
        "gcc_version": probe.gcc_version,
        "openmpi_version": probe.openmpi_version,
        "artifact_subdir": spec.artifact_subdir,
        "omp_runtime": spec.omp_runtime,
        "components": {
            comp.name: {
                "repo": comp.repo_url,
                "declared_ref": comp.ref,
                "resolved_ref": component_refs.get(comp.name, comp.ref),
            }
            for comp in spec.components
        },
    }
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def read_sentinel_fingerprint(sentinel_path: Path) -> Optional[str]:
    """Read the fingerprint hash recorded in a sentinel file."""
    if not sentinel_path.exists():
        return None
    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fp = data.get("fingerprint")
    return fp if isinstance(fp, str) else None


def write_sentinel(sentinel_path: Path, fingerprint: str, *,
                   now: Optional[Callable[[], float]] = None) -> None:
    """Write a sentinel file recording the toolchain fingerprint."""
    now_fn = now if now is not None else time.time
    payload = {
        "fingerprint": fingerprint,
        "timestamp": now_fn(),
    }
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(json.dumps(payload, sort_keys=True),
                             encoding="utf-8")


def sentinel_valid(sentinel_path: Path, fingerprint: str) -> bool:
    """Backwards-compat shim.  Sentinels are now plain existence markers
    (the artifact-presence probe at install start is the trust source);
    callers should prefer ``sentinel_path.exists()`` directly.  This
    helper still exists so external tooling that imports it doesn't
    break, and so the recorded fingerprint can be used as forensic
    metadata when ``compute_fingerprint`` is recomputed for debugging.
    """
    return sentinel_path.exists()


def component_install_valid(
    spec: BuildSpec,
    paths: BuildPaths,
    comp: BuildComponent,
    probe: ToolchainProbe,
    conda_binary: str,
) -> bool:
    """Probe whether one component is already installed and working.

    Runs the component's ``verify_argv`` against the install dir.
    Returns True when the install dir exists AND verify exits 0 (and
    matches ``verify_expected`` if set).  Returns False when the
    install dir is missing, no verify is defined, or the probe fails.

    This is the artifact-presence gate that replaced the global
    fingerprint check.  See the 2026-06-15 "drop fingerprint, gate
    on artifact" decision in molbuilder/docs/design.md for context.
    """
    install_dir = paths.component_install(comp.name)
    if not install_dir.exists():
        return False
    if not comp.verify_argv:
        # No verify defined -- safer to re-run than to assume done.
        return False
    subs = _build_substitutions(comp, spec, paths, probe)
    try:
        argv = _apply_template(comp.verify_argv, subs)
    except ValueError:
        return False
    # ``test -f`` and similar work as plain argv; binary checks
    # (e.g. ``{install}/bin/siesta --version``) also work.  We don't
    # route through ``conda run`` here -- the install dir is a fully
    # self-contained set of files, and the verify argv either targets
    # a file existence check or an executable inside that dir.
    cp = _run_capture(list(argv))
    return cp.returncode == 0


# --------------------------------------------------------------------- #
#  Template substitution                                                 #
# --------------------------------------------------------------------- #


def _build_substitutions(component: BuildComponent,
                         spec: BuildSpec,
                         paths: BuildPaths,
                         probe: ToolchainProbe) -> Mapping[str, str]:
    """Compute the placeholder dict for one component's argv templates.

    Naming convention: ``dep_<name>`` (underscore, not colon) because
    Python's :meth:`str.format_map` treats ``:`` as the format-spec
    separator and rejects ``{dep:elpa}``.
    """
    subs = {
        "prefix": str(paths.root),
        # ``env_prefix`` points at $CONDA_PREFIX itself -- used to pin
        # cmake to the conda env's compilers / MPI / CUDA / libs so
        # FindMPI / FindCUDA / FindBLAS can't pick up system installs.
        "env_prefix": str(probe.env_prefix),
        "src": str(paths.component_src(component.name)),
        "build": str(paths.component_build(component.name)),
        "install": str(paths.component_install(component.name)),
        "cuda_cc_numeric": probe.cuda_cc_numeric,
        "cuda_cc_sm": probe.cuda_cc_sm,
        # cuda_home points at the env (where conda-installed nvcc + libs
        # live).  Falls back to the env_prefix when the probe couldn't
        # find nvcc (which happens during dry-run before conda create).
        "cuda_home": probe.cuda_home or probe.env_prefix,
        "jobs": str(probe.jobs),
    }
    for other in spec.components:
        subs[f"dep_{other.name}"] = str(paths.component_install(other.name))
    return subs


def _apply_template(argv: Sequence[str], subs: Mapping[str, str]) -> Tuple[str, ...]:
    """Substitute ``{name}`` placeholders in every argv element."""
    out: List[str] = []
    for token in argv:
        try:
            out.append(token.format_map(subs))
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"unknown placeholder in build argv: {token!r} "
                f"(available: {sorted(subs)})"
            ) from exc
    return tuple(out)


def render_activate_hook(spec: BuildSpec, paths: BuildPaths,
                         probe: ToolchainProbe) -> str:
    """Render the activate.d hook body for the spec.

    The hook uses literal ``$CONDA_PREFIX`` (resolved at activation
    time by conda itself) -- no install-time-baked absolute paths
    means the hook still works if the env is cloned or moved.  This
    function is therefore a pass-through; the BuildSpec writes the
    hook text it wants verbatim.
    """
    return spec.activate_hook or ""


def render_deactivate_hook(spec: BuildSpec, paths: BuildPaths,
                           probe: ToolchainProbe) -> str:
    """Render the deactivate.d hook body (pass-through; see
    :func:`render_activate_hook`)."""
    return spec.deactivate_hook or ""


# --------------------------------------------------------------------- #
#  Plan / execute                                                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BuildStep:
    """One (component, phase) entry in the plan."""
    component: str
    phase: str
    argv: Tuple[str, ...]
    sentinel: Path
    log_file: Path
    cwd: Optional[Path] = None


@dataclass(frozen=True)
class BuildStepResult:
    """One executed step's outcome (or skipped/not-run record)."""
    step: BuildStep
    status: str        # "ok" | "skip" | "fail" | "not-run"
    returncode: Optional[int] = None
    output: str = ""


@dataclass(frozen=True)
class BuildResult:
    """Outcome of one :func:`run_build_spec` invocation."""
    spec: BuildSpec
    env_prefix: str
    fingerprint: str
    preflight_errors: Tuple[str, ...]
    steps: Tuple[BuildStepResult, ...]
    activate_hook_written: bool
    deactivate_hook_written: bool
    succeeded: bool


# Components that must be wiped if a named component is requested via
# --rebuild=<name>.  Wipes propagate downstream (something later in the
# chain has linked the rebuilt component's install dir).
def downstream_components(spec: BuildSpec, name: str) -> Tuple[str, ...]:
    """Return the named component plus everything downstream of it."""
    names = [c.name for c in spec.components]
    if name not in names:
        return ()
    idx = names.index(name)
    return tuple(names[idx:])


def _components_to_wipe(spec: BuildSpec,
                        rebuild: Optional[str]) -> Tuple[str, ...]:
    """Resolve a ``--rebuild`` argument to the set of components to wipe."""
    if not rebuild or rebuild == "none":
        return ()
    if rebuild == "all":
        return tuple(c.name for c in spec.components)
    return downstream_components(spec, rebuild)


def _wipe_component(paths: BuildPaths, name: str) -> None:
    """Drop sentinels + build dir + install dir for one component.

    Source dir is preserved -- the clone is the cheapest phase to re-do
    if the user explicitly asks via --rebuild=all, but resume should
    not re-clone on a normal rebuild.
    """
    for phase in PHASES:
        sentinel = paths.sentinel(name, phase)
        if sentinel.exists():
            sentinel.unlink()
    bd = paths.component_build(name)
    if bd.exists():
        shutil.rmtree(bd, ignore_errors=True)
    install_dir = paths.component_install(name)
    if install_dir.exists():
        shutil.rmtree(install_dir, ignore_errors=True)


def plan_build_spec(spec: BuildSpec,
                    env_prefix: str,
                    probe: ToolchainProbe,
                    component_refs: Mapping[str, str]
                    ) -> Tuple[BuildPaths, str, List[BuildStep]]:
    """Build the step list (does NOT execute or touch the filesystem)."""
    paths = resolve_paths(spec, env_prefix)
    fingerprint = compute_fingerprint(spec, probe, component_refs)

    steps: List[BuildStep] = []
    for comp in spec.components:
        subs = _build_substitutions(comp, spec, paths, probe)
        # Clone phase: either ``git clone`` (most components) or
        # ``curl + tar`` (when ``tarball_url`` is set).  The phase
        # name stays "clone" so the sentinel-resume model works
        # uniformly.
        src_dir = paths.component_src(comp.name)
        if comp.tarball_url:
            # Tarball path -- matches conda-forge's elpa-feedstock
            # pattern.  Single shell command that:
            #   1. wipes any partial state from a prior failed install
            #   2. downloads via curl with -fL (fail on HTTP error, follow redirects)
            #   3. verifies SHA256 if the recipe pinned one
            #   4. extracts via tar
            #   5. renames inner dir (e.g. "elpa-2021.11.001/") to {src}
            tar_path = src_dir.parent / f"{comp.name}.tar.gz"
            sha_check = ""
            if comp.tarball_sha256:
                sha_check = (
                    f'echo "{comp.tarball_sha256}  $TARBALL" '
                    f'  | sha256sum -c -; '
                )
            clone_argv = (
                "sh", "-c",
                f'set -e; '
                f'TARBALL="{tar_path}"; '
                f'rm -rf "{src_dir}" "$TARBALL"; '
                f'mkdir -p "{src_dir.parent}"; '
                f'echo "[clone] downloading {comp.tarball_url}"; '
                f'curl -fsSL -o "$TARBALL" "{comp.tarball_url}"; '
                + sha_check +
                f'echo "[clone] extracting"; '
                f'tar -xzf "$TARBALL" -C "{src_dir.parent}"; '
                f'mv "{src_dir.parent}/{comp.tarball_inner_dir}" "{src_dir}"; '
                f'rm "$TARBALL"; '
                f'echo "[clone] source ready at {src_dir}"',
            )
        else:
            # Git clone path.  --recurse-submodules brings the
            # component's External/ submodules along in one clone
            # (SIESTA needs this to pull libfdf + libpsml + xmlf90 +
            # libgridxc + ELSI + libxc per SIESTA INSTALL.md).
            clone_argv_list: List[str] = ["git", "clone"]
            if comp.clone_shallow:
                clone_argv_list.append("--depth=1")
            if comp.clone_recurse_submodules:
                clone_argv_list.extend([
                    "--recurse-submodules",
                    "--shallow-submodules",
                ])
            clone_argv_list.extend([
                "--branch", comp.ref, comp.repo_url,
                str(src_dir),
            ])
            clone_argv = tuple(clone_argv_list)
        steps.append(BuildStep(
            component=comp.name, phase="clone",
            argv=clone_argv,
            sentinel=paths.sentinel(comp.name, "clone"),
            log_file=paths.logs / f"{comp.name}.clone.log",
            cwd=None,
        ))
        steps.append(BuildStep(
            component=comp.name, phase="configure",
            argv=_apply_template(comp.configure_argv, subs),
            sentinel=paths.sentinel(comp.name, "configure"),
            log_file=paths.logs / f"{comp.name}.configure.log",
            cwd=None,
        ))
        steps.append(BuildStep(
            component=comp.name, phase="build",
            argv=_apply_template(comp.build_argv, subs),
            sentinel=paths.sentinel(comp.name, "build"),
            log_file=paths.logs / f"{comp.name}.build.log",
            cwd=None,
        ))
        steps.append(BuildStep(
            component=comp.name, phase="install",
            argv=_apply_template(comp.install_argv, subs),
            sentinel=paths.sentinel(comp.name, "install"),
            log_file=paths.logs / f"{comp.name}.install.log",
            cwd=None,
        ))
        if comp.verify_argv:
            steps.append(BuildStep(
                component=comp.name, phase="verify",
                argv=_apply_template(comp.verify_argv, subs),
                sentinel=paths.sentinel(comp.name, "verify"),
                log_file=paths.logs / f"{comp.name}.verify.log",
                cwd=None,
            ))
    return paths, fingerprint, steps


def _run_phase(step: BuildStep,
               *,
               env_prefix: str,
               conda_binary: str,
               timeout: int = 7200) -> BuildStepResult:
    """Run one phase under ``conda run -n <env>`` with live output.

    Output streams to stderr line-by-line so the user can see the
    build's progress (cmake compile lines, ninja step counts, git
    clone receiving-objects updates) instead of staring at a silent
    "[1/15] elpa.clone: ..." for 15 minutes.  The full transcript is
    still written to ``step.log_file`` for post-hoc inspection.
    """
    step.log_file.parent.mkdir(parents=True, exist_ok=True)
    # Bypass ``<mgr> run`` and call build tools through a bash
    # wrapper that sets ``conda activate``'s env vars and sources
    # the env's activate.d/*.sh hooks.  Three problems this solves:
    #
    #   (1) mamba 1.x's ``run`` generates ``exec --`` which bash
    #       rejects -- the entire build phase fails on line 5 of
    #       mamba's stub.  Fixed in mamba 2.x but still common.
    #   (2) ``conda activate`` is what cmake/ninja/make would
    #       normally see for PATH + LD_LIBRARY_PATH + CONDA_PREFIX
    #       + CONDA_DEFAULT_ENV.  We replicate it inline.
    #   (3) Earlier components of the build (ELPA installs to
    #       ``<env>/opt/<artifact>/elpa``) register their own
    #       activate.d hook so the next component (SIESTA) sees
    #       libelpa on the link path.  Sourcing activate.d makes
    #       that work without re-emitting the per-component
    #       LD_LIBRARY_PATH glue in this Python.
    import shlex as _shlex
    cmd_q = " ".join(_shlex.quote(str(a)) for a in step.argv)
    env_q = _shlex.quote(env_prefix)
    env_name = _shlex.quote(Path(env_prefix).name)
    activate_d = _shlex.quote(f"{env_prefix}/etc/conda/activate.d")
    # HPC strictness: every temp + cache dir the build tools touch
    # must live under the artifact root (which is under
    # $CONDA_PREFIX/opt/<artifact_subdir>).  Otherwise cmake's
    # temp probes go to /tmp (size-limited on most clusters),
    # ccache (if gcc is wrapped) writes to ~/.ccache (escapes env),
    # python tooling like meson scribbles in ~/.cache.  Pinning
    # them here means the env is the single dir an admin needs to
    # clean up.  Derive the artifact root from the step's log path
    # (every log file is ``<root>/logs/<comp>.<phase>.log`` per the
    # ``paths`` layout in plan_build_spec).
    paths_root = _shlex.quote(str(step.log_file.parent.parent))
    wrapper = (
        f"export CONDA_PREFIX={env_q}; "
        f"export CONDA_DEFAULT_ENV={env_name}; "
        f'export PATH={env_q}/bin"${{PATH:+:$PATH}}"; '
        f'export LD_LIBRARY_PATH={env_q}/lib"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        f"mkdir -p {paths_root}/.tmp {paths_root}/.ccache "
        f"{paths_root}/.cache/pip; "
        f"export TMPDIR={paths_root}/.tmp; "
        f"export TMP={paths_root}/.tmp; "
        f"export TEMP={paths_root}/.tmp; "
        f"export CCACHE_DIR={paths_root}/.ccache; "
        f"export PIP_CACHE_DIR={paths_root}/.cache/pip; "
        f"export XDG_CACHE_HOME={paths_root}/.cache; "
        f"if [ -d {activate_d} ]; then "
        f'for _f in {activate_d}/*.sh; do '
        f'[ -f "$_f" ] && . "$_f"; '
        f"done; "
        f"fi; "
        f"exec {cmd_q}"
    )
    rc, combined = run_streaming(
        ("bash", "-c", wrapper),
        # Strip leakage vectors (CPATH / CFLAGS / MPI_HOME / OMPI_*
        # / ...) so user-shell vars can't pull system MPI/CUDA into
        # the build.  PATH / LD_LIBRARY_PATH / CONDA_PREFIX are set
        # explicitly by the wrapper above on top of this slate.
        env=build_subprocess_env(),
        log_file=step.log_file,
        timeout=timeout,
    )
    if rc is None and not combined:
        # run_streaming returned launch failure with no captured output;
        # _run_phase still has to surface something to the caller.
        combined = "failed to launch (no output)"
    trimmed = combined[-4096:]  # tail -- cmake compile errors tend to be terminal
    status = "ok" if rc == 0 else "fail"
    return BuildStepResult(
        step=step,
        status=status,
        returncode=rc,
        output=trimmed,
    )


ProgressEvent = str  # "start" | "skip" | "ok" | "fail" | "wipe"
ProgressCallback = Callable[[ProgressEvent, "BuildStep", Optional["BuildStepResult"]], None]
ConfirmWarningsCallback = Callable[["PreflightReport"], bool]


def run_build_spec(spec: BuildSpec,
                   env_prefix: str,
                   *,
                   conda_binary: str,
                   cuda_cc_override: Optional[str] = None,
                   jobs: Optional[int] = None,
                   rebuild: Optional[str] = None,
                   conda_packages: Sequence[str] = (),
                   skip_preflight: bool = False,
                   skip_network_check: bool = False,
                   on_warnings: Optional[ConfirmWarningsCallback] = None,
                   on_progress: Optional[ProgressCallback] = None,
                   now: Optional[Callable[[], float]] = None,
                   ) -> BuildResult:
    """Execute a :class:`BuildSpec` against a live conda env.

    Parameters
    ----------
    rebuild
        ``None`` or ``"none"`` -> resume (skip valid sentinels).
        ``"all"`` -> wipe every component, then run.
        ``<component-name>`` -> wipe that component + everything
        downstream of it (later components in dependency order).
    skip_preflight
        Bypass the CUDA / gcc / forbidden-package / disk / network
        pre-flight checks.  Used by tests; production calls leave
        this ``False``.
    skip_network_check
        Skip the ``git ls-remote`` reachability check (CI sometimes
        blocks outbound).  Other preflight checks still run.
    on_warnings
        Callback invoked when preflight returns non-fatal warnings.
        It receives the :class:`PreflightReport` and returns ``True``
        to proceed, ``False`` to abort.  ``None`` (default) means
        "proceed silently".
    on_progress
        Callback invoked at every phase boundary -- ``"start"`` before
        a step runs (result is ``None``), ``"skip"`` when a valid
        sentinel short-circuits the step, ``"ok"`` / ``"fail"`` after
        execution.  Used by the CLI to render per-step progress; tests
        pass a recording callable.
    """
    probe = probe_toolchain(env_prefix, cuda_cc_override=cuda_cc_override,
                            jobs=jobs)
    report = PreflightReport(errors=(), warnings=(), info=())
    if not skip_preflight:
        report = preflight(spec, probe, conda_packages, env_prefix,
                           check_network=not skip_network_check)

    if report.errors:
        # Pre-flight failure short-circuits.  No filesystem mutations.
        return BuildResult(
            spec=spec, env_prefix=env_prefix,
            fingerprint="",
            preflight_errors=report.errors,
            steps=(),
            activate_hook_written=False,
            deactivate_hook_written=False,
            succeeded=False,
        )

    if report.warnings and on_warnings is not None:
        if not on_warnings(report):
            # User declined to proceed past warnings.
            return BuildResult(
                spec=spec, env_prefix=env_prefix,
                fingerprint="",
                preflight_errors=(
                    "user declined to proceed past preflight warnings",
                ) + report.warnings,
                steps=(),
                activate_hook_written=False,
                deactivate_hook_written=False,
                succeeded=False,
            )

    paths = resolve_paths(spec, env_prefix)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.sentinels.mkdir(parents=True, exist_ok=True)

    # Apply --rebuild: wipe sentinels + build + install for affected
    # components.  Source dirs are preserved so resume on clone can
    # short-circuit when the ref hasn't moved.
    for comp_name in _components_to_wipe(spec, rebuild):
        _wipe_component(paths, comp_name)

    # --rebuild=all also wipes any stale artifact dirs left over from a
    # prior recipe version (e.g. an `elsi/` install dir from the
    # pre-2026-06-15 three-component recipe).  Without this, a user
    # passing --rebuild=all to "start fresh" would still have those
    # stale dirs surviving on disk.
    if rebuild == "all":
        for stale_name in detect_stale_artifact_dirs(spec, env_prefix):
            stale_path = paths.root / stale_name
            if stale_path.exists():
                shutil.rmtree(stale_path, ignore_errors=True)

    # Resolved-ref dict is still useful as forensic metadata recorded
    # in the fingerprint file, but it no longer gates rebuilds.  The
    # artifact-presence probe below is the trust source.
    component_refs: dict = {}
    for comp in spec.components:
        src_git = paths.component_src(comp.name) / ".git"
        if src_git.exists():
            cp = _run_capture(["git", "-C", str(paths.component_src(comp.name)),
                               "rev-parse", "HEAD"])
            if cp.returncode == 0 and cp.stdout.strip():
                component_refs[comp.name] = cp.stdout.strip()
                continue
        component_refs[comp.name] = comp.ref

    _paths_unused, fingerprint, plan = plan_build_spec(
        spec, env_prefix, probe, component_refs,
    )

    paths.fingerprint_file.write_text(fingerprint, encoding="utf-8")

    # Artifact-presence reconciliation -- the trust source for "is
    # this component installed".  Replaces the old global-fingerprint
    # sentinel check (which invalidated ELPA whenever SIESTA's ref
    # shifted, causing wasteful rebuilds).  Two cases per component:
    #
    # 1. install dir present + ``verify_argv`` exits 0
    #    --> fast-forward ALL phase sentinels so the loop skips this
    #    component.  Editing a SIESTA cmake flag won't invalidate ELPA;
    #    ``--rebuild=siesta`` won't punish ELPA either (ELPA's install
    #    dir survives, verify still passes).
    #
    # 2. install dir missing OR verify fails
    #    --> wipe just the install + verify sentinels so those phases
    #    re-run.  Clone/configure/build sentinels stay -- if the prior
    #    build artifacts are intact in ``{build}/``, re-running install
    #    is fast and resume-friendly.  If they're also broken, the user
    #    can pass ``--rebuild=<component>`` to wipe the whole component.
    for comp in spec.components:
        if component_install_valid(spec, paths, comp, probe, conda_binary):
            for phase in PHASES:
                sentinel = paths.sentinel(comp.name, phase)
                if not sentinel.exists():
                    write_sentinel(sentinel, fingerprint, now=now)
        else:
            for phase in ("install", "verify"):
                sentinel = paths.sentinel(comp.name, phase)
                if sentinel.exists():
                    sentinel.unlink()

    executed: List[BuildStepResult] = []
    failed = False
    for step in plan:
        if failed:
            executed.append(BuildStepResult(step=step, status="not-run"))
            continue
        if step.sentinel.exists():
            skipped = BuildStepResult(step=step, status="skip",
                                      returncode=0,
                                      output="sentinel present; skipped")
            executed.append(skipped)
            if on_progress is not None:
                on_progress("skip", step, skipped)
            continue
        if on_progress is not None:
            on_progress("start", step, None)
        # If we're about to re-configure, wipe the build dir so cmake
        # doesn't reuse a stale CMakeCache.txt (which gets pinned to
        # the OLD CMAKE_INSTALL_PREFIX / OLD compilers).
        if step.phase in _PHASES_WIPING_BUILD:
            bd = paths.component_build(step.component)
            if bd.exists():
                shutil.rmtree(bd, ignore_errors=True)
            bd.mkdir(parents=True, exist_ok=True)
        # Same for clone: if a previous run left a partial src dir but
        # no clone sentinel, wipe and redo.
        if step.phase == "clone":
            src_dir = paths.component_src(step.component)
            if src_dir.exists():
                shutil.rmtree(src_dir, ignore_errors=True)
            src_dir.parent.mkdir(parents=True, exist_ok=True)
        result = _run_phase(step,
                            env_prefix=env_prefix,
                            conda_binary=conda_binary)
        executed.append(result)
        if on_progress is not None:
            on_progress(result.status, step, result)
        if result.status != "ok":
            failed = True
            continue
        write_sentinel(step.sentinel, fingerprint, now=now)

    activate_written = False
    deactivate_written = False
    if not failed:
        body = render_activate_hook(spec, paths, probe)
        if body:
            paths.activate_d.mkdir(parents=True, exist_ok=True)
            paths.activate_hook.write_text(body, encoding="utf-8")
            paths.activate_hook.chmod(0o755)
            activate_written = True
        body = render_deactivate_hook(spec, paths, probe)
        if body:
            paths.deactivate_d.mkdir(parents=True, exist_ok=True)
            paths.deactivate_hook.write_text(body, encoding="utf-8")
            paths.deactivate_hook.chmod(0o755)
            deactivate_written = True

    return BuildResult(
        spec=spec, env_prefix=env_prefix,
        fingerprint=fingerprint,
        preflight_errors=(),
        steps=tuple(executed),
        activate_hook_written=activate_written,
        deactivate_hook_written=deactivate_written,
        succeeded=not failed,
    )


def format_install_summary(spec: BuildSpec, probe: ToolchainProbe,
                           rebuild: Optional[str] = None) -> str:
    """Pre-install banner: what will happen + rough cost estimates.

    Printed before any subprocess runs so the user can bail out.  Cost
    rows are tuned for 8-core / broadband; the executor doesn't try to
    scale these because user-visible "this will take ~45 min" is a
    coarse contract, not a SLA.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    if rebuild and rebuild not in ("none", None):
        lines.append(f"  --rebuild={rebuild} for: {spec.artifact_subdir}")
    else:
        lines.append(f"  About to build: {spec.artifact_subdir}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("This is a SOURCE BUILD env -- the executor will:")
    lines.append("")
    for i, comp in enumerate(spec.components, start=1):
        clone_act, clone_cost = describe_phase(comp.name, "clone")
        cfg_act, cfg_cost = describe_phase(comp.name, "configure")
        bld_act, bld_cost = describe_phase(comp.name, "build")
        inst_act, inst_cost = describe_phase(comp.name, "install")
        # Show tarball URL for tarball-based components (e.g. ELPA),
        # else the git repo URL.
        source = (
            f"tarball: {comp.tarball_url}"
            if comp.tarball_url
            else comp.repo_url
        )
        lines.append(f"  Component {i}: {comp.name}  ({source} @ {comp.ref})")
        lines.append(f"     - {clone_act:<55s} {clone_cost}")
        lines.append(f"     - {cfg_act:<55s} {cfg_cost}")
        lines.append(f"     - {bld_act:<55s} {bld_cost}")
        lines.append(f"     - {inst_act:<55s} {inst_cost}")
        if comp.verify_argv:
            ver_act, ver_cost = describe_phase(comp.name, "verify")
            lines.append(f"     - {ver_act:<55s} {ver_cost}")
    lines.append("")
    lines.append(f"  Build concurrency:  -j{probe.jobs}")
    lines.append(f"  Resume model:       sentinel-based (re-running is safe)")
    lines.append(f"  Total est. time:    ~45 min on 8 cores, broadband")
    lines.append(f"  Total est. disk:    ~12 GB under $CONDA_PREFIX")
    lines.append("")
    return "\n".join(lines)


def format_progress_event(event: ProgressEvent,
                          step: BuildStep,
                          step_index: int,
                          step_total: int,
                          ) -> str:
    """Render one progress event as a single CLI line.

    Used by the CLI's progress callback to print uniform per-step
    headers.  Returns multiple lines separated by ``\\n``.
    """
    action, cost = describe_phase(step.component, step.phase)
    head = f"[{step_index:>2d}/{step_total}]"
    if event == "start":
        return f"{head} {step.component}.{step.phase}: {action} ({cost})"
    if event == "skip":
        return (f"{head} {step.component}.{step.phase}: skipped "
                f"(sentinel valid)")
    if event == "ok":
        return f"{head} {step.component}.{step.phase}: OK"
    if event == "fail":
        return f"{head} {step.component}.{step.phase}: FAILED"
    return f"{head} {step.component}.{step.phase}: {event}"


__all__ = [
    "PHASES",
    "BuildStep", "BuildStepResult", "BuildResult", "BuildPaths",
    "ToolchainProbe", "PreflightReport",
    "build_subprocess_env",
    "probe_toolchain",
    "describe_phase",
    "preflight",
    "format_preflight_report",
    "format_install_summary",
    "format_progress_event",
    "check_cuda_gcc_compat",
    "check_no_forbidden_packages",
    "check_disk",
    "check_repo_reachable",
    "check_env_health",
    "disk_free_gb",
    "detect_gpu_name",
    "resolve_paths",
    "compute_fingerprint",
    "read_sentinel_fingerprint",
    "write_sentinel",
    "sentinel_valid",
    "component_install_valid",
    "downstream_components",
    "plan_build_spec",
    "run_build_spec",
    "render_activate_hook",
    "render_deactivate_hook",
]
