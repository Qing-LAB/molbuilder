"""Post-install validation probes for env recipes.

Where :mod:`molbuilder.envs.doctor` answers "is the env present and
does the binary launch?", this module answers the next question:
**"does the env actually compute correctly?"**

For ``molbuilder-siesta-gpu`` -- the only recipe with a validator
today -- this runs four probes (~2 min wall-clock) that catch the
failure modes ``siesta --version`` cannot:

  1. binary-link sanity      siesta/tbtrans/phtrans present + version OK
  2. CUDA stack              nvidia-smi + libcuda.so.1 ctypes load
  3. ELPA GPU codepath       greps for the silent-CPU-fallback warning
                             (the load-bearing one -- catches the
                             single failure mode none of the others can)
  4. SIESTA ctest -L simple  the upstream "binary runs SCF" set (~110s)

The load-bearing probe is #3: ``nvidia-smi`` can report a perfectly
healthy GPU while ELPA silently runs on the CPU (elpa#15, same A100 +
``--enable-nvidia-sm80-gpu`` configuration we ship).  None of the
other probes catches that.

Excluded from the default suite:
  * ``elpa make check``: comprehensive test suite (~300+ validators,
    15-30 min wall-clock).  Wrong size for an interactive probe -- it's
    test coverage, not a smoke check.  Will live behind a future
    ``validate --deep`` flag for when you actually want to wait.
  * ``deviceQuery``: not in conda CUDA packages, redundant with #2.
  * CPU-vs-GPU energy cross-check at ~1e-4 eV: too loose (real ELPA2
    GPU agreement is ~1e-6 eV total; arXiv 2002.10991).  Future deep mode.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import builds as _builds
from .recipes import Recipe, recipe_by_name


# --------------------------------------------------------------------- #
#  Result dataclasses                                                    #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeResult:
    """One validation probe's outcome.

    Attributes
    ----------
    name
        Short slug shown in the report ("binary-links", "siesta ctest"…).
    passed
        ``True`` when the probe's pass criterion was met.  ``False``
        when the underlying command failed, the criterion didn't match,
        or the prerequisites for running it are missing (in which case
        ``detail`` says so explicitly).
    detail
        One-line human summary suitable for table display.
    output
        Captured stdout+stderr trimmed to 4 KiB so the CLI can dump it
        for a failing probe without flooding the terminal.
    """
    name: str
    passed: bool
    detail: str
    output: str = ""


@dataclass(frozen=True)
class ValidationReport:
    recipe_name: str
    env_prefix: str
    probes: Tuple[ProbeResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(p.passed for p in self.probes) and len(self.probes) > 0


# --------------------------------------------------------------------- #
#  Internal helpers                                                      #
# --------------------------------------------------------------------- #


# The silent-CPU-fallback signature.  ELPA prints this VERBATIM from
# ``src/elpa2/elpa2_template.F90`` when --enable-nvidia-gpu was set
# at build time but no GPU-aware kernel was selected at runtime, OR
# when the runtime CUDA context refuses to attach.  See elpa#15 --
# the exact A100 / sm_80 case we build for.
_ELPA_GPU_FALLBACK_RE = re.compile(
    r"GPU usage has been requested but compute kernel is set by the user as non-GPU",
    re.IGNORECASE,
)


def _trim(text: str, limit: int = 4096) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (trimmed, full length {len(text)} chars)"


def _run(argv, *, cwd: Optional[Path] = None,
         env: Optional[dict] = None,
         timeout: int = 120) -> Tuple[int, str]:
    """Run a subprocess and return (returncode, combined_output).

    Catches launch failures (binary not found, permission denied,
    timeout) as a returncode of -1 with the error in the output --
    callers never have to wrap this in try/except for the common path.
    """
    try:
        cp = subprocess.run(
            argv, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError,
            PermissionError, OSError) as exc:
        return -1, f"<<launch error>> {type(exc).__name__}: {exc}"
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _stack_root(env_prefix: str, recipe: Recipe) -> Path:
    """Return ``$env_prefix/opt/<artifact_subdir>`` for a source-build
    recipe (where build dirs + install dirs live)."""
    return Path(env_prefix) / "opt" / recipe.build_spec.artifact_subdir


# --------------------------------------------------------------------- #
#  Per-probe implementations                                             #
# --------------------------------------------------------------------- #


def _probe_binary_links(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 1: the conda-forge floor.

    Confirms the installed binaries exist (rpath/lib resolution works)
    and that ``siesta --version`` exits 0.  Anything before this and
    the user can't even ``conda activate && which siesta``.
    """
    siesta_install = _stack_root(env_prefix, recipe) / "siesta"
    bin_dir = siesta_install / "bin"
    # SIESTA 5.4.2 ships these binaries.  ``transiesta`` is NOT a
    # separate executable -- TranSiesta is a run mode of ``siesta``
    # itself (``%block ... transiesta`` in the .fdf).  Used to be its
    # own binary in 4.x; the merge landed pre-5.0.  ``siesta_qmmm`` is
    # also present but is a niche QM/MM driver -- not required for the
    # "binary is sane" check.
    required = ("siesta", "tbtrans", "phtrans")
    missing = [b for b in required if not (bin_dir / b).is_file()]
    if missing:
        return ProbeResult(
            name="binary-links",
            passed=False,
            detail=f"missing binary(s): {', '.join(missing)}",
            output=f"checked dir: {bin_dir}",
        )
    rc, out = _run([str(bin_dir / "siesta"), "--version"], timeout=15)
    if rc != 0:
        return ProbeResult(
            name="binary-links",
            passed=False,
            detail=f"siesta --version exited rc={rc}",
            output=_trim(out),
        )
    # Pull the version line out for the detail.
    version = ""
    for line in out.splitlines():
        if "Version" in line and ":" in line:
            version = line.split(":", 1)[1].strip()
            break
    return ProbeResult(
        name="binary-links",
        passed=True,
        detail=f"{len(required)} binaries present; siesta {version}".rstrip(),
        output=_trim(out),
    )


def _probe_siesta_ctest_simple(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 2: SIESTA's own ``-L simple`` ctest set.

    Runs the upstream tests labelled ``simple`` (currently
    00.BasisSets/default_basis + 01.PseudoPotentials/psf + .../full.psml)
    with ``-E verify`` so the libxc-version-sensitive energy-comparison
    subtests don't trip on conda-package drift.  This is the test set
    the SIESTA Tests/README explicitly recommends as
    "is the freshly compiled binary sane?".

    Output streams live (via ``_builds.run_streaming``) so the user
    sees ctest's own per-test progress instead of a 20 s silent gap.
    """
    build_dir = _stack_root(env_prefix, recipe) / "build" / "siesta"
    if not (build_dir / "CTestTestfile.cmake").is_file():
        return ProbeResult(
            name="siesta ctest",
            passed=False,
            detail=f"build dir missing CTestTestfile.cmake at {build_dir}",
        )
    # Prefer the env's ctest (matches the cmake we built with) over
    # /usr/bin/ctest (host's, possibly older).  Fall back if needed.
    env_ctest = Path(env_prefix) / "bin" / "ctest"
    ctest = str(env_ctest) if env_ctest.is_file() else shutil.which("ctest")
    if ctest is None:
        return ProbeResult(
            name="siesta ctest",
            passed=False,
            detail="ctest not found in env or on PATH",
        )
    start = time.monotonic()
    rc, out = _builds.run_streaming(
        [ctest, "-E", "verify", "-L", "simple", "--output-on-failure"],
        cwd=build_dir,
        timeout=120,
    )
    elapsed = time.monotonic() - start
    # ctest's tail line is the most reliable parse target.
    summary = ""
    for line in (out or "").splitlines():
        if "tests passed" in line:
            summary = line.strip()
    passed = (rc == 0)
    detail = summary or (f"ctest exited rc={rc}" if not passed
                         else "ctest -L simple passed")
    detail = f"{detail} ({elapsed:.1f}s)"
    return ProbeResult(
        name="siesta ctest",
        passed=passed,
        detail=detail,
        output=_trim(out or ""),
    )


def _probe_elpa_make_check(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 3: ELPA's ``make check``.

    Per ELPA INSTALL.md the canonical "shipped ELPA is sane" test.
    Runs the small validators built alongside the library.  Uses
    ``-k`` so one flake doesn't mask other failures.  Output streams
    live so the user sees per-validator pass/fail as they run.
    """
    build_dir = _stack_root(env_prefix, recipe) / "build" / "elpa"
    if not (build_dir / "Makefile").is_file():
        return ProbeResult(
            name="elpa make check",
            passed=False,
            detail=f"build dir missing Makefile at {build_dir}",
        )
    env_make = Path(env_prefix) / "bin" / "make"
    make = str(env_make) if env_make.is_file() else shutil.which("make")
    if make is None:
        return ProbeResult(
            name="elpa make check",
            passed=False,
            detail="make not found in env or on PATH",
        )
    start = time.monotonic()
    # CHECK_LEVEL=fast is the upstream knob for "smoke test only".
    # ``-j1`` forces ELPA's automake parallel-tests driver to run
    # tests SERIALLY -- each validator already spawns its own MPI
    # ranks for the eigenproblem, so running tests concurrently
    # multiplies CPU/GPU load and triggers oversubscription on
    # workstations.  ``MAKEFLAGS=-j1`` in the env defends against
    # the user's shell having ``-jN`` set globally.
    # Timeout 1800 s (30 min): realistic for 300+ validators on a
    # consumer GPU; the previous 900 s was based on an off-by-2OOM
    # research-summary estimate.
    rc, out = _builds.run_streaming(
        [make, "-j1", "check", "CHECK_LEVEL=fast", "-k"],
        cwd=build_dir,
        env=dict(os.environ, MAKEFLAGS="-j1"),
        timeout=1800,
    )
    elapsed = time.monotonic() - start
    passed = (rc == 0)
    out = out or ""
    # ELPA's check output ends in a "Testsuite summary" block on success;
    # on fail it prints "TOTAL: N | FAIL: M".  Either way, count PASS/FAIL
    # lines for a stable summary.
    n_pass = out.count("\nPASS:")
    n_fail = out.count("\nFAIL:")
    summary = f"{n_pass} PASS / {n_fail} FAIL"
    detail = f"{summary} ({elapsed:.1f}s)"
    return ProbeResult(
        name="elpa make check",
        passed=passed,
        detail=detail,
        output=_trim(out),
    )


def _probe_elpa_gpu_codepath(env_prefix: str,
                             recipe: Recipe) -> ProbeResult:
    """Probe 4: THE one that catches silent CPU fallback.

    Invokes one GPU-flavoured ELPA validator (1stage real-double on a
    1000x1000 random matrix) and asserts the stderr does NOT contain
    the verbatim warning ELPA prints when the GPU codepath wasn't
    actually used.  Without this check, an A100 system with a clean
    nvidia-smi can still silently fall back to CPU for every SIESTA
    SCF step -- and the only visible symptom is "DFT runs the same
    speed as the CPU env".

    Source of the warning string: ``src/elpa2/elpa2_template.F90`` in
    upstream ELPA; reported behaviour: elpa#15 (same A100 + sm_80
    build config we use).
    """
    build_dir = _stack_root(env_prefix, recipe) / "build" / "elpa"
    validator = build_dir / "validate_real_double_eigenvectors_1stage_gpu_random"
    if not validator.is_file():
        return ProbeResult(
            name="elpa gpu codepath",
            passed=False,
            detail=f"validator binary missing at {validator}",
        )
    # ELPA's GPU validators require >= 2 MPI ranks (the runtime
    # explicitly aborts with "must be run with more than 1 task" when
    # launched bare).  Use the env's mpirun -- both ranks share the
    # one GPU on a typical workstation, which ELPA handles fine.
    env_mpirun = Path(env_prefix) / "bin" / "mpirun"
    mpirun = str(env_mpirun) if env_mpirun.is_file() else shutil.which("mpirun")
    if mpirun is None:
        return ProbeResult(
            name="elpa gpu codepath",
            passed=False,
            detail="mpirun not found in env or on PATH",
        )
    # 1000 / 500 / 16 -- small enough to run in <5 s, large enough to
    # exercise the kernel meaningfully.  argv order is "na nev nblk"
    # per ELPA test/Fortran/test.F90:read_input_parameters_traditional.
    # ``--oversubscribe`` lets us run 2 ranks on a single-socket box
    # without conda's mpirun complaining about CPU count.
    rc, out = _run(
        [mpirun, "--oversubscribe", "-n", "2", str(validator),
         "1000", "500", "16"],
        cwd=build_dir,
        timeout=120,
    )
    fallback_hit = _ELPA_GPU_FALLBACK_RE.search(out) is not None
    if rc != 0:
        return ProbeResult(
            name="elpa gpu codepath",
            passed=False,
            detail=f"validator exited rc={rc}",
            output=_trim(out),
        )
    if fallback_hit:
        return ProbeResult(
            name="elpa gpu codepath",
            passed=False,
            detail=("silent CPU fallback warning detected -- "
                    "GPU codepath NOT exercised at runtime"),
            output=_trim(out),
        )
    # Positive signal -- the validator reports its kernel choice in
    # the banner.  Pull it out as the detail line for transparency.
    kernel_hint = ""
    for line in out.splitlines():
        if "kernel" in line.lower() and "gpu" in line.lower():
            kernel_hint = line.strip()
            break
    detail = ("GPU codepath exercised (no fallback warning); "
              + (kernel_hint or "1stage real-double 1000x1000 passed"))
    return ProbeResult(
        name="elpa gpu codepath",
        passed=True,
        detail=detail,
        output=_trim(out),
    )


def _probe_mps_available(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 6: NVIDIA MPS daemon binary present on host PATH.

    MPS (Multi-Process Service) is what lets >= 2 MPI ranks share one
    GPU concurrently for ELPA-GPU diag, since our ELPA tag
    (2021.11.001) has no NCCL.  Without MPS, multi-rank ELPA-GPU runs
    serialise on the CUDA driver context and the second rank pays
    serialisation overhead with no benefit -- which is why the wrapper
    auto-caps to ``mpi_np=2`` without MPS and to ``mpi_np=4`` with it.

    The ``nvidia-cuda-mps-control`` binary ships with the **NVIDIA
    host driver** (same package as ``nvidia-smi``).  It is NOT a
    conda package -- there is no `cuda-mps` or similar in conda-forge;
    the daemon is kernel-driver-coupled and lives on the host side.
    If ``nvidia-smi`` works (probe ``cuda stack``) and MPS is missing,
    the user's driver was installed without the MPS server -- some
    distro packages split it out (eg. Debian's
    ``nvidia-cuda-mps`` package).  Detail tells them where to get it.

    Soft fail: env still WORKS without MPS -- single-rank ELPA-GPU
    runs perfectly fine, and multi-rank runs auto-fall-back at the
    wrapper.  Marked FAIL only to surface it in the report; the
    wrapper handles the absence gracefully.
    """
    mps_ctrl = shutil.which("nvidia-cuda-mps-control")
    if mps_ctrl is None:
        return ProbeResult(
            name="mps daemon",
            passed=False,
            detail=("nvidia-cuda-mps-control not found on host PATH "
                    "-- multi-rank GPU runs will lose concurrency "
                    "(wrapper auto-caps to mpi_np=2); install via "
                    "host NVIDIA driver / `nvidia-cuda-mps` distro pkg"),
        )
    # Don't actually START the daemon (it would conflict with the
    # wrapper's per-job pipe dir, and the user may already have one
    # running for another workload).  ``-V`` just prints version and
    # exits -- safe + cheap.
    rc, out = _run([mps_ctrl, "-V"], timeout=5)
    if rc != 0:
        # The binary exists but won't run -- broken install.  Still
        # mark as fail but report the rc for diagnostics.
        return ProbeResult(
            name="mps daemon",
            passed=False,
            detail=f"nvidia-cuda-mps-control -V exited rc={rc}",
            output=_trim(out),
        )
    # First non-blank line of -V output is the version banner.
    version_line = next(
        (ln.strip() for ln in out.splitlines() if ln.strip()),
        "(version banner not parsed)",
    )
    return ProbeResult(
        name="mps daemon",
        passed=True,
        detail=f"MPS available -- {version_line}",
        output=_trim(out),
    )


def _probe_cuda_stack(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 5: CUDA driver + libcuda.so.1 loadability.

    SIESTA does NOT print the CUDA toolkit version at startup, so an
    explicit probe is required to catch the "driver too old for
    toolkit" case before the user's first DFT run.  Driver lives on
    the host (kernel-coupled); libcuda.so.1 must be findable via
    LD_LIBRARY_PATH set by the host driver install.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return ProbeResult(
            name="cuda stack",
            passed=False,
            detail="nvidia-smi not found on host PATH",
        )
    rc, out = _run(
        [nvidia_smi,
         "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader"],
        timeout=15,
    )
    if rc != 0 or not out.strip():
        return ProbeResult(
            name="cuda stack",
            passed=False,
            detail=f"nvidia-smi rc={rc}",
            output=_trim(out),
        )
    first_gpu = out.splitlines()[0].strip()
    # libcuda.so.1 lives in /usr/lib/x86_64-linux-gnu/ (driver pkg);
    # if it can't be dlopen'd, SIESTA's CUDA-linked code will segfault
    # on first call.
    py = Path(env_prefix) / "bin" / "python"
    if not py.is_file():
        py_str = shutil.which("python") or "python"
    else:
        py_str = str(py)
    rc2, out2 = _run(
        [py_str, "-c", "import ctypes; ctypes.CDLL('libcuda.so.1'); print('ok')"],
        timeout=15,
    )
    if rc2 != 0 or "ok" not in out2:
        return ProbeResult(
            name="cuda stack",
            passed=False,
            detail="libcuda.so.1 not loadable via ctypes",
            output=_trim(out + "\n---\n" + out2),
        )
    return ProbeResult(
        name="cuda stack",
        passed=True,
        detail=first_gpu,
        output=_trim(out),
    )


# --------------------------------------------------------------------- #
#  Public entry point                                                    #
# --------------------------------------------------------------------- #


# Dispatch table: recipe name -> ordered list of
# ``(slug, callable, runtime_hint)`` for each probe.
#
# * ``slug`` is the stable short identifier shown in the progress
#   markers ("[2/5] siesta ctest: starting ...").  It must match the
#   ``name`` field the probe writes into its ``ProbeResult``.
# * ``runtime_hint`` is the user-facing "~Ns" estimate so the terminal
#   is never silent for longer than the user expects.  Without these,
#   the slow probes (ctest, make check) look hung even when they're
#   making forward progress.
#
# Hardcoded rather than a recipe field because only one recipe has a
# validator today (siesta-gpu) and the probes are tightly coupled to
# its build-tree layout.  When another recipe needs a validator we'll
# extract a per-recipe ``validate_argv`` schema.
_RECIPE_PROBES = {
    "molbuilder-siesta-gpu": (
        # Order: cheap probes first so a misconfigured env fails fast
        # without burning time on the long probes before noticing
        # missing binaries.  Runtime hints come from MEASURED times on a
        # workstation GPU build (NVIDIA RTX 3060 Ti, sm_86).
        ("binary-links",       _probe_binary_links,         "~2s -- siesta + tbtrans + phtrans present"),
        ("cuda stack",         _probe_cuda_stack,           "~1s -- nvidia-smi + libcuda dlopen"),
        ("mps daemon",         _probe_mps_available,        "~1s -- nvidia-cuda-mps-control on PATH"),
        ("elpa gpu codepath",  _probe_elpa_gpu_codepath,    "~5s -- GPU validator + silent-fallback grep"),
        ("siesta ctest",       _probe_siesta_ctest_simple,  "~2min -- SIESTA -L simple (~90 tests, streams live)"),
        ("elpa make check",    _probe_elpa_make_check,      "~15-30min -- 300+ ELPA validators; PASS/SKIP "
                                                            "lines stream as tests complete; silent gaps "
                                                            "of 30-120s between markers are NORMAL "
                                                            "(each validator does GPU work in block-"
                                                            "buffered silence before printing its result)"),
    ),
}


def has_validator(recipe_name: str) -> bool:
    return recipe_name in _RECIPE_PROBES


def validate_recipe(recipe_name: str, env_prefix: str,
                    *, quiet: bool = False) -> ValidationReport:
    """Run the validator suite for ``recipe_name`` against an installed
    env at ``env_prefix``.  Returns a :class:`ValidationReport` even on
    partial failure so the CLI can render every probe's result.

    Unless ``quiet=True``, prints per-probe start/done markers to
    ``sys.stderr`` so the user sees forward progress while the longer
    probes (ctest, make check) run.  The slow probes additionally
    stream their subprocess output live -- see
    ``_probe_siesta_ctest_simple`` and ``_probe_elpa_make_check``.
    """
    recipe = recipe_by_name(recipe_name)
    if recipe is None:
        raise ValueError(f"unknown recipe `{recipe_name}`")
    probes = _RECIPE_PROBES.get(recipe_name)
    if probes is None:
        return ValidationReport(
            recipe_name=recipe_name,
            env_prefix=env_prefix,
            probes=(),
        )
    if recipe.build_spec is None:
        raise ValueError(
            f"recipe `{recipe_name}` has no build_spec -- the source-"
            f"build dirs needed by the validator don't exist"
        )
    total = len(probes)
    results: List[ProbeResult] = []
    for i, (slug, probe, hint) in enumerate(probes, start=1):
        if not quiet:
            sys.stderr.write(f"[{i}/{total}] {slug}: starting ({hint})\n")
            sys.stderr.flush()
        start = time.monotonic()
        result = probe(env_prefix, recipe)
        elapsed = time.monotonic() - start
        results.append(result)
        if not quiet:
            tag = "PASS" if result.passed else "FAIL"
            sys.stderr.write(
                f"[{i}/{total}] {slug}: {tag} ({elapsed:.1f}s) "
                f"-- {result.detail}\n"
            )
            sys.stderr.flush()
    return ValidationReport(
        recipe_name=recipe_name,
        env_prefix=env_prefix,
        probes=tuple(results),
    )


__all__ = [
    "ProbeResult",
    "ValidationReport",
    "has_validator",
    "validate_recipe",
]
