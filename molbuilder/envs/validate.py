"""Post-install validation probes for env recipes.

Where :mod:`molbuilder.envs.doctor` answers "is the env present and
does the binary launch?", this module answers the next question:
**"does the env actually compute correctly?"**

For ``molbuilder-siesta-gpu`` -- the only recipe with a validator
today -- this runs five small probes (~40 s total) that catch the
failure modes ``siesta --version`` cannot:

  1. binary-link sanity      siesta/tbtrans/phtrans present + version OK
  2. SIESTA ctest -L simple  the upstream "binary actually runs SCF" set
  3. ELPA make check         the upstream eigensolver self-test
  4. ELPA GPU codepath       grep for the silent-CPU-fallback warning
  5. CUDA stack              nvidia-smi + libcuda.so.1 ctypes load

The load-bearing one is #4: ``nvidia-smi`` can report a perfectly
healthy A100 while ELPA silently runs on the CPU (elpa#15, same
A100 + ``--enable-nvidia-sm80-gpu`` configuration we ship).  None of
the other probes catches that.

Skipped on purpose:
  * deviceQuery: not in conda CUDA packages, redundant with #5.
  * CPU-vs-GPU energy cross-check at ~1e-4 eV: too loose (real
    ELPA2 GPU agreement is ~1e-6 eV total; arXiv 2002.10991).
    Lives in ``validate --deep`` when we ship it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    required = ("siesta", "transiesta", "tbtrans")
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
    rc, out = _run(
        [ctest, "-E", "verify", "-L", "simple", "--output-on-failure"],
        cwd=build_dir,
        timeout=120,
    )
    elapsed = time.monotonic() - start
    # ctest's tail line is the most reliable parse target.
    summary = ""
    for line in out.splitlines():
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
        output=_trim(out),
    )


def _probe_elpa_make_check(env_prefix: str, recipe: Recipe) -> ProbeResult:
    """Probe 3: ELPA's ``make check``.

    Per ELPA INSTALL.md the canonical "shipped ELPA is sane" test.
    Runs the small validators built alongside the library.  Uses
    ``-k`` so one flake doesn't mask other failures.
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
    rc, out = _run(
        [make, "check", "CHECK_LEVEL=fast", "-k"],
        cwd=build_dir,
        timeout=300,
    )
    elapsed = time.monotonic() - start
    passed = (rc == 0)
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
    # 1000 / 500 / 16 -- small enough to run in <5 s, large enough to
    # exercise the kernel meaningfully.  argv order is "na nev nblk"
    # per ELPA test/Fortran/test.F90:read_input_parameters_traditional.
    rc, out = _run(
        [str(validator), "1000", "500", "16"],
        cwd=build_dir,
        timeout=60,
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


# Dispatch table: recipe name -> list of probe callables.  Hardcoded
# rather than a recipe field because only one recipe has a validator
# today (siesta-gpu) and the probes are tightly coupled to its
# build-tree layout.  When another recipe needs a validator we'll
# extract a per-recipe ``validate_argv`` schema.
_RECIPE_PROBES = {
    "molbuilder-siesta-gpu": (
        _probe_binary_links,
        _probe_siesta_ctest_simple,
        _probe_elpa_make_check,
        _probe_elpa_gpu_codepath,
        _probe_cuda_stack,
    ),
}


def has_validator(recipe_name: str) -> bool:
    return recipe_name in _RECIPE_PROBES


def validate_recipe(recipe_name: str, env_prefix: str) -> ValidationReport:
    """Run the validator suite for ``recipe_name`` against an installed
    env at ``env_prefix``.  Returns a :class:`ValidationReport` even on
    partial failure so the CLI can render every probe's result.
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
    results: List[ProbeResult] = []
    for probe in probes:
        results.append(probe(env_prefix, recipe))
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
