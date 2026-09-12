"""L1 tests for ``molbuilder.envs.validate``'s verdict.

`validate.py` had 598 lines of probe logic and **no test file at all** --
including the one probe two live engine docs name as the only canary for
ELPA's silent CPU fallback.  This covers the verdict, which is the part a
person acts on; the probes themselves need a built GPU env and ~30 minutes.
"""
from __future__ import annotations

import pytest

from molbuilder.envs import _cli as envs_cli
from molbuilder.envs.validate import (ProbeResult, ValidationReport,
                                      _RECIPE_PROBES)


def _report(*probes: ProbeResult) -> ValidationReport:
    return ValidationReport("molbuilder-siesta-gpu", "/fake/env", probes)


_OK = ProbeResult("binary-links", True, "siesta 5.4.2")
_MPS_ABSENT = ProbeResult("mps daemon", False,
                          "nvidia-cuda-mps-control not on PATH",
                          advisory=True)
_REAL_FAIL = ProbeResult("elpa gpu codepath", False,
                         "GPU requested but kernel is non-GPU")


def test_an_advisory_failure_does_not_condemn_a_working_env():
    """MPS being absent is not a broken env, and must not report as one.

    `_probe_mps_available`'s own docstring says "env still WORKS without MPS
    ... Marked FAIL only to surface it in the report" -- but the verdict
    required EVERY probe to pass, so a CPU-fine GPU-fine workstation whose
    distro simply splits out `nvidia-cuda-mps` was told "env not
    production-ready" and `validate` exited 1.
    """
    report = _report(_OK, _MPS_ABSENT)
    assert report.all_passed is True
    assert envs_cli._render_validation(report, show_output_on_fail=False) == 0


def test_a_real_failure_still_condemns_the_env():
    """The exemption is per-PROBE, not a general softening."""
    report = _report(_OK, _MPS_ABSENT, _REAL_FAIL)
    assert report.all_passed is False
    assert envs_cli._render_validation(report, show_output_on_fail=False) == 1


def test_no_probes_is_not_a_pass():
    """An empty suite must not read as a clean bill of health."""
    assert _report().all_passed is False


def test_the_long_upstream_suites_are_not_optional():
    """There is deliberately no way to skip them -- see the module docstring.

    A source build is verified by nobody upstream: it is compiled here
    against this host's CUDA, gcc and MPI, so upstream's own suite is the
    only evidence the binary computes correctly.  A knob was drafted and
    withdrawn 2026-09-12.  This test exists so re-adding one is a decision
    rather than a drift.
    """
    import inspect
    from molbuilder.envs import validate as V

    params = inspect.signature(V.validate_recipe).parameters
    for knob in ("skip", "only", "quick", "deep"):
        assert knob not in params, (
            f"`{knob}` would let a source build go unverified -- read the "
            f"module docstring before adding it")
    # And every registered probe is still in the list the CLI runs.
    slugs = [row[0] for row in _RECIPE_PROBES["molbuilder-siesta-gpu"]]
    for required in ("siesta ctest", "elpa make check"):
        assert required in slugs, f"{required} must stay in the suite"


@pytest.mark.parametrize("slug,fn,hint", _RECIPE_PROBES["molbuilder-siesta-gpu"])
def test_every_probe_row_carries_a_runtime_hint(slug, fn, hint):
    """The hint is what the CLI prints so the terminal is never silent for
    longer than a person expects -- the docstring's claim of "~2 min" for a
    ~30 min suite is exactly what a missing hint lets happen."""
    assert callable(fn)
    assert hint.strip(), f"{slug} has no runtime hint"
    assert any(u in hint for u in ("s ", "s -", "min")), (
        f"{slug}'s hint states no time: {hint!r}")
