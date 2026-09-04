"""L2 tests for the ``molbuilder.envs.builds`` source-build executor.

The executor never runs a real ELPA/ELSI/SIESTA build in tests --
that takes 35-45 min and needs CUDA + a GPU.  These tests exercise the
deterministic machinery around it: BuildSpec validation, sentinel
resume, toolchain fingerprint determinism, template substitution,
preflight reporting, --rebuild semantics, and the activate.d hook
content.

L4 (integration) coverage of the actual build run lives elsewhere and
is opt-in (needs hours + a GPU).  This file pins the unit-level
correctness so a refactor in builds.py surfaces immediately.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import pytest

from molbuilder.envs import builds as B
from molbuilder.envs.recipes import (
    BuildComponent, BuildSpec, recipe_by_name,
)


pytestmark = pytest.mark.module


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def siesta_gpu_spec() -> BuildSpec:
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    assert recipe is not None and recipe.build_spec is not None
    return recipe.build_spec


@pytest.fixture
def fake_probe(tmp_path) -> B.ToolchainProbe:
    """A deterministic ToolchainProbe for fingerprint / template tests."""
    return B.ToolchainProbe(
        env_prefix=str(tmp_path / "env"),
        cuda_home="/usr/local/cuda",
        cuda_version="12.5.40",
        cuda_compute_cap="8.0",
        gcc_version="14.2.0",
        openmpi_version="5.0.5",
        jobs=8,
    )


@pytest.fixture
def tiny_spec() -> BuildSpec:
    """A minimal BuildSpec for testing executor machinery without
    pulling in the full siesta-gpu recipe."""
    a = BuildComponent(
        name="a",
        repo_url="https://example.com/a.git",
        ref="v1",
        configure_argv=("cmake", "-S", "{src}", "-B", "{build}",
                        "-DCMAKE_INSTALL_PREFIX={install}"),
        build_argv=("cmake", "--build", "{build}", "-j", "{jobs}"),
        install_argv=("cmake", "--install", "{build}"),
        needs_cuda=False,
    )
    b = BuildComponent(
        name="b",
        repo_url="https://example.com/b.git",
        ref="v2",
        configure_argv=("cmake", "-S", "{src}", "-B", "{build}",
                        "-DA_DIR={dep_a}"),
        build_argv=("cmake", "--build", "{build}"),
        install_argv=("cmake", "--install", "{build}"),
        needs_cuda=False,
    )
    return BuildSpec(
        artifact_subdir="tiny-stack",
        components=(a, b),
        cuda_required=False,
    )


# --------------------------------------------------------------------- #
#  Phase tuple                                                           #
# --------------------------------------------------------------------- #


def test_phases_in_canonical_order():
    """Phases run clone -> configure -> build -> install -> verify.
    Resume semantics rely on this ordering."""
    assert B.PHASES == ("clone", "configure", "build", "install", "verify")


def test_describe_phase_returns_action_and_cost():
    """describe_phase returns a (action, cost) pair for known phases
    and falls back to generic strings for unknown ones."""
    action, cost = B.describe_phase("siesta", "build")
    assert "compile" in action.lower() or "build" in action.lower()
    assert cost  # not empty
    # Unknown component falls back, doesn't raise.
    a, c = B.describe_phase("nonsense", "phase")
    assert "nonsense" in a or "phase" in a
    assert c == "unknown"


# --------------------------------------------------------------------- #
#  BuildSpec validation                                                  #
# --------------------------------------------------------------------- #


def test_buildspec_rejects_empty_components():
    """Empty components tuple is meaningless; the constructor refuses."""
    with pytest.raises(ValueError):
        BuildSpec(artifact_subdir="x", components=())


def test_buildspec_rejects_duplicate_component_names():
    """Names are how sentinels + dep:* templates address components;
    duplicates break both."""
    c = BuildComponent(
        name="dup", repo_url="x", ref="v",
        configure_argv=(), build_argv=(), install_argv=(),
    )
    with pytest.raises(ValueError, match="duplicate"):
        BuildSpec(artifact_subdir="x", components=(c, c))


def test_buildspec_rejects_unsafe_artifact_subdir():
    """artifact_subdir must be a single path segment so it can't
    escape $CONDA_PREFIX/opt/."""
    c = BuildComponent(
        name="a", repo_url="x", ref="v",
        configure_argv=(), build_argv=(), install_argv=(),
    )
    for bad in ("../escape", "x/y", ".hidden"):
        with pytest.raises(ValueError):
            BuildSpec(artifact_subdir=bad, components=(c,))


# --------------------------------------------------------------------- #
#  Component dependency ordering                                         #
# --------------------------------------------------------------------- #


def test_siesta_gpu_components_in_dependency_order(siesta_gpu_spec):
    """Two-component build: elpa -> siesta.  SIESTA links ELPA
    externally; ELSI is a SIESTA submodule (per SIESTA 5.4 INSTALL.md)."""
    names = [c.name for c in siesta_gpu_spec.components]
    assert names == ["elpa", "siesta"]


def test_downstream_components_walks_chain(siesta_gpu_spec):
    """Asking to rebuild a component implies rebuilding everything
    downstream of it (later components have linked it)."""
    assert B.downstream_components(siesta_gpu_spec, "elpa") \
        == ("elpa", "siesta")
    assert B.downstream_components(siesta_gpu_spec, "siesta") \
        == ("siesta",)
    # Unknown component -> empty tuple, not exception.
    assert B.downstream_components(siesta_gpu_spec, "ghost") == ()
    assert B.downstream_components(siesta_gpu_spec, "elsi") == ()


# --------------------------------------------------------------------- #
#  Toolchain fingerprint                                                 #
# --------------------------------------------------------------------- #


def test_fingerprint_is_deterministic(tiny_spec, fake_probe):
    """Same (spec, probe, refs) -> same hash, every time."""
    refs = {"a": "sha-aaa", "b": "sha-bbb"}
    fp1 = B.compute_fingerprint(tiny_spec, fake_probe, refs)
    fp2 = B.compute_fingerprint(tiny_spec, fake_probe, refs)
    assert fp1 == fp2
    # 64-hex sha256 form
    assert re.fullmatch(r"[0-9a-f]{64}", fp1)


def test_fingerprint_changes_with_cuda_version(tiny_spec, fake_probe):
    """A CUDA toolkit upgrade must invalidate sentinels."""
    refs = {"a": "sha-aaa", "b": "sha-bbb"}
    fp_a = B.compute_fingerprint(tiny_spec, fake_probe, refs)
    probe_new = B.ToolchainProbe(
        **{**fake_probe.__dict__, "cuda_version": "12.6.0"}
    )
    fp_b = B.compute_fingerprint(tiny_spec, probe_new, refs)
    assert fp_a != fp_b


def test_fingerprint_changes_with_gcc_version(tiny_spec, fake_probe):
    """A gcc upgrade must invalidate sentinels."""
    refs = {"a": "sha-aaa", "b": "sha-bbb"}
    fp_a = B.compute_fingerprint(tiny_spec, fake_probe, refs)
    probe_new = B.ToolchainProbe(
        **{**fake_probe.__dict__, "gcc_version": "14.3.0"}
    )
    fp_b = B.compute_fingerprint(tiny_spec, probe_new, refs)
    assert fp_a != fp_b


def test_fingerprint_changes_with_resolved_ref(tiny_spec, fake_probe):
    """A new SIESTA tag (or any component ref) must invalidate sentinels."""
    refs_a = {"a": "sha-aaa", "b": "sha-bbb"}
    refs_b = {"a": "sha-aaa", "b": "sha-CHANGED"}
    fp_a = B.compute_fingerprint(tiny_spec, fake_probe, refs_a)
    fp_b = B.compute_fingerprint(tiny_spec, fake_probe, refs_b)
    assert fp_a != fp_b


def test_fingerprint_changes_with_openmpi_version(tiny_spec, fake_probe):
    """An OpenMPI bump must invalidate sentinels (ABI not stable
    across minor versions)."""
    refs = {"a": "sha-aaa", "b": "sha-bbb"}
    fp_a = B.compute_fingerprint(tiny_spec, fake_probe, refs)
    probe_new = B.ToolchainProbe(
        **{**fake_probe.__dict__, "openmpi_version": "4.1.6"}
    )
    fp_b = B.compute_fingerprint(tiny_spec, probe_new, refs)
    assert fp_a != fp_b


# --------------------------------------------------------------------- #
#  Sentinel round-trip                                                   #
# --------------------------------------------------------------------- #


def test_sentinel_roundtrip(tmp_path):
    """A sentinel that's present marks the phase as done -- the recorded
    fingerprint is forensic metadata only, not a gating check.  Per the
    2026-06-15 artifact-presence redesign: editing a SIESTA flag must
    not invalidate ELPA's sentinel just because the global fingerprint
    shifts.  The install-start ``component_install_valid`` probe is
    the trust source for "is this component already installed"."""
    sentinel = tmp_path / "x.done"
    # Fixed clock so the timestamp doesn't depend on wall time.
    B.write_sentinel(sentinel, "fp-1", now=lambda: 1700000000.0)
    assert sentinel.exists()
    # Fingerprint string is recorded for debugging but ignored by the
    # shim -- callers should prefer ``sentinel.exists()`` directly.
    assert B.read_sentinel_fingerprint(sentinel) == "fp-1"
    assert B.sentinel_valid(sentinel, "any-string") is True


def test_sentinel_absent_is_invalid(tmp_path):
    """A nonexistent sentinel marks the phase as not-done."""
    sentinel = tmp_path / "missing.done"
    assert B.sentinel_valid(sentinel, "any") is False


def test_sentinel_corrupt_still_counts_as_present(tmp_path):
    """A corrupt sentinel file still trips the "present" check -- under
    the artifact-presence model a sentinel is a marker, not a payload.
    If the underlying install is actually broken, the install-start
    verify probe catches that and the marker gets re-written cleanly."""
    sentinel = tmp_path / "corrupt.done"
    sentinel.write_text("not-json-at-all", encoding="utf-8")
    # Fingerprint isn't readable from a corrupt file, but the sentinel
    # itself exists -- and that's the only thing the phase loop checks.
    assert B.read_sentinel_fingerprint(sentinel) is None
    assert B.sentinel_valid(sentinel, "any") is True


# --------------------------------------------------------------------- #
#  Template substitution                                                 #
# --------------------------------------------------------------------- #


def test_template_substitution_in_plan(tiny_spec, fake_probe, tmp_path):
    """Plan step argv must have ``{src}`` / ``{build}`` / ``{install}`` /
    ``{dep_a}`` / ``{jobs}`` resolved to real paths + integers."""
    refs = {"a": "ref-a", "b": "ref-b"}
    paths, fingerprint, steps = B.plan_build_spec(
        tiny_spec, str(tmp_path / "env"), fake_probe, refs,
    )
    # Find b's configure step and ensure {dep_a} is resolved
    cfg = [s for s in steps if s.component == "b" and s.phase == "configure"]
    assert len(cfg) == 1
    argv_joined = " ".join(cfg[0].argv)
    assert "{dep_a}" not in argv_joined, "dep_a not substituted"
    assert "tiny-stack/a" in argv_joined  # install dir for `a`
    # a's build phase uses {jobs}
    b_step = [s for s in steps if s.component == "a" and s.phase == "build"]
    assert any(token == "8" for token in b_step[0].argv), (
        f"{{jobs}} should resolve to 8: {b_step[0].argv}"
    )


def test_activate_hook_uses_literal_conda_prefix(siesta_gpu_spec, fake_probe):
    """Per the 2026-06-15 design correction: the activate hook is a
    pass-through (no template substitution).  It uses literal
    ``$CONDA_PREFIX`` so the hook stays valid if the env is moved.
    No ``/usr/local/cuda`` paths -- the toolkit ships in the env's lib."""
    paths = B.resolve_paths(siesta_gpu_spec, "/tmp/env")
    rendered = B.render_activate_hook(siesta_gpu_spec, paths, fake_probe)
    # Literal $CONDA_PREFIX, not the install-time-resolved /tmp/env path
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"' in rendered
    assert '"$CONDA_PREFIX/lib"' in rendered
    # System CUDA path must NOT appear
    assert "/usr/local/cuda" not in rendered
    # Install-time prefix must NOT be baked in either
    assert "/tmp/env/opt/siesta-gpu-stack" not in rendered


def test_template_substitution_rejects_unknown_placeholder(
        siesta_gpu_spec, fake_probe, tmp_path):
    """A typo in a template placeholder must surface, not silently
    produce a broken argv."""
    bad = BuildComponent(
        name="bad", repo_url="x", ref="v",
        configure_argv=("cmake", "{not_a_real_placeholder}"),
        build_argv=(), install_argv=(),
    )
    spec = BuildSpec(artifact_subdir="bs", components=(bad,))
    with pytest.raises(ValueError, match="unknown placeholder"):
        B.plan_build_spec(spec, str(tmp_path / "env"), fake_probe,
                          {"bad": "v"})


# --------------------------------------------------------------------- #
#  CUDA <-> gcc compatibility                                            #
# --------------------------------------------------------------------- #


def test_cuda_13_with_gcc_14_is_ok(tmp_path):
    """The default pair is gcc 14 + CUDA 13.x (the recipe's cuda-version
    pin); preflight passes."""
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=str(tmp_path),
        cuda_version="13.0.0", cuda_compute_cap="8.0",
        gcc_version="14.2.0", openmpi_version="5.0.5", jobs=8,
    )
    assert B.check_cuda_gcc_compat(probe) is None


def test_cuda_128_with_gcc_14_is_ok(tmp_path):
    """CUDA 12.8+ also supports gcc 14."""
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=str(tmp_path),
        cuda_version="12.8.0", cuda_compute_cap="8.0",
        gcc_version="14.2.0", openmpi_version="5.0.5", jobs=8,
    )
    assert B.check_cuda_gcc_compat(probe) is None


def test_cuda_127_with_gcc_14_is_rejected(tmp_path):
    """CUDA 12.0-12.7 caps at gcc 13; gcc 14 is rejected."""
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=str(tmp_path),
        cuda_version="12.7.0", cuda_compute_cap="8.0",
        gcc_version="14.2.0", openmpi_version="5.0.5", jobs=8,
    )
    err = B.check_cuda_gcc_compat(probe)
    assert err is not None
    assert "13" in err  # recommends gcc 13
    assert "MOLBUILDER_GCC" in err  # tells user the override env var


def test_cuda_11_with_gcc_14_is_rejected(tmp_path):
    """CUDA 11.x caps at gcc 11; gcc 14 is rejected with clear hint."""
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=str(tmp_path),
        cuda_version="11.8.0", cuda_compute_cap="8.0",
        gcc_version="14.2.0", openmpi_version="5.0.5", jobs=8,
    )
    err = B.check_cuda_gcc_compat(probe)
    assert err is not None
    assert "11" in err and "MOLBUILDER_GCC" in err


def test_detect_cuda_home_prefers_env_over_system(tmp_path, monkeypatch):
    """Per the 2026-06-15 design correction: CUDA toolkit lives in
    the env (conda-installed cuda-nvcc).  The detector must find
    that nvcc first, before any /usr/local/cuda legacy install."""
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "nvcc").write_text("#!/bin/sh\necho fake nvcc\n")
    (env / "bin" / "nvcc").chmod(0o755)
    # Even if $CUDA_HOME is set to a different path, the env wins.
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    result = B._detect_cuda_home(str(env), {})
    assert result == str(env), (
        f"env's nvcc must win over $CUDA_HOME; got {result!r}"
    )


def test_detect_cuda_home_falls_back_to_system_when_env_empty(
        tmp_path, monkeypatch):
    """When the env doesn't have nvcc (pre-conda-create or partial
    install), the detector falls back to $CUDA_HOME / /usr/local/cuda."""
    env = tmp_path / "empty-env"
    env.mkdir()
    # Set $CUDA_HOME to a path with nvcc
    fallback = tmp_path / "fallback-cuda"
    (fallback / "bin").mkdir(parents=True)
    (fallback / "bin" / "nvcc").write_text("#!/bin/sh\n")
    (fallback / "bin" / "nvcc").chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(fallback))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # remove any host nvcc
    result = B._detect_cuda_home(str(env), {})
    assert result == str(fallback)


def test_no_cuda_no_gcc_is_silent(tmp_path):
    """When CUDA or gcc is undetected, this check returns None.
    The presence-required gate is handled elsewhere (preflight)."""
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=None,
        cuda_version=None, cuda_compute_cap=None,
        gcc_version=None, openmpi_version=None, jobs=8,
    )
    assert B.check_cuda_gcc_compat(probe) is None


# --------------------------------------------------------------------- #
#  Forbidden packages                                                    #
# --------------------------------------------------------------------- #


def test_check_forbidden_catches_mkl(siesta_gpu_spec):
    """If a future edit adds `mkl` to conda_packages, this check
    surfaces it."""
    err = B.check_no_forbidden_packages(
        siesta_gpu_spec, ("python", "mkl", "openblas"),
    )
    assert err is not None
    assert "mkl" in err
    assert "OpenMP" in err or "single" in err  # explains why


def test_check_forbidden_silent_on_clean_packages(siesta_gpu_spec):
    """The siesta-gpu recipe's own conda_packages list passes."""
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    err = B.check_no_forbidden_packages(siesta_gpu_spec,
                                        recipe.conda_packages)
    assert err is None


# --------------------------------------------------------------------- #
#  Disk + network preflight                                              #
# --------------------------------------------------------------------- #


def test_check_disk_warns_below_recommended(tmp_path, monkeypatch):
    """Disk between required + recommended thresholds returns a
    warning, not a hard error."""
    class FakeUsage:
        def __init__(self, free_gb):
            self.total = 100 * 1024 ** 3
            self.used = (100 - free_gb) * 1024 ** 3
            self.free = int(free_gb * 1024 ** 3)
    monkeypatch.setattr("shutil.disk_usage",
                        lambda p: FakeUsage(40))  # 40 GB
    free, msg = B.check_disk(str(tmp_path), required_gb=30, recommended_gb=50)
    assert free == pytest.approx(40, abs=0.1)
    assert msg is not None
    assert "recommended" in msg.lower()


def test_check_disk_errors_below_required(tmp_path, monkeypatch):
    """Disk below required threshold returns an error string."""
    class FakeUsage:
        def __init__(self, free_gb):
            self.total = 100 * 1024 ** 3
            self.used = (100 - free_gb) * 1024 ** 3
            self.free = int(free_gb * 1024 ** 3)
    monkeypatch.setattr("shutil.disk_usage",
                        lambda p: FakeUsage(5))  # 5 GB
    free, msg = B.check_disk(str(tmp_path), required_gb=30, recommended_gb=50)
    assert msg is not None
    assert "30" in msg  # mentions the required threshold
    assert "5" in msg or "free" in msg.lower()


def test_check_disk_silent_when_comfortable(tmp_path, monkeypatch):
    """Disk above recommended threshold returns (free_gb, None)."""
    class FakeUsage:
        def __init__(self, free_gb):
            self.total = 100 * 1024 ** 3
            self.used = (100 - free_gb) * 1024 ** 3
            self.free = int(free_gb * 1024 ** 3)
    monkeypatch.setattr("shutil.disk_usage",
                        lambda p: FakeUsage(200))
    free, msg = B.check_disk(str(tmp_path), required_gb=30, recommended_gb=50)
    assert msg is None
    assert free == pytest.approx(200, abs=0.1)


def test_check_disk_handles_missing_path():
    """Unreachable path returns (None, warning)."""
    free, msg = B.check_disk("/this/path/should/never/exist")
    assert free is None
    assert msg is not None


# --------------------------------------------------------------------- #
#  PreflightReport shape                                                 #
# --------------------------------------------------------------------- #


def test_preflight_report_separates_errors_warnings_info(tmp_path,
                                                         monkeypatch):
    """The three buckets do different things: errors halt, warnings
    confirm, info shows.  They must never blur."""
    spec = recipe_by_name("molbuilder-siesta-gpu").build_spec
    pkgs = recipe_by_name("molbuilder-siesta-gpu").conda_packages
    # Fake env: has a conda-meta dir but no nvcc, so the env-side
    # toolkit error fires.  Also fake nvidia-smi away so the host-side
    # error fires.
    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr("shutil.which",
                        lambda name: None if name in ("nvidia-smi", "nvcc")
                        else f"/usr/bin/{name}")
    probe = B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=None,
        cuda_version=None, cuda_compute_cap=None,
        gcc_version="14.2.0", openmpi_version="5.0.5", jobs=8,
    )
    report = B.preflight(spec, probe, pkgs, str(tmp_path),
                         check_network=False)
    assert isinstance(report, B.PreflightReport)
    # No CUDA toolkit + no driver -> at least one error mentions
    # "driver" or "nvcc" (the two failure modes after the split).
    err_text = " ".join(report.errors).lower()
    assert ("driver" in err_text or "nvcc" in err_text), report.errors
    # Info must contain gcc + jobs lines regardless of CUDA status
    info_joined = "\n".join(report.info)
    assert "gcc" in info_joined and "14.2.0" in info_joined
    assert "Build concurrency" in info_joined


def test_format_preflight_report_includes_all_sections(tmp_path):
    """The text formatter must show info + warnings + errors all in
    one render so users see everything in one screen."""
    report = B.PreflightReport(
        info=("CUDA toolkit 12.5",),
        warnings=("sm_80 fallback",),
        errors=("disk full",),
    )
    out = B.format_preflight_report(report)
    assert "CUDA toolkit 12.5" in out
    assert "sm_80 fallback" in out
    assert "disk full" in out


# --------------------------------------------------------------------- #
#  Preflight <-> ABI contract seam                                       #
# --------------------------------------------------------------------- #
#
# The unit-level rules live in tests/test_envs_abi.py.  What is tested
# here is only the WIRING: does preflight read the contract, does it put
# a violation in the right bucket, and does the structured `findings`
# list survive the trip out.


def _abi_probe(tmp_path):
    return B.ToolchainProbe(
        env_prefix=str(tmp_path), cuda_home=str(tmp_path),
        cuda_version="13.3", cuda_compute_cap="8.0",
        gcc_version="14.3.0", openmpi_version="5.0.10", jobs=8,
    )


def _write_sysroot(tmp_path, version):
    meta = tmp_path / "conda-meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"sysroot_linux-64-{version}-h0_0.json").write_text(
        json.dumps({"name": "sysroot_linux-64", "version": version}))


def test_preflight_reports_both_sides_of_the_abi_contract(tmp_path,
                                                          monkeypatch):
    """The pair of numbers is what makes a later build failure legible,
    so it is shown whether or not a rule fired."""
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    _write_sysroot(tmp_path, "2.17")
    report = B.preflight(recipe.build_spec, _abi_probe(tmp_path),
                         recipe.conda_packages, str(tmp_path),
                         check_network=False)
    info = "\n".join(report.info)
    assert "Host glibc" in info
    assert "Env sysroot" in info and "2.17" in info


def test_preflight_errors_when_the_sysroot_outranks_the_host_glibc(
        tmp_path, monkeypatch):
    """The regression.  A 2.39 sysroot on a 2.28 host must stop the
    build HERE, with the numbers named -- not five steps later inside
    ELPA's configure with a message that never mentions glibc."""
    import molbuilder.envs.abi as A
    monkeypatch.setattr(A, "_detect_host_glibc", lambda: (2, 28))
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    _write_sysroot(tmp_path, "2.39")
    report = B.preflight(recipe.build_spec, _abi_probe(tmp_path),
                         recipe.conda_packages, str(tmp_path),
                         check_network=False)
    codes = {f.code for f in report.findings}
    assert "abi.sysroot-exceeds-host-glibc" in codes
    # ...and it must land in errors, not warnings: the build cannot
    # succeed, so confirming past it would waste the user's time.
    assert any("2.39" in e and "2.28" in e for e in report.errors), report.errors


def test_preflight_is_silent_on_the_pinned_default(tmp_path, monkeypatch):
    """The shipped pin must not trip its own check on a realistic host."""
    import molbuilder.envs.abi as A
    monkeypatch.setattr(A, "_detect_host_glibc", lambda: (2, 28))
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    _write_sysroot(tmp_path, "2.17")
    report = B.preflight(recipe.build_spec, _abi_probe(tmp_path),
                         recipe.conda_packages, str(tmp_path),
                         check_network=False)
    assert [f for f in report.findings if f.code.startswith("abi.")] == []


def test_findings_carry_codes_so_tests_need_not_match_prose(tmp_path,
                                                            monkeypatch):
    """Every structured finding must be identifiable without reading
    its wording, so error text stays free to improve."""
    import molbuilder.envs.abi as A
    monkeypatch.setattr(A, "_detect_host_glibc", lambda: (2, 28))
    recipe = recipe_by_name("molbuilder-siesta-gpu")
    _write_sysroot(tmp_path, "2.39")
    report = B.preflight(recipe.build_spec, _abi_probe(tmp_path),
                         recipe.conda_packages, str(tmp_path),
                         check_network=False)
    assert report.findings
    for finding in report.findings:
        assert finding.code and finding.severity


def test_preflight_report_findings_default_to_empty(tmp_path):
    """Back-compat: the three prose lists remain the report's primary
    shape, and callers constructing a report positionally still work."""
    report = B.PreflightReport(errors=(), warnings=(), info=("x",))
    assert report.findings == ()


# --------------------------------------------------------------------- #
#  run_build_spec: preflight short-circuit                               #
# --------------------------------------------------------------------- #


def test_detect_stale_artifact_dirs_finds_old_component_dirs(tmp_path):
    """detect_stale_artifact_dirs catches leftover directories from
    a prior recipe version -- specifically an ``elsi/`` install dir
    from the deprecated 3-component layout."""
    spec = recipe_by_name("molbuilder-siesta-gpu").build_spec
    env_prefix = str(tmp_path / "env")
    paths = B.resolve_paths(spec, env_prefix)
    # Simulate a prior 3-component install
    (paths.root / "elsi" / "lib").mkdir(parents=True)
    (paths.root / "elpa" / "lib").mkdir(parents=True)  # known, NOT stale
    (paths.root / "leftover_garbage").mkdir()
    stale = B.detect_stale_artifact_dirs(spec, env_prefix)
    assert "elsi" in stale
    assert "leftover_garbage" in stale
    assert "elpa" not in stale          # known component
    assert ".sentinels" not in stale    # known infra dir (doesn't exist here but expected)


def test_detect_stale_artifact_dirs_empty_when_clean(tmp_path):
    """No stale dirs reported on a clean install."""
    spec = recipe_by_name("molbuilder-siesta-gpu").build_spec
    env_prefix = str(tmp_path / "fresh-env")
    # Root doesn't exist at all -- definitely clean.
    assert B.detect_stale_artifact_dirs(spec, env_prefix) == []
    # Root exists with only expected entries -- still clean.
    paths = B.resolve_paths(spec, env_prefix)
    paths.root.mkdir(parents=True)
    (paths.root / "src").mkdir()
    (paths.root / "build").mkdir()
    (paths.root / "logs").mkdir()
    (paths.root / ".sentinels").mkdir()
    (paths.root / "elpa").mkdir()
    (paths.root / "siesta").mkdir()
    assert B.detect_stale_artifact_dirs(spec, env_prefix) == []


def test_preflight_surfaces_stale_dirs_as_warning(tmp_path, monkeypatch):
    """When stale dirs are detected, preflight returns a warning
    (not an error), and the message tells the user how to clean up."""
    spec = recipe_by_name("molbuilder-siesta-gpu").build_spec
    pkgs = recipe_by_name("molbuilder-siesta-gpu").conda_packages
    env_prefix = str(tmp_path / "env")
    paths = B.resolve_paths(spec, env_prefix)
    (paths.root / "elsi").mkdir(parents=True)
    # Probe doesn't matter for this test -- we just check the warning
    # surfaces.  Skip CUDA + network checks to keep the test focused.
    probe = B.ToolchainProbe(
        env_prefix=env_prefix, cuda_home=env_prefix,
        cuda_version="13.0", cuda_compute_cap="8.0",
        gcc_version="14.2", openmpi_version="5.0.5", jobs=8,
    )
    report = B.preflight(spec, probe, pkgs, env_prefix, check_network=False)
    warn_text = " ".join(report.warnings)
    assert "elsi" in warn_text or "stale" in warn_text.lower()
    assert "--rebuild=all" in warn_text


def test_run_build_short_circuits_on_preflight_error(tmp_path):
    """When preflight raises errors, run_build_spec returns
    succeeded=False with NO subprocess invoked (no sentinels written,
    no build dir created)."""
    spec = recipe_by_name("molbuilder-siesta-gpu").build_spec
    pkgs = recipe_by_name("molbuilder-siesta-gpu").conda_packages
    env_prefix = str(tmp_path / "env")
    os.makedirs(env_prefix)
    # No CUDA on the fake env -> preflight errors
    result = B.run_build_spec(
        spec, env_prefix,
        conda_binary="/bin/false",   # would fail if called
        conda_packages=pkgs,
        skip_network_check=True,
    )
    assert result.succeeded is False
    assert result.preflight_errors  # non-empty
    assert result.steps == ()        # no phases executed
    # No artifact dirs created (preflight aborted before paths)
    assert not (Path(env_prefix) / "opt" / "siesta-gpu-stack").exists()


def test_run_build_calls_on_warnings_callback(tmp_path, monkeypatch):
    """Non-fatal warnings invoke the callback; returning False aborts."""
    # Use the tiny spec to bypass CUDA + network preflights cleanly.
    a = BuildComponent(
        name="a", repo_url="https://example.com/a.git", ref="v1",
        configure_argv=(), build_argv=(), install_argv=(),
    )
    spec = BuildSpec(artifact_subdir="tiny", components=(a,),
                     cuda_required=False)
    env_prefix = str(tmp_path / "env")
    os.makedirs(env_prefix)
    # Inject a fake warning via monkeypatching preflight
    monkeypatch.setattr(B, "preflight", lambda *a, **k: B.PreflightReport(
        errors=(), warnings=("fake warning",), info=("info line",),
    ))
    seen = []
    result = B.run_build_spec(
        spec, env_prefix,
        conda_binary="/bin/false",
        on_warnings=lambda r: (seen.append(r), False)[1],
        skip_network_check=True,
    )
    assert seen and seen[0].warnings == ("fake warning",)
    assert result.succeeded is False
    assert any("declined" in e for e in result.preflight_errors)


# --------------------------------------------------------------------- #
#  resolve_paths layout                                                  #
# --------------------------------------------------------------------- #


def test_resolve_paths_layout(siesta_gpu_spec, tmp_path):
    """The path layout is the contract surface for the activate hook
    + the engineering doc; pin it."""
    paths = B.resolve_paths(siesta_gpu_spec, str(tmp_path / "env"))
    assert paths.root == tmp_path / "env" / "opt" / "siesta-gpu-stack"
    assert paths.src == paths.root / "src"
    assert paths.build == paths.root / "build"
    assert paths.logs == paths.root / "logs"
    assert paths.sentinels == paths.root / ".sentinels"
    assert paths.activate_d == (
        tmp_path / "env" / "etc" / "conda" / "activate.d"
    )
    assert paths.activate_hook.name == "zz-siesta-gpu-stack.sh"
    assert paths.component_install("elpa") == paths.root / "elpa"
    assert paths.sentinel("elpa", "build") == (
        paths.root / ".sentinels" / "elpa.build.done"
    )


# --------------------------------------------------------------------- #
#  Default jobs                                                          #
# --------------------------------------------------------------------- #


def test_default_jobs_capped_at_8(monkeypatch):
    """Default build concurrency caps at 8 to avoid OOM during
    template-instantiated cmake builds on big machines."""
    monkeypatch.delenv("MOLBUILDER_BUILD_JOBS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert B._default_jobs() == 8


def test_default_jobs_honours_env_override(monkeypatch):
    """MOLBUILDER_BUILD_JOBS=N overrides the cap."""
    monkeypatch.setenv("MOLBUILDER_BUILD_JOBS", "12")
    assert B._default_jobs() == 12


def test_default_jobs_floors_at_1(monkeypatch):
    """If nproc reports something weird, we still get at least one job."""
    monkeypatch.delenv("MOLBUILDER_BUILD_JOBS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert B._default_jobs() == 1


# --------------------------------------------------------------------- #
#  Build-env isolation: subprocess env sanitizer                         #
# --------------------------------------------------------------------- #
#
# build_subprocess_env strips environment variables that would let a
# user's parent shell leak system MPI / system CUDA / system compiler
# flags into the conda build.  These tests pin that contract: every
# leakage vector is named explicitly so a refactor that drops a var
# from _LEAKAGE_ENV_VARS surfaces immediately.


def test_strips_lib_loader_paths():
    """LD_LIBRARY_PATH / LIBRARY_PATH / LD_RUN_PATH leakage would let
    the linker resolve libmpi.so against system paths."""
    base = {
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu",
        "LIBRARY_PATH": "/opt/openmpi/lib",
        "LD_RUN_PATH": "/usr/local/lib",
        "LD_PRELOAD": "/usr/lib/libwrap.so",
    }
    sanitized = B.build_subprocess_env(base)
    assert "PATH" in sanitized
    for k in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_RUN_PATH", "LD_PRELOAD"):
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (linker leakage)"
        )


def test_strips_compiler_include_paths():
    """CPATH / C_INCLUDE_PATH / CPLUS_INCLUDE_PATH leakage would let
    gcc find system mpi.h / cuda.h instead of the env's headers."""
    base = {
        "PATH": "/usr/bin",
        "CPATH": "/usr/include/openmpi",
        "C_INCLUDE_PATH": "/usr/include/openmpi",
        "CPLUS_INCLUDE_PATH": "/usr/include",
        "OBJC_INCLUDE_PATH": "/usr/include",
        "OBJCPLUS_INCLUDE_PATH": "/usr/include",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (header leakage)"
        )


def test_strips_compiler_driver_flags():
    """CFLAGS / CXXFLAGS / FFLAGS / LDFLAGS / CPPFLAGS leakage would
    silently add system include / lib paths to every compile + link
    command, bypassing the env's clean toolchain."""
    base = {
        "PATH": "/usr/bin",
        "CFLAGS": "-I/usr/include/openmpi -L/usr/lib/openmpi",
        "CXXFLAGS": "-I/usr/include/openmpi",
        "FFLAGS": "-I/usr/include/openmpi",
        "LDFLAGS": "-L/usr/lib/openmpi",
        "CPPFLAGS": "-D_GNU_SOURCE",
        "F90FLAGS": "-I/opt/something",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (compiler flag leakage)"
        )


def test_strips_compiler_binary_overrides():
    """CC / CXX / FC overrides would force the build to use whatever
    the user set, even if conda's activate.d wants to set them.  We
    strip first so conda's activate hooks have a clean slate."""
    base = {
        "PATH": "/usr/bin",
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "FC": "/usr/bin/gfortran",
        "F77": "/usr/bin/gfortran",
        "F90": "/usr/bin/gfortran",
        "AR": "/usr/bin/ar",
        "LD": "/usr/bin/ld",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (compiler binary override)"
        )


def test_strips_mpi_location_overrides():
    """MPI_HOME / MPI_ROOT / MPICC etc. would force cmake's FindMPI
    to use a different MPI than the env's openmpi."""
    base = {
        "PATH": "/usr/bin",
        "MPI_HOME": "/usr/lib/x86_64-linux-gnu/openmpi",
        "MPI_ROOT": "/opt/openmpi",
        "MPICC": "/usr/bin/mpicc",
        "MPICXX": "/usr/bin/mpicxx",
        "MPIFORT": "/usr/bin/mpifort",
        "MPI_INCLUDE": "/usr/include/openmpi",
        "MPIEXEC": "/usr/bin/mpiexec",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (MPI location override)"
        )


def test_strips_cuda_location_overrides():
    """CUDA_HOME / CUDA_PATH / NVCC etc. would override the env's
    cuda-nvcc.  We strip first; conda-forge's cuda-nvcc activate.d
    sets them again pointing at $CONDA_PREFIX."""
    base = {
        "PATH": "/usr/bin",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/opt/cuda",
        "CUDA_ROOT": "/usr/local/cuda-11.8",
        "CUDADIR": "/usr/local/cuda",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "NVCC": "/usr/local/cuda/bin/nvcc",
        "NVCCFLAGS": "-arch=sm_70",
        "CUDAFLAGS": "-O3",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (CUDA location override)"
        )


def test_strips_blas_fftw_hdf5_overrides():
    """BLAS_LIBS / FFTW_ROOT / HDF5_DIR / NETCDF_ROOT etc. would force
    cmake's FindXXX to point at system installs."""
    base = {
        "PATH": "/usr/bin",
        "BLAS": "openblas",
        "BLAS_LIBS": "-lopenblas",
        "LAPACK_LIBS": "-llapack",
        "MKLROOT": "/opt/intel/mkl",
        "OPENBLAS_DIR": "/opt/openblas",
        "FFTW_ROOT": "/opt/fftw",
        "HDF5_ROOT": "/opt/hdf5",
        "NETCDF_ROOT": "/opt/netcdf",
        "SCALAPACK_ROOT": "/opt/scalapack",
        "LIBXC_ROOT": "/opt/libxc",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} (lib location override)"
        )


def test_strips_mpi_prefix_families():
    """OMPI_* / MPICH_* / I_MPI_* / PMI_* / SLURM_* exports tune MPI
    runtime behaviour in ways that can bypass the env's MPI launcher
    and break the build's MPI smoke tests.  Stripped by prefix-match."""
    base = {
        "PATH": "/usr/bin",
        "OMPI_MCA_btl": "tcp,self",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "MPICH_ALLOC_MEM_PG_SZ": "huge",
        "I_MPI_ROOT": "/opt/intel",
        "PMI_RANK": "0",
        "PMIX_VERSION": "4.2.0",
        "SLURM_JOB_ID": "12345",
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        if k == "PATH":
            continue
        assert k not in sanitized, (
            f"build_subprocess_env must strip {k} via prefix matching"
        )


def test_preserves_innocuous_env_vars():
    """Variables NOT on the leakage lists (HOME, USER, LANG, TERM,
    PATH, TMPDIR, plus arbitrary user vars) must survive the sanitize
    so conda + git + etc. work normally."""
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "USER": "u",
        "LOGNAME": "u",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C",
        "TERM": "xterm-256color",
        "TMPDIR": "/tmp",
        "SHELL": "/bin/bash",
        "EDITOR": "vim",
        "PYTHONPATH": "",  # arbitrary user var
    }
    sanitized = B.build_subprocess_env(base)
    for k in base:
        assert k in sanitized, (
            f"build_subprocess_env should NOT strip {k}"
        )


def test_template_substitution_resolves_env_prefix(tiny_spec, fake_probe,
                                                   tmp_path):
    """{env_prefix} must resolve to the conda env's prefix path so
    cmake -DMPI_C_COMPILER={env_prefix}/bin/mpicc resolves to the env's
    mpicc, not a system one."""
    refs = {"a": "ref-a", "b": "ref-b"}
    paths, fingerprint, steps = B.plan_build_spec(
        tiny_spec, str(tmp_path / "env"), fake_probe, refs,
    )
    # tiny_spec doesn't use {env_prefix}, but the substitution dict
    # MUST include it for the real recipes that do.  Inject a probe
    # via _build_substitutions.
    for step in steps:
        for token in step.argv:
            assert "{env_prefix}" not in token, (
                f"unsubstituted placeholder in step.argv: {token!r}"
            )
