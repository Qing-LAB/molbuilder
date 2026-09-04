"""The build-time / run-time ABI contract (:mod:`molbuilder.envs.abi`).

These tests exist because of a specific failure that nothing in the
suite could see: a recipe declared ``sysroot_linux-64`` with no
version, the solver took the newest (2.39), and the resulting
toolchain compiled and linked cleanly while producing binaries the
host could not execute.  It surfaced five build steps later as
``configure: error: cannot run C++ compiled programs``.

Two lessons are encoded here:

  * assert on :attr:`Finding.code`, never on prose, so the messages
    stay free to improve;
  * cover the EXECUTION path, not just version arithmetic -- the whole
    class of bug is "every version looks fine and the binary still
    won't start".
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from molbuilder.envs import abi as A


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- #
#  Version values                                                        #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    ("2.39", (2, 39)),
    ("2.17", (2, 17)),
    ("glibc 2.43", (2, 43)),          # os.confstr shape
    ("6.12.0", (6, 12, 0)),
    ("7.0.0-30-generic", (7, 0, 0)),  # uname shape
    ("", None),
    (None, None),
    ("no-numbers-here", None),
])
def test_parse_version_handles_every_shape_this_module_receives(raw, expected):
    assert A.parse_version(raw) == expected


def test_versions_order_as_tuples_not_strings():
    """The bug hinged on 2.39 > 2.17.  String comparison gets this
    right by luck and 2.9 vs 2.17 wrong -- so it must be tuples."""
    assert A.parse_version("2.17") < A.parse_version("2.39")
    assert A.parse_version("2.9") < A.parse_version("2.17")


def test_format_version_round_trips_and_names_the_unknown():
    assert A.format_version((2, 17)) == "2.17"
    assert A.format_version(None) == "unknown"


# --------------------------------------------------------------------- #
#  Scope: ENV                                                            #
# --------------------------------------------------------------------- #


def _write_conda_meta(env_prefix: Path, name: str, version: str,
                      build: str = "h0_0", *, with_json_body: bool = True):
    meta = env_prefix / "conda-meta"
    meta.mkdir(parents=True, exist_ok=True)
    body = {"name": name, "version": version} if with_json_body else {}
    (meta / f"{name}-{version}-{build}.json").write_text(json.dumps(body))


def test_installed_package_version_reads_the_json_body(tmp_path):
    _write_conda_meta(tmp_path, "sysroot_linux-64", "2.39", "hc4b9eeb_6")
    assert A.installed_package_version(tmp_path, "sysroot_linux-64") == (2, 39)


def test_installed_package_version_falls_back_to_the_filename(tmp_path):
    """Package names contain hyphens, so the filename parse has to strip
    the known name rather than split on '-'."""
    _write_conda_meta(tmp_path, "sysroot_linux-64", "2.17", "h0157908_18",
                      with_json_body=False)
    assert A.installed_package_version(tmp_path, "sysroot_linux-64") == (2, 17)


def test_installed_package_version_is_none_when_absent(tmp_path):
    (tmp_path / "conda-meta").mkdir()
    assert A.installed_package_version(tmp_path, "sysroot_linux-64") is None


def test_installed_package_version_is_none_without_an_env(tmp_path):
    assert A.installed_package_version(tmp_path / "nope", "sysroot_linux-64") is None


def test_env_toolchain_reads_both_targets(tmp_path):
    _write_conda_meta(tmp_path, "sysroot_linux-64", "2.17")
    _write_conda_meta(tmp_path, "kernel-headers_linux-64", "3.10.0")
    env = A.EnvToolchain.from_env(tmp_path)
    assert env.sysroot == (2, 17)
    assert env.kernel_headers == (3, 10, 0)


# --------------------------------------------------------------------- #
#  Scope: HOST                                                           #
# --------------------------------------------------------------------- #


def test_host_platform_detect_reads_a_real_glibc():
    """Unlike the GPU probe, this one cannot come back empty-handed on a
    login node -- the C library is always loaded and always readable."""
    host = A.HostPlatform.detect()
    assert host.glibc is not None
    assert host.glibc[0] == 2          # glibc has been 2.x since 1997
    assert host.kernel is not None
    assert host.arch


# --------------------------------------------------------------------- #
#  The contract                                                          #
# --------------------------------------------------------------------- #


def _contract(sysroot, glibc, *, headers=None, kernel=(6, 0, 0)):
    return A.AbiContract(
        host=A.HostPlatform(glibc=glibc, kernel=kernel, arch="x86_64"),
        env=A.EnvToolchain(sysroot=sysroot, kernel_headers=headers),
    )


def test_sysroot_newer_than_host_glibc_is_an_error():
    """The exact shape of the reported bug: env 2.39, host 2.28."""
    findings = _contract((2, 39), (2, 28)).findings()
    codes = {f.code: f for f in findings}
    assert "abi.sysroot-exceeds-host-glibc" in codes
    assert codes["abi.sysroot-exceeds-host-glibc"].severity is A.Severity.ERROR


def test_sysroot_at_or_below_host_glibc_is_silent():
    assert _contract((2, 17), (2, 28)).findings() == ()
    assert _contract((2, 28), (2, 28)).findings() == ()


def test_the_pinned_default_is_clean_on_this_host():
    """2.17 must never fire against any host this suite runs on."""
    host = A.HostPlatform.detect()
    contract = A.AbiContract(host=host,
                             env=A.EnvToolchain(sysroot=(2, 17),
                                                kernel_headers=(3, 10, 0)))
    assert [f for f in contract.findings()
            if f.severity is A.Severity.ERROR] == []


def test_unknown_versions_never_produce_a_finding():
    """An unreadable version is a reason to stay quiet, not to block a
    build on a guess."""
    assert _contract(None, (2, 28)).findings() == ()
    assert _contract((2, 39), None).findings() == ()


def test_kernel_header_skew_is_a_warning_not_an_error():
    """Syscalls are additive; this is worth saying and not worth
    blocking on."""
    findings = _contract((2, 17), (2, 28), headers=(6, 12, 0),
                         kernel=(4, 18, 0)).findings()
    codes = {f.code: f for f in findings}
    assert codes["abi.kernel-headers-exceed-host-kernel"].severity \
        is A.Severity.WARNING


def test_every_rule_has_a_unique_stable_code():
    codes = [r.code for r in A.ABI_RULES]
    assert len(codes) == len(set(codes))
    assert all(c.startswith("abi.") and c.islower() for c in codes)


def test_every_rule_remediates_on_the_env_side():
    """A design invariant, not a coincidence.

    The host's C library is an input we accept; a finding whose fix was
    "upgrade your cluster's glibc" would be one the reader cannot act
    on.  Every knob this module points at lives in the recipe.
    """
    for rule in A.ABI_RULES:
        assert rule.remediation_scope is A.Scope.ENV
    findings = _contract((2, 39), (2, 28), headers=(6, 12, 0),
                         kernel=(4, 18, 0)).findings()
    assert findings
    for finding in findings:
        assert finding.scope is A.Scope.ENV


def test_every_rule_carries_a_remedy():
    """A finding the reader cannot act on is a finding that wastes their
    afternoon -- which is precisely what the raw autotools error did."""
    for rule in A.ABI_RULES:
        assert rule.remedy.strip()


def test_describe_always_reports_both_sides():
    """The pair of numbers is what makes a later failure legible, so it
    is emitted whether or not a rule fired."""
    lines = " ".join(_contract((2, 17), (2, 28)).describe())
    assert "2.17" in lines and "2.28" in lines


# --------------------------------------------------------------------- #
#  The execution check                                                   #
# --------------------------------------------------------------------- #


def _fake_compiler(env_prefix: Path, script: str) -> Path:
    binary = env_prefix / "bin" / "g++"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(script)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return binary


#: Emits a "binary" that exits 1 with the loader's real complaint --
#: what a 2.39-sysroot build does on a 2.28 host.
_COMPILER_MAKING_UNRUNNABLE_BINARIES = r"""#!/bin/bash
out=""
while [ $# -gt 0 ]; do
  [ "$1" = "-o" ] && { out="$2"; shift 2; continue; }
  shift
done
printf '#!/bin/bash\necho "%s: /lib64/libc.so.6: version \\`GLIBC_2.38'"'"' not found" >&2\nexit 1\n' "$out" > "$out"
chmod +x "$out"
"""

_COMPILER_THAT_FAILS_TO_COMPILE = """#!/bin/bash
echo "internal compiler error" >&2
exit 1
"""


def test_no_compiler_in_env_is_silent(tmp_path):
    """A dry run, or an env that has not been created yet.  Missing
    packages are check_env_health's job, not ours."""
    assert A.check_toolchain_executes(tmp_path) is None


@pytest.mark.slow
def test_a_working_toolchain_reports_nothing(tmp_path):
    host_cxx = "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    if host_cxx is None:
        pytest.skip("no host g++ to stand in for the env's compiler")
    (tmp_path / "bin").mkdir()
    os.symlink(host_cxx, tmp_path / "bin" / "g++")
    assert A.check_toolchain_executes(tmp_path) is None


@pytest.mark.slow
def test_binaries_that_compile_but_will_not_run_are_an_error(tmp_path):
    """The one that matters.  Every version check passes; the toolchain
    is still unusable."""
    _fake_compiler(tmp_path, _COMPILER_MAKING_UNRUNNABLE_BINARIES)
    finding = A.check_toolchain_executes(tmp_path)
    assert finding is not None
    assert finding.code == "abi.toolchain-cannot-run-binaries"
    assert finding.severity is A.Severity.ERROR


@pytest.mark.slow
def test_the_failure_message_names_the_version_skew_when_known(tmp_path):
    """The raw loader error is accurate and useless.  With a contract in
    hand the finding must say which knob to turn."""
    _fake_compiler(tmp_path, _COMPILER_MAKING_UNRUNNABLE_BINARIES)
    _write_conda_meta(tmp_path, "sysroot_linux-64", "2.39")
    contract = A.AbiContract(
        host=A.HostPlatform(glibc=(2, 28), kernel=(4, 18, 0), arch="x86_64"),
        env=A.EnvToolchain.from_env(tmp_path))
    rendered = A.check_toolchain_executes(tmp_path, contract=contract).render()
    assert "2.39" in rendered and "2.28" in rendered
    assert "MOLBUILDER_SYSROOT" in rendered


@pytest.mark.slow
def test_a_compiler_that_cannot_compile_is_only_a_warning(tmp_path):
    """The build will produce a far better error; this is an early
    signal, not the diagnosis."""
    _fake_compiler(tmp_path, _COMPILER_THAT_FAILS_TO_COMPILE)
    finding = A.check_toolchain_executes(tmp_path)
    assert finding.code == "abi.toolchain-cannot-compile"
    assert finding.severity is A.Severity.WARNING


def test_abi_never_imports_builds(tmp_path):
    """Dependency direction is load-bearing: builds -> abi, never back.
    A cycle here would make the contract untestable without a live
    build, which is the situation this module exists to escape."""
    source = Path(A.__file__).read_text()
    assert "from .builds" not in source
    assert "import builds" not in source
