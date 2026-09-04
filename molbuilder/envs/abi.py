"""The build-time / run-time ABI contract for a source-build env.

WHY THIS MODULE EXISTS (read before editing):

    A conda env is not a container.  It supplies the compiler, the
    headers, and the libraries a build links against -- but the
    binary it produces is loaded at runtime by the HOST's dynamic
    loader against the HOST's C library.  Conda ships no glibc
    runtime and deliberately does not point its binaries at the
    sysroot's loader (``readelf -p .interp`` on any conda binary
    reads ``/lib64/ld-linux-x86-64.so.2`` -- the host's).

    That splits every source build into two scopes that must be
    reasoned about SEPARATELY and then compared:

      :class:`HostPlatform`  what THIS MACHINE provides at RUN time.
                             Facts no conda package can change: the C
                             library, the kernel.  Read-only inputs.

      :class:`EnvToolchain`  what the env compiles AGAINST at BUILD
                             time.  Facts the RECIPE chooses: the
                             sysroot, the kernel headers.  Under our
                             control, and therefore our responsibility.

    glibc is backward compatible and NOT forward compatible: build
    against 2.17 and the binary runs on 2.17 through 2.43; build
    against 2.39 and it will not start on anything below 2.39.  So the
    contract is directional, and it is the only interesting question
    this module answers:

        for every (env floor, host ceiling) pair:  env <= host

    A violation is not a warning.  It produces a toolchain that
    compiles and links without complaint and then emits binaries the
    host cannot execute -- which surfaces downstream as a baffling
    ``configure: error: cannot run C++ compiled programs`` with no
    mention of glibc anywhere.  Naming the mismatch here is the whole
    point: the failure is trivially diagnosable at this layer and
    nearly undiagnosable three layers down.

DESIGN NOTES:

    * Rules are DATA (:data:`ABI_RULES`), not branches.  Adding a new
      build-vs-run pair is a row in that tuple, not an ``if`` in a
      function.  The evaluation loop never needs to change.

    * Findings carry a stable machine-readable :attr:`Finding.code`.
      Tests assert on codes, never on prose, so error wording stays
      free to improve without breaking the suite.

    * This module NEVER imports :mod:`molbuilder.envs.builds`.  The
      dependency runs one way (builds -> abi) so the contract stays
      testable without a live build.  Anything the checks need from
      the build environment (a sanitized env mapping, for instance)
      arrives as a parameter.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Version",
    "parse_version",
    "format_version",
    "Scope",
    "Severity",
    "Finding",
    "HostPlatform",
    "EnvToolchain",
    "AbiRule",
    "ABI_RULES",
    "AbiContract",
    "installed_package_version",
    "check_toolchain_executes",
]


# --------------------------------------------------------------------- #
#  Version values                                                        #
# --------------------------------------------------------------------- #
#
# A tuple of ints, compared lexicographically -- which is exactly the
# ordering glibc / kernel / sysroot versions want.  Deliberately NOT a
# full PEP 440 implementation: every version this module handles is a
# dotted numeric release.  ``builds.py`` still carries its own
# ``_cuda_tuple`` / ``_gcc_major`` parsers; folding those into this type
# is a worthwhile follow-up but is a separate, wider diff.

Version = Tuple[int, ...]

_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")


def parse_version(raw: Optional[str]) -> Optional[Version]:
    """Extract a dotted numeric version from ``raw``.

    Tolerant by design -- the same parser handles ``"2.39"``,
    ``"glibc 2.43"``, ``"6.12.0"``, and ``"7.0.0-30-generic"``, which
    are the four shapes this module actually receives.  Returns
    ``None`` when there is no number to read, and every caller treats
    ``None`` as "cannot judge" rather than as a failure.
    """
    if not raw:
        return None
    m = _VERSION_RE.search(str(raw))
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def format_version(version: Optional[Version]) -> str:
    """Render a :data:`Version` for humans; ``"unknown"`` for ``None``."""
    if not version:
        return "unknown"
    return ".".join(str(p) for p in version)


# --------------------------------------------------------------------- #
#  Findings                                                              #
# --------------------------------------------------------------------- #


class Scope(str, Enum):
    """Which side of the contract a fact belongs to.

    The distinction is load-bearing, not decorative: HOST facts are
    inputs we must accept, ENV facts are choices we can change.  Every
    remedy this module emits acts on the ENV side, because the HOST
    side is not ours to move.
    """

    HOST = "host"
    ENV = "env"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One structured preflight result.

    ``code`` is the stable identity -- dotted, lowercase, and never
    reworded once shipped.  ``summary`` / ``detail`` / ``remedy`` are
    prose and may be improved freely.

    ``scope`` says which side of the contract the REMEDY acts on, and
    it is always :attr:`Scope.ENV` for the rules in this module.  That
    is a design invariant, not a coincidence: the host's C library is
    an input we accept, so a finding that told the user to go upgrade
    their cluster's glibc would be a finding they cannot act on.
    ``test_envs_abi.py`` asserts it.
    """

    code: str
    severity: Severity
    summary: str
    detail: str = ""
    remedy: str = ""
    scope: Scope = Scope.ENV

    def render(self) -> str:
        """Flatten to the single-string form the CLI report consumes."""
        parts = [self.summary]
        if self.detail:
            parts.append(self.detail)
        if self.remedy:
            parts.append(self.remedy)
        return "  ".join(parts)


# --------------------------------------------------------------------- #
#  Scope: HOST -- what the machine provides at runtime                   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class HostPlatform:
    """Runtime facts of the machine this process is running on.

    Every field is something no conda package can change.  Unlike the
    GPU probe in :mod:`molbuilder.envs.builds`, these reads cannot
    return a false negative for "present but not visible": a missing
    ``nvidia-smi`` leaves you unable to distinguish "no GPU here" from
    "no GPU anywhere", whereas the C library is always loaded, always
    readable, and always the true version for this node.  That is why
    a violation of an ABI rule is an ERROR while the compute-cap
    fallback can only ever be a WARNING.
    """

    glibc: Optional[Version]
    kernel: Optional[Version]
    arch: str

    @classmethod
    def detect(cls) -> "HostPlatform":
        """Read the host's runtime facts.  Never raises."""
        return cls(
            glibc=_detect_host_glibc(),
            kernel=_detect_host_kernel(),
            arch=_detect_host_arch(),
        )


def _detect_host_glibc() -> Optional[Version]:
    """Host C-library version.

    ``os.confstr`` is a pure-stdlib read of the running libc -- no
    subprocess, no module load, nothing to be absent on a login node.
    ``platform.libc_ver()`` is the fallback for the (musl / non-glibc /
    restricted) cases where confstr is unavailable.
    """
    try:
        raw = os.confstr("CS_GNU_LIBC_VERSION")   # e.g. "glibc 2.43"
    except (ValueError, OSError, AttributeError):
        raw = None
    version = parse_version(raw)
    if version:
        return version
    try:
        import platform as _platform
        return parse_version(_platform.libc_ver()[1])
    except Exception:
        return None


def _detect_host_kernel() -> Optional[Version]:
    try:
        return parse_version(os.uname().release)
    except (AttributeError, OSError):
        return None


def _detect_host_arch() -> str:
    try:
        return os.uname().machine
    except (AttributeError, OSError):
        return "unknown"


# --------------------------------------------------------------------- #
#  Scope: ENV -- what the recipe chose to compile against                #
# --------------------------------------------------------------------- #


def installed_package_version(env_prefix: str, name: str) -> Optional[Version]:
    """Version of conda package ``name`` installed in ``env_prefix``.

    Reads ``conda-meta/<name>-<version>-<build>.json``.  The JSON's
    ``version`` field is authoritative; the filename is the fallback,
    parsed by stripping the (possibly hyphenated) package name off the
    front so ``sysroot_linux-64-2.39-hc4b9eeb_6.json`` still yields
    ``2.39``.
    """
    meta = Path(env_prefix, "conda-meta")
    if not meta.is_dir():
        return None
    for path in sorted(meta.glob(f"{name}-*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict) and data.get("name") not in (None, name):
            continue  # a longer package name that shares our prefix
        version = parse_version(data.get("version") if isinstance(data, dict) else None)
        if version:
            return version
        stem = path.stem[len(name) + 1:]           # "2.39-hc4b9eeb_6"
        version = parse_version(stem.rsplit("-", 1)[0])
        if version:
            return version
    return None


@dataclass(frozen=True)
class EnvToolchain:
    """Build-time facts the recipe selected, read back from a live env.

    These are the values a recipe is answerable for.  Every remedy in
    :data:`ABI_RULES` names the environment variable that moves one.
    """

    sysroot: Optional[Version]
    kernel_headers: Optional[Version]

    @classmethod
    def from_env(cls, env_prefix: str) -> "EnvToolchain":
        """Read the toolchain's targets from ``<env_prefix>/conda-meta``."""
        return cls(
            sysroot=installed_package_version(env_prefix, "sysroot_linux-64"),
            kernel_headers=installed_package_version(
                env_prefix, "kernel-headers_linux-64"),
        )


# --------------------------------------------------------------------- #
#  The contract: env floor <= host ceiling                               #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AbiRule:
    """One (env floor, host ceiling) pair that must not invert.

    Adding a rule is adding a row to :data:`ABI_RULES`.  The evaluator
    below is generic over every row and never needs editing -- which is
    the point: the next build-vs-run mismatch should be a data change,
    not a new special case bolted into a check function.
    """

    code: str
    env_attr: str
    host_attr: str
    subject: str
    severity: Severity
    remedy: str
    #: Which side the remedy acts on.  ENV for every rule we ship --
    #: see :class:`Finding`.  A future HOST-scoped rule would be one
    #: whose only fix is "use a different machine", and it should be
    #: worded that way rather than pretending to offer a knob.
    remediation_scope: Scope = Scope.ENV

    def evaluate(self, env: EnvToolchain, host: HostPlatform) -> Optional[Finding]:
        """Return a :class:`Finding` when this rule is violated.

        ``None`` when it holds, and also when either side is unknown --
        an unreadable version is a reason to stay quiet, never a reason
        to block a build on a guess.
        """
        env_version = getattr(env, self.env_attr, None)
        host_version = getattr(host, self.host_attr, None)
        if env_version is None or host_version is None:
            return None
        if env_version <= host_version:
            return None
        return Finding(
            code=self.code,
            severity=self.severity,
            summary=(
                f"{self.subject}: the env compiles against "
                f"{format_version(env_version)} but this host provides "
                f"{format_version(host_version)} at runtime."
            ),
            detail=(
                "Binaries built here will link cleanly and then fail to "
                "start, because the loader resolves against the host's "
                "library, not the env's sysroot.  Autotools reports this "
                "as `cannot run C compiled programs`, cmake as a failed "
                "compiler test -- neither mentions the version skew."
            ),
            remedy=self.remedy,
            scope=self.remediation_scope,
        )


#: The contract, as data.  Order is report order.
ABI_RULES: Tuple[AbiRule, ...] = (
    AbiRule(
        code="abi.sysroot-exceeds-host-glibc",
        env_attr="sysroot",
        host_attr="glibc",
        subject="C library (glibc)",
        severity=Severity.ERROR,
        remedy=(
            "Rebuild the env with a sysroot at or below the host's glibc: "
            "`MOLBUILDER_SYSROOT=2.17 molbuilder envs install <recipe> "
            "--clean`.  2.17 is conda-forge's portability floor and is "
            "what every prebuilt package in the env already targets."
        ),
    ),
    # Kernel headers newer than the running kernel are usually benign --
    # syscall numbers are additive and glibc probes at runtime for the
    # ones it needs -- so this is a WARNING.  It is still worth saying:
    # a build that compiles in a syscall the running kernel lacks fails
    # at runtime with ENOSYS, and this line is the only place that skew
    # is ever visible.
    AbiRule(
        code="abi.kernel-headers-exceed-host-kernel",
        env_attr="kernel_headers",
        host_attr="kernel",
        subject="Kernel headers",
        severity=Severity.WARNING,
        remedy=(
            "Usually harmless (syscalls are additive).  If a built binary "
            "fails with ENOSYS, pin the sysroot lower -- kernel-headers "
            "follows sysroot automatically."
        ),
    ),
)


@dataclass(frozen=True)
class AbiContract:
    """Both scopes plus the rules that relate them."""

    host: HostPlatform
    env: EnvToolchain

    @classmethod
    def probe(cls, env_prefix: str) -> "AbiContract":
        """Read both sides from the live host + a live env."""
        return cls(
            host=HostPlatform.detect(),
            env=EnvToolchain.from_env(env_prefix),
        )

    def findings(self) -> Tuple[Finding, ...]:
        """Every violated rule, in :data:`ABI_RULES` order."""
        out: List[Finding] = []
        for rule in ABI_RULES:
            found = rule.evaluate(self.env, self.host)
            if found is not None:
                out.append(found)
        return tuple(out)

    def describe(self) -> Tuple[str, ...]:
        """Info lines for the preflight report.

        Emitted whether or not a rule fires, because the pair of
        numbers is what makes a later failure legible.  Column widths
        match the surrounding report block in ``builds.py``.
        """
        lines = [
            f"Host glibc         {format_version(self.host.glibc):<10s}  "
            f"(runtime C library; sysroot must not exceed it)",
            f"Env sysroot        {format_version(self.env.sysroot):<10s}  "
            f"(what the toolchain compiles against)",
        ]
        return tuple(lines)


# --------------------------------------------------------------------- #
#  The executable check                                                  #
# --------------------------------------------------------------------- #
#
# The rules above catch the mismatch we understand.  This catches the
# whole family -- bad RPATH, a noexec build directory, an architecture
# mismatch, a half-installed toolchain -- by asking the only question
# that actually matters: does this compiler, on this host, produce a
# binary that runs?  It costs about two seconds and it is the check
# whose absence let a 2.39 sysroot reach a student's login node.

_EXEC_PROBE_SOURCE = "int main(void) { return 0; }\n"

#: Tried in order.  The conda triplet name always exists in a compiler
#: env; the bare name exists only after the recipe's shim step has run.
_CXX_CANDIDATES: Tuple[str, ...] = (
    "bin/x86_64-conda-linux-gnu-g++",
    "bin/g++",
    "bin/x86_64-conda-linux-gnu-gcc",
    "bin/gcc",
)


def _find_env_compiler(env_prefix: str) -> Optional[Path]:
    for rel in _CXX_CANDIDATES:
        candidate = Path(env_prefix, rel)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def check_toolchain_executes(env_prefix: str,
                             *,
                             contract: Optional[AbiContract] = None,
                             env: Optional[Mapping[str, str]] = None,
                             timeout: int = 120,
                             ) -> Optional[Finding]:
    """Compile and run a trivial program with the env's own compiler.

    Returns ``None`` when the toolchain works, or when there is no
    compiler to test (a dry run, or an env that has not been created
    yet -- missing packages are :func:`check_env_health`'s job, not
    ours).

    Parameters
    ----------
    contract
        When supplied and a rule is violated, its numbers are folded
        into the failure message.  This is what turns an opaque loader
        error into an actionable one, so callers should pass it.
    env
        Environment mapping for the compiler subprocess.  Callers in
        the build path should pass the same sanitized environment the
        build itself uses, so this check sees what the build will see.
    """
    compiler = _find_env_compiler(env_prefix)
    if compiler is None:
        return None

    # Compile inside the env prefix rather than $TMPDIR.  The env is
    # known-executable (conda's own binaries run from it), whereas a
    # cluster's /tmp is frequently mounted noexec -- which would make
    # this check report a failure the real build never hits.
    scratch_parent = env_prefix if os.access(env_prefix, os.W_OK) else None
    try:
        with tempfile.TemporaryDirectory(prefix=".mb-abi-probe-",
                                         dir=scratch_parent) as scratch:
            source = Path(scratch, "probe.c")
            binary = Path(scratch, "probe")
            source.write_text(_EXEC_PROBE_SOURCE)
            compiled = _quiet_run(
                [str(compiler), str(source), "-o", str(binary)],
                env=env, timeout=timeout)
            if compiled.returncode != 0:
                return Finding(
                    code="abi.toolchain-cannot-compile",
                    severity=Severity.WARNING,
                    summary=(
                        f"The env's compiler ({compiler.name}) could not "
                        f"compile a trivial program."
                    ),
                    detail=_tail(compiled.stderr or compiled.stdout),
                    remedy=(
                        "The build will report the underlying error in "
                        "more detail; this is an early warning, not the "
                        "diagnosis."
                    ),
                )
            ran = _quiet_run([str(binary)], env=env, timeout=timeout)
            if ran.returncode == 0:
                return None
            return _cannot_run_finding(compiler, ran, contract)
    except (OSError, subprocess.SubprocessError):
        # A probe that cannot run is not evidence that the toolchain is
        # broken.  Stay silent rather than block a build on our own
        # inability to test it.
        return None


def _cannot_run_finding(compiler: Path,
                        ran: subprocess.CompletedProcess,
                        contract: Optional[AbiContract],
                        ) -> Finding:
    """Build the ERROR for "compiled fine, would not run".

    When the contract explains why, lead with the explanation -- the
    raw loader message ("version `GLIBC_2.38' not found") is accurate
    but tells the reader nothing about which knob to turn.
    """
    explanation = ""
    remedy = (
        "Re-run with `MOLBUILDER_SYSROOT=2.17 ... --clean` if the env's "
        "sysroot is newer than this host's glibc; otherwise check that "
        "the build directory is not on a noexec mount."
    )
    if contract is not None:
        for finding in contract.findings():
            if finding.severity is Severity.ERROR:
                explanation = f"  {finding.summary}"
                remedy = finding.remedy
                break
    return Finding(
        code="abi.toolchain-cannot-run-binaries",
        severity=Severity.ERROR,
        summary=(
            f"The env's compiler ({compiler.name}) produces binaries this "
            f"host cannot execute."
        ),
        detail=(
            f"A trivial program compiled and linked, then exited "
            f"{ran.returncode} when run.{explanation}  "
            f"{_tail(ran.stderr or ran.stdout)}"
        ).strip(),
        remedy=remedy,
    )


def _quiet_run(argv: Sequence[str],
               *,
               env: Optional[Mapping[str, str]],
               timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout,
        env=dict(env) if env is not None else None,
    )


def _tail(text: Optional[str], limit: int = 300) -> str:
    """Last ``limit`` characters of compiler / loader output, flattened."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat[-limit:]
