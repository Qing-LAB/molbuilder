"""Diagnostic report for the conda envs molbuilder uses.

``molbuilder envs doctor`` produces a per-recipe report:

  * effective env name (after ``molbuilder.json`` overrides)
  * present / missing (from the capabilities snapshot)
  * verify command result, when present and verifiable

The report is a pure data structure (:class:`EnvReport`) so callers
(CLI, tests, future web-surface) all read the same shape; the CLI
renders it as a text table, tests assert against the fields.

This module performs no installation -- it only reports.  All side
effects live in :mod:`molbuilder.envs.install`.
"""
from __future__ import annotations

import fnmatch
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..diagnostics import Capabilities, get_capabilities
from .recipes import BUILTIN_RECIPES, Recipe


@dataclass(frozen=True)
class PackageAuditIssue:
    """A single mismatch between a recipe's declared packages and
    what's actually installed in the env.

    ``kind`` is one of:
      * ``"conda-missing"`` -- declared in recipe.conda_packages but
        no conda-meta record found
      * ``"conda-version"`` -- name matches but the installed version
        doesn't satisfy the recipe's pin
      * ``"conda-build"`` -- name + version match but the conda build
        string doesn't match the recipe's pattern (e.g., ``mpi_openmpi_*``)
      * ``"pip-missing"`` -- declared in recipe.pip_packages but no
        ``*.dist-info`` found in the env's site-packages
    """
    kind: str
    spec: str        # the recipe's declared spec
    found: str       # what's actually installed (or "(not found)")


@dataclass(frozen=True)
class PackageAudit:
    """Real package-presence audit for an env.

    Reads conda-meta/*.json + site-packages/*.dist-info directly --
    no subprocess, no ``conda list``, no ``pip list``.  Source of
    truth is the on-disk metadata that the package manager itself
    wrote at install time.
    """
    checked: bool          # False -> env missing or no python in env
    n_conda_declared: int
    n_pip_declared: int
    issues: Tuple[PackageAuditIssue, ...]


@dataclass(frozen=True)
class EnvReport:
    """One recipe's status as seen by ``doctor`` at report time.

    Attributes
    ----------
    recipe
        The :class:`Recipe` this report covers.
    effective_name
        Env name after applying ``molbuilder.json`` overrides; for
        routed recipes this may differ from ``recipe.name``.  For
        the host recipe (``category is None``) this equals
        ``recipe.name``.
    present
        ``True`` when ``effective_name`` appears in
        ``capabilities.conda_envs``.
    verify_ok
        ``True`` when the verify command exited 0 and (if set) its
        ``verify_expect_contains`` substring appeared in the combined
        stdout+stderr.  ``None`` when the env is missing or the
        verify command was not run (e.g., the recipe has no
        ``verify_argv`` set).
    verify_output
        First 2 KiB of the verify command's combined stdout+stderr
        when ``verify_ok`` is not ``None``; empty otherwise.  Trimmed
        because some verify commands emit MPI banners or warnings
        that would crowd the report.
    package_audit
        Real package-presence audit: every package in recipe.conda_packages
        AND recipe.pip_packages checked against the env's conda-meta/
        + site-packages/*.dist-info.  ``None`` when the env is missing.
    """
    recipe: Recipe
    effective_name: str
    present: bool
    verify_ok: Optional[bool]
    verify_output: str
    package_audit: Optional[PackageAudit] = None


# Conda spec parser -- matches the subset our recipes use:
#   "name"
#   "name=version"
#   "name=version=build"
#   "name>=version"  (and other comparators -- treated as name-only match)
#   "channel::name=version"
# Capture the comparator separately so we know whether to do a strict
# version glob match (``=``) or just a name-presence check (everything
# else).  Glob characters in version/build are honoured via fnmatch.
_CONDA_SPEC_RE = re.compile(
    r"^(?:[^:]+::)?"          # optional channel prefix (stripped)
    r"([A-Za-z0-9_.\-]+)"     # 1: name
    r"(?:([=<>!~]+)([^=]+))?"  # 2: comparator, 3: version (optional)
    r"(?:=([^=]+))?$"          # 4: build (optional)
)


def _parse_conda_spec(
    spec: str,
) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """Parse a conda spec into ``(name, comparator, version, build)``.

    Returns ``None`` for spec shapes we don't try to audit (e.g.,
    URL-based specs).  ``comparator`` is the literal operator we saw
    (``"="``, ``">="``, ``"<="``, ``"<"``, ``">"``, ``"!="``, ``"~="``);
    callers should only do exact glob matching when comparator is
    ``"="``.  Glob characters in version/build are preserved so
    fnmatch can match against the installed value.
    """
    m = _CONDA_SPEC_RE.match(spec.strip())
    if not m:
        return None
    name = m.group(1)
    comparator = m.group(2)
    version = m.group(3)
    build = m.group(4)
    return name, comparator, version, build


def _read_conda_meta(env_prefix: Path) -> Dict[str, Tuple[str, str]]:
    """Return ``{name: (version, build)}`` from ``<env>/conda-meta/*.json``.

    Each conda-meta JSON file is one installed package's record.
    Empty dict if the directory doesn't exist (env missing or
    corrupted).
    """
    out: Dict[str, Tuple[str, str]] = {}
    meta_dir = env_prefix / "conda-meta"
    if not meta_dir.is_dir():
        return out
    for f in meta_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = data.get("name")
        version = data.get("version", "")
        build = data.get("build", "")
        if name:
            out[name] = (version, build)
    return out


def _read_pip_packages(env_prefix: Path) -> Dict[str, str]:
    """Return ``{normalized_name: version}`` from site-packages dist-info.

    Walks every ``lib/python*/site-packages/*.dist-info/METADATA``,
    extracts ``Name`` + ``Version`` headers.  Normalizes the name to
    lowercase with ``-``/``_`` collapsed (PEP 503) so recipe specs
    like ``PeptideBuilder`` match metadata names like
    ``peptidebuilder``.
    """
    out: Dict[str, str] = {}
    lib = env_prefix / "lib"
    if not lib.is_dir():
        return out
    for py_dir in lib.glob("python*"):
        sp = py_dir / "site-packages"
        if not sp.is_dir():
            continue
        for dist_info in sp.glob("*.dist-info"):
            metadata = dist_info / "METADATA"
            if not metadata.is_file():
                continue
            try:
                text = metadata.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            name = ""
            version = ""
            for line in text.splitlines():
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                if name and version:
                    break
            if name:
                norm = re.sub(r"[-_.]+", "-", name).lower()
                out[norm] = version
    return out


def _normalize_pip_name(name: str) -> str:
    """PEP 503 normalization: lowercase + collapse -/_/."""
    # Strip extras like ``cupy-cuda13x[ctk]`` -> ``cupy-cuda13x``
    name = name.split("[")[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def audit_packages(env_prefix: Path, recipe: Recipe) -> PackageAudit:
    """Compare recipe's declared packages against what's on disk.

    Reads conda-meta/*.json + site-packages/*.dist-info -- no
    subprocess.  Source of truth is the on-disk metadata.
    """
    issues: List[PackageAuditIssue] = []
    conda_specs = list(recipe.conda_packages)
    pip_specs = list(recipe.pip_packages)
    # Build name-sets for optional packages so we can classify
    # missing ones as info-only (kind suffixed with ``-optional``).
    # Optional packages typically gate a non-default feature (GPU,
    # OAuth provider, etc.) -- the env still functions without them.
    optional_conda = {
        _parse_conda_spec(s)[0]
        for s in recipe.optional_conda_packages
        if _parse_conda_spec(s) is not None
    }
    optional_pip = {
        _normalize_pip_name(s) for s in recipe.optional_pip_packages
    }
    if not env_prefix.is_dir():
        return PackageAudit(
            checked=False,
            n_conda_declared=len(conda_specs),
            n_pip_declared=len(pip_specs),
            issues=(),
        )
    # --- conda packages ---
    installed_conda = _read_conda_meta(env_prefix)
    for spec in conda_specs:
        parsed = _parse_conda_spec(spec)
        if parsed is None:
            # Unrecognised shape -- don't try to audit, don't false-alarm.
            continue
        name, comparator, version_pat, build_pat = parsed
        if name not in installed_conda:
            kind = ("conda-missing-optional"
                    if name in optional_conda else "conda-missing")
            issues.append(PackageAuditIssue(
                kind=kind, spec=spec, found="(not found)",
            ))
            continue
        installed_version, installed_build = installed_conda[name]
        # Only do a strict version glob match when the spec uses ``=``.
        # For ``>=``, ``<=``, ``<``, ``>``, ``!=``, ``~=`` we skip the
        # version check -- proper semver comparison would need
        # ``packaging.version`` (not a host-env dep we want to lock in
        # for an audit pass).  The name-presence check still catches
        # missing packages, which is the common failure mode.
        if version_pat and version_pat != "*" and comparator == "=":
            # Conda treats ``=X.Y`` as ``starts with X.Y``.  fnmatch
            # would only match exact unless the user passed a glob,
            # so we expand bare versions to ``X.Y*``.
            pat = version_pat
            if not any(c in pat for c in "*?["):
                pat = pat + "*"
            if not fnmatch.fnmatchcase(installed_version, pat):
                issues.append(PackageAuditIssue(
                    kind="conda-version", spec=spec,
                    found=f"{name}={installed_version}",
                ))
                continue
        if build_pat and build_pat != "*":
            if not fnmatch.fnmatchcase(installed_build, build_pat):
                issues.append(PackageAuditIssue(
                    kind="conda-build", spec=spec,
                    found=f"{name}={installed_version}={installed_build}",
                ))
    # --- pip packages ---
    installed_pip = _read_pip_packages(env_prefix)
    for spec in pip_specs:
        norm = _normalize_pip_name(spec)
        if norm not in installed_pip:
            kind = ("pip-missing-optional"
                    if norm in optional_pip else "pip-missing")
            issues.append(PackageAuditIssue(
                kind=kind, spec=spec, found="(not found)",
            ))
    return PackageAudit(
        checked=True,
        n_conda_declared=len(conda_specs),
        n_pip_declared=len(pip_specs),
        issues=tuple(issues),
    )


def _effective_name(recipe: Recipe, caps: Capabilities) -> str:
    """The env name that ``conda run -n ...`` will hit.

    For routed recipes (``category`` set), this honours the
    ``molbuilder.json`` ``envs.<category>`` override.  For the host
    recipe (``category is None``), the recipe's default name is the
    answer -- there's no override slot for host today.
    """
    if recipe.category is not None:
        # env_for_category falls back to DEFAULT_ENV_NAMES when no
        # override is present, so this is always a non-None string.
        return caps.env_for_category(recipe.category) or recipe.name
    return recipe.name


def _run_verify(
    env_name: str,
    recipe: Recipe,
    conda_binary: str,
) -> Tuple[Optional[bool], str]:
    """Dispatch the recipe's verify command into the env.

    Returns ``(verify_ok, captured_output)``.  ``verify_ok`` is
    ``None`` when the recipe has no verify command (skipped, neither
    ok nor not ok).  Captured output is trimmed to 2 KiB to keep the
    text report compact.
    """
    if not recipe.verify_argv:
        return None, ""
    # Bypass ``conda run`` -- on mamba 2.x it generates a temp shell
    # stub (``/tmp/mamba*``) that uses ``exec --`` which bash rejects
    # with ``exec: --: invalid option``.  Use the same wrapper-script
    # bypass as install.py's _bypass_conda_run: resolve env prefix,
    # set conda activate's env vars manually, source activate.d, exec.
    from .install import _env_prefix, _bypass_conda_run
    prefix = _env_prefix(env_name, conda_binary)
    if prefix is None:
        return False, (
            f"verify could not resolve env prefix for `{env_name}`.  "
            f"Run `{conda_binary} env list` to confirm the env exists; "
            f"if it's there but we can't find it, file an issue."
        )
    raw_argv = (
        conda_binary, "run", "-n", env_name,
        "--no-capture-output", *recipe.verify_argv,
    )
    try:
        new_argv, _ = _bypass_conda_run(raw_argv, prefix)
    except ValueError:
        new_argv = raw_argv
    try:
        cp = subprocess.run(
            list(new_argv), capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, f"verify failed to launch: {exc}"
    combined = (cp.stdout or "") + (cp.stderr or "")
    trimmed = combined[:2048]
    # Exit-code check is gated by the recipe -- some binaries (tleap)
    # legitimately exit non-zero even after a healthy start, in which
    # case the substring check IS the verify.
    if recipe.verify_ignore_exit_code:
        ok = True
    else:
        ok = (cp.returncode == 0)
    if ok and recipe.verify_expect_contains:
        ok = recipe.verify_expect_contains in combined
    return ok, trimmed


def report_all(
    caps: Optional[Capabilities] = None,
    *,
    recipes: Tuple[Recipe, ...] = BUILTIN_RECIPES,
    run_verify: bool = True,
) -> List[EnvReport]:
    """Build an :class:`EnvReport` for every recipe.

    Parameters
    ----------
    caps
        Capabilities snapshot to read.  ``None`` (the default) reads
        the process singleton via :func:`get_capabilities`.
    recipes
        Recipes to report on; defaults to the built-in five.
    run_verify
        When ``False``, skip the verify-command dispatch (faster, for
        ``list``-style summaries).
    """
    caps = caps if caps is not None else get_capabilities()
    out: List[EnvReport] = []
    for recipe in recipes:
        effective = _effective_name(recipe, caps)
        present = caps.env_available(effective)
        if not present:
            out.append(EnvReport(
                recipe=recipe,
                effective_name=effective,
                present=False,
                verify_ok=None,
                verify_output="",
                package_audit=None,
            ))
            continue
        # Fast-mode (run_verify=False): skip both verify AND audit.
        # The audit needs subprocess to resolve env prefix; the fast
        # mode is for ``list``-style summaries that shouldn't shell
        # out per env.
        if not run_verify or caps.conda_binary is None:
            out.append(EnvReport(
                recipe=recipe,
                effective_name=effective,
                present=True,
                verify_ok=None,
                verify_output="",
                package_audit=None,
            ))
            continue
        # Resolve env prefix once -- shared between the verify bypass
        # and the package audit.
        from .install import _env_prefix
        prefix_str = _env_prefix(effective, caps.conda_binary)
        audit: Optional[PackageAudit] = None
        if prefix_str is not None:
            audit = audit_packages(Path(prefix_str), recipe)
        verify_ok, verify_out = _run_verify(
            effective, recipe, caps.conda_binary,
        )
        out.append(EnvReport(
            recipe=recipe,
            effective_name=effective,
            present=True,
            verify_ok=verify_ok,
            verify_output=verify_out,
            package_audit=audit,
        ))
    return out


__all__ = ["EnvReport", "PackageAudit", "PackageAuditIssue",
           "audit_packages", "report_all"]
