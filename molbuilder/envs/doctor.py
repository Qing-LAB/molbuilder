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
      * ``"pip-source"`` -- the package IS installed, but it came from
        somewhere other than the source the recipe records.  The only
        way to catch a package whose version cannot prove which tree it
        is (see :func:`_read_pip_dists`).  Informational when the
        record sets ``fallback_to_index``, because there the indexed
        build is a declared acceptable outcome and only an optional
        capability is missing.

    ``spec`` is what an installer must be HANDED to fix the issue --
    for a pip package with a recorded source that is the direct
    reference, not the bare name, so ``repair`` fetches from where the
    recipe says rather than from whatever the default index offers.
    It does NOT carry the install instruction.  It used to -- an
    ``install_flags`` copied off the record -- back when ``repair`` built
    its own command line.  Repair maps the issue to its ``PipPackage``
    by :attr:`name` now and reads the record, so a second copy of the
    instruction could only drift from the first.
    """
    kind: str
    spec: str        # what to hand an installer to fix this
    found: str       # what's actually installed (or "(not found)")
    #: The package's own name, so a caller can find the RECORD that says
    #: how to fix it.  An issue carries what is WRONG; the recipe says
    #: what to do about it, and repair reads the second rather than a
    #: flattened copy of the instruction.  Empty for conda issues, which
    #: are still repaired from their spec string.
    name: str = ""
    reason: Optional[str] = None   # why the source is unusual, if it is


@dataclass(frozen=True)
class PackageAudit:
    """Real package-presence audit for an env.

    Reads conda-meta/*.json + site-packages/*.dist-info directly --
    no subprocess, no ``conda list``, no ``pip list``.  Source of
    truth is the on-disk metadata that the package manager itself
    wrote at install time.
    """
    checked: bool          # False -> the env prefix is not a directory
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
        ``True`` when the step's own accept rule was satisfied -- see
        :meth:`molbuilder.envs.install.InstallStep.accepts`, which is the
        ONE place that rule lives.  Do not restate it here: the copy that
        stood here said "exited 0 and the substring appeared", which is
        false for a recipe setting ``verify_ignore_exit_code`` (tleap exits
        non-zero from a perfectly healthy start).  ``None`` when the env is missing or the
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
    m = _CONDA_SPEC_RE.fullmatch(spec.strip())
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


def _read_pip_dists(env_prefix: Path) -> Dict[str, Tuple[str, Optional[str]]]:
    """Return ``{normalized_name: (version, source_url_or_None)}``.

    ONE walk, ONE parse of each ``*.dist-info``.  Version and provenance
    are read together because they are two fields of the same record and
    two readers over the same file drift -- the name normalization in
    particular has to agree or the audit silently stops matching.

    Sources come from ``direct_url.json`` (PEP 610), which ``pip`` writes
    beside ``METADATA`` whenever a distribution came from a URL rather
    than an index.  That file is the only on-disk answer to "which tree
    is this?" for a package whose VERSION cannot say: ``pyscf-properties``
    declares ``0.1.0`` both on PyPI and on master.  ``None`` means the
    default index (no direct_url record).

    Name normalization is PEP 503 (lowercase, ``-``/``_``/``.``
    collapsed) so recipe names like ``PeptideBuilder`` match metadata
    names like ``peptidebuilder``.
    """
    out: Dict[str, Tuple[str, Optional[str]]] = {}
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
            if not name:
                continue
            url: Optional[str] = None
            direct = dist_info / "direct_url.json"
            if direct.is_file():
                try:
                    data = json.loads(direct.read_text(encoding="utf-8"))
                    url = data.get("url")
                    commit = (data.get("vcs_info") or {}).get("commit_id")
                    if url and commit:
                        url = f"{url}@{commit}"
                except (OSError, ValueError):
                    url = None
            out[_normalize_pip_name(name)] = (version, url)
    return out


def _split_vcs_ref(url: str) -> Tuple[str, Optional[str]]:
    """Split ``<repo>@<ref>`` into its parts.

    The trailing ``@`` is only a ref when what follows is not a path --
    ``git+ssh://git@host/o/r.git`` carries an ``@`` as USERINFO, and
    truncating there would silently compare the wrong thing.
    """
    head, sep, tail = url.rpartition("@")
    if sep and tail and "/" not in tail:
        return head, tail
    return url, None


def _canon_repo(url: str) -> str:
    """Repository identity: scheme prefix and ``.git`` suffix removed."""
    out = url.split("+", 1)[-1].rstrip("/")
    return out[:-4] if out.endswith(".git") else out


def _same_source(declared: str, installed: Optional[str]) -> bool:
    """Is ``installed`` provenance the one ``declared`` asked for?

    Both sides may carry a ref: a recipe may pin (``...git@<sha>``), and
    ``direct_url.json`` always records the COMMIT pip resolved.  So:

    * repositories must match, always;
    * a declared **commit SHA** must match the resolved commit -- pinning
      a SHA and silently running a different one is the whole failure a
      pin exists to prevent;
    * a declared **branch or tag** cannot be checked against a resolved
      commit, so the repository match is the answer.  Saying so is
      better than pretending a moving ref is verifiable.
    """
    if installed is None:
        return False
    decl_url, decl_ref = _split_vcs_ref(declared)
    inst_url, inst_commit = _split_vcs_ref(installed)
    if _canon_repo(decl_url) != _canon_repo(inst_url):
        return False
    if decl_ref and re.fullmatch(r"[0-9a-f]{7,40}", decl_ref):
        if inst_commit is None:
            return False
        # A short pin matches the long resolved id it prefixes.
        return (inst_commit.startswith(decl_ref)
                or decl_ref.startswith(inst_commit))
    return True


def _normalize_pip_name(name: str) -> str:
    """PEP 503 normalization: lowercase + collapse -/_/.

    Takes a bare project name.  Extras and URLs never reach here --
    ``PipPackage`` keeps them in their own fields and refuses a name
    carrying either, so there is nothing to strip off first.
    """
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def audit_packages(env_prefix: Path, recipe: Recipe) -> PackageAudit:
    """Compare recipe's declared packages against what's on disk.

    Reads conda-meta/*.json + site-packages/*.dist-info -- no
    subprocess.  Source of truth is the on-disk metadata.
    """
    issues: List[PackageAuditIssue] = []
    if not env_prefix.is_dir():
        return PackageAudit(
            checked=False,
            n_conda_declared=len(recipe.conda_packages),
            n_pip_declared=len(recipe.pip_packages),
            issues=(),
        )
    # --- conda packages ---
    #
    # ITERATE THE RECORDS, like the pip loop below.  This used to walk
    # `conda_specs` -- strings -- which forced a pre-pass building a
    # name-SET of the optional ones, parsed every optional spec twice, and
    # left `name` and `reason` empty on every conda issue.  Empty `name` is
    # what stopped `repair` from mapping a conda issue back to its record,
    # and empty `reason` is why `CondaPackage.reason` could never reach a
    # user.  A name-set matched by name is also the exact mechanism § 2.2
    # of the contract abolished for being able to miss silently.
    installed_conda = _read_conda_meta(env_prefix)
    for pkg in recipe.conda_packages:
        parsed = _parse_conda_spec(pkg.spec)
        if parsed is None:
            # Unrecognised shape -- don't try to audit, don't false-alarm.
            continue
        spec = pkg.spec
        name, comparator, version_pat, build_pat = parsed
        # `optional` suffixes EVERY conda kind, not just absence: an
        # optional package resolving to the wrong build must not fail the
        # health check for an env the recipe calls usable without it.
        suffix = "-optional" if pkg.optional else ""
        if name not in installed_conda:
            issues.append(PackageAuditIssue(
                kind=f"conda-missing{suffix}", name=name, spec=spec,
                found="(not found)", reason=pkg.reason,
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
                    kind=f"conda-version{suffix}", name=name, spec=spec,
                    found=f"{name}={installed_version}",
                    reason=pkg.reason,
                ))
                continue
        if build_pat and build_pat != "*":
            if not fnmatch.fnmatchcase(installed_build, build_pat):
                issues.append(PackageAuditIssue(
                    kind=f"conda-build{suffix}", name=name, spec=spec,
                    found=f"{name}={installed_version}={installed_build}",
                    reason=pkg.reason,
                ))
    # --- pip packages ---
    installed_dists = _read_pip_dists(env_prefix)
    for pkg in recipe.pip_packages:
        # Match on IDENTITY only.  dist-info records the project name,
        # never the URL it came from, so a source-bearing package would
        # read as permanently missing if the whole spec were normalized.
        norm = _normalize_pip_name(pkg.name)
        if norm not in installed_dists:
            kind = ("pip-missing-optional"
                    if pkg.optional else "pip-missing")
            issues.append(PackageAuditIssue(
                kind=kind, name=pkg.name, spec=pkg.spec(),
                found="(not found)", reason=pkg.reason,
            ))
            continue
        if pkg.source is not None:
            # Present -- but is it the tree the recipe asked for?  For a
            # package whose version is identical across sources this is
            # the ONLY on-disk answer.
            got = installed_dists[norm][1]
            if not _same_source(pkg.source, got):
                issues.append(PackageAuditIssue(
                    # An index build is an outcome the recipe ACCEPTS
                    # when it declares a fallback, so it degrades a
                    # capability rather than breaking the env.
                    kind=("pip-source-optional"
                          if pkg.fallback_to_index or pkg.optional
                          else "pip-source"),
                    name=pkg.name, spec=pkg.spec(),
                    found=(got or "(default index)"), reason=pkg.reason,
                ))
    return PackageAudit(
        checked=True,
        n_conda_declared=len(recipe.conda_packages),
        n_pip_declared=len(recipe.pip_packages),
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
    *,
    prefix: Optional[str],
) -> Tuple[Optional[bool], str]:
    """Dispatch the recipe's verify command into the env.

    Returns ``(verify_ok, captured_output)``.  ``verify_ok`` is
    ``None`` when the recipe has no verify command (skipped, neither
    ok nor not ok).  Captured output is trimmed to 2 KiB to keep the
    text report compact.

    Whether the output counts as a pass is NOT decided here -- it is
    decided by the step's own accept rule, the same one `install` used
    (:meth:`install.InstallStep.accepts`).  ``prefix`` is the caller's
    already-resolved env prefix, so this does not pay `_env_prefix` again.
    """
    # THE INSTALLER'S STEP, THE INSTALLER'S RUNNER.  This function used
    # to build the verify command itself, bypass ``conda run`` itself,
    # and re-implement the accept rule (exit code gated by the recipe,
    # then the substring) -- a fourth copy of a procedure that already
    # existed, and one that had already drifted to a different output
    # limit.  `verify_step_for` owns what verifying a recipe MEANS and
    # `run_step` owns how a command is dispatched, so asking them is the
    # only way this report can agree with what `install` just did.
    #
    # Imported inside the function because install.py imports
    # `_effective_name` from this module at import time; the cycle is
    # deliberate and this is the side that defers.
    from .install import run_step, verify_step_for
    step = verify_step_for(recipe, conda_binary, env_name)
    if step is None:
        return None, ""
    if prefix is None:
        return False, (
            f"verify could not resolve env prefix for `{env_name}`.  "
            f"Run `{conda_binary} env list` to confirm the env exists; "
            f"if it's there but we can't find it, file an issue."
        )
    # `sink=None` keeps the output captured rather than streamed: a
    # health report is read as a whole, not watched as it runs.
    done = run_step(step, prefix=prefix, timeout=60)
    # Trimmed tighter than the installer's excerpt on purpose -- this one
    # is printed inside a per-env block in a report covering every env.
    return bool(done.outcome.is_success), (done.output or "")[:2048]


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
        # Resolve env prefix ONCE and hand it down -- shared between the
        # verify dispatch and the package audit.  It used to be resolved
        # again inside `_run_verify` while this comment claimed otherwise,
        # paying `_env_prefix` twice per present env; that is the call the
        # installer budgets for (3-5 `env list` / `info --json` per recipe).
        from .install import _env_prefix
        prefix_str = _env_prefix(effective, caps.conda_binary)
        audit: Optional[PackageAudit] = None
        if prefix_str is not None:
            audit = audit_packages(Path(prefix_str), recipe)
        verify_ok, verify_out = _run_verify(
            effective, recipe, caps.conda_binary, prefix=prefix_str,
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
