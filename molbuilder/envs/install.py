"""Idempotent installer for the conda envs described by recipes.

``molbuilder envs install <name>`` is a thin wrapper around four
phases per recipe:

  1. ``conda create -n <env> -c <ch1> [-c <ch2>] ... <pkg1> <pkg2> ...``
  2. ``conda run -n <env> python -m pip install <pip-pkgs>``
     (skipped when ``recipe.pip_packages`` is empty)
  3. Each tuple in ``recipe.extra_steps`` dispatched via
     ``conda run -n <env> <argv>``.
  4. **(source-build recipes only)** ``builds.run_build_spec`` runs
     the recipe's :class:`BuildSpec`: clone + cmake + install for each
     component, with sentinel-resume.  Activate.d / deactivate.d
     hooks are rendered into the env's ``etc/conda/`` tree.
  5. Verify (re-uses :mod:`molbuilder.envs.doctor`).

The installer never deletes an existing env; if the env already
exists, phases 2-4 still run (so installing twice doesn't break --
``pip install`` and the extra steps are idempotent in practice; the
build_spec executor has its own sentinel-based resume).  That makes
``install`` safe to re-run when, e.g., the recipe gains a new pip
dependency or the user wants to rebuild from a new SIESTA tag.

Phases are reported as a list of :class:`InstallStep` instances so
the CLI can surface failures with the exact command that broke.
``--dry-run`` returns the step plan without executing.

This module performs side effects -- the only one in the package
besides :mod:`molbuilder.envs.builds`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ..diagnostics import Capabilities, get_capabilities
from . import builds as _builds
from .doctor import _effective_name
from .recipes import Recipe


@dataclass(frozen=True)
class InstallStep:
    """One command in the install plan.

    Attributes
    ----------
    label
        Short tag (``"conda create"``, ``"pip install"``,
        ``"extra"``, ``"verify"``).  Used by the CLI for the per-
        step header line.
    argv
        Full command argv as it would be invoked.  For dry runs the
        CLI prints this verbatim.
    returncode
        Process exit code, or ``None`` for a step that hasn't run
        (either because it was a dry run, or because an earlier step
        failed and we short-circuited).
    output
        First 2 KiB of combined stdout+stderr; empty for not-run
        steps.
    """
    label: str
    argv: Tuple[str, ...]
    returncode: Optional[int] = None
    output: str = ""


@dataclass(frozen=True)
class InstallResult:
    """Outcome of one ``install`` invocation."""
    recipe: Recipe
    effective_name: str
    steps: Tuple[InstallStep, ...]
    succeeded: bool
    build_result: Optional["_builds.BuildResult"] = None


def _plan(recipe: Recipe, env_name: str, conda: str) -> List[InstallStep]:
    """Build the step list without running anything."""
    steps: List[InstallStep] = []

    # Phase 1: conda create.
    create_argv: List[str] = [conda, "create", "-n", env_name, "-y"]
    for ch in recipe.channels:
        create_argv.extend(["-c", ch])
    create_argv.extend(recipe.conda_packages)
    steps.append(InstallStep(label="conda create",
                             argv=tuple(create_argv)))

    # Phase 2: pip install (one combined call if there's anything).
    if recipe.pip_packages:
        pip_argv = (
            conda, "run", "-n", env_name, "--no-capture-output",
            "python", "-m", "pip", "install",
            *recipe.pip_packages,
        )
        steps.append(InstallStep(label="pip install", argv=pip_argv))

    # Phase 3: extra dispatch-into-env steps.
    for extra in recipe.extra_steps:
        argv = (conda, "run", "-n", env_name, "--no-capture-output",
                *extra)
        steps.append(InstallStep(label="extra", argv=argv))

    # Phase 4: verify (only if the recipe declares one).
    if recipe.verify_argv:
        verify_argv = (conda, "run", "-n", env_name, "--no-capture-output",
                       *recipe.verify_argv)
        steps.append(InstallStep(label="verify", argv=verify_argv))

    return steps


def plan_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
) -> Tuple[str, List[InstallStep]]:
    """Return ``(effective_name, steps)`` for the recipe.

    Pure planner -- does not run anything.  Useful for ``--dry-run``
    + for tests that assert on the command shape without subprocess
    side effects.

    Raises
    ------
    RuntimeError
        When no ``conda`` binary is reachable; without it the plan's
        commands would be unrunnable so building one is pointless.
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = _effective_name(recipe, caps)
    return effective, _plan(recipe, effective, caps.conda_binary)


def _env_prefix(env_name: str, conda_binary: str) -> Optional[str]:
    """Return ``$CONDA_PREFIX`` for the named env, or ``None`` if not found.

    Two fallback paths so a partly-broken install still resolves a
    sane prefix:

    1. ``conda env list --json`` -- the canonical registry.  Works
       when conda properly tracks the env.
    2. ``conda info --json``'s ``envs_dirs`` + filesystem check --
       catches the case where conda's registry forgot the env (e.g.
       a previous ``conda env remove`` was interrupted, or
       ``conda create`` failed mid-flight leaving an orphan dir).
       Without this fallback, the build phase would error out with
       "could not resolve $CONDA_PREFIX; conda may have failed
       silently" even when the env directory IS on disk.
    """
    import json as _json
    # Fallback 1: registry
    cp = subprocess.run(
        [conda_binary, "env", "list", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    if cp.returncode == 0:
        try:
            envs = _json.loads(cp.stdout).get("envs", [])
        except (ValueError, KeyError):
            envs = []
        for prefix in envs:
            if Path(prefix).name == env_name:
                return prefix
    # Fallback 2: filesystem search in conda's envs_dirs
    try:
        info_cp = subprocess.run(
            [conda_binary, "info", "--json"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if info_cp.returncode != 0:
        return None
    try:
        info = _json.loads(info_cp.stdout)
    except (ValueError, KeyError):
        return None
    for envs_dir in info.get("envs_dirs", []):
        candidate = Path(envs_dir) / env_name
        if candidate.is_dir():
            return str(candidate)
    return None


# --------------------------------------------------------------------- #
#  Env-state probe -- diagnose conda env state up front                  #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvState:
    """Result of probing the conda env's current state.

    The conda env can be in one of five states from the install's
    point of view.  We resolve it ONCE at the start of the install
    (before any subprocess work runs) so a partly-broken env doesn't
    cause failures 10 minutes into ``conda create``.

    Attributes
    ----------
    name : str
        The env name we probed.
    listed_in_registry : bool
        ``conda env list --json`` includes the env.
    dir_exists : bool
        The env directory exists on disk under one of conda's
        ``envs_dirs``.
    has_conda_meta : bool
        The env directory has a ``conda-meta/`` subdirectory (the
        marker that conda itself uses to recognise a directory as a
        real env).
    prefix : Optional[str]
        Absolute path to the env if it was resolved by either the
        registry or the filesystem; ``None`` for a fresh install.
    """
    name: str
    listed_in_registry: bool
    dir_exists: bool
    has_conda_meta: bool
    prefix: Optional[str]

    @property
    def state_label(self) -> str:
        """One-word classification."""
        reg = self.listed_in_registry
        dir_ok = self.dir_exists and self.has_conda_meta
        if not reg and not self.dir_exists:
            return "FRESH"
        if reg and dir_ok:
            return "PRESENT"
        if not reg and dir_ok:
            return "ORPHAN"
        if reg and not self.dir_exists:
            return "GHOST"
        if self.dir_exists and not self.has_conda_meta:
            return "BROKEN"
        return "UNKNOWN"

    @property
    def can_resume(self) -> bool:
        """``conda create`` can be skipped and downstream phases run."""
        return self.state_label == "PRESENT"

    @property
    def needs_cleanup(self) -> bool:
        """User should run ``--clean`` or manually fix before installing."""
        return self.state_label in ("ORPHAN", "GHOST", "BROKEN")

    def describe(self) -> str:
        """Multi-line human description of the state + recommendation."""
        s = self.state_label
        lines = [
            f"  Env name:           {self.name}",
            f"  Registry lists it:  {'yes' if self.listed_in_registry else 'no'}",
            f"  Directory exists:   {'yes' if self.dir_exists else 'no'}",
            f"  conda-meta/ present:{' yes' if self.has_conda_meta else ' no'}",
        ]
        if self.prefix:
            lines.append(f"  Prefix path:        {self.prefix}")
        lines.append(f"  State:              {s}")
        if s == "FRESH":
            lines.append("  → conda create will run (fresh install).")
        elif s == "PRESENT":
            lines.append("  → conda create will be SKIPPED; install resumes from this env.")
        elif s == "ORPHAN":
            lines.append("  → ORPHAN: directory exists but conda's registry doesn't")
            lines.append("    track it.  conda create will refuse with `prefix already")
            lines.append("    exists`.  RECOMMENDED: re-run with --clean to wipe the")
            lines.append("    directory and start fresh.")
        elif s == "GHOST":
            lines.append("  → GHOST: conda's registry lists this env but the directory")
            lines.append("    is gone.  Fix manually with:")
            lines.append(f"      conda env remove -n {self.name} -y")
            lines.append("    or re-run with --clean which will do the same thing.")
        elif s == "BROKEN":
            lines.append("  → BROKEN: directory exists but is missing conda-meta/, so")
            lines.append("    it's not a real conda env.  Almost certainly residue from")
            lines.append("    a previous failed install.  RECOMMENDED: re-run with")
            lines.append("    --clean to wipe the directory and start fresh.")
        return "\n".join(lines)


def probe_env_state(env_name: str, conda_binary: str) -> EnvState:
    """Probe the conda env's current state.  Pure read; no side effects.

    Runs THREE independent checks (registry, conda-info envs_dirs,
    filesystem) and combines the results into an :class:`EnvState`.
    Cheap -- two ``conda`` subprocesses, ~100 ms each on a warm
    system, much less than the cost of a single failed
    ``conda create``.
    """
    import json as _json

    # Check 1: conda env list (the registry)
    listed = False
    prefix_from_registry: Optional[str] = None
    try:
        cp = subprocess.run(
            [conda_binary, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if cp.returncode == 0:
            envs = _json.loads(cp.stdout).get("envs", [])
            for prefix in envs:
                if Path(prefix).name == env_name:
                    listed = True
                    prefix_from_registry = prefix
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            ValueError, KeyError):
        pass

    # Checks 2 + 3: filesystem (conda's envs_dirs)
    dir_exists = False
    has_conda_meta = False
    prefix_from_fs: Optional[str] = None
    try:
        info_cp = subprocess.run(
            [conda_binary, "info", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if info_cp.returncode == 0:
            info = _json.loads(info_cp.stdout)
            for envs_dir in info.get("envs_dirs", []):
                candidate = Path(envs_dir) / env_name
                if candidate.is_dir():
                    dir_exists = True
                    prefix_from_fs = str(candidate)
                    if (candidate / "conda-meta").is_dir():
                        has_conda_meta = True
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            ValueError, KeyError):
        pass

    prefix = prefix_from_registry or prefix_from_fs
    return EnvState(
        name=env_name,
        listed_in_registry=listed,
        dir_exists=dir_exists,
        has_conda_meta=has_conda_meta,
        prefix=prefix,
    )


# NOTE: the helpers ``_env_listed_now`` and ``_env_prefix_dir_exists``
# used to live here.  Both were thin wrappers that duplicated logic
# already inside ``probe_env_state`` (the ONE source of truth for env
# presence).  Worse, they were combined via an ``OR`` in run_install
# that fired True for orphan directories (which conda create would
# then refuse with "prefix already exists"), shipping a false
# positive that masked --clean failures.  Replaced by direct
# ``probe_env_state(...).can_resume`` use in run_install; the helpers
# are gone, not deprecated, because nothing else called them.


def run_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
    skip_create_if_present: bool = True,
    rebuild: Optional[str] = None,
    build_on_warnings: Optional["_builds.ConfirmWarningsCallback"] = None,
    build_on_progress: Optional["_builds.ProgressCallback"] = None,
    build_skip_network_check: bool = False,
) -> InstallResult:
    """Execute the install plan, stopping at the first failed step.

    Parameters
    ----------
    skip_create_if_present
        When ``True`` (default) and the env already exists, the
        ``conda create`` step is reported as a no-op (returncode 0,
        output ``"env already exists; skipping create"``) and the
        remaining phases run normally.  This is what makes ``install``
        idempotent: re-running picks up new pip deps without trying
        to re-create the env.  Set ``False`` only in tests.
    rebuild
        For recipes carrying a ``build_spec``, forwarded to
        :func:`builds.run_build_spec`.  ``None`` or ``"none"`` resumes
        from sentinels; ``"all"`` rebuilds everything; a component
        name (``"elpa"``, ``"siesta"``) rebuilds that component plus
        everything downstream of it.  Ignored for non-build recipes.
    build_on_warnings, build_on_progress
        Optional callbacks forwarded to
        :func:`builds.run_build_spec` for source-build recipes.
        ``build_on_warnings(report) -> bool`` lets the CLI surface
        non-fatal preflight warnings + ask the user to confirm;
        ``build_on_progress(event, step, result)`` lets the CLI
        render per-phase progress.
    build_skip_network_check
        Skip the per-component ``git ls-remote`` reachability check.
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = _effective_name(recipe, caps)
    planned = _plan(recipe, effective, caps.conda_binary)
    # NOTE: we DELIBERATELY do not pre-compute ``env_exists`` from caps
    # here.  The conda-create skip decision below uses ``probe_env_state``
    # live -- the cached caps view can be stale (notably right after
    # --clean, when ``get_capabilities()`` returns the bound snapshot
    # rather than re-detecting), and trusting it caused the
    # 2026-06-15 "env already exists; conda may have failed silently"
    # regression.  See the conda-create branch below for the live probe.
    executed: List[InstallStep] = []
    succeeded = True

    # Reorder: conda-create + pip + extra_steps + (build_spec) + verify.
    # We pull the verify step out of `planned` and re-append it after
    # the build phase (if any) so verify runs against the built binary.
    verify_steps = [s for s in planned if s.label == "verify"]
    pre_verify = [s for s in planned if s.label != "verify"]

    # User-visible progress on every pre-build step.  Previously these
    # ran with capture_output=True so a 5-10 minute ``conda create``
    # downloading several GB of CUDA packages looked frozen to the
    # user.  We now stream stdout+stderr line-by-line to sys.stderr
    # via the same ``run_streaming`` helper used by the source-build
    # phases, so the install pipeline shows continuous progress.
    total_pre = len(pre_verify)
    for i, step in enumerate(pre_verify, start=1):
        if step.label == "conda create" and skip_create_if_present:
            # Live re-check via the existing EnvState machine -- the
            # ONE source of truth for "is this env actually installable-
            # into".  ``can_resume`` is True iff registry-list AND
            # dir-exists AND conda-meta -- the same three signals an
            # orphan dir / partial install would fail.  This replaces
            # an earlier inline OR that ALSO ORed in the cached
            # ``caps.env_available`` (stale right after --clean) and
            # whose two "independent" live probes both delegated to
            # ``_env_prefix`` and so were not independent at all.
            state = probe_env_state(effective, caps.conda_binary)
            if state.can_resume:
                executed.append(InstallStep(
                    label=step.label, argv=step.argv,
                    returncode=0,
                    output=f"env `{effective}` already exists; "
                           f"skipping create",
                ))
                sys.stderr.write(
                    f"[{i}/{total_pre}] {step.label}: "
                    f"SKIPPED (env `{effective}` already exists -- "
                    f"if you want a fresh env, re-run with --clean)\n"
                )
                sys.stderr.flush()
                continue
            # If the env is in a broken / orphan / ghost state,
            # ``conda create`` will refuse with "prefix already exists".
            # The state probe already classified the case; surface a
            # diagnostic so the user knows to re-run with --clean
            # instead of staring at a cryptic conda error.
            if state.needs_cleanup:
                executed.append(InstallStep(
                    label=step.label, argv=step.argv,
                    returncode=None,
                    output=f"env `{effective}` is in state "
                           f"{state.state_label} -- re-run with --clean "
                           f"to wipe before installing.",
                ))
                sys.stderr.write(
                    f"[{i}/{total_pre}] {step.label}: "
                    f"BLOCKED (env state {state.state_label}; "
                    f"re-run with --clean)\n"
                )
                sys.stderr.flush()
                succeeded = False
                break
        sys.stderr.write(
            f"[{i}/{total_pre}] {step.label}: starting "
            f"(streaming output below; this may take 5-15 min for "
            f"conda create with large package sets)\n"
        )
        sys.stderr.flush()
        rc, combined = _builds.run_streaming(
            list(step.argv),
            sink=sys.stderr,
            timeout=3600,
        )
        if rc is None:
            executed.append(InstallStep(
                label=step.label, argv=step.argv,
                returncode=None,
                output=combined or "step failed to launch",
            ))
            sys.stderr.write(
                f"[{i}/{total_pre}] {step.label}: FAILED to launch\n"
            )
            sys.stderr.flush()
            succeeded = False
            break
        # Keep first 4096 chars for the failure-recap CLI output; the
        # full output was already streamed to the user's terminal.
        trimmed = combined[:4096]
        executed.append(InstallStep(
            label=step.label, argv=step.argv,
            returncode=rc, output=trimmed,
        ))
        if rc != 0:
            sys.stderr.write(
                f"[{i}/{total_pre}] {step.label}: FAILED (rc={rc})\n"
            )
            sys.stderr.flush()
            succeeded = False
            break
        sys.stderr.write(
            f"[{i}/{total_pre}] {step.label}: OK\n"
        )
        sys.stderr.flush()

    # Build-spec phase: only if recipe declares one AND nothing failed
    # before it.  The build executor has its own sentinel resume; we
    # surface it as a single InstallStep so the CLI's per-step view
    # still works.
    build_result: Optional[_builds.BuildResult] = None
    if succeeded and recipe.build_spec is not None:
        prefix = _env_prefix(effective, caps.conda_binary)
        if prefix is None:
            executed.append(InstallStep(
                label="build", argv=("internal", "resolve-env-prefix"),
                returncode=None,
                output=f"could not resolve $CONDA_PREFIX for env "
                       f"{effective!r}; conda may have failed silently.",
            ))
            succeeded = False
        else:
            build_result = _builds.run_build_spec(
                recipe.build_spec, prefix,
                conda_binary=caps.conda_binary,
                rebuild=rebuild,
                conda_packages=recipe.conda_packages,
                skip_network_check=build_skip_network_check,
                on_warnings=build_on_warnings,
                on_progress=build_on_progress,
            )
            if build_result.preflight_errors:
                executed.append(InstallStep(
                    label="build:preflight",
                    argv=("preflight",),
                    returncode=None,
                    output="\n".join(build_result.preflight_errors),
                ))
                succeeded = False
            else:
                for sresult in build_result.steps:
                    executed.append(InstallStep(
                        label=f"build:{sresult.step.component}.{sresult.step.phase}",
                        argv=sresult.step.argv,
                        returncode=sresult.returncode,
                        output=sresult.output,
                    ))
                if not build_result.succeeded:
                    succeeded = False

    if succeeded and verify_steps:
        for step in verify_steps:
            sys.stderr.write(f"[verify] {step.label}: starting\n")
            sys.stderr.flush()
            rc, combined = _builds.run_streaming(
                list(step.argv),
                sink=sys.stderr,
                timeout=3600,
            )
            if rc is None:
                executed.append(InstallStep(
                    label=step.label, argv=step.argv,
                    returncode=None,
                    output=combined or "step failed to launch",
                ))
                sys.stderr.write(
                    f"[verify] {step.label}: FAILED to launch\n"
                )
                sys.stderr.flush()
                succeeded = False
                break
            trimmed = combined[:4096]
            executed.append(InstallStep(
                label=step.label, argv=step.argv,
                returncode=rc, output=trimmed,
            ))
            if (not recipe.verify_ignore_exit_code
                    and rc != 0):
                sys.stderr.write(
                    f"[verify] {step.label}: FAILED (rc={rc})\n"
                )
                sys.stderr.flush()
                succeeded = False
                break
            if (recipe.verify_expect_contains
                    and recipe.verify_expect_contains not in combined):
                succeeded = False
                executed[-1] = InstallStep(
                    label=step.label, argv=step.argv,
                    returncode=rc,
                    output=(trimmed
                            + f"\n(missing expected substring "
                              f"`{recipe.verify_expect_contains}`)"),
                )
                sys.stderr.write(
                    f"[verify] {step.label}: FAILED (missing expected "
                    f"output: {recipe.verify_expect_contains!r})\n"
                )
                sys.stderr.flush()
                break
            sys.stderr.write(f"[verify] {step.label}: OK\n")
            sys.stderr.flush()

    return InstallResult(
        recipe=recipe,
        effective_name=effective,
        steps=tuple(executed),
        succeeded=succeeded,
        build_result=build_result,
    )


__all__ = [
    "InstallStep",
    "InstallResult",
    "plan_install",
    "run_install",
]
