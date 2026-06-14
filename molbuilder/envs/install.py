"""Idempotent installer for the conda envs described by recipes.

``molbuilder envs install <name>`` is a thin wrapper around four
phases per recipe:

  1. ``conda create -n <env> -c <ch1> [-c <ch2>] ... <pkg1> <pkg2> ...``
  2. ``conda run -n <env> python -m pip install <pip-pkgs>``
     (skipped when ``recipe.pip_packages`` is empty)
  3. Each tuple in ``recipe.extra_steps`` dispatched via
     ``conda run -n <env> <argv>``.
  4. Verify (re-uses :mod:`molbuilder.envs.doctor`).

The installer never deletes an existing env; if the env already
exists, phases 2-3 still run (so installing twice doesn't break --
``pip install`` and the extra steps are idempotent in practice).
That makes ``install`` safe to re-run when, e.g., the recipe gains
a new pip dependency.

Phases are reported as a list of :class:`InstallStep` instances so
the CLI can surface failures with the exact command that broke.
``--dry-run`` returns the step plan without executing.

This module performs side effects -- the only one in the package.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..diagnostics import Capabilities, get_capabilities
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


def run_install(
    recipe: Recipe,
    *,
    caps: Optional[Capabilities] = None,
    skip_create_if_present: bool = True,
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
    """
    caps = caps if caps is not None else get_capabilities()
    if caps.conda_binary is None:
        raise RuntimeError(
            "conda CLI not found; install conda before invoking "
            "`molbuilder envs install`."
        )
    effective = _effective_name(recipe, caps)
    planned = _plan(recipe, effective, caps.conda_binary)
    env_exists = caps.env_available(effective)
    executed: List[InstallStep] = []
    succeeded = True

    for step in planned:
        if (step.label == "conda create" and env_exists
                and skip_create_if_present):
            executed.append(InstallStep(
                label=step.label, argv=step.argv,
                returncode=0,
                output="env already exists; skipping create",
            ))
            continue
        try:
            cp = subprocess.run(
                step.argv, capture_output=True, text=True, timeout=3600,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            executed.append(InstallStep(
                label=step.label, argv=step.argv,
                returncode=None,
                output=f"step failed to launch: {exc}",
            ))
            succeeded = False
            break
        combined = (cp.stdout or "") + (cp.stderr or "")
        trimmed = combined[:2048]
        executed.append(InstallStep(
            label=step.label, argv=step.argv,
            returncode=cp.returncode, output=trimmed,
        ))
        # For the verify step, exit-code gating is recipe-controlled;
        # everything else is plain "rc != 0 -> fail".
        if step.label == "verify":
            if (not recipe.verify_ignore_exit_code
                    and cp.returncode != 0):
                succeeded = False
                break
            if (recipe.verify_expect_contains
                    and recipe.verify_expect_contains not in combined):
                succeeded = False
                executed[-1] = InstallStep(
                    label=step.label, argv=step.argv,
                    returncode=cp.returncode,
                    output=(trimmed
                            + f"\n(missing expected substring "
                              f"`{recipe.verify_expect_contains}`)"),
                )
                break
        elif cp.returncode != 0:
            succeeded = False
            break

    return InstallResult(
        recipe=recipe,
        effective_name=effective,
        steps=tuple(executed),
        succeeded=succeeded,
    )


__all__ = [
    "InstallStep",
    "InstallResult",
    "plan_install",
    "run_install",
]
