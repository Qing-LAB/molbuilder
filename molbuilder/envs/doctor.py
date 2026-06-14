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

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..diagnostics import Capabilities, get_capabilities
from .recipes import BUILTIN_RECIPES, Recipe


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
    """
    recipe: Recipe
    effective_name: str
    present: bool
    verify_ok: Optional[bool]
    verify_output: str


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
    argv = (
        conda_binary, "run", "-n", env_name,
        "--no-capture-output", *recipe.verify_argv,
    )
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
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
            ))
            continue
        if not run_verify or caps.conda_binary is None:
            out.append(EnvReport(
                recipe=recipe,
                effective_name=effective,
                present=True,
                verify_ok=None,
                verify_output="",
            ))
            continue
        verify_ok, verify_out = _run_verify(
            effective, recipe, caps.conda_binary,
        )
        out.append(EnvReport(
            recipe=recipe,
            effective_name=effective,
            present=True,
            verify_ok=verify_ok,
            verify_output=verify_out,
        ))
    return out


__all__ = ["EnvReport", "report_all"]
