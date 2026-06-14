"""Conda env management surface.

The package exposes three things:

  * ``run_in_env`` / ``run_tool`` -- the historical subprocess-dispatch
    helpers (moved from ``molbuilder/envs.py`` to ``_dispatch.py`` when
    the package landed; back-compat re-exports here).
  * ``Recipe`` + the registry of recipes for each declared env --
    see :mod:`molbuilder.envs.recipes`.
  * The doctor / install helpers that drive the CLI subcommands --
    see :mod:`molbuilder.envs.doctor` + :mod:`molbuilder.envs.install`.

``import subprocess`` lives at module scope so existing tests that
patch ``molbuilder.envs.subprocess.run`` keep working unchanged
(the attribute resolves to the same stdlib ``subprocess`` object
that ``_dispatch`` imports, so monkeypatching either name has the
same effect).
"""
from __future__ import annotations

import shutil      # re-exported for test monkeypatching; see module docstring
import subprocess  # re-exported for test monkeypatching; see module docstring

from ._dispatch import run_in_env, run_tool
from .recipes import (
    BUILTIN_RECIPES,
    Recipe,
    recipe_by_name,
    recipe_for_category,
)

__all__ = [
    "run_in_env",
    "run_tool",
    "Recipe",
    "BUILTIN_RECIPES",
    "recipe_by_name",
    "recipe_for_category",
    "subprocess",
    "shutil",
]
