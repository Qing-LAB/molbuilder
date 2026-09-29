"""Conda env management surface.

The package exposes three things:

  * ``run_in_env`` / ``run_tool`` -- dispatch a command into a named env,
    or to the env the tool is routed to -- see
    :mod:`molbuilder.envs._dispatch`.
  * ``Recipe`` + the registry of recipes for each declared env --
    see :mod:`molbuilder.envs.recipes`.
  * The doctor / install helpers that drive the CLI subcommands --
    see :mod:`molbuilder.envs.doctor` + :mod:`molbuilder.envs.install`.

"""
from __future__ import annotations

from ._dispatch import route, run_in_env, run_tool
from .recipes import (
    Recipe,
    builtin_recipes,
    recipe_by_name,
    recipe_for_category,
)

__all__ = [
    "run_in_env",
    "run_tool",
    "route",
    "Recipe",
    "builtin_recipes",
    "recipe_by_name",
    "recipe_for_category",
]
