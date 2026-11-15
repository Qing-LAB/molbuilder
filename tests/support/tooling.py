"""The opt-in test tooling, and the ONE rule for a test that needs it
(`process/testing.md` § 4, *one rule for missing tooling*; the user,
2026-11-14: "unify the mechanism").

The tooling is the host recipe's opt-in records (`ops/env-framework.md`
§ 3.2a), all installed by one command, :data:`INSTALL`.  A test that needs a
piece of it and finds it absent skips, naming that command -- never an
error, never a reason of its own.  Two places meet the rule, and both ask
this module: the harness's `_node_esm.node()`, the one way a test reaches
Node, and `conftest.py`'s collection hook, for every test that asks for a
fixture of the browser plugin (:func:`browser_fixtures_asked`) -- skipped
there, before any fixture of the test is set up.

What a test needs of the MACHINE beyond the host env -- an env of its own
recipe, a visible GPU -- follows the same rule
(`process/testing.md` § 3): :func:`unmet` is the one answer and its reason.
The collection hook asks it, in one place and after the browser's reason,
for the needs a real run's fixture declares
(`support.real_runs.makes_a_real_run`) and those a test states of its own
body (:func:`needs`).
"""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from typing import Optional

import pytest

from molbuilder.diagnostics import DEFAULT_ENV_NAMES
from molbuilder.envs.hints import install_cmd, opt_in_cmd

#: The host env, as its recipe names it -- always its own name
#: (`envs.recipes.effective_name`).
HOST_ENV = "molbuilder"
#: The command that installs every piece of the test tooling -- the host
#: env's opt-in records, as `doctor` names it (`envs.hints.opt_in_cmd`).
INSTALL = opt_in_cmd(HOST_ENV)

#: The browser plugin's pytest name (`pytest-playwright`'s entry point), and
#: how a reason names it.
BROWSER_PLUGIN = "playwright"
BROWSER = "the browser plugin (pytest-playwright)"
#: How a reason names Node.
NODE = "Node (the host env's opt-in nodejs)"

#: EVERY FIXTURE THE BROWSER PLUGIN PROVIDES -- its public ones.  Declared
#: here because a machine without the plugin cannot ask it; checked against
#: the plugin itself where it is installed
#: (`test_missing_test_tooling.py`), so a plugin that grows one is noticed.
BROWSER_FIXTURES = frozenset({
    "browser", "browser_channel", "browser_context_args", "browser_name",
    "browser_type", "browser_type_launch_args", "connect_options", "context",
    "delete_output_dir", "device", "is_chromium", "is_firefox", "is_webkit",
    "launch_browser", "new_context", "output_path", "page", "playwright",
})


#: A need beside the envs: a visible NVIDIA GPU.
GPU = "an NVIDIA GPU"
#: Every need a test may state: an env named as its recipe is, or the GPU.
NEEDS = frozenset({HOST_ENV, *DEFAULT_ENV_NAMES.values(), GPU})


def known(stated) -> tuple:
    """``stated``, each one of :data:`NEEDS` -- or a ``ValueError`` naming
    the ones that are not."""
    unknown = sorted(set(stated) - NEEDS)
    if unknown:
        raise ValueError(f"not a need a test can state: {unknown}; a need is "
                         f"one of {sorted(NEEDS)}")
    return tuple(stated)


def unmet(need: str) -> Optional[str]:
    """Why ``need`` -- one of :data:`NEEDS` -- is not met on this machine:
    no GPU visible, or the env not in the env manager's list, naming the
    command that installs it; ``None`` when it is met."""
    from _road import env_available, gpu_visible
    known([need])
    if need == GPU:
        return None if gpu_visible() else "no NVIDIA GPU is visible here"
    if env_available(need):
        return None
    return f"the {need} env is not installed here: {install_cmd(need)}"


def why_unmet(stated) -> str:
    """Every unmet need's reason among ``stated``, each said once, joined --
    empty when all are met."""
    return "; ".join(dict.fromkeys(
        w for w in map(unmet, sorted(stated)) if w))


def needs(*stated: str):
    """The mark for a test, or a module, whose own body uses ``stated``: the
    collection hook skips it where one is unmet (:func:`unmet`)."""
    if not stated:
        raise TypeError("needs() names at least one need")
    return pytest.mark.needs(*known(stated))


def missing(what: str) -> str:
    """The one reason a test that needs ``what`` skips with -- "not
    available": not installed, or a run told not to load it."""
    return f"{what} is not available in this env ({sys.prefix}): {INSTALL}"


def node() -> Optional[Path]:
    """The env's own Node -- the host recipe's opt-in ``nodejs`` -- under the
    prefix of the env this suite runs in, whatever else is first on
    ``PATH``; ``None`` when it is not installed."""
    found = Path(sys.prefix) / "bin" / "node"
    return found if found.exists() else None


@functools.cache
def _host_requires() -> frozenset:
    from molbuilder.envs.recipes import recipe_by_name
    return frozenset(re.split(r"[=<>!~ ]", spec)[0]
                     for spec in recipe_by_name(HOST_ENV).conda_set())


def host_requires(package: str) -> bool:
    """Whether the host env requires ``package`` -- a `conda` line of
    `host-env.txt`, on every install (`process/testing.md` § 3): a test is
    never skipped for one."""
    return package in _host_requires()


def needs_backend(name: str) -> None:
    """A test needs the structure builder's ``name`` backend
    (`builders/backends`): one the host env requires (``rdkit``) is never
    skipped for -- missing, the build fails with the backend's own words,
    the env broken; a tool outside the env (``threedna``, ``amber``) not
    found here skips the test, naming it."""
    from molbuilder.builders.backends import available_backends
    if available_backends().get(name) or host_requires(name):
        return
    pytest.skip(f"the {name} builder backend -- a tool outside the env -- "
                "is not installed here")


def browser_plugin(config) -> bool:
    """Whether the browser plugin is loaded in this pytest -- what provides
    the browser fixtures, not merely the library under it."""
    return config.pluginmanager.hasplugin(BROWSER_PLUGIN)


def browser_fixtures_asked(item) -> frozenset:
    """The browser plugin's fixtures ``item`` asks for, directly or through
    another fixture -- not a name the test parametrizes itself, which is a
    value of its own (`test_frame_set_units` parametrizes a ``page``)."""
    own = set(getattr(getattr(item, "callspec", None), "params", {}) or {})
    return frozenset((set(getattr(item, "fixturenames", ())) - own)
                     & BROWSER_FIXTURES)
