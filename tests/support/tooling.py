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
"""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from typing import Optional

import pytest

from molbuilder.envs.hints import opt_in_cmd

#: The command that installs every piece of the test tooling -- the host
#: env's opt-in records, as `doctor` names it (`envs.hints.opt_in_cmd`).
INSTALL = opt_in_cmd("molbuilder")

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
                     for spec in recipe_by_name("molbuilder").conda_set())


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
