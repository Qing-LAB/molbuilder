"""FIELD TEST -- one reading of the env manager's registry, on what this machine's
manager really answers (`ops/env-framework.md` § 2, `installation.md` M2).

Goal: does this env exist, and where -- one answer, from the gate and from the
state machine alike.  An env made here with ``--prefix`` outside the manager's
own folder is the case that decides it: the manager lists it, and an env it lists
is never reported FRESH, which is the one answer that makes a duplicate env.

The env is empty, and made under this test's own home and folder, so nothing of
the machine's changes.

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import json
import subprocess

import pytest

from molbuilder import diagnostics as D
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install as I


def _make_env(manager, prefix):
    done = subprocess.run(
        [manager, "create", "-y", "-q", "--offline", "-p", str(prefix)],
        capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr


@pytest.fixture
def manager():
    """The one the product would use."""
    found, _source = D._find_conda_binary()
    if not found:
        pytest.skip("no env manager on this machine")
    return found


def test_an_env_the_manager_lists_has_one_answer(manager, tmp_path):
    made = tmp_path / "scratch" / "mb-gpu"
    _make_env(manager, made)
    listed = json.loads(subprocess.run(
        [manager, "env", "list", "--json"], check=True, capture_output=True,
        text=True, timeout=120).stdout)
    assert str(made) in listed["envs"], "the manager did not list what it made"

    got = D.conda_env_prefixes(manager)
    assert got["mb-gpu"] == str(made)
    # The installation is listed by the registry and is not an env.  A manager
    # reporting no details cannot say which it is, so nothing is excluded: an
    # env it lists and this map omits would read as FRESH, and be made twice.
    if listed.get("envs_details"):
        for prefix, details in listed["envs_details"].items():
            if details.get("base"):
                assert prefix not in got.values(), got
    else:
        assert sorted(got.values()) == sorted(listed["envs"]), got

    set_capabilities(Capabilities(runtime_config={}, conda_binary=manager,
                                  conda_envs=got))
    caps = D.get_capabilities()
    state = I.probe_env_state("mb-gpu", manager)
    assert caps.env_available("mb-gpu") is True
    assert state.state is I.EnvPresence.PRESENT, state.describe()
    assert caps.env_prefix("mb-gpu") == state.prefix == str(made)


def test_an_env_the_snapshot_never_saw_is_still_looked_for(manager, tmp_path):
    """A snapshot that does not know an env is not evidence of absence: the env
    may have been created a second ago, by this very install."""
    set_capabilities(Capabilities(runtime_config={}, conda_binary=manager,
                                  conda_envs={}))
    fresh = tmp_path / "just-made"
    _make_env(manager, fresh)
    assert I._env_prefix("just-made", manager) == str(fresh)
