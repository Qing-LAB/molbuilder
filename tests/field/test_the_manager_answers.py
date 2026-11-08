"""FIELD TEST -- one reading of the env manager's registry, on what this machine's
manager really answers (`ops/env-framework.md` § 2, `installation.md` M2).

Goal: does this env exist, and where -- one answer, from the gate, the state
machine and the doctor alike.  An env made here with ``--prefix`` outside the
manager's own folder is the case that decides it: the manager lists it, and an
env it lists is never reported FRESH, which is the one answer that makes a
duplicate env.

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


def test_the_doctor_says_present_what_the_manager_lists(manager, tmp_path,
                                                        machine_config):
    """`envs doctor`'s present column (`envs.doctor.report_all`), on what
    this machine's manager lists, under the name `molbuilder.json` gives a
    category (``envs.<category>``, `configuration.md` § 4): an env made here,
    renamed in for SIESTA, is present; a name nothing made, renamed in for
    PySCF, is missing.  THE SILENT FAILURES: the rename read past, so the
    doctor reports on an env the engines never enter; and an env reported
    present that the manager does not list.  Written in 2.67.11 and not run
    yet: it needs a probed target's record, so it first runs in the field
    pass on Sol (the plan's *Only the user's Sol runs can show*)."""
    from molbuilder.envs import doctor
    _make_env(manager, tmp_path / "scratch" / "mb-doctor")
    machine_config({"envs": {"siesta": "mb-doctor",
                             "pyscf": "mb-doctor-never-made"}})
    by = {r.recipe.category: r
          for r in doctor.report_all(D.detect(), run_verify=False)}
    assert (by["siesta"].effective_name, by["siesta"].present) == (
        "mb-doctor", True)
    assert (by["pyscf"].effective_name, by["pyscf"].present) == (
        "mb-doctor-never-made", False)


def test_an_env_the_snapshot_never_saw_is_still_looked_for(manager, tmp_path):
    """A snapshot that does not know an env is not evidence of absence: the env
    may have been created a second ago, by this very install."""
    set_capabilities(Capabilities(runtime_config={}, conda_binary=manager,
                                  conda_envs={}))
    fresh = tmp_path / "just-made"
    _make_env(manager, fresh)
    assert I._env_prefix("just-made", manager) == str(fresh)
