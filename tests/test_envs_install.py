"""L1 tests for ``molbuilder.envs.install``.

Plan generation + execution shape; no real conda commands.  We use
the SIESTA recipe (single conda package, no pip, has verify) and
the tests recipe (conda + pip + extra step) to cover the relevant
phase combinations.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install
from molbuilder.envs.recipes import recipe_by_name


def _bind(*, conda_envs=(), conda_binary="/usr/bin/conda"):
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary=conda_binary,
        conda_envs=frozenset(conda_envs),
    ))


# --------------------------------------------------------------------- #
#  plan_install: pure planner                                            #
# --------------------------------------------------------------------- #


def test_plan_includes_create_and_verify_for_siesta():
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    name, steps = install.plan_install(recipe)
    labels = [s.label for s in steps]
    assert "conda create" in labels
    assert "verify" in labels
    assert "pip install" not in labels  # siesta has no pip pkgs
    assert "extra" not in labels


def test_plan_includes_pip_and_extras_for_tests_recipe():
    _bind()
    recipe = recipe_by_name("molbuilder-tests")
    name, steps = install.plan_install(recipe)
    labels = [s.label for s in steps]
    assert labels == ["conda create", "pip install", "extra", "verify"]


def test_plan_conda_create_has_channels_in_order():
    """MDtools lists dacase before conda-forge -- order is load-
    bearing for the solver (dacase wins for ambertools-dac=26)."""
    _bind()
    recipe = recipe_by_name("molbuilder-MDtools")
    name, steps = install.plan_install(recipe)
    create = next(s for s in steps if s.label == "conda create")
    argv = list(create.argv)
    # Channels appear in declared order, each preceded by -c.
    chan_idx = [i for i, a in enumerate(argv) if a == "-c"]
    assert len(chan_idx) == 2
    assert argv[chan_idx[0] + 1] == "dacase"
    assert argv[chan_idx[1] + 1] == "conda-forge"


def test_plan_raises_without_conda_binary():
    _bind(conda_binary=None)
    recipe = recipe_by_name("molbuilder-siesta")
    with pytest.raises(RuntimeError, match="conda CLI not found"):
        install.plan_install(recipe)


# --------------------------------------------------------------------- #
#  run_install: execution                                                #
# --------------------------------------------------------------------- #


def _stub(returncode=0, stdout="", stderr=""):
    cp = MagicMock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def test_run_install_succeeds_when_all_steps_zero(monkeypatch):
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout="siesta 5.4.2"))
    result = install.run_install(recipe)
    assert result.succeeded is True
    assert result.recipe.name == "molbuilder-siesta"
    # Both create and verify ran.
    assert [s.label for s in result.steps] == ["conda create", "verify"]


def test_run_install_short_circuits_on_create_failure(monkeypatch):
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    calls = []
    def fake_run(*a, **kw):
        calls.append(a)
        # First call (create) fails; we should never see a second.
        return _stub(1, stderr="CondaPackagesNotFoundError")
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    result = install.run_install(recipe)
    assert result.succeeded is False
    assert len(calls) == 1, (
        "create failure must short-circuit; got "
        f"{len(calls)} subprocess.run calls"
    )


def test_run_install_skips_create_when_env_already_present(monkeypatch):
    """Idempotency: re-running install when the env exists should
    skip create and still run pip/extras/verify."""
    _bind(conda_envs=("molbuilder-tests",))
    recipe = recipe_by_name("molbuilder-tests")
    calls = []
    def fake_run(*a, **kw):
        calls.append(a)
        return _stub(0, stdout="Version 1.40")
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    result = install.run_install(recipe)
    assert result.succeeded is True
    # Create step is reported (it's in the plan) but its output marks
    # it as skipped.
    create = next(s for s in result.steps if s.label == "conda create")
    assert "already exists" in create.output
    # subprocess.run was called for pip + extra + verify (3 times),
    # not for create.
    assert len(calls) == 3


def test_run_install_verify_substring_failure_is_fatal(monkeypatch):
    """A verify step that exits 0 but lacks the expected substring
    must fail the install -- catches a silent regression where the
    binary is in the env but not the right binary."""
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    # First call (conda create) succeeds with a normal install log,
    # second (verify) returns 0 but no "siesta" substring.
    outputs = iter([
        _stub(0, stdout="solving..."),    # conda create
        _stub(0, stdout="oops wrong binary"),  # verify
    ])
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: next(outputs))
    result = install.run_install(recipe)
    assert result.succeeded is False
    verify_step = next(s for s in result.steps if s.label == "verify")
    assert "missing expected substring" in verify_step.output


def test_run_install_verify_ignore_exit_respects_substring(monkeypatch):
    """MDtools: verify exits 1 (tleap behaviour) but substring
    matches -> succeed.  Mirrors the production verify."""
    _bind()
    recipe = recipe_by_name("molbuilder-MDtools")
    outputs = iter([
        _stub(0, stdout="solving..."),  # conda create
        _stub(1, stdout="Welcome to LEaP!"),  # verify -> rc=1 but OK
    ])
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: next(outputs))
    result = install.run_install(recipe)
    assert result.succeeded is True
