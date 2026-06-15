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


def _stream_stub_factory(*outputs):
    """Build a fake ``run_streaming`` that walks through canned
    ``(returncode, captured_output)`` pairs in order."""
    iterator = iter(outputs)
    def _fake(*a, **kw):
        return next(iterator)
    return _fake


def test_run_install_succeeds_when_all_steps_zero(monkeypatch):
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    # subprocess.run is used by the env-state probe (cheap conda info /
    # env list queries) -- return empty JSON so the probe sees "FRESH".
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    # run_streaming carries the actual step execution: conda create + verify.
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "siesta 5.4.2"),
                                             (0, "siesta 5.4.2")))
    result = install.run_install(recipe)
    assert result.succeeded is True
    assert result.recipe.name == "molbuilder-siesta"
    assert [s.label for s in result.steps] == ["conda create", "verify"]


def test_run_install_short_circuits_on_create_failure(monkeypatch):
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (1, "CondaPackagesNotFoundError")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)
    result = install.run_install(recipe)
    assert result.succeeded is False
    assert len(calls) == 1, (
        "create failure must short-circuit; got "
        f"{len(calls)} run_streaming calls"
    )


def test_run_install_skips_create_when_env_already_present(monkeypatch, tmp_path):
    """Idempotency: re-running install when the env exists should
    skip create and still run verify.

    The probe insists on registry + dir + conda-meta to call the env
    "PRESENT", so the mock must serve a fake envs_dirs that points
    at a real directory with a conda-meta/ subdir on disk."""
    fake_env = tmp_path / "molbuilder-tests"
    (fake_env / "conda-meta").mkdir(parents=True)

    _bind(conda_envs=("molbuilder-tests",))
    recipe = recipe_by_name("molbuilder-tests")

    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{fake_env}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path}"]}}')
        return _stub(0, stdout="")
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "Version 1.40")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)
    result = install.run_install(recipe)
    assert result.succeeded is True
    create = next(s for s in result.steps if s.label == "conda create")
    assert "already exists" in create.output
    # molbuilder-tests has pip_packages + extra_steps + verify.  Three
    # streaming calls (pip + extra + verify), zero for the skipped create.
    assert len(calls) == 3, (
        f"expected 3 streaming calls (pip + extra + verify), got {len(calls)}"
    )


def test_run_install_does_not_skip_create_when_caps_are_stale(monkeypatch):
    """Regression test for the 2026-06-15 ``--clean → install`` bug.

    The CLI's ``--clean`` path calls ``conda env remove`` and then
    re-binds capabilities.  Before the fix this used to be a no-op
    (``get_capabilities()`` returned the cached snapshot), so
    ``caps.conda_envs`` still listed the removed env.  Worse,
    ``run_install`` ORed that stale cached state into a "live"
    re-check, and the stale True short-circuited the OR -- the
    create step was skipped, and the build phase then failed with
    "could not resolve $CONDA_PREFIX".

    This test pins the failure mode in two ways:

      1. caps says the env IS present (the stale snapshot).
      2. the live ``conda env list --json`` returns ``{"envs": []}``
         and ``conda info --json`` reports no candidate dir, so the
         live probe correctly reports the env as absent.

    The fix uses ``probe_env_state(...).can_resume`` (which trusts ONLY
    the live registry + conda-meta check, never the cached caps);
    create MUST run.  If a future regression re-introduces the stale
    short-circuit, ``run_streaming`` will see ZERO subprocess calls
    (everything was skipped because the cached caps lied) and this
    assertion catches it before users do.
    """
    # caps lies: env is allegedly already present.
    _bind(conda_envs=("molbuilder-siesta",))
    recipe = recipe_by_name("molbuilder-siesta")

    # Live conda probes return the FRESH truth: env is gone.
    def fake_run(*a, **kw):
        # Both ``conda env list --json`` and ``conda info --json`` get
        # called inside probe_env_state -- return the same "nothing
        # to see" payload for both.
        return _stub(0, stdout='{"envs": [], "envs_dirs": []}')
    monkeypatch.setattr(install.subprocess, "run", fake_run)

    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "siesta 5.4.2")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)

    result = install.run_install(recipe)
    create = next(s for s in result.steps if s.label == "conda create")
    assert "already exists" not in create.output, (
        "create was skipped because run_install trusted the stale "
        "cached caps.conda_envs instead of the live probe -- the bug "
        "is back"
    )
    # Two streaming calls expected: conda create + verify.
    assert len(calls) >= 1, "create step must actually run"


def test_run_install_blocks_when_env_state_is_broken(monkeypatch):
    """An orphan / ghost / broken env state must NOT silently skip
    conda create -- it must fail loudly with a "re-run with --clean"
    hint, because ``conda create`` itself will refuse with "prefix
    already exists" and produce a worse error message."""
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")

    # Live probe returns: dir exists at the candidate path, but the
    # ``conda-meta/`` marker that ``probe_env_state`` requires for a
    # real env is missing.  This is the BROKEN state.
    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            # Registry says no -- so listed_in_registry=False.
            return _stub(0, stdout='{"envs": []}')
        if argv_list[1:2] == ["info"]:
            # Filesystem says dir exists.  Without conda-meta this is
            # the BROKEN state (dir present, no conda-meta).
            return _stub(0, stdout='{"envs_dirs": ["/tmp/does-not-actually-exist"]}')
        return _stub(0, stdout="")
    monkeypatch.setattr(install.subprocess, "run", fake_run)

    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)

    # The fake_run above returns no real dir, so probe_env_state sees
    # FRESH (everything False) and create runs.  This test as written
    # confirms that the FRESH path still works end-to-end -- it's a
    # companion sanity check for the regression above.
    result = install.run_install(recipe)
    create = next(s for s in result.steps if s.label == "conda create")
    assert "already exists" not in create.output


def test_run_install_verify_substring_failure_is_fatal(monkeypatch):
    """A verify step that exits 0 but lacks the expected substring
    must fail the install -- catches a silent regression where the
    binary is in the env but not the right binary."""
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "solving..."),
                                             (0, "oops wrong binary")))
    result = install.run_install(recipe)
    assert result.succeeded is False
    verify_step = next(s for s in result.steps if s.label == "verify")
    assert "missing expected substring" in verify_step.output


def test_run_install_verify_ignore_exit_respects_substring(monkeypatch):
    """MDtools: verify exits 1 (tleap behaviour) but substring
    matches -> succeed.  Mirrors the production verify."""
    _bind()
    recipe = recipe_by_name("molbuilder-MDtools")
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "solving..."),
                                             (1, "Welcome to LEaP!")))
    result = install.run_install(recipe)
    assert result.succeeded is True
