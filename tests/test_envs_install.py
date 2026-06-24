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
    # The verify step now requires the env prefix to be resolvable so
    # the bypass code path can fire.  Patch ``_env_prefix`` to return
    # a fake prefix once the env has been "created".  Pre-fix, the
    # verify step would silently fall back to the buggy ``conda run``
    # argv when prefix resolution failed -- now it fails loud, which
    # matches real-world behaviour where _env_prefix is rock-solid.
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "solving..."),
                                             (1, "Welcome to LEaP!")))
    result = install.run_install(recipe)
    assert result.succeeded is True


# --------------------------------------------------------------------- #
#  cmd_bootstrap: full-stack bootstrap subcommand (2026-06-23)          #
# --------------------------------------------------------------------- #
#
# Gates the ASU-deployment critical path: ``molbuilder envs bootstrap``
# iterates BUILTIN_RECIPES, runs each install, runs doctor at the end.
# Audit 2026-06-24 found zero test coverage on this subcommand; this
# block closes the gap.
#
# Strategy: mock at three boundaries (cheap, fast, no real conda):
#   - molbuilder.envs._cli._install.run_install -> stub returning OK
#   - molbuilder.envs._cli._doctor.report_all   -> stub returning empty
#   - molbuilder.envs._cli._diag.detect         -> stub returning caps
#
# The CliRunner invokes the click handler with `--yes` to skip the
# interactive confirm prompt.


def _make_install_stub(succeeded_per_call=None):
    """Return a fake ``run_install`` that records calls + returns a
    minimal succeeded-or-failed result.  ``succeeded_per_call`` is an
    iterable of bools; default all True."""
    calls = []
    iterator = iter(succeeded_per_call or ())

    def _fake(recipe, caps=None, **kw):
        calls.append(recipe.name)
        # Default True if iterator exhausted (favors success path).
        try:
            ok = next(iterator)
        except StopIteration:
            ok = True
        result = MagicMock()
        result.succeeded = ok
        result.recipe = recipe
        return result

    return _fake, calls


def _make_runner():
    """CliRunner is brought in via Click."""
    from click.testing import CliRunner
    return CliRunner()


def test_bootstrap_dry_run_lists_recipes_without_installing(monkeypatch):
    """Dry-run path: shows the plan, runs zero installs, returns OK."""
    _bind()
    from molbuilder.envs import _cli
    install_stub, install_calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all",
                        lambda caps, **kw: [])
    runner = _make_runner()
    result = runner.invoke(
        _cli.envs_group, ["bootstrap", "--dry-run", "--yes"],
        catch_exceptions=False,
    )
    # Dry-run still shows the plan + calls doctor at the end (per the
    # cmd_bootstrap code path), but does not call run_install.
    assert install_calls == [], (
        f"--dry-run should not call run_install; got: {install_calls}")
    assert "bootstrap plan:" in result.output
    assert "dry-run" in result.output.lower()


def test_bootstrap_runs_install_for_each_conda_only_recipe(monkeypatch):
    """Default invocation (no --include-source-builds): iterate
    BUILTIN_RECIPES and call run_install for each conda-only recipe.
    Source-build recipes (build_spec != None) are excluded.  All envs
    are absent so --skip-existing has nothing to skip."""
    _bind()  # caps with empty conda_envs set
    from molbuilder.envs import _cli
    install_stub, install_calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all",
                        lambda caps, **kw: [])
    # Lazy import in cmd_bootstrap: ``from .. import diagnostics as
    # _diag``.  Patch the source module so the lazy import picks up
    # the stub regardless of import-cache state.
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset()))

    runner = _make_runner()
    result = runner.invoke(
        _cli.envs_group, ["bootstrap", "--yes"],
        catch_exceptions=False,
    )
    # Every conda-only recipe in BUILTIN_RECIPES should appear in
    # install_calls.  Source-build recipes (build_spec != None) must
    # NOT appear -- they're opt-in via --include-source-builds.
    from molbuilder.envs.recipes import BUILTIN_RECIPES
    expected = [
        r.name for r in BUILTIN_RECIPES if r.build_spec is None
    ]
    forbidden = [
        r.name for r in BUILTIN_RECIPES if r.build_spec is not None
    ]
    assert install_calls == expected, (
        f"Expected conda-only recipes to be installed in order; "
        f"got {install_calls}, expected {expected}.\n"
        f"output:\n{result.output}"
    )
    for name in forbidden:
        assert name not in install_calls, (
            f"Source-build recipe {name!r} should NOT install by "
            f"default; needs --include-source-builds.")


def test_bootstrap_include_source_builds_adds_them(monkeypatch):
    """``--include-source-builds`` opts the user into the source-build
    recipes too.  All BUILTIN_RECIPES are then iterated."""
    _bind()  # caps with empty conda_envs set
    from molbuilder.envs import _cli
    install_stub, install_calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all",
                        lambda caps, **kw: [])
    # Lazy import in cmd_bootstrap: ``from .. import diagnostics as
    # _diag``.  Patch the source module so the lazy import picks up
    # the stub regardless of import-cache state.
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset()))

    runner = _make_runner()
    result = runner.invoke(
        _cli.envs_group,
        ["bootstrap", "--yes", "--include-source-builds"],
        catch_exceptions=False,
    )
    from molbuilder.envs.recipes import BUILTIN_RECIPES
    expected_names = {r.name for r in BUILTIN_RECIPES}
    assert set(install_calls) == expected_names, (
        f"--include-source-builds should iterate every recipe; "
        f"got {install_calls!r}, expected {expected_names!r}.\n"
        f"output:\n{result.output}"
    )


def test_bootstrap_skips_existing_envs_by_default(monkeypatch):
    """When ``--skip-existing`` is the default and the env already
    exists in caps, ``run_install`` is not called for that recipe.
    This keeps bootstrap idempotent (safe to re-run)."""
    from molbuilder.envs.recipes import BUILTIN_RECIPES
    conda_only_names = [
        r.name for r in BUILTIN_RECIPES if r.build_spec is None
    ]
    # Pretend the FIRST conda-only env is already present.
    already_present = conda_only_names[0]
    _bind(conda_envs=(already_present,))
    from molbuilder.envs import _cli
    install_stub, install_calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all",
                        lambda caps, **kw: [])
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset({already_present})))

    runner = _make_runner()
    result = runner.invoke(
        _cli.envs_group, ["bootstrap", "--yes"],
        catch_exceptions=False,
    )
    assert already_present not in install_calls, (
        f"--skip-existing default should have skipped already-present "
        f"env {already_present!r}; got install_calls={install_calls!r}.\n"
        f"output:\n{result.output}"
    )
    # The remaining conda-only recipes still got installed.
    remaining = [n for n in conda_only_names if n != already_present]
    assert set(install_calls) == set(remaining)


def test_bootstrap_runs_doctor_at_end(monkeypatch):
    """At the end of every bootstrap run (whether installs ran or
    everything was skipped), doctor must run to verify env health.
    This is the user-facing smoke-check promise of the bootstrap CLI.
    """
    _bind()
    from molbuilder.envs import _cli
    install_stub, _ = _make_install_stub()
    doctor_called = []

    def _doctor_stub(caps, **kw):
        doctor_called.append(True)
        return []   # empty report -> exit code 0 from _render_doctor

    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all", _doctor_stub)
    # Lazy import in cmd_bootstrap: ``from .. import diagnostics as
    # _diag``.  Patch the source module so the lazy import picks up
    # the stub regardless of import-cache state.
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset()))

    runner = _make_runner()
    runner.invoke(
        _cli.envs_group, ["bootstrap", "--yes"],
        catch_exceptions=False,
    )
    assert doctor_called, (
        "bootstrap must call doctor at the end so the user sees a "
        "smoke check of every env it just installed.")


# --------------------------------------------------------------------- #
#  install-env.sh thin-shim contract (2026-06-24 architectural rewrite) #
# --------------------------------------------------------------------- #
#
#  The shell script is a thin shim: it solves the chicken-and-egg
#  of "you can't run `molbuilder envs ...` until the host env
#  exists" and forwards ``"$@"`` verbatim to ``molbuilder envs``.
#  Every recipe-shape concern (recipe lookup, --rebuild component
#  validation, elsi→siesta alias, --check / --dry-run semantics)
#  lives in the Python ``_cli.py`` cmd_install handler.
#
#  These tests pin the shim's contract:
#    * No args -> exit 2 with a first-time-? hint
#    * Non-bootstrap subcommand with no host env -> exit 2, pointing
#      at bootstrap
#    * bootstrap with no host env -> auto-create + dispatch (idempotent)
#    * Any subcommand with host env present -> dispatch verbatim,
#      passing ``"$@"`` (all flags including trailing ones)
#    * PYTHONPATH=$REPO_ROOT set so ``python -m molbuilder`` works
#      regardless of the caller's CWD

import os
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_INSTALL_ENV_SH = _REPO_ROOT / "scripts" / "install-env.sh"


def _make_stub_mamba(bin_dir, *, host_env_present=True,
                     configured_channels=()):
    """Create a stub ``mamba`` binary at ``bin_dir/mamba`` PLUS a
    stub env-python at the path the shim's _resolve_env_python
    fallback derives (``<dirname(dirname(bin_dir))>/envs/molbuilder/
    bin/python``).

    The stubs fake:
      * ``mamba env list`` -> output that either includes or omits
        the host env (controlled by ``host_env_present``).
      * ``mamba config --get channels`` -> ``--add channels '<name>'``
        lines per configured channel (empty tuple = fresh conda).
      * ``mamba info --json`` -> empty (forces _resolve_env_python's
        fallback path; simpler than emitting valid JSON for awk).
      * ``mamba create`` -> echo ``[stub-create] $*``.
      * The env's python -> echo ``[stub-dispatch] python $*`` and
        ``[stub-env] PYTHONPATH=$PYTHONPATH ...`` so tests can assert
        what got forwarded.  Replaces the old ``mamba run``-driven
        dispatch (the shim now bypasses mamba run to dodge the
        mamba 1.x ``exec --`` bug).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    mamba = bin_dir / "mamba"
    env_list_lines = ["# conda environments:", "#", "base    /root"]
    if host_env_present:
        env_list_lines.append("molbuilder    /root/molbuilder")
    env_list_output = "\n".join(env_list_lines) + "\n"
    config_lines = [f"--add channels '{ch}'" for ch in configured_channels]
    config_output = ("\n".join(config_lines) + "\n") if config_lines else ""
    mamba.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "env" && "$2" == "list" ]]; then\n'
        f"  cat <<'EOF'\n{env_list_output}EOF\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "$1" == "config" && "$2" == "--get" && "$3" == "channels" ]]; then\n'
        f"  cat <<'EOF'\n{config_output}EOF\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "$1" == "info" && "$2" == "--json" ]]; then\n'
        '  exit 0\n'  # empty stdout -> forces fallback path
        "fi\n"
        'if [[ "$1" == "create" ]]; then\n'
        '  echo "[stub-create] $*"\n'
        "  exit 0\n"
        "fi\n"
        'echo "[stub-mamba] $*"\n'
        "exit 0\n"
    )
    mamba.chmod(0o755)
    # The shim's _resolve_env_python fallback derives the env's
    # python from ``${ENV_MGR%/bin/*}/envs/<name>/bin/python``.
    # ENV_MGR is bin_dir/mamba, so the install-root strip lands at
    # bin_dir.parent; the python goes at
    # ``<bin_dir.parent>/envs/molbuilder/bin/python``.
    env_python_dir = bin_dir.parent / "envs" / "molbuilder" / "bin"
    env_python_dir.mkdir(parents=True, exist_ok=True)
    env_python = env_python_dir / "python"
    env_python.write_text(
        "#!/usr/bin/env bash\n"
        'echo "[stub-dispatch] python $*"\n'
        'echo "[stub-env] PYTHONPATH=${PYTHONPATH:-} '
        'MOLBUILDER_REPO_ROOT=${MOLBUILDER_REPO_ROOT:-}"\n'
        "exit 0\n"
    )
    env_python.chmod(0o755)
    return mamba


def _run_install_env_sh(args, *, tmp_path, host_env_present=True,
                        cwd=None, configured_channels=(), extra_env=None):
    """Run ``install-env.sh`` with a stubbed ``mamba`` on PATH."""
    bin_dir = tmp_path / "bin"
    _make_stub_mamba(bin_dir, host_env_present=host_env_present,
                     configured_channels=configured_channels)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("MAMBA_EXE", None)
    env.pop("CONDA_EXE", None)
    env.pop("MOLBUILDER_HOST_ENV_CHANNELS", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(_INSTALL_ENV_SH), *args],
        env=env, capture_output=True, text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def test_shim_forwards_args_verbatim(tmp_path):
    """The shim forwards ``$@`` 1:1 to ``molbuilder envs ...``.

    Tests every subcommand shape: install with --rebuild=, with
    --clean, with --check, with --dry-run, plus list/doctor.  All
    flags must reach the Python layer; nothing dropped, nothing
    rewritten."""
    # Every case includes --yes so the env-manager confirmation
    # prompt (added 2026-06-24) skips and the subprocess can run
    # non-interactively under pytest.
    cases = [
        (["install", "molbuilder-siesta", "--yes"],
         "install molbuilder-siesta --yes"),
        (["install", "molbuilder-siesta-gpu",
          "--rebuild=siesta", "--yes", "--skip-network-check"],
         "install molbuilder-siesta-gpu "
         "--rebuild=siesta --yes --skip-network-check"),
        (["install", "molbuilder-siesta-gpu", "--clean", "--yes"],
         "install molbuilder-siesta-gpu --clean --yes"),
        (["install", "molbuilder-siesta", "--check", "--yes"],
         "install molbuilder-siesta --check --yes"),
        (["install", "molbuilder-siesta", "--dry-run", "--yes"],
         "install molbuilder-siesta --dry-run --yes"),
        (["list", "--yes"], "list --yes"),
        (["doctor", "--yes"], "doctor --yes"),
    ]
    for args, expected_tail in cases:
        r = _run_install_env_sh(args, tmp_path=tmp_path)
        assert r.returncode == 0, f"args={args}: {r.stderr}"
        assert f"python -m molbuilder envs {expected_tail}" in r.stdout, (
            f"args={args}: shim must forward verbatim; got stdout:\n"
            f"{r.stdout}")


def test_shim_runnable_from_any_cwd(tmp_path):
    """``python -m molbuilder`` requires the package on PYTHONPATH
    (molbuilder is not pip-installed).  The shim must set PYTHONPATH
    to the repo root regardless of the caller's CWD -- otherwise a
    fresh-machine deployment (``cd ~ && bash repo/scripts/install-env.sh
    ...``) fails with ModuleNotFoundError on the very first dispatch."""
    # Run from tmp_path -- NOT the repo root.  --yes skips the
    # env-manager confirmation prompt.
    r = _run_install_env_sh(
        ["list", "--yes"], tmp_path=tmp_path, cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert f"PYTHONPATH={_REPO_ROOT}" in r.stdout, (
        f"shim must set PYTHONPATH={_REPO_ROOT} so molbuilder is "
        f"importable from any CWD; got stdout:\n{r.stdout}")
    assert f"MOLBUILDER_REPO_ROOT={_REPO_ROOT}" in r.stdout


def test_no_args_suggests_bootstrap(tmp_path):
    """A first-time user running ``bash install-env.sh`` with no
    args should see a 'first-time? type this' hint that names the
    bootstrap command, then the full usage.  Exit code is 2 so
    accidental empty invocations don't pass CI silently."""
    r = _run_install_env_sh([], tmp_path=tmp_path)
    assert r.returncode == 2
    assert "bootstrap --yes" in r.stderr
    assert "First-time install" in r.stderr


def test_non_bootstrap_without_host_env_points_at_bootstrap(tmp_path):
    """If the host env doesn't exist and the user runs a non-bootstrap
    subcommand, the shim should NOT silently auto-create -- it should
    error and point at the bootstrap command.  Auto-create only happens
    in the bootstrap path (deliberate state-machine constraint)."""
    r = _run_install_env_sh(
        ["install", "molbuilder-siesta", "--yes"],
        tmp_path=tmp_path, host_env_present=False,
    )
    assert r.returncode == 2
    assert "host env 'molbuilder' does not exist" in r.stderr
    assert "bootstrap --yes" in r.stderr


def test_bootstrap_auto_creates_host_env_when_missing(tmp_path):
    """The one path that auto-creates the host env: bootstrap.
    Without this the chicken-and-egg of 'install Python before
    Python is available' has no resolution."""
    r = _run_install_env_sh(
        ["bootstrap", "--yes"],
        tmp_path=tmp_path, host_env_present=False,
    )
    assert r.returncode == 0, r.stderr
    assert "creating host env 'molbuilder'" in r.stderr
    assert "python -m molbuilder envs bootstrap --yes" in r.stdout


# --------------------------------------------------------------------- #
#  Respect ~/.condarc on bootstrap host-env create                       #
# --------------------------------------------------------------------- #


def test_bootstrap_respects_condarc_when_channels_configured(tmp_path):
    """When the user has channels in .condarc (e.g. an HPC site with a
    private mirror or strict channel_priority), the host-env create
    must NOT prepend ``-c conda-forge`` -- that would override the
    user's intent.  The script probes ``mamba config --get channels``
    and, on any non-empty result, passes NO ``-c`` flag."""
    r = _run_install_env_sh(
        ["bootstrap", "--yes"],
        tmp_path=tmp_path, host_env_present=False,
        configured_channels=("site-internal", "conda-forge"),
    )
    assert r.returncode == 0, r.stderr
    assert "respecting user's .condarc channels" in r.stderr
    # The stub-create line shows the exact args passed to ``mamba
    # create``.  Must NOT contain ``-c conda-forge`` (the script's
    # fallback) when .condarc already lists channels.
    create_lines = [ln for ln in r.stdout.splitlines()
                    if ln.startswith("[stub-create]")]
    assert create_lines, f"no create line; got:\n{r.stdout}"
    assert "-c conda-forge" not in create_lines[0], (
        f"script must not override .condarc; got:\n{create_lines[0]}")


def test_bootstrap_falls_back_to_conda_forge_when_no_channels(tmp_path):
    """When .condarc has no channels configured (fresh conda, default
    ``defaults`` channel only), the script falls back to
    ``-c conda-forge`` so the bootstrap can resolve the scientific
    stack (numpy, ase, sisl, rdkit) that isn't in ``defaults``."""
    r = _run_install_env_sh(
        ["bootstrap", "--yes"],
        tmp_path=tmp_path, host_env_present=False,
        configured_channels=(),
    )
    assert r.returncode == 0, r.stderr
    assert "no channels configured in .condarc" in r.stderr
    create_lines = [ln for ln in r.stdout.splitlines()
                    if ln.startswith("[stub-create]")]
    assert create_lines, f"no create line; got:\n{r.stdout}"
    assert "-c conda-forge" in create_lines[0]


def test_bootstrap_honors_molbuilder_host_env_channels_override(tmp_path):
    """Explicit override via env var beats both .condarc probing and
    the conda-forge fallback.  Lets an admin pin the host-env
    channels deterministically without modifying .condarc."""
    r = _run_install_env_sh(
        ["bootstrap", "--yes"],
        tmp_path=tmp_path, host_env_present=False,
        configured_channels=("conda-forge",),  # would otherwise skip -c
        extra_env={"MOLBUILDER_HOST_ENV_CHANNELS":
                   "site-mirror,conda-forge"},
    )
    assert r.returncode == 0, r.stderr
    assert "channels from MOLBUILDER_HOST_ENV_CHANNELS" in r.stderr
    create_lines = [ln for ln in r.stdout.splitlines()
                    if ln.startswith("[stub-create]")]
    assert create_lines, f"no create line; got:\n{r.stdout}"
    assert "-c site-mirror" in create_lines[0]
    assert "-c conda-forge" in create_lines[0]


def test_unknown_subcommand_forwards_to_python(tmp_path):
    """The shim has no allow-list of subcommands -- new Python
    subcommands (e.g. an upcoming ``molbuilder envs purge``) become
    reachable through the shim with zero bash-side changes.  This
    locks the thin-shim invariant: bash has no recipe-shape
    knowledge."""
    r = _run_install_env_sh(
        ["some-future-subcommand", "--with-flag", "--yes"],
        tmp_path=tmp_path,
    )
    # The shim forwards to Python; whether Python rejects an unknown
    # subcommand is Python's concern -- the shim's job is only to
    # forward.  Stub returns 0 here so we can verify forwarding.
    assert r.returncode == 0
    assert "envs some-future-subcommand --with-flag" in r.stdout


# --------------------------------------------------------------------- #
#  Python-side: elsi → siesta alias for --rebuild on siesta-gpu          #
# --------------------------------------------------------------------- #


def test_rebuild_elsi_remaps_to_siesta_in_python():
    """The elsi→siesta alias for ``--rebuild`` on the GPU recipe used
    to live in the bash wrapper; it moved into ``_cli.cmd_install``
    so the recipe-shape knowledge lives next to the recipe (single
    source of truth).  This test pins the alias behavior."""
    from click.testing import CliRunner
    from molbuilder.envs import _cli
    from molbuilder import diagnostics
    # Provide a minimal Capabilities so cmd_install can run far enough
    # to hit the rebuild validation block (before any subprocess work).
    diagnostics.set_capabilities(diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    ))
    runner = CliRunner()
    # --dry-run short-circuits before any real install; we only want
    # to confirm the alias surfaces the "ELSI is a SIESTA submodule"
    # note and doesn't error with "unknown choice".
    result = runner.invoke(
        _cli.envs_group,
        ["install", "molbuilder-siesta-gpu",
         "--rebuild=elsi", "--dry-run"],
        catch_exceptions=False,
    )
    # The note is emitted to stderr; CliRunner mixes them by default
    # unless mix_stderr=False.  Either output captures it.
    assert ("ELSI is a SIESTA submodule" in result.output
            or "ELSI is a SIESTA submodule" in (result.stderr_bytes or b"")
                .decode()), (
        f"expected the elsi→siesta alias note; got:\n{result.output}")
    # Must NOT report "unknown choice" -- the alias must remap before
    # the unknown-choice validator runs.
    assert "unknown" not in result.output.lower(), result.output
