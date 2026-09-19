"""L1 tests for ``molbuilder.envs.install``.

Plan generation + execution shape; no real conda commands.  We use
the SIESTA recipe (single conda package, no pip, has verify) for the
create+verify path and a SYNTHETIC recipe (conda + pip + extra step +
verify) for the all-phases path.  The synthetic recipe is used instead
of a registry entry because no built-in recipe currently declares an
``extra_steps`` install phase -- the planner logic under test is
recipe-shape-driven, so an in-test recipe is the honest fixture.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from molbuilder import diagnostics as _diag
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install
from molbuilder.envs.recipes import (CondaPackage, PipPackage, Recipe,
                                     recipe_by_name)


# A recipe that exercises EVERY install phase: conda create + pip
# install + an extra step + verify.  Mirrors the shape the retired
# browser-E2E env used to provide.
_ALL_PHASES_RECIPE = Recipe(
    name="synth-install-env",
    category=None,
    description="Synthetic recipe: conda + pip + extra + verify.",
    channels=("conda-forge",),
    conda_packages=("python=3.12", "pip"),
    pip_packages=(PipPackage("some-pip-only-tool"),),
    extra_steps=(("python", "-m", "some_tool", "post-install"),),
    verify_argv=("some-tool", "--version"),
    verify_expect_contains="Version",
)


def _bind(*, conda_envs=(), conda_binary="/usr/bin/conda"):
    """Bind a synthetic snapshot.

    `conda_envs` may be a mapping of ``{name: prefix}`` -- which is what the
    real snapshot carries since 2026-09-12 -- or a bare sequence of names for
    the tests that only ask whether an env exists.  Wrapping a mapping in
    `frozenset` threw the prefixes away, and `_env_prefix` then fell through to
    a live registry read against a fake manager path.
    """
    envs_arg = (dict(conda_envs) if isinstance(conda_envs, dict)
                else {name: f"/prefix/{name}" for name in conda_envs})
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary=conda_binary,
        conda_envs=envs_arg,
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


def test_plan_includes_pip_and_extras_for_all_phases_recipe():
    _bind()
    recipe = _ALL_PHASES_RECIPE
    name, steps = install.plan_install(recipe)
    labels = [s.label for s in steps]
    # `conda install` is its own step so that an env which ALREADY EXISTS
    # can still receive a package the recipe gained.
    assert labels == ["conda create", "conda install", "pip install",
                      "extra", "verify"]


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


# The manager is faked at `diagnostics.subprocess.run`: both readers of the
# manager's documents -- `conda_env_prefixes` (env list --json) and
# `manager_info` (info --json) -- live there since 2026-09-13.  Faking
# `install.subprocess` intercepted only the second, and only until that reader
# moved.
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
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    # The verify step now requires the env prefix to be resolvable so
    # the bypass code path can fire.  Patch ``_env_prefix`` to return
    # a fake prefix once the env has been "created".  Pre-fix, the
    # verify step would silently fall back to the buggy ``conda run``
    # argv when prefix resolution failed -- now it fails loud, which
    # matches real-world behaviour where _env_prefix is rock-solid.
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    # run_streaming carries the step execution: create + the conda set +
    # verify.  The prefix above does not exist, so the audit gate finds every
    # declared package absent and the set is dispatched.
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "siesta 5.4.2"),
                                             (0, "siesta 5.4.2"),
                                             (0, "siesta 5.4.2")))
    result = install.run_install(recipe)
    assert result.succeeded is True
    assert result.recipe.name == "molbuilder-siesta"
    assert [s.label for s in result.steps] == ["conda create", "conda install",
                                               "verify"]


def test_conda_set_is_not_dispatched_when_the_env_already_has_it(
        monkeypatch, tmp_path):
    """A CURRENT env pays nothing -- `env-framework.md` § 4.4's other branch.

    The gate exists because an unconditional `conda install` is not a no-op.
    Measured 2026-09-17 on conda 26.7.1: the full declared list against a
    SATISFIED `molbuilder-siesta` still took 26 s and wanted to update
    `ca-certificates` and `openssl` to newer builds -- so installing on every
    run would drift a healthy env each time anyone re-ran bootstrap.

    Here `conda-meta/` carries a record for every declared package, so the
    step is SKIPPED and no solve is dispatched for it.
    """
    recipe = _ALL_PHASES_RECIPE
    fake_env = tmp_path / recipe.name
    meta = fake_env / "conda-meta"
    meta.mkdir(parents=True)
    for spec in recipe.conda_set():
        name = spec.split("=")[0]
        (meta / f"{name}-1.0-h0.json").write_text(
            json.dumps({"name": name, "version": "1.0", "build": "h0"}))

    _bind(conda_envs={recipe.name: str(fake_env)})

    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{fake_env}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path}"]}}')
        return _stub(0, stdout="")
    monkeypatch.setattr(_diag.subprocess, "run", fake_run)
    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "Version 1.40")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)

    result = install.run_install(recipe)
    assert result.succeeded is True
    conda_set = next(s for s in result.steps if s.label == "conda install")
    assert conda_set.outcome is install.Outcome.SKIPPED
    assert conda_set.returncode is None, (
        "a step that did not run has no exit code")
    assert "already installed" in conda_set.output
    # A CONDA-level install -- `<mgr> install ...` -- not the word anywhere.
    # `python -m pip install` is dispatched here too and is not a solve.
    dispatched = [a[0] for a in calls]
    solves = [argv for argv in dispatched if list(argv)[1:2] == ["install"]]
    assert not solves, (
        f"no conda solve may be dispatched for a satisfied env; got {solves}")


def test_opt_in_is_the_difference_between_the_two_plans():
    """One claim, both directions: `opt_in` is exactly what `--with-dev-tools`
    adds and a default run leaves out -- across all three kinds.

    The host env's test tooling is the case: a conda package (nodejs), two pip
    packages (playwright, pytest-playwright) and an extra step (the chromium
    download).  It was four hand-typed commands in `install-env.sh --help`
    until 2026-09-18, which is what a package the registry cannot express
    looks like.
    """
    _bind()
    recipe = recipe_by_name("molbuilder")
    flat = lambda plan: " ".join(" ".join(s.argv) for s in plan)
    default = flat(install.plan_install(recipe)[1])
    opted = flat(install.plan_install(recipe, include_opt_in=True)[1])
    for name in ("nodejs", "playwright", "pytest-playwright", "chromium"):
        assert name not in default, f"{name} is opt-in; a default run must skip it"
        assert name in opted, f"--with-dev-tools must install {name}"
    # The chromium download is the only extra step this recipe has, so a
    # default plan dispatches none at all.
    assert not [s for s in install.plan_install(recipe)[1]
                if s.role is install.StepRole.EXTRA]


def test_opt_in_absence_is_reported_but_does_not_fail_the_audit(tmp_path):
    """An opt-in package nobody asked for is a CHOICE, not a defect.

    Counting it as REQUIRED missing would make `doctor` red on every machine
    that simply did not want the test tooling.  Reported all the same --
    absence nobody can see is how four commands came to live in a help
    comment.
    """
    from molbuilder.envs import doctor as _doc
    recipe = Recipe(
        name="synth-optin-env", category=None, description="d",
        channels=("conda-forge",),
        conda_packages=("python=3.12", CondaPackage("nodejs", opt_in="why")),
        pip_packages=(PipPackage("playwright", opt_in="why"),),
    )
    (tmp_path / "conda-meta").mkdir()
    audit = _doc.audit_packages(tmp_path, recipe)
    kinds = {i.kind for i in audit.issues}
    assert "conda-missing-opt-in" in kinds
    assert "pip-missing-opt-in" in kinds
    # `python` IS a required gap here -- the synthetic prefix has an empty
    # conda-meta -- and that is right.  The claim is narrower: the two
    # OPT-IN packages are not.
    required = {i.name for i in audit.issues if not i.optional}
    assert "nodejs" not in required and "playwright" not in required, (
        f"an opt-in package that was never asked for is not a REQUIRED gap; "
        f"required={sorted(required)}")


def test_the_interpreter_reaches_an_env_that_already_exists(monkeypatch,
                                                            tmp_path):
    """python is delivered to a PRESENT env, not only to a fresh one.

    It is the one declared package `conda create` also carries, and create is
    SKIPPED for an env that exists -- so while `conda_set` excluded python
    "because create already placed it", `install` could never deliver it to a
    machine that had bootstrapped before the pin existed.  That is verbatim
    the hole the create/install split was made to close,
    reopened for the very package that change adds to `molbuilder-siesta`.

    The env here is the real pre-2026-09-18 shape: siesta, numactl and git on
    disk, no python.
    """
    recipe = recipe_by_name("molbuilder-siesta")
    fake_env = tmp_path / "molbuilder-siesta"
    meta = fake_env / "conda-meta"
    meta.mkdir(parents=True)
    for name in ("siesta", "numactl", "git"):        # note: NO python
        (meta / f"{name}-1.0-h0.json").write_text(
            json.dumps({"name": name, "version": "1.0", "build": "h0"}))
    _bind(conda_envs={recipe.name: str(fake_env)})

    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{fake_env}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path}"]}}')
        return _stub(0, stdout="")
    monkeypatch.setattr(_diag.subprocess, "run", fake_run)
    calls = []
    def fake_stream(*a, **kw):
        calls.append(a[0])
        return (0, "siesta 5.4.2")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)

    result = install.run_install(recipe)
    assert result.succeeded is True
    solves = [list(c) for c in calls if list(c)[1:2] == ["install"]]
    assert solves, (
        "create was skipped and no conda install was dispatched -- the env "
        "keeps whatever it has, and `install` reports success over it")
    assert any(a.startswith("python=") for a in solves[0]), (
        f"the declared interpreter must be in the solve; got {solves[0]}")


def test_run_install_short_circuits_on_create_failure(monkeypatch):
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    monkeypatch.setattr(_diag.subprocess, "run",
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
    recipe = _ALL_PHASES_RECIPE
    fake_env = tmp_path / recipe.name
    (fake_env / "conda-meta").mkdir(parents=True)

    _bind(conda_envs=(recipe.name,))

    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{fake_env}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path}"]}}')
        return _stub(0, stdout="")
    monkeypatch.setattr(_diag.subprocess, "run", fake_run)
    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "Version 1.40")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)
    result = install.run_install(recipe)
    assert result.succeeded is True
    create = next(s for s in result.steps if s.label == "conda create")
    assert "already exists" in create.output
    # SKIPPED, and claiming no exit code.  It used to record
    # `returncode=0`, which made "I did not do this" indistinguishable
    # from "I did this and it worked" to every reader of the result.
    assert create.outcome is install.Outcome.SKIPPED
    assert create.returncode is None
    assert create.outcome.is_success is True
    # AND THE CONDA SET STILL GOES IN.  This is the half `install` could not
    # do before 2026-09-17: the env exists, so create is skipped -- and the
    # declared packages went with it, because `conda create` was the only
    # place they were ever named.  `conda-meta/` here
    # is empty, so the gate finds them absent and dispatches.
    conda_set = next(s for s in result.steps if s.label == "conda install")
    assert conda_set.outcome is install.Outcome.OK
    assert "install" in conda_set.argv and "create" not in conda_set.argv
    # Four streaming calls (conda set + pip + extra + verify), zero for the
    # skipped create.
    assert len(calls) == 4, (
        f"expected 4 streaming calls (conda set + pip + extra + verify), "
        f"got {len(calls)}"
    )


def test_run_install_blocks_when_env_state_is_broken(monkeypatch, tmp_path):
    """An env directory with no `conda-meta/` BLOCKS the install and says
    `--clean`, instead of letting `conda create` fail cryptically.

    THE FAILURE THIS CATCHES.  `probe_env_state` returns BROKEN for a directory
    that exists without `conda-meta/` (`install.py:401`) -- a half-finished or
    interrupted install.  `run_install` must stop there (`:650`,
    `state.needs_cleanup`): if it proceeds, `conda create` refuses with "prefix
    already exists" and the person is left staring at a conda error with no idea
    that `--clean` is the answer.  Worse, nothing downstream runs, so a
    `succeeded is True` here would report a working env that is not one.

    WHY THIS TEST EXISTS AT ALL.  A test of this NAME was deleted on 2026-09-08
    because its own body comment admitted the mismatch -- *"the fake_run above
    returns no real dir, so probe_env_state sees FRESH ... this test as written
    confirms that the FRESH path still works"*.  It had promised a gate the
    suite did not have, which is worse than an absent test: the audit that read
    it counted the gate as covered.  This is the gate, actually exercised.

    Contract: `ops/environments.md` -- the state machine and `--clean`;
    recorded as a coverage gap by the 2026-09-08 audit
    (`process/test-audit-findings.md` § 3.5).

    BROKEN needs all three of: registered, directory present, no `conda-meta/`.
    The sibling above serves the same fixture WITH `conda-meta` and gets
    PRESENT, so the two differ by exactly the signal under test.
    """
    recipe = _ALL_PHASES_RECIPE
    fake_env = tmp_path / recipe.name
    fake_env.mkdir(parents=True)          # ... and deliberately NO conda-meta/

    _bind(conda_envs=(recipe.name,))

    def fake_run(argv, *a, **kw):
        argv_list = list(argv) if not isinstance(argv, str) else [argv]
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{fake_env}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path}"]}}')
        return _stub(0, stdout="")
    monkeypatch.setattr(_diag.subprocess, "run", fake_run)

    calls = []
    def fake_stream(*a, **kw):
        calls.append(a)
        return (0, "Version 1.40")
    monkeypatch.setattr(install._builds, "run_streaming", fake_stream)

    result = install.run_install(recipe)

    assert result.succeeded is False, (
        "a BROKEN env reported a successful install; the person is told they "
        "have a working environment that has no conda-meta/")
    create = next(s for s in result.steps if s.label == "conda create")
    assert create.returncode is None, (
        f"the blocked step must not claim an exit code: {create.returncode!r}")
    assert "BROKEN" in create.output and "--clean" in create.output, (
        f"the refusal does not tell the person what to do: {create.output!r}")
    # NOTHING may run past the gate -- pip, extra steps and verify would all
    # execute against an env that does not exist.
    assert calls == [], f"work ran past the BROKEN gate: {len(calls)} call(s)"


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
    monkeypatch.setattr(_diag.subprocess, "run", fake_run)

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


# A recipe whose only pip package is OPTIONAL, so the failure below is
# the one `optional` exists to survive.
_OPTIONAL_PIP_RECIPE = Recipe(
    name="synth-optional-env",
    category=None,
    description="Synthetic recipe: one optional pip package.",
    channels=("conda-forge",),
    conda_packages=("python=3.12", "pip"),
    pip_packages=(
        PipPackage("some-gpu-wheel", optional=True,
                   reason="GPU only; the env is a full CPU env without it"),
    ),
    verify_argv=("some-tool", "--version"),
    verify_expect_contains="Version",
)


def test_optional_package_failure_degrades_without_stopping(monkeypatch):
    """An optional package that cannot be installed must leave the env
    DEGRADED and the install SUCCEEDING, with every later phase still run.

    This is the promise `PipPackage.optional` makes, and it has been
    broken in production once already: the runner's launch-failure branch
    skipped the optional check and aborted an install it should have
    survived.  Nothing caught that, because every test here drove either
    a clean run or a required failure -- a mutation making DEGRADED stop
    the install passed the whole suite.
    """
    _bind()
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory(
                            (0, "solving..."),          # conda create
                            (0, "solving..."),          # the conda set
                            (1, "No matching distribution found"),
                            (0, "Version 1.0"),         # verify STILL RUNS
                        ))
    result = install.run_install(_OPTIONAL_PIP_RECIPE)

    assert result.succeeded is True, (
        "one unavailable optional wheel must not take the env down")
    pip_step = next(s for s in result.steps
                    if s.label == "pip install some-gpu-wheel")
    assert pip_step.outcome is install.Outcome.DEGRADED
    assert pip_step.outcome.stops_the_install is False
    # DEGRADED is precisely where the two predicates differ: the env did
    # NOT get what this step was for, and the install still stands.
    assert pip_step.outcome.is_success is False
    verify = next(s for s in result.steps if s.label == "verify")
    assert verify.outcome is install.Outcome.OK, (
        "the phase after a degraded step must still run")


def test_launch_failure_of_optional_step_also_degrades(monkeypatch):
    """The specific regression: `rc is None` -- the process never
    launched -- must not be treated as a special case that forgets
    `fatal`."""
    _bind()
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory(
                            (0, "solving..."),          # conda create
                            (0, "solving..."),          # the conda set
                            (None, ""),                 # never launched
                            (0, "Version 1.0"),
                        ))
    result = install.run_install(_OPTIONAL_PIP_RECIPE)
    assert result.succeeded is True
    pip_step = next(s for s in result.steps
                    if s.label == "pip install some-gpu-wheel")
    assert pip_step.outcome is install.Outcome.DEGRADED
    assert pip_step.returncode is None
    assert "failed to launch" in pip_step.output


def test_index_fallback_is_flagless_and_recovers(monkeypatch):
    """The declared alternative must carry NO force flags, and succeeding on
    it must report RECOVERED under the argv that actually ran.

    BOTH halves were unenforced until 2026-09-12, and each mutation passed
    the entire suite:

      * adding `flags=pkg.install_flags()` to the fallback -- the contract
        warns this is how "an unreachable source during an unrelated install
        would overwrite a good git tree with a worse one", i.e. silent env
        corruption;
      * collapsing RECOVERED into OK -- the state exists precisely so a
        recovered step is never reported under the command that failed.
    """
    pkg = PipPackage("pyscf-properties",
                     source="git+https://github.com/pyscf/properties.git",
                     force=True, fallback_to_index=True,
                     reason="never released to PyPI with the IR module")
    step = install.pip_step_for(pkg, "/usr/bin/conda", "molbuilder-pySCF")

    # The PRIMARY attempt forces, because an installed version cannot prove
    # it is the declared build.
    assert "--force-reinstall" in step.argv
    assert "--no-deps" in step.argv
    assert ("pyscf-properties @ git+https://github.com/pyscf/properties.git"
            in step.argv)

    # The ALTERNATIVE forces nothing.  Its job is to make sure the package
    # EXISTS, never to replace what is already there.
    assert len(step.fallbacks) == 1
    indexed = step.fallbacks[0]
    assert "--force-reinstall" not in indexed, (
        "the index fallback must not force -- an unreachable source during "
        "an unrelated install would overwrite a good tree with a worse one")
    assert "--no-deps" not in indexed
    assert "pyscf-properties" in indexed

    # Source unreachable, index fine.
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory(
                            (1, "Could not resolve host: github.com"),
                            (0, "Successfully installed pyscf-properties")))
    done = install.run_step(step, prefix=None)
    assert done.outcome is install.Outcome.RECOVERED, (
        "a step that succeeded on its alternative is RECOVERED, not OK")
    assert done.argv == indexed, (
        "the step must report the argv that RAN, not the one that failed")


def test_bare_strings_normalise_for_both_kinds():
    """One rule for both package kinds -- a bare name becomes a record.

    The conda half was covered only by accident (every recipe uses bare
    conda strings, so breaking it errors at import); the pip half was not
    covered at all, and deleting it passed the whole suite.
    """
    r = Recipe(
        name="synth-normalise", category=None, description="d",
        channels=("conda-forge",),
        conda_packages=("numpy", CondaPackage("cupy", optional=True,
                                              reason="GPU only")),
        pip_packages=("pubchempy", PipPackage("cupy-cuda13x",
                                              optional=True)),
    )
    # Bare names became records, required by default.
    assert r.conda_packages[0] == CondaPackage("numpy")
    assert r.pip_packages[0] == PipPackage("pubchempy")
    assert r.conda_packages[0].optional is False
    assert r.pip_packages[0].optional is False
    # Records written out explicitly pass through untouched.
    assert r.conda_packages[1].optional is True
    assert r.conda_packages[1].reason == "GPU only"
    assert r.pip_packages[1].optional is True
    # And the derived views hand the plain strings back.
    assert r.conda_specs == ("numpy", "cupy")
    assert r.pip_specs == ("pubchempy", "cupy-cuda13x")


def test_build_phases_carry_the_verdict_builds_gave_them(monkeypatch):
    """A sentinel-skipped build phase is SKIPPED, not OK, and an abandoned
    one is not reported at all.

    The adapter used to derive the outcome from the return code, which got
    two of `builds.py`'s four states wrong: "skip" carries `returncode=0`
    so a phase that did NOT run reported `OK (rc=0)` -- the exact confusion
    `_undispatched` was written to end -- and "not-run" carries `None` so
    one real ELPA failure was announced as FAILED once per remaining phase.
    """
    from molbuilder.envs import builds as B

    recipe = recipe_by_name("molbuilder-siesta-gpu")
    _bind()
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: "/fake/envs/x")
    monkeypatch.setattr(install._builds, "run_streaming",
                        lambda *a, **kw: (0, "ok"))

    def mkstep(component, phase):
        return B.BuildStep(component=component, phase=phase,
                           argv=("cmake", "--build", "."),
                           sentinel=Path("/nonexistent"),
                           log_file=Path("/tmp/x/logs/a.b.log"))

    monkeypatch.setattr(install._builds, "run_build_spec",
                        lambda *a, **kw: B.BuildResult(
                            spec=recipe.build_spec,
                            env_prefix="/fake/envs/x",
                            activate_hook_written=False,
                            deactivate_hook_written=False,
                            succeeded=False,
                            steps=(
                                B.BuildStepResult(step=mkstep("elpa", "clone"),
                                                  status="skip", returncode=0,
                                                  output="sentinel present"),
                                B.BuildStepResult(step=mkstep("elpa", "build"),
                                                  status="fail", returncode=2,
                                                  output="boom"),
                                B.BuildStepResult(step=mkstep("siesta", "build"),
                                                  status="not-run"),
                            ),
                            preflight_errors=(),
                        ))
    result = install.run_install(recipe)

    by_label = {s.label: s for s in result.steps}
    assert by_label["build:elpa.clone"].outcome is install.Outcome.SKIPPED, (
        "a sentinel-skipped phase must not claim it ran")
    assert by_label["build:elpa.build"].outcome is install.Outcome.FAILED
    assert "build:siesta.build" not in by_label, (
        "a phase that was never reached is not reported, the same way the "
        "install loop does not record steps after it stops")
    assert result.succeeded is False, (
        "builds.py owns its own verdict and the installer must read it")


def test_run_install_verify_substring_failure_is_fatal(monkeypatch):
    """A verify step that exits 0 but lacks the expected substring
    must fail the install -- catches a silent regression where the
    binary is in the env but not the right binary."""
    _bind()
    recipe = recipe_by_name("molbuilder-siesta")
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "solving..."),
                                             (0, "solving..."),
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
    monkeypatch.setattr(_diag.subprocess, "run",
                        lambda *a, **kw: _stub(0, stdout='{"envs": []}'))
    monkeypatch.setattr(install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(install._builds, "run_streaming",
                        _stream_stub_factory((0, "solving..."),
                                             (0, "solving..."),
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
        return _fake_result(recipe, ok)

    return _fake, calls


def _fake_result(recipe, ok=True):
    """An `InstallResult` as the installer really returns one.

    A bare `MagicMock` invents a truthy attribute for anything asked of it, so
    `if result.build_result.activate_hook_written:` came out TRUE for a
    conda-only recipe and the recap then dereferenced a `build_spec` that is
    None -- an AttributeError that aborted a whole `bootstrap` run and looked
    like a product bug.  A fake has to look like the thing it stands for.
    """
    result = MagicMock()
    result.succeeded = ok
    result.recipe = recipe
    result.build_result = None
    result.steps = []
    return result


def _dispatch_log(monkeypatch, output=""):
    """Record what goes through THE ONE DOOR, and answer success.

    `builds.dispatch_into_env` is the single place a command enters an env
    (env-framework.md § 5.6), so recording there records everything the
    installer does -- the `--clean` wipe included, since that is a step now.
    One substitution, of the function the design names, and the assertions read
    the sequence the product itself produces.
    """
    from molbuilder.envs import builds as _b

    seen = []

    def _door(argv, prefix, **kw):
        seen.append([str(a) for a in argv])
        return (0, output)

    monkeypatch.setattr(_b, "dispatch_into_env", _door)
    return seen


def _make_runner():
    """CliRunner is brought in via Click."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture(autouse=True)
def _home_is_not_the_developers(monkeypatch, tmp_path):
    """The install CLI tees a per-recipe log under ``~/.molbuilder/logs``
    even when ``run_install`` is stubbed -- the tee is the CLI's, opened
    around the stub.  Untouched, every bootstrap test dropped real-named
    zero-byte logs into the developer's actual home (found 2026-08-28).
    ``_log_root`` resolves lazily now, so isolating HOME here is enough."""
    monkeypatch.setenv("HOME", str(tmp_path))


def test_the_dry_run_says_what_the_machine_must_provide(monkeypatch):
    """`Recipe.system_preconditions` is accurate and was rendered NOWHERE
    until 2026-09-14 -- the one place it belongs is where somebody is
    deciding whether to start a 25-minute source build.

    And it must say what is OPTIONAL.  The NVIDIA driver is not needed to
    build: CUDA comes from conda into the env, ELPA is a diagonalization
    library whose CPU kernels work without a GPU, and the driver matters at
    RUN time for the GPU path only.  A preconditions list that read as
    "you need a driver" would turn people away from an env that installs
    and runs fine on their laptop.
    """
    from molbuilder.envs import _cli as envs_cli
    _bind(conda_binary="/fake/conda")
    result = _make_runner().invoke(
        envs_cli.envs_group, ["install", "molbuilder-siesta-gpu", "--dry-run"],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "This machine has to provide:" in result.output, result.output
    driver = [l for l in result.output.splitlines() if "NVIDIA driver" in l]
    assert driver, result.output
    assert "OPTIONAL" in " ".join(driver), (
        "the driver line must say it is optional: it is not needed to "
        f"build.\n{result.output}")


def _cli_hint(*args):
    """The remedy speller the CLI uses, so a test never re-types a command."""
    from molbuilder.envs.hints import fix_cmd
    return fix_cmd(*args)


def test_the_notebook_env_is_not_installed_by_bootstrap(monkeypatch):
    """The FACT, not the rule (user, 2026-09-14: *"jupyter notebook is
    provided as optional ... install of this should be explicit just like the
    siesta-gpu env"*).

    The sibling test above derives its expectation from `opt_in`, so it holds
    whatever that field says -- delete `opt_in` from the notebook recipe and
    it still passes, while every bootstrapped machine quietly grows a
    notebook server.  This one names the env, and reads the output a person
    sees: not planned, and the command that installs it is printed.
    """
    from molbuilder.envs import _cli as envs_cli
    _bind(conda_binary="/fake/conda")
    result = _make_runner().invoke(
        envs_cli.envs_group, ["bootstrap", "--dry-run", "--yes"],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "molbuilder-jupyternb" in out, out
    assert "opt-in" in out, out
    # ...named as NOT being installed, with the explicit command beside it.
    assert "NOT installing" in out, out
    assert _cli_hint("install", "molbuilder-jupyternb", "--yes") in out, out


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


def test_dry_run_writes_nothing_when_every_env_is_present(monkeypatch,
                                                          tmp_path):
    """`--dry-run` was honoured only when there was something to install.

    With every env already present, control took the other branch and fell
    through to `_seed_config` -- which PROMPTS and CREATES the config directory,
    molbuilder.json, environment.json, secrets/ and environments/.  A dry run
    that asks questions and writes files, against a flag whose help says "do not
    install anything" and `env-framework.md` 480's *"--dry-run means nothing gets
    installed, including by the shim"*.  It also spent the full verify+audit pass
    on every env.
    """
    from molbuilder.envs import _cli
    from molbuilder.envs.recipes import BUILTIN_RECIPES
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cfg))
    # Every env present -> --skip-existing empties the plan.
    _bind(conda_envs=tuple(r.name for r in BUILTIN_RECIPES))
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect", lambda: Capabilities(
        runtime_config={}, conda_binary="/c/bin",
        conda_envs=frozenset(r.name for r in BUILTIN_RECIPES)))

    def _boom(*a, **k):
        raise AssertionError("a --dry-run must not seed the config directory")
    monkeypatch.setattr(_cli, "_seed_config", _boom)
    doctor_ran = []
    monkeypatch.setattr(_cli._doctor, "report_all",
                        lambda caps, **kw: doctor_ran.append(1) or [])

    result = _make_runner().invoke(
        _cli.envs_group, ["bootstrap", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert not cfg.exists(), f"--dry-run created {cfg}"
    assert not doctor_ran, "a dry run spent the full doctor pass"
    assert "dry-run" in result.output.lower(), result.output


def test_bootstrap_gives_a_source_build_the_same_eyes_as_install(monkeypatch):
    """`bootstrap --include-source-builds` showed no preflight at all.

    It called `run_install(recipe, caps=caps)` with neither build callback, and
    `run_build_spec` reads `on_warnings=None` as "proceed silently" --
    `format_preflight_report` is only ever called FROM that callback, so the
    report was not merely unconfirmed, it was never rendered: no missing-driver
    notice, no compute-capability fallback, no free-space reminder, no per-phase
    progress.  Its own help promised the opposite: *"the user is asked to confirm
    before each source build starts unless --yes is also given."*
    """
    from molbuilder.envs import _cli
    seen = {}

    def _fake_run_install(recipe, **kw):
        seen[recipe.name] = (kw.get("build_on_warnings"),
                             kw.get("build_on_progress"))
        return _fake_result(recipe)

    _bind()
    monkeypatch.setattr(_cli._install, "run_install", _fake_run_install)
    monkeypatch.setattr(_cli._doctor, "report_all", lambda caps, **kw: [])
    monkeypatch.setattr(_cli, "_render_doctor", lambda reports: 0)
    monkeypatch.setattr(_cli, "_seed_config", lambda *a, **k: None)
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect", lambda: Capabilities(
        runtime_config={}, conda_binary="/c/bin", conda_envs=frozenset()))

    _make_runner().invoke(
        _cli.envs_group,
        ["bootstrap", "--yes", "--include-source-builds"])

    from molbuilder.envs.recipes import BUILTIN_RECIPES
    source_builds = [r.name for r in BUILTIN_RECIPES
                     if r.build_spec is not None]
    assert source_builds, "no source-build recipe to check"
    for name in source_builds:
        assert name in seen, f"{name} was not installed: {sorted(seen)}"
        on_warnings, on_progress = seen[name]
        assert on_warnings is not None, (
            f"{name} built with no preflight callback -- every warning "
            f"auto-accepted and the report never printed")
        assert on_progress is not None, (
            f"{name} built with no progress callback -- no per-phase output")


def test_the_shim_runs_a_readonly_verb_with_no_terminal_and_no_flag():
    """`install-env.sh list` must work with nothing on stdin.

    `require_conda` asked "use this env manager?" for EVERY verb, and with no
    terminal it exited 2 with *"pass --yes to skip"* -- while `list`, `doctor`,
    `validate` and `repair` define no `--yes` option, so the remedy it printed
    was rejected by click ("Error: No such option '--yes'").  The canonical
    entry point could not run a health check in CI at all, and `cmd_doctor`'s
    docstring advertises itself as being for exactly that.

    Only `bootstrap` and `install` create or change an env with the chosen
    manager, so only they confirm it -- and both accept `--yes`, so the message
    stays true wherever it is still reached.

    A real invocation, because the defect is only visible WITHOUT a terminal and
    no amount of reading the bash proves the exit code.  Self-skipping, and
    gated on the manager the PACKAGE detects rather than a hard-coded path.
    """
    import subprocess as _sp
    from pathlib import Path as _P
    from molbuilder import diagnostics

    caps = diagnostics.detect()
    if not (caps.conda_binary and caps.env_available("molbuilder")):
        pytest.skip("needs a detectable conda and the `molbuilder` host env")

    script = _P(__file__).resolve().parents[1] / "scripts" / "install-env.sh"
    proc = _sp.run(["bash", str(script), "list"],
                   stdin=_sp.DEVNULL, capture_output=True, text=True,
                   timeout=180)
    combined = proc.stdout + proc.stderr
    assert "no TTY for confirmation" not in combined, (
        "a read-only verb still demands a terminal; in CI this is exit 2 and "
        f"the --yes it suggests is not an option on `list`:\n{combined}")
    assert proc.returncode == 0, (
        f"`install-env.sh list` exited {proc.returncode} with no stdin:\n"
        f"{combined}")


# `_recording_manager` and `_asked` stood here until 2026-09-13: a fake manager
# binary in a temp directory, recorded as `envs.manager`, with a log file.  It
# was the right shape for the question it was first asked -- "did the wipe reach
# a real `conda env remove`" -- while the wipe was a private dispatch.  Once the
# wipe became a step in the plan, the plan and the result answer the same
# question with nothing built: `plan_install` is pure, and every step carries an
# outcome.  A sandbox that size is for testing a system; this is one order in
# one list.


def test_bootstrap_hard_stops_on_a_wrecked_env_like_install_does(monkeypatch):
    """D7 -- `bootstrap` was a LOSSY COPY of `install`'s orchestration.

    It ran no env-state probe, so it never met the ORPHAN / GHOST / BROKEN hard
    stop: handed a directory conda's registry does not know about, it drove a
    `conda create` straight at it, which conda refuses with *"prefix already
    exists"* after the person has waited for the solve.  `install` diagnosed
    that in a second, before touching anything, since the state machine was
    written.

    A state in, an event sequence out: `EnvState` says ORPHAN, and the door
    must see no `create`.  Both are the installer's own vocabulary -- there is
    nothing here to fake but the reading itself.
    """
    _bind(conda_envs={})
    from molbuilder.envs import _cli
    seen = _dispatch_log(monkeypatch)
    monkeypatch.setattr(
        _cli._install, "probe_env_state",
        lambda name, binary: install.EnvState(
            name=name, listed_in_registry=False,     # the registry does not
            dir_exists=True, has_conda_meta=True,    # know this directory
            prefix=f"/prefix/{name}", manager=binary))
    monkeypatch.setattr(_cli._doctor, "report_all", lambda caps, **kw: [])
    monkeypatch.setattr(_cli, "_render_doctor", lambda reports: 0)
    monkeypatch.setattr(_cli, "_seed_config", lambda *a, **k: None)

    result = _make_runner().invoke(_cli.envs_group, ["bootstrap", "--yes"])

    creates = [a for a in seen if len(a) > 1 and a[1] == "create"]
    assert not creates, (
        f"bootstrap drove a `conda create` at an env conda's registry does "
        f"not know about:\n{seen}\n{result.output}")
    assert "ORPHAN" in result.output, result.output
    assert "refused" in result.output.lower(), (
        "the report must say which recipe was refused and why:\n"
        + result.output)


def test_the_environment_canary_notices_a_change(monkeypatch, tmp_path):
    """The safety net in `conftest.py` has to work, or it is worse than none.

    Asked through the snapshot -- `{name: prefix}` is what the fingerprint
    reads -- so this needs no manager and no real env: point it at a directory,
    add a package the way pip would, and ask again.
    """
    import sys as _sys

    fake_env = tmp_path / "envs" / "pretend"
    site = fake_env / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    _bind(conda_envs={"pretend": str(fake_env)}, conda_binary="/fake/conda")

    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import conftest

    before = conftest._env_fingerprint()
    assert "pretend" in before, before

    (site / "something-1.0.dist-info").mkdir()
    after = conftest._env_fingerprint()

    assert after != before, (
        "a package appeared in an env and the canary saw nothing")


def test_the_clean_plan_removes_the_env_before_it_creates_one():
    """`--clean` is the one wipe-and-reinstall door, and the wipe and the
    create cannot get out of step -- which is why there is no separate `envs
    remove` verb at all.

    Read off the PLAN, which runs nothing: `plan_install` is pure, so the order
    is a property of the plan rather than something you have to execute to
    find out.  The wipe used to be dispatched by the CLI on the side, which is
    why this could not be asked before -- and why `--dry-run` could not show it.

    Two defects met here, one from each direction: `--clean` was refused for
    conda-only recipes (four of the five registered envs, so the remedy
    `doctor` prints was a usage error), and when the refusal was lifted the
    wipe was still gated on `build_spec` -- accepted, wiped nothing, printed
    "install OK".
    """
    _bind(conda_envs={"molbuilder-pySCF": "/prefix/molbuilder-pySCF"})
    recipe = recipe_by_name("molbuilder-pySCF")

    _name, plain = install.plan_install(recipe)
    _name, wiped = install.plan_install(recipe, clean=True)

    assert [s.label for s in plain][0] == "conda create"
    labels = [s.label for s in wiped]
    assert labels[0] == "remove env molbuilder-pySCF", labels
    assert labels[1] == "conda create", labels
    assert labels[1:] == [s.label for s in plain], (
        "--clean changed more than the wipe", labels)


def test_the_clean_run_removes_a_PRESENT_env_and_installs_into_the_new_one(
        monkeypatch):
    """`--clean` against the one state it exists for: an env that IS there.

    Until 2026-09-13 this test faked the env ABSENT, and the removal step
    carried the CREATE role -- so the runner asked "does it already exist?"
    first and answered SKIPPED for every present env.  `--clean` never removed
    anything, and the test could not see it because it never gave the skip a
    chance to fire (K-L1).

    Three facts, from the sequence the installer itself produces: the removal
    is dispatched, before the create; every step carries an outcome and the
    run succeeds; and every step AFTER the create is addressed at the
    directory the manager put the NEW env in -- not the one the old env had,
    which the startup snapshot still lists until the installer voids it.
    """
    from molbuilder import diagnostics as _diag
    old, new = "/prefix/old-pySCF", "/prefix/new-pySCF"
    _bind(conda_envs={"molbuilder-pySCF": old}, conda_binary="/fake/conda")
    recipe = recipe_by_name("molbuilder-pySCF")
    seen: list = []

    def _door(argv, prefix, **kw):
        seen.append(([str(a) for a in argv], prefix))
        return (0, _VERIFY_OUTPUT)

    monkeypatch.setattr(install._builds, "dispatch_into_env", _door)

    def _probe(name, binary):
        # The fake manager's state follows the commands it was given: the env
        # is there until an `env remove` has gone through the door, and gone
        # after.  A probe that answered PRESENT forever would make the create
        # after the removal look like a resume -- which is what the first
        # version of this test did.
        removed = any(argv[1] == "env" for argv, _p in seen)
        if removed:
            return install.EnvState(name=name, listed_in_registry=False,
                                    dir_exists=False, has_conda_meta=False,
                                    prefix=None, manager=binary)
        return install.EnvState(name=name, listed_in_registry=True,
                                dir_exists=True, has_conda_meta=True,
                                prefix=old, manager=binary)

    monkeypatch.setattr(install, "probe_env_state", _probe)
    # What a fresh reading of the machine says once the env has been
    # re-created: the manager put it somewhere else.
    monkeypatch.setattr(_diag, "detect", lambda: Capabilities(
        runtime_config={}, conda_binary="/fake/conda",
        conda_envs={"molbuilder-pySCF": new}))

    result = install.run_install(recipe, clean=True)

    labels = [s.label for s in result.steps]
    assert labels[0] == "remove env molbuilder-pySCF", labels
    assert all(s.outcome is not None for s in result.steps), (
        "a step with no outcome is a step the verdict cannot account for")
    assert result.succeeded, [(s.label, s.outcome) for s in result.steps]
    verbs = [argv[1] for argv, _p in seen]
    assert "env" in verbs, f"the removal was never dispatched: {seen}"
    assert verbs.index("env") < verbs.index("create"), seen
    # ...and HANDED the directory the registry gave, for the door to address
    # it by (M2; review B-L2, 2026-09-14) -- the door is faked here, so what
    # it was handed is what can be seen; the re-addressing itself is asserted
    # through the real door in test_envs_enters_the_env_through_the_manager.
    _removal, handed = next((a, p) for a, p in seen if a[1] == "env")
    assert handed == old, seen
    entered = [(argv, p) for argv, p in seen if argv[1] == "run"]
    assert entered, seen
    assert all(p == new for _argv, p in entered), (
        "a step after the create was addressed at the OLD directory:\n"
        + "\n".join(f"  {p}: {' '.join(a)}" for a, p in entered))


def test_the_clean_run_on_a_fresh_machine_skips_the_removal_and_creates(
        monkeypatch):
    """`env remove -n` exits 1 for an env that is not there (measured on
    conda 26.7.1, 2026-09-14) -- so `--clean` on a fresh machine, or the
    re-run after a `--clean` whose create then failed, died at step 1.  The
    removal is SKIPPED when nothing is on disk, and the create runs."""
    from molbuilder import diagnostics as _diag
    new = "/prefix/new-pySCF"
    _bind(conda_envs={}, conda_binary="/fake/conda")
    recipe = recipe_by_name("molbuilder-pySCF")
    seen: list = []

    def _door(argv, prefix, **kw):
        seen.append(([str(a) for a in argv], prefix))
        return (0, _VERIFY_OUTPUT)

    monkeypatch.setattr(install._builds, "dispatch_into_env", _door)
    fresh = install.EnvState(name="molbuilder-pySCF", listed_in_registry=False,
                             dir_exists=False, has_conda_meta=False,
                             prefix=None, manager="/fake/conda")
    monkeypatch.setattr(install, "probe_env_state", lambda name, binary: fresh)
    # After the create the manager knows the env; with a fake manager the
    # resolver is told directly.
    monkeypatch.setattr(install, "_env_prefix", lambda name, binary: new)

    result = install.run_install(recipe, clean=True, env_state=fresh)

    removal = result.steps[0]
    assert removal.label == "remove env molbuilder-pySCF"
    assert removal.outcome is install.Outcome.SKIPPED, removal.outcome
    assert not any(argv[1] == "env" for argv, _p in seen), (
        "a removal was dispatched with nothing on disk to remove")
    assert any(argv[1] == "create" for argv, _p in seen), "the create did not run"
    assert result.succeeded, [(s.label, s.outcome) for s in result.steps]


#: What molbuilder-pySCF's verify step requires of its own output.
_VERIFY_OUTPUT = ("pyscf 2.13, geometric 1.1, prop: polarizability OK\n"
                  "  IR: analytic dmu/dR available (pyscf.prop.infrared)")


def test_an_env_outside_envs_dirs_is_PRESENT_not_GHOST(monkeypatch, tmp_path):
    """GHOST must mean what `env-framework.md` § 2.1 says: a registry entry
    whose directory is gone.

    The probe used to measure `dir_exists` by searching conda's `envs_dirs`
    for `<dir>/<name>`, ignoring the prefix the registry had just handed it two
    lines above.  An env created with `--prefix` outside any of those
    directories -- `/scratch`, a project tree, a module-provided root -- is
    listed by the registry with a perfectly healthy directory, and the probe
    called it GHOST anyway.  GHOST then hard-stops `install` and prints
    `env remove` as the fix: the program recommending the destruction of a
    working env (`installation.md` M2 and M5 in one defect).

    Asking the manager where the env is, instead of guessing from a search
    path, is also one subprocess instead of two -- asserted here, because the
    saving is the point: the prefix is not derived at all any more.
    """
    out_of_tree = tmp_path / "scratch" / "mb-out-of-tree"
    (out_of_tree / "conda-meta").mkdir(parents=True)
    seen: list = []

    def fake_run(argv, *a, **kw):
        argv_list = list(argv)
        seen.append(argv_list[1:3])
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{out_of_tree}"]}}')
        if argv_list[1:2] == ["info"]:
            # envs_dirs deliberately does NOT contain the env.
            return _stub(0, stdout=f'{{"envs_dirs": ["{tmp_path / "envs"}"]}}')
        return _stub(0, stdout="")

    monkeypatch.setattr(_diag.subprocess, "run", fake_run)
    state = install.probe_env_state("mb-out-of-tree", "/fake/conda")

    assert state.state is install.EnvPresence.PRESENT, (
        f"a healthy env outside envs_dirs reported {state.state_label}; "
        f"prefix={state.prefix}")
    assert state.can_resume is True
    assert state.needs_cleanup is False
    assert state.prefix == str(out_of_tree)
    assert ["info"] not in [c[:1] for c in seen], (
        f"the registry answered with the prefix; `info --json` was still "
        f"paid for: {seen}")


def test_a_registry_entry_whose_directory_is_gone_is_still_GHOST(
        monkeypatch, tmp_path):
    """The other half: measuring the named prefix must not make GHOST
    unreachable.  A registry entry pointing at a directory that is not there
    is exactly what GHOST is for, and `install` must still hard-stop on it."""
    gone = tmp_path / "scratch" / "was-here"

    def fake_run(argv, *a, **kw):
        argv_list = list(argv)
        if argv_list[1:3] == ["env", "list"]:
            return _stub(0, stdout=f'{{"envs": ["{gone}"]}}')
        if argv_list[1:2] == ["info"]:
            return _stub(0, stdout='{"envs_dirs": []}')
        return _stub(0, stdout="")

    monkeypatch.setattr(_diag.subprocess, "run", fake_run)
    state = install.probe_env_state("was-here", "/fake/conda")

    assert state.state is install.EnvPresence.GHOST, state.describe()
    assert state.needs_cleanup is True
    assert state.can_resume is False
    # M3: the remedy it prints names the detected manager.
    assert state.remove_cmd() == "/fake/conda env remove -n was-here -y"


def test_clean_REFUSES_to_remove_the_env_molbuilder_IS_RUNNING_FROM(
        monkeypatch, tmp_path):
    """The self-destruct `--clean` opened up, and nothing stood in its way.

    Opening `--clean` to conda-only recipes put the HOST recipe in scope -- it
    is conda-only -- and the host env is the env this interpreter runs from.
    `doctor` prints `install <name> --clean --yes` for any env whose verify
    failed, the host included, and `--yes` means no prompt stands between that
    copy-paste and a machine with no molbuilder: the removal succeeds, then
    `conda create` runs from a prefix that no longer exists, and a failure
    between the two leaves nothing installed.  conda's own "cannot remove
    current environment" guard never fires because the shim dispatches
    `<prefix>/bin/python` WITHOUT activating.

    The assertion is the EVENT SEQUENCE: nothing reached the door at all.  That
    is the whole claim -- a refusal is a refusal only if nothing was
    dispatched -- and the door is where every dispatch goes, so an empty record
    is the proof.
    """
    import sys as _sys

    # The real condition: the host env's prefix IS this interpreter's prefix.
    _bind(conda_envs={"molbuilder": _sys.prefix}, conda_binary="/fake/conda")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    from molbuilder.envs import _cli
    seen = _dispatch_log(monkeypatch)
    monkeypatch.setattr(
        _cli._install, "probe_env_state",
        lambda name, binary: install.EnvState(
            name=name, listed_in_registry=True, dir_exists=True,
            has_conda_meta=True, prefix=_sys.prefix, manager=binary))

    result = _make_runner().invoke(
        _cli.envs_group, ["install", "molbuilder", "--clean", "--yes"])

    assert seen == [], (
        f"`--clean --yes` on the env molbuilder runs from dispatched "
        f"something:\n{seen}\n{result.output}")
    assert result.exit_code == 2, (
        f"refusal must be an error, not a warning it then ignores "
        f"(exit {result.exit_code})\n{result.output}")
    assert "RUNNING FROM" in result.output, result.output
    # M3: the manual route names the DETECTED manager, not a literal `conda`.
    assert f"/fake/conda env remove --prefix {_sys.prefix} -y" in result.output, \
        result.output


def test_an_advisory_probe_is_one_word_and_out_of_the_count(capsys):
    """H11 and E7, which are one confusion seen from two sides.

    An advisory probe is one the env is usable without -- MPS being absent is
    the example.  The live line called it FAIL while the table beneath called
    it NOTE (two words for one fact), and the summary counted it as a failure
    while the VERDICT ignored it -- so a run with one real failure beside an
    absent MPS printed "4/6 checks passed" and a reader could not reconcile
    the number with the outcome.

    One word, from the result; one counting rule, the verdict's.
    """
    from molbuilder.envs import _cli
    from molbuilder.envs.validate import ProbeResult, ValidationReport

    report = ValidationReport(
        recipe_name="molbuilder-siesta-gpu", env_prefix="/prefix",
        probes=(
            ProbeResult(name="ctest", passed=True, detail="12 passed"),
            ProbeResult(name="elpa", passed=True, detail="ok"),
            ProbeResult(name="gpu-fallback", passed=False,
                        detail="ran on CPU"),
            ProbeResult(name="mps", passed=False, detail="no MPS daemon",
                        advisory=True),
        ))

    assert [p.tag for p in report.probes] == ["PASS", "PASS", "FAIL", "NOTE"]

    code = _cli._render_validation(report, show_output_on_fail=False)
    out = capsys.readouterr()
    printed = out.out + out.err

    assert code == 1, "a real failure must fail the command"
    assert "2/3 required checks passed" in printed, printed
    assert "1 advisory" in printed, printed
    assert "[NOTE] mps" in printed.replace("  ", " ") or "[NOTE]" in printed


def test_the_host_env_name_has_a_persistent_home(monkeypatch, tmp_path):
    """D13 -- `"envs": {"host": "mb-dev"}` validated and was ignored.

    The `envs` block takes any string->string pair, so writing it was accepted;
    nothing read it, because the name was resolved from the config only for
    RECIPES WITH A CATEGORY and the host recipe has none.  So the host env's
    name lived only in `$MOLBUILDER_HOST_ENV`: export it, install, open a new
    shell without it, and `doctor` reports the host recipe against `molbuilder`
    again -- while `install molbuilder` from there builds the second host env
    the override existed to avoid.

    Same rule as every other env now: the config is the home, the variable is
    an override for one invocation.
    """
    import json

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "molbuilder.json").write_text(json.dumps({"envs": {"host": "mb-dev"}}))
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cfg))
    monkeypatch.delenv("MOLBUILDER_HOST_ENV", raising=False)

    from molbuilder.diagnostics import detect
    from molbuilder.envs.recipes import effective_name

    host = recipe_by_name("molbuilder")
    assert effective_name(host, detect()) == "mb-dev"

    # and the variable still wins for one invocation
    monkeypatch.setenv("MOLBUILDER_HOST_ENV", "mb-other")
    assert effective_name(host, detect()) == "mb-other"

    # a routed recipe is unaffected by either
    siesta = recipe_by_name("molbuilder-siesta")
    assert effective_name(siesta, detect()) == "molbuilder-siesta"


def test_every_fix_command_a_recipe_prints_names_a_registered_recipe():
    """A remedy in product code has to be runnable.

    The GPU recipe's verify step warns when bare `gcc` resolves outside the env
    and told the user to `molbuilder envs install siesta-gpu` -- which
    `recipe_by_name` does not match, because it takes canonical names only.  So
    the one warning that says your GPU env is missing its toolchain shims handed
    you `unknown recipe 'siesta-gpu'`, exit 2.  Measured 2026-09-12.

    Checks the RESULT -- the names a recipe's own shell actually prints -- rather
    than how any hint is spelled.
    """
    import re
    from molbuilder.envs.recipes import BUILTIN_RECIPES

    registered = {r.name for r in BUILTIN_RECIPES}
    bad = []
    for recipe in BUILTIN_RECIPES:
        blob = " ".join(recipe.verify_argv or ())
        for m in re.finditer(r"install\s+([A-Za-z0-9][A-Za-z0-9._-]*)", blob):
            cited = m.group(1)
            if cited.startswith("-"):
                continue
            if cited not in registered:
                bad.append(f"{recipe.name} prints `install {cited}`")
    assert not bad, (
        "a recipe prints a fix command naming an unregistered recipe, so "
        f"copy-pasting it errors: {bad}.  Registered: {sorted(registered)}")


def test_bootstrap_warns_about_a_readonly_config_root_BEFORE_installing(
        monkeypatch, tmp_path):
    """A read-only config root is known at minute 0; say so then.

    Seeding runs after every install, and that order is right -- it must not
    discard built envs.  But its preconditions were only tested there, so a
    read-only $HOME surfaced after forty minutes of conda work.  The warning now
    comes before the first install, and crucially before the Proceed prompt, so
    an interactive user can decline (user: *"if it's read only, then we should
    give the warning early rather than wait at the very end"*).

    The assertion that matters is ORDERING -- the warning must appear ahead of
    the first recipe banner in the output, not merely appear.
    """
    _bind()
    from molbuilder.envs import _cli
    import os as _os

    ro = tmp_path / "ro"
    ro.mkdir()
    _os.chmod(ro, 0o500)
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(ro / "cfg"))

    install_stub, calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    monkeypatch.setattr(_cli._doctor, "report_all", lambda caps, **kw: [])
    monkeypatch.setattr(_cli, "_render_doctor", lambda reports: 0)
    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset()))
    try:
        result = _make_runner().invoke(_cli.envs_group, ["bootstrap", "--yes"])
        out = result.output
        assert "will NOT be seeded" in out, (
            f"no early warning about the read-only config root:\n{out}")
        assert "is not writable" in out, f"the reason is not named:\n{out}"
        assert calls, "the envs must still install -- this is a warning"
        # `[1/N] <recipe>` is the banner bootstrap prints as each install
        # starts -- a recipe NAME is no good as a marker here, because
        # "molbuilder" also appears in the env-manager line and in every
        # `bash scripts/install-env.sh ...` hint.
        assert "[1/" in out, f"no install banner to order against:\n{out}"
        assert out.index("will NOT be seeded") < out.index("[1/"), (
            "the warning came AFTER the installs started, which is the whole "
            f"defect:\n{out}")
    finally:
        _os.chmod(ro, 0o700)


def test_bootstrap_without_yes_and_without_a_terminal_says_so_up_front(
        monkeypatch):
    """The documented form is `bootstrap` with no `--yes`, which has two
    questions to ask.  Under nohup / CI / a batch step there is nobody to ask,
    so the seeding aborts at the end -- knowable at the start."""
    _bind()
    from molbuilder.envs import _cli
    monkeypatch.setattr(_cli._hints, "stdin_can_answer", lambda: False)
    assert _cli._warn_about_seeding_now(auto_yes=False) is True
    # ...and --yes is exactly the answer, so it must NOT warn then.
    assert _cli._warn_about_seeding_now(auto_yes=True) is False


def test_bootstrap_that_could_not_seed_the_config_exits_nonzero(monkeypatch):
    """A bootstrap that installed everything and seeded NOTHING is not a
    success, and the exit code is the only part a script reads.

    The seeding failure is deliberately non-fatal on the spot -- a read-only
    $HOME must not throw away forty minutes of built envs, nor take the doctor
    report with it.  But until 2026-09-12 it was not RECORDED either: the
    except block printed a warning and the exit code came from doctor alone, so
    `bootstrap` returned 0 having created no config directory at all.  Every
    later verb then refuses for want of `script_generation.activation`.

    The realistic trigger is the form the guide shows: `bootstrap` without
    `--yes` has two questions to ask, and with no tty on stdin (nohup, CI, a
    batch step) click aborts on the first.
    """
    _bind()
    from molbuilder.envs import _cli
    install_stub, _calls = _make_install_stub()
    monkeypatch.setattr(_cli._install, "run_install", install_stub)
    # Doctor is HEALTHY -- that is the whole point: its exit code must not be
    # what speaks for a bootstrap that failed to seed.
    monkeypatch.setattr(_cli._doctor, "report_all", lambda caps, **kw: [])
    monkeypatch.setattr(_cli, "_render_doctor", lambda reports: 0)

    def _refuse(*a, **kw):
        raise OSError("Read-only file system: '/config/molbuilder.json'")
    monkeypatch.setattr(_cli, "_seed_config", _refuse)

    from molbuilder import diagnostics as _diag_mod
    monkeypatch.setattr(_diag_mod, "detect",
                        lambda: Capabilities(
                            runtime_config={}, conda_binary="/c/bin",
                            conda_envs=frozenset()))

    result = _make_runner().invoke(_cli.envs_group, ["bootstrap", "--yes"])
    assert result.exit_code != 0, (
        "bootstrap seeded no config directory and still reported success; "
        f"exit_code={result.exit_code}\n{result.output}")
    assert "could not seed the config directory" in result.output, (
        "the reason must still be printed -- a non-zero exit with no stated "
        f"cause is its own defect:\n{result.output}")


def test_bootstrap_runs_install_for_each_conda_only_recipe(monkeypatch):
    """Default invocation: install the DEFAULT STACK and nothing a recipe
    says is opt-in.

    The rule is `Recipe.opt_in is None` (2026-09-14).  It was
    `build_spec is None` -- a proxy that meant "expensive, so ask first" and
    happened to hold for the only opt-in env there was; the notebook env is
    cheap, conda-only and still opt-in.  All envs are absent here so
    --skip-existing has nothing to skip."""
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
        r.name for r in BUILTIN_RECIPES if r.opt_in is None
    ]
    forbidden = [
        r.name for r in BUILTIN_RECIPES if r.opt_in is not None
    ]
    assert install_calls == expected, (
        f"Expected the default stack to be installed in order; "
        f"got {install_calls}, expected {expected}.\n"
        f"output:\n{result.output}"
    )
    for name in forbidden:
        assert name not in install_calls, (
            f"Source-build recipe {name!r} should NOT install by "
            f"default; needs --include-source-builds.")


def test_bootstrap_include_source_builds_adds_them(monkeypatch):
    """``--include-source-builds`` opts the user into the source-build
    recipes too -- the default stack plus every source build."""
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
    # The flag is about SOURCE BUILDS, and since 2026-09-14 that is not the
    # same set as "everything opt-in": the notebook env is opt-in and is not
    # a source build, so this flag does not reach it.  That is the point of
    # the separation -- `--include-source-builds` opts into a COST, and an
    # optional feature is chosen by name.
    expected_names = {r.name for r in BUILTIN_RECIPES
                      if r.opt_in is None or r.build_spec is not None}
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
        r.name for r in BUILTIN_RECIPES if r.opt_in is None
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
    """Create a stub ``mamba`` binary at ``bin_dir/mamba`` PLUS the
    env-python it REPORTS, out of tree at ``<bin_dir.parent>/scratch/
    molbuilder/bin/python``.

    Out of tree on purpose.  The env is nowhere near the manager binary --
    no rule of thumb about install roots reaches it -- so the shim resolves
    it only by asking the manager where it is (`installation.md` M2), which
    is the arrangement every module-provided mamba on an HPC login node
    actually has.

    The stubs fake:
      * ``mamba env list`` -> output that either includes or omits
        the host env (controlled by ``host_env_present``), naming the
        prefix the same way a real manager does -- and listing it from
        then on once ``create`` has run, which is the one behaviour that
        makes "create, then resolve" work.  The stub used to report the
        env absent forever, and the shim found a python regardless by
        deriving one from the manager's own path; with that derivation
        gone (`installation.md` M2) a stub that does not honour its own
        create is simply a broken manager.
      * ``mamba config --get channels`` -> ``--add channels '<name>'``
        lines per configured channel (empty tuple = fresh conda).
      * ``mamba info --json`` -> empty, so ``env list`` is the only
        answer; resolving anyway proves one reading suffices.
      * ``mamba create`` -> echo ``[stub-create] $*``.
      * The env's python -> echo ``[stub-dispatch] python $*`` and
        ``[stub-env] PYTHONPATH=$PYTHONPATH ...`` so tests can assert
        what got forwarded.  Replaces the old ``mamba run``-driven
        dispatch (the shim now bypasses mamba run to dodge the
        mamba 1.x ``exec --`` bug).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    mamba = bin_dir / "mamba"
    env_prefix = bin_dir.parent / "scratch" / "molbuilder"
    env_list_lines = ["# conda environments:", "#", "base    /root"]
    if host_env_present:
        env_list_lines.append(f"molbuilder    {env_prefix}")
    env_list_output = "\n".join(env_list_lines) + "\n"
    created_marker = bin_dir.parent / ".stub-created"
    created_line = f"molbuilder    {env_prefix}"
    config_lines = [f"--add channels '{ch}'" for ch in configured_channels]
    config_output = ("\n".join(config_lines) + "\n") if config_lines else ""
    mamba.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "env" && "$2" == "list" ]]; then\n'
        f"  cat <<'EOF'\n{env_list_output}EOF\n"
        f'  if [[ -e "{created_marker}" ]]; then echo "{created_line}"; fi\n'
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
        f'  : > "{created_marker}"\n'
        "  exit 0\n"
        "fi\n"
        'echo "[stub-mamba] $*"\n'
        "exit 0\n"
    )
    mamba.chmod(0o755)
    # The python goes where the stub manager just said the env is, which
    # is what a real manager guarantees and the only thing the shim relies
    # on.  Put it there even when host_env_present is False: the shim is
    # then meant to create the env first, and a test that wants to see the
    # create must not fail one step earlier for want of a python.
    env_python_dir = env_prefix / "bin"
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


def test_shim_bootstrap_dry_run_refuses_to_create_the_host_env(tmp_path):
    """``--dry-run`` must not perform the one install this script owns.

    The shim creates the host env before handing off, and the Python
    layer is what honours --dry-run -- so on a fresh machine the flag
    that promises "do not install" used to trigger a multi-GB conda
    create and only THEN print a plan.  Planning genuinely needs the env
    (the planner lives in it), so the answer is to say so, not to
    install.
    """
    r = _run_install_env_sh(["bootstrap", "--dry-run", "--yes"],
                            tmp_path=tmp_path, host_env_present=False)
    assert r.returncode == 2, (
        f"expected a refusal, got rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    assert "[stub-create]" not in (r.stdout + r.stderr), (
        "--dry-run created the host env anyway")
    assert "bootstrap --yes" in r.stderr, (
        "the refusal must name the command that makes the dry run work")


def test_shim_bootstrap_creates_host_env_without_dry_run(tmp_path):
    """The control for the test above: the same invocation WITHOUT
    --dry-run still auto-creates, which is bootstrap's whole job."""
    r = _run_install_env_sh(["bootstrap", "--yes"],
                            tmp_path=tmp_path, host_env_present=False)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "[stub-create]" in (r.stdout + r.stderr), (
        "bootstrap must create the host env when it is missing")


def test_shim_help_needs_no_tty(tmp_path):
    """``<subcommand> --help`` must reach the Python CLI without a TTY.

    The usage text points at `<subcommand> --help` as THE flag reference --
    deliberately, so Python stays the single source of truth instead of the
    shim carrying a second copy that drifts.  That pointer is only true if
    asking for help does not stop at the env-manager confirmation: it used
    to, and in any non-TTY (CI, a pipe, an editor shell) exited 2 with "no
    TTY for confirmation" instead of printing anything.
    """
    r = _run_install_env_sh(["repair", "--help"], tmp_path=tmp_path)
    assert r.returncode == 0, (
        f"--help must not need a TTY or a --yes: rc={r.returncode}\n"
        f"{r.stdout}\n{r.stderr}")
    assert "no TTY for confirmation" not in r.stderr
    assert "[stub-dispatch]" in (r.stdout + r.stderr), (
        "--help must be forwarded to the Python CLI, which owns the flags")


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


def test_activation_wrapper_survives_a_command_containing_shell_metacharacters():
    """The generated shell must not break the command it carries.

    The wrapper builds one bash string around the step's own command, so every
    quoting decision in it is load-bearing.  It has been got wrong before: the
    diagnostic line it used to emit logged the inner command with
    ``echo "[bypass] cmd={shlex.quote(...)}"``, and shlex.quote emits SINGLE
    quotes, inert inside a DOUBLE-quoted echo -- so a command whose own text
    contained a double quote closed the echo early and the remainder was parsed
    as shell.  The first step with one (the siesta-gpu toolchain shims, whose
    body has ``B="$CONDA_PREFIX/bin"`` and a ``link() {`` function) died with
    ``syntax error near unexpected token '('``, raised by the DIAGNOSTIC line,
    about a command that was perfectly valid.  Those echoes are gone with the
    wrapper's demotion to a fallback; the quoting they exposed is still here.

    Everything a real step throws at it: double quotes, parentheses, a
    function definition, ``$(( ))`` arithmetic, single quotes, and a
    backslash.
    """
    import shlex
    import subprocess

    from molbuilder.envs.builds import activation_wrapper

    nasty = ('set -e; B="$X/bin"; f() { echo "a(b)"; }; n=$((1+1)); '
             "g='single'; h=\"embedded 'single' inside double\"; "
             'printf "%s\\n" "$B$n$g$h"')
    argv = ("conda", "run", "-n", "someenv", "--no-capture-output",
            "bash", "-c", nasty)
    new_argv = activation_wrapper(argv, "/tmp/does-not-matter")

    wrapper = new_argv[-1]
    cp = subprocess.run(["bash", "-n", "-c", wrapper],
                        capture_output=True, text=True, timeout=30)
    assert cp.returncode == 0, (
        f"generated wrapper is not valid bash:\n{cp.stderr}")

    # A second, independent lexer must also find the quoting balanced --
    # `bash -n` and shlex disagreeing would mean we got lucky, not right.
    shlex.split(wrapper)          # raises ValueError if a quote dangles

    # The wrapper carries the inner script verbatim -- it is exec'd, so any
    # re-quoting of it would change what runs.
    assert 'f() { echo "a(b)"; }' in wrapper


def test_the_install_log_lands_in_the_home_of_the_moment(monkeypatch, tmp_path):
    """The log root is a QUESTION, asked when asked.  As a module constant
    it froze the developer's real home at import time, and no amount of
    later HOME isolation could redirect it -- which is how every full-suite
    run left five zero-byte install logs in the real ``~/.molbuilder/logs``
    (2026-08-28).  Mutation: restore the constant and this fails."""
    from molbuilder.envs import _cli
    monkeypatch.setenv("HOME", str(tmp_path))
    p = _cli._resolve_install_log_path("molbuilder-siesta")
    assert str(p).startswith(str(tmp_path)), (
        f"the install log escaped the current HOME: {p}")
