"""The doctor's report carries its own fix commands (user, 2026-08-20).

The rule: **every problem the doctor detects ends with the exact command
that fixes it**, in the ONE spelling that works from a bare shell in any
folder (``bash <checkout>/scripts/install-env.sh ...``, `envs.hints.fix_cmd`)
-- a person looking at a broken env
must never have to tour the docs to learn the verb.

The hints are KIND-PRECISE because bare ``repair`` skips version/build
mismatches by design: a version-only failure hinted at bare repair would
prescribe a no-op.
"""
from __future__ import annotations

import pytest

from molbuilder.envs._cli import _fix_cmd, _render_doctor
from molbuilder.envs.doctor import (EnvReport, PackageAudit,
                                    PackageAuditIssue)
from molbuilder.diagnostics import DEFAULT_ENV_NAMES
from molbuilder.envs.recipes import builtin_recipes

# CHOSEN BY THE PROPERTY UNDER TEST, not by index.  These tests split on
# whether a recipe has post-install steps, because that is what decides how
# many commands `doctor` prints -- so the fixtures ask for the property
# rather than trusting a position in the registry to keep it.
# `extra_set()`, not `extra_steps`: the DEFAULT set is what decides how many
# commands doctor prints, and the two differ -- `_HOST`
# declares an opt-in chromium step that a default install never dispatches.
# A BACKEND's: the host env's fixes are its own (`hints.install_again_cmd`),
# rows below.
RECIPE = next(r for r in builtin_recipes()
              if r.category is not None and not r.extra_set())
NAME = RECIPE.name
#: The host env's recipe -- the env the install script runs its verbs from.
_HOST_RECIPE = next(r for r in builtin_recipes() if r.category is None)

#: Each that DOES have post-install steps on a default run, for the other
#: branch -- the host env's (its path file) among them.
RECIPES_WITH_STEPS = [r for r in builtin_recipes() if r.extra_set()]


def _report(**kw):
    base = dict(recipe=RECIPE, effective_name=NAME, present=True,
                verify_ok=True, verify_output="", package_audit=None)
    base.update(kw)
    return EnvReport(**base)


def _audit(*issues):
    return PackageAudit(checked=True, n_conda_declared=4, n_pip_declared=1,
                        issues=tuple(issues))


def _issue(kind, spec="somepkg=1.0"):
    # `optional` is a field the audit sets; a hand-built issue sets it the
    # way the audit does -- opt-in absence included (doctor.py: an opt-in
    # package nobody asked for is not a REQUIRED gap).
    return PackageAuditIssue(kind=kind, spec=spec, found="(not found)",
                             optional=kind.endswith(("-optional", "-opt-in")))


def _render(capsys, rep):
    code = _render_doctor([rep])
    # ONE readouterr: a second call reads streams the first already
    # drained.
    cap = capsys.readouterr()
    return code, cap.out + cap.err


#: (which env is missing, its recipe, the fix doctor names): the host env is
#: made by `bootstrap`, which makes the molbuilder this command runs -- the
#: install script refuses `install` for it -- and any other by `install`
#: (`ops/installation.md` § 2: every problem ends with its exact fix command).
MISSING_ROWS = [
    ("the host env", _HOST_RECIPE, ("bootstrap", "--yes")),
    ("a backend env", RECIPE, ("install", NAME, "--yes")),
]


@pytest.mark.parametrize("which, recipe, fix", MISSING_ROWS,
                         ids=[r[0] for r in MISSING_ROWS])
def test_a_missing_env_carries_the_command_that_makes_it(capsys, which,
                                                         recipe, fix):
    """`ops/installation.md` § 2: a missing env's `next:` line is its exact
    fix command -- one the install script runs.

    THE SILENT FAILURE: a missing host env told to run `install molbuilder
    --yes`, which the install script refuses."""
    code, out = _render(capsys, _report(recipe=recipe,
                                        effective_name=recipe.name,
                                        present=False, verify_ok=None))
    assert code == 0            # missing is informational, as before
    assert "next:    " + _fix_cmd(*fix) in out, out
    assert "molbuilder envs install" not in out, (
        "the second spelling is back -- one spelling, the shell form")


def test_an_opt_in_row_never_silences_a_required_failure(capsys):
    """Both reports, and the exit code still says FAILED.

    The opt-in row was added BETWEEN the audit chain's first `if` and its
    `elif` on 2026-09-18, which re-parented the `elif`/`else` onto it -- so
    for the one recipe that has opt-in packages (the host env, on every
    machine that has not run `--with-dev-tools`) the FAILED branch became
    unreachable: a REQUIRED missing package printed nothing and `doctor`
    exited 0 over it.  `bootstrap` keys its verdict on that exit code, so it
    would have announced "every env installed and verified" over the exact
    failure its verdict block was written to catch.

    Rendered, not inspected: the defect was invisible to a test that asked
    `PackageAuditIssue` about its own fields, because the fields were right
    and the branch that read them was unreachable.
    """
    rep = _report(package_audit=_audit(_issue("conda-missing"),
                                       _issue("conda-missing-opt-in",
                                              spec="nodejs")))
    code, out = _render(capsys, rep)
    assert code == 1, "an opt-in absence must not turn a failure into a pass"
    assert "audit:   FAILED" in out
    assert "next:    " + _fix_cmd("repair", NAME) in out
    # ...and the opt-in row is still there, with the command that adds it.
    assert "opt-in:" in out
    assert "add:     " + _fix_cmd("install", NAME, "--with-dev-tools",
                                  "--yes") in out


def test_required_missing_packages_carry_the_repair_command(capsys):
    rep = _report(package_audit=_audit(_issue("conda-missing")))
    code, out = _render(capsys, rep)
    assert code == 1
    assert "next:    " + _fix_cmd("repair", NAME) in out


@pytest.mark.parametrize("recipe", RECIPES_WITH_STEPS, ids=lambda r: r.name)
def test_a_recipe_with_post_install_steps_needs_both_verbs(capsys, recipe):
    """**Neither verb alone finishes it**, so doctor names both.

    `repair` installs what the audit reported and does not re-run
    `extra_steps`.  `install` runs the whole plan but SKIPS THE CREATE STEP on
    an env that already exists -- and conda packages enter a plan only through
    create, so it adds no missing package.  The host env's step, its path
    file naming this checkout, is `bootstrap`'s (`ops/deployment.md` § 0).

    Measured 2026-09-14: `install` on a present env goes from "conda create:
    SKIPPED" straight to verify, having installed nothing.
    """
    rep = _report(recipe=recipe, effective_name=recipe.name,
                  package_audit=_audit(_issue("conda-missing")))
    code, out = _render(capsys, rep)
    assert code == 1
    # The one that installs the missing package comes FIRST.
    assert "next:    " + _fix_cmd("repair", recipe.name) in out
    again = (_fix_cmd("bootstrap", "--yes") if recipe.category is None
             else _fix_cmd("install", recipe.name, "--yes"))
    assert "then:    " + again in out
    assert "post-install steps" in out


def test_a_version_only_failure_hints_the_flag_that_actually_fixes_it(
        capsys):
    rep = _report(package_audit=_audit(_issue("conda-version")))
    code, out = _render(capsys, rep)
    assert code == 1
    assert _fix_cmd("repair", NAME, "--include-version-fix") in out
    assert "makes repair rebuild those" in out


def test_optional_only_names_the_enable_command_and_stays_ok(capsys):
    rep = _report(package_audit=_audit(_issue("pip-missing-optional")))
    code, out = _render(capsys, rep)
    assert code == 0            # optional-only is not a failure
    assert _fix_cmd("repair", NAME, "--include-optional") in out


#: (which env failed its verify, its recipe, the rebuild doctor names): a
#: backend's by `install --clean`; the host env's -- the install script runs
#: its verbs from it, so its `--clean` is refused there -- by removing it from
#: a shell where it is not active, then `bootstrap` (`installation.md` M5).
FAILED_VERIFY_ROWS = [
    ("a backend env", RECIPE,
     [_fix_cmd("install", NAME, "--clean", "--yes")]),
    ("the host env", _HOST_RECIPE,
     [f"/opt/mgr/bin/micromamba env remove -n {_HOST_RECIPE.name} -y",
      _fix_cmd("bootstrap", "--yes")]),
]


@pytest.mark.parametrize("which, recipe, rebuild", FAILED_VERIFY_ROWS,
                         ids=[r[0] for r in FAILED_VERIFY_ROWS])
def test_a_failed_verify_offers_repair_then_the_rebuild(capsys, which,
                                                        recipe, rebuild):
    """`ops/installation.md` § 2: every problem ends with its exact fix
    command -- one the install script runs.

    THE SILENT FAILURE: a host env told `install molbuilder --clean --yes`,
    which the install script refuses from inside that env."""
    rep = _report(recipe=recipe, effective_name=recipe.name,
                  verify_ok=False, verify_output="boom",
                  manager="/opt/mgr/bin/micromamba")
    code, out = _render(capsys, rep)
    assert code == 1
    assert "next:    " + _fix_cmd("repair", recipe.name) in out, out
    for line in rebuild:
        assert line in out, out
    if recipe.category is None:
        assert _fix_cmd("install", recipe.name, "--clean", "--yes") \
            not in out, out


def test_a_printed_env_command_runs_typed_in_any_folder(tmp_path):
    """`ops/installation.md` § 2 (the user, 2026-11-22: "Launcher by full
    path"): an env command molbuilder prints names the install script by
    the checkout's full path, so typed in a folder outside the checkout --
    a calculation's -- it runs.  Asked for its usage, which changes nothing.
    THE SILENT FAILURE: `bash scripts/install-env.sh ...` typed there,
    answered "No such file or directory"."""
    import os
    import subprocess
    from molbuilder.envs import hints
    done = subprocess.run(["bash", "-c", _fix_cmd("--help")], cwd=tmp_path,
                          env={**os.environ, "HOME": str(tmp_path)},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "env installer" in done.stdout, done.stdout
    # ...AND WHAT IT PRINTS TO TYPE names it so too -- its examples, and the
    # first step it names when asked nothing.
    assert _fix_cmd("bootstrap", "--yes") in done.stdout, done.stdout
    asked_nothing = subprocess.run(
        ["bash", "-c", hints.LAUNCHER], cwd=tmp_path,
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=120)
    assert _fix_cmd("bootstrap", "--yes") in asked_nothing.stderr, \
        asked_nothing.stderr


@pytest.mark.parametrize("calculation, template", [
    ("optimization", None),
    ("vibration", {"compute_raman": True}),
], ids=["a PySCF optimization, its geomeTRIC remedy",
        "a PySCF vibration with Raman, its pyscf-properties remedy"])
def test_a_checkout_path_the_shell_quotes_still_writes_a_deck_that_parses(
        tmp_path, monkeypatch, calculation, template):
    """A PySCF deck prepared for this machine names the env command in its
    own Python, by this checkout's full path, standing there word for word
    whatever the path holds (`ops/installation.md` § 2; `execution/
    script-preparation.md` § 4.3: check catches a generated program that
    does not parse).  Prepared on the road, it parses.  THE SILENT FAILURE:
    the quoted path closing a string early -- prep refusing every PySCF
    calculation of a checkout in a folder with an apostrophe, a fault the
    person cannot fix."""
    from molbuilder.envs import hints
    from support.road import (QUOTED_CHECKOUT_SEEN, QUOTED_LAUNCHER, WATER,
                              described_and_prepared, road_structure)
    monkeypatch.setattr(hints, "LAUNCHER", QUOTED_LAUNCHER)
    _dest, _stage, deck = described_and_prepared(
        tmp_path, monkeypatch, road_structure(WATER), "pyscf",
        calculation=calculation, template=template)
    compile(deck, "deck.py", "exec")
    assert QUOTED_CHECKOUT_SEEN in deck, "the deck names no remedy"


@pytest.mark.parametrize("env", ["pyscf", "siesta-gpu"])
def test_a_checkout_path_the_shell_quotes_still_writes_an_env_check_that_parses(
        monkeypatch, env):
    """An env's check is a program -- PySCF's Python run by ``python -c``,
    the GPU build's shell run by ``bash -c`` -- naming its repair command
    (`ops/installation.md` § 2), which stands in it word for word whatever
    the path holds.  THE SILENT FAILURE: a healthy env reported failing, its
    check never parsed."""
    import subprocess
    from molbuilder.envs import hints, recipes
    from support.road import QUOTED_CHECKOUT_SEEN, QUOTED_LAUNCHER
    monkeypatch.setattr(hints, "LAUNCHER", QUOTED_LAUNCHER)
    # THE RECIPES BUILT AFRESH, as a process started in that checkout builds
    # them: the registry is built once a process (`builtin_recipes`, cached).
    monkeypatch.setattr(recipes, "builtin_recipes",
                        recipes.builtin_recipes.__wrapped__)
    program, flag, check = recipes.recipe_by_name(
        DEFAULT_ENV_NAMES[env]).verify_argv
    assert QUOTED_CHECKOUT_SEEN in check, "the check names no repair"
    if program == "python":
        compile(check, "check", "exec")
    else:
        parsed = subprocess.run([program, "-n", flag, check],
                                capture_output=True, text=True, timeout=60)
        assert parsed.returncode == 0, parsed.stderr


def test_a_host_env_that_does_not_find_molbuilder_is_pointed_back(capsys):
    """`ops/deployment.md` § 0, *When something moves*: the checkout moved,
    the host env's check fails -- molbuilder not found -- and doctor names
    `bootstrap`, which writes the path file again, after `repair`, which
    installs packages only and runs no extra step.  THE SILENT FAILURE: a
    moved checkout answered with `repair` alone, which finds nothing to do,
    and doctor failing again."""
    rep = _report(recipe=_HOST_RECIPE, effective_name=_HOST_RECIPE.name,
                  verify_ok=False,
                  verify_output="ModuleNotFoundError: No module named "
                                "'molbuilder'",
                  manager="/opt/mgr/bin/micromamba")
    code, out = _render(capsys, rep)
    assert code == 1
    assert "next:    " + _fix_cmd("repair", _HOST_RECIPE.name) in out, out
    assert "then:    " + _fix_cmd("bootstrap", "--yes") in out, out


def test_a_failed_verify_on_the_RUNNING_env_does_not_offer_clean(capsys):
    """`installation.md` M5 -- a printed remedy may not destroy working state.

    `--clean --yes` on the env molbuilder runs from removes the prefix holding
    the interpreter that is mid-install; the removal succeeds and the
    reinstall cannot.  So for THAT env the report offers repair and the
    three-step rebuild from outside, and never the one-liner.

    The hint rule still holds: the rebuild is named, with the DETECTED manager
    (M3), not replaced by a pointer to the docs.
    """
    import sys as _sys

    rep = _report(verify_ok=False, verify_output="boom",
                  prefix=_sys.prefix, manager="/opt/mgr/bin/micromamba")
    code, out = _render(capsys, rep)

    assert code == 1
    assert "next:    " + _fix_cmd("repair", NAME) in out, out
    assert _fix_cmd("install", NAME, "--clean", "--yes") not in out, (
        "doctor still hands over the command that deletes the env it is "
        f"reporting on:\n{out}")
    assert f"/opt/mgr/bin/micromamba env remove -n {NAME} -y" in out, out


def test_the_closing_line_points_at_the_hints_not_the_docs(capsys):
    rep = _report(verify_ok=False, verify_output="")
    _code, out = _render(capsys, rep)
    assert "next:" in out, "the combined capture must include stderr"
    assert "docs/ops/installation.md" not in out
    assert "`next:` fix command" in out, (
        "the closing line must point at the hints")


#: (the command a person runs on a missing env): every surface that finds
#: an env missing prints the one fix `hints.install_again_cmd` gives --
#: doctor's rows above, these two here.
MISSING_COMMANDS = ["repair", "validate"]


@pytest.mark.parametrize("command", MISSING_COMMANDS)
@pytest.mark.parametrize("which, recipe, fix", MISSING_ROWS,
                         ids=[r[0] for r in MISSING_ROWS])
def test_a_missing_env_named_to_a_command_carries_its_fix(
        command, which, recipe, fix):
    """`ops/installation.md` § 2: every problem ends with its exact fix
    command -- `repair` and `validate` on an env that is not there, as
    `doctor` on one.

    THE SILENT FAILURE: `repair molbuilder` on a machine whose host env is
    gone told to `install molbuilder --yes`, which the install script
    refuses."""
    from click.testing import CliRunner
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.envs._cli import envs_group
    set_capabilities(Capabilities(runtime_config={},
                                  conda_binary="/usr/bin/conda",
                                  conda_envs={}))
    result = CliRunner().invoke(envs_group, [command, recipe.name])
    said = result.output + (getattr(result, "stderr", "") or "")
    assert _fix_cmd(*fix) in said, said


#: (the command, what it is typed with, the command the sentence names --
#: its verb and recipe): every verb that needs the package manager, refused
#: in the one sentence.
_SOURCE_BUILT = DEFAULT_ENV_NAMES["siesta-gpu"]
NO_MANAGER_COMMANDS = [
    ("doctor", ["doctor"], ["doctor"]),
    ("repair", ["repair", NAME], ["repair", NAME]),
    ("clean", ["clean", _SOURCE_BUILT], ["clean", _SOURCE_BUILT]),
    ("validate", ["validate", _SOURCE_BUILT], ["validate", _SOURCE_BUILT]),
    ("install", ["install", NAME, "--check"], ["install", NAME]),
    ("bootstrap", ["bootstrap", "--dry-run"], ["bootstrap"])]
#: (why there is none, what the resolver said, the sentence's fix): none
#: loaded -- load it; the one molbuilder.json records not runnable -- fix
#: the record, since a recorded manager wins (`hints.no_manager`).
NO_MANAGER_CAUSES = [
    ("none found", None, "Load it in this shell"),
    ("the recorded one cannot run",
     "molbuilder.json envs.manager = '/gone/bin/mamba' is not an executable "
     "file on this machine", "Fix `envs.manager` there"),
]


@pytest.mark.parametrize("cause, refusal, fix", NO_MANAGER_CAUSES,
                         ids=[r[0] for r in NO_MANAGER_CAUSES])
@pytest.mark.parametrize("command, words, named", NO_MANAGER_COMMANDS,
                         ids=[r[0] for r in NO_MANAGER_COMMANDS])
def test_a_command_with_no_package_manager_says_the_one_sentence(
        command, words, named, cause, refusal, fix):
    """`ops/installation.md`, *The install script is not the CLI*: a command
    that finds no package manager "says so in one sentence: load it in this
    shell and run the command again" -- and, where molbuilder.json's
    recorded one is what does not run, that the record is the fix.

    THE SILENT FAILURE: `validate` telling a person whose env is there to
    install it again; `clean` naming another verb; "load it" where loading
    cannot help."""
    from click.testing import CliRunner
    from molbuilder.diagnostics import Capabilities, set_capabilities
    from molbuilder.envs._cli import envs_group
    set_capabilities(Capabilities(runtime_config={}, conda_binary=None,
                                  conda_binary_source=refusal,
                                  conda_envs={}))
    result = CliRunner().invoke(envs_group, words)
    said = result.output + (getattr(result, "stderr", "") or "")
    assert result.exit_code != 0, said
    assert f"`{_fix_cmd(*named)}` cannot run" in said, said
    assert fix in said, said
