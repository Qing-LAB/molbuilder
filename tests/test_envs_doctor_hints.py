"""The doctor's report carries its own fix commands (user, 2026-08-20).

The rule: **every problem the doctor detects ends with the exact command
that fixes it**, in the ONE spelling that works from a bare shell
(``bash scripts/install-env.sh ...``) -- a person looking at a broken env
must never have to tour the docs to learn the verb.  Until this landed the
report listed the problems and closed with "See above +
docs/ops/installation.md", and its missing-env hint used the other
spelling (``molbuilder envs install``).

The hints are KIND-PRECISE because bare ``repair`` skips version/build
mismatches by design: a version-only failure hinted at bare repair would
prescribe a no-op.
"""
from __future__ import annotations

import pytest

from molbuilder.envs._cli import _fix_cmd, _render_doctor
from molbuilder.envs.doctor import (EnvReport, PackageAudit,
                                    PackageAuditIssue)
from molbuilder.envs.recipes import BUILTIN_RECIPES

# CHOSEN BY THE PROPERTY UNDER TEST, not by index.  These tests split on
# whether a recipe has post-install steps, because that is what decides how
# many commands `doctor` prints -- so the fixtures ask for the property
# rather than trusting a position in the registry to keep it.
RECIPE = next(r for r in BUILTIN_RECIPES if not r.extra_steps)
NAME = RECIPE.name

#: One that DOES have post-install steps, for the other branch.
RECIPE_WITH_STEPS = next(r for r in BUILTIN_RECIPES if r.extra_steps)


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
    # way the audit does.
    return PackageAuditIssue(kind=kind, spec=spec, found="(not found)",
                             optional=kind.endswith("-optional"))


def _render(capsys, rep):
    code = _render_doctor([rep])
    # ONE readouterr: a second call reads streams the first already
    # drained, so `.err` was always "" and the closing-line pin below
    # passed vacuously (this review's own catch).
    cap = capsys.readouterr()
    return code, cap.out + cap.err


def test_a_missing_env_carries_the_install_command(capsys):
    code, out = _render(capsys, _report(present=False, verify_ok=None))
    assert code == 0            # missing is informational, as before
    assert "next:    " + _fix_cmd("install", NAME, "--yes") in out
    assert "molbuilder envs install" not in out, (
        "the second spelling is back -- one spelling, the shell form")


def test_required_missing_packages_carry_the_repair_command(capsys):
    rep = _report(package_audit=_audit(_issue("conda-missing")))
    code, out = _render(capsys, rep)
    assert code == 1
    assert "next:    " + _fix_cmd("repair", NAME) in out


def test_a_recipe_with_post_install_steps_needs_both_verbs(capsys):
    """**Neither verb alone finishes it**, so doctor names both.

    `repair` installs what the audit reported and does not re-run
    `extra_steps`.  `install` runs the whole plan but SKIPS THE CREATE STEP on
    an env that already exists -- and conda packages enter a plan only through
    create, so it adds no missing package.

    This printed `install` ALONE until 2026-09-14, justified by a worked
    example (the host env's `ipykernel` kernelspec) that no longer exists.
    The only recipe with post-install steps left is the GPU env, and a missing
    conda package there was being answered with the one command that provably
    cannot install it -- measured the same day: `install` on a present env goes
    from "conda create: SKIPPED" straight to verify, having installed nothing.
    """
    rep = _report(recipe=RECIPE_WITH_STEPS,
                  effective_name=RECIPE_WITH_STEPS.name,
                  package_audit=_audit(_issue("conda-missing")))
    code, out = _render(capsys, rep)
    assert code == 1
    # The one that installs the missing package comes FIRST.
    assert "next:    " + _fix_cmd("repair", RECIPE_WITH_STEPS.name) in out
    assert "then:    " + _fix_cmd(
        "install", RECIPE_WITH_STEPS.name, "--yes") in out
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


def test_a_failed_verify_offers_repair_then_the_rebuild(capsys):
    rep = _report(verify_ok=False, verify_output="boom")
    code, out = _render(capsys, rep)
    assert code == 1
    assert "next:    " + _fix_cmd("repair", NAME) in out
    assert _fix_cmd("install", NAME, "--clean", "--yes") in out


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
