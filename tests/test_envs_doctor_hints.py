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

RECIPE = BUILTIN_RECIPES[0]
NAME = RECIPE.name


def _report(**kw):
    base = dict(recipe=RECIPE, effective_name=NAME, present=True,
                verify_ok=True, verify_output="", package_audit=None)
    base.update(kw)
    return EnvReport(**base)


def _audit(*issues):
    return PackageAudit(checked=True, n_conda_declared=4, n_pip_declared=1,
                        issues=tuple(issues))


def _issue(kind, spec="somepkg=1.0"):
    return PackageAuditIssue(kind=kind, spec=spec, found="(not found)")


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


def test_the_closing_line_points_at_the_hints_not_the_docs(capsys):
    rep = _report(verify_ok=False, verify_output="")
    _code, out = _render(capsys, rep)
    assert "next:" in out, "the combined capture must include stderr"
    assert "docs/ops/installation.md" not in out
    assert "`next:` fix command" in out, (
        "the closing line must point at the hints")
