"""L1 tests for ``molbuilder.envs.doctor``.

Pins the report shape and the present/missing logic without running
real conda subprocesses; the verify_argv dispatch is exercised by
plugging a fake conda binary into capabilities and patching
``subprocess.run``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from molbuilder import envs
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import doctor
from molbuilder.envs.recipes import BUILTIN_RECIPES, Recipe


def _bind(*, conda_envs=(), conda_binary="/usr/bin/conda",
          envs_override=None):
    cfg = {"envs": envs_override} if envs_override is not None else {}
    set_capabilities(Capabilities(
        runtime_config=cfg,
        conda_binary=conda_binary,
        conda_envs=frozenset(conda_envs),
    ))


# --------------------------------------------------------------------- #
#  report_all: shape + present/missing                                   #
# --------------------------------------------------------------------- #


def test_report_all_marks_every_env_missing_on_empty_machine():
    _bind(conda_envs=())
    reports = doctor.report_all(run_verify=False)
    assert {r.recipe.name for r in reports} == {
        r.name for r in BUILTIN_RECIPES
    }
    for r in reports:
        assert r.present is False
        assert r.verify_ok is None
        assert r.verify_output == ""


def test_report_all_marks_envs_present_when_in_caps():
    _bind(conda_envs=("molbuilder", "molbuilder-siesta"))
    reports = doctor.report_all(run_verify=False)
    by_name = {r.effective_name: r for r in reports}
    assert by_name["molbuilder"].present is True
    assert by_name["molbuilder-siesta"].present is True
    assert by_name["molbuilder-pySCF"].present is False


def test_report_all_honours_envs_override():
    """When molbuilder.json renames siesta -> my-siesta, the
    doctor's effective_name picks up the override."""
    _bind(
        conda_envs=("my-siesta",),
        envs_override={"siesta": "my-siesta"},
    )
    reports = doctor.report_all(run_verify=False)
    siesta = next(r for r in reports
                  if r.recipe.category == "siesta")
    assert siesta.effective_name == "my-siesta"
    assert siesta.present is True


def test_report_all_skip_verify_does_not_call_subprocess(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    called = []
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda *a, **kw: called.append(a) or
                                         MagicMock(returncode=0,
                                                   stdout="", stderr=""))
    doctor.report_all(run_verify=False)
    assert called == [], (
        "run_verify=False must not invoke subprocess at all"
    )


# --------------------------------------------------------------------- #
#  Verify command dispatch + exit-code semantics                         #
# --------------------------------------------------------------------- #


def _stub_completed(stdout="", stderr="", returncode=0):
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def test_verify_ok_when_returncode_zero_and_substring_matches(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="siesta 5.4.2",
                                         returncode=0),
    )
    reports = doctor.report_all()
    siesta = next(r for r in reports if r.recipe.category == "siesta")
    assert siesta.verify_ok is True


def test_verify_fails_when_returncode_nonzero(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="siesta 5.4.2",
                                         returncode=1),
    )
    reports = doctor.report_all()
    siesta = next(r for r in reports if r.recipe.category == "siesta")
    assert siesta.verify_ok is False


def test_verify_fails_when_substring_missing(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="unrelated text",
                                         returncode=0),
    )
    reports = doctor.report_all()
    siesta = next(r for r in reports if r.recipe.category == "siesta")
    assert siesta.verify_ok is False


def test_verify_ignore_exit_code_only_checks_substring(monkeypatch):
    """The MDtools recipe sets verify_ignore_exit_code=True;
    a non-zero exit must still report OK when the substring is
    present (mirrors tleap's real behaviour)."""
    _bind(conda_envs=("molbuilder-MDtools",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="Welcome to LEaP!",
                                         returncode=1),
    )
    reports = doctor.report_all()
    md = next(r for r in reports if r.recipe.category == "mdtools")
    assert md.verify_ok is True


def test_verify_ignore_exit_code_still_checks_substring(monkeypatch):
    """If ignore_exit_code=True, substring is the ONLY signal;
    missing substring must still fail."""
    _bind(conda_envs=("molbuilder-MDtools",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="no banner here",
                                         returncode=0),
    )
    reports = doctor.report_all()
    md = next(r for r in reports if r.recipe.category == "mdtools")
    assert md.verify_ok is False


def test_verify_output_trimmed_to_2k(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda *a, **kw: _stub_completed(stdout="x" * 10000,
                                         returncode=0),
    )
    reports = doctor.report_all()
    siesta = next(r for r in reports if r.recipe.category == "siesta")
    assert len(siesta.verify_output) <= 2048
