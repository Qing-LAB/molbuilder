"""L1 tests for ``molbuilder.envs.doctor``.

Pins the report shape and the present/missing logic without running
real conda subprocesses; the verify_argv dispatch is exercised by
plugging a fake conda binary into capabilities and patching
``subprocess.run``.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from molbuilder import envs
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import doctor
from molbuilder.envs.recipes import BUILTIN_RECIPES, PipPackage, Recipe


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
    # doctor.py now bypasses ``<mgr> run`` via install._bypass_conda_run
    # (mamba 2.x ``exec --`` workaround); patch _env_prefix so the
    # bypass code path runs.
    import molbuilder.envs.install as _install
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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
    import molbuilder.envs.install as _install
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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
    import molbuilder.envs.install as _install
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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
    import molbuilder.envs.install as _install
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
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


# --------------------------------------------------------------------- #
#  Provenance: which TREE is installed, when the version cannot say      #
# --------------------------------------------------------------------- #

def _fake_dist(site_packages, name, version, *, url=None, commit=None):
    """Write the dist-info pip would write for one installed package."""
    d = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    if url is not None:
        body = {"url": url}
        if commit is not None:
            body["vcs_info"] = {"vcs": "git", "commit_id": commit}
        (d / "direct_url.json").write_text(json.dumps(body), encoding="utf-8")


@pytest.mark.parametrize("declared_source,installed_url,installed_commit,expect", [
    # The case the whole mechanism exists for: same NAME, same VERSION,
    # different tree.  Only provenance can tell these apart.
    ("git+https://example.invalid/o/r.git", None, None, "pip-source"),
    ("git+https://example.invalid/o/r.git",
     "https://example.invalid/o/r.git", "abc1234", None),
    ("git+https://example.invalid/o/r.git",
     "https://example.invalid/other/r.git", "abc1234", "pip-source"),
    # A ref-pinned declaration must not read as a mismatch just for
    # carrying a ref -- a branch cannot be checked against a commit.
    ("git+https://example.invalid/o/r.git@main",
     "https://example.invalid/o/r.git", "abc1234", None),
    # ...but a pinned SHA is checked, because that is what a pin is for.
    ("git+https://example.invalid/o/r.git@abc1234",
     "https://example.invalid/o/r.git", "abc1234def", None),
    ("git+https://example.invalid/o/r.git@deadbee",
     "https://example.invalid/o/r.git", "abc1234def", "pip-source"),
    # No source declared -> provenance is not this recipe's business.
    (None, "https://example.invalid/anything.git", "abc1234", None),
])
def test_audit_detects_which_tree_is_installed(
        tmp_path, declared_source, installed_url, installed_commit, expect):
    """A package declared with a source is audited on PROVENANCE.

    ``pyscf-properties`` declares version ``0.1.0`` both on PyPI and on
    master, so a version comparison passes on an env that lacks the
    feature.  The audit reads PEP 610 ``direct_url.json`` instead.  Every
    row here holds the version FIXED so nothing but provenance can be
    doing the work.
    """
    sp = tmp_path / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    _fake_dist(sp, "widget", "0.1.0",
               url=installed_url, commit=installed_commit)
    recipe = Recipe(
        name="synthetic", category=None, description="d",
        channels=("conda-forge",), conda_packages=("python",),
        pip_packages=(PipPackage("widget", source=declared_source),),
    )
    # The synthetic prefix has no conda-meta, so the declared conda
    # package audits as missing.  This test is about pip provenance.
    kinds = [i.kind for i in doctor.audit_packages(tmp_path, recipe).issues
             if i.kind.startswith("pip-")]
    assert kinds == ([expect] if expect else [])


def test_a_source_mismatch_the_recipe_accepts_is_only_informational(tmp_path):
    """``fallback_to_index`` means the indexed build is a declared
    acceptable outcome, so landing on it degrades a capability rather
    than breaking the env -- and ``doctor`` must not call it FAILED."""
    sp = tmp_path / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    _fake_dist(sp, "widget", "0.1.0")          # from the index
    recipe = Recipe(
        name="synthetic", category=None, description="d",
        channels=("conda-forge",), conda_packages=("python",),
        pip_packages=(PipPackage("widget",
                                 source="git+https://example.invalid/o/r.git",
                                 fallback_to_index=True,
                                 reason="why this source is unusual"),),
    )
    (issue,) = [i for i in doctor.audit_packages(tmp_path, recipe).issues
                if i.kind.startswith("pip-")]
    assert issue.kind == "pip-source-optional"
    assert issue.found == "(default index)"
    # repair must be handed the SOURCE and the flags, not a bare name --
    # a plain install of a force-flagged package exits 0 and changes
    # nothing, because the versions are identical.
    assert issue.spec.endswith("git+https://example.invalid/o/r.git")
    assert issue.reason == "why this source is unusual"
