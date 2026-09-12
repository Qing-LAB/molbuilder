"""L1 tests for ``molbuilder.envs.doctor``.

Pins the report shape and the present/missing logic without running
real conda subprocesses; the verify_argv dispatch is exercised by
plugging a fake conda binary into capabilities and patching
``subprocess.run``.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import builds as _builds, doctor, install as _install
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


def test_report_all_skip_verify_dispatches_nothing(monkeypatch):
    _bind(conda_envs=("molbuilder-siesta",))
    called = []
    monkeypatch.setattr(_builds, "run_streaming",
                        lambda *a, **kw: called.append(a) or (0, ""))
    doctor.report_all(run_verify=False)
    assert called == [], (
        "run_verify=False must not dispatch anything at all"
    )


# --------------------------------------------------------------------- #
#  Verify semantics                                                      #
# --------------------------------------------------------------------- #
#
# The accept rule -- is this output a pass? -- lives in ONE place,
# `InstallStep.accepts`, and both `install` and `doctor` ask it.  So it is
# tested ONCE, here, as a table; five near-identical tests used to drive
# the same rule through `report_all` with different stub outputs, back
# when doctor carried its own copy of it.
#
# What still needs its own test is the WIRING: that doctor's verdict
# really comes from the rule rather than from a local re-derivation.  One
# case proves that, and it has to be the case where the rule and the
# naive "rc == 0" answer DISAGREE -- otherwise a doctor that ignored the
# rule entirely would still pass.


@pytest.mark.parametrize("ignore_rc,expect,rc,output,accepted", [
    # The ordinary rule: the exit code decides.
    (False, None, 0, "anything", True),
    (False, None, 1, "anything", False),
    # An expected substring is an ADDITIONAL requirement, so a command
    # that exits 0 while the thing we asked about is absent is a failure.
    (False, "siesta", 0, "siesta 5.4.2", True),
    (False, "siesta", 0, "unrelated text", False),
    (False, "siesta", 1, "siesta 5.4.2", False),
    # `ignore_exit_code` hands the verdict to the substring ENTIRELY --
    # tleap exits 1 from a healthy start, so its banner is the signal.
    (True, "Welcome to LEaP!", 1, "Welcome to LEaP!", True),
    (True, "Welcome to LEaP!", 0, "no banner here", False),
    # ...but it never extends to a process that did not run.  Ignoring an
    # exit code is not ignoring a missing command.
    (True, "Welcome to LEaP!", None, "", False),
    (True, None, None, "", False),
    (False, None, None, "", False),
])
def test_accept_rule(ignore_rc, expect, rc, output, accepted):
    step = _install.InstallStep(
        label="verify", argv=("true",),
        ignore_exit_code=ignore_rc, expect_contains=expect,
    )
    assert step.accepts(rc, output) is accepted


def test_doctor_verify_uses_the_accept_rule(monkeypatch):
    """MDtools sets verify_ignore_exit_code=True, so a non-zero exit with
    the banner present is a PASS.  A doctor that decided on the exit code
    itself -- as it did while it carried its own copy of the rule --
    reports False here."""
    _bind(conda_envs=("molbuilder-MDtools",))
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(_builds, "run_streaming",
                        lambda *a, **kw: (1, "Welcome to LEaP!"))
    reports = doctor.report_all()
    md = next(r for r in reports if r.recipe.category == "mdtools")
    assert md.verify_ok is True


def test_verify_output_trimmed_to_2k(monkeypatch):
    """The installer keeps 4 KiB; this report keeps 2 KiB, because it
    prints one block per env and there are a dozen envs."""
    _bind(conda_envs=("molbuilder-siesta",))
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: f"/fake/envs/{env_name}")
    monkeypatch.setattr(_builds, "run_streaming",
                        lambda *a, **kw: (0, "x" * 10000))
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
