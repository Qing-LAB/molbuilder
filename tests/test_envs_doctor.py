"""L1 tests for ``molbuilder.envs.doctor``.

Pins the accept rule, what `repair` dispatches, and what the audit reads of a
package's source.  The present/missing column is read on what this machine's
own manager lists, a field test (`tests/field/test_the_manager_answers.py`).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import doctor, install as _install
from molbuilder.envs.recipes import PipPackage, Recipe, recipe_by_name


def _bind(*, conda_envs=(), conda_binary="/usr/bin/conda"):
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary=conda_binary,
        conda_envs=frozenset(conda_envs),
    ))


# --------------------------------------------------------------------- #
#  Verify semantics                                                      #
# --------------------------------------------------------------------- #
#
# The accept rule -- is this output a pass? -- lives in ONE place,
# `InstallStep.accepts`, and both `install` and `doctor` ask it.  So it is
# tested ONCE, here, as a table.
#
# What still needs its own test is the WIRING: that doctor's verdict
# really comes from the rule rather than from a local re-derivation.  One
# case proves that, and it has to be the case where the rule and the
# naive "rc == 0" answer DISAGREE -- otherwise a doctor that ignored the
# rule entirely would still pass.


@pytest.mark.parametrize("expect,rc,output,accepted", [
    # The exit code is always part of the verdict.
    (None, 0, "anything", True),
    (None, 1, "anything", False),
    # An expected substring is an ADDITIONAL requirement, so a command
    # that exits 0 while the thing we asked about is absent is a failure.
    ("siesta", 0, "siesta 5.4.2", True),
    ("siesta", 0, "unrelated text", False),
    ("siesta", 1, "siesta 5.4.2", False),
    # A process that did not run is never a pass.
    ("siesta", None, "siesta 5.4.2", False),
    (None, None, "", False),
])
def test_accept_rule(expect, rc, output, accepted):
    step = _install.InstallStep(
        label="verify", argv=("true",), expect_contains=expect,
    )
    assert step.accepts(rc, output) is accepted


def test_doctor_verify_uses_the_accept_rule(monkeypatch, tmp_path):
    """A recipe that expects a banner: a command that exits 0 WITHOUT it is a
    FAIL.  A doctor that decided on the exit code itself reports True here.
    The manager is a real process that exits 0 and prints nothing, so no
    answer is typed."""
    mgr = tmp_path / "mgr"
    mgr.write_text("#!/bin/sh\nexit 0\n")
    mgr.chmod(0o755)
    _bind(conda_envs=("synthetic",), conda_binary=str(mgr))
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: str(tmp_path))
    recipe = Recipe(name="synthetic", category=None, description="d",
                    channels=("conda-forge",), conda_packages=("python",),
                    verify_argv=("true",), verify_expect_contains="LEaP")
    (report,) = doctor.report_all(recipes=(recipe,))
    assert report.verify_ok is False


# --------------------------------------------------------------------- #
#  repair: an issue says what is WRONG, the RECORD says what to do         #
# --------------------------------------------------------------------- #


def test_repair_runs_the_step_the_installer_would(monkeypatch):
    """`repair` must map each issue back to its record and dispatch the
    installer's own step -- not a command of its own.

    This is the whole of the contract's section 5, and it had no test:
    replacing `pip_step_for(pkg, ...)` with a plain
    `pip install <name>` passed the entire suite.  That regression is
    invisible AND useless -- pip treats the bare name as already
    satisfied, so repair would report OK and the re-audit would
    immediately re-report the same issue.
    """
    from click.testing import CliRunner
    from molbuilder.envs import _cli

    recipe = recipe_by_name("molbuilder-pySCF")
    pkg = next(p for p in recipe.pip_packages
               if p.name == "pyscf-properties")
    _bind(conda_envs=(recipe.name,))
    monkeypatch.setattr(_install, "_env_prefix",
                        lambda env_name, conda_binary: "/fake/envs/x")

    # The audit says: present, but from the wrong tree.
    issue = doctor.PackageAuditIssue(
        kind="pip-source", name=pkg.name, spec=pkg.spec(),
        found="(default index)", reason=pkg.reason)
    monkeypatch.setattr(doctor, "audit_packages",
                        lambda prefix, rec: doctor.PackageAudit(
                            checked=True, n_conda_declared=0,
                            n_pip_declared=1, issues=(issue,)))

    dispatched = []

    def fake_run_step(step, **kw):
        dispatched.append(step)
        return step.__class__(label=step.label, argv=step.argv,
                              returncode=0, output="ok",
                              outcome=_install.Outcome.OK)
    monkeypatch.setattr(_install, "run_step", fake_run_step)

    CliRunner().invoke(_cli.cmd_repair, ["molbuilder-pySCF"])

    assert len(dispatched) == 1, "exactly one step should be dispatched"
    argv = dispatched[0].argv
    # It is the RECORD's step: the source, and the force flags that ride
    # with it because an installed version cannot prove it is that build.
    assert any("git+https://github.com/pyscf/properties.git" in a
               for a in argv), (
        f"repair dispatched a command that does not name the recorded "
        f"source: {argv}")
    assert "--force-reinstall" in argv
    assert "--no-deps" in argv
    # ...and it still carries the declared alternative.
    assert dispatched[0].fallbacks, (
        "the step must keep the index fallback the record declares")


# --------------------------------------------------------------------- #
#  Provenance: which TREE is installed, when the version cannot say      #
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def widget(tmp_path_factory):
    """A small package in a git repository, installed by real pip both ways:
    from the repository, and from a folder standing for an index.  What the
    audit reads is what pip wrote."""
    pytest.importorskip("setuptools", reason="building a package needs it, "
                                             "with no network to fetch it")
    root = tmp_path_factory.mktemp("widget")
    env = {**os.environ, "HOME": str(root), "PIP_NO_CACHE_DIR": "1",
           "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}

    def run(*argv, cwd=None):
        done = subprocess.run(argv, cwd=cwd, env=env, text=True,
                              capture_output=True)
        assert done.returncode == 0, (argv, done.stdout, done.stderr)
        return done.stdout.strip()

    def pip(*argv):
        return run(sys.executable, "-m", "pip", "--no-input", *argv,
                   "--no-deps", "--no-index")

    repo = root / "repo"
    (repo / "widget").mkdir(parents=True)
    (repo / "widget" / "__init__.py").write_text("")
    (repo / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "widget"\nversion = "0.1.0"\n')
    git = ("git", "-c", "user.name=t", "-c", "user.email=t@example.invalid")
    run(*git, "init", "-q", "-b", "main", cwd=repo)
    run(*git, "add", ".", cwd=repo)
    run(*git, "commit", "-qm", "0.1.0", cwd=repo)
    source = f"git+file://{repo}"
    from_repository = root / "from-repository"
    pip("install", "--no-build-isolation", "--prefix", str(from_repository),
        source)
    wheels = root / "wheels"
    pip("wheel", "--no-build-isolation", "-w", str(wheels), str(repo))
    from_index = root / "from-index"
    pip("install", "--find-links", str(wheels), "--prefix", str(from_index),
        "widget")
    return {"source": source, "commit": run(*git, "rev-parse", "HEAD", cwd=repo),
            "from_repository": from_repository, "from_index": from_index}


def _a_recipe_of(package):
    return Recipe(name="a-recipe", category=None, description="d",
                  channels=("conda-forge",), conda_packages=("python",),
                  pip_packages=(package,))


@pytest.mark.parametrize("declared, expect", [
    ("{source}", None),
    # A branch cannot be checked against a commit, so it is no mismatch...
    ("{source}@main", None),
    # ...and a pinned SHA is checked, because that is what a pin is for.
    ("{source}@{commit}", None),
    ("{source}@{commit:.7}", None),
    ("{source}@deadbee", "pip-source"),
    # Same name, same version, another tree: only provenance can tell.
    ("git+file:///elsewhere/repo", "pip-source"),
    # No source declared -> provenance is not this recipe's business.
    (None, None),
])
def test_audit_detects_which_tree_is_installed(widget, declared, expect):
    """A package declared with a source is audited on PROVENANCE.

    ``pyscf-properties`` declares version ``0.1.0`` both on PyPI and on
    master, so a version comparison passes on an env that lacks the
    feature.  The audit reads PEP 610 ``direct_url.json`` instead, as pip wrote
    it for the install from the repository.
    """
    source = declared.format(**widget) if declared else None
    # This prefix has no conda-meta, so the declared conda package audits as
    # missing.  This test is about pip provenance.
    kinds = [i.kind for i in doctor.audit_packages(
                 widget["from_repository"],
                 _a_recipe_of(PipPackage("widget", source=source))).issues
             if i.kind.startswith("pip-")]
    assert kinds == ([expect] if expect else [])


def test_a_source_mismatch_the_recipe_accepts_is_only_informational(widget):
    """``fallback_to_index`` means the indexed build is a declared
    acceptable outcome, so landing on it degrades a capability rather
    than breaking the env -- and ``doctor`` must not call it FAILED."""
    recipe = _a_recipe_of(PipPackage("widget", source=widget["source"],
                                     fallback_to_index=True,
                                     reason="why this source is unusual"))
    (issue,) = [i for i in doctor.audit_packages(widget["from_index"],
                                                 recipe).issues
                if i.kind.startswith("pip-")]
    assert issue.kind == "pip-source-optional"
    assert issue.found == "(default index)"
    # repair must be handed the SOURCE and the flags, not a bare name --
    # a plain install of a force-flagged package exits 0 and changes
    # nothing, because the versions are identical.
    assert issue.spec.endswith(widget["source"])
    assert issue.reason == "why this source is unusual"
