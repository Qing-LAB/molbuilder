"""Does this env exist, and where?  ONE reading answers both.

`docs/ops/env-framework.md` § 2 (the `EnvState` machine) and the handover's Z3.
Three readers of one registry document used to give three answers:

  * `diagnostics._list_conda_envs` kept only envs whose parent directory was
    literally called ``envs`` -- so an env created with ``--prefix`` elsewhere
    was invisible, and `caps.env_available` said NO about an env
    `probe_env_state` reported PRESENT.  `repair` then said *"env does not
    exist.  Install it first"* about a healthy env.
  * `install._env_prefix` resolved four ways, paying a 1.2 s registry read per
    recipe -- five of them per `doctor` report.
  * `probe_env_state` parsed the same document a third way.

And the CLI probed the env, printed the state, then `run_install` probed it
again one call later: the same two JSON documents, twice in a row.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from molbuilder import diagnostics as D
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install as I

#: What molbuilder-pySCF's verify step requires of its own output.  Every recipe
#: carries such a substring (env-framework.md 5.2), so a dispatch stub returning
#: "ok" fails the install for a reason that has nothing to do with the test.
_VERIFY_OK = ("pyscf 2.13, geometric 1.1, prop: polarizability OK\n"
              "  IR: analytic dmu/dR available (pyscf.prop.infrared)")


def _registry(tmp_path, prefixes, *, base=None, details=True):
    """A manager binary whose `env list --json` reports `prefixes`.

    Shaped like a real one: `envs` plus `envs_details`, where the manager gives
    each env the NAME it knows it by and flags its base installation.  Pass
    ``details=False`` for a manager that reports only `envs` -- mamba and
    micromamba may -- which is the conservative path.
    """
    p = tmp_path / "mgr"
    payload = {"envs": [str(x) for x in prefixes]}
    if details:
        payload["envs_details"] = {
            str(x): {"name": ("base" if str(x) == str(base)
                              else os.path.basename(str(x))),
                     "base": str(x) == str(base)}
            for x in prefixes
        }
    body = json.dumps(payload)
    p.write_text("#!/bin/sh\nif [ \"$2\" = list ]; then cat <<'EOJ'\n"
                 + body + "\nEOJ\nelse echo '{}'; fi\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


# --------------------------------------------------------------------------- #
#  One reader                                                                 #
# --------------------------------------------------------------------------- #

def test_the_registry_answers_with_prefixes_and_drops_the_installation_root(
        tmp_path):
    root = tmp_path / "miniconda3"
    mgr = _registry(tmp_path, [root,
                               root / "envs" / "molbuilder",
                               root / "envs" / "notebook"], base=root)

    got = D.conda_env_prefixes(mgr)

    assert got == {"molbuilder": str(root / "envs" / "molbuilder"),
                   "notebook": str(root / "envs" / "notebook")}, got
    assert "miniconda3" not in got, (
        "the installation root is listed by the registry and is not an env")


def test_an_env_outside_envs_dirs_is_VISIBLE(tmp_path):
    """The H2 half that had a consequence.  The old reader filtered on the
    parent directory being named `envs`, which drops the installation root (the
    point) and every `--prefix` env (not the point).  Since Phase 1a an env is
    addressed by its prefix, so such an env is perfectly usable -- it was only
    ever unreachable by `-n`, and nothing addresses by name any more."""
    root = tmp_path / "miniconda3"
    out_of_tree = tmp_path / "scratch" / "mb-gpu"
    mgr = _registry(tmp_path, [root, root / "envs" / "molbuilder", out_of_tree],
                    base=root)

    got = D.conda_env_prefixes(mgr)

    assert got.get("mb-gpu") == str(out_of_tree), got


def test_the_gate_and_the_state_machine_now_agree(tmp_path, monkeypatch):
    """The disagreement, as an operator met it: `doctor` said MISSING and
    `install` resolved a prefix for the same env, because the gate and the probe
    read the registry differently."""
    root = tmp_path / "miniconda3"
    out_of_tree = tmp_path / "scratch" / "mb-gpu"
    (out_of_tree / "conda-meta").mkdir(parents=True)
    mgr = _registry(tmp_path, [root, out_of_tree], base=root)

    set_capabilities(D.Capabilities(runtime_config={}, conda_binary=mgr,
                                   conda_envs=D.conda_env_prefixes(mgr)))
    caps = D.get_capabilities()
    state = I.probe_env_state("mb-gpu", mgr)

    assert caps.env_available("mb-gpu") is True, (
        "the gate still cannot see an env the state machine calls PRESENT")
    assert state.state is I.EnvPresence.PRESENT, state.describe()
    assert caps.env_prefix("mb-gpu") == state.prefix == str(out_of_tree)


def test_a_failed_registry_read_is_an_empty_answer_not_an_exception(tmp_path):
    """Every caller asks by membership, so "no envs" has to be sayable."""
    broken = tmp_path / "mgr"
    broken.write_text("#!/bin/sh\nexit 3\n")
    broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
    assert D.conda_env_prefixes(str(broken)) == {}
    assert D.conda_env_prefixes(str(tmp_path / "not-a-file")) == {}


# --------------------------------------------------------------------------- #
#  The reading already taken is not paid for again                            #
# --------------------------------------------------------------------------- #

def test_env_prefix_uses_the_snapshot_and_runs_no_subprocess(monkeypatch):
    """`doctor` asks this once per recipe and the answer was already on the
    snapshot.  A registry read costs 1.2 s warm, so five recipes paid six
    seconds to re-learn what startup had just read."""
    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/fake/mgr",
        conda_envs={"molbuilder-pySCF": "/opt/envs/molbuilder-pySCF"}))

    def _no(*a, **k):
        raise AssertionError("a subprocess was launched for a known env")

    monkeypatch.setattr(I.subprocess, "run", _no)

    assert I._env_prefix("molbuilder-pySCF", "/fake/mgr") \
        == "/opt/envs/molbuilder-pySCF"


def test_an_env_the_snapshot_never_saw_is_still_looked_for(tmp_path):
    """A snapshot that does not know an env is not evidence of absence: the
    env may have been created a second ago, by this very install."""
    root = tmp_path / "miniconda3"
    fresh = root / "envs" / "just-made"
    fresh.mkdir(parents=True)
    mgr = _registry(tmp_path, [root, fresh], base=root)
    set_capabilities(Capabilities(runtime_config={}, conda_binary=mgr,
                                 conda_envs={}))

    assert I._env_prefix("just-made", mgr) == str(fresh)


def test_the_install_probes_this_env_once_not_three_times(monkeypatch, tmp_path):
    """H6.  The CLI probed, printed `state.describe()`, and then `run_install`
    probed again one call later -- instrumented at three reads of the same two
    JSON documents per install.  The reading is handed over instead."""
    from molbuilder.envs import _cli

    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/fake/mgr",
        conda_envs={"molbuilder-pySCF": str(tmp_path / "prefix")}))
    probes: list = []
    real_probe = I.probe_env_state

    def counting(name, binary):
        probes.append(name)
        return I.EnvState(name=name, listed_in_registry=True, dir_exists=True,
                          has_conda_meta=True, prefix=str(tmp_path / "prefix"),
                          manager=binary)

    monkeypatch.setattr(_cli._install, "probe_env_state", counting)
    monkeypatch.setattr(I, "probe_env_state", counting)
    ran: list = []
    # The verify step has its own accept rule (`expect_contains`), so a stub
    # returning "ok" makes the install FAIL for a reason this test is not about.
    monkeypatch.setattr(I._builds, "dispatch_into_env",
                        lambda argv, prefix, **kw: (ran.append(tuple(argv)),
                                                    (0, _VERIFY_OK))[1])
    monkeypatch.setattr(_cli._install, "_env_prefix",
                        lambda name, binary: str(tmp_path / "prefix"))

    from click.testing import CliRunner
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(
        _cli.envs_group, ["install", "molbuilder-pySCF", "--yes"])

    assert result.exit_code == 0, result.output
    assert len(probes) == 1, (
        f"the env was probed {len(probes)} times for one install: {probes}\n"
        f"{result.output}")
    assert any("create" in " ".join(a) for a in ran) is False, (
        "create was dispatched for an env the probe called PRESENT")


# RETIRED 2026-09-13: `test_after_clean_the_stale_reading_is_DROPPED_and_
# create_still_runs` stood here.  It asserted the same fact as
# `test_envs_install.py::test_clean_wipes_the_env_and_then_creates_it_again` --
# that a wipe is followed by a real `conda create` -- but through three
# monkeypatched internals, and with `conda env remove` behind them.  The
# replacement drives the same path against a manager that writes down what it
# was asked to do, with nothing patched, so it cannot remove anything even when
# the code is wrong.  Two tests for one fact, and this was the weaker one.


def test_a_manager_that_reports_no_details_hides_nothing(tmp_path):
    """mamba and micromamba may report only `envs`.  Nothing in that document
    says which prefix is an installation, so NOTHING is excluded -- and that is
    the safer of two wrong answers, measured rather than argued.

    The first attempt kept the old rule there ("only prefixes whose parent is
    called `envs`"), and three tests of the state machine failed at once: an env
    the registry lists but this map omits reads as **FRESH** to
    `probe_env_state`, whereupon ``conda create -n <name>`` makes a SECOND env
    beside the real one, under `envs_dirs`, while the real one sits untouched
    somewhere else.  An installation root listed under its directory's basename
    is by contrast a name nothing ever asks about: the gates ask about
    `molbuilder-*`.
    """
    root = tmp_path / "miniconda3"
    out_of_tree = tmp_path / "scratch" / "mb-gpu"
    mgr = _registry(tmp_path, [root, root / "envs" / "molbuilder", out_of_tree],
                    details=False)

    got = D.conda_env_prefixes(mgr)

    assert got == {"miniconda3": str(root),
                   "molbuilder": str(root / "envs" / "molbuilder"),
                   "mb-gpu": str(out_of_tree)}, got


def test_the_probe_sees_an_out_of_tree_env_on_EITHER_kind_of_manager(tmp_path):
    """The property the reader exists to protect, through the state machine: an
    env the registry lists is never reported FRESH, whatever the manager says
    about its own base.  FRESH is the one answer that leads to a duplicate env.
    """
    root = tmp_path / "miniconda3"
    out_of_tree = tmp_path / "scratch" / "mb-gpu"
    (out_of_tree / "conda-meta").mkdir(parents=True)

    for details in (True, False):
        mgr = _registry(tmp_path, [root, out_of_tree],
                        base=root, details=details)
        state = I.probe_env_state("mb-gpu", mgr)
        assert state.state is I.EnvPresence.PRESENT, (
            f"envs_details={details}: {state.describe()}")
        assert state.prefix == str(out_of_tree)
