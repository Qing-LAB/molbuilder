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

What a real manager answers is read from it, on the target:
`tests/field/test_the_manager_answers.py`.
"""
from __future__ import annotations

import stat

from molbuilder import diagnostics as D
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.envs import install as I

#: What molbuilder-pySCF's verify step requires of its own output.  Every recipe
#: carries such a substring (env-framework.md 5.2), so a dispatch stub returning
#: "ok" fails the install for a reason that has nothing to do with the test.
_VERIFY_OK = ("pyscf 2.13, geometric 1.1, prop: polarizability OK\n"
              "  IR: analytic dmu/dR available (pyscf.prop.infrared)")


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

    monkeypatch.setattr(D.subprocess, "run", _no)

    assert I._env_prefix("molbuilder-pySCF", "/fake/mgr") \
        == "/opt/envs/molbuilder-pySCF"


def test_the_install_probes_this_env_once_not_three_times(monkeypatch, tmp_path):
    """H6.  The CLI probed, printed `state.describe()`, and then `run_install`
    probed again one call later -- instrumented at three reads of the same two
    JSON documents per install.  The reading is handed over instead."""
    from molbuilder.envs import _cli

    set_capabilities(Capabilities(
        runtime_config={}, conda_binary="/fake/mgr",
        conda_envs={"molbuilder-pySCF": str(tmp_path / "prefix")}))
    probes: list = []

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
    result = CliRunner().invoke(
        _cli.envs_group, ["install", "molbuilder-pySCF", "--yes"])

    assert result.exit_code == 0, result.output
    assert len(probes) == 1, (
        f"the env was probed {len(probes)} times for one install: {probes}\n"
        f"{result.output}")
    assert any("create" in " ".join(a) for a in ran) is False, (
        "create was dispatched for an env the probe called PRESENT")
