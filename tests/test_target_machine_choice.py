"""Which machine a `prep` is FOR is the user's answer, never a default.

`preparing-for-another-machine.md` § 4.  Four ways the wrong machine could
be chosen; two already refused before this file existed (C2, C4) and are
pinned here so they stay refused, two did not (C1, C3).

**Every case drives the real `prep` command.** The unit-level probe that
first surfaced C2 said the target was ignored -- `resolve_target` returns
an existing snapshot before it ever reads `target` -- and running the case
showed the guard fires further along. A test that reasons about one
function would have "found" a bug that is not there.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from molbuilder.scheduler import AmbiguousTarget, UnknownTarget

_REC = {"schema": "molbuilder/environment@2", "domains": [], "topology": {},
        "site": {}, "source": {}}


@pytest.fixture
def machines(tmp_path, monkeypatch):
    """A home with named target records, and a bundle to prep."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cfg = tmp_path / "home" / ".config" / "molbuilder"
    (cfg / "environments").mkdir(parents=True)

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv("MOLBUILDER_PROJECTS", str(tree))

    def _write(name: str, body=None, *, scheduler="slurm"):
        p = cfg / "environments" / f"{name}.json"
        p.write_text(body if body is not None
                     else json.dumps({**_REC, "scheduler": scheduler}))
        return p

    def _this_machine(scheduler="workstation"):
        (cfg / "environment.json").write_text(
            json.dumps({**_REC, "scheduler": scheduler}))

    return type("M", (), {"write": staticmethod(_write),
                          "this_machine": staticmethod(_this_machine),
                          "tree": tree, "cfg": cfg})


def _prep(bundle: str, *extra):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner()
    init = r.invoke(jobset_group, [
        "init", "--structure", "P/structure/h2.xyz", "--bundle", bundle,
        "--shape", "flat", "--engine", "pyscf"])
    assert init.exit_code == 0, init.output
    return r.invoke(jobset_group,
                    ["prep", "run", "coarse", "--bundle", bundle, *extra])


class TestTheUserChoosesTheMachine:

    def test_c1_several_machines_and_no_target_is_refused(self, machines):
        """The failure `--target` exists for, one level up: the flag was
        added so a cluster prep is not measured against the desk, but with
        several records and no flag the desk was still chosen silently."""
        machines.write("sol")
        machines.write("agave")
        machines.this_machine()
        res = _prep("P/optimization/w")
        assert res.exit_code != 0, res.output
        assert "several machines could be meant" in res.output
        # It names every choice, including staying here -- AS SOMETHING A
        # PERSON CAN TYPE.  This asserted "omit --target" until 2026-08-24,
        # which is what the message used to say and was self-contradictory:
        # omitting the flag is the action that produced this refusal, so the
        # instruction it gave could not be followed.  With any named record
        # on file, preparing for the box in front of you was impossible.
        assert "--target sol" in res.output
        assert "--target agave" in res.output
        from molbuilder.scheduler.record import LOCAL_TARGET
        assert f"--target {LOCAL_TARGET}" in res.output
        assert "omit --target" not in res.output, (
            "the refusal is telling the user to do the thing that caused it")

    def test_c1_one_machine_and_no_target_still_proceeds(self, machines):
        """No ambiguity, so no question.  A refusal here would tax every
        single-machine user for a problem they do not have."""
        machines.this_machine()
        res = _prep("P/optimization/w")
        assert res.exit_code == 0, res.output

    def test_c1_asks_nothing_of_a_reprep(self, machines):
        """A calculation that already carries a snapshot HAS its answer --
        asking again would make every second `prep` need a flag."""
        machines.write("sol")
        machines.this_machine()
        first = _prep("P/optimization/w", "--target", "sol")
        assert first.exit_code == 0, first.output
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        again = CliRunner().invoke(jobset_group, [
            "prep", "run", "coarse", "--bundle", "P/optimization/w"])
        assert again.exit_code == 0, again.output

    def test_c3_a_named_record_that_will_not_read_is_an_error(self, machines):
        """`read_environment` answers absent / unreadable / wrong-schema
        with one `None` so callers have one thing to check.  For a NAMED
        target that single answer must not mean "try the next scope":
        the user said which machine, and silence hands them another."""
        machines.write("sol", '{"schema": "molbuilder/environment@99"}')
        machines.this_machine()
        res = _prep("P/optimization/x", "--target", "sol")
        assert res.exit_code != 0, res.output
        assert "cannot be read" in res.output
        assert "probe --write --name sol" in res.output

    def test_c4_an_unknown_target_names_the_known_ones(self, machines):
        """Refused before this file existed; pinned so it stays."""
        machines.write("sol")
        machines.this_machine()
        res = _prep("P/optimization/y", "--target", "nosuch")
        assert res.exit_code != 0, res.output
        assert "no machine record named 'nosuch'" in res.output
        assert "sol" in res.output

    def test_c2_a_contradicting_target_is_refused(self, machines):
        """Refused before this file existed.  Pinned because reading
        `resolve_target` alone suggests otherwise -- it returns an existing
        snapshot before consulting `target`, and the guard is further on."""
        machines.write("sol", scheduler="slurm")
        machines.write("agave", scheduler="workstation")
        machines.this_machine()
        # Prepped for one machine -- named, because two exist (C1).
        first = _prep("P/optimization/z", "--target", "agave")
        assert first.exit_code == 0, first.output
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        res = CliRunner().invoke(jobset_group, [
            "prep", "run", "coarse", "--bundle", "P/optimization/z",
            "--target", "sol"])
        assert res.exit_code != 0, res.output
        assert "does not match the machine" in res.output
        # and the snapshot is NOT quietly rewritten to the new target
        rec = json.loads(
            (machines.tree / "P" / "optimization" / "z" / "environment.json"
             ).read_text())
        assert rec["scheduler"] == "workstation"


class TestTheBootstrapIsSaidToNotTravel:

    def test_a_remote_target_warns_when_the_preamble_is_this_machines(
            self, machines):
        """§ 3: `--target` carries MEASUREMENTS.  A preamble is a
        preference, so it stays local and the wrapper would run on the
        target with lines written for here."""
        machines.write("sol")
        machines.this_machine()
        (machines.cfg / "molbuilder.json").write_text(json.dumps(
            {"script_generation": {"preamble": "source /home/local/conda.sh",
                                   "activation": "conda activate"}}))
        res = _prep("P/optimization/w", "--target", "sol")
        assert res.exit_code == 0, res.output
        assert "prepped for 'sol'" in res.output
        assert "preambles CONCATENATE" in res.output

    def test_no_warning_without_a_target(self, machines):
        """Prepping for this machine is the case the local config is FOR."""
        machines.this_machine()
        (machines.cfg / "molbuilder.json").write_text(json.dumps(
            {"script_generation": {"preamble": "source /home/local/conda.sh",
                                   "activation": "conda activate"}}))
        res = _prep("P/optimization/w")
        assert res.exit_code == 0, res.output
        assert "prepped for" not in res.output


class TestTheCasesReadingFoundThatPokingDidNot:
    """Three defects that only a full read of `machine_for` shows, because
    each needs a *combination* a passing test never reaches.
    """

    def test_c1_fires_with_named_targets_and_no_local_record(self, machines):
        """The commonest cluster setup: records for the clusters, nothing
        probed on the laptop.

        The first cut required a local record before calling it ambiguous,
        so this case fell through to a FRESH PROBE of the machine the user
        is sitting at -- silently prepping for the laptop, which is exactly
        what the refusal exists to stop.  "This machine" is always a
        candidate, so any named record makes the question real.
        """
        machines.write("sol")
        # deliberately NO machines.this_machine()
        res = _prep("P/optimization/w")
        assert res.exit_code != 0, res.output
        assert "several machines could be meant" in res.output

    def test_c3_fires_even_when_the_bundle_is_already_prepped(self, machines):
        """`record_scopes` puts the calculation's snapshot FIRST, so a check
        living inside the resolution loop never reaches the target scope on
        an already-prepped bundle -- the same flag would refuse for a fresh
        folder and stay silent for a prepped one.  The named target is
        therefore validated whole, before any scope is walked."""
        machines.write("sol")
        machines.this_machine()
        first = _prep("P/optimization/w", "--target", "sol")
        assert first.exit_code == 0, first.output
        # sol's record goes bad AFTER the bundle was prepped
        machines.write("sol", '{"schema": "molbuilder/environment@99"}')
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        res = CliRunner().invoke(jobset_group, [
            "prep", "run", "coarse", "--bundle", "P/optimization/w",
            "--target", "sol"])
        assert res.exit_code != 0, res.output
        assert "cannot be read" in res.output

    def test_an_unreadable_target_never_silently_becomes_this_machine(
            self, machines):
        """The property all of C1/C3 protect, stated once: no spelling of
        the flags may end with a machine the user did not name."""
        machines.write("sol", '{"nonsense": true}')
        machines.this_machine(scheduler="workstation")
        res = _prep("P/optimization/w", "--target", "sol")
        assert res.exit_code != 0
        bundle = machines.tree / "P" / "optimization" / "w"
        assert not (bundle / "environment.json").is_file(), (
            "a refused prep must not leave this machine's record behind")
