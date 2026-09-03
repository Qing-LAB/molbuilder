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

#: A record as `jobset probe --write` writes one.  It carries
#: `script_generation` because a probed machine states how a shell enters
#: an environment there -- a FACT about the machine (`configuration.md`
#: § 5 M-1), and since 2026-08-24 the field the generator reads.  Without
#: it `prep --target <name>` refuses rather than bake THIS machine's
#: activation into a wrapper bound for another one, so a fixture that
#: omitted it stopped every target test at that refusal instead of at the
#: guard it meant to exercise.
_REC = {"schema": "molbuilder/environment@2", "domains": [], "topology": {},
        "site": {}, "source": {},
        "script_generation": {"preamble": "module load mamba",
                              "activation": "source activate"}}


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

    def test_c1_the_BENCH_arm_refuses_in_words_not_a_traceback(self, machines):
        """The same which-machine question, asked by `prep bench` -- which
        reached the environment BEFORE the run arm's catch and leaked the
        refusal as a raw traceback (workflow.md § 9: a gate refuses with
        the reason, never a stack trace; found live 2026-08-28)."""
        machines.write("sol")
        machines.this_machine()
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        r = CliRunner()
        init = r.invoke(jobset_group, [
            "init", "--structure", "P/structure/h2.xyz",
            "--bundle", "P/optimization/wb",
            "--shape", "flat", "--engine", "pyscf"])
        assert init.exit_code == 0, init.output
        # a bench needs an axis to measure; declare one the way the tab does
        import json as _json
        tj = machines.tree / "P" / "optimization" / "wb" / "task.json"
        d = _json.loads(tj.read_text()); d["bench"] = {"threads": [1, 2]}
        tj.write_text(_json.dumps(d))
        res = r.invoke(jobset_group,
                       ["prep", "bench", "coarse",
                        "--bundle", "P/optimization/wb"])
        assert res.exit_code != 0, res.output
        assert "several machines could be meant" in res.output
        assert "--target sol" in res.output
        assert "Traceback" not in res.output, (
            "the bench arm answered a user question with a stack trace")

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


# `TestTheBootstrapIsSaidToNotTravel` was RETIRED 2026-08-25 with
# `runtime_config.bootstrap_travels`.  Its own docstring stated the rule
# `preparing-for-another-machine.md` § 3 retracted on 2026-08-24 -- *"a
# preamble is a preference, so it stays local"*.  Since that date the
# bootstrap rides the machine's probed record and `runwrap` reads the
# record and nothing else (see `_REC` above, which carries
# `script_generation` precisely because the generator now requires it).
# The warning it pinned therefore fired on every named-target prep while
# the wrapper being generated was correct.

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

    def test_naming_THIS_MACHINE_reads_the_same_records_as_naming_nothing(
            self, machines):
        """`--target this` says WHICH machine; it is not a second way of
        finding one.

        `workflow.md` § 5 step 1: *"read the record — the calculation's,
        else this machine's"*.  `LOCAL_TARGET` took a private road that read
        ONLY the machine scope, so naming the local machine DISCARDED the
        calculation's own snapshot while saying nothing kept it -- one box,
        two answers, and the explicit road was the worse one.

        It reaches everybody.  The task-setup tab can only send the label
        `(this machine)`, which the prep door translates to `LOCAL_TARGET`,
        so every local prep from the browser took that road; and `--target
        this` is what C1's own refusal above tells people to type.

        **The scenario is a bundle that has been carried.**  It was prepped
        where a machine record existed, and it is prepped again somewhere
        that has none of its own -- which is the case the snapshot exists
        for (`running-a-job.md` § 3.1: read from a record, and the local box
        is no exception).  Before the fix this refused in REMOTE words --
        *"on this, run jobset probe --write --name this, then copy the
        record into ~/.config/ here"* -- advice that means nothing for the
        box you are sitting at.
        """
        machines.this_machine(scheduler="slurm")
        first = _prep("P/optimization/w")
        assert first.exit_code == 0, first.output
        bundle = machines.tree / "P" / "optimization" / "w"
        snapshot = bundle / "environment.json"
        assert snapshot.is_file(), "step 1 wrote no snapshot to read back"

        # The machine scope goes away; only the calculation's own record is
        # left, which is exactly what a carried bundle has.
        (machines.cfg / "environment.json").unlink()

        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        res = CliRunner().invoke(jobset_group, [
            "prep", "run", "coarse", "--bundle", "P/optimization/w",
            "--target", "this"])
        assert res.exit_code == 0, (
            "naming this machine refused a bundle carrying its own record:\n"
            + res.output)

        # And it read THAT record -- not a fresh probe of the box, which
        # would be the guess § 3.1 forbids.
        from molbuilder.scheduler.record import machine_for
        named = machine_for(bundle, target="this")
        silent = machine_for(bundle, target=None)
        assert named is not None and silent is not None
        assert named.to_dict() == silent.to_dict(), (
            "naming this machine and naming nothing resolved DIFFERENT "
            "records for the same box")

    def test_no_machine_record_at_all_is_a_REFUSAL_naming_the_probe(
            self, machines):
        """**A machine that has not been probed cannot be prepped for**
        *(user, 2026-09-02: "all environments have to be explicitly probed
        and stored. no environment json, error")*.

        Step 1 used to run a fresh probe here and write the answer down.
        That reads as helpful and is the guess `running-a-job.md` § 3.1
        forbids: the numbers the wrapper then carries come from *whichever
        box happened to run prep*, which for a bundle described at a desk
        and run on a cluster is the wrong machine -- and the number looks
        exactly like a right one.

        The refusal has to carry the command, because "no machine record"
        is only actionable if you are told the one thing that makes one.
        """
        from molbuilder.scheduler import machine_scope_path

        # NOTHING is probed: no named records, and the record the suite's
        # own fixture writes for every test is removed.
        Path(machine_scope_path()).unlink(missing_ok=True)

        res = _prep("P/optimization/w")
        assert res.exit_code != 0, (
            "a bundle with no machine record anywhere was prepped anyway -- "
            "against what, then?\n" + res.output)
        assert "no machine record" in res.output, res.output
        assert "jobset probe --write" in res.output, (
            "the refusal does not name the command that fixes it: "
            + res.output)
        bundle = machines.tree / "P" / "optimization" / "w"
        assert not (bundle / "environment.json").is_file(), (
            "a refused prep left a machine record behind -- it probed after "
            "all, and the next prep would silently use it")
