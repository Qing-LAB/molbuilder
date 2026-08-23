"""`jobset machines` — the answer to "did the record I copied over arrive?"

Why this file exists
====================

Preparing a calculation for another machine (`preparing-for-another-machine
.md`) is: probe THERE, copy the record HERE, `prep --target NAME`.  Step two
had no confirmation.  The browser could list the records
(`GET /api/task-setup/machines`); the terminal could not, so a user at a
shell learned whether their `scp` had landed by running `prep --target` and
reading the refusal.

Two things are pinned here:

  * the listing DESCRIBES an unreadable record rather than skipping it — a
    dropped record looks exactly like one that was never copied, which is the
    opposite of the question being asked;
  * the terminal and the browser read the SAME function, so they cannot
    disagree about which machines exist or when a choice is required.

Every test runs against an isolated ``HOME`` — the developer's own
`~/.config/molbuilder` must never decide whether these pass.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from molbuilder.jobset._cli import jobset_group


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A HOME with no machine records at all."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    d = tmp_path / ".config" / "molbuilder" / "environments"
    d.mkdir(parents=True)
    return tmp_path


def _write_record(home, name, *, valid=True):
    p = (home / ".config" / "molbuilder" / "environments" / f"{name}.json")
    if not valid:
        p.write_text("{ truncated")
        return p
    # Build a record through the real writer, then file it under the given
    # name: a record's CONTENT is a measurement and the machine it describes
    # is its FILENAME -- which is exactly why a copied-in file is discovered
    # by its stem, and why this test can stand in for an `scp`.
    from molbuilder.scheduler import (Environment, Site, Topology,
                                        write_environment)
    env = Environment(
        scheduler="slurm",
        topology=Topology(sockets=2, cores_per_socket=64),
        site=Site(partition="general", qos="public", account=None),
        domains=[],
        source={"scheduler": "flag"},
        detected_at="2026-08-22T00:00:00+00:00",
    )
    write_environment(env, p)
    return p


def _run(args):
    return CliRunner().invoke(jobset_group, args)


class TestTheListingAnswersTheCopyQuestion:

    def test_a_copied_record_is_listed_by_its_filename(self, home):
        _write_record(home, "sol")
        r = _run(["machines"])
        assert r.exit_code == 0, r.output
        assert "sol" in r.output
        # ...and the path is shown, because that path IS the copy destination
        assert str(home / ".config" / "molbuilder" / "environments") in r.output

    def test_an_unreadable_record_is_shown_and_marked(self, home):
        """Not hidden.  The user wrote it; hiding it leaves them waiting for
        something that cannot happen."""
        _write_record(home, "broken", valid=False)
        r = _run(["machines"])
        assert r.exit_code == 0, r.output
        assert "broken" in r.output
        assert "cannot be read" in r.output
        assert "probe --write --name broken" in r.output

    def test_this_machine_is_offered_too(self, home):
        r = _run(["machines"])
        assert "(this machine)" in r.output

    def test_with_no_targets_it_teaches_the_whole_workflow(self, home):
        """The empty state is where a user most needs the instructions."""
        r = _run(["machines"])
        assert "probe --write --name" in r.output
        assert "environments" in r.output      # where to copy it TO

    def test_the_local_remedy_does_not_invent_a_name(self, home):
        """This machine's record is written by a bare `probe --write`.

        The message said ``--name (this machine)`` until 2026-08-22 -- not a
        name, and not a runnable command.
        """
        r = _run(["machines"])
        assert "--name (this machine)" not in r.output


class TestTheTerminalAndTheBrowserCannotDisagree:

    def test_both_read_the_same_function(self, home):
        """`GET /api/task-setup/machines` and this verb serve one list.

        Not "produce equal output" -- the same call, so a change to the rule
        reaches both or neither.
        """
        pytest.importorskip("flask")
        _write_record(home, "sol")
        from molbuilder.scheduler import known_machines, choice_required
        from molbuilder.web.app import create_app

        served = create_app(config={}).test_client() \
            .get("/api/task-setup/machines").get_json()
        direct = known_machines()
        assert [m["name"] for m in served["machines"]] == \
               [m["name"] for m in direct]
        assert served["choice_required"] is choice_required(direct)

    def test_a_named_record_makes_the_choice_required(self, home):
        """Rule: "this machine" is always a candidate, so ANY named record
        makes the question real (`preparing-for-another-machine.md` § 4, C1)."""
        from molbuilder.scheduler import choice_required
        assert choice_required() is False
        _write_record(home, "sol")
        assert choice_required() is True
        assert "--target" in _run(["machines"]).output


class TestTheHelpNamesOnlyLiveVerbs:

    def test_jobset_help_names_live_verbs(self):
        """The group help said ``describe`` for days after the verb became
        ``init`` -- so `--help` recommended a command the CLI rejects.

        Any ``verb`` in the group's help text that looks like one of ours must
        resolve to a registered command.
        """
        import re
        registered = set(jobset_group.commands)
        help_text = jobset_group.__doc__ or ""
        cited = set(re.findall(r"``([a-z][a-z-]+)``", help_text))
        # only judge words that are plausibly verbs of THIS group
        suspects = {c for c in cited if "." not in c and "-" not in c}
        unknown = {c for c in suspects
                   if c not in registered and c not in {"prep", "job", "set"}}
        assert not unknown, (
            f"`jobset --help` names {sorted(unknown)}, which are not "
            f"registered commands ({sorted(registered)}).  A help text that "
            f"recommends a nonexistent verb is worse than none.")
