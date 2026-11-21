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

Every test runs against conftest's isolated config root — the developer's
own `~/.config/molbuilder` must never decide whether these pass.
"""
from __future__ import annotations


import pytest
from click.testing import CliRunner

from molbuilder.jobset._cli import jobset_group
from support.machines import SLURM


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A config root with no named records -- this machine's is conftest's."""
    from molbuilder.config_dir import ensure_private_dir
    from molbuilder.scheduler import environments_dir
    ensure_private_dir(environments_dir())
    return tmp_path


def _write_record(home, name, *, valid=True, domains=()):
    from molbuilder.scheduler import named_environment_path
    p = named_environment_path(name)
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
        scheduler=SLURM,
        topology=Topology(sockets=2, cores_per_socket=64),
        site=Site(partition="general", qos="public"),
        domains=list(domains),
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
        from molbuilder.scheduler import environments_dir
        assert str(environments_dir()) in r.output

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
        """This machine's record is written by a bare `probe --write`."""
        r = _run(["machines"])
        assert "--name (this machine)" not in r.output


class TestTheTerminalAndTheBrowserCannotDisagree:

    def test_both_read_the_same_function(self, home):
        """`GET /api/task-setup/machines` and this verb serve one list.

        Not "produce equal output" -- the same call, so a change to the rule
        reaches both or neither.
        """
        _write_record(home, "sol")
        from molbuilder.scheduler import known_machines, choice_required
        from molbuilder.web.app import create_app

        served = create_app(config={}).test_client() \
            .get("/api/task-setup/machines").get_json()
        direct = known_machines()
        assert [m["name"] for m in served["machines"]] == \
               [m["name"] for m in direct]
        assert served["choice_required"] is choice_required(direct)

    def test_a_queue_is_said_by_the_machines_it_lists_else_its_largest(
            self, home):
        """`scheduler.md` § 2 and R3: a queue's machines are the machines it
        lists, each with the size it states; a record that lists no
        machines is read by its largest machine's figures -- as the fit
        reads them.  A listed machine that states no memory (a probe that
        read none for it) shows none, never the queue's largest figure,
        which the fit does not hold it to; the queue table and the machines
        listing say the same."""
        from molbuilder.scheduler import Domain, known_machines
        queues = [Domain(name="unlisted", partition="u", qos="public",
                         max_cores=128, max_mem_gb=512),
                  Domain(name="listed", partition="l", qos="public",
                         node_types=[{"cores": 48, "mem_gb": 256}]),
                  Domain(name="no-mem", partition="n", qos="public",
                         node_types=[{"cores": 64, "nodes": 4}],
                         max_mem_gb=1000)]
        _write_record(home, "sol", domains=queues)
        assert [(q.core_range, q.mem_range) for q in queues] == [
            ("128", "512 GB"), ("48", "256 GB"), ("64", "-")]
        summary = next(m["summary"] for m in known_machines()
                       if m["name"] == "sol")
        assert "48-128 cores" in summary and "256-512 GB" in summary, summary

    def test_a_queue_row_that_is_not_one_is_said_and_kept(self, home):
        """`scheduler.md` R2: *"Nothing is refused and nothing is dropped; a
        typo stops being invisible."*  A hand-edited queue row that spells
        ``qos`` as ``QoS`` is no queue -- not offered, never read as one
        with no QoS -- and the machine's line says it, while the record
        keeps the row as written when it is written again."""
        import json
        from molbuilder.scheduler import (Domain, known_machines,
                                          read_environment, write_environment)
        path = _write_record(home, "sol", domains=[
            Domain(name="debug", partition="htc", qos="debug")])
        record = json.loads(path.read_text())
        record["domains"].append({"name": "htc", "partition": "htc",
                                  "QoS": "public"})
        path.write_text(json.dumps(record))
        sol = next(m for m in known_machines() if m["name"] == "sol")
        assert "?? not understood: queue htc (no qos)" in sol["summary"], (
            sol["summary"])
        env = read_environment(path)
        assert [q.name for q in env.domains] == ["debug"]
        write_environment(env, path)
        assert {"name": "htc", "partition": "htc", "QoS": "public"} in \
            json.loads(path.read_text())["domains"]

    def test_a_named_record_makes_the_choice_required(self, home):
        """Rule: "this machine" is always a candidate, so ANY named record
        makes the question real (`preparing-for-another-machine.md` § 4, C1)."""
        from molbuilder.scheduler import choice_required
        assert choice_required() is False
        _write_record(home, "sol")
        assert choice_required() is True
        assert "--target" in _run(["machines"]).output
