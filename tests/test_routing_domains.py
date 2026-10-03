"""Submission domains — the named ``(partition, qos)`` menu ``submit --domain``
resolves against.

Renamed from ``test_bench_routing.py`` on 2026-08-17: the `bench` command group
is gone, and routing was never a benchmark concern.

**The menu moved house the same day** (N4). It was ``scheduler.routing`` in the
person's ``molbuilder.json``; it is now ``domains`` in ``environment.json``,
because a reachable domain is what ``sinfo``/``sacctmgr`` measured and
`configuration.md` § 5 M-1 puts measurements in the machine record.

**And the record is its ONLY home** *(2026-10-02)*: a declared
``scheduler.routing`` stood in as a fallback when nothing was probed, and it
described a machine its author was not on.  A target's queues are probed on
that machine and its record copied here; the whole ``scheduler`` block is
refused by name (`configuration.md` § 4, § 5 M-1).
"""

import json
from pathlib import Path

import pytest

from molbuilder.scheduler import (FILENAME, Domain, Environment, Site,
                                    Topology, write_environment)
from molbuilder.runtime_config import (RuntimeConfigError, get_routing,
                                       read_config)
from molbuilder.scheduler.quantities import parse_walltime

_DOMAINS = [
    Domain(name="debug",  max_time="0-00:15:00", partition="htc", qos="debug"),
    Domain(name="htc",    max_time="0-04:00:00", partition="htc", qos="public"),
    Domain(name="public", max_time="7-00:00:00", partition="public",
           qos="public"),
]


def _write_record(where, domains=_DOMAINS, gpu_type=None):
    return write_environment(
        Environment(scheduler="slurm",
                    topology=Topology(cores_per_socket=64, gpu_type=gpu_type),
                    site=Site(partition="public"),
                    domains=list(domains)),
        Path(where) / FILENAME)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Isolated cwd + $HOME + XDG.

    Both readers below consult the CWD-first server-wide scope and the per-user
    machine scope, so without this the verdicts depend on the developer's own
    ``molbuilder.json`` — caught 2026-08-12 the moment that file gained a real
    ``scheduler.routing``, and again on 2026-08-17 when N4 made that key an
    error and thirteen tests in OTHER files failed for that reason alone.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()


# ---- parse_walltime --------------------------------------------------- #

@pytest.mark.parametrize("s,secs", [
    ("0-00:15:00", 900),
    ("0-04:00:00", 4 * 3600),
    ("7-00:00:00", 7 * 24 * 3600),
    ("04:00:00", 4 * 3600),     # HH:MM:SS
    ("30", 1800),               # bare = minutes (SLURM)
    ("2-12", (2 * 24 + 12) * 3600),
    ("", 0),
])
def test_parse_walltime(s, secs):
    assert parse_walltime(s) == secs


def test_parse_walltime_garbage_raises():
    with pytest.raises(ValueError):
        parse_walltime("soon")


# ---- the menu now comes from the machine record ----------------------- #

def test_get_routing_reads_the_calculations_record(tmp_path):
    _write_record(tmp_path)
    out = get_routing(project_dir=tmp_path)
    assert [d.name for d in out] == ["debug", "htc", "public"]
    assert out[1].partition == "htc" and out[1].qos == "public"
    assert out[0].max_time == "0-00:15:00"


def test_get_routing_is_empty_without_a_record(tmp_path):
    assert get_routing(project_dir=tmp_path) == []


def test_get_routing_is_empty_on_a_workstation(tmp_path):
    """M-2: one shape for both machines.  A workstation record is a valid
    record that simply has no domains — not a missing file, and not an error."""
    write_environment(Environment(scheduler="workstation",
                                  topology=Topology(cores_per_socket=10)),
                      tmp_path / FILENAME)
    assert get_routing(project_dir=tmp_path) == []


def test_the_calculations_record_wins_over_the_machines(tmp_path):
    """M-3, at the reader a person actually feels it through.

    A folder carried to a cluster reads the record `prep` snapshotted beside
    it, not this machine's.
    """
    machine = tmp_path / "home" / ".config" / "molbuilder"
    machine.mkdir(parents=True)
    _write_record(machine, [Domain(name="elsewhere", partition="p", qos="q")])
    _write_record(tmp_path)
    assert [d.name for d in get_routing(project_dir=tmp_path)] == \
        ["debug", "htc", "public"]


def test_an_unknown_column_survives_the_type(tmp_path):
    """R10 as a property of the TYPE, not of one branch.

    ``Domain.extra`` carries what this reader does not check, so a column an
    operator drafts is indistinguishable from one it was built to know --
    which is the opposite of the 2026-08-12 defect, where drafting a column
    and not writing one looked the same.
    """
    row = {"name": "x", "partition": "p", "qos": "q", "invented_by_hand": 7}
    write_environment(
        Environment(scheduler="slurm", topology=Topology(),
                    domains=[Domain.from_row(row)]), tmp_path / FILENAME)
    got = get_routing(project_dir=tmp_path)[0]
    assert got.extra["invented_by_hand"] == 7
    # ...and it is in `extra`, NOT promoted to a field: a drafted
    # column must stay distinguishable from a declared one, which is
    # exactly what R2 relies on to say admission is total.
    assert not hasattr(got, "invented_by_hand")


# ---- where a value came from is displayed, not inferred --------------- #

# `test_a_refusal_names_WHICH_file_carries_the_key` retired 2026-10-02 (W54): a refusal naming its file (R10) is `test_auth_config.py`'s door test; the `scheduler` refusal a row of `tests/data/molbuilder_json.toml`.


def test_provenance_shows_which_record_supplied_the_domains(tmp_path):
    """`config_provenance` exists to answer *"where did that setting come
    from?"*.  When the domains moved to ``environment.json`` it kept reporting
    the old home, so a correctly-probed cluster displayed "(none)"."""
    from molbuilder.runtime_config import config_provenance, format_provenance
    machine = tmp_path / "home" / ".config" / "molbuilder"
    machine.mkdir(parents=True)
    _write_record(machine, [Domain(name="elsewhere", partition="p", qos="q")])
    _write_record(tmp_path)

    prov = config_provenance(project_dir=tmp_path)
    assert prov["domains"] == ["debug", "htc", "public"], \
        "provenance must follow the domains to the record that won"

    env_rows = [s for s in prov["sources"] if s["scope"] == "environment"]
    assert [s["via"] for s in env_rows] == ["calculation", "machine"]
    assert all(s["found"] for s in env_rows)
    # the calculation's record is listed FIRST, which is the order it wins in
    assert env_rows[0]["path"] == str(tmp_path / "environment.json")
    assert str(machine) in format_provenance(prov)


# ---- no card in the person's config (`scheduler.md` R2a) ----------------- #

# ---- the gpu column: two spellings, ONE reading ------------------------- #
#
# Two things write the column and neither is wrong: a probe maps gres type to
# per-node count, and a person describes one device.  What WAS wrong is that
# two call sites each read the raw column and only one understood both -- so
# the documented hand-declared row made `prep bench` refuse, naming
# ``mem_gb``/``per_node``/``type`` as GPU types.  `Domain.devices` is the one
# reading (`execution/scheduler.md` § 4, "Device").

def test_a_silent_or_unreadable_column_states_no_count():
    """R3 applies to devices: *the row does not say* is ``None``, never zero.
    A count we cannot read must not read as a domain with no devices, or
    admission refuses work the record never ruled out."""
    for column in (None, {}, "gpu:a100:4", {"a100": "many"},
                   {"type": "a100"}):
        row = Domain(name="g", partition="general", qos="public", gpu=column)
        assert all(d.per_node is None for d in row.devices), column
    # ...and the unreadable-COUNT cases still name the device they saw
    assert Domain(name="g", partition="general", qos="public",
                  gpu={"a100": "many"}).devices[0].type == "a100"


def test_the_users_own_spelling_survives_the_round_trip():
    """`devices` INTERPRETS the column; it never rewrites it.  The row stays
    the operator's to edit, in the words they wrote it in."""
    written = {"type": "a100", "per_node": 4, "mem_gb": 80}
    row = Domain.from_row({"name": "g", "partition": "general",
                           "qos": "public", "gpu": written})
    assert row.devices[0].type == "a100"
    assert row.to_row()["gpu"] == written


def test_a_column_the_reader_does_not_understand_is_SAID(tmp_path, monkeypatch):
    """A misspelling must stop being invisible -- without being refused.

    An unrecognised column in a routing row is KEPT (R10, and
    `test_an_unknown_column_survives_the_type` above pins it: a retired key
    a record still carries, or a column of an operator's own, must
    survive).  That makes a TYPO indistinguishable from a deliberate extra:
    `max_tme` lands in `extra` exactly as `node_type` does, `max_time` then
    reads as unstated, and a job asking more wall than the queue allows is
    admitted (R3).  `scheduler.md` § 4 called it *"a bug waiting for someone
    to misspell it"*.

    Neither refusing nor dropping is available, so the machine list SAYS SO.
    Both readers get it from one place -- `jobset machines` prints this
    summary and `GET /api/task-setup/machines` serves it -- because the
    terminal and the browser must not be able to disagree about a machine.

    MUTATION THIS MUST FAIL AGAINST: drop the `uninterpreted` bits from the
    summary.  The record still parses, every other fact still prints, and the
    typo is silent again.
    """
    from molbuilder.scheduler.record import (Domain, Environment, Topology,
                                             known_machines)

    cfg = tmp_path / "molbuilder"
    (cfg / "environments").mkdir(parents=True)
    env = Environment(
        scheduler="slurm", topology=Topology(sockets=2, cores_per_socket=24),
        domains=[Domain.from_row({"name": "gpu", "partition": "gpu",
                                  "qos": "public",
                                  "max_tme": "1-00:00:00"})])
    (cfg / "environments" / "sol.json").write_text(env.to_json())
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cfg))

    sol = next(m for m in known_machines() if m["name"] == "sol")
    assert "max_tme" in sol["summary"], (
        "a column the reader could not interpret is not reported at all -- "
        "a typo in max_time silently lifts the queue's wall")
    assert "??" in sol["summary"], (
        "the marker is what makes it catch the eye in a line of ordinary "
        "facts (user, 2026-09-06)")


def test_a_record_the_reader_fully_understands_says_nothing_extra(tmp_path,
                                                                  monkeypatch):
    """The other half: the notice must not cry wolf on a clean record."""
    from molbuilder.scheduler.record import (Domain, Environment, Topology,
                                             known_machines)

    cfg = tmp_path / "molbuilder"
    (cfg / "environments").mkdir(parents=True)
    env = Environment(
        scheduler="slurm", topology=Topology(sockets=2, cores_per_socket=24),
        domains=[Domain.from_row({"name": "gpu", "partition": "gpu",
                                  "qos": "public",
                                  "max_time": "1-00:00:00"})])
    (cfg / "environments" / "sol.json").write_text(env.to_json())
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cfg))

    sol = next(m for m in known_machines() if m["name"] == "sol")
    assert "??" not in sol["summary"], (
        f"a correctly spelled record was flagged: {sol['summary']!r}")
