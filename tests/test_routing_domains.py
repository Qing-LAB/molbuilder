"""Submission domains — the named ``(partition, qos)`` menu ``submit --domain``
resolves against.

The menu is ``domains`` in ``environment.json``, because a reachable domain is
what ``sinfo``/``sacctmgr`` measured and `configuration.md` § 5 M-1 puts
measurements in the machine record.

**The record is its ONLY home** *(2026-10-02)*: a target's queues are probed on
that machine and its record copied here; the whole ``scheduler`` block is
refused by name (`configuration.md` § 4, § 5 M-1).
"""

import pytest

from conftest import write_machine_record
from molbuilder.config_dir import ensure_private_dir
from molbuilder.runtime_config import get_routing
from molbuilder.scheduler import (Domain, Environment, Topology,
                                  environments_dir, known_machines,
                                  machine_scope_path, named_environment_path,
                                  write_environment)
from molbuilder.scheduler.quantities import parse_walltime
from support.machines import SLURM
from support.road import describe_calculation, jobset

pytestmark = pytest.mark.usefixtures("in_a_folder_of_its_own")

_DOMAINS = [
    Domain(name="debug",  max_time="0-00:15:00", partition="htc", qos="debug"),
    Domain(name="htc",    max_time="0-04:00:00", partition="htc", qos="public"),
    Domain(name="public", max_time="7-00:00:00", partition="public",
           qos="public"),
]


def _this_machine_lists(domains):
    """This machine's record, where a probe writes it, listing ``domains``."""
    write_machine_record(scheduler=SLURM,
                         topology=Topology(cores_per_socket=64),
                         domains=list(domains))


@pytest.fixture
def carried(tmp_path, monkeypatch):
    """A calculation prepared on a machine listing `debug`, `htc` and
    `public`, which keeps its copy of that record (`configuration.md` M-3);
    the machine's own record then lists `elsewhere` -- the folder as it reads
    when carried to another machine, or after a new probe."""
    _this_machine_lists(_DOMAINS)
    bundle = describe_calculation(tmp_path, monkeypatch)
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this", "--no-sbatch")
    assert r.exit_code == 0, r.output
    _this_machine_lists([Domain(name="elsewhere", partition="p", qos="q")])
    return bundle


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

def test_the_calculations_record_wins_over_the_machines(carried):
    """M-3, at the reader a person actually feels it through.

    A folder carried to a cluster reads the record `prep` snapshotted beside
    it, not this machine's -- which answers a question that names no folder.
    """
    out = get_routing(project_dir=carried)
    assert [d.name for d in out] == ["debug", "htc", "public"]
    assert out[1].partition == "htc" and out[1].qos == "public"
    assert out[0].max_time == "0-00:15:00"
    assert [d.name for d in get_routing()] == ["elsewhere"]


def test_get_routing_is_empty_where_no_queue_can_be_named(tmp_path):
    """M-2: one shape for both machines.  A workstation record -- the one
    this suite's machine holds -- is a valid record that simply has no
    domains, not a missing file and not an error; and with no record at all
    the menu is empty too."""
    assert get_routing(project_dir=tmp_path) == []
    machine_scope_path().unlink()
    assert get_routing(project_dir=tmp_path) == []


def test_an_unknown_column_survives_the_type():
    """R10 as a property of the TYPE, not of one branch.

    ``Domain.extra`` carries what this reader does not check, so a column an
    operator drafts is indistinguishable from one it was built to know --
    which is the opposite of the 2026-08-12 defect, where drafting a column
    and not writing one looked the same.
    """
    row = {"name": "x", "partition": "p", "qos": "q", "invented_by_hand": 7}
    _this_machine_lists([Domain.from_row(row)])
    got = get_routing()[0]
    assert got.extra["invented_by_hand"] == 7
    # ...and it is in `extra`, NOT promoted to a field: a drafted
    # column must stay distinguishable from a declared one, which is
    # exactly what R2 relies on to say the fit is total.
    assert not hasattr(got, "invented_by_hand")


# ---- where a value came from is displayed, not inferred --------------- #


def test_provenance_shows_which_record_supplied_the_domains(carried):
    """`config_provenance` exists to answer *"where did that setting come
    from?"*.  When the domains moved to ``environment.json`` it kept reporting
    the old home, so a correctly-probed cluster displayed "(none)"."""
    from molbuilder.runtime_config import config_provenance, format_provenance
    prov = config_provenance(project_dir=carried)
    assert prov["domains"] == ["debug", "htc", "public"], \
        "provenance must follow the domains to the record that won"

    env_rows = [s for s in prov["sources"] if s["scope"] == "environment"]
    assert [s["via"] for s in env_rows] == ["calculation", "machine"]
    assert all(s["found"] for s in env_rows)
    # the calculation's record is listed FIRST, which is the order it wins in
    assert env_rows[0]["path"] == str(carried / "environment.json")
    assert str(machine_scope_path().parent) in format_provenance(prov)


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
    the fit says a queue holds no GPU it may well carry."""
    for column in (None, "gpu:a100:4"):
        row = Domain(name="g", partition="general", qos="public", gpu=column)
        assert row.devices == (), column
    # ...and an unreadable COUNT still names the device it saw
    device = Domain(name="g", partition="general", qos="public",
                    gpu={"a100": "many"}).devices[0]
    assert (device.type, device.per_node) == ("a100", None)


def test_the_users_own_spelling_survives_the_round_trip():
    """`devices` INTERPRETS the column; it never rewrites it -- the record
    keeps the words the probe wrote, every type in the order written."""
    written = {"a100": 4, "a100.20gb": 16}
    row = Domain.from_row({"name": "g", "partition": "general",
                           "qos": "public", "gpu": written})
    assert [(d.type, d.per_node) for d in row.devices] == [
        ("a100", 4), ("a100.20gb", 16)]
    assert row.to_row()["gpu"] == written


@pytest.mark.parametrize("column, said", [("max_tme", True),
                                          ("max_time", False)],
                         ids=["misspelled", "understood"])
def test_a_column_the_reader_does_not_understand_is_SAID(column, said):
    """A misspelling must stop being invisible -- without being refused, and
    without crying wolf on a clean record.

    An unrecognised column in a routing row is KEPT (R10, and
    `test_an_unknown_column_survives_the_type` above pins it: a retired key
    a record still carries, or a column of an operator's own, must
    survive).  That makes a TYPO indistinguishable from a deliberate extra:
    `max_tme` lands in `extra` exactly as `node_type` does, `max_time` then
    reads as unstated, and a job asking more wall than the queue allows is
    never said to be over it (R4).  `scheduler.md` § 4 called it *"a bug waiting for someone
    to misspell it"*.

    Neither refusing nor dropping is available, so the machine list SAYS SO.
    Both readers get it from one place -- `jobset machines` prints this
    summary and `GET /api/task-setup/machines` serves it -- because the
    terminal and the browser must not be able to disagree about a machine.

    MUTATION THIS MUST FAIL AGAINST: drop the `uninterpreted` bits from the
    summary.  The record still parses, every other fact still prints, and the
    typo is silent again.
    """
    ensure_private_dir(environments_dir())
    write_environment(
        Environment(
            scheduler=SLURM, topology=Topology(sockets=2, cores_per_socket=24),
            domains=[Domain.from_row({"name": "gpu", "partition": "gpu",
                                      "qos": "public",
                                      column: "1-00:00:00"})]),
        named_environment_path("sol"))
    summary = next(m for m in known_machines() if m["name"] == "sol")["summary"]
    # The marker is what makes it catch the eye in a line of ordinary facts
    # (user, 2026-09-06).
    assert ("??" in summary) is said, summary
    assert (column in summary) is said, summary
