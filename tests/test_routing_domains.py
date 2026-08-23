"""Submission domains — the named ``(partition, qos)`` menu ``submit --domain``
resolves against.

Renamed from ``test_bench_routing.py`` on 2026-08-17: the `bench` command group
is gone, and routing was never a benchmark concern.

**The menu moved house the same day** (N4). It was ``scheduler.routing`` in the
person's ``molbuilder.json``; it is now ``domains`` in ``environment.json``,
because a reachable domain is what ``sinfo``/``sacctmgr`` measured and
`configuration.md` § 5 M-1 puts measurements in the machine record.

**Corrected the same day**: the key is NOT an error.  You can only probe the
machine you are standing on, so a workstation describing a cluster must be
able to DECLARE its domains.  Probed wins where both exist; declared is the
fallback, and carries the operator's own columns through whole.
"""

import json
from pathlib import Path

import pytest

from molbuilder.scheduler import (FILENAME, Domain, Environment, Site,
                                    Topology, write_environment)
from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                       RuntimeConfigError, get_routing,
                                       get_scheduler)
from molbuilder.scheduler.probe import parse_walltime


def _write_config(tmp_path, scheduler_block):
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
        json.dumps({"scheduler": scheduler_block}))


_SCHED = {"kind": "slurm",
          "directives": {"partition": "public", "qos": "public"}}

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
    _write_config(tmp_path, _SCHED)
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


# ---- the old home is DECLARED capability, not an error ---------------- #

def test_declared_routing_is_read_when_nothing_was_probed(tmp_path):
    """**You can only probe the machine you are standing on.**

    N4 refused ``scheduler.routing`` outright, on the rule "domains are
    probed, not declared".  That bricked `prep` on a workstation whose config
    described a cluster -- not an edge case but the ordinary way this is used:
    describe here, run there, and the cluster cannot be probed from here.  The
    axis is fact vs preference; a fact may be declared.
    """
    _write_config(tmp_path, dict(_SCHED, routing=[
        {"name": "gpu", "partition": "general", "qos": "public",
         "max_time": "7-00:00:00"}]))
    assert [d.name for d in get_routing(project_dir=tmp_path)] == ["gpu"]
    assert get_scheduler(project_dir=tmp_path) is not None


def test_declared_rows_keep_the_operators_own_columns(tmp_path):
    """R10 (2026-08-12), which the N4 draft retired on a false premise.

    I justified dropping that guard with *"nobody hand-writes a column
    there"*.  The developer's own config hand-writes five per row --
    ``node_type``, ``max_cores``, ``max_mem_gb``, ``default_mem_per_core_gb``,
    ``gpu{}`` -- and the memory ones are values NO probe may invent, by the
    prober's own note.
    """
    _write_config(tmp_path, dict(_SCHED, routing=[
        {"name": "gpu", "partition": "general", "qos": "public",
         "max_time": "7-00:00:00", "node_type": "gpu-a100",
         "max_cores": 48, "max_mem_gb": 512,
         "gpu": {"type": "a100", "per_node": 4, "mem_gb": 80}}]))
    row = get_routing(project_dir=tmp_path)[0]
    assert row.node_type == "gpu-a100"
    assert row.max_cores == 48
    assert row.gpu == {"type": "a100", "per_node": 4, "mem_gb": 80}


def test_one_shape_whichever_source_answered(tmp_path):
    """**The same row in, the same row out** — from either source.

    `get_routing` built a 4-key mapping by hand on the probed branch and
    passed declared rows through whole on the other, so a caller got 4 keys or
    6 depending on which file answered *the same function*.  That is two
    representations of one concept, which is what `Domain.from_row` /
    `to_row` exist to remove: both branches now build the same object.
    """
    row = {"name": "gpu", "partition": "general", "qos": "public",
           "max_time": "7-00:00:00", "node_type": "gpu-a100",
           "max_cores": 48, "gpu": {"type": "a100", "mem_gb": 80}}

    write_environment(
        Environment(scheduler="slurm", topology=Topology(),
                    domains=[Domain.from_row(row)]), tmp_path / FILENAME)
    probed = get_routing(project_dir=tmp_path)

    (tmp_path / FILENAME).unlink()
    _write_config(tmp_path, dict(_SCHED, routing=[row]))
    declared = get_routing(project_dir=tmp_path)

    # Typed since phase 3 (2026-08-23): `get_routing` used to flatten its
    # Domains back to dicts on the way out, which is what let a caller reach
    # for a key nothing declared.  The claim is unchanged and now stronger --
    # one SHAPE means one TYPE, compared as objects rather than as mappings
    # that happen to match.
    assert probed == declared, (
        "one function must not return two shapes:\n"
        f"  probed  : {probed}\n  declared: {declared}")
    assert probed == [Domain.from_row(row)]
    assert all(isinstance(d, Domain) for d in probed)


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


def test_a_probed_record_beats_a_declared_one(tmp_path):
    """Standing on the machine beats a hand-written note about it."""
    _write_config(tmp_path, dict(_SCHED, routing=[
        {"name": "declared", "partition": "p", "qos": "q",
         "max_time": "1-00:00:00"}]))
    _write_record(tmp_path)
    assert [d.name for d in get_routing(project_dir=tmp_path)] == \
        ["debug", "htc", "public"]


def test_a_workstation_record_does_not_mask_a_declared_cluster(tmp_path):
    """The case that broke: this box is a workstation (no domains), the config
    describes a cluster.  An empty probed list must fall THROUGH to the
    declaration rather than shadow it."""
    write_environment(Environment(scheduler="workstation",
                                  topology=Topology(cores_per_socket=10)),
                      tmp_path / FILENAME)
    _write_config(tmp_path, dict(_SCHED, routing=[
        {"name": "sol-gpu", "partition": "general", "qos": "public",
         "max_time": "7-00:00:00"}]))
    assert [d.name for d in get_routing(project_dir=tmp_path)] == ["sol-gpu"]


# ---- where a value came from is displayed, not inferred --------------- #

@pytest.mark.parametrize("scope", ["machine-cwd", "machine-xdg", "bundle"])
def test_a_refusal_names_WHICH_file_carries_the_key(tmp_path, scope):
    """A refusal must point at the file a person has to edit.

    Three files can supply a `scheduler` block -- cwd, XDG, and the bundle's
    `.molbuilder.json` -- so a message quoting the generic ``molbuilder.json``
    names three and answers none.  This file learned that once (R10,
    2026-08-12) and N4 reintroduced it, costing thirteen confusing failures
    whose real cause was a config two directories up.

    Checked here on ``kind``, a refusal that is still live: the routing
    refusal it was written for is gone (routing is declared capability now),
    but the naming rule outlived it.
    """
    block = dict(_SCHED, kind="pbs")           # not a supported scheduler
    if scope == "bundle":
        _write_config(tmp_path, block)
        expected = tmp_path / PROJECT_CONFIG_FILENAME
    elif scope == "machine-cwd":
        (tmp_path / "molbuilder.json").write_text(
            json.dumps({"scheduler": block}))
        expected = tmp_path / "molbuilder.json"
    else:
        xdg = tmp_path / "home" / ".config" / "molbuilder"
        xdg.mkdir(parents=True)
        (xdg / "molbuilder.json").write_text(json.dumps({"scheduler": block}))
        expected = xdg / "molbuilder.json"

    with pytest.raises(RuntimeConfigError) as exc:
        get_scheduler(project_dir=tmp_path)
    assert str(expected) in str(exc.value), (
        f"the refusal must name {expected}; got: {exc.value}")


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


# ---- gpu.default_type: probed is the default, configured is the override #

def test_gpu_default_type_falls_back_to_the_probed_card(tmp_path):
    """M-1's payoff.  Two files each held a probed GPU type — the record's
    ``topology.gpu_type`` and ``scheduler.gpu.default_type`` — and only the
    first reached the code that sizes a run."""
    _write_config(tmp_path, dict(_SCHED, gpu={"partition": "public"}))
    _write_record(tmp_path, gpu_type="a100")
    assert get_scheduler(project_dir=tmp_path)["gpu"]["default_type"] == "a100"


def test_a_configured_gpu_type_still_wins(tmp_path):
    """Which card you WANT stays a choice: a site that wants the a30 says so."""
    _write_config(tmp_path, dict(_SCHED,
                                 gpu={"partition": "public",
                                      "default_type": "a30"}))
    _write_record(tmp_path, gpu_type="a100")
    assert get_scheduler(project_dir=tmp_path)["gpu"]["default_type"] == "a30"


# ---- the gpu column: two spellings, ONE reading ------------------------- #
#
# Two things write the column and neither is wrong: a probe maps gres type to
# per-node count, and a person describes one device.  What WAS wrong is that
# two call sites each read the raw column and only one understood both -- so
# the documented hand-declared row made `prep bench` refuse, naming
# ``mem_gb``/``per_node``/``type`` as GPU types.  `Domain.devices` is the one
# reading (`execution/scheduler.md` § 4, "Device").

def test_a_probed_gpu_column_reads_as_its_types():
    row = Domain(name="g", partition="general", qos="public",
                 gpu={"a100": 4, "a100.20gb": 16})
    assert {(d.type, d.per_node) for d in row.devices} == \
        {("a100", 4), ("a100.20gb", 16)}
    assert all(d.mem_gb is None for d in row.devices), \
        "sinfo does not report device memory; inventing it would be a lie"


def test_a_declared_gpu_column_reads_as_ONE_device():
    """`asu-sol.md` § 5.3's spelling.  Three keys, one device -- the reading
    that the map-only reader got backwards."""
    row = Domain(name="g", partition="general", qos="public",
                 gpu={"type": "a100", "per_node": 4, "mem_gb": 80})
    assert len(row.devices) == 1
    d = row.devices[0]
    assert (d.type, d.per_node, d.mem_gb) == ("a100", 4, 80.0)


def test_both_spellings_answer_the_same_question():
    """The fact is *what one node offers*.  Same answer, either spelling --
    which is the property every caller of `devices` relies on."""
    probed = Domain(name="g", partition="general", qos="public",
                    gpu={"a100": 4})
    declared = Domain(name="g", partition="general", qos="public",
                      gpu={"type": "a100", "per_node": 4, "mem_gb": 80})
    assert [(d.type, d.per_node) for d in probed.devices] == \
           [(d.type, d.per_node) for d in declared.devices]


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
