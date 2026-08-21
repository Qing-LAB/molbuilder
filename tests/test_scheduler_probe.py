"""`jobset probe`: parse live sinfo/sacctmgr -> the reachable domains that go
into ``environment.json`` (`configuration.md` § 5).  Fixtures are the REAL ASU
Sol output captured 2026-06-29, so the derivation is validated against the
actual cluster.

Renamed from ``test_bench_probe.py`` on 2026-08-17: the verb it names has not
been ``molbuilder bench probe-scheduler`` since the `bench` group was deleted,
and nothing it covers is a benchmark.
"""

import pytest

from molbuilder.scheduler_probe import (derive_domains, parse_allowed_qos,
                                        parse_qos, parse_sinfo)

# Real Sol `sinfo -h -o "%P|%30l|%D|%40G"` (pipe-delimited; untruncated).
_SINFO = """\
arm|7-00:00:00|4|gpu:gh200:1
fpga|7-00:00:00|4|(null)
gaudi|7-00:00:00|10|gpu:hl225:8
general|14-00:00:00|4|gpu:a100:4
general|14-00:00:00|61|(null)
general|14-00:00:00|8|gpu:l40:4
highmem|7-00:00:00|11|(null)
htc*|4:00:00|51|gpu:a100:4
htc*|4:00:00|3|gpu:a100.20gb:16
htc*|4:00:00|134|(null)
lightwork|1-00:00:00|1|gpu:a100.20gb:16
public|7-00:00:00|52|gpu:a100:4
public|7-00:00:00|5|gpu:a30:3
public|7-00:00:00|107|(null)
"""

# Real Sol `sacctmgr -nP show qos format=Name,MaxWall,Flags` (subset).
_QOS = """\
public||DenyOnLimit
private||DenyOnLimit
debug|00:15:00|DenyOnLimit,OverPartQOS
long|14-00:00:00||PartitionTimeLimit
class|1-00:00:00|DenyOnLimit
htc|04:00:00|
"""

# Real Sol `sacctmgr -nP show assoc user=$USER format=QOS`.
_ASSOC = "debug,htc,private,public\n"


def _sol():
    return (parse_sinfo(_SINFO), parse_qos(_QOS), parse_allowed_qos(_ASSOC))


# --------------------------------------------------------------------- #
#  the parsers                                                          #
# --------------------------------------------------------------------- #


def test_parse_sinfo_merges_partitions_and_gres():
    parts = {p.name: p for p in parse_sinfo(_SINFO)}
    assert parts["htc"].has_gpu and parts["htc"].gpu_types["a100"] == 4
    assert parts["htc"].nodes == 51 + 3 + 134          # node groups summed
    assert parts["fpga"].has_gpu is False
    assert parts["arm"].gpu_types == {"gh200": 1}


def test_parse_qos_maxwall():
    q = parse_qos(_QOS)
    assert q["debug"][0] == "00:15:00" and q["debug"][1] == 900
    assert q["public"] == (None, None)                 # no QoS wall ceiling
    assert q["htc"][1] == 4 * 3600


def test_parse_allowed_qos():
    assert parse_allowed_qos(_ASSOC) == {"debug", "htc", "private", "public"}


# --------------------------------------------------------------------- #
#  the derivation -- facts only (configuration.md § 5, M-1)             #
# --------------------------------------------------------------------- #


def test_every_reachable_partition_becomes_a_domain():
    """**No GPU filter** — the change that made a CPU benchmark possible.

    `derive_scheduler_block` kept only partitions carrying the chosen full GPU
    type, so Sol's four GPU partitions were the whole menu and `highmem` — a
    real partition a CPU job wants — was invisible.  A probe reports what is
    there; what a run wants is the person's.
    """
    domains, _ = derive_domains(*_sol())
    names = {d["name"] for d in domains}
    assert {"highmem", "fpga", "lightwork", "arm", "gaudi"} <= names, \
        "a CPU-only or non-NVIDIA partition is still being filtered out"
    assert {"htc", "public", "general"} <= names


def test_domains_are_ordered_cheapest_ceiling_first():
    domains, _ = derive_domains(*_sol())
    assert [d["name"] for d in domains][:3] == ["debug", "htc", "lightwork"]
    assert domains[-1]["name"] == "general"            # 14 days, the longest


def test_the_wall_is_the_smaller_of_partition_and_qos():
    doms = {d["name"]: d for d in derive_domains(*_sol())[0]}
    # htc's partition limit is 4h and the public QoS has no ceiling -> 4h
    assert doms["htc"]["max_time"] == "4:00:00"
    # debug's QoS ceiling (15 min) is smaller than any partition limit
    assert doms["debug"] == {"name": "debug", "max_time": "00:15:00",
                             "partition": "htc", "qos": "debug",
                             "gpu": {"a100": 4, "a100.20gb": 16}}
    assert doms["general"]["max_time"] == "14-00:00:00"


def test_no_debug_domain_when_the_qos_is_not_held():
    parts, qos, _ = _sol()
    domains, _ = derive_domains(parts, qos, {"public"})
    assert "debug" not in [d["name"] for d in domains]


def test_a_domain_is_never_a_preference():
    """M-1: a probe writes facts, never a default.

    `derive_scheduler_block` returned a `directives` block whose partition was
    `route_parts[0]` — *the cheapest* — plus `gpu.default_type` from a ranking
    that preferred full cards over MIG and anything over l40.  Both are choices,
    and both are the person's.  A domain carries only what was measured.
    """
    domains, _ = derive_domains(*_sol())
    assert domains, "expected some domains from the Sol fixture"
    for d in domains:
        # ``gpu`` joined 2026-08-21 (`generator.md` § 4.3a) and it IS a
        # measurement: the partition's gres inventory from sinfo, present
        # only where sinfo reported one.  A default_type-style RANKING of
        # that inventory would be the preference this test bars.
        assert set(d) <= {"name", "partition", "qos", "max_time", "gpu"}, \
            f"a domain gained a field that is not a measurement: {sorted(d)}"
    by = {d["name"]: d for d in domains}
    assert by["htc"]["gpu"] == {"a100": 4, "a100.20gb": 16}
    assert "gpu" not in by["fpga"], "no inventory -> no key, never null"


def test_the_qos_assumption_is_stated_not_hidden():
    _, notes = derive_domains(*_sol())
    assert any("ASSUMPTION" in n and "per-partition QoS list" in n
               for n in notes), \
        "the one guess this derivation makes must reach the user"


# --------------------------------------------------------------------- #
#  degenerate inputs                                                    #
# --------------------------------------------------------------------- #


def test_no_allowed_qos_falls_back_and_says_so():
    parts, qos, _ = _sol()
    domains, notes = derive_domains(parts, qos, set())
    assert all(d["qos"] == "public" for d in domains
               if d["name"] != "debug")
    assert any("could not read your allowed QoS" in n for n in notes)


def test_empty_probe_is_safe():
    """Not on a cluster: empty text -> no crash, no domains, a note."""
    domains, notes = derive_domains(parse_sinfo(""), parse_qos(""),
                                    parse_allowed_qos(""))
    assert domains == []
    assert any("no partitions" in n for n in notes)


@pytest.mark.parametrize("gone", ["best_gpu_type", "derive_scheduler_block"])
def test_the_preference_deriving_helpers_are_gone(gone):
    """Deleted 2026-08-17 (N3), not left unused.

    Both existed to pick something FOR you — a default GPU type, a default
    partition and QoS — and write it into a person's config file.  M-1 moved
    that decision back to the person, so the code that made it has no caller.
    """
    import molbuilder.scheduler_probe as probe
    assert not hasattr(probe, gone)


# --------------------------------------------------------------------- #
#  The probe VERB — declared facts and per-difference consent           #
#  (roadmap § 0.2, N3+; configuration.md § 5 M-1/M-6)                   #
# --------------------------------------------------------------------- #


def _cli(args, **kw):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, ["probe", *args], **kw)


def test_set_declares_typed_facts_and_the_source_says_flag(tmp_path):
    """M-1's declared door: a fact the probe cannot see from here arrives
    by --set, typed by the schema itself, and the record's source admits
    it (`flag`)."""
    import json
    r = _cli(["--set", "gpus_per_node=4", "--set", "gpu_type=a100",
              "--set", "mem_total_gb=1536.5",
              "--write", "--yes", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    d = json.loads((tmp_path / "environment.json").read_text())
    assert d["topology"]["gpus_per_node"] == 4
    assert d["topology"]["gpu_type"] == "a100"
    assert d["topology"]["mem_total_gb"] == 1536.5
    assert d["source"]["topology"].endswith("flag")


def test_set_refuses_unknown_keys_and_mistyped_values_by_name(tmp_path):
    r = _cli(["--set", "max_gpus=4"])
    assert r.exit_code != 0
    assert "no topology field 'max_gpus'" in r.output
    assert "gpus_per_node" in r.output          # the vocabulary, offered
    r = _cli(["--set", "gpus_per_node=four"])
    assert r.exit_code != 0 and "'four' is not int" in r.output
    r = _cli(["--set", "gpus_per_node"])
    assert r.exit_code != 0 and "KEY=VALUE" in r.output


def test_scheduler_flag_forces_the_kind_and_names_the_named_file(tmp_path):
    """--scheduler declares the kind (source `flag`); --name lands the
    record in <out>/<name>.json — how a workstation holds a cluster."""
    import json
    r = _cli(["--name", "sol-x", "--scheduler", "slurm",
              "--set", "cores_per_socket=64",
              "--write", "--yes", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    d = json.loads((tmp_path / "sol-x.json").read_text())
    assert d["scheduler"] == "slurm"
    assert d["source"]["scheduler"] == "flag"
    assert d["topology"]["cores_per_socket"] == 64
    assert "--target sol-x" in r.output


def _two_envs():
    from molbuilder.environment import Domain, Environment, Topology
    before = Environment(scheduler="workstation",
                         topology=Topology(gpus_per_node=4, gpu_type="a100"),
                         detected_at="2026-08-01T00:00:00+00:00")
    probed = Environment(scheduler="workstation",
                         topology=Topology(gpus_per_node=1, gpu_type="rtx"),
                         detected_at="2026-08-19T00:00:00+00:00")
    return before, probed


def test_consent_no_keeps_the_record_yes_takes_the_probe(monkeypatch,
                                                         capsys):
    """Per difference the user picks which value survives; No (the
    default) keeps the record, so a weaker probe cannot erase a declared
    fact."""
    import click
    from molbuilder.jobset._cli import _probe_consent_merge
    before, probed = _two_envs()
    monkeypatch.setattr(click, "confirm", lambda *a, **k: False)
    out = _probe_consent_merge(before, probed, yes=False)
    assert out.topology.gpus_per_node == 4 and out.topology.gpu_type == "a100"
    assert out.detected_at == "2026-08-19T00:00:00+00:00"   # stamp follows
    assert "kept recorded" in capsys.readouterr().out
    before, probed = _two_envs()
    monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
    out = _probe_consent_merge(before, probed, yes=False)
    assert out.topology.gpus_per_node == 1 and out.topology.gpu_type == "rtx"


def test_consent_eof_keeps_everything_silence_is_no(monkeypatch, capsys):
    """A scripted probe without --yes gets EOF at the first question and
    the record survives whole — an unanswered question declines."""
    import click
    from molbuilder.jobset._cli import _probe_consent_merge
    before, probed = _two_envs()

    def _abort(*a, **k):
        raise click.exceptions.Abort()
    monkeypatch.setattr(click, "confirm", _abort)
    out = _probe_consent_merge(before, probed, yes=False)
    assert out.topology.gpus_per_node == 4 and out.topology.gpu_type == "a100"
    text = capsys.readouterr().out
    assert "silence is no" in text


def test_consent_yes_flag_asks_nothing(monkeypatch):
    import click
    from molbuilder.jobset._cli import _probe_consent_merge

    def _explode(*a, **k):
        raise AssertionError("--yes must not ask")
    monkeypatch.setattr(click, "confirm", _explode)
    before, probed = _two_envs()
    out = _probe_consent_merge(before, probed, yes=True)
    assert out.topology.gpus_per_node == 1


def test_an_unchanged_record_is_said_not_reasked(capsys):
    from molbuilder.jobset._cli import _probe_consent_merge
    before, _ = _two_envs()
    before2, _ = _two_envs()
    out = _probe_consent_merge(before, before2, yes=False)
    assert out.topology.gpus_per_node == 4
    assert "already says this" in capsys.readouterr().out


def test_domains_diff_as_one_fact(monkeypatch, capsys):
    """The reachable-domain SET is one question, not one per row."""
    import click
    from molbuilder.environment import Domain
    from molbuilder.jobset._cli import _probe_consent_merge
    before, probed = _two_envs()
    probed.topology = before.topology            # isolate the domains diff
    probed.domains = [Domain(name="short", partition="p", qos="q",
                             max_time="1:00:00")]
    asked = []
    monkeypatch.setattr(click, "confirm",
                        lambda msg, **k: (asked.append(msg), True)[1])
    out = _probe_consent_merge(before, probed, yes=False)
    assert len(asked) == 1 and "domains" in asked[0]
    assert [d.name for d in out.domains] == ["short"]
