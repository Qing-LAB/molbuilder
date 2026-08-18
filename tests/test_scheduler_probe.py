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
                             "partition": "htc", "qos": "debug"}
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
        assert set(d) == {"name", "partition", "qos", "max_time"}, \
            f"a domain gained a field that is not a measurement: {sorted(d)}"


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
