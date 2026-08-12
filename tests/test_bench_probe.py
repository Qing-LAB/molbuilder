"""probe-scheduler: parse live sinfo/sacctmgr -> proposed scheduler block
(execution/job-system.md § 7).  Fixtures are the REAL ASU Sol output captured
2026-06-29, so the derivation is validated against the actual cluster."""

from molbuilder.scheduler_probe import (best_gpu_type, derive_scheduler_block,
                                     parse_allowed_qos, parse_qos, parse_sinfo)

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


def test_best_gpu_type_prefers_full_plentiful_non_l40():
    parts = parse_sinfo(_SINFO)
    # a100 (full, 51+3? no -- a100 only counts a100-bearing nodes) is the
    # most plentiful FULL non-l40 card; MIG (a100.20gb), l40 excluded.
    assert best_gpu_type(parts) == "a100"


def test_derive_matches_hand_built_sol_table():
    parts = parse_sinfo(_SINFO)
    block, notes = derive_scheduler_block(parts, parse_qos(_QOS),
                                          parse_allowed_qos(_ASSOC))
    assert block is not None
    assert block["gpu"]["default_type"] == "a100"
    # Only a100-bearing partitions become domains (arm/gaudi/l40/MIG dropped),
    # ordered cheapest -> most general, with a debug domain first.
    doms = {d["name"]: d for d in block["routing"]}
    assert [d["name"] for d in block["routing"]] == \
        ["debug", "htc", "public", "general"]
    assert doms["debug"] == {"name": "debug", "max_time": "00:15:00",
                             "partition": "htc", "qos": "debug"}
    assert doms["htc"]["partition"] == "htc" and doms["htc"]["qos"] == "public"
    assert doms["htc"]["max_time"] == "4:00:00"
    assert doms["public"]["partition"] == "public"
    assert doms["public"]["max_time"] == "7-00:00:00"
    assert doms["general"]["max_time"] == "14-00:00:00"
    # directives default = cheapest GPU partition + an allowed QoS (public).
    assert block["directives"] == {"partition": "htc", "qos": "public"}
    # gpu.mem is a site policy, so probing does not invent a value
    assert "mem" not in block["gpu"]


def test_derive_no_debug_when_not_allowed():
    block, _ = derive_scheduler_block(
        parse_sinfo(_SINFO), parse_qos(_QOS), {"public"})   # no debug QoS
    assert "debug" not in [d["name"] for d in block["routing"]]


def test_derive_none_without_gpu_partitions():
    no_gpu = "highmem|7-00:00:00|11|(null)\nfpga|7-00:00:00|4|(null)\n"
    block, notes = derive_scheduler_block(parse_sinfo(no_gpu), {}, {"public"})
    assert block is None
    assert any("no GPU" in n for n in notes)


def test_derive_empty_probe_is_safe():
    # Not on a cluster: empty text -> no crash, block None.
    block, notes = derive_scheduler_block(parse_sinfo(""), parse_qos(""),
                                          parse_allowed_qos(""))
    assert block is None
