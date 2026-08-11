"""Submission-domain routing -- recommend + explicit select + per-point job
names (execution/job-system.md § 6, § 4.4; execution/job-system.md § 6).

* ``get_routing`` reads/validates the named-domain menu (survives the
  scheduler-block normalisation);
* ``parse_walltime`` parses SLURM time strings; ``recommend_domain`` /
  ``fitting_domains`` pick by walltime/memory;
* ``bake_run_bench`` bakes the explicit-selection gate (menu on no domain,
  fit-refuse on a too-small domain, accept on a fitting one);
* the sweep names each point ``-J job-gpu-G<g>K<k>C<c>``.
"""

import json
import subprocess

import pytest

from molbuilder.bench.adapters import (SlurmAdapter, domain_fits,
                                       fitting_domains, parse_walltime,
                                       recommend_domain)
from molbuilder.bench.generate import bake_run_bench
from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                       RuntimeConfigError, get_routing)

_DOMAINS = [
    {"name": "debug",  "max_time": "0-00:15:00", "partition": "htc",
     "qos": "debug"},
    {"name": "htc",    "max_time": "0-04:00:00", "partition": "htc",
     "qos": "public"},
    {"name": "public", "max_time": "7-00:00:00", "partition": "public",
     "qos": "public"},
]


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


# ---- recommend / fit -------------------------------------------------- #

def test_recommend_cheapest_fitting_domain():
    # 10 min fits all -> cheapest (debug); 3h fits htc+public -> htc; 5d -> public
    assert recommend_domain(_DOMAINS, parse_walltime("0-00:10:00"), None) == "debug"
    assert recommend_domain(_DOMAINS, parse_walltime("0-03:00:00"), None) == "htc"
    assert recommend_domain(_DOMAINS, parse_walltime("5-00:00:00"), None) == "public"


def test_recommend_none_when_over_all_ceilings():
    assert recommend_domain(_DOMAINS, parse_walltime("9-00:00:00"), None) is None


def test_mem_cap_domain_skipped_when_job_mem_unknown():
    doms = [{"name": "big", "max_time": "7-00:00:00", "max_mem_gb": 256,
             "partition": "highmem", "qos": "public"}]
    # job_mem None -> can't prove it fits a capped domain (§ 4.3)
    assert not domain_fits(doms[0], parse_walltime("01:00:00"), None)
    assert domain_fits(doms[0], parse_walltime("01:00:00"), 128)
    assert not domain_fits(doms[0], parse_walltime("01:00:00"), 300)
    assert fitting_domains(doms, parse_walltime("01:00:00"), None) == []


# ---- get_routing ------------------------------------------------------ #

def _write(tmp_path, sched):
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
        json.dumps({"scheduler": sched}))


def test_get_routing_reads_and_survives_normalise(tmp_path):
    _write(tmp_path, {"kind": "slurm",
                      "directives": {"partition": "public", "qos": "public"},
                      "routing": _DOMAINS})
    out = get_routing(project_dir=tmp_path)
    assert [d["name"] for d in out] == ["debug", "htc", "public"]
    assert out[1]["partition"] == "htc" and out[1]["qos"] == "public"


def test_get_routing_absent_is_empty(tmp_path):
    _write(tmp_path, {"kind": "slurm",
                      "directives": {"partition": "public", "qos": "public"}})
    assert get_routing(project_dir=tmp_path) == []


def test_get_routing_duplicate_name_raises(tmp_path):
    _write(tmp_path, {"kind": "slurm",
                      "directives": {"partition": "public", "qos": "public"},
                      "routing": [_DOMAINS[1], _DOMAINS[1]]})
    with pytest.raises(RuntimeConfigError, match="duplicate domain"):
        get_routing(project_dir=tmp_path)


def test_get_routing_missing_field_raises(tmp_path):
    _write(tmp_path, {"kind": "slurm",
                      "directives": {"partition": "public", "qos": "public"},
                      "routing": [{"name": "x", "max_time": "01:00:00",
                                   "partition": "htc"}]})  # no qos
    with pytest.raises(RuntimeConfigError, match=r"routing\[0\]\.qos"):
        get_routing(project_dir=tmp_path)


# ---- bake_run_bench: the explicit-selection gate ---------------------- #

def _bake(tmp_path, **kw):
    return bake_run_bench(tmp_path, SlurmAdapter(), 64, "submit",
                          routing=_DOMAINS, **kw).read_text()


def test_run_bench_menu_when_no_domain(tmp_path):
    text = _bake(tmp_path, recommend="htc",
                 fitting=_DOMAINS[1:], job_time="0-04:00:00")
    # Bakes the menu + recommendation + the explicit-select exit path.
    assert "--domain <name>" in text
    assert "recommended for this run" in text and "$_rec" in text
    assert 'case "$_dom" in' in text
    # the syntax is valid bash
    assert subprocess.run(["bash", "-n", str(tmp_path / "run-bench")],
                          capture_output=True).returncode == 0


def test_run_bench_domain_resolves_partition_qos(tmp_path):
    text = _bake(tmp_path, recommend="htc",
                 fitting=_DOMAINS[1:], job_time="0-04:00:00")
    assert '_cpu_pq="-p htc -q public"' in text       # htc domain
    assert '_cpu_pq="-p htc -q debug"' in text        # debug domain
    assert 'export MB_GPU_PQ="$_gpu_pq"' in text


def test_run_bench_fitting_set_baked(tmp_path):
    # Only htc+public fit a 4h run; debug must be absent from the fit set
    # so selecting it is refused at runtime.
    text = _bake(tmp_path, recommend="htc",
                 fitting=_DOMAINS[1:], job_time="0-04:00:00")
    assert '_fitting="htc public"' in text


def test_run_bench_exec_domain_default_baked(tmp_path):
    text = _bake(tmp_path, exec_domain="htc", recommend="htc",
                 fitting=_DOMAINS[1:], job_time="0-04:00:00")
    assert '_dom="htc"' in text                        # standing default
