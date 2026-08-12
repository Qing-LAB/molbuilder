"""Submission-domain routing -- recommend + explicit select + per-point job
names (execution/job-system.md § 6, § 4.4; execution/job-system.md § 6).

* ``get_routing`` reads/validates the named-domain menu (survives the
  scheduler-block normalisation);
* ``parse_walltime`` parses SLURM time strings.
"""

import json

import pytest

from molbuilder.bench.adapters import parse_walltime
from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                       RuntimeConfigError, get_routing)


def _write(tmp_path, scheduler_block):
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(
        json.dumps({"scheduler": scheduler_block}))


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Isolated cwd + $HOME + XDG: `get_routing` deep-merges the CWD-first
    server-wide scope, so without this the verdicts depend on the repo-root
    ``molbuilder.json`` — caught 2026-08-12 the moment that file gained a
    real ``scheduler.routing`` (the configured test posture)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()


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


# `recommend_cheapest_fitting_domain` & friends, and the baked run-bench
# quartet, were RETIRED 2026-08-12 (u5): the domain-fit helpers' one
# caller was the deleted `bench prep`, and the baked launcher died with
# the bundle.  A submit-side recommendation rebuilds against
# `scheduler.routing` where it is read, when it is designed.
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



