"""The shared versioned-document helpers (molbuilder/persist.py)."""
from __future__ import annotations

import pytest

from molbuilder import persist


# --------------------------------------------------------------------- #
#  schema_major / check_schema_major                                     #
# --------------------------------------------------------------------- #

def test_schema_major_extracts_major_or_empty():
    assert persist.schema_major("molbuilder/job-set@1") == "1"
    assert persist.schema_major("molbuilder/bench-result@2") == "2"
    assert persist.schema_major("no-at-sign") == ""     # no @ -> "" (never matches)
    assert persist.schema_major("") == ""
    assert persist.schema_major(None) == ""


def test_check_schema_major_accepts_same_major():
    # same major -> OK.  Minors aren't encoded in the string: the convention
    # is a single integer major (@1) that STAYS PUT when fields are added, so
    # "tolerate minor bumps" means the string is unchanged, not "@1.4".
    persist.check_schema_major("molbuilder/x@1", "molbuilder/x@1")


def test_check_schema_major_rejects_different_major():
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema_major("molbuilder/x@2", "molbuilder/x@1")


def test_check_schema_major_rejects_missing_at():
    # a bare/garbage schema has no major and must never match.
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema_major("garbage", "molbuilder/x@1")


def test_check_schema_major_label_prefixes_message():
    with pytest.raises(ValueError, match="job-set schema mismatch"):
        persist.check_schema_major("x@9", "x@1", label="job-set")


# --------------------------------------------------------------------- #
#  read_json / write_json                                               #
# --------------------------------------------------------------------- #

def test_write_json_roundtrips_and_is_pretty(tmp_path):
    obj = {"schema": "molbuilder/x@1", "jobs": [1, 2, 3]}
    p = persist.write_json(tmp_path / "d.json", obj)
    assert p.is_file()
    text = p.read_text()
    assert text.endswith("\n")            # trailing newline
    assert "\n  " in text                 # indented (pretty)
    assert persist.read_json(p) == obj    # lossless


# --------------------------------------------------------------------- #
#  the three adopters route through the shared check                     #
# --------------------------------------------------------------------- #

def test_adopters_use_the_shared_check():
    from molbuilder.jobset.model import JobSet, SCHEMA as JS
    from molbuilder.bench.environment import Environment, SCHEMA as ENV
    from molbuilder.bench.result import BenchResult, SCHEMA as RES

    for cls, good in ((JobSet, JS), (Environment, ENV), (BenchResult, RES)):
        # wrong major -> the unified "schema mismatch" error, every artifact.
        bad = {"schema": good.rsplit("@", 1)[0] + "@99"}
        with pytest.raises(ValueError, match="schema mismatch"):
            cls.from_dict(bad)
