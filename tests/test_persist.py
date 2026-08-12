"""The shared versioned-document helpers (molbuilder/persist.py)."""
from __future__ import annotations

import json
import os

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


def test_schema_major_ignores_a_minor_component():
    """job-contracts.md § 6.1: 'checked major-only, TOLERATING same-major
    minor bumps, rejecting a different major.'

    Until 2026-08-07 the whole post-@ token was compared, so ``@1.4`` was
    rejected against ``@1`` with a message claiming the major differed when
    it did not.  Nothing shipped carried a minor, so nothing had exercised
    it; ``molbuilder/task@1``'s reader is the first that did.
    """
    assert persist.schema_major("molbuilder/task@1.4") == "1"
    assert persist.schema_major("molbuilder/task@2.0.1") == "2"
    persist.check_schema_major("molbuilder/task@1.4", "molbuilder/task@1")
    with pytest.raises(ValueError):
        persist.check_schema_major("molbuilder/task@2.0", "molbuilder/task@1")


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
    from molbuilder.environment import Environment, SCHEMA as ENV
    from molbuilder.bench.result import BenchResult, SCHEMA as RES

    for cls, good in ((JobSet, JS), (Environment, ENV), (BenchResult, RES)):
        # wrong major -> the unified "schema mismatch" error, every artifact.
        bad = {"schema": good.rsplit("@", 1)[0] + "@99"}
        with pytest.raises(ValueError, match="schema mismatch"):
            cls.from_dict(bad)


def test_write_json_is_atomic_no_tmp_left(tmp_path):
    """C2 + U8: write_json publishes via a UNIQUE temp + os.replace -- no
    temp file of any name remains,
    content is complete."""
    p = tmp_path / "cfg.json"
    persist.write_json(p, {"schema": "molbuilder/x@1", "archive_globs": ["*.DM"]})
    assert p.is_file()
    assert not (tmp_path / "cfg.json.tmp").exists()      # tmp cleaned by replace
    assert persist.read_json(p)["archive_globs"] == ["*.DM"]
    # overwrite (the set path) -- still atomic, still no tmp
    persist.write_json(p, {"schema": "molbuilder/x@1", "archive_globs": ["*.chk"]})
    assert not (tmp_path / "cfg.json.tmp").exists()
    assert persist.read_json(p)["archive_globs"] == ["*.chk"]


# --------------------------------------------------------------------- #
#  U8 -- the unique-temp shape (checkpoint's lesson, adopted here)      #
# --------------------------------------------------------------------- #


def test_write_json_uses_a_unique_temp_never_the_derived_name(tmp_path,
                                                              monkeypatch):
    """The derived ``<target>.tmp`` is the trap checkpointing.md § 6 names:
    two concurrent writers agree on one temp path and one installs the
    other's half-written bytes.  The temp must be mkstemp-unique."""
    import molbuilder.persist as persist
    seen = {}
    real = os.replace

    def spy(src, dst):
        seen["src"] = str(src)
        return real(src, dst)
    monkeypatch.setattr(persist.os, "replace", spy)
    p = tmp_path / "cfg.json"
    persist.write_json(p, {"schema": "molbuilder/x@1"})
    assert seen["src"] != str(tmp_path / "cfg.json.tmp")
    assert "cfg.json." in seen["src"] and seen["src"].endswith(".tmp")


def test_write_json_preserves_the_targets_mode(tmp_path):
    """mkstemp creates 0600; a rewrite must keep the target's own mode
    (and a fresh file gets 0644, the ordinary-create answer)."""
    import molbuilder.persist as persist
    p = tmp_path / "cfg.json"
    persist.write_json(p, {"a": 1})
    assert (p.stat().st_mode & 0o777) == 0o644
    os.chmod(p, 0o600)
    persist.write_json(p, {"a": 2})
    assert (p.stat().st_mode & 0o777) == 0o600


def test_write_json_fails_clean_on_an_unserialisable_object(tmp_path):
    """Serialisation happens BEFORE the temp exists: the target is
    untouched and no litter is left."""
    import pytest
    import molbuilder.persist as persist
    p = tmp_path / "cfg.json"
    persist.write_json(p, {"a": 1})
    with pytest.raises(TypeError):
        persist.write_json(p, {"bad": object()})
    assert json.loads(p.read_text()) == {"a": 1}
    assert list(tmp_path.iterdir()) == [p]


def test_write_json_tmp_dir_stages_outside_the_target_dir(tmp_path,
                                                          monkeypatch):
    """For a target inside a checkpointed folder, tmp_dir points the
    staging somewhere never stored -- the litter of a crash cannot be
    committed into history."""
    import molbuilder.persist as persist
    stage = tmp_path / "never-stored"
    stage.mkdir()
    seen = {}
    real = os.replace

    def spy(src, dst):
        seen["src"] = str(src)
        return real(src, dst)
    monkeypatch.setattr(persist.os, "replace", spy)
    p = tmp_path / "calc" / "job-set.json"
    p.parent.mkdir()
    persist.write_json(p, {"a": 1}, tmp_dir=stage)
    assert seen["src"].startswith(str(stage))
