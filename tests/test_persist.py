"""The shared versioned-document helpers (molbuilder/persist.py)."""
from __future__ import annotations

import json
import os
import stat

import pytest

from molbuilder import persist


# --------------------------------------------------------------------- #
#  schema_major / check_schema                                     #
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
    it did not.
    """
    assert persist.schema_major("molbuilder/task@1.4") == "1"
    assert persist.schema_major("molbuilder/task@2.0.1") == "2"
    persist.check_schema("molbuilder/task@1.4", "molbuilder/task@1")
    with pytest.raises(ValueError):
        persist.check_schema("molbuilder/task@2.0", "molbuilder/task@1")


def test_check_schema_accepts_same_major():
    # same major -> OK.  Minors aren't encoded in the string: the convention
    # is a single integer major (@1) that STAYS PUT when fields are added, so
    # "tolerate minor bumps" means the string is unchanged, not "@1.4".
    persist.check_schema("molbuilder/x@1", "molbuilder/x@1")


def test_check_schema_rejects_different_major():
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema("molbuilder/x@2", "molbuilder/x@1")


def test_check_schema_rejects_missing_at():
    # a bare/garbage schema has no major and must never match.
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema("garbage", "molbuilder/x@1")


def test_check_schema_label_prefixes_message():
    with pytest.raises(ValueError, match="job-set schema mismatch"):
        persist.check_schema("x@9", "x@1", label="job-set")


# --------------------------------------------------------------------- #
#  read_json / write_json                                               #
# --------------------------------------------------------------------- #

def test_write_json_roundtrips_and_is_pretty(tmp_path):
    """THE one byte shape (`persist.json_text`): pretty, a trailing newline,
    a character beyond ASCII as itself -- a region label's `α`, a unit's
    `Å` -- and read back as it was.  THE SILENT FAILURE: two writers of one
    format writing the same value two ways, so a file's bytes depend on
    which door wrote it."""
    obj = {"schema": "molbuilder/x@1", "jobs": [1, 2, 3],
           "label": "α-helix", "unit": "Å"}
    p = persist.write_json(tmp_path / "d.json", obj)
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n")            # trailing newline
    assert "\n  " in text                 # indented (pretty)
    assert '"α-helix"' in text and '"Å"' in text, text   # as themselves
    assert persist.read_json(p) == obj    # lossless


def _saved_with_a_byte_order_mark(p):
    """An editor's save: a byte-order mark put first."""
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())


@pytest.mark.parametrize("road", ["the disk", "a plan holding the file",
                                  "a plan that does not hold it"])
@pytest.mark.parametrize("edited, schema, reads", [
    # (what a person did to the description, the schema it is read as,
    # whether it reads)
    (None, "task", True),
    (_saved_with_a_byte_order_mark, "task", True),
    (None, "job-request", False),
], ids=["as describe wrote it", "saved by an editor", "read as another record"])
def test_a_record_reads_as_one_object_of_its_schema(
        tmp_path, monkeypatch, road, edited, schema, reads):
    """`job-contracts.md` § 6.1: a record is read as one JSON object of its
    schema (`persist.read_object(path, schema=...)`), a byte-order mark an
    editor put first read past (`persist.text_of`) -- the same from the disk,
    from a plan holding the file and through a plan to the disk where it
    holds none (`Plan.read_object`).  The record is the description
    `molbuilder describe` wrote, a file a person may edit
    (`execution/job-request.md` § 7).  THE SILENT FAILURES: a reader
    composing the steps by hand that reads past another record's schema; a
    prep refusing a record the disk's reader reads (the plan decoded plain
    UTF-8, to 2.69.18)."""
    from molbuilder import job_request, task
    from molbuilder.jobset.planned import Plan
    from support.road import described, road_tree
    record = described(road_tree(tmp_path, monkeypatch)) / "task.json"
    as_written = json.loads(record.read_text(encoding="utf-8"))
    if edited is not None:
        edited(record)
    asked = {"task": task.SCHEMA, "job-request": job_request.SCHEMA}[schema]
    if road == "the disk":
        def read():
            return persist.read_object(record, schema=asked)
    elif road == "a plan holding the file":
        held, at = Plan(), tmp_path / "held.json"
        held.bytes(at, record.read_bytes())

        def read():
            return held.read_object(at, schema=asked)
    else:
        def read():
            return Plan().read_object(record, schema=asked)
    if reads:
        assert read() == as_written
    else:
        # REFUSED AS NOT READING, in words, never a reader's crash (§ 6.1).
        with pytest.raises(ValueError):
            read()


def test_a_prep_plan_holds_what_the_one_writer_writes(tmp_path, monkeypatch):
    """`job-contracts.md` § 6.1: the one writer takes a prep's plan
    (`write_json(..., plan=...)`) -- nothing on the disk until the plan is
    carried out, then the same bytes the disk's write gives.  THE SILENT
    FAILURES: a preview writing into the calculation (a plan's write reaching
    the disk before it is carried out), a prep writing other bytes than the
    disk's writer."""
    from molbuilder import task
    from molbuilder.jobset.planned import Plan
    from support.road import described, road_tree
    obj = persist.read_object(described(road_tree(tmp_path, monkeypatch))
                              / "task.json", schema=task.SCHEMA)
    plan = Plan()
    p = persist.write_json(tmp_path / "planned.json", obj, plan=plan)
    assert not p.exists(), "a plan writes nothing until it is carried out"
    assert plan.read_object(p, schema=task.SCHEMA) == obj
    plan.carry_out()
    disk = persist.write_json(tmp_path / "disk.json", obj)
    assert p.read_bytes() == disk.read_bytes()


# --------------------------------------------------------------------- #
#  the three adopters route through the shared check                     #
# --------------------------------------------------------------------- #

def test_adopters_use_the_shared_check():
    from molbuilder.jobset.model import JobSet, SCHEMA as JS
    from molbuilder.scheduler import Environment, SCHEMA as ENV
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


def test_write_bytes_mode_forces_owner_only_instead_of_widening(tmp_path):
    """`mode=0o600` is how a CREDENTIAL is written (`configuration.md` § 2.3).

    The default widens to 0644 because a shared artifact should not inherit
    mkstemp's 0600.  Both halves are asserted here, on a fresh file and over a loose one, so
    neither can be dropped without a failure.
    """
    import molbuilder.persist as persist
    fresh = tmp_path / "key"
    persist.write_bytes(fresh, b"s3cret", mode=0o600)
    assert (fresh.stat().st_mode & 0o777) == 0o600
    assert fresh.read_bytes() == b"s3cret"

    # A file that ARRIVED loose is not inherited from -- `mode` wins over the
    # target's own, which the default path would have kept.
    loose = tmp_path / "arrived-0644"
    loose.write_bytes(b"old")
    os.chmod(loose, 0o644)
    persist.write_bytes(loose, b"new", mode=0o600)
    assert (loose.stat().st_mode & 0o777) == 0o600

    # And the default still widens, which is the other half of the rule.
    shared = tmp_path / "artifact.json"
    persist.write_bytes(shared, b"{}")
    assert (shared.stat().st_mode & 0o777) == 0o644


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


def test_check_schema_rejects_the_wrong_artifact_at_the_same_major():
    """THE U9 pin: the check compared MAJORS ONLY until 2026-08-12, so any
    @1 artifact parsed as any other -- a task.json handed to the
    Environment reader sailed through the gate and failed later, with a
    message naming the wrong thing.  The name half is what says WHICH
    artifact this is."""
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema("molbuilder/task@1", "molbuilder/environment@1")
    with pytest.raises(ValueError, match="schema mismatch"):
        persist.check_schema("molbuilder/job-set@1",
                             "molbuilder/bench-result@1")
    # and the name helper itself
    assert persist.schema_name("molbuilder/task@1") == "molbuilder/task"
    assert persist.schema_name("garbage") == ""


def test_write_bytes_exclusive_refuses_to_replace_and_leaves_the_original(
        tmp_path):
    """`exclusive=True` CREATES the name or fails -- it never clobbers.

    The session key is why it exists: two servers starting together both reach
    that write, and under last-writer-wins the second installs a new key over
    the first, signing out everyone already logged in.  The caller needs the
    loser to LOSE, so it can read the winner's key back and sign with that.
    """
    p = tmp_path / "secret_key"
    persist.write_bytes(p, b"first-one-wins", mode=0o600)

    with pytest.raises(FileExistsError):
        persist.write_bytes(p, b"second-one-must-not", mode=0o600,
                            exclusive=True)

    # The point of the whole exercise: the original survived intact.
    assert p.read_bytes() == b"first-one-wins"
    # and the refused attempt staged a temp, which must not be left behind
    assert list(tmp_path.iterdir()) == [p]


def test_write_bytes_exclusive_creates_owner_only_and_clears_its_temp(
        tmp_path):
    """The absent case still writes, at the mode asked for, with no litter.

    `os.link` leaves the staged temp under its own name rather than renaming
    it away, so this half is not implied by the refusal case above: an
    `exclusive` create that forgot to unlink would pass that test and drop a
    second, readable copy of the secret beside it.
    """
    p = tmp_path / "secret_key"
    persist.write_bytes(p, b"x" * 32, mode=0o600, exclusive=True)

    assert p.read_bytes() == b"x" * 32
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [p]
