"""The MANIFEST is the archive's identity, so its format is exact.

**Contract:** [`job-contracts.md`](?doc=execution/job-contracts.md) § 6.1 — the
canonical format and why each rule is there; [`checkpointing.md`](?doc=execution/checkpointing.md)
§ 3 — the archive is named by the sha256 of this file; I1 — that is what makes
an archive impossible to modify without becoming a different archive.

**The one property everything here defends:** *one set of files has exactly one
possible MANIFEST, byte for byte.* Every rule below serves it, and the reason it
matters is not tidiness — two spellings of the same content would be two
different archives, so a writer and a reader that disagree about a byte disagree
about where the data lives.

These are written from the format spec, not from the parser. A test that read
the implementation could only confirm the code still says what it said.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from molbuilder.checkpoint import (
    EMPTY_MANIFEST_DIGEST,
    CheckpointError,
    archive_dir,
    format_manifest,
    manifest_digest,
    parse_manifest,
)

A = "a" * 64
B = "b" * 64


# ------------------------------------------------------------------ #
#  The property: one content, one MANIFEST                            #
# ------------------------------------------------------------------ #


def test_entry_order_does_not_change_the_bytes():
    """Two machines archiving the same files must produce the same archive.

    They will not enumerate a directory in the same order, so the format sorts.
    If it did not, the same content would land at two different names.
    """
    one = format_manifest([(A, 10, "a/one.DM"), (B, 20, "z/two.DM")])
    two = format_manifest([(B, 20, "z/two.DM"), (A, 10, "a/one.DM")])
    assert one == two
    assert manifest_digest(one) == manifest_digest(two)


def test_different_content_is_a_different_archive():
    """I1 made structural: changed content cannot occupy the same name."""
    one = format_manifest([(A, 10, "job.DM")])
    two = format_manifest([(B, 10, "job.DM")])
    assert manifest_digest(one) != manifest_digest(two)


def test_the_emitted_form_is_exactly_three_tab_separated_fields():
    raw = format_manifest([(A, 10, "sub/job.DM")]).decode("ascii")
    assert raw == f"{A}\t10\tsub/job.DM\n"


def test_round_trip():
    raw = format_manifest([(A, 10, "a.DM"), (B, 0, "b/c.DM")])
    assert parse_manifest(raw, "x") == {"a.DM": (A, 10), "b/c.DM": (B, 0)}


# ------------------------------------------------------------------ #
#  A state that archived nothing                                      #
# ------------------------------------------------------------------ #


def test_an_empty_manifest_is_legal_and_has_a_fixed_name():
    """"this state archived nothing" is a legal answer, distinct from a
    missing archive, which means the archive was lost (§ 6.1)."""
    assert parse_manifest(b"", "x") == {}
    assert format_manifest([]) == b""
    assert EMPTY_MANIFEST_DIGEST == hashlib.sha256(b"").hexdigest()


def test_every_state_with_no_big_files_shares_one_archive():
    """A consequence of naming by content (§ 3), and a welcome one."""
    assert manifest_digest(format_manifest([])) == EMPTY_MANIFEST_DIGEST


# ------------------------------------------------------------------ #
#  Refused, never repaired                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("raw, why", [
    (b"short\n",                          "fewer than three fields"),
    ((A + "\t10\ta.DM").encode(),         "no final newline"),
    ((A + "\t10\ta\n\n").encode(),        "a blank line"),
    ((A + "\t010\ta\n").encode(),         "a leading zero — a second spelling of one size"),
    ((A.upper() + "\t10\ta\n").encode(),  "uppercase hex — a second spelling of one digest"),
    (("z" * 64 + "\t10\ta\n").encode(),   "not hex at all"),
    ((A + "\t10\ta\t\n").encode(),        "a trailing empty field"),
    ((A + "  10  a.DM\n").encode(),       "two spaces, which is not the separator"),
    ((B + "\t1\tz\n" + A + "\t1\ta\n").encode(), "unsorted"),
    ((A + "\t1\tdup\n" + A + "\t1\tdup\n").encode(), "a duplicate key"),
    (b"\xef\xbb\xbf" + (A + "\t1\ta\n").encode(), "a BOM"),
    ((A + "\t1\ta\r\n").encode(),         "CRLF"),
    ("{}\t1\tcafé\n".format(A).encode("utf-8"), "non-ASCII in the key"),
])
def test_a_deviation_is_refused_and_named(raw, why):
    """No field-count fallback, no header skip, no BOM tolerance.

    Every one of these is a byte sequence a lenient reader could 'understand',
    and every one of them would put the same content at a different name than
    the writer chose.
    """
    with pytest.raises(CheckpointError) as exc:
        parse_manifest(raw, "the-archive")
    assert "the-archive" in str(exc.value), (
        "the message must say which archive to look at: " + why)


@pytest.mark.parametrize("key", [
    "/absolute",
    "../escape",
    "sub/../../escape",
    "./here",
    ".git/config",
    ".binsnapshots/x",
    "sub/.hidden/x",
    "back\\slash",
    "",
])
def test_a_key_that_could_steer_a_restore_out_of_the_folder_is_refused(key):
    """The restore writes to these paths, so the format is the boundary.

    Dot-prefixed components are rejected wholesale rather than by naming
    `.git` and `.binsnapshots`: a blocklist of two is a blocklist somebody
    extends the store with a third and forgets.
    """
    with pytest.raises(CheckpointError):
        parse_manifest(f"{A}\t1\t{key}\n".encode("utf-8", "surrogateescape"),
                       "x")


# ------------------------------------------------------------------ #
#  The writer refuses what the reader would refuse                    #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("entries, why", [
    ([("nothex", 1, "a")],       "a digest that is not 64 hex characters"),
    ([(A, -1, "a")],             "a negative size"),
    ([(A, True, "a")],           "a bool, which is an int subclass and is not a size"),
    ([(A, 1, "../x")],           "a key that escapes the folder"),
    ([(A, 1, "a"), (A, 2, "a")], "the same key twice"),
])
def test_the_writer_cannot_emit_what_the_reader_would_reject(entries, why):
    """An archive that traps itself is unrecoverable.

    If the writer could emit a MANIFEST its own parser refuses, that archive
    could never be verified or restored again — and under content addressing it
    would also be unreachable, since the digest names a file nothing can read.
    """
    with pytest.raises(CheckpointError):
        format_manifest(entries)


# ------------------------------------------------------------------ #
#  The archive's name                                                 #
# ------------------------------------------------------------------ #


def test_the_archive_directory_is_the_digest():
    raw = format_manifest([(A, 10, "job.DM")])
    assert archive_dir(Path("/calc"), manifest_digest(raw)) == (
        Path("/calc") / ".binsnapshots" / manifest_digest(raw))


@pytest.mark.parametrize("name", ["4f9ca71", "a" * 40, "not-a-digest", ""])
def test_an_archive_name_that_is_not_a_digest_is_refused(name):
    """Including a 40-character commit sha.

    Naming an archive after the state it belongs to is the scheme content
    addressing replaced (§ 3); accepting one would reintroduce an archive whose
    name does not prove its content.
    """
    with pytest.raises(CheckpointError):
        archive_dir(Path("/calc"), name)
