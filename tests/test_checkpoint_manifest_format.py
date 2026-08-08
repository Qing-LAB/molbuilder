"""Tests gating the canonical MANIFEST lockdown.

Format spec: docs/execution/job-contracts.md § 6.1.
Invariants:  docs/execution/checkpointing.md (I1, I2, A2).

Real filesystem (pytest tmp_path), real git via subprocess.  No mocks
-- the bug class that prompted the lockdown (writer format diverging
from parser expectations) is invisible to mocked tests.

Coverage (checkpointing.md § 13):
  * canonical write/read round-trip
  * canonical write is sorted + atomic
  * each malformed variant raises with a specific reason
  * legacy 2-column MANIFEST -> migration round-trips
  * migration aborts on hash mismatch; original untouched
  * restore on un-migrated 2-column MANIFEST raises with migration hint
  * CLI ``molbuilder snapshot migrate-manifest`` end-to-end
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from molbuilder.checkpoint import (
    Repo, CheckpointError,
    _format_canonical_manifest, _parse_canonical_manifest,
)


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"],
                       capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _have_git(),
    reason="git not on PATH; checkpoint tests need git ≥ 2.20",
)


def _seed(tmp_path: Path) -> Path:
    """Stub working dir with one .fdf + one fake big binary."""
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x00" * 2048)
    return tmp_path


# ----------------------------------------------------------------- #
#  job-contracts.md § 6.1 — canonical format round-trips                             #
# ----------------------------------------------------------------- #


def test_canonical_write_round_trips_through_canonical_parser(tmp_path):
    """Repo.checkpoint() writes a MANIFEST that the canonical parser
    accepts on a read-back."""
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    # MANIFEST exists at the HEAD's archive dir.
    sha = repo._head_sha()
    manifest = tmp_path / ".binsnapshots" / sha / "MANIFEST"
    assert manifest.is_file()
    raw = manifest.read_bytes()
    parsed = _parse_canonical_manifest(raw, where=str(manifest.parent))
    assert "siesta-test.DM" in parsed
    sha256, size = parsed["siesta-test.DM"]
    assert size == 2048
    assert sha256 == hashlib.sha256(b"\x00" * 2048).hexdigest()


def test_canonical_manifest_is_sorted_by_filename(tmp_path):
    """job-contracts.md § 6.1: entries are alphabetical by filename so two machines
    archiving the same files produce identical MANIFEST bytes."""
    # Drop three files that, if archived in glob order, would NOT be
    # alphabetical.  Names chosen so a naïve traversal probably yields
    # a non-sorted order.
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    for name in ("zeta.DM", "alpha.DM", "mid.DM"):
        (tmp_path / name).write_bytes(b"\x00" * 16)
    repo = Repo(str(tmp_path))
    repo.init()
    sha = repo._head_sha()
    manifest_text = (tmp_path / ".binsnapshots" / sha
                     / "MANIFEST").read_text()
    names = [line.split("  ")[2] for line in manifest_text.splitlines()]
    assert names == sorted(names)
    assert names == ["alpha.DM", "mid.DM", "zeta.DM"]


def test_canonical_manifest_ends_with_newline(tmp_path):
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    sha = repo._head_sha()
    raw = (tmp_path / ".binsnapshots" / sha / "MANIFEST").read_bytes()
    assert raw.endswith(b"\n")
    # And contains no CR.
    assert b"\r" not in raw


def test_format_canonical_manifest_sorts(tmp_path):
    """The formatter sorts even when the caller passes unsorted input."""
    entries = [
        ("a" * 64, 100, "zeta.DM"),
        ("b" * 64, 200, "alpha.DM"),
        ("c" * 64, 300, "middle.DM"),
    ]
    text = _format_canonical_manifest(entries)
    lines = text.rstrip("\n").split("\n")
    names = [line.split("  ")[2] for line in lines]
    assert names == ["alpha.DM", "middle.DM", "zeta.DM"]


# ----------------------------------------------------------------- #
#  job-contracts.md § 6.1 — malformed variants all raise (parametrised)              #
# ----------------------------------------------------------------- #


def _good_line() -> str:
    return ("a" * 64) + "  " + "1024" + "  " + "siesta-test.DM"


# Each case yields a (raw_bytes, expected_substring_in_error_message).
# Strict canonical parser must reject all of them with a specific reason.
_MALFORMED_CASES = [
    pytest.param(
        b"\xef\xbb\xbf" + (_good_line() + "\n").encode("ascii"),
        "BOM",
        id="bom",
    ),
    pytest.param(
        (_good_line() + "\r\n").encode("ascii"),
        "CR",
        id="crlf",
    ),
    pytest.param(
        _good_line().encode("ascii"),     # no final newline
        "missing final newline",
        id="no-trailing-newline",
    ),
    pytest.param(
        # Non-ASCII byte (U+00E9 "é" UTF-8 = 0xc3 0xa9)
        (("a" * 64) + "  " + "1024" + "  " + "caf\xe9.DM\n").encode("utf-8"),
        "non-ASCII",
        id="non-ascii",
    ),
    pytest.param(
        # 2-column sha256sum default form -- detected explicitly so the
        # error message tells the user to run migrate-manifest.
        (("a" * 64) + "  " + "siesta-test.DM\n").encode("ascii"),
        "migrate-manifest",
        id="legacy-2-col",
    ),
    pytest.param(
        # 4-column (extra trailing field).
        (("a" * 64) + "  " + "1024" + "  " + "siesta-test.DM"
         + "  " + "extra\n").encode("ascii"),
        "fields",
        id="4-col",
    ),
    pytest.param(
        # Uppercase hex in sha256.
        (("A" * 64) + "  " + "1024" + "  " + "siesta-test.DM\n"
         ).encode("ascii"),
        "lowercase hex",
        id="uppercase-hex",
    ),
    pytest.param(
        # Non-integer size.
        (("a" * 64) + "  " + "abc" + "  " + "siesta-test.DM\n"
         ).encode("ascii"),
        "decimal integer",
        id="non-int-size",
    ),
    pytest.param(
        # Leading-zero size ("01024").
        (("a" * 64) + "  " + "01024" + "  " + "siesta-test.DM\n"
         ).encode("ascii"),
        "leading zeros",
        id="leading-zero-size",
    ),
    pytest.param(
        # Filename with embedded space.
        (("a" * 64) + "  " + "1024" + "  " + "siesta test.DM\n"
         ).encode("ascii"),
        "non-printable or whitespace",
        id="filename-embedded-space",
    ),
    pytest.param(
        # Filename starting with dot.
        (("a" * 64) + "  " + "1024" + "  " + ".hidden\n"
         ).encode("ascii"),
        "dot",
        id="filename-dotfile",
    ),
    # NOTE: a key CONTAINING a separator is now legal -- keys are repo-relative
    # POSIX paths, because the archive walk must reach nested run folders (L2 in
    # docs/execution/checkpointing.md).  What stays illegal is a key that could
    # direct a restore out of the run directory or into git's own state:
    pytest.param(
        # Absolute path.
        (("a" * 64) + "  " + "1024" + "  " + "/etc/passwd\n"
         ).encode("ascii"),
        "absolute",
        id="filename-absolute",
    ),
    pytest.param(
        # Parent-directory traversal.
        (("a" * 64) + "  " + "1024" + "  " + "coarse/../../escape.DM\n"
         ).encode("ascii"),
        "parent-directory",
        id="filename-traversal",
    ),
    pytest.param(
        # Empty component (double separator).
        (("a" * 64) + "  " + "1024" + "  " + "coarse//job.DM\n"
         ).encode("ascii"),
        "empty",
        id="filename-empty-component",
    ),
    pytest.param(
        # Dot-directory component -- would reach .git / .binsnapshots.
        (("a" * 64) + "  " + "1024" + "  " + ".git/objects/x\n"
         ).encode("ascii"),
        "dot-prefixed component",
        id="filename-dot-directory",
    ),
    pytest.param(
        # Windows separator.
        (("a" * 64) + "  " + "1024" + "  " + "coarse\\job.DM\n"
         ).encode("ascii"),
        "backslash",
        id="filename-backslash",
    ),
    pytest.param(
        # MANIFEST self-reference.
        (("a" * 64) + "  " + "1024" + "  " + "MANIFEST\n"
         ).encode("ascii"),
        "self-reference",
        id="manifest-self-reference",
    ),
    pytest.param(
        # Out-of-order entries (zeta before alpha).
        ((("a" * 64) + "  " + "1024" + "  " + "zeta.DM\n"
          + ("b" * 64) + "  " + "1024" + "  " + "alpha.DM\n")
         ).encode("ascii"),
        "sorted",
        id="unsorted",
    ),
    pytest.param(
        # A line that is only a newline: a BLANK line, which the format
        # forbids.  (A wholly empty file is legal -- see the note below.)
        b"\n",
        "blank",
        id="single-blank-line",
    ),
    # NOTE: a COMPLETELY EMPTY MANIFEST is no longer malformed -- it is the
    # canonical "this commit archived nothing", written so that a MISSING
    # archive directory unambiguously means the archive was lost.  Its positive
    # test is `test_empty_manifest_means_archived_nothing` below.  A file with
    # a blank LINE is still malformed (`single-blank-line`, above): that is a
    # separator with nothing to separate, which is a truncation signature.
    pytest.param(
        # Duplicate filename.
        (((("a" * 64) + "  " + "1024" + "  " + "alpha.DM\n") * 2)
         ).encode("ascii"),
        "more than once",
        id="duplicate-filename",
    ),
]


def test_empty_manifest_means_archived_nothing(tmp_path):
    """A zero-byte MANIFEST parses to no entries rather than raising.

    It is how a commit says "there were no big binaries here", which is what
    lets a MISSING archive directory mean "the archive was lost" -- one signal,
    one meaning (docs/execution/checkpointing.md, S1 and L7)."""
    assert _parse_canonical_manifest(b"", where=str(tmp_path)) == {}


@pytest.mark.parametrize("raw,reason_substr", _MALFORMED_CASES)
def test_malformed_manifest_raises_with_specific_reason(
        raw, reason_substr, tmp_path):
    """Every malformed shape raises CheckpointError; the message names
    the specific reason so the user can fix it."""
    with pytest.raises(CheckpointError) as exc_info:
        _parse_canonical_manifest(raw, where=str(tmp_path))
    assert reason_substr in str(exc_info.value)


# ----------------------------------------------------------------- #
#  checkpointing.md I1 — legacy migration                                         #
# ----------------------------------------------------------------- #


def _seed_and_init(tmp_path: Path) -> Repo:
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    return repo


def _write_2col_manifest(arch: Path, files: dict) -> None:
    """Write a sha256sum-style 2-column MANIFEST: `<sha>  <name>`."""
    body = "".join(f"{sha}  {name}\n" for name, sha in files.items())
    (arch / "MANIFEST").write_text(body, encoding="ascii")


def test_migrate_manifest_converts_legacy_2col_to_canonical(tmp_path):
    """Real-world case: TJ-BDT-Au111's bash-rolled MANIFEST is 2-col;
    migrate-manifest produces a canonical 3-col MANIFEST that the
    canonical parser accepts."""
    repo = _seed_and_init(tmp_path)
    sha = repo._head_sha()
    arch = tmp_path / ".binsnapshots" / sha
    # Overwrite the canonical MANIFEST with a 2-col legacy version
    # (truth source: the .DM file actually present in the archive).
    dm = arch / "siesta-test.DM"
    sha256 = hashlib.sha256(dm.read_bytes()).hexdigest()
    _write_2col_manifest(arch, {"siesta-test.DM": sha256})
    # Pre-condition: canonical parser refuses this format.
    with pytest.raises(CheckpointError, match="migrate-manifest"):
        _parse_canonical_manifest((arch / "MANIFEST").read_bytes(),
                                  where=str(arch))
    # Migrate.
    repo.tag("baseline", message="before migrate")
    result = repo.migrate_manifest("baseline")
    assert "siesta-test.DM" in result
    assert result["siesta-test.DM"] == (sha256, 2048)
    # Post-condition: canonical parser now accepts.
    parsed = _parse_canonical_manifest(
        (arch / "MANIFEST").read_bytes(), where=str(arch))
    assert parsed == result


def test_migrate_manifest_is_idempotent_on_canonical(tmp_path):
    """If the MANIFEST is already canonical, migrate is a no-op (returns
    the parsed contents without rewriting)."""
    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    arch = tmp_path / ".binsnapshots" / repo._head_sha()
    mtime_before = (arch / "MANIFEST").stat().st_mtime_ns
    result = repo.migrate_manifest("baseline")
    mtime_after = (arch / "MANIFEST").stat().st_mtime_ns
    assert "siesta-test.DM" in result
    # File untouched (mtime preserved).
    assert mtime_before == mtime_after


def test_migrate_manifest_aborts_on_sha_mismatch(tmp_path):
    """If the recorded sha256 doesn't match the file on disk, migration
    aborts AND the original MANIFEST is left untouched (checkpointing.md I1)."""
    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    # Write a 2-col MANIFEST whose sha256 does NOT match the file.
    bad_sha = "f" * 64
    _write_2col_manifest(arch, {"siesta-test.DM": bad_sha})
    legacy_bytes = (arch / "MANIFEST").read_bytes()
    with pytest.raises(CheckpointError, match="integrity check failed"):
        repo.migrate_manifest("baseline")
    # Original MANIFEST untouched (byte-for-byte).
    assert (arch / "MANIFEST").read_bytes() == legacy_bytes
    # And there is no leftover MANIFEST.tmp.
    assert not (arch / "MANIFEST.tmp").exists()


def test_migrate_manifest_handles_self_reference_in_legacy(tmp_path):
    """sha256sum * includes the MANIFEST file itself in the output;
    migrate-manifest skips the self-reference entry rather than
    refusing.  The canonical output does not list MANIFEST."""
    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    dm_sha = hashlib.sha256((arch / "siesta-test.DM").read_bytes()).hexdigest()
    manifest_sha = "0" * 64    # bogus -- skipped, not verified
    body = (f"{manifest_sha}  MANIFEST\n"
            f"{dm_sha}  siesta-test.DM\n")
    (arch / "MANIFEST").write_text(body, encoding="ascii")
    result = repo.migrate_manifest("baseline")
    assert "siesta-test.DM" in result
    assert "MANIFEST" not in result


# ----------------------------------------------------------------- #
#  checkpointing.md A2 — restore refuses legacy MANIFEST (with hint)              #
# ----------------------------------------------------------------- #


def test_restore_on_legacy_2col_manifest_raises_with_migration_hint(tmp_path):
    """User flow that prompted the lockdown: restoring against a
    legacy 2-col MANIFEST raises with a message naming the
    migrate-manifest CLI, rather than silently no-op'ing.
    """
    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    # Overwrite with legacy 2-col MANIFEST (DM bytes haven't changed).
    dm_sha = hashlib.sha256((arch / "siesta-test.DM").read_bytes()).hexdigest()
    _write_2col_manifest(arch, {"siesta-test.DM": dm_sha})
    # Make a tracked-text change so restore has something to do.
    (tmp_path / "siesta-test.fdf").write_text("changed\n")
    repo.checkpoint(message="touch fdf")
    # Now try to restore -- canonical parser raises with the migration
    # hint.  The .DM bytes on disk are NOT modified (we don't proceed
    # past the parser refusal).
    with pytest.raises(CheckpointError, match="migrate-manifest"):
        repo.restore("baseline")


# ----------------------------------------------------------------- #
#  CLI end-to-end via click.testing.CliRunner                        #
# ----------------------------------------------------------------- #


def test_cli_migrate_manifest_converts_in_place(tmp_path):
    """`molbuilder snapshot migrate-manifest <ref>` invoked via the
    real CLI runner converts the in-place MANIFEST and reports the
    archived file list."""
    from molbuilder.cli import cli

    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    dm_sha = hashlib.sha256((arch / "siesta-test.DM").read_bytes()).hexdigest()
    _write_2col_manifest(arch, {"siesta-test.DM": dm_sha})

    runner = CliRunner()
    result = runner.invoke(cli, [
        "snapshot", "migrate-manifest", "baseline",
        "--path", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "migrated MANIFEST" in result.output
    assert "siesta-test.DM" in result.output
    # And the on-disk MANIFEST is now canonical.
    parsed = _parse_canonical_manifest(
        (arch / "MANIFEST").read_bytes(), where=str(arch))
    assert "siesta-test.DM" in parsed


def test_cli_migrate_manifest_already_canonical_is_zero_exit(tmp_path):
    """Idempotent: invoking migrate-manifest on an already-canonical
    archive exits 0 without rewriting the file."""
    from molbuilder.cli import cli

    repo = _seed_and_init(tmp_path)
    repo.tag("baseline", message="first")
    arch = tmp_path / ".binsnapshots" / repo._resolve_ref("baseline")
    mtime_before = (arch / "MANIFEST").stat().st_mtime_ns

    runner = CliRunner()
    result = runner.invoke(cli, [
        "snapshot", "migrate-manifest", "baseline",
        "--path", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    mtime_after = (arch / "MANIFEST").stat().st_mtime_ns
    assert mtime_before == mtime_after
