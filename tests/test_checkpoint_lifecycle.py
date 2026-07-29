"""End-to-end tests for the run-checkpoints module.

Covers the user-facing lifecycle: init -> checkpoint -> tag -> change
files -> checkpoint again -> restore to tag.  Both text files and big
binaries (.DM-like) round-trip correctly.

See docs/execution/running-a-job.md § 6 for the full design contract.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from molbuilder.checkpoint import (
    Repo, Checkpoint, RepoState,
    CheckpointError, DirtyWorkingTreeError, NoSuchRefError,
    NestedRepoRefusedError,
)


# Skip the entire module if `git` is not on PATH (unlikely on CI but
# possible on minimal containers).  Module-level skip via the standard
# pytest pattern.
def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True,
                       check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _have_git(),
    reason="git not on PATH; run-checkpoints tests require git >= 2.20",
)


# ----------------------------------------------------------------- #
#  Lifecycle smoke                                                   #
# ----------------------------------------------------------------- #


def _seed_working_dir(tmp_path: Path) -> Path:
    """Create a stub working dir with one .fdf + one fake big binary."""
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    # 2 KB binary that matches the *.DM big-binary pattern.
    (tmp_path / "siesta-test.DM").write_bytes(b"\x00" * 2048)
    return tmp_path


def test_init_creates_repo_and_first_commit(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    assert repo.initialized
    # .gitignore was written.
    assert (tmp_path / ".gitignore").is_file()
    # .binsnapshots created.
    assert (tmp_path / ".binsnapshots").is_dir()
    # There is exactly one commit and the big binary is archived.
    cps = repo.list_checkpoints()
    assert len(cps) == 1
    head = cps[0]
    assert "initial state" in head.summary
    assert head.has_archive          # the .DM was archived
    assert head.archive_bytes == 2048


def test_init_is_idempotent(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    head_before = repo._head_sha()
    repo.init()                       # second call is a no-op
    head_after = repo._head_sha()
    assert head_before == head_after


def test_init_refuses_when_nested_working_dir_present(tmp_path):
    """P5: a directory containing nested working dirs (sub-dirs with
    .fdf / .py / .run.sh) cannot be init'd."""
    _seed_working_dir(tmp_path)
    nested = tmp_path / "stage4-subdir"
    nested.mkdir()
    (nested / "child.fdf").write_text("SystemLabel inner\n")
    repo = Repo(str(tmp_path))
    with pytest.raises(NestedRepoRefusedError,
                       match="nested working dirs"):
        repo.init()


def test_checkpoint_lifecycle_with_tag_and_restore(tmp_path):
    """Full E2E: init, tag, change files, checkpoint, restore.
    Both text and binary round-trip."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first save point")

    # Change both text and the big binary.
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test_v2\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x42" * 4096)

    cp2 = repo.checkpoint(message="iter 2: tweaked fdf + larger DM")
    assert cp2 is not None
    assert cp2.has_archive
    assert cp2.archive_bytes == 4096

    # Now restore to baseline.
    restored = repo.restore("baseline")
    assert "siesta-test.DM" in restored
    # Text rewound to v1.
    assert (tmp_path / "siesta-test.fdf").read_text() == "SystemLabel test\n"
    # Binary rewound to original 2 KB content.
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\x00" * 2048


def test_checkpoint_on_clean_tree_is_polite_noop(tmp_path):
    """User clicks 'Checkpoint now' on a clean tree: returns None, no
    commit, no error."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    cps_before = len(repo.list_checkpoints())
    result = repo.checkpoint(message="should be a no-op")
    assert result is None
    cps_after = len(repo.list_checkpoints())
    assert cps_before == cps_after


def test_restore_refuses_on_dirty_tree(tmp_path):
    """The user must explicitly checkpoint or discard first."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    # Make a dirty change.
    (tmp_path / "siesta-test.fdf").write_text("dirty change\n")
    with pytest.raises(DirtyWorkingTreeError,
                       match="uncommitted changes"):
        repo.restore("baseline")


def test_restore_unknown_ref_raises(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    with pytest.raises(NoSuchRefError, match="no such ref"):
        repo.restore("does-not-exist")


def test_tag_requires_message(tmp_path):
    """Per § 11 decision 3: tags are always annotated; empty message
    is refused."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    with pytest.raises(CheckpointError, match="message must be non-empty"):
        repo.tag("nope", message="")


def test_archive_integrity_check_catches_corruption(tmp_path):
    """If a binary in the archive is corrupted (sha256 mismatch) the
    restore refuses rather than silently overwriting."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    # Corrupt the archived binary.
    sha = repo._resolve_ref("baseline")
    archive = tmp_path / ".binsnapshots" / sha / "siesta-test.DM"
    archive.write_bytes(b"\xff" * 2048)        # different content
    # Reset working-tree change so restore can proceed past the dirty check.
    # (No working-tree dirty here.)
    with pytest.raises(CheckpointError,
                       match="integrity check failed"):
        repo.restore("baseline")


def test_state_reports_initialized_and_head(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    pre = repo.state()
    assert pre.initialized is False
    assert pre.head is None
    repo.init()
    post = repo.state()
    assert post.initialized is True
    assert post.head is not None
    assert post.dirty is False


def test_state_reports_dirty_after_edit(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    (tmp_path / "siesta-test.fdf").write_text("dirty\n")
    state = repo.state()
    assert state.dirty is True


def test_list_decorates_with_tags(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first save point")
    cps = repo.list_checkpoints()
    assert cps
    head = cps[0]
    assert any("baseline" in r for r in head.refs)


def test_resolve_ref_public_method(tmp_path):
    """Public ``Repo.resolve_ref`` (added 2026-06-26 to keep blueprints
    out of underscore-prefixed methods).  Returns the commit SHA on
    success, raises NoSuchRefError on a bogus ref."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    sha = repo.resolve_ref("baseline")
    assert len(sha) == 40
    # Annotated-tag peel: the public method returns the commit SHA,
    # not the tag-object SHA.
    assert sha == repo._head_sha()
    with pytest.raises(NoSuchRefError):
        repo.resolve_ref("nope")


def test_diff_public_method_with_pathspec(tmp_path):
    """Public ``Repo.diff`` returns unified-diff text; both refs are
    validated before invoking git diff, so an unknown ref surfaces as
    NoSuchRefError rather than a CheckpointError wrapping git's
    error string."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel v2\n")
    repo.checkpoint(message="iter 2")
    out = repo.diff("baseline", "HEAD", pathspec=["*.fdf"])
    assert "SystemLabel test" in out
    assert "SystemLabel v2" in out
    # Unknown ref -> NoSuchRefError, not a generic CheckpointError.
    with pytest.raises(NoSuchRefError):
        repo.diff("baseline", "ghost")


def test_diff_public_method_without_pathspec(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    (tmp_path / "siesta-test.fdf").write_text("changed\n")
    repo.checkpoint(message="iter 2")
    # No pathspec -- whole repo.
    out = repo.diff("baseline", "HEAD")
    assert "SystemLabel test" in out


def test_default_gitignore_excludes_big_binaries(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    body = (tmp_path / ".gitignore").read_text()
    for pat in ("*.DM", "*.HSX", "*.TSHS", ".binsnapshots/"):
        assert pat in body, f"{pat} missing from .gitignore"


def test_user_supplied_gitignore_is_not_overwritten(tmp_path):
    """If the user wrote a .gitignore before calling init, we don't
    clobber it."""
    custom = "*.foo\nmy-custom-pattern/\n"
    (tmp_path / ".gitignore").write_text(custom)
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    assert (tmp_path / ".gitignore").read_text() == custom


# ----------------------------------------------------------------- #
#  Phase 4: experimental branching (run-checkpoints.md § 4.5)        #
# ----------------------------------------------------------------- #


def test_branch_forks_and_isolates_the_alternative(tmp_path):
    """P6: a branch carries an experiment without disturbing the original
    path. Checkpoints after `branch` land on the branch; the original
    branch's converged state is recoverable."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("converged", message="good geometry")
    original = (tmp_path / "siesta-test.fdf").read_text()

    # fork an experiment and change the input on the branch.
    repo.branch("try-tzp")
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\nPAO.BasisSize TZP\n")
    cp = repo.checkpoint(message="experiment: TZP basis")
    assert cp is not None
    # the experimental checkpoint is on the new branch.
    state = repo.state()
    assert state.current_branch == "try-tzp"

    # the converged tag still restores the ORIGINAL input (isolation).
    repo.restore("converged")
    assert (tmp_path / "siesta-test.fdf").read_text() == original


def test_branch_rejects_duplicate_name(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.branch("exp")
    with pytest.raises(CheckpointError):
        repo.branch("exp")            # already exists -> git errors, wrapped


def test_branch_rejects_empty_name(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    with pytest.raises(CheckpointError, match="non-empty"):
        repo.branch("   ")


def test_cli_snapshot_branch(tmp_path):
    from click.testing import CliRunner
    from molbuilder.cli import cli
    _seed_working_dir(tmp_path)
    Repo(str(tmp_path)).init()
    r = CliRunner().invoke(cli, ["snapshot", "branch", "stage4-tzp",
                                 "-p", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "stage4-tzp" in r.output
    assert Repo(str(tmp_path)).state().current_branch == "stage4-tzp"


# ----------------------------------------------------------------- #
#  Binary-archive integrity on restore (§ 10.3 data-safety) —       #
#  the "verify sha256+size BEFORE touching the working tree" path.   #
# ----------------------------------------------------------------- #


def _sole_archive_dir(tmp_path: Path) -> Path:
    dirs = [d for d in (tmp_path / ".binsnapshots").iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _clean_head_with_corruptible_v1(tmp_path):
    """init (c1, .DM=zeros) tagged 'v1', then a clean 'c2' (.DM=0x22) so the
    working tree matches HEAD -- no dirty-binary refusal (#2) fires, and we
    can corrupt v1's OLD archive to exercise restore-time integrity (#3).
    Returns (repo, v1_archive_dir)."""
    from molbuilder.checkpoint import _archive_dir
    _seed_working_dir(tmp_path)                          # .DM = zeros
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    v1 = repo.resolve_ref("v1")
    # change text too -- a binary-only change is treated as "clean" by
    # checkpoint(), so we need a text edit to force a real c2 commit+archive.
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n# c2\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x22" * 2048)
    repo.checkpoint(message="c2")                        # HEAD now matches working
    return repo, _archive_dir(tmp_path, v1)


def test_restore_aborts_on_tampered_archived_binary(tmp_path):
    """A corrupted archived binary (bytes changed, MANIFEST sha unchanged)
    must abort restore and NOT copy the bad bytes into the working tree."""
    repo, v1arch = _clean_head_with_corruptible_v1(tmp_path)
    (v1arch / "siesta-test.DM").write_bytes(b"\xFF" * 2048)   # sha mismatch
    with pytest.raises(CheckpointError, match="integrity check failed"):
        repo.restore("v1")
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\x22" * 2048  # untouched


def test_restore_aborts_on_archived_binary_size_mismatch(tmp_path):
    repo, v1arch = _clean_head_with_corruptible_v1(tmp_path)
    (v1arch / "siesta-test.DM").write_bytes(b"\x00" * 1024)   # size mismatch
    with pytest.raises(CheckpointError, match="integrity check failed"):
        repo.restore("v1")


def test_restore_aborts_on_missing_archived_file(tmp_path):
    repo, v1arch = _clean_head_with_corruptible_v1(tmp_path)
    (v1arch / "siesta-test.DM").unlink()                 # MANIFEST still lists it
    with pytest.raises(CheckpointError, match="refusing to restore"):
        repo.restore("v1")


def test_restore_refuses_on_uncommitted_binary_change(tmp_path):
    """#2 SAFETY: a modified-but-not-checkpointed big binary must NOT be
    silently overwritten by restore -- git can't see gitignored binaries, so
    restore checks them separately and refuses (P3: the user decides)."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x77" * 2048)   # uncommitted edit
    with pytest.raises(DirtyWorkingTreeError, match="uncommitted binary changes"):
        repo.restore("v1")
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\x77" * 2048  # preserved


def test_restore_include_binaries_false_skips_integrity_and_binaries(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    (tmp_path / "siesta-test.DM").write_bytes(b"\xAB" * 2048)
    # even a corrupt archive is fine when binaries are skipped.
    (_sole_archive_dir(tmp_path) / "siesta-test.DM").write_bytes(b"\xFF" * 999)
    restored = repo.restore("v1", include_binaries=False)
    assert restored == []
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\xAB" * 2048  # untouched


def test_archive_total_bytes_reports_archived_size(tmp_path):
    _seed_working_dir(tmp_path)                     # one 2048-byte .DM
    repo = Repo(str(tmp_path))
    repo.init()
    assert repo.archive_total_bytes() == 2048


def test_restore_is_atomic_corrupt_binary_leaves_text_untouched(tmp_path):
    """SAFETY (§ 10.3 atomicity): a corrupt binary archive must abort the
    WHOLE restore -- the TEXT must NOT be rewound either. Before the
    verify-before-git-restore fix this left a half-restored tree (text@ref,
    binaries@current)."""
    from molbuilder.checkpoint import _archive_dir
    _seed_working_dir(tmp_path)                       # .fdf v1, .DM v1
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    v1_sha = repo.resolve_ref("v1")
    # advance BOTH text and binary to a v2 state and commit.
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test_v2\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x22" * 2048)
    repo.checkpoint(message="v2")
    # corrupt v1's archived binary (sha mismatch, same size).
    (_archive_dir(tmp_path, v1_sha) / "siesta-test.DM").write_bytes(b"\xFF" * 2048)

    with pytest.raises(CheckpointError, match="integrity check failed"):
        repo.restore("v1")
    # NEITHER text nor binary was touched -- restore was all-or-nothing.
    assert (tmp_path / "siesta-test.fdf").read_text() == "SystemLabel test_v2\n"
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\x22" * 2048


def test_checkpoint_produces_a_self_consistent_archive(tmp_path):
    """SAVE-side integrity: a freshly-written archive verifies against its own
    MANIFEST (the sha256+size recorded match the bytes actually archived)."""
    from molbuilder.checkpoint import _verify_archived_binaries
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    expected = _verify_archived_binaries(tmp_path, repo.resolve_ref("HEAD"))
    assert expected["siesta-test.DM"] == (
        __import__("hashlib").sha256(b"\x00" * 2048).hexdigest(), 2048)


def test_archive_detects_copy_corruption_at_save(tmp_path, monkeypatch):
    """SAVE-side fidelity: a silently-corrupting copy (disk error) must make
    archiving FAIL loudly -- never record the corrupt copy's own sha as truth
    (which would later 'verify' against itself and restore bad bytes)."""
    import molbuilder.checkpoint as cp
    _seed_working_dir(tmp_path)                      # .DM = 2048 zeros
    def _corrupting_copy(src, dst, *a, **k):
        Path(dst).write_bytes(b"\xFF" * 2048)        # wrong bytes
    monkeypatch.setattr(cp.shutil, "copy2", _corrupting_copy)
    repo = cp.Repo(str(tmp_path))
    with pytest.raises(cp.CheckpointError, match="corrupt"):
        repo.init()                                  # init archives the .DM


# ----------------------------------------------------------------- #
#  #1: missing-archive is a LOUD warning, never a silent text-only    #
#      restore, for a project that uses big binaries.                 #
# ----------------------------------------------------------------- #


def test_missing_archive_warning_flags_lost_archive(tmp_path):
    """A ref whose archive is gone, in a binary-using project, warns."""
    import shutil as _sh
    from molbuilder.checkpoint import _archive_dir
    _seed_working_dir(tmp_path)                          # project HAS a .DM
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    _sh.rmtree(_archive_dir(tmp_path, repo.resolve_ref("v1")))   # lost archive
    w = repo.missing_archive_warning("v1")
    assert w is not None and "NO binary archive" in w


def test_missing_archive_warning_none_for_binary_free_project(tmp_path):
    """A legitimately text-only checkpoint must NOT warn (no false alarm)."""
    (tmp_path / "job.fdf").write_text("SystemLabel test\n")   # no big binary
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="text only")
    assert repo.missing_archive_warning("v1") is None


def test_cli_restore_warns_on_missing_archive(tmp_path):
    import shutil as _sh
    from click.testing import CliRunner
    from molbuilder.checkpoint import _archive_dir
    from molbuilder.cli import cli
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("v1", message="baseline")
    v1 = repo.resolve_ref("v1")
    # clean c2 so restore's dirty-binary guard passes, then lose v1's archive.
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n# c2\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x22" * 2048)
    repo.checkpoint(message="c2")
    _sh.rmtree(_archive_dir(tmp_path, v1))
    r = CliRunner().invoke(cli, ["snapshot", "restore", "v1",
                                 "-p", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "WARNING" in r.output and "NO binary archive" in r.output


# ----------------------------------------------------------------- #
#  Engine-aware, persistent, editable big-binary classification      #
#  (run-checkpoints.md § 9) -- the unified accessor.                  #
# ----------------------------------------------------------------- #


def test_init_pyscf_engine_archives_chk_not_dm(tmp_path):
    """PySCF's .chk is a big binary; SIESTA's .DM is not, for a pyscf repo."""
    (tmp_path / "job.py").write_text("# pyscf script\n")
    (tmp_path / "job.chk").write_bytes(b"\x00" * 4096)     # PySCF checkpoint
    repo = Repo(str(tmp_path))
    repo.init(engine="pyscf")
    assert "*.chk" in repo.archive_globs()
    assert "*.DM" not in repo.archive_globs()
    # .chk archived (not git-committed), and gitignored.
    sha = repo.resolve_ref("HEAD")
    assert (tmp_path / ".binsnapshots" / sha / "job.chk").is_file()
    assert "*.chk" in (tmp_path / ".gitignore").read_text()


def test_init_siesta_engine_uses_siesta_globs(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    assert "*.DM" in repo.archive_globs() and "*.chk" not in repo.archive_globs()


def test_init_no_engine_uses_safe_union(tmp_path):
    """Unspecified engine -> union, so nothing is missed regardless of engine."""
    repo = Repo(str(tmp_path))
    (tmp_path / "note.txt").write_text("x")
    repo.init()
    globs = repo.archive_globs()
    assert "*.DM" in globs and "*.chk" in globs          # both engines covered


def test_set_archive_globs_persists_and_syncs_gitignore(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.set_archive_globs(["*.DM", "*.chk", "*.custombin"])
    # persisted (read back through the unified accessor)
    assert Repo(str(tmp_path)).archive_globs() == ["*.DM", "*.chk", "*.custombin"]
    # .gitignore regenerated to match
    gi = (tmp_path / ".gitignore").read_text()
    assert "*.custombin" in gi and "*.chk" in gi
    # a *.custombin file is now treated as a big binary
    (tmp_path / "x.custombin").write_bytes(b"\x00" * 512)
    cp = repo.checkpoint(message="add custom binary")
    assert cp is not None and cp.has_archive


def test_set_archive_globs_rejects_empty(tmp_path):
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    with pytest.raises(CheckpointError, match="cannot be empty"):
        repo.set_archive_globs([])


def test_cli_snapshot_config_show_and_set(tmp_path):
    from click.testing import CliRunner
    from molbuilder.cli import cli
    _seed_working_dir(tmp_path)
    Repo(str(tmp_path)).init(engine="siesta")
    runner = CliRunner()
    # show
    r = runner.invoke(cli, ["snapshot", "config", "-p", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "*.DM" in r.output
    # set
    r = runner.invoke(cli, ["snapshot", "config", "-p", str(tmp_path),
                            "--set", "*.DM,*.chk"])
    assert r.exit_code == 0, r.output
    assert Repo(str(tmp_path)).archive_globs() == ["*.DM", "*.chk"]


# ----------------------------------------------------------------- #
#  Fresh-eye review fixes (2026-07)                                   #
# ----------------------------------------------------------------- #


def test_overlapping_globs_do_not_duplicate_archive_entries(tmp_path):
    """C1: overlapping globs (*.DM + *.D*) must not list a file twice ->
    duplicate MANIFEST entries would trap the checkpoint (restore rejects)."""
    from molbuilder.checkpoint import _list_big_binaries
    _seed_working_dir(tmp_path)                          # sim... sits as siesta-test.DM
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.set_archive_globs(["*.DM", "*.D*"])             # overlapping
    found = _list_big_binaries(tmp_path)
    assert [p.name for p in found].count("siesta-test.DM") == 1   # deduped
    # and a real checkpoint -> restore roundtrip works (no dup MANIFEST).
    (tmp_path / "siesta-test.fdf").write_text("v2\n")
    repo.checkpoint(message="c2")
    repo.tag("v2", message="v2")
    (tmp_path / "siesta-test.fdf").write_text("v3\n")
    repo.checkpoint(message="c3")
    repo.restore("v2")                                   # would raise if duplicated
    assert (tmp_path / "siesta-test.fdf").read_text() == "v2\n"


def test_corrupt_head_manifest_gives_clear_error_not_dirty(tmp_path):
    """C4: a corrupt HEAD MANIFEST must say 'corrupt/unreadable', not the
    misleading 'uncommitted binary changes'."""
    _seed_working_dir(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.tag("v1", message="v1")
    manifest = _sole_archive_dir(tmp_path) / "MANIFEST"
    manifest.write_text("this is not a valid manifest\n")   # corrupt HEAD archive
    with pytest.raises(CheckpointError, match="corrupt|unreadable"):
        repo.restore("v1")


def test_archive_copy_corruption_leaves_no_partial_archive(tmp_path, monkeypatch):
    """#3: a corrupting copy at save fails loudly AND leaves NO final archive
    dir (only the throwaway .tmp is cleaned) -- never a partial archive."""
    import molbuilder.checkpoint as cp
    _seed_working_dir(tmp_path)
    monkeypatch.setattr(cp.shutil, "copy2",
                        lambda s, d, *a, **k: Path(d).write_bytes(b"\xFF" * 8))
    repo = cp.Repo(str(tmp_path))
    with pytest.raises(cp.CheckpointError, match="corrupt"):
        repo.init(engine="siesta")
    snaps = tmp_path / ".binsnapshots"
    # no *.tmp and no partial per-sha archive left behind
    leftovers = [d.name for d in snaps.iterdir() if d.is_dir()] if snaps.is_dir() else []
    assert all(not n.endswith(".tmp") for n in leftovers), leftovers


def test_restore_copy_failure_is_clear_error(tmp_path, monkeypatch):
    """C3: a copy fault AFTER verification surfaces as a clear CheckpointError,
    not a raw OSError."""
    import molbuilder.checkpoint as cp
    _seed_working_dir(tmp_path)
    repo = cp.Repo(str(tmp_path))
    repo.init(engine="siesta")
    repo.tag("v1", message="v1")
    # verify uses read_bytes (still real); only the copy step is broken.
    def _boom(s, d, *a, **k):
        raise OSError("simulated disk fault")
    monkeypatch.setattr(cp.shutil, "copy2", _boom)
    with pytest.raises(cp.CheckpointError, match="copying .* failed"):
        repo.restore("v1")
