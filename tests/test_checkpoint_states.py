"""States, where you stand, and going back — the `Repo` API (§ 5, § 7.1).

**Contract:** [`checkpointing.md`](?doc=execution/checkpointing.md) § 5 (state,
tag, where you stand) · § 7.1 (going back and trying something else with the
original intact) · S1 · L3 · A5 · A6.

Real git, real files, no mocks. § 13.3: these rules are about what is on disk
after an operation, and a mocked filesystem stays green while the real predicate
reads the wrong names.

The size limit is set small in the fixture so the **size gate** is what decides,
not a name — that is S1b, and it is the path that had a live defect.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from molbuilder.checkpoint import (
    CheckpointError,
    DirtyWorkingTreeError,
    MANIFEST_NAME,
    NoSuchRefError,
    Repo,
    archive_dir,
    archive_key,
    parse_manifest,
    walk_files,
)
from molbuilder.runtime_config import PROJECT_CONFIG_FILENAME

BIG = b"\x01" * 5000       # over the fixture's limit
SMALL = "text\n"


@pytest.fixture()
def calc(tmp_path):
    """A calculation folder whose store is decided by SIZE, not by name."""
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / PROJECT_CONFIG_FILENAME).write_text(json.dumps(
        {"checkpoint": {"size_limit_bytes": 1024, "engines": {"generic": []}}}))
    (root / "job.fdf").write_text("SystemLabel job\n")
    repo = Repo(str(root))
    repo.init(note="set up")
    return repo


def _tracked(repo):
    out = subprocess.run(["git", "ls-files"], cwd=repo.path,
                         capture_output=True, text=True, check=True).stdout
    return set(out.split())


def _archived(repo, state):
    man = archive_dir(repo.root, state.archive) / MANIFEST_NAME
    return set(parse_manifest(man.read_bytes(), "x"))


# ------------------------------------------------------------------ #
#  S1 — every regular file in exactly one store                       #
# ------------------------------------------------------------------ #


def test_every_regular_file_is_in_exactly_one_store(calc):
    """Never both, never neither — walked, with no allow-list.

    Both halves matter: `tracked` alone passes while a big file sits in no
    store at all, which is the branch that loses data.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "sub").mkdir()
    (calc.root / "sub" / "nested.bin").write_bytes(BIG)
    (calc.root / "sub" / "notes.txt").write_text(SMALL)
    (calc.root / ".scratch").mkdir()
    (calc.root / ".scratch" / "hidden.bin").write_bytes(BIG)
    state = calc.save("everything at once")

    tracked, archived = _tracked(calc), _archived(calc, state)
    for path in walk_files(calc.root):
        key = archive_key(calc.root, path)
        assert (key in tracked) != (key in archived), (
            f"S1 violated for {key!r}: tracked={key in tracked} "
            f"archived={key in archived}")
    assert "sub/nested.bin" in archived, "the walk must reach subdirectories"
    assert ".scratch/hidden.bin" in archived, (
        "S1 exempts no category but the two stores; a hidden directory is an "
        "ordinary directory")


def test_the_two_stores_are_never_stored(calc):
    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("one big file")
    for key in _tracked(calc) | _archived(calc, state):
        assert not key.startswith((".git/", ".binsnapshots/"))


# ------------------------------------------------------------------ #
#  § 5 — where you stand, and what "unsaved" means                    #
# ------------------------------------------------------------------ #


def test_a_folder_is_clean_immediately_after_saving(calc):
    """Including a file that is big by SIZE rather than by name.

    This is the defect that made the size gate unusable: such a file cannot be
    named in `.gitignore`, so git sees it as untracked and reported it as added
    forever — the folder never read clean and every restore demanded --force.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("saved a big file")
    assert calc.status().clean


def test_unsaved_is_measured_against_where_you_stand_not_the_newest(calc):
    """The whole reason the baseline is what it is (§ 5).

    Against the newest state, going back would make the entire folder read as
    modified — the warning would fire on every restore, about content that is
    already saved, and people would learn to click through it.
    """
    (calc.root / "job.XV").write_text("stage 1\n")
    first = calc.save("stage 1")
    (calc.root / "job.XV").write_text("stage 2\n")
    calc.save("stage 2")

    calc.restore(first.id)
    status = calc.status()
    assert status.standing_at == first.id
    assert status.clean, "standing at an older state is not 'unsaved work'"


def test_the_three_shapes_are_all_reported(calc):
    """A5: changed, added and deleted are all lost when the folder is made
    equal to a target, so all three are named."""
    (calc.root / "keep.txt").write_text(SMALL)
    (calc.root / "gone.txt").write_text(SMALL)
    calc.save("two files")

    (calc.root / "keep.txt").write_text("edited\n")
    (calc.root / "fresh.txt").write_text(SMALL)
    (calc.root / "gone.txt").unlink()

    status = calc.status()
    assert "keep.txt" in status.changed
    assert "fresh.txt" in status.added
    assert "gone.txt" in status.deleted
    assert set(status.unsaved()) == {"keep.txt", "fresh.txt", "gone.txt"}


def test_a_changed_big_file_is_seen_even_though_git_cannot_see_it(calc):
    """L7's sibling: big files are outside git, so the MANIFEST is their record."""
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    assert "big.bin" in calc.status().changed


def test_a_big_file_only_change_still_produces_a_state(calc):
    """L7.  `git status` is clean when only a gitignored file changed, and a
    save that trusted it would report 'nothing to do' about work you believed
    you had saved."""
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    second = calc.save("only the big file changed")
    assert second is not None and second.id != first.id
    assert second.archive != first.archive


def test_saving_an_unchanged_folder_produces_nothing(calc):
    assert calc.save("nothing happened") is None


# ------------------------------------------------------------------ #
#  § 7.1 — going back, and the original staying intact                #
# ------------------------------------------------------------------ #


def test_going_back_and_saving_forks_without_declaring_anything(calc):
    """The claim § 7.1 is built on: both attempts share a parent.

    Nothing was named, no branch verb was run, and the list can show that they
    are alternatives because each state records where it came from.
    """
    (calc.root / "job.XV").write_text("stage 1\n")
    stage1 = calc.save("stage 1 converged")
    (calc.root / "job.XV").write_text("200 Ry\n")
    attempt_a = calc.save("stage 2 at 200 Ry")

    calc.restore(stage1.id)
    (calc.root / "job.XV").write_text("300 Ry\n")
    attempt_b = calc.save("stage 2 at 300 Ry")

    assert attempt_a.parent == attempt_b.parent == stage1.id
    assert attempt_a.id != attempt_b.id


def test_the_first_attempt_is_intact_after_the_second(calc):
    """Nothing is ever rewritten: restoring changed the FOLDER, not the state."""
    (calc.root / "job.XV").write_text("stage 1\n")
    stage1 = calc.save("stage 1")
    (calc.root / "big.bin").write_bytes(BIG)
    attempt_a = calc.save("attempt A")

    calc.restore(stage1.id)
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    calc.save("attempt B")

    calc.restore(attempt_a.id)
    assert (calc.root / "big.bin").read_bytes() == BIG


def test_a_restore_removes_what_the_target_did_not_have(calc):
    """A5, and without a warning: those files are not lost, they are in the
    state that holds them.  Leaving a later stage's output in a folder claiming
    to be an earlier one is exactly the file a run picks up unasked."""
    early = calc.save("early") or calc.standing_at()
    (calc.root / "later.txt").write_text(SMALL)
    (calc.root / "later.bin").write_bytes(BIG)
    calc.save("later")

    calc.restore(early.id)
    assert not (calc.root / "later.txt").exists()
    assert not (calc.root / "later.bin").exists()


# ------------------------------------------------------------------ #
#  A5 — warn, then obey                                               #
# ------------------------------------------------------------------ #


def test_a_restore_refuses_unsaved_work_and_names_it(calc):
    (calc.root / "job.XV").write_text("saved\n")
    first = calc.save("first")
    (calc.root / "job.XV").write_text("unsaved edit\n")
    (calc.root / "new.bin").write_bytes(BIG)

    with pytest.raises(DirtyWorkingTreeError) as exc:
        calc.restore(first.id)
    message = str(exc.value)
    assert "job.XV" in message and "new.bin" in message, (
        "a count tells nobody anything; the files must be named")


def test_force_accepts_the_loss_and_completes(calc):
    (calc.root / "job.XV").write_text("saved\n")
    first = calc.save("first")
    (calc.root / "job.XV").write_text("unsaved\n")
    calc.restore(first.id, force=True)
    assert (calc.root / "job.XV").read_text() == "saved\n"
    assert calc.status().clean


def test_a_restore_that_refuses_changes_nothing(calc):
    """A2: the refusals come before the first byte moves."""
    (calc.root / "job.XV").write_text("saved\n")
    first = calc.save("first")
    (calc.root / "job.XV").write_text("unsaved\n")
    before = (calc.root / "job.XV").read_text()
    with pytest.raises(DirtyWorkingTreeError):
        calc.restore(first.id)
    assert (calc.root / "job.XV").read_text() == before


def test_an_unknown_state_is_refused_before_anything_else(calc):
    (calc.root / "job.XV").write_text("unsaved\n")
    with pytest.raises(NoSuchRefError):
        calc.restore("no-such-state")


# ------------------------------------------------------------------ #
#  A6 — nothing you saved becomes unreachable                         #
# ------------------------------------------------------------------ #


def test_every_state_survives_any_amount_of_wandering(calc):
    made = []
    for i in range(4):
        (calc.root / "job.XV").write_text(f"step {i}\n")
        made.append(calc.save(f"step {i}"))
    calc.restore(made[0].id, force=True)
    calc.restore(made[2].id, force=True)
    calc.restore(made[1].id, force=True)

    listed = {s.id for s in calc.states()}
    for state in made:
        assert state.id in listed, "a saved state disappeared (A6)"
        assert calc.restore(state.id, force=True).id == state.id


def test_a_state_saved_after_a_restore_is_still_reachable(calc):
    """The hazard A6 exists for: work saved while standing at an old state
    must not become unreferenced when you move away again."""
    first = calc.save("first") or calc.standing_at()
    (calc.root / "job.XV").write_text("a\n")
    calc.save("a")
    calc.restore(first.id)
    (calc.root / "job.XV").write_text("b\n")
    forked = calc.save("saved while standing at an old state")

    calc.restore(first.id, force=True)
    subprocess.run(["git", "gc", "--prune=now", "--quiet"],
                   cwd=calc.path, check=True)
    assert forked.id in {s.id for s in calc.states()}
    assert calc.restore(forked.id, force=True).id == forked.id


# ------------------------------------------------------------------ #
#  L3 — a note, and the calculation's name                            #
# ------------------------------------------------------------------ #


def test_a_state_without_a_note_is_refused(calc):
    (calc.root / "job.XV").write_text("x\n")
    for empty in ("", "   ", "\n"):
        with pytest.raises(CheckpointError):
            calc.save(empty)


def test_every_state_names_its_calculation(calc):
    """A folder can be copied to a cluster or opened a year later, and a
    history whose states say only "stage 2 converged" cannot say which
    calculation that was."""
    (calc.root / "job.XV").write_text("x\n")
    state = calc.save("stage 2 converged")
    assert state.calculation == "BDT_Au_relax"
    assert "BDT_Au_relax" not in state.note, (
        "the name rides in its own field; it must not pollute what you wrote")


def test_a_calculation_name_needing_repair_is_refused(tmp_path):
    """Nothing is normalised: silently fixing an id would decouple the
    history's name from the folder's."""
    root = tmp_path / "has spaces & punctuation!"
    root.mkdir()
    with pytest.raises(CheckpointError):
        Repo(str(root)).init(note="set up")


# ------------------------------------------------------------------ #
#  I2b — the records are checked, not trusted                         #
# ------------------------------------------------------------------ #


def test_every_state_carries_its_archive_digest(calc):
    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("with an archive")
    assert state.archive
    assert (archive_dir(calc.root, state.archive) / MANIFEST_NAME).is_file()


def test_a_state_that_archived_nothing_still_carries_a_digest(calc):
    """That is what makes "this state had no big files" and "this state's
    archive is gone" two different observations."""
    (calc.root / "only.txt").write_text(SMALL)
    state = calc.save("no big files here")
    assert state.archive


def test_a_tampered_manifest_is_refused_on_restore(calc):
    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("first")
    (calc.root / "job.XV").write_text("move away\n")
    calc.save("second")

    man = archive_dir(calc.root, state.archive) / MANIFEST_NAME
    man.write_bytes(man.read_bytes().replace(b"big.bin", b"big.bio"))
    with pytest.raises(CheckpointError) as exc:
        calc.restore(state.id, force=True)
    assert "MANIFEST" in str(exc.value) or "modified" in str(exc.value)


def test_a_hand_edited_ignore_block_is_detected_and_repaired(calc):
    gi = calc.root / ".gitignore"
    assert not calc.status().ignore_edited
    gi.write_text(gi.read_text().replace(".binsnapshots/",
                                         ".binsnapshots/\n*.XV"))
    assert calc.status().ignore_edited, (
        "an edit inside the markers must be detectable — that is what "
        "regeneration buys for a derived record (I2b)")
    (calc.root / "job.XV").write_text("x\n")
    calc.save("regenerates the block")
    assert not calc.status().ignore_edited


# ------------------------------------------------------------------ #
#  Tags                                                               #
# ------------------------------------------------------------------ #


def test_a_tag_names_a_state_and_restores_it(calc):
    (calc.root / "job.XV").write_text("good\n")
    state = calc.save("the geometry I trust")
    calc.tag("stage1-good", "geometry I trust", at=state.id)
    (calc.root / "job.XV").write_text("elsewhere\n")
    calc.save("elsewhere")

    assert calc.restore("stage1-good", force=True).id == state.id
    assert [t.name for t in calc.tags()] == ["stage1-good"]
    assert "stage1-good" in calc.states()[0].tags or any(
        "stage1-good" in s.tags for s in calc.states())


def test_nothing_tags_a_state_on_your_behalf(calc):
    """L4: the namespace is yours alone, which is what makes your own tags
    easy to see.  Stage completions used to be tagged automatically."""
    for i in range(3):
        (calc.root / "job.XV").write_text(f"{i}\n")
        calc.save(f"stage {i} converged")
    assert calc.tags() == []


# ------------------------------------------------------------------ #
#  L1 — one repository per calculation                                #
# ------------------------------------------------------------------ #


def test_a_folder_of_independent_calculations_is_refused(tmp_path):
    """One history over several calculations would rewind all of them."""
    from molbuilder.checkpoint import NestedRepoRefusedError
    root = tmp_path / "topic"
    (root / "run_a").mkdir(parents=True)
    (root / "run_b").mkdir()
    (root / "run_a" / "job.fdf").write_text("x\n")
    (root / "run_b" / "job.fdf").write_text("x\n")
    with pytest.raises(NestedRepoRefusedError):
        Repo(str(root)).init(note="set up")


def test_a_folder_that_declares_them_one_calculation_is_accepted(tmp_path):
    """The description is what says these stages are one unit of work."""
    root = tmp_path / "BDT_Au_relax"
    (root / "01_coarse").mkdir(parents=True)
    (root / "01_coarse" / "job.fdf").write_text("x\n")
    (root / "task.json").write_text('{"engine": "siesta"}')
    state = Repo(str(root)).init(note="set up")
    assert state.calculation == "BDT_Au_relax"


def test_a_repository_inside_a_repository_is_refused(tmp_path):
    from molbuilder.checkpoint import NestedRepoRefusedError
    root = tmp_path / "outer"
    inner = root / "inner"
    inner.mkdir(parents=True)
    (inner / "job.fdf").write_text("x\n")
    Repo(str(inner)).init(note="inner")
    (root / "task.json").write_text("{}")
    with pytest.raises(NestedRepoRefusedError):
        Repo(str(root)).init(note="outer")


# ------------------------------------------------------------------ #
#  I2a — a restore replays the save and consults nothing              #
# ------------------------------------------------------------------ #


def test_a_restore_ignores_the_configuration_entirely(calc, tmp_path):
    """The contract's own test for I2a: save, then change the classification
    beyond recognition -- and the restored tree must be byte-identical.

    This is what makes moving the classification safe (S1c): every archive
    already written stays restorable, because a restore replays what the save
    recorded rather than re-deriving it.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "small.txt").write_text(SMALL)
    state = calc.save("the state to come back to")
    before = {archive_key(calc.root, p): p.read_bytes()
              for p in walk_files(calc.root)}

    (calc.root / "big.bin").write_bytes(b"\x09" * 5000)
    (calc.root / "stray.bin").write_bytes(BIG)
    calc.save("moved on")

    # Change the classification beyond recognition, then delete it outright.
    cfg = calc.root / PROJECT_CONFIG_FILENAME
    cfg.write_text(json.dumps({"checkpoint": {
        "size_limit_bytes": 99_999_999,          # nothing is big any more
        "engines": {"generic": []}}}))
    calc.restore(state.id, force=True)
    after = {archive_key(calc.root, p): p.read_bytes()
             for p in walk_files(calc.root) if p.name != PROJECT_CONFIG_FILENAME}
    expected = {k: v for k, v in before.items() if k != PROJECT_CONFIG_FILENAME}
    assert after == expected, (
        "a restore re-derived what to do from the config instead of replaying "
        "what the save recorded (I2a)")


# ------------------------------------------------------------------ #
#  A1 / A2 / I2 — the archive is verified, not trusted                #
# ------------------------------------------------------------------ #


def test_a_copy_corrupted_on_the_way_to_disk_is_caught_at_save(calc,
                                                              monkeypatch):
    """§ 6: the copy is re-hashed rather than trusted.

    If the MANIFEST's checksum came from the copy alone, a copy corrupted on
    the way to disk would be *self-consistent* -- it would verify against its
    own bad checksum forever and be restored as truth.  Monkeypatching the copy
    is the only way to produce a disk fault on demand; everything either side
    of it is real.
    """
    import molbuilder.checkpoint as cp

    def corrupting_copy(src, dst, *a, **k):
        Path(dst).write_bytes(b"\x00" * 5000)
        return dst

    (calc.root / "big.bin").write_bytes(BIG)
    monkeypatch.setattr(cp.shutil, "copy2", corrupting_copy)
    with pytest.raises(CheckpointError) as exc:
        calc.save("a save that must not succeed")
    assert "corrupt" in str(exc.value).lower()


def test_a_corrupt_archive_is_refused_and_the_folder_is_untouched(calc):
    """A2: verify before mutating, in that order.

    A restore that half-completes leaves text from one state and big files from
    another -- a folder no save ever held, and nothing can diagnose it
    afterwards.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    target = calc.save("the state to come back to")
    (calc.root / "job.XV").write_text("moved on\n")
    calc.save("moved on")

    payload = archive_dir(calc.root, target.archive) / "big.bin"
    payload.write_bytes(b"\x00" * 5000)          # right size, wrong bytes

    before = {archive_key(calc.root, p): p.read_bytes()
              for p in walk_files(calc.root)}
    with pytest.raises(CheckpointError):
        calc.restore(target.id, force=True)
    after = {archive_key(calc.root, p): p.read_bytes()
             for p in walk_files(calc.root)}
    assert after == before, "the folder changed before the check refused"


def test_a_partial_archive_never_appears_at_the_published_name(calc,
                                                               monkeypatch):
    """A1: build, verify, publish -- a reader never meets a half-written one."""
    import molbuilder.checkpoint as cp

    def die(src, dst, *a, **k):
        raise OSError("disk full")

    (calc.root / "big.bin").write_bytes(BIG)
    monkeypatch.setattr(cp.shutil, "copy2", die)
    with pytest.raises(OSError):
        calc.save("this save dies mid-copy")
    published = [d for d in (calc.root / ".binsnapshots").iterdir()
                 if d.is_dir() and not d.name.endswith(".tmp")]
    for adir in published:
        assert (adir / MANIFEST_NAME).is_file(), (
            f"{adir.name} is published but has no MANIFEST -- a reader would "
            f"take it for complete")


# ------------------------------------------------------------------ #
#  § 3 — the archive is named by content                              #
# ------------------------------------------------------------------ #


def test_two_states_with_the_same_big_files_share_one_archive(calc):
    """Not a copy avoided -- the same directory, named by what it holds."""
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("first")
    (calc.root / "note.txt").write_text("only the text changed\n")
    second = calc.save("second")

    assert first.archive == second.archive
    assert len([d for d in (calc.root / ".binsnapshots").iterdir()
                if d.is_dir()]) == 2, (
        "one archive for the big file, one empty archive for the first state")


def test_changing_a_big_file_makes_a_different_archive(calc):
    """I1 made structural: content cannot change under a name."""
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    second = calc.save("second")
    assert first.archive != second.archive
    assert (archive_dir(calc.root, first.archive) / "big.bin"
            ).read_bytes() == BIG, "the older archive was modified in place"


# ------------------------------------------------------------------ #
#  S1a — the ignore file is generated, and only the block is ours     #
# ------------------------------------------------------------------ #


def test_a_users_own_ignore_entries_are_left_alone(calc):
    gi = calc.root / ".gitignore"
    gi.write_text("# mine\nnotes.private\n\n" + gi.read_text())
    (calc.root / "job.XV").write_text("x\n")
    calc.save("regenerates only the marked block")
    text = gi.read_text()
    assert "notes.private" in text, "a user's own entries are theirs"
    assert text.count("=== molbuilder checkpoint BEGIN ===") == 1


def test_the_generated_block_contains_nothing_but_archive_patterns(calc):
    """S1a's test, and no fixture can be too short for it.

    If git is told to skip something the archive does not take, that file is in
    no store -- and this catches it without any file existing at all.
    """
    from molbuilder.checkpoint import ARCHIVE_DIR
    always = calc._classification()["always_large"]
    text = (calc.root / ".gitignore").read_text()
    block = text.split("=== molbuilder checkpoint BEGIN ===")[1] \
                .split("=== molbuilder checkpoint END ===")[0]
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert line in set(always) | {ARCHIVE_DIR + "/"}, (
            f"{line!r} is ignored by git and taken by nothing")
