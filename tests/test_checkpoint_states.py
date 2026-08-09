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

import os
import subprocess
from pathlib import Path

import pytest

from molbuilder.runtime_config import (
    get_checkpoint,
    get_checkpoint_engines,
)
from molbuilder.checkpoint import (
    CheckpointError,
    DirtyWorkingTreeError,
    MANIFEST_NAME,
    NoSuchRefError,
    Repo,
    _SHA256_RE,
    archive_dir,
    archive_key,
    parse_manifest,
    walk_files,
)

BIG = b"\x01" * 5000       # over the fixture's limit
SMALL = "text\n"


@pytest.fixture()
def calc(tmp_path, checkpoint_config):
    """A calculation folder whose store is decided by SIZE, not by name.

    The classification is set in the server-wide config (S1c) -- the folder
    itself carries none, which is what the rule says and what the walk below
    would otherwise have to make an exception for.
    """
    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    repo = Repo(str(root))
    repo.init(note="set up")
    return repo


def _published(repo):
    """The archives on disk -- staging directories are not archives.

    A publisher stages under `<digest>.<random>` and renames into place, so a
    name that is not a bare sha256 is somebody mid-write.
    """
    return [d for d in (repo.root / ".binsnapshots").iterdir()
            if d.is_dir() and _SHA256_RE.match(d.name)]


def _as_of_the_state(path, repo):
    """Give `path` the timestamp of the state the folder stands at.

    § 7.2 defines the cheap read's blind spot against the STATE's clock -- "a
    rewrite to exactly the same size inside the same second as the save" -- so
    a test of it has to set that relationship rather than hope for it.  Left to
    the wall clock these pass or fail on where the save's sub-second fraction
    happened to land, which is a coin toss dressed as an assertion.
    """
    from molbuilder.checkpoint import _epoch_of
    at = _epoch_of(repo.standing_at().at)
    os.utime(path, (at, at))


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
    assert status.standing_at.id == first.id
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
    """L7's sibling: big files are outside git, so the MANIFEST is their record.

    Asked exactly, because that is the guarantee -- see the two tests below for
    which question gets asked when.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    assert "big.bin" in calc.status(deep=True).changed


def test_a_resized_big_file_is_seen_by_the_cheap_read_too(calc):
    """The display does not need content to answer most of the time.

    A different size IS an answer, so the common cases -- a run that grew a
    density matrix, a file deleted, a file appearing -- all show without
    anything being read.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(BIG + b"more")
    assert "big.bin" in calc.status().changed


def test_the_cheap_read_may_miss_a_same_size_rewrite_and_that_is_the_deal(calc):
    """The accepted blind spot, asserted so nobody "fixes" it by hashing.

    A state's timestamp is whole seconds and a file's is not, so a same-size
    rewrite inside that second is invisible to the display.  That costs
    NOTHING: no byte moves on a status call, and the next real operation
    compares content.  Paying for certainty here would mean reading gigabytes
    every time a directory is opened.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)      # same size...
    _as_of_the_state(calc.root / "big.bin", calc)             # ...same second
    assert calc.status().clean, (
        "if this starts failing, the cheap read began hashing -- check that "
        "the display is not paying for exactness it does not need")


def test_a_restore_still_refuses_a_same_size_rewrite(calc):
    """And this is why missing it in the display is safe.

    The moment the folder is about to change, the question is asked exactly.
    A user who was told "nothing unsaved" a second ago is still stopped here,
    because the operation checks content and the badge never did.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("first")
    (calc.root / "job.XV").write_text("moved on\n")
    calc.save("moved on")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)      # same size...
    _as_of_the_state(calc.root / "big.bin", calc)             # ...same second

    assert calc.status().clean, "precondition: the cheap read misses it"
    with pytest.raises(DirtyWorkingTreeError) as exc:
        calc.restore(first.id)
    assert "big.bin" in str(exc.value), (
        "the operation must check content, whatever the display said")


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
    # `A or any(...)` made the first clause unfalsifiable and asked only whether
    # SOME state carries the tag.  The claim is that the tagged state carries it.
    tagged = [s for s in calc.states() if "stage1-good" in s.tags]
    assert [s.id for s in tagged] == [state.id], (
        "the tag must sit on the state it was applied to, and on no other")


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


def test_a_restore_ignores_the_configuration_entirely(calc, checkpoint_config):
    """The contract's own test for I2a: save, then change the classification
    beyond recognition -- and the restored tree must be byte-identical.

    This is what makes the classification's single home safe (S1c): every
    archive already written stays restorable, because a restore replays what
    the save recorded rather than re-deriving it.

    The comparison is now over the WHOLE tree with no exclusions.  It used to
    exempt the config file, because the config lived inside the folder being
    compared -- an exemption that existed only to hide the rule S1c is about.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "small.txt").write_text(SMALL)
    state = calc.save("the state to come back to")
    before = {archive_key(calc.root, p): p.read_bytes()
              for p in walk_files(calc.root)}

    (calc.root / "big.bin").write_bytes(b"\x09" * 5000)
    (calc.root / "stray.bin").write_bytes(BIG)
    calc.save("moved on")

    # Change the classification beyond recognition: nothing is big any more.
    checkpoint_config(size_limit_bytes=99_999_999, engines={"generic": []})
    calc.restore(state.id, force=True)
    after = {archive_key(calc.root, p): p.read_bytes()
             for p in walk_files(calc.root)}
    assert after == before, (
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
    for adir in _published(calc):
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
    assert len(_published(calc)) == 2, (
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
    always = calc.classification()["always_large"]
    text = (calc.root / ".gitignore").read_text()
    block = text.split("=== molbuilder checkpoint BEGIN ===")[1] \
                .split("=== molbuilder checkpoint END ===")[0]
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert line in set(always) | {ARCHIVE_DIR + "/"}, (
            f"{line!r} is ignored by git and taken by nothing")


# ================================================================== #
#  A realistic folder, not a two-file fixture                        #
#                                                                    #
#  Everything above proves a rule on the smallest case that can show #
#  it.  A calculation folder is a nested staged tree with several    #
#  stages, attempts, shared pseudopotentials linked rather than      #
#  copied, and big files at depth -- and several rules only have     #
#  teeth there (L2, S1's symlink carve-out, the dedup property).     #
# ================================================================== #


def _staged(root, stages=("01_coarse", "02_tight"), attempts=2, dm=BIG):
    """A calculation folder shaped like `project-layout.md` § 1.1's nested form."""
    (root / "task.json").write_text('{"engine": "siesta"}')
    (root / "Au.psml").write_text("<pseudo/>\n")
    (root / "job.fdf.template").write_text("SystemLabel job\n")
    for stage in stages:
        sdir = root / stage
        sdir.mkdir(exist_ok=True)
        (sdir / "job.fdf").write_text(f"# {stage}\n")
        # a stage LINKS the shared pseudopotential rather than copying it
        link = sdir / "Au.psml"
        if not link.exists():
            link.symlink_to("../Au.psml")
        for n in range(attempts):
            run = sdir / f"run-{n}"
            run.mkdir(exist_ok=True)
            (run / "run.json").write_text('{"mode": "local"}')
            (run / "job.out").write_text(f"{stage} attempt {n}\n")
            (run / "job.XV").write_text(f"{stage}-{n} coords\n")
            (run / "job.DM").write_bytes(dm + stage.encode() + str(n).encode())
    return root


@pytest.fixture()
def staged(tmp_path, checkpoint_config):
    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    _staged(root)
    repo = Repo(str(root))
    repo.init(note="a staged calculation, two stages, two attempts each")
    return repo


def test_s1_holds_over_a_real_staged_tree(staged):
    """L2 and S1 together, at depth, with symlinks in the tree.

    A flat fixture cannot fail this: the defect it guards is that a gitignore
    pattern with no slash matches at EVERY level while a top-level-only walk
    matches at one, so a nested big file falls between the two stores.
    """
    state = staged.standing_at()
    tracked, archived = _tracked(staged), _archived(staged, state)
    for path in walk_files(staged.root):
        key = archive_key(staged.root, path)
        assert (key in tracked) != (key in archived), (
            f"S1 violated at depth for {key!r}")
    assert len([k for k in archived if k.endswith("job.DM")]) == 4, (
        "every stage's every attempt must be archived, not just the top one")
    assert "01_coarse/run-0/job.XV" in tracked, "small files at depth go to git"


def test_symlinks_are_not_stored_and_survive_a_restore(staged):
    """S1's carve-out: a link has no content of its own.

    Storing it as a regular file would both duplicate the target and restore a
    real file where a link belongs -- and a saved tree is full of links,
    because a stage links the shared pseudopotentials rather than copying them.
    """
    state = staged.standing_at()
    archived = _archived(staged, state)
    assert "01_coarse/Au.psml" not in archived
    link = staged.root / "01_coarse" / "Au.psml"
    assert link.is_symlink()

    (staged.root / "01_coarse" / "run-0" / "job.DM").write_bytes(b"\x09" * 5000)
    staged.save("moved on")
    staged.restore(state.id, force=True)
    assert link.is_symlink(), "a restore turned a link into a real file"
    assert link.resolve() == (staged.root / "Au.psml").resolve()


def test_a_restore_recreates_directories_that_had_been_removed(staged):
    """The archive's keys are paths, so a restore must rebuild the tree."""
    state = staged.standing_at()
    import shutil as _sh
    (staged.root / "03_extra").mkdir()
    (staged.root / "03_extra" / "job.DM").write_bytes(BIG)
    staged.save("a third stage")
    _sh.rmtree(staged.root / "01_coarse")
    staged.save("removed the first stage entirely")

    staged.restore(state.id, force=True)
    assert (staged.root / "01_coarse" / "run-1" / "job.DM").is_file()
    assert not (staged.root / "03_extra").exists()
    assert staged.status().clean


def test_identical_content_is_stored_once(staged):
    """§ 12's *Disk cost* property: a second save of an unchanged 2 GB binary
    costs a directory entry.  Untested until now, and it is the reason the
    archive is content-addressed rather than merely checksummed."""
    first = staged.standing_at()
    (staged.root / "task.json").write_text('{"engine": "siesta", "v": 2}')
    second = staged.save("only the description changed")

    assert first.archive == second.archive, (
        "the big files did not change, so it is the same archive")

    # now change ONE of the four, and the other three must be shared by inode
    (staged.root / "01_coarse" / "run-0" / "job.DM").write_bytes(b"\x07" * 5000)
    third = staged.save("one attempt re-run")
    assert third.archive != first.archive

    old_dir = archive_dir(staged.root, first.archive)
    new_dir = archive_dir(staged.root, third.archive)
    shared = [k for k in _archived(staged, third)
              if (old_dir / k).is_file()
              and (old_dir / k).stat().st_ino == (new_dir / k).stat().st_ino]
    assert len(shared) == 3, (
        f"3 unchanged big files should be hard-linked, not copied; {len(shared)} were")


# ------------------------------------------------------------------ #
#  S7 — a file that changes category leaves the store it came from    #
# ------------------------------------------------------------------ #


def test_a_file_that_grows_past_the_limit_leaves_git(calc):
    """The contract calls this routine once the gate is a size, because files
    grow: an .EIG at 8 MB last save and 12 MB this one crosses by itself, with
    nobody deciding anything.  Tracked *and* archived is S1's other losing
    branch -- a large blob committed on every save from then on."""
    small = calc.root / "grows.bin"
    small.write_bytes(b"x" * 100)
    first = calc.save("small enough for git")
    assert "grows.bin" in _tracked(calc)
    assert "grows.bin" not in _archived(calc, first)

    small.write_bytes(BIG)
    second = calc.save("now it is big")
    assert "grows.bin" in _archived(calc, second)
    assert "grows.bin" not in _tracked(calc), (
        "it stayed in git as well -- a blob on every save from now on (S7)")


def test_a_file_that_shrinks_below_the_limit_leaves_the_archive(calc):
    """The same rule in the other direction, which is the half people forget."""
    big = calc.root / "shrinks.bin"
    big.write_bytes(BIG)
    first = calc.save("big")
    assert "shrinks.bin" in _archived(calc, first)

    big.write_bytes(b"x" * 100)
    second = calc.save("small now")
    assert "shrinks.bin" in _tracked(calc)
    assert "shrinks.bin" not in _archived(calc, second)


# ------------------------------------------------------------------ #
#  A history with real shape                                          #
# ------------------------------------------------------------------ #


def test_a_deep_forked_history_lists_children_before_parents(staged):
    """`states()` is ordered topologically, not by date.

    Several states can share a second -- a scripted sweep saves faster than the
    clock ticks -- and a date tie would print a child above its own parent.
    """
    made = [staged.standing_at()]
    for i in range(3):                                  # a trunk
        (staged.root / "task.json").write_text(f'{{"v": {i}}}')
        made.append(staged.save(f"trunk {i}"))
    forks = []
    for i in range(3):                                  # three from one point
        staged.restore(made[1].id, force=True)
        (staged.root / "task.json").write_text(f'{{"fork": {i}}}')
        forks.append(staged.save(f"fork {i}"))

    listed = [s.id for s in staged.states()]
    assert len(listed) == len(made) + len(forks)
    position = {sid: n for n, sid in enumerate(listed)}
    for state in staged.states():
        if state.parent in position:
            assert position[state.id] < position[state.parent], (
                "a state was listed after the state it came from")
    assert {f.parent for f in forks} == {made[1].id}


def test_i2c_the_warning_names_a_file_the_classification_no_longer_matches(
        calc, checkpoint_config):
    """I2c's own stated test.

    A `.DM` archived while `*.DM` was classified big; the classification is
    narrowed; the file is modified and an earlier state restored.  The warning
    must still name it, because the MANIFEST -- not the classification -- is
    what says the restore will write over it.
    """
    checkpoint_config(size_limit_bytes=99_999_999,
                      engines={"generic": ["*.DM"]})
    (calc.root / "job.DM").write_bytes(b"\x01" * 100)
    first = calc.save("with a .DM")
    (calc.root / "job.XV").write_text("moved on\n")
    calc.save("moved on")

    checkpoint_config(size_limit_bytes=99_999_999, engines={"generic": []})
    (calc.root / "job.DM").write_bytes(b"\x02" * 100)

    with pytest.raises(DirtyWorkingTreeError) as exc:
        calc.restore(first.id)
    assert "job.DM" in str(exc.value), (
        "the warning omitted a file the restore will overwrite (I2c)")


# ------------------------------------------------------------------ #
#  S8 — bare git, and what molbuilder can honestly say about it       #
# ------------------------------------------------------------------ #


def test_a_folder_bare_git_pulled_out_of_step_is_not_acted_on(calc):
    """§ 2.0 tells you not to; nothing stops you; this is what happens.

    `git checkout` an earlier state and the text rewinds while every big file
    stays where it was -- inputs from one state, files from another.  The
    operation that would act on it checks CONTENT (§ 7.2) and refuses, naming
    the files that differ.
    """
    (calc.root / "job.fdf").write_text("stage 1 inputs\n")
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("stage 1")
    (calc.root / "job.fdf").write_text("stage 2 inputs\n")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    calc.save("stage 2")

    subprocess.run(["git", "checkout", "--quiet", first.id],
                   cwd=calc.path, check=True)
    assert (calc.root / "job.fdf").read_text() == "stage 1 inputs\n"
    assert (calc.root / "big.bin").read_bytes()[0] == 2, (
        "precondition: git left the big file alone, which is the whole hazard")

    with pytest.raises(DirtyWorkingTreeError) as exc:
        calc.restore(first.id)
    assert "big.bin" in str(exc.value), (
        "it must name the file that differs, not merely refuse")


def test_saving_such_a_folder_records_what_is_there_and_restores_correctly(calc):
    """Not a refusal, deliberately.

    § 1 promises to save what is on disk, not to adjudicate how it got there --
    and the state that results is internally consistent, which the restore
    proves.
    """
    (calc.root / "job.fdf").write_text("stage 1 inputs\n")
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("stage 1")
    (calc.root / "job.fdf").write_text("stage 2 inputs\n")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    calc.save("stage 2")
    subprocess.run(["git", "checkout", "--quiet", first.id],
                   cwd=calc.path, check=True)

    mixed = calc.save("what bare git left behind")
    assert mixed is not None
    (calc.root / "big.bin").write_bytes(b"\x09" * 5000)
    calc.save("moved on again")

    calc.restore(mixed.id, force=True)
    assert (calc.root / "job.fdf").read_text() == "stage 1 inputs\n"
    assert (calc.root / "big.bin").read_bytes()[0] == 2
    assert calc.status(deep=True).clean


# ------------------------------------------------------------------ #
#  S9 — two saves of one folder cannot corrupt each other             #
# ------------------------------------------------------------------ #


def test_concurrent_saves_of_the_same_content_cannot_corrupt_the_archive(calc):
    """Content addressing does this rather than a lock.

    Two savers of the same big files compute the same digest and publish to the
    same path with the same bytes -- the race has no wrong outcome to reach.
    What must hold afterwards is that every archive present verifies (I2).
    """
    import threading
    from molbuilder.checkpoint import big_files, publish_archive, verify_archive

    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "other.bin").write_bytes(b"\x05" * 6000)
    cls = calc.classification()
    big = big_files(calc.root, int(cls["size_limit_bytes"]),
                    cls["always_large"])

    results, errors = [], []

    def publish():
        try:
            results.append(publish_archive(calc.root, big))
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=publish) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"a concurrent publish failed: {errors}"
    assert len(set(results)) == 1, "same content must mean one archive name"
    verify_archive(calc.root, results[0])             # raises if damaged

    for adir in _published(calc):
        verify_archive(calc.root, adir.name)


# ------------------------------------------------------------------ #
#  § 3 — "exactly one store" is true of the object database too       #
# ------------------------------------------------------------------ #


def _loose_blob_bytes(repo):
    """Total size of every blob git holds, reachable or not.

    `git add` writes a blob the moment it stages a file, and `rm --cached`
    only drops the index entry -- so a big file that reached `add` leaves its
    bytes behind whatever the index says afterwards.  Asking git what it holds
    is the only question that catches that; `ls-files` cannot.
    """
    out = subprocess.run(
        ["git", "cat-file", "--batch-all-objects",
         "--batch-check=%(objecttype) %(objectsize)"],
        cwd=repo.path, capture_output=True, text=True, check=True).stdout
    total = 0
    for line in out.splitlines():
        kind, _, size = line.partition(" ")
        if kind == "blob":
            total += int(size)
    return total


def test_a_big_file_never_enters_gits_object_store(calc):
    """S1's losing branch, in the one place a walk of the tree cannot see it.

    A file big by SIZE cannot be named in `.gitignore`, so it used to be handed
    to `git add -A` and only unstaged afterwards.  `add` hashes, compresses and
    WRITES: the blob landed in `.git/objects` and stayed there -- re-created on
    every save, for a file § 3 says never goes into git.  `git ls-files` shows
    nothing wrong, which is why this asks the object database instead.
    """
    payload = b"\x07" * 200_000                  # 200 KB, far over the 1 KB limit
    (calc.root / "big.bin").write_bytes(payload)
    (calc.root / "small.txt").write_text(SMALL)
    calc.save("one big file, one small")

    assert "big.bin" not in _tracked(calc), "precondition: it is not tracked"
    assert _loose_blob_bytes(calc) < len(payload), (
        "the big file's bytes are in .git/objects; unstaging it does not "
        "remove the blob `git add` already wrote (§ 3, S1)")


def test_repeated_saves_do_not_grow_git_by_the_big_file(calc):
    """The cost was paid AGAIN on every save, which is what made it fatal.

    Ten saves of an unchanged density matrix meant ten copies in the object
    store of a file the archive already holds once.
    """
    (calc.root / "big.bin").write_bytes(b"\x07" * 200_000)
    calc.save("first")
    baseline = _loose_blob_bytes(calc)
    for n in range(3):
        (calc.root / "big.bin").write_bytes(bytes([n]) * 200_000)
        calc.save(f"rewrite {n}")
    assert _loose_blob_bytes(calc) - baseline < 200_000, (
        "each save added another copy of the big file to git's object store")


# ------------------------------------------------------------------ #
#  I2b — damage is named, not absorbed                                #
# ------------------------------------------------------------------ #


def test_a_lost_archive_is_named_rather_than_read_as_unsaved_work(calc):
    """Two outcomes, not three: it matches, or it is refused (I2b).

    With the archive gone there is no record of what the state held, so every
    archived file looks like something you just created.  Reporting "1 unsaved"
    is the exact opposite of the truth -- those files were saved and are now
    unreachable -- and it invites a save that would paper over the loss.
    """
    import shutil as _sh
    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("with an archive")
    _sh.rmtree(archive_dir(calc.root, state.archive))

    with pytest.raises(CheckpointError) as exc:
        calc.status()
    message = str(exc.value)
    assert "missing" in message or "lost" in message
    assert state.short in message, "it must say WHICH state's archive is gone"


def test_a_tampered_manifest_is_caught_by_the_cheap_read_too(calc):
    """The digest check costs one small file, so the display can afford it.

    A record that does not hash to the name it is stored under is damage
    wherever it is noticed, and noticing it only at restore time means the
    panel shows a healthy folder until the moment somebody acts on it.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("with an archive")
    man = archive_dir(calc.root, state.archive) / MANIFEST_NAME
    man.write_bytes(man.read_bytes().replace(b"big.bin", b"big.bio"))

    with pytest.raises(CheckpointError) as exc:
        calc.status()
    assert "modified" in str(exc.value) or MANIFEST_NAME in str(exc.value)


# ------------------------------------------------------------------ #
#  Paths git does not hand back verbatim                              #
# ------------------------------------------------------------------ #


def test_a_big_file_whose_path_has_a_space_still_reads_saved(calc):
    """`git status --porcelain` QUOTES a path containing a space.

    Read as a bare path that arrives as `"01 coarse/job.DM"` -- quotes and all
    -- which matches no archive key, so the guard that keeps git quiet about
    big files never fired: the file read as unsaved immediately after being
    saved, forever, and every restore demanded --force.
    """
    stage = calc.root / "01 coarse"
    stage.mkdir()
    (stage / "job.DM").write_bytes(BIG)
    state = calc.save("a stage directory with a space in its name")

    assert "01 coarse/job.DM" in _archived(calc, state)
    status = calc.status()
    assert status.clean, (
        f"a saved big file read as unsaved: {status.unsaved()}")


def test_a_rename_is_reported_as_the_two_files_it_touched(calc):
    """git reports a rename as one record naming BOTH paths.

    Taken as a single name it becomes the string `old -> new`, which names
    nothing on disk: it cannot be found, saved or restored, and the person
    reading the warning is shown a path that does not exist.
    """
    (calc.root / "before.txt").write_text(SMALL)
    calc.save("with a file to rename")
    subprocess.run(["git", "mv", "before.txt", "after.txt"],
                   cwd=calc.path, check=True)

    status = calc.status()
    assert "before.txt" in status.deleted
    assert "after.txt" in status.added
    assert not any("->" in name for name in status.unsaved()), (
        "a rename was reported as one path naming two files")


def test_a_name_the_archive_cannot_carry_is_refused_with_the_way_out(calc):
    """S1 cannot be met for this file, so the refusal has to be usable.

    The MANIFEST is ASCII, so a big file named outside it cannot be archived.
    That is a real limit; what makes it survivable is saying which file it is
    and what to do -- rename it, or raise the limit so git takes it instead.
    """
    (calc.root / "résumé.bin").write_bytes(BIG)
    with pytest.raises(CheckpointError) as exc:
        calc.save("a file the record cannot carry")
    message = str(exc.value)
    assert "résumé.bin" in message, "it must name the file"
    assert "rename" in message.lower() and "size_limit_bytes" in message, (
        "a refusal with no way out is a dead end")


def test_a_restore_keeps_a_tracked_file_whose_path_has_a_space(calc):
    """The same quoting hazard as above, in the place where BYTES MOVE.

    A restore removes what the target did not hold, and the set it keeps came
    from `git ls-files` split on whitespace -- so `01 coarse/job.fdf` arrived as
    `01` and `coarse/job.fdf`, the real key matched neither, and the file was
    deleted as a leftover.  Nothing put it back: only *archived* files are
    copied afterwards, and this one is small enough for git.

    So the file the target state holds is gone from the folder while the state
    still claims it -- silent loss during the one operation § 1's promise is
    entirely about.
    """
    stage = calc.root / "01 coarse"
    stage.mkdir()
    (stage / "job.fdf").write_text("SystemLabel job\n")
    (stage / "big.bin").write_bytes(BIG)
    target = calc.save("a stage whose directory name has a space")

    (calc.root / "later.txt").write_text(SMALL)
    calc.save("moved on")
    calc.restore(target.id, force=True)

    assert (stage / "job.fdf").is_file(), (
        "a restore deleted a tracked file the target state holds")
    assert (stage / "job.fdf").read_text() == "SystemLabel job\n"
    assert (stage / "big.bin").read_bytes() == BIG
    assert not (calc.root / "later.txt").exists()
    assert calc.status(deep=True).clean


def test_an_unreadable_file_stops_the_save_rather_than_being_skipped(calc):
    """S1 exempts nothing, so "could not read it" is not a reason to omit it.

    Silently leaving it out produces a snapshot of most of a folder, which the
    contract says is not a snapshot of the folder -- and the omission would only
    be discovered by a restore that did not bring the file back.
    """
    victim = calc.root / "locked.bin"
    victim.write_bytes(BIG)
    os.chmod(victim, 0o000)
    try:
        with pytest.raises(CheckpointError) as exc:
            calc.save("a file nobody can read")
        assert "locked.bin" in str(exc.value), "it must name the file"
    finally:
        os.chmod(victim, 0o644)


# ================================================================== #
#  § 13.1 — the fixture is GENERATED FROM THE CONFIG                 #
#                                                                    #
#  "A walk can only judge files the fixture created, and no fixture  #
#  ever created a `.MD`."  Every test above uses an empty pattern    #
#  list, so until now no test had ever exercised a real always-large #
#  family at all -- `*.TBT.AVTRANS_*` was matched by nothing,        #
#  anywhere.  These walk the shipped classification instead of a     #
#  hand-written list, so adding a pattern extends them and adding an #
#  engine gets a suite for free.                                     #
# ================================================================== #


def _a_name_matching(pattern, n):
    """A concrete filename that the glob names.

    `*.DM` -> `s0.DM`; `*.TBT.AVTRANS_*` -> `s1.TBT.AVTRANS_s1`.  Derived from
    the pattern rather than listed beside it, because a list of example names is
    a second copy of the classification (§ 13.3).
    """
    return pattern.replace("*", f"s{n}").replace("?", "q")


@pytest.fixture()
def shipped(tmp_path, checkpoint_config):
    """A folder under the SHIPPED engine patterns and a tiny size limit.

    Omitting `engines` leaves the real ones in place, so the patterns under test
    are the ones molbuilder actually ships.  The limit is small only so the
    unlisted-file half runs quickly.
    """
    checkpoint_config(size_limit_bytes=1024)
    return tmp_path


def test_every_pattern_the_config_names_is_archived(shipped):
    """For every engine, for every pattern it names: a file, and it is stored.

    The files are made deliberately **small** -- ten bytes, far under the limit
    -- so only the pattern can send them to the archive.  If a family stopped
    being matched, these land in git and the assertion fires; that is exactly
    how `*.MD` should have been caught.
    """
    for engine in get_checkpoint_engines():
        _one_engines_families_are_archived(shipped, engine)


def _one_engines_families_are_archived(shipped, engine):
    patterns = get_checkpoint(engine)["always_large"]
    root = shipped / f"calc_{engine}"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    made = []
    for n, pattern in enumerate(patterns):
        name = _a_name_matching(pattern, n)
        (root / name).write_bytes(b"tiny bytes")
        made.append((pattern, name))

    repo = Repo(str(root))
    state = repo.init(engine=engine, note="every configured family, one file")
    archived, tracked = _archived(repo, state), _tracked(repo)

    for pattern, name in made:
        assert name in archived, (
            f"{engine}: {name!r} matches the configured family {pattern!r} and "
            f"must be archived, but it is not in the MANIFEST")
        assert name not in tracked, (
            f"{engine}: {name!r} is archived AND tracked -- S1's other losing "
            f"branch, a blob committed on every save from now on")
    if not patterns:
        assert engine == "generic", (
            "only `generic` may name no family; § 4 says the others name the "
            "ones that are always large")


def test_the_size_gate_decides_for_a_file_no_pattern_names(shipped):
    """S1b, per engine: an unlisted file either side of the limit.

    No pattern is involved in either direction -- this is the measurement, and
    it must give the same answer whichever engine is named, because an engine
    entry may only let a family *skip* the measuring.
    """
    for engine in get_checkpoint_engines():
        _the_gate_decides(shipped, engine)


def _the_gate_decides(shipped, engine):
    root = shipped / f"gate_{engine}"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    (root / "unlisted_over.xyz").write_bytes(b"\x01" * 4096)     # over 1 KB
    (root / "unlisted_under.xyz").write_bytes(b"\x01" * 10)      # under

    repo = Repo(str(root))
    state = repo.init(engine=engine, note="one either side of the limit")
    archived, tracked = _archived(repo, state), _tracked(repo)

    assert "unlisted_over.xyz" in archived and "unlisted_over.xyz" not in tracked
    assert "unlisted_under.xyz" in tracked and "unlisted_under.xyz" not in archived


def test_the_generated_ignore_block_matches_the_engines_patterns(shipped):
    """S1a per engine: git is told to skip exactly what the archive takes.

    The block is compared against the configuration, not against a list written
    here -- so a pattern that reaches `.gitignore` without reaching the archive
    fails without any file existing.
    """
    for engine in get_checkpoint_engines():
        _the_block_matches(shipped, engine)


def _the_block_matches(shipped, engine):
    from molbuilder.checkpoint import ARCHIVE_DIR
    root = shipped / f"ignore_{engine}"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    repo = Repo(str(root))
    repo.init(engine=engine, note="set up")

    allowed = set(get_checkpoint(engine)["always_large"]) | {ARCHIVE_DIR + "/"}
    text = (root / ".gitignore").read_text()
    block = text.split("=== molbuilder checkpoint BEGIN ===")[1] \
                .split("=== molbuilder checkpoint END ===")[0]
    for line in block.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert line in allowed, (
                f"{engine}: {line!r} is ignored by git and taken by nothing")


# ------------------------------------------------------------------ #
#  A5's three shapes name the FILE, not the store it happened to be in #
# ------------------------------------------------------------------ #


def test_a_file_that_grows_past_the_limit_reads_changed_not_added(calc):
    """A5: *added* means the state held no file at this path — in either store.

    `job.EIG` at 8 MB lives in git; it grows to 12 MB and belongs in the
    archive.  Looking only at the archive record of the standing state finds
    nothing and calls it "added" — but the state had that file all along, and
    what actually happened to it is that its contents changed.
    """
    grows = calc.root / "job.EIG"
    grows.write_bytes(b"x" * 100)
    calc.save("small enough for git")
    assert "job.EIG" in _tracked(calc), "precondition: the state holds it in git"

    grows.write_bytes(BIG)                       # now over the limit
    status = calc.status()
    assert "job.EIG" in status.changed, (
        "the state held this file; growing past the limit changed it, it did "
        "not create it")
    assert "job.EIG" not in status.added


def test_a_file_that_only_gets_reclassified_is_not_unsaved(calc, checkpoint_config):
    """Moving the limit must not make an untouched folder read as unsaved.

    Nothing was written.  The state holds those exact bytes and a restore gives
    them back, so there is nothing at risk — and a warning that fires when
    nothing is wrong is how § 7.2 says people learn to ignore the real one.
    """
    steady = calc.root / "job.EIG"
    steady.write_bytes(b"x" * 2000)              # over 1024: archived
    calc.save("archived at the current limit")
    assert calc.status().clean

    checkpoint_config(size_limit_bytes=99_999_999, engines={"generic": []})
    status = calc.status()                        # nothing on disk was touched
    assert status.clean, (
        f"moving the size limit made an untouched folder look unsaved: "
        f"{status.unsaved()}")


def test_a_tracked_file_the_config_reclassifies_leaves_git_on_the_next_save(
        calc, checkpoint_config):
    """S7 reached by a CONFIG change, and the path a git flag silently guards.

    `git check-ignore` consults the index, so a tracked file is reported as NOT
    ignored — which is what puts it on the save's exclusion list, which is what
    keeps `git add` from writing its blob.  Adding `--no-index` there would call
    it ignored, drop it from the list, and hand it straight back to `add`.
    """
    victim = calc.root / "job.DM"
    victim.write_bytes(b"\x01" * 200)             # under the limit: git takes it
    calc.save("small, so git holds it")
    assert "job.DM" in _tracked(calc)
    before = _loose_blob_bytes(calc)

    checkpoint_config(size_limit_bytes=1024, engines={"generic": ["*.DM"]})
    victim.write_bytes(b"\x02" * 200_000)
    state = calc.save("the classification now calls .DM always-large")

    assert "job.DM" in _archived(calc, state), "it must reach the archive"
    assert "job.DM" not in _tracked(calc), "and leave git (S7)"
    assert _loose_blob_bytes(calc) - before < 200_000, (
        "its blob was written into git on the way past")
    assert calc.status().clean


# ------------------------------------------------------------------ #
#  I3 — restore is the only operation that changes a file you made    #
# ------------------------------------------------------------------ #


def test_only_restore_changes_a_file_you_made(calc):
    """Saving reads your files; listing and tagging touch only the history.

    The exception is named by WHAT may be written -- the generated `.gitignore`
    and the two stores -- rather than by which verb writes it.  Excepting a verb
    would pass a save that quietly rewrote an input.
    """
    (calc.root / "job.XV").write_text("coords\n")
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("something to look at")

    def mine():
        return {archive_key(calc.root, p): p.read_bytes()
                for p in walk_files(calc.root)
                if archive_key(calc.root, p) != ".gitignore"}

    before = mine()
    calc.status()
    calc.status(deep=True)
    calc.states()
    calc.standing_at()
    calc.tags()
    calc.classification()
    calc.calculation()
    calc.tag("a-name", "why it matters")
    calc.save("a second save reads, and writes only its own files")
    assert mine() == before, "an operation other than restore changed a file"
