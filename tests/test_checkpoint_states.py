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
    CalculationNameError,
    CheckpointError,
    DirtyWorkingTreeError,
    MANIFEST_NAME,
    NoSuchRefError,
    Repo,
    _SHA256_RE,
    archive_dir,
    parse_manifest,
)

BIG = b"\x01" * 5000       # over the fixture's limit
SMALL = "text\n"


def walk_independently(root: Path):
    """Every regular file, found WITHOUT the module's own walk.

    **S1's test bullet says this in so many words:** *"The walk must not reuse
    the same skip rule the classification uses, or it can only ever agree with
    it."*  Importing `walk_files` and iterating it is exactly that reuse --
    `big_files` is ``[p for p in walk_files(root) if ...]``, so a skip added
    inside it vanishes from **both** sides of an S1 assertion at once and a
    file in no store reads as compliant.  That is the `*.MD` failure one level
    up: § 13.1 fixed *a walk can only judge files the fixture created*, and
    this fixes *a walk can only judge files the classification looked at*.

    So the exclusions are written here, literally, and they are the whole of
    S1's list: the two stores, which cannot contain themselves.  Symlinks are
    skipped because S1 carves them out by name -- *"regular is load-bearing"*.
    Keys are derived with `relative_to` rather than `archive_key` for the same
    independence reason.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", ".binsnapshots")]
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path


def key_of(root: Path, path: Path) -> str:
    """The archive key, computed without the module that computes archive keys."""
    return path.relative_to(root).as_posix()


def tree_bytes(root: Path) -> dict:
    """Every file you made, by key -- for before/after comparisons."""
    return {key_of(root, p): p.read_bytes() for p in walk_independently(root)}


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

    § 7.2 measures the cheap read against the STATE's clock, not the wall
    clock, so a test of the blind spot's **near edge** has to set that
    relationship rather than hope for it.  Left to the wall clock these pass or
    fail on where the save's sub-second fraction happened to land, which is a
    coin toss dressed as an assertion.

    Only the near edge needs this.  The far edge -- a timestamp *older* than
    the state -- is reached by ageing the file outright, since any amount of
    "older" does, and its test says so.
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
    for path in walk_independently(calc.root):
        key = key_of(calc.root, path)
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
    """The accepted blind spot at its NEAR edge, asserted so nobody "fixes" it
    by hashing.

    A state's timestamp is whole seconds and a file's is not, so a same-size
    rewrite inside that second is invisible to the display.  That costs
    NOTHING: no byte moves on a status call, and the next real operation
    compares content.  Paying for certainty here would mean reading gigabytes
    every time a directory is opened.

    **This is one edge of the blind spot, not the whole of it** -- the two tests
    below hold the other edge and the door § 7.2 closes.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)      # same size...
    _as_of_the_state(calc.root / "big.bin", calc)             # ...same second
    assert calc.status().clean, (
        "if this starts failing, the cheap read began hashing -- check that "
        "the display is not paying for exactness it does not need")


def test_the_blind_spot_reaches_a_timestamp_older_than_the_state(calc):
    """§ 7.2's real width, which the test above understates.

    That test pins a file dated *at* the state, so it passes for a blind spot
    one second wide **and** for one that is unbounded in the other direction.
    The comparison is *newer than the state*, so a file dated BEFORE the state
    slips through however old it is -- and the contract now says so rather than
    calling it "a rewrite inside the same second".

    **Not hypothetical.**  § 2 sends you to `rsync` for moving a calculation
    between machines, and `rsync -a` -- like `cp -p` and `tar -x` -- carries the
    source's timestamp along with the bytes.  So a folder can receive a
    different density matrix of the same size, dated an hour before the state it
    lands in, and the badge says saved.

    Both halves are asserted: the display misses it, and the exact read does
    not.  Without the second half this would pass for a hole rather than a
    blind spot.
    """
    import time

    big = calc.root / "big.bin"
    big.write_bytes(BIG)
    calc.save("first")

    big.write_bytes(b"\x02" * 5000)                  # same size, other bytes...
    an_hour_ago = time.time() - 3600                 # ...dated BEFORE the state
    os.utime(big, (an_hour_ago, an_hour_ago))

    assert calc.status().clean, (
        "the cheap read either began hashing, or started comparing timestamps "
        "for equality -- and equality reports every restored folder as unsaved, "
        "which is the false alarm § 7.2 exists to prevent")
    assert "big.bin" in calc.status(deep=True).changed, (
        "the exact read must still see it; a blind spot the operations share "
        "is not a blind spot, it is a hole")


def test_a_rerun_that_writes_identical_bytes_still_reads_clean(calc):
    """The door § 7.2 closes, held shut.

    The tempting repair for the test above is to compare each big file against
    the archive's own copy rather than against the state's clock: one `copy2`
    makes both, so their timestamps agree by construction, and an `rsync -a`
    arrival would be caught exactly.

    **Identical content is stored once (§ 12) is what breaks it.**  An archive's
    copy carries the timestamp of the FIRST time those bytes were archived --
    here by the hard-link route, because `moving.bin` changed so a genuinely new
    archive was built while `steady.bin` was linked into it.  The archive's
    timestamp therefore stays behind the working file's from then on, and under
    that repair this folder would read unsaved permanently, with nothing a user
    could do about it.

    So this asserts the outcome the repair would destroy.  `steady.bin` is aged
    an hour before the first save, or the two timestamps land close enough
    together that the test passes without the mechanism ever being exercised.
    """
    import time

    steady, moving = calc.root / "steady.bin", calc.root / "moving.bin"
    steady.write_bytes(BIG)
    moving.write_bytes(b"\x05" * 6000)
    an_hour_ago = time.time() - 3600
    os.utime(steady, (an_hour_ago, an_hour_ago))
    first = calc.save("first")

    steady.write_bytes(BIG)                    # byte-identical, timestamp = now
    moving.write_bytes(b"\x06" * 6000)         # and something else really moved
    second = calc.save("a rerun -- one output identical, one not")

    assert second.archive != first.archive, (
        "precondition: a new archive, so the link path is what carries "
        "steady.bin across rather than the whole directory being reused")
    old_copy = archive_dir(calc.root, first.archive) / "steady.bin"
    new_copy = archive_dir(calc.root, second.archive) / "steady.bin"
    assert old_copy.stat().st_ino == new_copy.stat().st_ino, (
        "precondition: unchanged content is hard-linked, not copied (§ 12)")
    assert new_copy.stat().st_mtime < steady.stat().st_mtime - 60, (
        "precondition: the archive's copy is dated well before the working "
        "file, which is exactly what the repair would trip over")

    assert calc.status().clean, (
        "a rerun that produced byte-identical output read as unsaved.  That is "
        "what comparing against the archive's own copy would do -- and it would "
        "never clear (§ 7.2)")


def test_an_unreadable_state_timestamp_means_check_rather_than_assume(
        calc, monkeypatch):
    """§ 7.2: *"If a state's time cannot be parsed, the file is hashed rather
    than declared changed — slower, never wrong, and never a false alarm that
    trains people to ignore the real one."*

    A sentence of the contract with no test.  This uses the cheap read's own
    blind spot as the probe, which is the only way to tell the two failures
    apart: a same-size rewrite inside the save's second is invisible to the
    timestamp comparison — the test above pins exactly that.  Make the
    timestamp unreadable and the *same folder* must now report the change,
    which can only happen if it hashed.

    Asserting merely that nothing crashed would pass for code that read
    "unreadable" as "unchanged" and quietly dropped the file.
    """
    import molbuilder.checkpoint as cp

    (calc.root / "big.bin").write_bytes(BIG)
    calc.save("first")
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)      # same size...
    _as_of_the_state(calc.root / "big.bin", calc)             # ...same second
    assert calc.status().clean, "precondition: the cheap read misses it"

    monkeypatch.setattr(cp, "_epoch_of", lambda iso: None)
    assert "big.bin" in calc.status().changed, (
        "an unreadable timestamp was read as 'unchanged'.  It must mean "
        "*check* -- the slow answer is never wrong, and the wrong cheap "
        "answer here loses a file")


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


def test_a_restored_big_file_keeps_its_own_timestamp(calc):
    """The archive copy carries the file's own mtime, and § 7.2 leans on it.

    The cheap read rules a large file out with two `stat` fields: the size
    matches its record, and it has not been touched since the state was saved.
    A restore that stamped the copy with **now** would give every restored big
    file a timestamp far newer than the older state it just returned to — so
    the folder would read *everything unsaved* the moment you went back, on
    every restore of an earlier state.  That is § 7.2's false alarm, fired by
    the one operation the warning matters most for.

    `shutil.copy2` is what prevents it, and until now it was covered only by
    accident: a test about recreating directories happened to assert `.clean`.
    Retire that test and the guarantee would have gone with it, silently.
    """
    import time

    big = calc.root / "big.bin"
    big.write_bytes(BIG)
    # AGE IT, and this line is the test.  A real `.DM` is written when the run
    # produces it and then sits there for hours.  With a freshly written file
    # the save and the restore land inside any sane tolerance, so the check
    # passes for code that stamps `now` -- which is exactly what the first
    # version of this test did: it was green and proved nothing.
    saved_mtime = time.time() - 3600
    os.utime(big, (saved_mtime, saved_mtime))

    target = calc.save("the state to come back to")
    big.write_bytes(b"\x02" * 5000)
    calc.save("moved on")
    calc.restore(target.id, force=True)

    assert big.stat().st_mtime == pytest.approx(saved_mtime, abs=1), (
        "the restored copy was stamped with the time of the restore rather "
        "than carrying the mtime the file had when it was saved -- an hour of "
        "difference, so this cannot pass by the test running quickly")
    # The consequence, on the path that actually reads it: the cheap read
    # rules the file out by size and timestamp and must not call it unsaved.
    status = calc.status()
    assert status.clean, (
        f"the folder read as unsaved immediately after a restore: "
        f"{status.unsaved()}")


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


def test_where_you_stand_moves_only_where_section_5_says_it_does(calc):
    """§ 5's third row, which makes three separate claims and had one test.

    *"`init` puts you at the first state, `save` puts you at the one it just
    made, `restore` puts you at the one you asked for."*  Only the third was
    asserted directly; the other two were implied by tests that would have
    failed for other reasons first, which is not the same as being checked.
    """
    first = calc.standing_at()
    assert first is not None and first.note == "set up", (
        "init must leave you standing at the state it made")

    (calc.root / "job.XV").write_text("one\n")
    saved = calc.save("stage 1")
    assert calc.standing_at().id == saved.id, (
        "save must move you onto the state it just made -- that is what makes "
        "the next save's parent this one (§ 7.1)")

    calc.restore(first.id)
    assert calc.standing_at().id == first.id


def test_nothing_but_save_and_init_ever_records_a_state(calc):
    """§ 2: *"It is not automatic. Nothing is ever saved without you saying
    so."*

    I3 makes the same promise about **your files**; this is the same promise
    about **the history**, and it had no test.  A read that quietly recorded a
    state would be invisible: the folder still works, the list simply grows
    states nobody asked for, and the notes would be whatever the code invented
    -- which L3 exists to prevent.

    `restore` is included deliberately.  It is the one verb that changes the
    folder, so it is the one most likely to be given a "save first for you"
    convenience, which A5 rules out in so many words.
    """
    (calc.root / "job.XV").write_text("coords\n")
    first = calc.save("something to stand on")
    (calc.root / "job.XV").write_text("later\n")
    calc.save("and somewhere to come back from")

    before = [s.id for s in calc.states()]
    calc.status()
    calc.status(deep=True)
    calc.states()
    calc.states(limit=1)
    calc.standing_at()
    calc.tags()
    calc.classification()
    calc.calculation()
    calc.tag("a-name", "why it matters")
    calc.restore(first.id, force=True)

    assert [s.id for s in calc.states()] == before, (
        "an operation other than `save` recorded a state")


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


def test_force_does_not_ask_and_so_cannot_be_refused_by_where_you_stand(calc):
    """§ 7: *`--force` is that answer given in advance, so the question is not
    asked at all.*

    Working out what is unsaved reads the **standing** state's record.  Asking
    anyway made a forced restore fail for a reason about the state you are
    LEAVING: lose that archive and `restore`, `status` and `list` all refused
    together, so no verb could move the folder anywhere — while the target's
    archive verified perfectly the whole time.  § 2.0 promises the verbs cover
    the work.

    The target is damaged in **no** way here; only the standing state is.  That
    separation is the test: A2 must still refuse for the target (asserted in
    its own tests), and must not refuse for this.
    """
    import shutil as _sh
    from molbuilder.checkpoint import verify_archive

    (calc.root / "a.bin").write_bytes(BIG)
    target = calc.save("a state I want to come back to")
    (calc.root / "a.bin").write_bytes(b"\x02" * 5000)
    standing = calc.save("where the folder stands now")
    _sh.rmtree(archive_dir(calc.root, standing.archive))

    verify_archive(calc.root, target.archive)      # precondition: target is fine
    with pytest.raises(CheckpointError) as exc:
        calc.status()
    assert standing.short in str(exc.value), (
        "precondition: the STANDING state is what is damaged")

    assert calc.restore(target.id, force=True).id == target.id, (
        "a forced restore was refused because of the state it was leaving")
    assert (calc.root / "a.bin").read_bytes() == BIG
    assert calc.status().clean, "and the folder is usable again afterwards"


def test_force_does_not_read_the_working_tree_it_is_about_to_overwrite(calc):
    """The other half of the same sentence: asking costs, and force skips it.

    On a real calculation "what is unsaved" hashes the whole density-matrix set
    — end to end, for a message `--force` guarantees nobody will see.  § 6
    refuses to double a *save* for less than that.

    Asserted behaviourally rather than by counting hashes: a large file that
    **cannot be read** is left in the folder.  Computing the answer opens it and
    dies; skipping the question does not.  The file is not in the target, so the
    restore's job is to remove it — which needs the directory, not the file.
    """
    (calc.root / "keep.bin").write_bytes(BIG)
    target = calc.save("the state to come back to")

    victim = calc.root / "unreadable.bin"
    victim.write_bytes(b"\x07" * 5000)
    calc.save("with a file the target does not have")
    os.chmod(victim, 0o000)
    try:
        with pytest.raises(PermissionError):
            calc.status(deep=True)               # precondition: asking opens it
        calc.restore(target.id, force=True)
    finally:
        if victim.exists():
            os.chmod(victim, 0o644)
    assert not victim.exists(), "the leftover was not removed"
    assert (calc.root / "keep.bin").read_bytes() == BIG


def test_an_unknown_state_is_refused_before_anything_else(calc):
    (calc.root / "job.XV").write_text("unsaved\n")
    with pytest.raises(NoSuchRefError):
        calc.restore("no-such-state")


def test_all_three_shapes_are_named_then_force_makes_the_folder_equal(calc):
    """A5's stated test, end to end and in one place.

    *"With an unsaved edit, an unsaved new file AND a deleted one, a
    non-interactive restore stops and changes nothing while naming all three;
    `--force` completes; afterwards the folder equals the target state
    exactly, with no rescue copies and no leftovers from where you stood."*

    It existed only as four tests covering pieces, and **the deleted shape had
    never reached a refusal message** -- `status.deleted` was asserted, the
    warning was not.  A user consenting to a loss was told about two of the
    three things they were about to lose.
    """
    (calc.root / "keep.txt").write_text(SMALL)
    (calc.root / "gone.txt").write_text(SMALL)
    (calc.root / "big.bin").write_bytes(BIG)
    target = calc.save("the state to come back to")
    expected = tree_bytes(calc.root)

    (calc.root / "later.txt").write_text("from where I stood\n")
    calc.save("moved on")

    (calc.root / "keep.txt").write_text("edited\n")      # changed
    (calc.root / "fresh.bin").write_bytes(BIG)           # added
    (calc.root / "gone.txt").unlink()                    # deleted

    before = tree_bytes(calc.root)
    with pytest.raises(DirtyWorkingTreeError) as exc:
        calc.restore(target.id)
    message = str(exc.value)
    for name in ("keep.txt", "fresh.bin", "gone.txt"):
        assert name in message, (
            f"the warning does not name {name!r}; nobody can consent to a "
            f"loss they were not told about")
    assert tree_bytes(calc.root) == before, "a refusal changed the folder"

    calc.restore(target.id, force=True)
    assert tree_bytes(calc.root) == expected, (
        "the folder does not equal the target state exactly")
    assert not (calc.root / "later.txt").exists(), (
        "a leftover from where you stood survived the restore")
    strays = sorted(k for k in tree_bytes(calc.root)
                    if k.endswith((".orig", ".rej", ".bak", "~")))
    assert strays == [], (
        f"a rescue copy was left behind: {strays}.  A5 is explicit -- no "
        f"stash, no move-aside, no automatic save-before-restore, no .orig")
    assert calc.status(deep=True).clean


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


def test_a_bare_git_init_does_not_smuggle_a_bad_name_into_the_history(
        tmp_path, checkpoint_config):
    """L3's refusal had a way around it, and § 2.0 says people take that road.

    `init` refuses a name needing repair — but only on a folder it is
    initialising. Somebody who ran `git init` here by hand (which § 2.0 says
    happens, and does not forbid) reached an already-initialised folder, so
    `init` returned early and `save` wrote the raw directory name straight into
    every state: `Calculation: has spaces!`, in a history `init` would have
    refused outright.

    Both halves are asserted, because a refusal with no way forward would just
    move the problem: the save is refused, and `init --calculation` is the verb
    that repairs the folder.
    """
    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    root = tmp_path / "has spaces!"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    subprocess.run(["git", "init", "-q", "."], cwd=root, check=True)

    repo = Repo(str(root))
    with pytest.raises(CalculationNameError) as exc:
        repo.save("stage 1 converged")
    assert "--calculation" in str(exc.value), (
        "the refusal must name the verb that fixes it")
    assert not repo.states(), "nothing may have been recorded"

    repo.init(calculation="BDT_Au_relax", note="named at last")
    state = repo.save("stage 1 converged")
    assert state.calculation == "BDT_Au_relax"


# ------------------------------------------------------------------ #
#  L3 — the note and the name go through ONE parser (§ 15)            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("note", [
    "one line",
    "a note\nover several\nlines",
    "trailing whitespace   ",
    "a colon: in the middle",
    "an em dash — and an é",
    "stage 2 at 300 Ry -- forces now below 0.02",
])
def test_the_note_survives_the_round_trip_through_the_one_parser(note):
    """§ 15: written and read through **one parser so the two cannot drift**.

    Four functions implement that -- `message_with_trailers`, `note_of`,
    `calculation_of`, `trailer_of` -- and **no test called any of them**.
    Drift between the writer and the reader is precisely what a round trip
    catches and an end-to-end save does not: a save that mangles a note still
    saves, and the mangling is only visible a month later in a list.
    """
    from molbuilder.checkpoint import (calculation_of, message_with_trailers,
                                       note_of, trailer_of)
    digest = "a" * 64
    message = message_with_trailers(note, digest, "BDT_Au_relax")
    assert note_of(message) == note.strip()
    assert calculation_of(message) == "BDT_Au_relax"
    assert trailer_of(message) == digest


def test_a_note_that_looks_like_a_trailer_stays_in_the_note(calc):
    """The hazard the parser's own docstring names, and nothing tested it.

    A note that begins *"Calculation: rerun with a tighter mesh"* must stay a
    note.  Scanning the whole message for `Key: value` would eat it: the line
    vanishes from what the person wrote **and becomes the calculation's
    name**, so a history would claim to belong to a calculation nobody has.

    What prevents it is that only the **last** block counts, and what makes
    that hold is the blank line the writer inserts.  Nothing pinned it, so
    changing that separator would forge a name with the suite green.
    """
    forged = "Calculation: not-the-real-one\nManifest-SHA256: " + "f" * 64
    (calc.root / "job.XV").write_text("x\n")
    state = calc.save(forged)

    assert state.calculation == "BDT_Au_relax", (
        "a note forged the calculation's name")
    assert state.archive != "f" * 64, "a note forged the archive digest"
    assert state.note == forged, (
        "what the person wrote was eaten as metadata and is gone from the list")


# ------------------------------------------------------------------ #
#  S5 — identity is calculation-level, the run index is not           #
# ------------------------------------------------------------------ #


def test_the_calculation_is_named_the_same_however_many_runs_appear(staged):
    """S5, in the half checkpointing owns and can answer today.

    *"Identity is calculation-level; the run index is invocation-level.
    Nothing about a run may change the id, or the files it produced would be
    orphaned by the act of producing them."*  Attempts come and go inside the
    folder -- `run-0`, `run-1`, more added later -- and every state must still
    name one calculation, or a single calculation's history would split under
    its own outputs.

    The other half of S5 -- that the *run layer* never rewrites the id -- waits
    on a run layer to test, and § 13.4 says so.
    """
    names = {staged.standing_at().calculation}
    for n in (2, 3):
        for stage in ("01_coarse", "02_tight"):
            run = staged.root / stage / f"run-{n}"
            run.mkdir()
            (run / "run.json").write_text('{"mode": "local"}')
            (run / "job.DM").write_bytes(BIG + stage.encode() + bytes([n]))
        names.add(staged.save(f"attempt {n} in both stages").calculation)

    assert names == {"BDT_Au_relax"}, (
        f"the calculation's identity moved with a run index: {names}")
    assert {s.calculation for s in staged.states()} == {"BDT_Au_relax"}


# ------------------------------------------------------------------ #
#  The Python surface itself (§ 10.4, § 15)                           #
# ------------------------------------------------------------------ #


def test_states_honours_the_limit_the_python_example_uses(calc):
    """§ 10.4's example is literally `repo.states(limit=5)`.

    A documented parameter no test ever passed is a promise nobody checked.
    Both halves matter: that it truncates, and that it truncates from the
    **newest** end -- a limit that returned an arbitrary slice would make the
    documented call show the oldest five, which is the opposite of useful.
    """
    for n in range(6):
        (calc.root / "job.XV").write_text(f"{n}\n")
        calc.save(f"step {n}")
    everything = calc.states()
    assert len(everything) == 7, "six saves plus the state `init` made"

    assert len(calc.states(limit=3)) == 3
    assert [s.id for s in calc.states(limit=3)] == \
           [s.id for s in everything[:3]], (
        "a limit must take the newest states, not an arbitrary three")
    assert len(calc.states(limit=99)) == 7, (
        "a limit past the end is not an error")


def test_no_python_caller_can_ask_for_a_text_only_restore(calc):
    """A4 names **three** surfaces, and this is the third.

    The CLI flag and the route field are each tested on their own surface.
    The Python keyword is the one the other two called, and nothing asserted
    it had gone -- **the absence of the parameter is the rule**, exactly as it
    is for the classification's directory argument (S1c).
    """
    import inspect
    params = set(inspect.signature(Repo.restore).parameters)
    assert params == {"self", "state", "force"}, (
        f"Repo.restore takes {sorted(params)}; a restore returns the whole "
        f"folder or it does not happen (A4)")


def test_a_missing_git_is_its_own_error_and_says_how_to_fix_it(calc,
                                                               monkeypatch):
    """One of the five subclasses § 15 names, referenced by no test.

    It is separated from the rest *because it is one a person can act on* --
    "a surface that cannot tell them apart reports 'fix your input' for a
    broken disk."  That separation is worth nothing unless the class is
    actually raised where git is actually absent.
    """
    import molbuilder.checkpoint as cp
    from molbuilder.checkpoint import GitNotInstalledError

    def no_git(*a, **k):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(cp.subprocess, "run", no_git)
    with pytest.raises(GitNotInstalledError) as exc:
        calc.states()
    message = str(exc.value).lower()
    assert "git" in message
    assert "install" in message or "conda" in message, (
        "a refusal a person is meant to resolve must say how")


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


def test_a_deleted_manifest_line_is_refused(calc):
    """I2b's first named edit, and it was untested.

    *"Delete a MANIFEST line and that file is simply not restored -- it reads
    exactly like 'this archive never held it'."*  Under content addressing the
    digest catches it, which is the design working rather than a second check;
    asserting it is what keeps that true if the naming scheme ever moves.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "other.bin").write_bytes(b"\x05" * 6000)
    state = calc.save("two big files")
    (calc.root / "job.XV").write_text("moved on\n")
    calc.save("moved on")

    man = archive_dir(calc.root, state.archive) / MANIFEST_NAME
    kept = [ln for ln in man.read_bytes().splitlines(keepends=True)
            if b"\tbig.bin" not in ln]
    assert len(kept) == 1, "precondition: exactly one entry was removed"
    man.write_bytes(b"".join(kept))

    with pytest.raises(CheckpointError) as exc:
        calc.restore(state.id, force=True)
    assert MANIFEST_NAME in str(exc.value) or "modified" in str(exc.value)


def test_a_sha_edited_to_match_a_swapped_file_is_still_refused(calc):
    """I2b's worst named edit: the one that returns wrong bytes AND succeeds.

    *"Change a sha AND the file to match and a restore returns the wrong bytes
    and reports success."*  Self-consistency is exactly what a bad disk or a
    deliberate edit produces, so the check cannot be the MANIFEST against its
    own contents.  It is the MANIFEST against **the name it is stored under**,
    which is the commit's trailer and inherits git's own hashing (I2b).

    The precondition below is the point of the test: the archive is made
    internally consistent first, so anything that only looked inward would
    pass.
    """
    import hashlib
    from molbuilder.checkpoint import parse_manifest as _parse

    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("first")
    (calc.root / "job.XV").write_text("moved on\n")
    calc.save("moved on")

    adir = archive_dir(calc.root, state.archive)
    swapped = b"\x42" * 5000
    (adir / "big.bin").write_bytes(swapped)
    man = adir / MANIFEST_NAME
    man.write_bytes(man.read_bytes().replace(
        hashlib.sha256(BIG).hexdigest().encode(),
        hashlib.sha256(swapped).hexdigest().encode()))
    assert _parse(man.read_bytes(), "x")["big.bin"][0] == \
        hashlib.sha256(swapped).hexdigest(), (
        "precondition: the record and the bytes now agree with each other")

    with pytest.raises(CheckpointError) as exc:
        calc.restore(state.id, force=True)
    assert MANIFEST_NAME in str(exc.value), (
        "the refusal must name the RECORD as what failed, not the file")


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


def test_the_documents_own_advice_for_reading_one_old_file(calc):
    """§ 2, § 2.0, § 5.1 and A5 all point at `git show <state>:<path>`.

    That is right for the text half and **cannot work for the other**.  A
    large file is deliberately never in any commit (§ 3) — kept out of
    `git add` by pathspec and dropped from the index — so git has nothing to
    show.  The advice appears five times in this contract and once in the
    guide with no such qualification, which sends somebody holding a 2 GB
    `.DM` to a command that answers *path does not exist*.

    Both halves are pinned here, and the second assertion doubles as a canary:
    if `git show` ever starts producing the large file, a big file has reached
    git, which is S1's losing branch rather than an improvement to the advice.
    """
    (calc.root / "job.DM").write_bytes(BIG)
    state = calc.save("a text file and a large one")

    def git_show(path):
        return subprocess.run(["git", "show", f"{state.id}:{path}"],
                              cwd=calc.path, capture_output=True)

    assert git_show("job.fdf").returncode == 0, (
        "the advice must work for the half it is actually about")
    assert git_show("job.DM").returncode != 0, (
        "a large file is in a commit tree; the archive is supposed to be the "
        "only place it lives (§ 3, S1)")

    # What does work, derived from the state's own record rather than guessed.
    assert (archive_dir(calc.root, state.archive) / "job.DM").read_bytes() == BIG


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


def test_a_whole_staged_calculation_tags_nothing(staged):
    """L4's stated test: *"run a **full staged calculation** and assert the
    tag list is empty until somebody types `snapshot tag`."*

    Three plain saves in a flat folder cannot fail for the reason the rule
    exists -- nothing in `save` was ever going to tag.  The retired mechanism
    was per-**stage**, `<id>/<stage>/<UTC>`, so the fixture with stages and
    attempts is the one this has to run over.  Then a tag is typed, and it is
    the only one, which is what makes the empty list above meaningful rather
    than a repo where tagging is simply broken.
    """
    for stage in ("01_coarse", "02_tight"):
        for n in range(2):
            run = staged.root / stage / f"run-{n}"
            (run / "job.out").write_text(f"{stage} attempt {n} converged\n")
            (run / "job.DM").write_bytes(BIG + stage.encode() + bytes([n + 7]))
            staged.save(f"{stage} attempt {n} converged")

    assert len(staged.states()) >= 4, "precondition: a history with shape"
    assert staged.tags() == [], "something tagged a state on your behalf (L4)"

    staged.tag("chosen", "the geometry I decided on")
    assert [t.name for t in staged.tags()] == ["chosen"]


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


def test_a_restore_replays_the_save_under_every_mutation_the_rule_names(
        calc, checkpoint_config):
    """I2a's stated test in full: *"change the engine, empty the hint list,
    **delete the config outright** — and assert the restored tree is
    byte-identical either way."*

    Only the middle mutation used to be exercised, by a separate test this
    one **replaces**: its whole body is the `an_empty_hint_list` case below,
    so keeping both would have been one assertion wearing two names, and the
    weaker name is the one a later reader trusts.

    Deleting the file is the decisive case, and for a different reason than
    the other two: those change the answer a config read would *give*, so a
    restore that consulted it might still land in the right place by luck.
    With no file at all a read either falls back to the 10 MB default — under
    which nothing in this fixture is large — or fails outright. Neither can be
    mistaken for replaying the record.

    The comparison is over the WHOLE tree with no exclusions. It used to
    exempt the config file, because the config lived *inside* the folder being
    compared — an exemption that existed only to hide the rule S1c is about.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "small.txt").write_text(SMALL)
    target = calc.save("the state to come back to")
    expected = tree_bytes(calc.root)

    (calc.root / "big.bin").write_bytes(b"\x09" * 5000)
    (calc.root / "stray.bin").write_bytes(BIG)
    moved = calc.save("moved on")

    cfg = checkpoint_config(size_limit_bytes=1024, engines={"generic": []})

    def a_different_engine():
        # A hint list that names the file, under a name nothing else uses.
        checkpoint_config(size_limit_bytes=1024,
                          engines={"generic": ["*.txt"], "vasp": ["*.bin"]})

    def an_empty_hint_list():
        checkpoint_config(size_limit_bytes=99_999_999, engines={"generic": []})

    def no_config_at_all():
        cfg.unlink()

    for mutate in (a_different_engine, an_empty_hint_list, no_config_at_all):
        calc.restore(moved.id, force=True)         # go away again
        mutate()
        calc.restore(target.id, force=True)
        assert tree_bytes(calc.root) == expected, (
            f"a restore changed with the configuration ({mutate.__name__}); "
            f"it must replay what the save recorded (I2a)")


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

    before = tree_bytes(calc.root)
    with pytest.raises(CheckpointError):
        calc.restore(target.id, force=True)
    after = tree_bytes(calc.root)
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


def test_a_save_killed_as_the_record_is_written_publishes_nothing(calc,
                                                                  monkeypatch):
    """A1 at the seam the copy test cannot reach.

    *"Kill the process between EACH step"* -- and only the copy step had a
    kill point.  Here every copy is made and verified and the process dies as
    the MANIFEST is written, which is the last moment before the rename.
    Nothing may appear under a published name: the staging directory is the
    whole of what is lost.
    """
    import molbuilder.checkpoint as cp

    (calc.root / "big.bin").write_bytes(BIG)
    before = {d.name for d in _published(calc)}
    real = cp._atomic_write_bytes

    def die(target, data, tmp_dir=None):
        """Fail on the MANIFEST only — the seam this test is named for.

        A stub that failed on *every* atomic write would kill the save at
        `.gitignore` instead, which is a different moment with a different
        guarantee, and the test would still be green while asserting nothing
        about the archive.
        """
        if target.name == MANIFEST_NAME:
            raise OSError("disk full")
        return real(target, data, tmp_dir)

    monkeypatch.setattr(cp, "_atomic_write_bytes", die)
    with pytest.raises(OSError):
        calc.save("dies as the record is written")

    assert {d.name for d in _published(calc)} == before, (
        "an archive appeared at a published name after the save died")
    assert not [p for p in (calc.root / ".binsnapshots").iterdir()
                if not _SHA256_RE.match(p.name)], (
        "the staging directory was left behind; it is cleaned up on any exit")


def test_an_archive_published_but_never_committed_costs_nothing(calc,
                                                                monkeypatch):
    """The one interruption § 6 describes, and § 12 calls the only garbage.

    *"The worst an interruption leaves behind is an archive nothing points at
    yet -- no lost data, and the next save writes the identical bytes to the
    identical path."*  Both halves asserted: the folder still works, and the
    next save **adopts that exact archive** rather than building a second one,
    which is what makes a repeat harmless by construction rather than by
    locking (S9).
    """
    import molbuilder.checkpoint as cp
    real = cp.publish_archive

    def publish_then_die(root, big):
        real(root, big)
        raise OSError("killed after publishing, before the commit")

    (calc.root / "big.bin").write_bytes(BIG)
    monkeypatch.setattr(cp, "publish_archive", publish_then_die)
    with pytest.raises(OSError):
        calc.save("dies between publish and commit")
    monkeypatch.undo()

    orphans = {d.name for d in _published(calc)}
    state = calc.save("the next save, after the interruption")
    assert state is not None, "the interrupted work was not saveable afterwards"
    assert state.archive in orphans, (
        "the next save built a second archive instead of reusing the one the "
        "interrupted save already published -- same content, same digest, "
        "same path (§ 6)")
    assert calc.status().clean


# ------------------------------------------------------------------ #
#  I2 — a MANIFEST is authoritative: exists, size, sha256             #
# ------------------------------------------------------------------ #


def test_a_save_refuses_to_adopt_a_damaged_archive(calc):
    """§ 1's promise, broken by the one operation that makes it.

    Publishing is *create if absent*, and the absent check was the file being
    there — so a save whose big files had not changed **returned the existing
    digest on the strength of the name alone**. The name proves what an archive
    *should* hold, never that it still does: with the archive damaged, the new
    state carried a digest that no longer verified, and the save said *saved*
    about a state that could not be returned to.

    I2b allows two outcomes and not three — *it matches, or it is refused* —
    and skipping the question was the third. The refusal has to be usable, so
    it names the remedy: the archive is content-addressed, so deleting it and
    saving again rebuilds it byte-identically from the files still on disk.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("state 1")

    # Truncated, which is what a disk that filled up leaves behind -- and what
    # the save's CHEAP check is for.  Same-size damage is deliberately not
    # caught here; the test below pins that half.
    (archive_dir(calc.root, first.archive) / "big.bin").write_bytes(BIG[:100])

    # Move away and bring the same big content back: the save would otherwise
    # reuse that archive, because identical content means an identical digest.
    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    calc.save("state 2")
    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "note.txt").write_text(SMALL)

    with pytest.raises(CheckpointError) as exc:
        calc.save("state 3 -- same big content as state 1")
    message = str(exc.value)
    assert "damaged" in message or "does not match" in message
    assert "rm -rf" in message and ".binsnapshots" in message, (
        "a refusal with no way out is a dead end; the archive is named by its "
        "content, so removing it and saving again is the remedy")

    # And the remedy actually works, which is what makes it a remedy.
    import shutil as _sh
    _sh.rmtree(archive_dir(calc.root, first.archive))
    rebuilt = calc.save("state 3, after removing the damaged archive")
    assert rebuilt is not None
    from molbuilder.checkpoint import verify_archive
    verify_archive(calc.root, rebuilt.archive)


def test_the_save_checks_cheaply_and_the_restore_checks_exactly(calc):
    """The other half of the split, pinned so neither side drifts back.

    **This is git's own model, and the reason is git's own too.**  A commit
    never re-reads the object store — corrupt a loose object and `git commit`
    exits 0, while `git checkout` answers *"inflate: data stream error"* and
    restores nothing.  Verification lives on the read, where zlib hands it over
    for free, and in `fsck`.

    Ours cannot be free on the read, because the archive is raw bytes — so the
    restore asks explicitly, and that is what this asserts.  The save asks
    cheaply: the record's own digest, then existence and size.  A same-size
    byte flip is therefore **invisible at save time on purpose**, and the save
    that follows it succeeds.

    Both halves are here in one test because they are one decision.  Assert
    only the second and somebody "hardens" the save and doubles every save in
    the system; assert only the first and the restore's exactness looks
    optional.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    first = calc.save("state 1")

    # Same size, different bytes: only a checksum can tell.
    (archive_dir(calc.root, first.archive) / "big.bin").write_bytes(
        b"\x99" * len(BIG))

    (calc.root / "big.bin").write_bytes(b"\x02" * 5000)
    calc.save("state 2")
    (calc.root / "big.bin").write_bytes(BIG)          # back to state 1's bytes
    (calc.root / "note.txt").write_text(SMALL)

    reused = calc.save("state 3 -- reuses state 1's archive")
    assert reused is not None, (
        "the save must not pay for a full re-hash of an archive it is only "
        "pointing at; that is the doubling this split exists to avoid")
    assert reused.archive == first.archive

    # ...and the exact question is asked where the bytes are about to be used.
    with pytest.raises(CheckpointError) as exc:
        calc.restore(first.id, force=True)
    assert "sha256" in str(exc.value), (
        "the restore must compare content -- it is our inflate check, and "
        "nothing else in the system performs it")


def test_a_reused_inode_is_checked_like_a_copied_one(calc):
    """The same assumption, in the hard-link path.

    Unchanged content is hard-linked rather than copied, and the link used to
    skip the re-hash on the grounds that it is *"the same inode that was
    verified when it was first written."*  Verified **when written** is true;
    unchanged **since** is the assumption, and I2b exists because that
    assumption fails. A damaged inode would be linked into a brand-new archive
    whose MANIFEST disagrees with its own bytes from the moment it appears.

    Two big files here, and only one of them changes — so the other is the one
    offered for reuse, which is the path under test.
    """
    steady = calc.root / "steady.bin"
    moving = calc.root / "moving.bin"
    steady.write_bytes(BIG)
    moving.write_bytes(b"\x05" * 6000)
    first = calc.save("two large files")

    # Damage the file that will be offered for reuse next time.
    (archive_dir(calc.root, first.archive) / "steady.bin").write_bytes(
        b"\x99" * len(BIG))
    moving.write_bytes(b"\x06" * 6000)          # a genuinely new archive

    with pytest.raises(CheckpointError) as exc:
        calc.save("only the moving file changed")
    assert "steady.bin" in str(exc.value), (
        "the damaged inode was linked into the new archive unchecked")


def test_verify_runs_over_every_archive_in_the_folder(calc):
    """I2's own stated test, which the contract calls the most valuable one.

    It had no test whose subject it was: `verify_archive` was reached as a
    side assertion inside the concurrency test and by the restore path.  A
    folder accumulates archives, and the claim is about all of them.
    """
    from molbuilder.checkpoint import verify_archive

    digests = set()
    for n in range(4):
        (calc.root / "big.bin").write_bytes(bytes([n]) * 5000)
        (calc.root / f"note{n}.txt").write_text(SMALL)
        digests.add(calc.save(f"round {n}").archive)
    assert len(digests) == 4, "precondition: four distinct archives"

    on_disk = {d.name for d in _published(calc)}
    assert digests <= on_disk
    for name in on_disk:
        verify_archive(calc.root, name)        # raises on any of the three


def test_a_manifest_entry_whose_file_is_gone_is_refused(calc):
    """I2's first condition: **the file exists**.

    Losing one payload out of an otherwise healthy archive is not the same
    failure as losing the archive, and only the per-entry check can see it:
    the MANIFEST still hashes to its own name, so the digest check passes.
    """
    from molbuilder.checkpoint import verify_archive

    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("with an archive")
    (archive_dir(calc.root, state.archive) / "big.bin").unlink()

    with pytest.raises(CheckpointError) as exc:
        verify_archive(calc.root, state.archive)
    assert "big.bin" in str(exc.value), "it must name the entry that is gone"


def test_a_manifest_entry_whose_size_disagrees_is_refused(calc):
    """I2's second condition: **its size matches**.

    A truncated copy is the ordinary shape of a disk that filled up mid-write,
    and it is cheaper to detect than a checksum -- so it is named for what it
    is rather than reported as a content mismatch.
    """
    from molbuilder.checkpoint import verify_archive

    (calc.root / "big.bin").write_bytes(BIG)
    state = calc.save("with an archive")
    (archive_dir(calc.root, state.archive) / "big.bin").write_bytes(BIG[:100])

    with pytest.raises(CheckpointError) as exc:
        verify_archive(calc.root, state.archive)
    message = str(exc.value)
    assert "big.bin" in message and "bytes" in message


def test_a_corrupt_archive_is_refused_before_you_are_asked_to_accept_a_loss(
        calc):
    """§ 7: *two refusals, then one question, and **the order is the rule**.*

    The reason is stated there: *"nobody should be asked to accept a loss for
    an operation that then fails for an unrelated reason."*  The existing
    corruption test passes `force=True`, which skips the question entirely --
    so it proved the archive is checked, never that it is checked FIRST.

    Both conditions are true here at once: the target's archive is corrupt AND
    there is unsaved work.  The archive must be what refuses.
    """
    (calc.root / "big.bin").write_bytes(BIG)
    target = calc.save("the state to come back to")
    (calc.root / "big.bin").write_bytes(b"\x03" * 5000)   # a DIFFERENT archive,
    calc.save("moved on")                                 # so only the target's
    (archive_dir(calc.root, target.archive)               # is damaged
     / "big.bin").write_bytes(b"\x00" * 5000)
    (calc.root / "unsaved.txt").write_text("work I have not saved\n")

    assert not calc.status().clean, "precondition: there is a loss to accept"
    with pytest.raises(CheckpointError) as exc:
        calc.restore(target.id)
    assert not isinstance(exc.value, DirtyWorkingTreeError), (
        "the user was asked to accept a loss for a restore that cannot happen")
    assert "sha256" in str(exc.value) or "I2" in str(exc.value)


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
    for path in walk_independently(staged.root):
        key = key_of(staged.root, path)
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


def test_a_symlink_to_a_big_file_is_not_followed_into_the_archive(calc):
    """S1's carve-out, at the size where following a link actually costs.

    **The existing symlink test could not fail for this.**  The staged fixture
    links a *small* pseudopotential, so treating links as ordinary files
    changes nothing observable there — the target is under the limit either
    way and still lands in git.  Breaking the walk to follow links failed not
    one test in the suite.

    Over the limit the two behaviours separate, in both directions at once:
    the target is archived a **second** time under the link's path, and the
    restore then writes a real file where a link belongs.  A stage that linked
    a shared file comes back holding its own private copy, and the layout the
    next run reads is quietly not the one that was saved.
    """
    real = calc.root / "shared.DM"
    real.write_bytes(BIG)
    stage = calc.root / "01_coarse"
    stage.mkdir()
    (stage / "shared.DM").symlink_to("../shared.DM")

    state = calc.save("a stage that links a large shared file")
    archived = _archived(calc, state)
    assert "shared.DM" in archived, "the real file is stored once, where it is"
    assert "01_coarse/shared.DM" not in archived, (
        "the link was followed and its target archived a second time under "
        "the link's path — S1: a link has no content of its own")

    calc.restore(state.id, force=True)
    link = stage / "shared.DM"
    assert link.is_symlink(), "a restore turned a link into a real file"
    assert link.resolve() == real.resolve()
    assert link.read_bytes() == BIG


def test_a_link_an_ignore_pattern_would_swallow_is_still_saved(
        tmp_path, checkpoint_config):
    """S1's carve-out is from the ARCHIVE, not from the snapshot.

    An always-large entry matches a **name**, and a name says nothing about a
    link: `stage2/job.DM -> ../stage1/job.DM` is twenty bytes of path text that
    `.gitignore` sent to the store which does not take links.  It landed in
    NEITHER -- `add` skipped it as ignored, the archive skipped it as a link --
    so a restore neither brought it back nor removed it, and § 3's *exactly one
    store* quietly did not hold.

    **Every symlink fixture in this file names no always-large family**, so
    nothing here could reach it: with no pattern, a link is not ignored and git
    takes it without being asked.  The engine entry is the whole test.

    These are the links `jobset/materialize.py` lays to carry a file forward
    from the stage that produced it, so the shape is the real one.
    """
    checkpoint_config(size_limit_bytes=1024,
                      engines={"generic": [], "siesta": ["*.DM"]})
    root = tmp_path / "BDT_Au_relax"
    (root / "stage1").mkdir(parents=True)
    (root / "stage2").mkdir()
    (root / "task.json").write_text('{"engine": "siesta"}')
    (root / "stage1" / "job.DM").write_bytes(BIG)
    (root / "stage2" / "job.DM").symlink_to("../stage1/job.DM")

    repo = Repo(str(root))
    state = repo.init(engine="siesta", note="a stage carrying the one before it")

    tracked, archived = _tracked(repo), _archived(repo, state)
    assert "stage1/job.DM" in archived, "the real file goes to the archive"
    assert "stage2/job.DM" not in archived, (
        "the link was followed and its target archived a second time")
    assert "stage2/job.DM" in tracked, (
        "the link is in NO store: `.gitignore` matched its name and the "
        "archive does not take links, so nothing holds it (S1)")

    # And it survives the round trip AS A LINK.
    link = root / "stage2" / "job.DM"
    link.unlink()
    repo.save("the link was removed")
    repo.restore(state.id, force=True)
    assert link.is_symlink(), (
        "a state that held a link came back without it -- a restore must "
        "recreate the layout, which is what S1 leans on")
    assert link.resolve() == (root / "stage1" / "job.DM").resolve()


def test_a_stray_link_that_no_state_holds_is_removed_by_a_restore(
        tmp_path, checkpoint_config):
    """A5's removal half, for the link nothing ever saved.

    **This test was written wrong first, and mutation testing is what said
    so.**  It saved the link before restoring — which makes it *tracked*, so
    git's own checkout removes it and the sweep under test never ran.  Green,
    and proving nothing.

    The real gap is the link that was **never saved** and whose name an ignore
    pattern matches: `git checkout` does not know it, `git clean` without `-x`
    will not touch it, and the leftover sweep walked only regular files.  So a
    stray `job.DM -> ../stage1/job.DM` — the shape `jobset/materialize.py` lays
    at prep time — survived a restore of a state that never had it, pointing
    the next run at another stage's output.

    No `force` is needed, and that is the point rather than a convenience: git
    says nothing about an ignored untracked path and the archive holds no
    links, so the folder reads **clean**.  It is a leftover, not unsaved work,
    and A5 removes leftovers without asking because they are not a loss.
    """
    checkpoint_config(size_limit_bytes=1024,
                      engines={"generic": [], "siesta": ["*.DM"]})
    root = tmp_path / "BDT_Au_relax"
    (root / "stage1").mkdir(parents=True)
    (root / "stage2").mkdir()
    (root / "task.json").write_text('{"engine": "siesta"}')
    (root / "stage1" / "job.DM").write_bytes(BIG)
    repo = Repo(str(root))
    early = repo.init(engine="siesta", note="no link yet")

    stray = root / "stage2" / "job.DM"
    stray.symlink_to("../stage1/job.DM")          # laid, never saved
    assert "stage2/job.DM" not in _tracked(repo), (
        "precondition: untracked, so git's checkout cannot be what removes it")
    assert repo.status().clean, (
        "precondition: invisible to status, so this is a leftover rather than "
        "unsaved work -- and no force is needed to walk past it")

    repo.restore(early.id)
    assert not stray.is_symlink() and not stray.exists(), (
        "a stray link survived a restore of a state that never held it; the "
        "next run would pick it up unasked (A5)")


def test_a_restore_never_writes_through_a_hard_link_in_its_way(
        tmp_path, checkpoint_config):
    """A5: the folder equals the target **exactly** — and `copy2` did not.

    It truncates its destination in place, so it writes *through* a hard link.
    Two archived paths sharing an inode, restored to a state that held them
    with **different** content: the second copy lands in both, one path ends up
    holding the other's bytes, and the restore reports success. § 1's promise
    is that a state you saved is one you can return to.

    **The names must match an always-large family or the bug cannot appear** —
    otherwise `git clean` removes both files before the copy, and every
    ordinary fixture in this file is in exactly that position. That is why
    nothing here found it.
    """
    checkpoint_config(size_limit_bytes=1024, engines={"generic": ["*.DM"]})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / "job.fdf").write_text("x\n")
    a, b = root / "a.DM", root / "b.DM"
    a.write_bytes(b"\xAA" * 5000)
    b.write_bytes(b"\xBB" * 5000)
    repo = Repo(str(root))
    target = repo.init(note="two large files, holding different bytes")

    b.unlink()
    os.link(a, b)                            # one inode under two names
    assert a.stat().st_ino == b.stat().st_ino, "precondition: hard-linked"
    (root / "note.txt").write_text("moved on\n")
    repo.save("the two paths were linked together")

    repo.restore(target.id, force=True)
    assert a.read_bytes() == b"\xAA" * 5000, (
        "the restore wrote b.DM's bytes through the shared inode and into "
        "a.DM; the folder does not equal the target state (A5)")
    assert b.read_bytes() == b"\xBB" * 5000
    assert a.stat().st_ino != b.stat().st_ino, (
        "the target held two independent files, so the restore must leave two")
    assert repo.status(deep=True).clean


def test_a_restore_never_follows_a_symlink_standing_where_a_file_belongs(
        tmp_path, checkpoint_config):
    """The same flaw, pointed outward — and this one can leave the folder.

    `copy2` follows a symlink at the destination, so the restored bytes land on
    whatever it points at while the link stays where a real file belongs. The
    state's own file is then not restored *and* an unrelated file is
    overwritten, which may be outside the calculation entirely.
    """
    checkpoint_config(size_limit_bytes=1024, engines={"generic": ["*.DM"]})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / "job.fdf").write_text("x\n")
    (root / "job.DM").write_bytes(BIG)
    repo = Repo(str(root))
    target = repo.init(note="a real large file")

    bystander = tmp_path / "not-even-in-the-calculation.dat"
    bystander.write_bytes(b"\x33" * 5000)
    (root / "job.DM").unlink()
    (root / "job.DM").symlink_to(bystander)
    # NOT saved, and that is the whole setup.  A link that was saved is
    # tracked, so git's own checkout removes it before the copy runs and this
    # can never fire -- the first version of this test saved it, passed
    # without the fix, and pinned nothing.  Untracked *and* matching an
    # always-large name is what survives: `git clean` leaves ignored paths
    # alone, and the leftover sweep keeps anything the target state holds.
    assert (root / "job.DM").is_symlink()

    repo.restore(target.id, force=True)
    assert not (root / "job.DM").is_symlink(), (
        "the link was left standing where the state holds a real file")
    assert (root / "job.DM").read_bytes() == BIG
    assert bystander.read_bytes() == b"\x33" * 5000, (
        "the restore wrote through the link and overwrote a file outside the "
        "calculation folder")


def test_identical_content_in_one_save_is_stored_once(calc):
    """§ 12's *Disk cost*, inside a single save.

    The property held only ACROSS saves: the index of what is already archived
    is built from **published** archives, so two paths carrying the same bytes
    in the same save were each copied in full. A stage that carries its
    predecessor's 2 GB density matrix forward — by copy, or by a hard link,
    which the walk sees as two ordinary files — cost 4 GB of archive, every
    save.
    """
    from molbuilder.checkpoint import verify_archive

    (calc.root / "s1").mkdir()
    (calc.root / "s2").mkdir()
    (calc.root / "s1" / "job.DM").write_bytes(BIG)
    (calc.root / "s2" / "job.DM").write_bytes(BIG)     # same bytes, other path
    state = calc.save("two stages holding the same output")

    adir = archive_dir(calc.root, state.archive)
    one, two = adir / "s1" / "job.DM", adir / "s2" / "job.DM"
    assert one.is_file() and two.is_file(), "both paths are in the archive"
    assert one.read_bytes() == two.read_bytes() == BIG
    assert one.stat().st_ino == two.stat().st_ino, (
        "identical content was copied twice inside one archive (§ 12)")
    verify_archive(calc.root, state.archive)      # and it is still a valid one

    # The archive is named by its MANIFEST, and that did not change -- so this
    # is a storage detail, not a new archive format.
    calc.restore(state.id, force=True)
    assert (calc.root / "s1" / "job.DM").read_bytes() == BIG
    assert (calc.root / "s2" / "job.DM").read_bytes() == BIG
    assert (calc.root / "s1" / "job.DM").stat().st_ino != \
           (calc.root / "s2" / "job.DM").stat().st_ino, (
        "sharing an inode is the ARCHIVE's business; the restored folder gets "
        "independent files, which is what the state describes")


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


def test_a_file_that_crossed_the_limit_restores_from_either_state(calc):
    """S7's round trip, which the two tests above do not attempt.

    They assert *which store* the file lands in and stop — neither restores
    anything. But a file that crossed the limit is held **in git by one state
    and in the archive by the next**, so the history contains two states that
    disagree about which store owns that path. A restore then has to fetch it
    from a different place depending on which state you asked for, and the two
    mechanisms that could get it wrong are the ones that *delete*: git's own
    checkout, and the leftover sweep that removes anything neither record
    claims.

    Both directions of the crossing, and both directions of travel.
    """
    small_bytes = b"x" * 100
    grows = calc.root / "grows.bin"
    shrinks = calc.root / "shrinks.bin"

    grows.write_bytes(small_bytes)          # git
    shrinks.write_bytes(BIG)                # archive
    before = calc.save("one small, one large")
    assert "grows.bin" in _tracked(calc)
    assert "shrinks.bin" in _archived(calc, before)

    grows.write_bytes(BIG)                  # -> archive
    shrinks.write_bytes(small_bytes)        # -> git
    after = calc.save("they swapped stores")
    assert "grows.bin" in _archived(calc, after)
    assert "shrinks.bin" in _tracked(calc)

    # Back to the state where the sizes -- and the stores -- were the other way.
    calc.restore(before.id, force=True)
    assert grows.read_bytes() == small_bytes, (
        "the small version did not come back from git")
    assert shrinks.read_bytes() == BIG, (
        "the large version did not come back from the archive")
    assert calc.status(deep=True).clean, (
        f"the folder does not equal the state it stands at: "
        f"{calc.status(deep=True).unsaved()}")

    # And forward again, which is the crossing in the other direction.
    calc.restore(after.id, force=True)
    assert grows.read_bytes() == BIG
    assert shrinks.read_bytes() == small_bytes
    assert calc.status(deep=True).clean


def test_a_file_that_crossed_the_limit_is_still_only_in_one_store_afterwards(
        calc):
    """The other half: S1 must hold in **every** state, not just the newest.

    A restore that brought the file back from the archive while git's checkout
    also wrote it would leave one path claimed by both stores — S1's *never
    both*, reached by travelling rather than by saving. The walk here is the
    independent one, for the reason S1's bullet gives.
    """
    grows = calc.root / "grows.bin"
    grows.write_bytes(b"x" * 100)
    small_state = calc.save("small enough for git")
    grows.write_bytes(BIG)
    big_state = calc.save("now it is big")

    for state in (small_state, big_state, small_state):
        calc.restore(state.id, force=True)
        tracked, archived = _tracked(calc), _archived(calc, state)
        for path in walk_independently(calc.root):
            key = key_of(calc.root, path)
            assert (key in tracked) != (key in archived), (
                f"after restoring {state.short}, {key!r} is "
                f"tracked={key in tracked} archived={key in archived}")


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


def test_the_warning_names_a_file_the_classification_no_longer_matches(
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


def test_two_concurrent_saves_of_one_folder_cannot_corrupt_it(calc):
    """S9's stated test is *"two concurrent **saves** of the same folder"*.

    The test above races four `publish_archive` calls, which is the archive
    half only.  A `save` also regenerates `.gitignore`, stages an index,
    commits and writes a ref -- and the contract's claim that *"git serialises
    its own index already"* was an assumption **no test checked**.

    The rule is *cannot corrupt*, not *cannot interleave*, so a save losing a
    race for git's index lock is a **refusal** and is allowed.  What is not
    allowed: a failure that is anything other than a refusal, an archive that
    does not verify afterwards, or a folder left unusable.
    """
    import threading
    from molbuilder.checkpoint import verify_archive

    (calc.root / "big.bin").write_bytes(BIG)
    (calc.root / "other.bin").write_bytes(b"\x05" * 6000)

    done, failures = [], []
    ready = threading.Barrier(2)

    def saver(n):
        # A separate handle, the way two processes would reach one folder.
        repo = Repo(str(calc.root))
        try:
            # A timeout, not a bare wait: if the other thread dies before it
            # reaches the barrier, a bare wait hangs the suite forever and the
            # failure looks like an infrastructure problem instead of a bug.
            ready.wait(timeout=60)
            done.append(repo.save(f"concurrent save {n}"))
        except Exception as exc:                      # noqa: BLE001
            failures.append(exc)

    threads = [threading.Thread(target=saver, args=(n,)) for n in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert all(isinstance(f, CheckpointError) for f in failures), (
        f"a concurrent save failed with something that is not a refusal: "
        f"{failures!r}")
    for adir in _published(calc):
        verify_archive(calc.root, adir.name)
    assert any(s is not None for s in done) or failures, (
        "neither save recorded anything and neither refused")

    # Whatever the interleaving was, the folder is usable afterwards.
    calc.save("after the race")
    assert calc.status(deep=True).clean


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


def test_a_folder_with_many_big_files_saves_in_one_go(calc):
    """Volume, at the one place the shape of the code makes it a question.

    `save` names **every** big file as its own `:(exclude,literal)` pathspec
    on a **single** `git add` command line, so the argument list grows with
    the number of large files.  Every other test in this file has at most a
    handful, and § 3.1's own prose contemplates a 500-file archive -- so this
    runs that many.

    Three things must hold at once, and the third is why the test is here
    rather than being a benchmark: they are all archived, **none reaches
    git's object database**, and the folder reads clean afterwards.  The
    obvious remedy if this ever fails for argument length is to chunk the
    `add`, and a chunk boundary is exactly where a big file slips back into
    git -- so the assertion below is what makes that remedy safe to attempt.
    """
    for n in range(500):
        (calc.root / f"run-{n:03d}.DM").write_bytes(BIG + bytes([n % 251]))
    state = calc.save("five hundred large files")

    archived, tracked = _archived(calc, state), _tracked(calc)
    assert len([k for k in archived if k.endswith(".DM")]) == 500
    assert not [k for k in tracked if k.endswith(".DM")], (
        "a big file reached git's index")
    assert _loose_blob_bytes(calc) < len(BIG), (
        "big files' bytes are in .git/objects; `add` writes a blob for every "
        "path it is handed")
    assert calc.status().clean


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
        return {k: v for k, v in tree_bytes(calc.root).items()
                if k != ".gitignore"}

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
