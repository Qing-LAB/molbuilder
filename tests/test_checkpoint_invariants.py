"""The checkpoint invariants, asserted.

`docs/execution/checkpointing.md` is a list of things that must always be true,
each written so a test can assert it.  This file is that assertion for the six
which are checkable against the code as it stands and had no test naming them:
**S5, I2, I3, I4, A1, A2**.  The other six assertable ones are already pinned —
S1 and S1a and L2 and L7 in `test_checkpoint_nested_layout.py`, I1 in
`test_checkpoint_manifest_format.py`, L1 in `test_checkpoint_repo_scope.py`.

**Each test is written from the contract's own "how to check" clause**, not from
reading the implementation.  Where the contract names the method — *"corrupt one
byte"*, *"grep for git as a command word, not as a substring"*, *"exactly two
hits and no third"* — that method is used literally, because it was chosen to
catch the failure the invariant exists to prevent rather than to be easy.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from molbuilder.checkpoint import (
    Repo, CheckpointError, DirtyWorkingTreeError,
)


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(not _have_git(), reason="git not on PATH")

_SRC = Path(__file__).resolve().parent.parent / "molbuilder"


def _run_dir(root: Path, label: str = "job") -> Path:
    """A directory with one of each kind of file the classification sorts."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{label}.fdf").write_text(f"SystemLabel {label}\n")
    (root / f"{label}.run.sh").write_text("#!/bin/bash\nsiesta < job.fdf\n")
    (root / f"{label}.XV").write_text("coords\n")          # small: git's
    (root / f"{label}.DM").write_bytes(b"density" * 100)   # big: the archive's
    (root / f"{label}.out").write_text("Job completed\n")
    return root


# ------------------------------------------------------------------ #
#  I2 — a MANIFEST is authoritative for its archive                   #
#  "the single most valuable test in the system"                      #
# ------------------------------------------------------------------ #

def test_I2_every_manifest_entry_matches_its_file(tmp_path):
    """For every entry in every archive: the file exists, its size equals the
    recorded bytes, and its sha256 equals the recorded sha.

    The contract says to run this over **every** archive in a repository, not
    just the newest — a history is only as good as its oldest reachable point.
    """
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")

    # Three checkpoints, each changing the big binary, so there are three
    # archives to walk rather than one.
    for i in range(3):
        (root / "job.DM").write_bytes(f"density-v{i}".encode() * 100)
        (root / "job.out").write_text(f"Job completed {i}\n")
        repo.checkpoint(f"step {i}")

    manifests = list((root / ".binsnapshots").glob("*/MANIFEST"))
    assert len(manifests) >= 3, f"expected an archive per checkpoint, got {len(manifests)}"

    checked = 0
    for man in manifests:
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            sha, size, key = line.split(None, 2)
            f = man.parent / key
            assert f.is_file(), f"{man}: entry {key!r} has no file"
            data = f.read_bytes()
            assert len(data) == int(size), f"{man}: {key} size {len(data)} != {size}"
            assert hashlib.sha256(data).hexdigest() == sha, \
                f"{man}: {key} sha mismatch — the archive cannot be trusted"
            checked += 1
    assert checked >= 3, "no entries were actually verified"


# ------------------------------------------------------------------ #
#  A2 — restore verifies before it mutates                            #
# ------------------------------------------------------------------ #

def test_A2_restore_refuses_a_corrupt_archive_and_changes_nothing(tmp_path):
    """Corrupt one byte of the target ref's archive and attempt a restore: it
    refuses, **and the worktree is byte-identical to what it was**.

    The second half is the point.  A restore that detects corruption after
    replacing half the files has still destroyed the working state.
    """
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")
    first = repo.list_checkpoints()[0]

    (root / "job.DM").write_bytes(b"second" * 100)
    (root / "job.out").write_text("Job completed 2\n")
    repo.checkpoint("second")

    before = {p.name: p.read_bytes()
              for p in root.iterdir() if p.is_file()}

    # Corrupt exactly one byte of the target archive's copy.
    target = next((root / ".binsnapshots" / first.sha).glob("*.DM"))
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    with pytest.raises(CheckpointError):
        repo.restore(first.sha)

    after = {p.name: p.read_bytes()
             for p in root.iterdir() if p.is_file()}
    assert after == before, (
        "restore mutated the worktree before discovering the corruption")


def test_A2_restore_refuses_a_dirty_worktree_before_touching_it(tmp_path):
    """The same ordering, one gate earlier: uncommitted work is not silently
    overwritten by a restore."""
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")
    first = repo.list_checkpoints()[0]

    (root / "job.out").write_text("uncommitted work\n")
    with pytest.raises(DirtyWorkingTreeError):
        repo.restore(first.sha)
    assert (root / "job.out").read_text() == "uncommitted work\n"


# ------------------------------------------------------------------ #
#  A1 — archiving is build, verify, swap, then delete                 #
# ------------------------------------------------------------------ #

def test_A1_a_failure_mid_archive_leaves_the_old_archive_whole(tmp_path,
                                                               monkeypatch):
    """Interrupt a checkpoint's archive step; afterwards the archive set is
    either the old one or the new one, **never a mixture**.

    The contract says to kill the process between steps.  Raising from inside
    the copy is the same cut with a stack trace attached: what matters is that
    the published archive is not half-replaced.
    """
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")
    first = repo.list_checkpoints()[0]

    good = {p.name: p.read_bytes()
            for p in (root / ".binsnapshots" / first.sha).iterdir()
            if p.is_file()}
    assert good, "the first checkpoint archived nothing; fixture is wrong"

    import molbuilder.checkpoint as cp
    real_copy = cp.shutil.copy2
    calls = {"n": 0}

    def exploding_copy(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] > 1:                     # let one land, then fail
            raise OSError("simulated interruption mid-archive")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(cp.shutil, "copy2", exploding_copy)
    (root / "job.DM").write_bytes(b"new" * 200)
    (root / "job.HSX").write_bytes(b"hsx" * 200)
    (root / "job.out").write_text("Job completed 2\n")

    with pytest.raises(OSError, match="simulated interruption"):
        repo.checkpoint("this one is interrupted")

    monkeypatch.undo()
    still = {p.name: p.read_bytes()
             for p in (root / ".binsnapshots" / first.sha).iterdir()
             if p.is_file()}
    assert still == good, (
        "an interrupted archive damaged the previously published one")


# ------------------------------------------------------------------ #
#  I4 — a generated wrapper contains no git                           #
# ------------------------------------------------------------------ #

_GIT_AS_COMMAND = re.compile(r"(^|[;&|(\s])git\s")


def test_I4_no_generated_wrapper_invokes_git(tmp_path):
    """A wrapper runs on a compute node with no repository and no molbuilder;
    a `git` call there fails or, worse, touches a repository that happens to be
    above it.

    Matched as a **command word**, per the contract: `digits` and `logging` are
    not violations, and a check that flags them is one somebody will disable.
    """
    from molbuilder.runwrap import render_run_wrapper

    deck = tmp_path / "job.fdf"
    deck.write_text("SystemLabel job\nNumberOfAtoms 4\n")
    rendered = [render_run_wrapper(deck, mpi_np=1),
                render_run_wrapper(deck, mpi_np=4)]

    for text in rendered:
        for n, line in enumerate(text.splitlines(), 1):
            bare = line.split("#", 1)[0]        # comments may mention git
            assert not _GIT_AS_COMMAND.search(bare), \
                f"line {n} invokes git: {line.strip()!r}"

    # The matcher must not be so loose that it can never fail, nor so tight it
    # misses a real call -- pin both directions.
    assert not _GIT_AS_COMMAND.search("digits=4")
    assert not _GIT_AS_COMMAND.search("_logging git_style=1")
    assert _GIT_AS_COMMAND.search("git commit -m x")
    assert _GIT_AS_COMMAND.search("cd $d && git add .")


# ------------------------------------------------------------------ #
#  I3 — warm state is moved or restored, never incidentally lost      #
# ------------------------------------------------------------------ #

_WARM_SUFFIXES = (".XV", ".DM", ".CG", ".chk")


def _warm_destroying_lines(text: str):
    """Lines that delete or truncate something whose name looks like warm state.

    Covers the three shapes the contract names: an ``unlink``, an ``rmtree``, and
    a **truncating open** — `open(p, "w")` on a `.DM` empties it just as surely
    as removing it, and is the one a reviewer's eye slides over.
    """
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        bare = line.split("#", 1)[0]
        destroys = (re.search(r"\b(unlink|rmtree)\s*\(", bare)
                    or re.search(r"\bopen\s*\([^)]*[\"']w[b+]*[\"']", bare))
        if not destroys:
            continue
        if any(suf in bare for suf in _WARM_SUFFIXES) or \
           re.search(r"warm|restart", bare, re.I):
            out.append((n, line.strip()))
    return out


def test_I3_the_detector_can_actually_fire():
    """A positive control, because the assertion below is that a search finds
    NOTHING — and a search that can never find anything passes forever while
    proving nothing.  This is what makes the next test falsifiable.

    **What it cannot see, stated rather than implied.** The detector is
    line-local: it recognises a deletion only when the same line also names a
    warm suffix or says "warm"/"restart".  A deletion whose target was bound
    three lines earlier, or whose only clue is a comment, passes it.  So this
    is a tripwire for the obvious shapes, not a proof of absence — the
    invariant's real guarantee is the design (one mover, no deleter), and this
    catches a regression that walks into it.
    """
    planted = [
        'os.unlink(run_dir / f"{label}.DM")',
        'shutil.rmtree(warm_dir)',
        'open(path / "job.XV", "w").close()',
        'shutil.rmtree(self.restart_dir)',
    ]
    for line in planted:
        assert _warm_destroying_lines(line), f"detector missed: {line!r}"
    for benign in ['open(log, "r")', 'shutil.copy2(src, dst)',
                   'out.unlink()  # the rendered wrapper']:
        assert not _warm_destroying_lines(benign), f"false positive: {benign!r}"


def test_I3_no_python_path_destroys_warm_state():
    """Warm state is **moved** (`--cold`, to `<basename>-restart-aside-<UTC>/`)
    or **replaced wholesale** (a restore, via git checkout and the archive).
    Nothing deletes or truncates it.

    A third path would be a *silent* loss: the user asked for neither operation,
    so nothing reports it, and the absence surfaces as an unexplained cold start
    hours later.

    **Scope, stated because it is narrower than the invariant.** The contract's
    check is "exactly two hits and no third" over every path that writes into a
    run directory.  Neither of those two is a Python call — the move-aside is
    `mv` in the rendered wrapper (covered by the next test) and the restore goes
    through git and the archive — so over `molbuilder/` the correct count is
    **zero**, and that is what is asserted here.
    """
    hits = []
    for src in sorted(_SRC.rglob("*.py")):
        for n, line in _warm_destroying_lines(src.read_text()):
            hits.append(f"{src.relative_to(_SRC)}:{n}: {line}")

    assert not hits, (
        "a Python path removes or truncates warm state; it is moved aside or "
        "replaced, never destroyed:\n  " + "\n  ".join(hits))


def test_I3_cold_restart_moves_warm_files_aside_rather_than_deleting(tmp_path):
    """The behavioural half: `--cold` must **move**, so the previous state is
    recoverable from the aside directory."""
    from molbuilder.runwrap import render_run_wrapper

    deck = tmp_path / "job.fdf"
    deck.write_text("SystemLabel job\nNumberOfAtoms 4\n")
    text = render_run_wrapper(deck, mpi_np=1)

    assert "-restart-aside-" in text, "no move-aside directory is named"
    cold = text[text.index("_cold"):]
    assert " mv " in cold, "the cold path must move warm files"
    assert not re.search(r"rm\s+-[rf]*f[rf]*\s+\S*\.(XV|DM|CG)", text), \
        "the cold path must not delete warm files"


# ------------------------------------------------------------------ #
#  S5 — identity is calculation-level; the run index is invocation-   #
#       level                                                          #
# ------------------------------------------------------------------ #

def test_S5_no_identity_is_derived_from_a_run(tmp_path):
    """An id must be knowable before the calculation exists, so nothing may
    derive it from a run's output, a timestamp, or a run index — an id that
    depended on a result would change exactly when the calculation worked.

    **This asserts the shipped half only, and the other half cannot be tested
    yet.** The contract's check is two claims: *no code path derives an id from
    a run*, and *`stages.json`'s `run.id` is read, never recomputed*. The second
    has nothing to test against — `stages.json` and its reader are proposed, not
    built (`engines/stages.md § 6`) — so what is pinned here is the first, at
    the place it would actually go wrong today: the run index advances across
    invocations while the basename every warm file is keyed by does not, and no
    warm filename carries the index. When the reader lands, item 6 of the plan
    owes this invariant its second assertion.
    """
    from molbuilder.runwrap import render_run_wrapper

    deck = tmp_path / "bdt_au_relax.fdf"
    deck.write_text("SystemLabel bdt_au_relax\nNumberOfAtoms 4\n")
    text = render_run_wrapper(deck, mpi_np=1)

    # The basename is a literal, fixed at generation.
    assert "bdt_au_relax" in text
    # …and the run index is resolved at run time, separately.
    assert "_run_n" in text, "the run index should be a runtime variable"
    assert "-run${_run_n}" in text or "-run$_run_n" in text, \
        "outputs are indexed by the runtime run number"

    # The two must not be entangled: no warm-file name carries the index.
    for suf in ("XV", "DM", "CG"):
        assert not re.search(r"run\$?\{?_run_n\}?[^\n]*\." + suf, text), \
            f"the .{suf} name must not depend on the run index"


def test_S5_the_id_is_read_from_the_deck_not_recomputed(tmp_path):
    """`run-identity.md § 3` rule 1: an id is read, never recomputed — so two
    wrappers generated from the same deck carry the same basename regardless of
    when they were generated."""
    from molbuilder.runwrap import render_run_wrapper

    deck = tmp_path / "bdt.fdf"
    deck.write_text("SystemLabel bdt\nNumberOfAtoms 4\n")
    a = render_run_wrapper(deck, mpi_np=1)
    b = render_run_wrapper(deck, mpi_np=1)

    def basenames(t):
        return set(re.findall(r"\bbdt\b", t))
    assert basenames(a) == basenames(b)
    assert "bdt" in a


# ------------------------------------------------------------------ #
#  L5 — a checkpoint costs what changed, not what exists              #
# ------------------------------------------------------------------ #

def _du(root: Path) -> int:
    """Disk actually consumed, counting a hard-linked file once — which is the
    number L5 is about.  Summing `st_size` would count every link in full and
    report no saving at all."""
    seen, total = set(), 0
    for f in root.rglob("*"):
        if f.is_file():
            st = f.stat()
            if st.st_ino in seen:
                continue
            seen.add(st.st_ino)
            total += st.st_size
    return total


def test_L5_a_second_checkpoint_of_unchanged_binaries_costs_near_zero(tmp_path):
    """Checkpoint a folder twice with the binaries untouched between them; the
    second checkpoint's *incremental* disk cost is near zero.

    This is the contract's own check, verbatim.  Automatic checkpoints fire
    twice per stage, so without it a five-stage mission pays ten full copies of
    its `.DM` set and the folder this design exists to keep manageable becomes
    the reason the disk fills.
    """
    root = _run_dir(tmp_path / "calc")
    big = b"x" * 400_000
    (root / "job.DM").write_bytes(big)
    (root / "job.HSX").write_bytes(big + b"y")

    repo = Repo(str(root))
    repo.init(engine="siesta")
    after_first = _du(root / ".binsnapshots")
    assert after_first > 700_000, "the fixture's binaries were not archived"

    # Text changes; the binaries do not.
    (root / "job.out").write_text("Job completed, again\n")
    assert repo.checkpoint("second") is not None
    after_second = _du(root / ".binsnapshots")

    growth = after_second - after_first
    assert growth < 10_000, (
        f"the second checkpoint cost {growth} bytes of disk for binaries that "
        f"did not change; L5 requires near zero")


def test_L5_a_changed_binary_is_stored_again(tmp_path):
    """The other half, and it is not a defect: in a flat directory one
    `<id>.DM` is overwritten every stage, so its content genuinely differs and
    a fresh copy is correct (`project-layout.md § 6.2`).  Reuse is by CONTENT,
    so this needs no special case for the directory shape."""
    root = _run_dir(tmp_path / "calc")
    (root / "job.DM").write_bytes(b"a" * 300_000)
    repo = Repo(str(root))
    repo.init(engine="siesta")
    after_first = _du(root / ".binsnapshots")

    (root / "job.DM").write_bytes(b"b" * 300_000)      # a real change
    repo.checkpoint("stage 2 overwrote it")
    growth = _du(root / ".binsnapshots") - after_first

    assert growth > 250_000, (
        "a changed binary must be stored, not silently aliased to the old one")


def test_L5_reuse_never_aliases_different_content(tmp_path):
    """The failure that would make L5 worse than the disk it saves: two
    different results sharing one archived copy.  Each checkpoint's archive must
    restore its OWN bytes."""
    root = _run_dir(tmp_path / "calc")
    (root / "job.DM").write_bytes(b"first" * 50_000)
    repo = Repo(str(root))
    repo.init(engine="siesta")
    first = repo.list_checkpoints()[0]

    (root / "job.DM").write_bytes(b"secnd" * 50_000)
    repo.checkpoint("second")

    a = (root / ".binsnapshots" / first.sha / "job.DM").read_bytes()
    assert a == b"first" * 50_000, "the first archive was overwritten by reuse"

    repo.restore(first.sha)
    assert (root / "job.DM").read_bytes() == b"first" * 50_000


def test_L5_does_not_link_against_a_rotted_candidate(tmp_path):
    """The index knows only what a MANIFEST *claims*.  If an archived file has
    rotted, linking to it would record a sha its bytes do not have — turning a
    cheap save into a corrupt one.  A candidate is hashed before it is trusted,
    so a damaged one is copied past rather than reused."""
    root = _run_dir(tmp_path / "calc")
    payload = b"z" * 200_000
    (root / "job.DM").write_bytes(payload)
    repo = Repo(str(root))
    repo.init(engine="siesta")
    first = repo.list_checkpoints()[0]

    # Rot the archived copy without touching its MANIFEST.
    rotted = root / ".binsnapshots" / first.sha / "job.DM"
    rotted.write_bytes(b"q" * 200_000)

    (root / "job.out").write_text("again\n")
    second = repo.checkpoint("second")
    assert second is not None

    fresh = root / ".binsnapshots" / second.sha / "job.DM"
    assert fresh.read_bytes() == payload, (
        "the new archive linked to rotted bytes instead of copying the source")


# ------------------------------------------------------------------ #
#  L3 — every commit and tag names its calculation                    #
#  L4 — the tag namespace is stage completions only                   #
# ------------------------------------------------------------------ #

from datetime import datetime, timezone            # noqa: E402
from molbuilder.checkpoint import (                # noqa: E402
    checkpoint_message, stage_completion_tag, parse_stage_completion_tag,
    utc_stamp,
)

_WHEN = datetime(2026, 8, 6, 22, 14, 3, tzinfo=timezone.utc)


def test_L3_the_forms_match_the_contract_examples():
    """`engines/stages.md § 7.3` gives a worked example of each form.  If the
    code and the document disagree, one of them is wrong and a reader cannot
    tell which — so the document's own strings are the fixture."""
    assert checkpoint_message(
        "bdt_au_relax_c6h4s2au38", "tight", "relaxation converged, 41 steps"
    ) == "bdt_au_relax_c6h4s2au38 · tight · relaxation converged, 41 steps"

    assert stage_completion_tag("bdt_au_relax_c6h4s2au38", "tight", _WHEN) == \
        "bdt_au_relax_c6h4s2au38/tight/20260806T221403Z"


def test_L3_a_tag_parses_into_three_parts_led_by_the_id():
    """The contract's check, verbatim: every tag parses into exactly three
    parts of which the first equals the folder's id."""
    tag = stage_completion_tag("bdt_au", "coarse", _WHEN)
    parsed = parse_stage_completion_tag(tag)
    assert parsed is not None
    assert parsed[0] == "bdt_au" and parsed[1] == "coarse"
    assert len(tag.split("/")) == 3


def test_L3_names_are_refused_rather_than_normalised():
    """A name that is not ref-safe is refused, **not rewritten**.  Silently
    normalising would decouple the history's name from the folder's — and the
    id is chosen from a set that already survives both a filename and a git ref
    (`run-identity.md § 3`), so a name needing repair is a bug upstream."""
    for bad in ("bdt au", "bdt/au", "bdt.au", "", "bdt:au"):
        with pytest.raises(CheckpointError):
            stage_completion_tag(bad, "tight", _WHEN)
        with pytest.raises(CheckpointError):
            checkpoint_message(bad, "tight", "converged")
    for bad_stage in ("tight run", "tight/er", "", "tight-er"):
        with pytest.raises(CheckpointError):
            stage_completion_tag("bdt_au", bad_stage, _WHEN)


def test_L3_the_timestamp_is_ref_legal():
    """The ISO form's colons are not legal in a git ref, which is why the stamp
    is compact.  A ref with a colon is not a cosmetic problem — git refuses it."""
    stamp = utc_stamp(_WHEN)
    assert stamp == "20260806T221403Z"
    assert ":" not in stamp and " " not in stamp


def test_L4_a_hand_made_tag_is_not_mistaken_for_an_automatic_one():
    """Only stage completions are tagged by molbuilder; a user tagging by hand
    is their own business.  The parser must therefore recognise *its own* form
    and decline everything else, or a roll-up of "every checkpoint of this
    stage" would sweep in tags nobody meant that way."""
    for hand in ("v1", "before-the-rewrite", "bdt_au/tight",
                 "bdt_au/tight/not-a-time", "a/b/c/d",
                 "bdt_au/tight/20260806T221403"):        # missing the Z
        assert parse_stage_completion_tag(hand) is None


def test_L4_stage_tags_are_hierarchical_and_globbable(tmp_path):
    """`git tag --list '<id>/tight/*'` must answer "every checkpoint of one
    stage, oldest to newest" — the question a user returning to a mission
    actually asks.  Asserted against real git, not string manipulation."""
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")

    made = []
    for stage, second in (("coarse", 1), ("tight", 2), ("tight", 3)):
        when = _WHEN.replace(second=second)
        label = stage_completion_tag("bdt_au", stage, when)
        repo.tag(label, message=checkpoint_message("bdt_au", stage, "converged"))
        made.append(label)

    out = subprocess.run(["git", "tag", "--list", "bdt_au/tight/*"],
                         cwd=str(root), capture_output=True, text=True).stdout
    listed = out.split()
    assert listed == sorted(listed), "the stamp must sort oldest to newest"
    assert len(listed) == 2, f"expected only tight's tags, got {listed}"
    assert all(t.startswith("bdt_au/tight/") for t in listed)


def test_L4_a_colliding_stage_tag_is_refused_not_suffixed(tmp_path):
    """Two checkpoints of one stage inside the same second collide, and the
    contract says that is **refused, not suffixed** — like every other name in
    this design.  A silently suffixed tag would make the roll-up above return
    something the user never created."""
    root = _run_dir(tmp_path / "calc")
    repo = Repo(str(root))
    repo.init(engine="siesta")

    label = stage_completion_tag("bdt_au", "tight", _WHEN)
    repo.tag(label, message=checkpoint_message("bdt_au", "tight", "converged"))
    with pytest.raises(CheckpointError):
        repo.tag(label, message=checkpoint_message("bdt_au", "tight", "again"))
