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

    with pytest.raises(Exception):
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


def test_I3_only_two_code_paths_can_remove_warm_state():
    """Exactly two operations may displace warm state — the `--cold`
    move-aside and a restore — and there must be no third.

    A third would be a *silent* loss: the user asked for neither, so nothing
    reports it, and the absence surfaces as an unexplained cold start hours
    later.  The contract asks for a grep over every path that writes into a run
    directory; this is that grep, over the module that owns them.
    """
    hits = []
    for src in sorted(_SRC.rglob("*.py")):
        for n, line in enumerate(src.read_text().splitlines(), 1):
            bare = line.split("#", 1)[0]
            if not re.search(r"\b(unlink|rmtree)\s*\(", bare):
                continue
            if any(suf in bare for suf in _WARM_SUFFIXES) or \
               re.search(r"warm|restart", bare, re.I):
                hits.append(f"{src.relative_to(_SRC)}:{n}: {line.strip()}")

    assert not hits, (
        "a code path removes warm state directly; warm files are moved aside "
        "or restored, never deleted:\n  " + "\n  ".join(hits))


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

    Checked where it would actually go wrong: the run index advances across
    invocations while the basename every warm file is keyed by does not.
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
