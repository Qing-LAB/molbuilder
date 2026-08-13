"""No git on a compute node (I4).

**Contract:** [`checkpointing.md`](?doc=execution/checkpointing.md) I4 — *a
generated wrapper contains no git*, because a wrapper that committed would need
git on the compute node, which `running-a-job.md § 2` forbids.

**The contract's test is about the EMITTED SCRIPT, and it used to be missing.**
Everything here once checked Python import graphs — that `runwrap.py` does not
import the checkpoint module. That is a real property and it is kept below, but
it is not I4: a template could hard-code a `git rev-parse` line into the bash it
emits without importing anything, and the import check stays green while the
`.sbatch` dies on a node with no git. The rule is about what lands in the file
that SLURM runs, so that is what is read.

I4's own wording sets the bar for *how* to read it: `git` must be matched **as a
command word** — `digits` and `logging` are not violations, and a check that
flags them is one somebody will disable.
"""
from __future__ import annotations

import inspect
import json
import re
import shutil
import subprocess

import pytest

from molbuilder.runwrap import render_run_wrapper, write_run_wrapper


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The renders resolve script_generation config from cwd + HOME/XDG
    (G-3, 2026-08-13): unsandboxed, every wrapper here was rendered with
    the developer's repo-root molbuilder.json folded in, so what I4 was
    checked against varied by machine -- and failed in an isolated cwd
    (the writer rightly REFUSES with no activation declared, which is
    exactly what running the file isolated exposed).  One sandboxed cwd +
    HOME for the file, with the activation DECLARED by the test."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cwd = tmp_path_factory.mktemp("cwd")
    (cwd / "molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    monkeypatch.chdir(cwd)


#: `git` as a COMMAND WORD: at the start of a line, after a shell operator
#: (`|`, `&&`, `;`, `$(`, backtick…), after a keyword that introduces a command
#: (`then`, `do`, `else`), or after a **runner that executes its argument**.
#: Never inside another identifier.  This is I4's own instruction, and the
#: reason for it is that a blunt substring search flags `digits` and gets
#: switched off.
#:
#: The runner alternative was added in the second review.  `xargs git push` and
#: `sudo git commit` invoke git exactly as much as a bare `git` does, and the
#: pattern saw neither -- so a wrapper could have shipped one and stayed green
#: while dying on a node with no git, which is the whole failure I4 names.
_GIT_COMMAND = re.compile(
    r"(?:^|[|&;(`]|\$\(|\b(?:then|do|else|xargs|sudo|env|time|nohup|exec"
    r"|command|eval)\b)\s*git\b",
    re.MULTILINE)


#: `git` as a bare word, for the emitted files that are not shell scripts.
#: Still word-bounded rather than a substring search -- "digits" contains the
#: three letters, which is the trap I4 names.
_GIT_WORD = re.compile(r"\bgit\b")


def _emit(tmp_path, **kwargs):
    """Write a real wrapper for a real script and return every emitted file."""
    script = tmp_path / "job.fdf"
    script.write_text("SystemLabel job\n")
    write_run_wrapper(script, env="molbuilder-siesta", **kwargs)
    return [p for p in tmp_path.iterdir() if p.is_file() and p != script]


def _shell(files):
    """The emitted files bash actually runs.

    A wrapper drops more than scripts beside the job -- `mb_monitor.py` rides
    along and is Python.  Reading that as bash proves nothing; it gets the
    word-level check below instead, because it runs on the node too.
    """
    return [p for p in files
            if p.name.endswith((".run.sh", ".sbatch"))]


# ------------------------------------------------------------------ #
#  I4 — the emitted script                                            #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("kwargs", [
    {"emit_sbatch": False},
    {"emit_sbatch": True, "time": "01:00:00", "mem": "8G", "cpus_per_task": 4},
    {"emit_sbatch": False, "continue_retries": 2},
    {"emit_sbatch": False, "mpi_np": 4, "omp_threads": 2},
])
def test_no_emitted_wrapper_invokes_git(tmp_path, kwargs):
    """I4's stated test, over every shape the generator produces.

    A wrapper runs where there may be no git at all, so a `git` command word in
    one is not a style problem — it is a job that fails on the node, after the
    queue time was already spent.
    """
    emitted = _emit(tmp_path, **kwargs)
    assert _shell(emitted), "the generator emitted no shell script to check"
    for script in _shell(emitted):
        text = script.read_text()
        hit = _GIT_COMMAND.search(text)
        assert hit is None, (
            f"{script.name} invokes git as a command word "
            f"({text[max(0, hit.start() - 40):hit.end() + 40]!r}). A compute "
            f"node is not required to have git (checkpointing.md I4).")

    # Everything else that rides along to the node -- today the monitor -- gets
    # the word-level check.  It runs there as well, so it may not need git
    # either, and the shell-operator pattern would not see a Python call.
    for other in emitted:
        if other in _shell(emitted):
            continue
        hit = _GIT_WORD.search(other.read_text())
        assert hit is None, (
            f"{other.name} is shipped to the compute node and mentions git "
            f"(checkpointing.md I4).")


def test_the_check_does_not_fire_on_words_that_merely_contain_git(tmp_path):
    """I4 names this trap itself: `digits` and `logging` are not violations.

    A check that flags them is one somebody disables, and a disabled check
    protects nothing.  Asserted directly so the pattern above cannot be
    "tightened" into uselessness.
    """
    innocent = ("echo digits\nLOG=logging.txt\necho legit\n"
                "# the legitimate word gitlab appears in a comment\n"
                "MSG=\"no digits here\"\n")
    assert _GIT_COMMAND.search(innocent) is None
    for guilty in ("git rev-parse HEAD\n", "  git add -A\n",
                   "x=$(git status)\n", "true && git commit\n",
                   "if true; then git log; fi\n",
                   # A runner still runs it.  These were invisible until the
                   # second review, and each is a real way to ship git into a
                   # wrapper without typing it at the start of a line.
                   "xargs git push\n", "sudo git commit\n", "env git status\n",
                   "exec git log\n", "command git add .\n"):
        assert _GIT_COMMAND.search(guilty) is not None, guilty


def test_the_rendered_text_is_what_gets_written(tmp_path):
    """`render_*` and `write_*` must not drift, or the check reads a draft.

    Compared without the PROVENANCE timestamp (the 67cee62a class, G-3):
    generated-at is wall clock at seconds precision and the two renders
    run in sequence, so byte equality flaked across a second boundary."""
    def _logic(text: str) -> str:
        return "\n".join(l for l in text.splitlines()
                         if not l.lstrip("# ").startswith("generated-at"))
    script = tmp_path / "job.fdf"
    script.write_text("SystemLabel job\n")
    rendered = render_run_wrapper(script, env="molbuilder-siesta")
    written = write_run_wrapper(script, env="molbuilder-siesta",
                                emit_sbatch=False)
    assert _logic(written.read_text()) == _logic(rendered)


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not on PATH")
def test_the_emitted_wrapper_is_valid_bash(tmp_path):
    """A syntax check, because everything above reads the file as text.

    A wrapper that greps clean and does not parse fails just as hard on the
    node, and `bash -n` is the cheapest way to know it is a script at all.
    """
    for script in _shell(_emit(tmp_path, emit_sbatch=False)):
        result = subprocess.run(["bash", "-n", str(script)],
                                capture_output=True, text=True)
        assert result.returncode == 0, (
            f"{script.name} is not valid bash:\n{result.stderr}")


# ------------------------------------------------------------------ #
#  The coupling check — kept, but it is not I4                        #
# ------------------------------------------------------------------ #


def test_runwrap_does_not_import_checkpoint():
    """Defence in depth, one layer above the emitted script.

    If the wrapper generator cannot reach the checkpoint module, it cannot grow
    a call that emits git by accident.  This does not replace the check above:
    hard-coded bash needs no import.
    """
    import molbuilder.runwrap as runwrap
    src = inspect.getsource(runwrap)
    for needle in ("from molbuilder.checkpoint", "import molbuilder.checkpoint",
                   "from .checkpoint", "import .checkpoint"):
        assert needle not in src, (
            f"runwrap.py contains {needle!r}; the wrapper must be git-agnostic "
            f"(checkpointing.md I4).")


def test_checkpoint_module_not_in_runwrap_namespace():
    """The same, at runtime: nothing from checkpoint is reachable via runwrap."""
    import molbuilder.runwrap as runwrap
    for name in ("Repo", "checkpoint", "CheckpointError", "Checkpoint"):
        value = getattr(runwrap, name, None)
        if value is None:
            continue
        assert getattr(value, "__module__", None) != "molbuilder.checkpoint", (
            f"runwrap.{name} resolves to molbuilder.checkpoint.{name}.")


def test_checkpoint_module_does_not_import_runwrap():
    """Loose coupling in both directions; runwrap is heavy."""
    import molbuilder.checkpoint as cp
    src = inspect.getsource(cp)
    assert "from molbuilder.runwrap" not in src
    assert "import molbuilder.runwrap" not in src
