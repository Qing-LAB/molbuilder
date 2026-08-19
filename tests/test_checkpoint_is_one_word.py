"""The tool is called **checkpoint**, everywhere a person can see it.

Contract: [`checkpointing.md`](docs/execution/checkpointing.md) — the feature,
the module, the blueprint, the HTTP routes and the UI have always said
*checkpoint*.  The CLI group said *snapshot* until 2026-08-18, which is the
defect this file exists to keep closed: **one tool, one word.**

**What is deliberately NOT covered, and why.** ``snapshot`` is an ordinary
English noun and the contract uses it as one — *"a state is a saved snapshot of
the whole folder"*.  This checks the places a name is a NAME: the command a
person types, and the commands molbuilder tells them to type.  It does not
police prose, because a rule that did would push the documents into worse
English to satisfy a grep.

**And two on-disk names stay as they are** (user, 2026-08-18): the archive
directory ``.binsnapshots/`` and the ``.gitignore`` markers.  They live inside
calculation folders that already exist; renaming them would orphan real
archives to no one's benefit, and neither is a name a person is asked to type.
That is the line: **a verb someone types is the tool's name; a file on disk is
the data's.**
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from molbuilder import cli as _cli

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "molbuilder"

#: The verbs the group offers.  Named here so a renamed or dropped subcommand
#: shows up as a failure rather than as a silently smaller tool.
SUBCOMMANDS = {"init", "save", "list", "tag", "restore", "config"}


def test_the_group_a_person_types_is_checkpoint():
    assert "checkpoint" in _cli.cli.commands, sorted(_cli.cli.commands)
    assert "snapshot" not in _cli.cli.commands, (
        "the old verb is still registered; a rename is a rename, not an alias "
        "(no-backward-compat rule)")


def test_the_group_offers_exactly_these_verbs():
    assert set(_cli.cli.commands["checkpoint"].commands) == SUBCOMMANDS


#: Files whose ``snapshot`` mentions are the on-disk names, the English noun,
#: or an unrelated function (``system_load.snapshot``).  Everything else in the
#: package is searched for a COMMAND spelled with the retired verb.
_COMMAND_RE = re.compile(
    r"(?<!bin)snapshot[s]?\s+(?:init|save|list|tag|restore|config)\b")


def _python_sources():
    for p in sorted(PKG.rglob("*.py")):
        yield p
    for p in sorted((REPO / "tests").glob("*.py")):
        if p.name != Path(__file__).name:
            yield p


def test_nothing_names_a_command_with_the_retired_verb():
    """**The one that would have caught the live bug.**

    ``molbuilder checkpoint list`` printed *"`snapshot save` keeps them"* after
    the rename — a command that no longer exists, in output a person reads at
    exactly the moment they are deciding whether to save.  The rename had
    replaced ``molbuilder snapshot`` and left the bare spellings.
    """
    offences = []
    for path in _python_sources():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _COMMAND_RE.search(line):
                offences.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offences, (
        "a command is spelled with the retired verb:\n  "
        + "\n  ".join(offences)
        + "\n\nThe tool is `molbuilder checkpoint`.  If the word here is the "
          "ordinary noun, reword it so it does not read as a command.")


@pytest.mark.parametrize("marker", ["# === molbuilder checkpoint BEGIN ===",
                                    "# === molbuilder checkpoint END ==="])
def test_the_gitignore_markers_are_what_existing_folders_hold(marker):
    """**A persisted name is not a rename's business.**

    These delimit the managed region of a calculation's ``.gitignore``, and the
    writer finds the existing block BY them: change the string and an already
    initialised folder grows a second block on its next save while the stale
    one keeps excluding files.  Pinned so a future sweep of the word cannot
    reach them by accident.
    """
    from molbuilder import checkpoint as _ck
    assert marker in (_ck._GITIGNORE_BEGIN, _ck._GITIGNORE_END)


def test_the_archive_directory_keeps_its_name():
    """Same rule, same reason: ``.binsnapshots/`` sits inside folders that
    already exist, and it names the DATA -- whole copies of the large files --
    rather than the tool."""
    from molbuilder import checkpoint as _ck
    assert _ck.ARCHIVE_DIR == ".binsnapshots"
