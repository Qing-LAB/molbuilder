"""**A contract must not say the same thing twice.**

`web/molview.md` § 11.6 carried **109 identical lines twice**, 155 lines apart
— the whole *"What it reads / What it shows / when the marks appear / what a
consumer may reach"* passage — and then the two copies diverged into **two
statements of one rule**, quoting two different things the user said on the
same day.  One had a door table, the other an exemption paragraph.  Neither was
wrong; they were simply two homes for one fact, which is the thing this project
does not allow in code and had not been checking in prose.

Nothing announced it.  It surfaced only because a paragraph was being added to
that section on 2026-09-03 and there were two places it could go.

**Why a run and not a line.**  Repeating one sentence is often deliberate — a
rule restated where a reader meets it, a heading reused, a table row that reads
the same in two tables.  A long consecutive RUN is different: it can only be a
copy, because nobody writes fifteen identical lines twice on purpose.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

#: A line short enough to repeat innocently ("---", "| | |", "```", a stub row).
_SUBSTANTIVE = 55

#: How many consecutive substantive lines make a copy rather than a coincidence.
#: The molview duplication was 109; a fenced example block or a repeated
#: worked-example is nowhere near this long.
_RUN = 15


def _live_docs():
    return [p for p in sorted(DOCS.rglob("*.md"))
            if "archive" not in p.relative_to(DOCS).parts]


def _longest_repeated_run(lines):
    """The longest run of substantive lines that occurs twice, and where.

    Substantive lines only, so a stretch of blank lines or table separators
    cannot be mistaken for repeated prose.
    """
    marked = [(i, l.strip()) for i, l in enumerate(lines)
              if len(l.strip()) >= _SUBSTANTIVE]
    seen: dict[str, int] = {}
    best = (0, None, None)
    for pos in range(len(marked)):
        text = marked[pos][1]
        if text not in seen:
            seen[text] = pos
            continue
        # A repeat: walk both forward while they agree.
        first = seen[text]
        n = 0
        while (pos + n < len(marked)
               and marked[first + n][1] == marked[pos + n][1]):
            n += 1
        if n > best[0]:
            best = (n, marked[first][0] + 1, marked[pos][0] + 1)
        seen.setdefault(text, first)
    return best


def test_no_live_document_repeats_a_long_passage():
    offenders = []
    for p in _live_docs():
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        n, first, second = _longest_repeated_run(lines)
        if n >= _RUN:
            offenders.append(
                f"{p.relative_to(DOCS)}: {n} substantive lines repeated — "
                f"line {first} and line {second}")

    assert not offenders, (
        "these documents contain a copied passage:\n  " + "\n  ".join(offenders)
        + "\n\nTwo copies of one passage drift: an edit lands in one of them, "
          "and a reader believes whichever they reach first.  Merge them, "
          "keeping anything the second copy says that the first does not — "
          "in molview.md § 11.6 the second copy was the only place carrying "
          "`requireMatch` and one of the two user quotations."
    )


def test_the_detector_finds_a_copy_when_there_is_one():
    """The mutation guard.  A detector that answered "no run" for everything
    would make the test above vacuously green — which is exactly how the
    duplication survived: nothing was looking."""
    body = [f"a substantive line of prose, number {i}, long enough to count"
            for i in range(20)]
    doc = ["intro"] + body + ["something else entirely in between"] + body
    n, first, second = _longest_repeated_run(doc)
    assert n == 20, f"the copy was not found: {n}"
    assert first == 2 and second == 23, (first, second)

    # And it does not cry copy over ordinary prose.
    n2, _, _ = _longest_repeated_run(["intro"] + body + ["tail"])
    assert n2 == 0, f"unique prose read as a repeat: {n2}"
