"""Docs-structure contract — the permanent new-tree rules.

The docs tree was reorganized 2026-07-26 (Waves 0–9) and closed out
2026-07-28 (Wave 10): the legacy ``old_docs/`` tree was archived verbatim to
``docs/archive/old_docs/`` and the migration ledger to
``docs/archive/MIGRATION.md``.  The freeze + keep-and-mark rules that policed
the migration (former rule 1: every old_docs file has a ledger row, the
``_migrated_`` filename mark agrees with the ledger status) are retired with
the migration itself — ``old_docs/`` no longer exists at the project root.

The permanent rules (test-enforced):

  1. R1 (index)  — every .md under docs/ (outside archive/) is linked from
                   docs/README.md, the ONE index.
  2. R2 (header) — every .md under docs/ (outside archive/, minus the index
                   file) opens with the provenance header
                   (**Role:** ... / **Domain:** ...).
  3. Links       — every internal doc link uses the document-module
                   convention (``?doc=<docs-root-path>``) and resolves.
  4. Citations   — every ``<doc>.md § N`` a document aims at another
                   document resolves to a real heading there.
  5. Tables      — no markdown table has rows orphaned from its header.

Rules 4 and 5 were added 2026-08-11, after a five-pass review found by hand
exactly what they now find automatically.  Both earn their place by what they
caught:

* **4** — ``running-a-job.md`` sent a reader to ``architecture.md`` § 7 for a
  config map that is in § 8, and four documents cited an **archived** file as
  the source of their open questions.  A citation is the one part of a contract
  a reader follows blindly.
* **5** — ``stages.md`` § 5's four-destination table was cut in half by a
  blockquote, so two rows rendered with no header; ``science/validation.md``'s
  **R5** and **R6** sat forty-six lines from theirs, at the end of an unrelated
  section.  A split table does not look broken in the source and is unreadable
  in the browser, which is why it survived so long.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# Index files: exempt from the provenance header (they ARE the index).
INDEX_FILES = {"README.md"}


def _new_tree_mds() -> list:
    """Every .md under docs/ outside archive/ (the rule-governed set)."""
    return [p for p in DOCS.rglob("*.md")
            if "archive" not in p.relative_to(DOCS).parts]


# --------------------------------------------------------------------- #
#  1. R1 — the ONE index                                                #
# --------------------------------------------------------------------- #


def test_every_new_tree_doc_is_indexed_in_readme():
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    missing = []
    for p in _new_tree_mds():
        rel = p.relative_to(DOCS).as_posix()
        if rel in INDEX_FILES:
            continue
        # Index links use the document-module convention: ?doc=<docs-rel-path>
        # (the Documents tab serves docs via /documents?doc=..., never the raw
        # .md path — see docs/README.md § the link convention).
        if f"(?doc={rel})" not in readme:
            missing.append(rel)
    assert not missing, (
        f"docs not linked from docs/README.md (rule R1): {sorted(missing)}")


# --------------------------------------------------------------------- #
#  2. R2 — provenance header                                            #
# --------------------------------------------------------------------- #


def test_every_new_tree_doc_has_a_provenance_header():
    bad = []
    for p in _new_tree_mds():
        rel = p.relative_to(DOCS).as_posix()
        if rel in INDEX_FILES:
            continue
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:15])
        if "**Role:**" not in head or "**Domain:**" not in head:
            bad.append(rel)
    assert not bad, (
        "docs missing the provenance header (**Role:** / **Domain:** in "
        f"the first 15 lines — rule R2): {sorted(bad)}")


# --------------------------------------------------------------------- #
#  3. Internal doc links use the document-module convention + resolve    #
# --------------------------------------------------------------------- #
#
# The Documents tab serves docs through the module (/documents?doc=<rel>),
# never as raw .md paths (a raw relative .md href 404s in the rendered view).
# So every internal doc link must be ``?doc=<docs-root-relative-path>`` and
# that path must resolve to a real file under docs/.

_LINK = re.compile(r"\]\(([^)\s]+)\)")   # any link target


def test_internal_doc_links_use_doc_module_convention():
    dangling = []       # ?doc= links whose target file is missing
    raw = []            # raw relative .md links that ignore the convention
    for p in DOCS.rglob("*.md"):
        # archive/ is verbatim history: its docs' internal links point at
        # the tree layout of THEIR day and may dangle by design.  Only the
        # archive's own index stays link-checked.
        rel_parts = p.relative_to(DOCS).parts
        if "archive" in rel_parts and p.name != "README.md":
            continue
        # old_docs/ under archive/ is a verbatim frozen snapshot;
        # its internal links point at the old tree layout.
        if "old_docs" in rel_parts:
            continue
        for m in _LINK.finditer(p.read_text(encoding="utf-8")):
            target = m.group(1)
            if target.startswith(("http://", "https://", "#")):
                continue
            if target.startswith("?doc="):
                doc_path = target[len("?doc="):].split("#", 1)[0]
                if not (DOCS / doc_path).resolve().is_file():
                    dangling.append(f"{p.relative_to(REPO)} -> {target}")
            elif target.split("#", 1)[0].endswith(".md"):
                raw.append(f"{p.relative_to(REPO)} -> {target}")
    assert not dangling, f"dangling ?doc= links (target missing): {dangling}"
    assert not raw, (
        "raw relative .md links must use the document-module convention "
        f"`?doc=<docs-root-path>` (the Documents tab serves via ?doc=): {raw}")


# --------------------------------------------------------------------- #
#  4. Citations — a "§ N" that points somewhere must arrive             #
# --------------------------------------------------------------------- #

#: ``[`a/b.md`](?doc=…)`` -> ``` `b.md` ```, so the gap between the name and a
#: following ``§`` is prose rather than a URL full of dots.
_DOC_LINK = re.compile(r"\[`?([^\]`]+?)`?\]\(\?doc=[^)]*\)")

#: A citation is the doc's name and a section with only spaces, commas or
#: backticks between them.  Anything longer is a new sentence, and its ``§``
#: belongs to the citing document -- e.g. "…see `project-layout.md`. § 7.1
#: below defers to it", where § 7.1 is the CITING file's own section.
_CITATION = re.compile(
    r"`?([A-Za-z0-9_./-]+\.md)`?[ ,]{0,3}§+\s*([0-9]+(?:\.[0-9a-z]+)*)")

_HEADING = re.compile(r"^#{2,6}\s+§?\s*([0-9]+(?:\.[0-9a-z]+)*[a-z]?)\b")
_HEADING_DOTTED = re.compile(r"^#{2,6}\s+([0-9]+[a-z]?)\.\s")


def _headings(path: Path) -> set:
    found = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for pat in (_HEADING, _HEADING_DOTTED):
            m = pat.match(line)
            if m:
                found.add(m.group(1))
    return found


def test_every_cross_document_section_citation_resolves():
    """A reader follows a `§` without checking it, so a wrong one is worse
    than none: it spends their time and returns them a false answer.

    Only citations aimed at a document whose *name is unique* in the tree are
    checked -- an ambiguous basename cannot be resolved to one file, and
    guessing would make this guard the thing that is wrong.  Archived docs are
    skipped as targets: they are frozen, so their sections cannot be repaired.
    """
    docs = _new_tree_mds()
    by_name = {}
    for p in docs:
        by_name.setdefault(p.name, []).append(p)
    headings = {p: _headings(p) for p in docs}

    broken = []
    for p in docs:
        text = _DOC_LINK.sub(lambda m: f"`{m.group(1)}`",
                             p.read_text(encoding="utf-8", errors="replace"))
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in _CITATION.finditer(line):
                name, section = Path(m.group(1)).name, m.group(2)
                targets = by_name.get(name)
                if not targets or len(targets) != 1:
                    continue
                target = targets[0]
                if target == p:
                    continue
                if section not in headings[target]:
                    broken.append(
                        f"{p.relative_to(DOCS)}:{lineno} -> {name} § {section}")

    assert not broken, (
        "these citations name a section that does not exist in the target:\n  "
        + "\n  ".join(sorted(broken))
        + "\n\nEither the section moved (repoint the citation) or it was "
          "renumbered (find where the content went -- do NOT just delete the "
          "number, which is how a pointer becomes prose nobody can follow).")


# --------------------------------------------------------------------- #
#  5. Tables — no row orphaned from its header                          #
# --------------------------------------------------------------------- #


def test_no_markdown_table_has_rows_orphaned_from_its_header():
    """A run of ``|`` lines whose second line is not ``---`` renders as a
    table with no header -- every cell shifts up one and the column meanings
    are lost.

    It happens when prose or a blockquote is written BETWEEN two rows, which
    looks entirely reasonable in the source.  The repair is to move the
    interruption below the completed table, never to delete rows.
    """
    orphans = []
    for p in _new_tree_mds():
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].lstrip().startswith("|"):
                i += 1
                continue
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            block = lines[start:i]
            if len(block) < 2 or "---" not in block[1]:
                orphans.append(
                    f"{p.relative_to(DOCS)}:{start + 1}  {block[0][:70]}")

    assert not orphans, (
        "these table rows have no header above them -- something was written "
        "between two rows of one table:\n  " + "\n  ".join(orphans)
        + "\n\nMove the interrupting prose or blockquote BELOW the whole "
          "table; do not drop the rows.")
