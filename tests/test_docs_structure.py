"""Docs-structure contract — the migration ledger + the new-tree rules.

The docs tree was reorganized 2026-07-26 (docs/README.md § Structure): the
legacy tree is FROZEN at ``old_docs/`` and every doc migrates to the new
domain-structured ``docs/`` through a per-doc reconcile gate, tracked in
``docs/MIGRATION.md`` (the ledger).  This file makes the rules mechanical:

  1. FREEZE      — every file under old_docs/ has a ``pending`` ledger row
                   (nothing new may be born in the old tree), and every
                   ``pending`` row still points at a real old_docs file
                   (a moved doc must flip its row's status in the same
                   commit).
  2. R1 (index)  — every .md under docs/ (outside archive/) is linked from
                   docs/README.md, the ONE index.
  3. R2 (header) — every .md under docs/ (outside archive/, minus the two
                   index files) opens with the provenance header
                   (**Role:** ... / **Domain:** ...).
  4. Links       — every relative ``.md`` link inside the new tree resolves.

When the last ledger row closes and old_docs/ is deleted, drop rule 1 and
keep 2-4 (they are the permanent structure rules).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OLD = REPO / "old_docs"
LEDGER = DOCS / "MIGRATION.md"

# Index files: exempt from the provenance header (they ARE the index).
INDEX_FILES = {"README.md", "MIGRATION.md"}

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*([^|]+?)\s*\|")


def _ledger_rows() -> dict:
    """{old_docs-relative path: status} parsed from the ledger table."""
    rows = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m and m.group(1) != "old_docs/ file":       # skip the header row
            rows[m.group(1)] = m.group(2)
    return rows


def _old_files() -> set:
    return {p.relative_to(OLD).as_posix()
            for p in OLD.rglob("*") if p.is_file()}


def _new_tree_mds() -> list:
    """Every .md under docs/ outside archive/ (the rule-governed set)."""
    return [p for p in DOCS.rglob("*.md")
            if "archive" not in p.relative_to(DOCS).parts]


# --------------------------------------------------------------------- #
#  1. Freeze — old tree ↔ ledger                                        #
# --------------------------------------------------------------------- #


def test_every_old_docs_file_is_a_pending_ledger_row():
    if not OLD.is_dir():
        return  # migration complete; freeze rule retired
    rows = _ledger_rows()
    unlisted = sorted(_old_files() - set(rows))
    assert not unlisted, (
        "files in the FROZEN old_docs/ tree with no ledger row (new docs "
        f"must be born under docs/ — README.md R6): {unlisted}")
    stale = sorted(f for f in _old_files() if rows.get(f, "pending") != "pending")
    assert not stale, (
        "ledger rows marked moved/merged/archived but the file still "
        f"exists in old_docs/: {stale}")


def test_every_pending_ledger_row_still_exists():
    if not OLD.is_dir():
        return
    have = _old_files()
    gone = sorted(f for f, status in _ledger_rows().items()
                  if status == "pending" and f not in have)
    assert not gone, (
        "ledger rows still 'pending' but the old_docs/ file is gone — "
        f"flip the row's status in the same commit as the move: {gone}")


# --------------------------------------------------------------------- #
#  2. R1 — the ONE index                                                #
# --------------------------------------------------------------------- #


def test_every_new_tree_doc_is_indexed_in_readme():
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    missing = []
    for p in _new_tree_mds():
        rel = p.relative_to(DOCS).as_posix()
        if rel in INDEX_FILES:
            continue
        if f"({rel})" not in readme:
            missing.append(rel)
    assert not missing, (
        f"docs not linked from docs/README.md (rule R1): {sorted(missing)}")


# --------------------------------------------------------------------- #
#  3. R2 — provenance header                                            #
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
#  4. No dangling relative .md links in the new tree                    #
# --------------------------------------------------------------------- #

_LINK = re.compile(r"\]\(([^)#\s]+\.md)(#[^)]*)?\)")


def test_no_dangling_md_links_in_new_tree():
    dangling = []
    for p in DOCS.rglob("*.md"):
        for m in _LINK.finditer(p.read_text(encoding="utf-8")):
            target = m.group(1)
            if target.startswith(("http://", "https://")):
                continue
            if not (p.parent / target).resolve().is_file():
                dangling.append(f"{p.relative_to(REPO)} -> {target}")
    assert not dangling, f"dangling .md links: {dangling}"
