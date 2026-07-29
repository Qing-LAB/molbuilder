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
