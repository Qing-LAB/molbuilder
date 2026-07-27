"""Docs-structure contract — the migration ledger + the new-tree rules.

The docs tree was reorganized 2026-07-26 (docs/README.md § Structure): the
legacy tree is FROZEN at ``old_docs/`` and every doc migrates to the new
domain-structured ``docs/`` through a per-doc reconcile gate, tracked in
``docs/MIGRATION.md`` (the ledger).

**Keep-and-mark policy (user, 2026-07-26).** Migrated old docs are NOT
deleted during the migration — they are kept in ``old_docs/`` and *marked
done* by prefixing the filename with ``_migrated_`` (e.g.
``protocols/web-api.md`` → ``protocols/_migrated_web-api.md``). The whole
frozen tree therefore survives until closeout, when the new tree is
cross-checked against it to prove nothing was missed; only then is
``old_docs/`` deleted. The ledger row still records the outcome
(moved / merged-into / archived); the prefix is the filesystem mirror of
"this one is done".

This file makes the rules mechanical:

  1. FREEZE      — every file under old_docs/ (prefix stripped) has a ledger
                   row (nothing new may be born in the old tree), and the
                   filesystem mark agrees with the ledger status: a
                   ``_migrated_``-prefixed file has a non-``pending`` status,
                   an un-prefixed file is still ``pending``.
  2. R1 (index)  — every .md under docs/ (outside archive/) is linked from
                   docs/README.md, the ONE index.
  3. R2 (header) — every .md under docs/ (outside archive/, minus the two
                   index files) opens with the provenance header
                   (**Role:** ... / **Domain:** ...).
  4. Links       — every relative ``.md`` link inside the new tree resolves.

Grandfathered exception: the Wave-H archives were relocated verbatim
(``git mv`` into docs/archive/) BEFORE this policy, so their old_docs files
are gone. A non-``pending`` row whose file is simply absent is allowed
(rule 1) — the keep-and-mark rule binds only files that still exist.

When old_docs/ is deleted at closeout, drop rule 1 and keep 2-4 (they are
the permanent structure rules).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OLD = REPO / "old_docs"
LEDGER = DOCS / "MIGRATION.md"

# Filename prefix marking an old_docs file as "migrated / worked on".
DONE_PREFIX = "_migrated_"

# Index files: exempt from the provenance header (they ARE the index).
INDEX_FILES = {"README.md", "MIGRATION.md"}

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*([^|]+?)\s*\|")


def _ledger_rows() -> dict:
    """{old_docs-relative logical path: status} parsed from the ledger table."""
    rows = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m and m.group(1) != "old_docs/ file":       # skip the header row
            rows[m.group(1)] = m.group(2)
    return rows


def _logical(rel: str) -> tuple[str, bool]:
    """Map an old_docs path to (logical original path, is_marked_done).

    Strips the DONE_PREFIX from the basename so the ledger key (the original
    name) is recovered.
    """
    parent, _, name = rel.rpartition("/")
    marked = name.startswith(DONE_PREFIX)
    if marked:
        name = name[len(DONE_PREFIX):]
    return (f"{parent}/{name}" if parent else name), marked


def _old_files() -> dict:
    """{logical old_docs path: is_marked_done} for every file under old_docs/."""
    out = {}
    for p in OLD.rglob("*"):
        if p.is_file():
            logical, marked = _logical(p.relative_to(OLD).as_posix())
            out[logical] = marked
    return out


def _new_tree_mds() -> list:
    """Every .md under docs/ outside archive/ (the rule-governed set)."""
    return [p for p in DOCS.rglob("*.md")
            if "archive" not in p.relative_to(DOCS).parts]


# --------------------------------------------------------------------- #
#  1. Freeze + keep-and-mark — old tree ↔ ledger                        #
# --------------------------------------------------------------------- #


def test_every_old_docs_file_is_a_ledger_row():
    if not OLD.is_dir():
        return  # migration complete; freeze rule retired
    rows = _ledger_rows()
    unlisted = sorted(f for f in _old_files() if f not in rows)
    assert not unlisted, (
        "files in the FROZEN old_docs/ tree with no ledger row (new docs "
        f"must be born under docs/ — README.md R6): {unlisted}")


def test_mark_agrees_with_ledger_status():
    """A _migrated_-prefixed file must be non-pending; an un-prefixed file
    must be pending. This keeps the filesystem mark and the ledger honest."""
    if not OLD.is_dir():
        return
    rows = _ledger_rows()
    prefixed_but_pending, unprefixed_but_done = [], []
    for logical, marked in _old_files().items():
        status = rows.get(logical, "pending")
        if marked and status == "pending":
            prefixed_but_pending.append(logical)
        if not marked and status != "pending":
            unprefixed_but_done.append(logical)
    assert not prefixed_but_pending, (
        "old_docs files prefixed `_migrated_` but still `pending` in the "
        f"ledger — set the ledger status in the same commit: {sorted(prefixed_but_pending)}")
    assert not unprefixed_but_done, (
        "ledger says migrated (moved/merged/archived) but the old_docs file "
        "is NOT prefixed `_migrated_` — rename it to mark it done (keep-and-"
        f"mark policy): {sorted(unprefixed_but_done)}")


def test_every_pending_ledger_row_still_exists():
    if not OLD.is_dir():
        return
    have = _old_files()
    gone = sorted(f for f, status in _ledger_rows().items()
                  if status == "pending" and f not in have)
    assert not gone, (
        "ledger rows still 'pending' but the old_docs/ file is gone — a "
        "migrated file must be KEPT and prefixed `_migrated_`, and its row "
        f"status set, in the same commit: {gone}")


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
        # Index links use the document-module convention: ?doc=<docs-rel-path>
        # (the Documents tab serves docs via /documents?doc=..., never the raw
        # .md path — see docs/README.md § the link convention).
        if f"(?doc={rel})" not in readme:
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
#  4. Internal doc links use the document-module convention + resolve    #
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
