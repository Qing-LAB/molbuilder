"""No active source may cite a retired documentation path.

The 2026-07 docs migration moved every doc into the domain tree under
``docs/`` and archived the legacy layout to ``docs/archive/old_docs/``.
The closeout audit (docs/audit-2026-07-28-document-migration.md, P0)
found ~319 references in active code/tests still pointing at retired
locations; they were repointed 2026-07-29.  This test keeps it that way:

  1. No retired-layout path may appear in active sources — neither the
     retired directories (``docs/protocols/``, ``docs/types/``,
     ``docs/tabs/``) nor the retired root-level docs.  Directory
     prefixes are matched on their own so even a reference wrapped
     across source lines is caught.
  2. Stronger, future-proof: every ``docs/**.md`` path an active source
     mentions must EXIST on disk — so the next doc move can't strand
     references the way the migration did.

Archive content is historical evidence, not authority (archive
README): if a comment genuinely means the old document, it must say
``docs/archive/old_docs/…`` explicitly — that form is allowed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SCAN_DIRS = ("molbuilder", "tests", "scripts")
SCAN_ROOT_FILES = ("README.md", "pyproject.toml")
EXTS = {".py", ".js", ".html", ".css", ".md", ".toml", ".sh", ".json",
        ".cfg", ".yaml", ".yml"}

# Files whose text legitimately narrates the retired layout.
ALLOWLIST = {
    "tests/test_docs_structure.py",   # closeout history in its docstring
    "tests/test_no_retired_doc_paths.py",   # this file
}

# Retired locations.  ``docs/archive/old_docs/`` is the sanctioned way
# to cite history, so ``old_docs/`` only trips when NOT under archive.
RETIRED = re.compile(
    r"docs/(?:protocols|types|tabs)/"
    r"|docs/(?:config|deployment|README_install|installation"
    r"|job-execution|science|MIGRATION)\.md"
    r"|(?<![\w/])README_install\.md"
    r"|(?<!archive/)(?<![\w])old_docs/"
)

DOC_PATH = re.compile(r"docs/[A-Za-z0-9_\-./]+\.md")


def _scan_files():
    for base in SCAN_DIRS:
        for p in (REPO / base).rglob("*"):
            if (p.is_file() and p.suffix in EXTS
                    and "__pycache__" not in p.parts
                    and ".git" not in p.parts):
                yield p
    for name in SCAN_ROOT_FILES:
        p = REPO / name
        if p.is_file():
            yield p


def _read(p: Path):
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_no_active_source_cites_a_retired_doc_path():
    hits = []
    for p in _scan_files():
        rel = p.relative_to(REPO).as_posix()
        if rel in ALLOWLIST:
            continue
        text = _read(p)
        if text is None:
            continue
        for m in RETIRED.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(0)}")
    assert not hits, (
        "active sources cite RETIRED doc paths — repoint each to its "
        "owner in the domain tree (the migration ledger "
        "docs/archive/MIGRATION.md maps old -> new).\n"
        "  NOT a remedy: rewriting the citation as docs/archive/old_docs/... "
        "That form was recommended here until 2026-08-10 and it is the wrong "
        "answer for CODE — docs/README.md calls the archive 'Not a source of "
        "truth', so a docstring that specifies behaviour by citing it is "
        "describing live code with a document nobody maintains.  Cite the "
        "archive only to narrate history, never to specify.\n  "
        + "\n  ".join(sorted(hits)))


def test_every_cited_doc_path_exists():
    """Any ``docs/**.md`` an active source names must exist on disk."""
    missing = []
    for p in _scan_files():
        rel = p.relative_to(REPO).as_posix()
        if rel in ALLOWLIST:
            continue
        text = _read(p)
        if text is None:
            continue
        for m in DOC_PATH.finditer(text):
            if not (REPO / m.group(0)).is_file():
                line = text.count("\n", 0, m.start()) + 1
                missing.append(f"{rel}:{line}: {m.group(0)}")
    assert not missing, (
        "active sources cite docs paths that do not exist — a doc "
        "moved without its references (update them in the same "
        "commit):\n  " + "\n  ".join(sorted(missing)))


# ===================================================================== #
#  A bare filename is a citation too — and it was the blind spot        #
# ===================================================================== #
#
#  The two tests above catch a retired PATH (`docs/protocols/x.md`) and a
#  path that does not resolve.  A docstring saying `slurm-integration.md
#  § 4.3` is neither: no `docs/` prefix, so no pattern matches, and it reads
#  exactly like a citation of a live contract.
#
#  That is not hypothetical.  The 2026-07-29 sweep repointed ~319 PATH
#  references and this guard has been green ever since, while ~230 bare
#  citations of 29 archived-only documents sat untouched in the package.
#  Found 2026-08-10 while chasing one redundant import.


def _archived_only_basenames() -> set[str]:
    """Doc basenames that exist ONLY under ``docs/archive/``.

    A basename with a live twin (``design.md``, ``structure-periodicity.md``,
    ``README.md``) is excluded: a bare citation of one of those resolves to
    the live document, which is correct and common.  What is left can only
    mean the archived file.
    """
    docs = REPO / "docs"
    archived = {p.name for p in (docs / "archive").rglob("*.md")}
    live = {p.name for p in docs.rglob("*.md")
            if "archive" not in p.relative_to(docs).parts}
    return archived - live


_ARCHIVED_ONLY = _archived_only_basenames()
_BARE_ARCHIVED = re.compile(
    r"(?<![\w/.-])(" + "|".join(re.escape(n) for n in sorted(_ARCHIVED_ONLY))
    + r")") if _ARCHIVED_ONLY else None

# These narrate the archive rather than specifying against it.
_NARRATION_OK = {
    "tests/test_docs_structure.py",
    "tests/test_bench_generate.py",
    "tests/test_no_retired_doc_paths.py",
}


def measure_bare_archived_citations() -> list[str]:
    """``file:line: basename`` for every bare citation of an archived-only doc."""
    if _BARE_ARCHIVED is None:
        return []
    hits = []
    for p in _scan_files():
        rel = p.relative_to(REPO).as_posix()
        if rel in _NARRATION_OK or rel in ALLOWLIST:
            continue
        text = _read(p)
        if text is None:
            continue
        for m in _BARE_ARCHIVED.finditer(text):
            # The explicit `docs/archive/old_docs/...` form is excluded by the
            # lookbehind; this is the bare one.
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(1)}")
    return hits


@pytest.mark.xfail(strict=True, reason=(
    "OPEN — decision 32 in docs/archive/2026-08-19-staged-runs-implementation-plan.md "
    "§ 8.  277 bare citations of 31 archived-only documents; the biggest are "
    "parse-module.md (46) and job-execution.md (41).  slurm-integration.md was "
    "the largest and is now CLOSED — all 51 repointed to running-a-job.md and "
    "job-system.md.  It is also the worked example for the rest: a successor "
    "is found by reading what the live doc POINTS AT (job-system § 6 names its "
    "two owners in its first sentence), not by grepping successors for words "
    "you expect to see — that method gave the wrong answer and a "
    "recommendation to un-archive a correctly-superseded document.  Run "
    "`python -m tests.test_no_retired_doc_paths` for the current count.  "
    "Strict, so this fails loudly the moment the last one is resolved."))
def test_no_active_source_cites_an_archived_doc_as_authority():
    hits = measure_bare_archived_citations()
    assert not hits, (
        f"{len(hits)} bare citations of archived-only documents.  "
        "docs/README.md: the archive is 'Not a source of truth', so code "
        "specified by one is code with no maintained contract:\n  "
        + "\n  ".join(sorted(hits)[:40])
        + (f"\n  ... and {len(hits) - 40} more" if len(hits) > 40 else ""))


if __name__ == "__main__":                      # a number, on demand
    hits = measure_bare_archived_citations()
    by_doc: dict[str, int] = {}
    for h in hits:
        by_doc[h.rsplit(": ", 1)[1]] = by_doc.get(h.rsplit(": ", 1)[1], 0) + 1
    print(f"bare citations of archived-only docs: {len(hits)} "
          f"across {len(by_doc)} documents")
    for name, n in sorted(by_doc.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {name}")
