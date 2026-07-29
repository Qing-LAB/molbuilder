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

REPO = Path(__file__).resolve().parents[1]

SCAN_DIRS = ("molbuilder", "tests", "scripts")
SCAN_ROOT_FILES = ("README.md", "HANDOFF.md", "pyproject.toml")
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
        "docs/archive/MIGRATION.md maps old -> new), or cite "
        "docs/archive/old_docs/... explicitly for history:\n  "
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
