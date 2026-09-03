"""No active source may cite a retired documentation path.

The 2026-07 docs migration moved every doc into the domain tree under
``docs/`` and archived the legacy layout to ``docs/archive/old_docs/``.
The closeout audit (docs/archive/2026-07-28-document-migration.md, P0)
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
    # (?<!old_): ``docs/archive/old_docs/tabs/…`` is the sanctioned
    # history citation (the comment above), and ``old_docs/tabs/``
    # CONTAINS the substring ``docs/tabs/`` -- without the lookbehind
    # this arm flagged exactly the citations the old_docs arm permits.
    r"(?<!old_)docs/(?:protocols|types|tabs)/"
    # Same collision, one arm down: ``old_docs/job-execution.md`` ENDS
    # WITH ``docs/job-execution.md``, so without this lookbehind the
    # sanctioned history citation trips the retired-root-level arm --
    # and the two tests then disagree about the one form each requires.
    r"|(?<!old_)docs/(?:config|deployment|README_install|installation"
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


def test_no_active_source_cites_an_archived_doc_as_authority():
    """No active source may specify its behaviour by citing an archived-only
    document.

    **CLOSED 2026-09-03** (decision 32 in
    ``docs/archive/2026-08-19-staged-runs-implementation-plan.md`` § 8).  This
    was ``xfail(strict=True)`` at 277 bare citations of 31 documents; it is now
    an ordinary guard, and the count is zero.

    An archive records how a decision was REACHED.  A citation that specifies
    live behaviour needs a document somebody maintains -- ``docs/README.md``
    calls the archive "Not a source of truth", so code specified by one is code
    with no maintained contract, and a rule that moves becomes invisible.  Each
    citation was resolved by reading the code it sits beside, checking whether
    the live contract already stated the rule, WRITING it there when it did
    not, and only then re-pointing.

    Three failure modes turned up, and they want different fixes -- which is
    why "grep and replace the name" was never the method:

    1. **A pointer aimed at the wrong document.**  The successor is found by
       reading what the live doc POINTS AT (``job-system`` § 6 names its two
       owners in its first sentence), not by grepping successors for words you
       expect to see -- that method gave the wrong answer and a recommendation
       to un-archive a correctly-superseded document.
       (``slurm-integration.md``, 51.)
    2. **Prose that had gone false under a correct pointer.**  Half of
       ``parse-module.md``'s 46 were module headers saying the legacy
       ``molbuilder/parsers/`` package "stays in place until H4" -- H4b deleted
       it on 2026-06-21.  A reader was being told a parallel implementation
       exists.  Those became one provenance line each; only the rule-citing
       ones moved.
    3. **A rule with no live home at all.**  The citation was load-bearing and
       nothing maintained said it -- so the rule was written from the code
       first (the app shell, the poll's keep-on-``undefined`` apply rule, the
       two bias walks' opposite failure policies, the five spectra selectors,
       "no silent absorption").

    Narrating history is still fine, and is what the sanctioned
    ``docs/archive/old_docs/...`` form is for; specifying against it is not.
    """
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
