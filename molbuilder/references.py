"""The bibliography door -- one reader over ``docs/science/references.bib``.

THE ONE HOME for citations (user, 2026-08-21): every scientific argument
in the validation design starts from a confirmed reference, and this
module is how code reaches it.  Catalogue items name keys
(``Item.refs``); the form resolves them here so the help expander shows
the real title and DOI; ``tests/test_catalogue_refs.py`` pins every
named key to an entry, so an invented citation fails CI instead of
reaching a user.  Engine-manual citations are NOT here -- they ride the
catalogue's ``manual`` key, which names the manual and section directly.

Deliberately a minimal parser: it reads the fields a person needs to
FIND the paper (author, title, journal, volume, pages, year, doi) and
nothing else.  The .bib file stays the source of truth for BibTeX
consumers; this module never writes it.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

#: The bib's one location -- beside the science docs it serves.
BIB_PATH = Path(__file__).resolve().parent.parent / "docs" / "science" / "references.bib"

_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)
_FIELD_RE = re.compile(
    r"^\s*(author|title|journal|booktitle|publisher|volume|pages|year|doi)"
    r"\s*=\s*\{(.*?)\}\s*,?\s*$",
    re.M | re.S)


def _clean(v: str) -> str:
    v = re.sub(r"\s+", " ", v).strip()
    return v.replace("{", "").replace("}", "")


@lru_cache(maxsize=1)
def _entries() -> Dict[str, Dict[str, str]]:
    text = BIB_PATH.read_text(encoding="utf-8")
    out: Dict[str, Dict[str, str]] = {}
    marks = list(_ENTRY_RE.finditer(text))
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks)
                    else len(text)]
        fields = {k: _clean(v) for k, v in _FIELD_RE.findall(body)}
        out[m.group(2)] = fields
    return out


def known_keys() -> frozenset:
    """Every key the bib defines -- what the CI pin checks against."""
    return frozenset(_entries())


def citation_for(key: str) -> Optional[Dict[str, str]]:
    """The fields a person needs to find the paper, or ``None`` for an
    unknown key (the caller decides whether that is an error; the form
    simply omits, because the TEST is where an unknown key must fail)."""
    e = _entries().get(key)
    if e is None:
        return None
    parts = []
    if e.get("author"):
        first = e["author"].split(" and ")[0]
        parts.append(first + (" et al." if " and " in e["author"] else ""))
    for f in ("journal", "booktitle", "publisher"):
        if e.get(f):
            parts.append(e[f])
            break
    if e.get("volume"):
        parts.append("vol. " + e["volume"])
    if e.get("year"):
        parts.append("(" + e["year"] + ")")
    return {
        "key":   key,
        "title": e.get("title", ""),
        "text":  ", ".join(parts),
        "doi":   e.get("doi", ""),
    }
