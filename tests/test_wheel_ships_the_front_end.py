"""**A built wheel must carry every file the served pages ask for.**

The failure this guards is silent in the only place anyone looks: the repo
works, because Flask serves `molbuilder/web/static/` off disk.  An INSTALLED
copy serves what the wheel packaged, and on 2026-09-03 a wheel built from this
tree was measured carrying **90 of the 141** static files — the whole MolView
module (13 files), the structure page (9), VibrationView (5), and the
Task-setup, Documents, Transport and This-machine tabs' own assets.  Every
directory added after the hand-kept glob list was last edited was missing, and
nothing said so.

It had happened before, on the other side of the same list: no `.toml` pattern
existed until 2026-08-14, so neither the parameter catalogue nor the warm-file
rules shipped — *"it worked from the repo and would have failed on an installed
copy"* (`pyproject.toml`'s own note).

So the rule is not "keep the list current".  It is:

> **Every file under `web/static/` is matched by a package-data pattern, and
> every static asset a template names exists on disk.**

The first half is what a recursive glob buys and this test keeps honest.  The
second is its mirror: a pattern that ships everything is no help if a template
points at a file nobody wrote.

**Matched the way setuptools matches, not with `fnmatch` alone.**  `fnmatch`'s
`*` crosses `/`, so `web/static/vendor/*/*` "covers" `lib/molview/model.js`
under it and the first measurement of this defect came back **zero missing**.
Per-segment matching is what setuptools' `glob` actually does, and it is what
makes the count here real.
"""
from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "molbuilder"
STATIC = PKG / "web" / "static"
TEMPLATES = PKG / "web" / "templates"


def _package_data_patterns() -> list[str]:
    with open(REPO / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    return cfg["tool"]["setuptools"]["package-data"]["molbuilder"]


def _matches(rel: str, pattern: str) -> bool:
    """One path against one pattern, the way a filesystem glob works.

    ``**`` spans directories; every other segment matches within one.
    """
    parts = rel.split("/")
    pats = pattern.split("/")
    if "**" in pats:
        head = pats[: pats.index("**")]
        tail = pats[pats.index("**") + 1:]
        if len(parts) < len(head) + len(tail):
            return False
        if not all(fnmatch.fnmatch(a, b) for a, b in zip(parts, head)):
            return False
        if tail and not all(fnmatch.fnmatch(a, b)
                            for a, b in zip(parts[len(parts) - len(tail):], tail)):
            return False
        return True
    if len(parts) != len(pats):
        return False
    return all(fnmatch.fnmatch(a, b) for a, b in zip(parts, pats))


def test_every_static_file_is_packaged():
    """No file the server can hand out is left out of the wheel."""
    pats = _package_data_patterns()
    missing = sorted(
        str(f.relative_to(PKG))
        for f in STATIC.rglob("*")
        if f.is_file() and not any(_matches(str(f.relative_to(PKG)), p)
                                   for p in pats)
    )
    assert not missing, (
        f"{len(missing)} static file(s) match no package-data pattern, so a "
        "built wheel does not carry them and an installed copy serves a "
        "broken page:\n  " + "\n  ".join(missing[:40])
        + "\n\nDo not add another per-directory glob -- that list is what "
          "failed.  The tree is covered recursively; a file outside it needs "
          "a pattern that cannot go stale."
    )


def test_the_recursive_pattern_is_actually_recursive():
    """The mutation guard for the matcher above.

    A matcher that answered ``True`` for everything would make the first test
    vacuously green -- which is exactly what `fnmatch` alone did.  So: a file
    two directories deep must NOT be matched by a one-level pattern.
    """
    assert not _matches("web/static/lib/molview/model.js", "web/static/lib/*.js")
    assert not _matches("web/static/lib/molview/model.js", "web/static/vendor/*/*")
    assert _matches("web/static/lib/molview/model.js", "web/static/**/*")
    assert _matches("web/static/style.css", "web/static/**/*")


#: ``url_for('static', filename='…')`` — the only way a template names an asset.
_URL_FOR = re.compile(
    r"url_for\(\s*['\"]static['\"]\s*,\s*filename\s*=\s*['\"]([^'\"]+)['\"]")


def test_every_asset_a_template_names_exists():
    """The mirror: shipping everything does not help if a page asks for a
    file nobody wrote.  A missing one is a 404 in the browser console and a
    feature that silently does nothing."""
    dangling = []
    for tpl in sorted(TEMPLATES.rglob("*.html")):
        text = tpl.read_text(encoding="utf-8")
        for m in _URL_FOR.finditer(text):
            rel = m.group(1)
            if not (STATIC / rel).is_file():
                line = text.count("\n", 0, m.start()) + 1
                dangling.append(f"{tpl.relative_to(REPO)}:{line}: {rel}")
    assert not dangling, (
        "templates name static files that do not exist:\n  "
        + "\n  ".join(sorted(dangling))
    )


def test_the_template_scan_sees_the_templates():
    """A path filter that matched nothing would make the test above pass by
    checking nothing -- the shape this suite exists to catch elsewhere."""
    tpls = list(TEMPLATES.rglob("*.html"))
    assert len(tpls) >= 8, f"only {len(tpls)} templates found under {TEMPLATES}"
    named = sum(len(_URL_FOR.findall(t.read_text(encoding="utf-8"))) for t in tpls)
    assert named >= 40, f"only {named} url_for('static') references parsed"
