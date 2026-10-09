"""Every vendored browser asset ships with its notice, its release and its
citation (`molbuilder/web/static/vendor/README.md`, the inventory; user,
2026-10-09: "explicit citation and license for all embedded packages ...
should have the right files and information included in the package/git file
structure").

An artifact lint over the vendor directory as it ships (`testing.md` § 6,
*Artifact lints*): a third-party file added without an inventory row, a row
whose license file is not beside it, or a component with no citation is a
redistribution without its notice -- found here rather than by a reader of
the wheel.  The package-data rule ships the whole directory
(`pyproject.toml`, `web/static/**/*`), so what is checked here is what ships.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

VENDOR = (Path(__file__).resolve().parent.parent
          / "molbuilder" / "web" / "static" / "vendor")
README = VENDOR / "README.md"


def _section(text: str, title: str) -> str:
    start = text.index(f"## {title}")
    rest = text[start + 3:]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def _rows():
    """The inventory table's rows: ``(component, files cell, license cell)``."""
    rows = []
    for line in _section(README.read_text(), "Inventory").splitlines():
        if not line.startswith("| ") or line.startswith("| Component") \
                or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[-1]))
    return rows


def _shipped():
    return sorted(str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*")
                  if p.is_file() and p.name != "README.md"
                  and "__pycache__" not in p.parts)


def test_every_vendored_file_is_an_inventory_rows_file_or_its_notice():
    covered = []
    for _name, files, notice in _rows():
        covered += [g for g in re.findall(r"`([^`]+)`", files)
                    if not g.startswith("/")]
        covered += re.findall(r"`([^`]+)`", notice)
        covered += re.findall(r"\]\(([^)]+)\)", notice)
    orphans = [f for f in _shipped()
               if not any(fnmatch.fnmatch(f, g) for g in covered)]
    assert not orphans, (
        f"vendored with no inventory row naming them: {orphans} -- add the "
        f"component's row, its license file and its citation to "
        f"static/vendor/README.md")


def test_every_rows_license_file_is_beside_its_files():
    missing = [(name, link) for name, _files, notice in _rows()
               for link in re.findall(r"\]\(([^)]+)\)", notice)
               if not (VENDOR / link).is_file()]
    assert not missing, missing


def test_every_component_is_cited():
    citations = _section(README.read_text(), "Citations")
    cited = set(re.findall(r"^- \*\*([^*]+)\*\*", citations, re.M))
    uncited = [name for name, _f, _n in _rows() if name not in cited]
    assert not uncited, f"no citation line under ## Citations for {uncited}"
