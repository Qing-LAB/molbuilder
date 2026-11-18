"""Render the contract's manifest from the catalogue.

`docs/execution/project-layout.md` § 5 lists every file molbuilder writes into
a calculation folder -- what it is for, who writes it, the one function its
readers ask, what losing it costs -- and the engines' files molbuilder names
(§ 5.4).  Those rows are the catalogue's, `runfiles.WRITTEN` and
`runfiles.ENGINE_FILES`: the ONE source (user, 2026-10-04: "Generate from
catalogue"), the Task setup card and the Results tab's file card read it too,
and this tool writes its tables into the document, between the markers it
owns:

    <!-- manifest:<level> -->  ...  <!-- /manifest -->

Usage, from the repository's root:
    python -m tools.manifest           # rewrite the tables in place
    python -m tools.manifest --check   # exit 1 when the document differs

A DOOR THAT DOES NOT EXIST IS REFUSED, not rendered: a contract that names a
function the code does not have is how `project-layout.md` § 4.5 came to send
readers to `paths.path_for` (found 2026-10-04).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from molbuilder.catalogue_tables import (ENGINE_HEAD, WRITTEN_HEAD,
                                        door_resolves, engine_cells,
                                        markdown_table, written_cells)
from molbuilder.runfiles import ENGINE_FILES, WRITTEN

DOC = Path(__file__).resolve().parent.parent / "docs/execution/project-layout.md"

#: The tables, in the order § 5 holds them: one per level, and the engines'
#: files molbuilder names (§ 5.4, `runfiles.ENGINE_FILES`).
LEVELS = ("calculation", "stage", "run", "bench", "launch", "transient")
ENGINES = "engines"

_MARK = re.compile(r"(<!-- manifest:([a-z]+) -->\n)(.*?)(<!-- /manifest -->)",
                   re.S)



def _checked(row, table: str):
    """``row``, refused when its door names a function the code lacks: a
    contract must not send a reader to one
    (`catalogue_tables.door_resolves`)."""
    if row.door and not door_resolves(row.door):
        raise SystemExit(
            f"runfiles.{table}: {row.role or row.name!r} names the door "
            f"{row.door!r}, which does not exist -- a contract must not "
            f"send a reader to a function the code lacks.")
    return row


def engines_table() -> str:
    """§ 5.4's rows: the engines' files molbuilder names."""
    return markdown_table(ENGINE_HEAD, (
        engine_cells(_checked(e, "ENGINE_FILES")) for e in ENGINE_FILES))


def table(level: str) -> str:
    """One level's rows, as the markdown table § 5 holds."""
    return markdown_table(WRITTEN_HEAD, (
        written_cells(_checked(a, "WRITTEN"))
        for a in WRITTEN if a.level == level))


def render_into(doc: str) -> str:
    """``doc`` with every marked table rendered from the catalogue."""
    def one(m):
        level = m.group(2)
        if level == ENGINES:
            return m.group(1) + engines_table() + m.group(4)
        if level not in LEVELS:
            raise SystemExit(f"manifest marker for an unknown level: {level}")
        return m.group(1) + table(level) + m.group(4)
    return _MARK.sub(one, doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the document differs from the catalogue")
    args = ap.parse_args(argv)
    text = DOC.read_text(encoding="utf-8")
    rendered = render_into(text)
    if args.check:
        if rendered != text:
            print(f"{DOC}: § 5 differs from the catalogue "
                  f"(runfiles.WRITTEN, ENGINE_FILES) -- run "
                  f"`python -m tools.manifest`", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
