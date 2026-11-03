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
import importlib
import re
import sys
from pathlib import Path

from molbuilder.runfiles import (ENGINE_FILES, WRITTEN, Artifact, EngineFile,
                                 names_made, shown)

DOC = Path(__file__).resolve().parent.parent / "docs/execution/project-layout.md"

#: The tables, in the order § 5 holds them: one per level, and the engines'
#: files molbuilder names (§ 5.4, `runfiles.ENGINE_FILES`).
LEVELS = ("calculation", "stage", "run", "bench", "launch", "transient")
ENGINES = "engines"

_MARK = re.compile(r"(<!-- manifest:([a-z]+) -->\n)(.*?)(<!-- /manifest -->)",
                   re.S)

_ENGINE = {"siesta": "SIESTA", "pyscf": "PySCF"}


def _name(a: "Artifact | EngineFile") -> str:
    """The file's name, one spelling -- a row of either table, its fields
    shown by name (`runfiles.shown`)."""
    if a.name:
        return f"`{shown(a.name)}`"
    head = "<base>" if a.staged else "<label>"
    return f"`{head}{shown(a.role)}`"


def _notes(a: Artifact) -> str:
    """Which engine, which calculation, and when only sometimes."""
    bits = []
    if a.engine:
        bits.append(_ENGINE.get(a.engine, a.engine))
    if a.calculation:
        bits.append(", ".join(a.calculations))
    out = f" *({', '.join(bits)})*" if bits else ""
    if a.only:
        out += f" — only: {a.only}"
    return out


def door_resolves(door: str) -> bool:
    """Whether ``door``, a dotted path under ``molbuilder``, names something
    that exists -- a module's function, class, or a class's method."""
    parts = door.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module("molbuilder." + ".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return False
        return True
    return False


def _carried(e: EngineFile) -> str:
    """What the engine's shipped restart-file list says of it
    (`warmfiles`): carried to a continuing run, known to the list, the
    engine's scratch -- each name the row makes for a run of ours asked
    (`runfiles.names_made`) -- or nothing."""
    from molbuilder.warmfiles import is_scratch, warm_list
    if not e.role:
        return ""
    if any(is_scratch("L" + m) for m in names_made(e.role)):
        return "scratch"
    listed = warm_list(e.engine)
    if e.role in listed.carry:
        return "carried"
    if e.role in listed.suffixes:
        return "listed"
    return ""


def engines_table() -> str:
    """§ 5.4's rows: the engines' files molbuilder names."""
    lines = ["| file | what it is | the restart list | the door |",
             "|---|---|---|---|"]
    for e in ENGINE_FILES:
        if e.door and not door_resolves(e.door):
            raise SystemExit(
                f"runfiles.ENGINE_FILES: {e.role or e.name!r} names the "
                f"door {e.door!r}, which does not exist -- a contract must "
                f"not send a reader to a function the code lacks.")
        door = f"`{e.door}`" if e.door else "none"
        lines.append(f"| {_name(e)} *({_ENGINE.get(e.engine, e.engine)})* "
                     f"| {e.what} | {_carried(e) or '—'} | {door} |")
    return "\n".join(lines) + "\n"


def table(level: str) -> str:
    """One level's rows, as the markdown table § 5 holds."""
    lines = ["| file | what it is for | written by | the door | kind |",
             "|---|---|---|---|---|"]
    for a in WRITTEN:
        if a.level != level:
            continue
        if a.door and not door_resolves(a.door):
            raise SystemExit(
                f"runfiles.WRITTEN: {a.role or a.name!r} names the door "
                f"{a.door!r}, which does not exist -- a contract must not "
                f"send a reader to a function the code lacks.")
        door = f"`{a.door}`" if a.door else "none"
        lines.append(f"| {_name(a)}{_notes(a)} | {a.what} | {a.writer} "
                     f"| {door} | {a.kind} |")
    return "\n".join(lines) + "\n"


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
