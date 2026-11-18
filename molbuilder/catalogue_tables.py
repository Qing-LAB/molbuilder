"""THE RUN-FILE CATALOGUE AS TABLE CELLS -- one rendering of each row of
`runfiles.WRITTEN` and `runfiles.ENGINE_FILES`, for every page that shows it.

MODULE  catalogue_tables (L2; the catalogue (`runfiles`), the engines' words
        (`engine_table`) and their restart-file lists (`warmfiles`))
ROLE    :func:`written_cells` / :func:`engine_cells` -- a row's cells as a
        markdown table shows them, under :data:`WRITTEN_HEAD` /
        :data:`ENGINE_HEAD`: its file (:func:`file_cell`), what it is, who
        writes it, its door (:func:`door_cell`); :func:`markdown_table` --
        the one writer of these pages' tables; :func:`door_resolves` --
        whether a row's door names something the code has
USED-BY `tools/manifest.py`, which writes `execution/project-layout.md`
        § 5's tables from them and refuses a door the code lacks; the
        Documents tab's file index (`web/doc_indexes.py`)
"""
from __future__ import annotations

import importlib
from typing import Iterable, List, Sequence

from .runfiles import Artifact, EngineFile, compose, names_made, shown_file


def _engine_words(engine: str) -> str:
    """The engine's name as a person reads it -- its record's words."""
    from .engine_table import engine as _engine
    return _engine(engine).words


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


def door_cell(row: "Artifact | EngineFile") -> str:
    """The row's door -- the one function its readers ask -- or *none*."""
    return f"`{row.door}`" if row.door else "none"


def _notes(a: Artifact) -> str:
    """Which engine, which calculation, and when only sometimes."""
    bits = []
    if a.engine:
        bits.append(_engine_words(a.engine))
    if a.calculation:
        bits.append(", ".join(a.calculations))
    out = f" *({', '.join(bits)})*" if bits else ""
    if a.only:
        out += f" — only: {a.only}"
    return out


def file_cell(row: "Artifact | EngineFile") -> str:
    """The row's file (`runfiles.shown_file`), with which engine and which
    calculation -- a catalogue file -- or the engine that writes it -- an
    engine's file."""
    if isinstance(row, EngineFile):
        return f"`{shown_file(row)}` *({_engine_words(row.engine)})*"
    return f"`{shown_file(row)}`{_notes(row)}"


def carried(e: EngineFile) -> str:
    """What the engine's shipped restart-file list says of it
    (`warmfiles`): carried to a continuing run, known to the list, the
    engine's scratch -- each name the row makes for a run of ours asked
    (`runfiles.names_made`) -- or nothing."""
    from .warmfiles import is_scratch, warm_list
    if not e.role:
        return ""
    if any(is_scratch(compose("L", m)) for m in names_made(e.role)):
        return "scratch"
    listed = warm_list(e.engine)
    if e.role in listed.carry:
        return "carried"
    if e.role in listed.suffixes:
        return "listed"
    return ""


#: THE COLUMNS of a catalogue file's row (:func:`written_cells`) and of an
#: engine file's (:func:`engine_cells`).
WRITTEN_HEAD = ("file", "what it is for", "written by", "the door", "kind")
ENGINE_HEAD = ("file", "what it is", "the restart list", "the door")


def markdown_table(head: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """``rows`` under ``head`` as one markdown table -- the one writer of
    the catalogue's tables and the Documents tab's indexes."""
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def written_cells(a: Artifact) -> List[str]:
    """A catalogue file's cells: file, what it is for, who writes it, its
    door, its kind."""
    return [file_cell(a), a.what, a.writer, door_cell(a), a.kind]


def engine_cells(e: EngineFile) -> List[str]:
    """An engine file's cells: file, what it is, what the restart list says
    of it, its door."""
    return [file_cell(e), e.what, carried(e) or "—", door_cell(e)]


__all__ = ["ENGINE_HEAD", "WRITTEN_HEAD", "carried", "door_cell",
           "door_resolves", "engine_cells", "file_cell", "markdown_table",
           "written_cells"]
