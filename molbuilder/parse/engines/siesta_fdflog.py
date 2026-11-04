"""SIESTA's own ``fdf.<timestamp>.log`` -- every key the engine read, with the
value it read.  Its name is the catalogue's (`runfiles.SIESTA_FDF_LOG`),
read back through the catalogue's doors: SIESTA 5 names the log after the
moment it opened it (``fdf.20260925T194936.695.log``, ``Src/reinit_m.F90``).

`model/parse.md` § 5d.3: the run record's third column, *what the engine
used*, is read here and never echoed from the deck.

The fdf library writes one line per key SIESTA looks up: ``<key> <value>
[<unit>] [# default value]`` -- the marker saying the deck does not carry the
label -- followed by ``# above item originally: <key> <value> [<unit>]`` when
it converted the spelling it was given.  ``#:block?`` / ``#:defined?`` lines
are its own lookups and carry no value.  A block's rows are the lines it writes
with one leading space (libfdf's ``fdf_bline``), between ``%block`` and
``%endblock`` at column 0; a line at column 0 inside a block is a lookup of its
own (``LatticeConstant`` inside ``%block LatticeVectors``, which SIESTA looks up
as it reads the block), and a block echoed with no ``%endblock`` holds the rows
it wrote.

**A key may be read more than once, and not always to the same value.**  Each
call site passes its own default, so a key the deck does not set can read
differently at each: TranSIESTA reads ``TS.Contours.Eq.Pole`` per chemical
potential as 1.5 eV, then -- with no ``contour.eq`` -- as the continued
fraction's own π·60·kT·0.7, then as 1.5 eV again to print it
(``Src/m_ts_chem_pot.F90``), and the 42-pole run of 2026-09-25 used the second.
So a key read to several values carries them all, in order, and no single
value: which one a call site used is that call site's, and a reader that
picked one would be choosing for it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from molbuilder.parse.base import FileParser
from molbuilder.parse.fdf import fdf_key
from molbuilder.parse.types import EngineParamsResult, ParseResult, ParseWarning
from molbuilder.runfiles import SIESTA_FDF_LOG, find_fixed, fixed_fields


def _opened_at(stamp: str) -> Optional[str]:
    """A log's ``stamp`` field (``20260925T194936.695``) as the moment
    SIESTA opened it, to the second: ``2026-09-25T19:49:36``."""
    from datetime import datetime
    try:
        return datetime.strptime(stamp.split(".")[0],
                                 "%Y%m%dT%H%M%S").isoformat()
    except ValueError:
        return None
_ORIGINAL = re.compile(r"^#\s*above item originally:\s*\S+\s+(.*?)\s*$",
                       re.IGNORECASE)


def used_log_of(d: Path, out_head: str) -> Optional[Path]:
    """SIESTA's ``fdf.<stamp>.log`` for the ``.out`` whose head this is --
    what the output's catalogue row names for ``used_log``
    (`runfiles.WRITTEN`).

    THE FDF LOG IS PAIRED BY ITS STAMP: SIESTA opens it in the second it
    prints ``>> Start of run`` -- exact on 122 of 122 real outputs, and
    SIESTA's own naming, so it holds in a folder molbuilder did not write.
    """
    from . import siesta_grammar as _G
    facts: Dict[str, Any] = {}
    for line in out_head.splitlines():
        if _G.read_launch_line(line, facts) == "run_start_local":
            break
    start = facts.get("run_start_local")
    if not start:
        return None
    return next((p for p, f in find_fixed(d, SIESTA_FDF_LOG)
                 if _opened_at(f["stamp"]) == start), None)


def read_fdf_log(text: str, source: str = "<text>"
                 ) -> Tuple[Dict[str, Dict[str, Any]],
                            Dict[str, List[str]],
                            List[ParseWarning]]:
    """``(params, blocks, warnings)`` from an fdf log's text.

    ``params`` and ``blocks`` are keyed by fdf's own label rule
    (`parse/fdf.fdf_key`: case, ``.``, ``-`` and ``_`` ignored), so a deck's
    ``kgrid_Monkhorst_Pack`` finds the log's ``kgrid.MonkhorstPack``.  Each
    ``params`` entry has ``key`` as the log spells it and, when every reading
    agrees, the reading itself: ``value``, ``number`` and ``unit`` when it is
    a quantity, ``default`` when the deck does not carry the label, and
    ``original`` when SIESTA converted the spelling it was given.  A key read
    to several values has ``readings`` instead -- each distinct one, in order.
    """
    spelled: Dict[str, str] = {}
    read: Dict[str, List[Dict[str, Any]]] = {}
    blocks: Dict[str, List[str]] = {}
    warnings: List[ParseWarning] = []
    last = None
    block = None
    block_line = 0
    in_rows = False
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        # ONE LEADING SPACE IS A BLOCK'S ROW, or libfdf's own note on reading
        # one (``#:(Backspace to)``); every other line starts at column 0.
        if line.startswith(" "):
            if block is not None and not line.lstrip().startswith("#:"):
                blocks[block].append(line)
                in_rows = True
            continue
        in_rows = False
        low = line.lower()
        if low.startswith("%endblock"):
            block = None
            continue
        if low.startswith("%block"):
            parts = line.split(None, 1)
            block = fdf_key(parts[1].strip()) if len(parts) > 1 else ""
            block_line = line_no
            blocks[block] = []
            continue
        if line.startswith("#"):
            m = _ORIGINAL.match(line.strip())
            if m and last is not None:
                last["original"] = m.group(1)
            continue
        body, _, comment = line.partition("#")
        toks = body.split()
        if not toks:
            continue
        key, rest = toks[0], toks[1:]
        reading: Dict[str, Any] = {
            "value": " ".join(rest),
            "default": "default value" in comment.lower()}
        if rest:
            try:
                reading["number"] = float(rest[0])
            except ValueError:
                pass
            if len(rest) == 2 and "number" in reading:
                reading["unit"] = rest[1]
        label = fdf_key(key)
        spelled.setdefault(label, key)
        read.setdefault(label, []).append(reading)
        last = reading
    # A LOG THAT STOPS AMONG A BLOCK'S ROWS is cut short; one that went on to
    # other lookups read as many rows as it wanted (``Optical.Vector``).
    if block is not None and in_rows:
        warnings.append(ParseWarning(source=source, line_no=block_line,
                                     snippet=block,
                                     error=f"block {block!r} never closed",
                                     category="fdf-log"))
    params: Dict[str, Dict[str, Any]] = {}
    for label, readings in read.items():
        distinct: List[Dict[str, Any]] = []
        # ONE READING PER VALUE, case-blind: fdf matches an option's value
        # the way it matches its label, so `CG` and `cg` are one setting read
        # twice, not two.  The first spelling is kept.
        for r in readings:
            if all(str(r["value"]).casefold() != str(d["value"]).casefold()
                   for d in distinct):
                distinct.append(r)
        params[label] = ({"key": spelled[label], **distinct[0]}
                         if len(distinct) == 1 else
                         {"key": spelled[label], "readings": distinct})
    return params, blocks, warnings


class SiestaFdfLogParser(FileParser):
    """SIESTA's fdf log: the engine's account of every setting it read."""

    name   = "siesta-fdf-log"
    label  = "SIESTA's own fdf log (fdf.*.log)"
    hint   = "the fdf.<timestamp>.log SIESTA writes beside its .out"
    output = EngineParamsResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return fixed_fields(SIESTA_FDF_LOG, Path(path).name) is not None

    @classmethod
    def parse(cls, path: Path) -> EngineParamsResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        params, blocks, warnings = read_fdf_log(text, source=str(path))
        return EngineParamsResult(
            **ParseResult.envelope(cls.name, str(path)),
            params=params, blocks=blocks, parse_warnings=warnings)
