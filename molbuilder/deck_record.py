"""The molbuilder record a generated deck carries -- its fenced blocks, and the one reader of a block's JSON.

MODULE  deck_record (L1, the job as described; the standard library only)
ROLE    the grammar of every molbuilder block in a generated file -- the block
        names, the BEGIN/END fences (:func:`begin_marker`,
        :func:`end_marker`, :data:`MARKER_RE`) -- and the ONE reader of a
        block's JSON payload (:func:`read_json_block`), with the readers of
        the two blocks a job reads beside itself: ENGINE-OFFSET and VIBRATION
USED-BY script_emit (writes every block and reads the rest through these),
        the SIESTA vibration's finish (`spectra.siesta_vibration`, beside
        the job), runs.declared, transport/compose, runwrap and
        pyscf/input (the effective-parameters fence)
TRAVELS in ``mb_vibration.pyz`` beside a SIESTA force-constant job
        (`runwrap.VIBRATION_COMPANIONS`)

`execution/job-contracts.md` § 3.1 owns the grammar.  Its own module because
a job reads blocks where the template does not exist: the grammar and its
reader are the part a job needs, and they need nothing but the standard
library.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# Block names used in the markers.  Centralised so a typo doesn't
# silently produce a file the parser refuses.
BLOCK_HEADER        = "header"
BLOCK_PROVENANCE    = "provenance"
BLOCK_BENCH_MARKS   = "bench-marks"
BLOCK_ATOM_METADATA = "atom-metadata"
#: Where a deck's atoms were placed (`model/structure-periodicity.md` § 6.0).
BLOCK_ENGINE_OFFSET = "engine-offset"
#: What a SIESTA force-constant deck's finish reads that no SIESTA keyword
#: states (`engines/vibration.md` § 5.3): the stationarity criterion, the
#: person's statement, the ladder's relaxation record.
BLOCK_VIBRATION     = "vibration"
BLOCK_USER_CUSTOM   = "user-custom"
#: The parameters the ENGINE actually holds, recorded into the run log at
#: startup.  Shared between engines on purpose: the deck says what was asked
#: for, this says what was heard, and one reader should be able to compare
#: them without knowing which engine wrote it.
BLOCK_PARAMETERS    = "effective-parameters"


def begin_marker(name: str) -> str:
    """Return the literal BEGIN marker line for a reserved block."""
    return f"# === molbuilder {name} BEGIN ==="


def end_marker(name: str) -> str:
    """Return the literal END marker line for a reserved block."""
    return f"# === molbuilder {name} END ==="


# Regex matching either marker for any block.  Group 1: block name;
# group 2: BEGIN | END.
#
# The name is one lowercase word (``header``, ``bench-marks``) OR two words
# (``item mesh_cutoff``) -- the second form is job-contracts.md § 3.7's item
# block, whose marker carries the FIELD's name.  That is what lets prep walk
# a template and rebuild a config without an .fdf parser, so the name has to
# reach the marker; underscores are allowed there because field names have
# them.  Every consumer already filters on ``group(1) != BLOCK_<x>``, so
# widening the name pattern cannot make an item block look like a reserved
# one (checked across all six consumers, 2026-08-07).
MARKER_RE = re.compile(
    r"^#\s*===\s+molbuilder\s+([a-z-]+(?:\s+[A-Za-z0-9_]+)?)"
    r"\s+(BEGIN|END)\s+===\s*$"
)


#: THE STAMPS a file molbuilder writes carries -- WHEN it was written and BY
#: WHICH BUILD -- which say nothing of what it computes: every ISO date-time
#: (a deck's and a wrapper's ``generated-at``, the atom-metadata fence's
#: ``created_at``, the pipeline log's banner), the ``generator-version``
#: line's value (``git <sha>``, ``unknown`` when git did not answer in time)
#: and the progress seed's ``wall_time``.
_STAMPS = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    rb"|^(?:#[ \t]*generator-version[ \t]).*$"
    rb"|wall_time: [0-9.]+", re.M)


def without_stamps(data: bytes) -> bytes:
    """``data`` with its :data:`_STAMPS` masked -- what two files are
    compared on when the question is what they compute, not when they were
    written: whether two decks are one calculation
    (`script_emit.same_calculation`), and whether two plans write one thing
    (`jobset.planned.Plan.identity`).  Fields, never whole fences: the
    atom-metadata fence that holds ``created_at`` holds the region partition
    too."""
    return _STAMPS.sub(b"<stamp>", data)


def _brace_delta(line: str) -> int:
    """``{`` minus ``}`` on one line, counting only braces OUTSIDE strings.

    Counting every brace would let a brace inside a JSON *string* close the
    walk early: a region named ``a}b`` -- valid JSON on the wire, written
    correctly by :func:`emit_atom_metadata` -- would make the whole
    ATOM-METADATA block unreadable, and the labels AND the frozen set vanish
    with no message.

    JSON strings cannot contain a literal newline, so the in-string state
    never has to carry across lines.
    """
    depth = 0
    in_str = False
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depth


def read_json_block(text: str, name: str) -> Optional[Dict[str, Any]]:
    """Find the molbuilder block ``name`` in ``text`` and return its JSON
    payload as a dict -- the ONE reader for every JSON-payload block
    (ATOM-METADATA, ENGINE-OFFSET, VIBRATION), so their parsing cannot
    drift.

    Returns ``None`` when:
      * No such block is present.
      * The block markers are unbalanced.
      * The JSON between markers fails to parse.

    Comment-prefix-per-line is stripped before JSON parsing.
    """
    lines = text.splitlines()
    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != name:
            continue
        if m.group(2) == "BEGIN":
            begin_idx = i
            end_idx = None
        elif m.group(2) == "END" and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    # Inner lines: strip leading "# " (or "#") to recover JSON.
    inner: List[str] = []
    for raw in lines[begin_idx + 1: end_idx]:
        if raw.startswith("# "):
            inner.append(raw[2:])
        elif raw.startswith("#"):
            inner.append(raw[1:])
        else:
            inner.append(raw)
    # Brace-balance walk so the extractor accepts BOTH pretty-printed
    # JSON (molbuilder's emit_atom_metadata via json.dumps indent=2)
    # AND compact / single-line JSON.  The contract on the wire is
    # "valid JSON inside the block"; how the writer formatted it isn't
    # load-bearing.
    json_lines: List[str] = []
    saw_open = False
    brace_depth = 0
    for line in inner:
        stripped = line.strip()
        if not saw_open:
            if not stripped or not stripped.startswith("{"):
                continue
            saw_open = True
        json_lines.append(line)
        brace_depth += _brace_delta(stripped)
        if brace_depth <= 0:
            break
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


def extract_engine_offset(text: str) -> Optional[Dict[str, Any]]:
    """The ENGINE-OFFSET block's payload -- ``{applied_offset, cell,
    axis_kind, stated}`` -- or ``None`` for a deck that carries none.  The deck's
    own coordinates carry no offset; ``applied_offset`` is the correction that
    was added to the design's to produce them."""
    return read_json_block(text, BLOCK_ENGINE_OFFSET)


def extract_vibration_record(text: str) -> Optional[Dict[str, Any]]:
    """The VIBRATION block's payload -- ``{stage, force_criterion_ev_ang,
    temperature_K, already_relaxed, relaxation, relaxation_stage,
    masses_amu, molbuilder_version}`` (format ``molbuilder-vibration/v3``), built by
    `spectra.siesta_vibration.vibration_record` and written by
    `script_emit.emit_vibration_record` -- or ``None`` for a deck that
    carries none: every deck but a SIESTA force-constant one."""
    return read_json_block(text, BLOCK_VIBRATION)


__all__ = ["BLOCK_HEADER", "BLOCK_PROVENANCE", "BLOCK_BENCH_MARKS",
           "BLOCK_ATOM_METADATA", "BLOCK_ENGINE_OFFSET", "BLOCK_VIBRATION",
           "BLOCK_USER_CUSTOM", "BLOCK_PARAMETERS", "MARKER_RE",
           "begin_marker", "end_marker", "read_json_block",
           "extract_engine_offset", "extract_vibration_record",
           "without_stamps"]
