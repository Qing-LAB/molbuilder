"""The molbuilder record a generated deck carries -- its fenced blocks, and the one reader of a block's JSON.

MODULE  deck_record (L1, the job as described; beside a job the standard
        library and the progress log's words at load, and the order record's
        class where a deck's is read)
ROLE    the grammar of every molbuilder block in a generated file -- the block
        names, the BEGIN/END fences (:func:`begin_marker`,
        :func:`end_marker`, :data:`MARKER_RE`) -- and the ONE reader of a
        block's JSON payload (:func:`read_json_block`), with the readers of
        each block -- ATOM-METADATA and its order record
        (:func:`order_record_of`), ENGINE-OFFSET, VIBRATION, SHARED, PROVENANCE
        and who wrote the file (:func:`written_by`), BENCH-MARKS, the
        effective-parameters rows
USED-BY script_emit (writes every block through these),
        the SIESTA vibration's finish (`spectra.siesta_vibration`, beside
        the job), runs.declared, parse (a folder's engine, a run's record),
        runwrap and pyscf/input (the effective-parameters fence)
TRAVELS in ``mb_vibration.pyz`` beside each call of a SIESTA vibration
        (`runwrap.VIBRATION_COMPANIONS`)

`execution/job-contracts.md` § 3.1 owns the grammar.  Its own module because
a job reads blocks where the template does not exist: the grammar and its
reader are the part a job needs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:                                        # inside molbuilder
    from .trajectory_log.keys import WALL_TIME as _WALL_TIME
except ImportError:                         # beside a job, in a bundle
    from keys import WALL_TIME as _WALL_TIME


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
#: What a transmission point that shares energies with a finished
#: calculation's point computes, and with which point
#: (`transport.energies.Contours`; `engines/transport.md` § 2a.16).
BLOCK_SHARED        = "shared"
#: A PySCF script's run settings, live lines at its head, which every other
#: line reads by name (`execution/job-contracts.md` § 3.1): what the
#: same-calculation door sets aside for a PySCF script.
BLOCK_RUN_SETTINGS  = "run-settings"
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


#: Either fence of any block.  Group 1: the block's name; group 2: BEGIN | END.
MARKER_RE = re.compile(
    r"^#\s*===\s+molbuilder\s+([a-z-]+)\s+(BEGIN|END)\s+===\s*$")


#: The name a PySCF vibration script gives the release that wrote it, for its
#: result (`engines/vibration.md` § 6.2).
BUILD_NAME = "MOLBUILDER_VERSION"

#: THE STAMPS a file molbuilder writes carries -- WHEN it was written and BY
#: WHICH BUILD -- which say nothing of what it computes: every ISO date-time
#: (a deck's and a wrapper's ``generated-at``, the atom-metadata fence's
#: ``created_at``, the pipeline log's banner), the ``generator-version``
#: line's value (``git <sha>``, ``unknown`` when git did not answer in time),
#: the line of a PySCF vibration script naming the release that wrote it
#: (:data:`BUILD_NAME`), and the progress seed's ``wall_time``.
_STAMPS = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    rb"|^(?:#[ \t]*generator-version[ \t]).*$"
    rb"|^" + BUILD_NAME.encode() + rb" = .*$"
    rb"|" + re.escape(_WALL_TIME.encode()) + rb" [0-9.]+", re.M)


def without_stamps(data: bytes) -> bytes:
    """``data`` with its :data:`_STAMPS` masked -- what two files are
    compared on when the question is what they compute, not when they were
    written: whether two decks are one calculation (:func:`same_calculation`,
    a PySCF script's reading), and whether two plans write one thing
    (`jobset.planned.Plan.identity`).  Fields, never whole fences: the
    atom-metadata fence that holds ``created_at`` holds the region partition
    too."""
    return _STAMPS.sub(b"<stamp>", data)


class DeckRecordUnreadable(ValueError):
    """A deck carries the molbuilder block it names, and the block cannot be
    read: its fences are not a BEGIN/END pair, it holds no JSON object, or
    its JSON does not parse.  The block is a record molbuilder writes and
    nobody edits (`execution/job-contracts.md` § 3.1), so this is refused,
    never read as a deck that records nothing."""


def _unreadable(name: str, why: str) -> DeckRecordUnreadable:
    return DeckRecordUnreadable(
        f"its molbuilder {name} block {why}.  The block is a record "
        f"molbuilder writes, not one edited by hand: restore the folder to "
        f"a saved state (`molbuilder checkpoint restore`), or prepare the "
        f"stage again")


def _first_pair(text: str, name: str) -> Tuple[Optional[List[str]], bool]:
    """The lines between the first complete BEGIN/END pair of the block
    ``name`` in ``text`` (``None`` when there is none), and whether either of
    its fences is there at all -- the scan every reader of a block by its
    first pair asks.  USER-CUSTOM takes the outermost pair instead, on purpose
    (`script_emit._user_custom_span`)."""
    lines = text.splitlines()
    begin: Optional[int] = None
    seen = False
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m or m.group(1) != name:
            continue
        seen = True
        if m.group(2) == "BEGIN":
            begin = i
        elif begin is not None:
            return lines[begin + 1: i], True
    return None, seen


def read_json_block(text: str, name: str) -> Optional[Dict[str, Any]]:
    """Find the molbuilder block ``name`` in ``text`` and return its JSON
    payload as a dict -- the ONE reader for every JSON-payload block
    (ATOM-METADATA, ENGINE-OFFSET, VIBRATION, SHARED), so their parsing cannot
    drift.

    Returns ``None`` when no such block is present: the deck records
    nothing of it.  A block that IS present and cannot be read -- its
    markers not a BEGIN/END pair, no JSON object between them, or JSON that
    fails to parse -- raises :class:`DeckRecordUnreadable`.

    The object starts at the first line opening with ``{``, past the block's
    ``# format:`` line, each line's comment prefix off; however its writer
    laid it out, the JSON decoder finds where it ends."""
    inner, seen = _first_pair(text, name)
    if not seen:
        return None
    if inner is None:
        raise _unreadable(name, "has no BEGIN/END pair of fences")
    lines = [re.sub(r"^# ?", "", raw) for raw in inner]
    opens = next((i for i, line in enumerate(lines)
                  if line.lstrip().startswith("{")), None)
    if opens is None:
        raise _unreadable(name, "holds no JSON object")
    try:
        return json.JSONDecoder().raw_decode(
            "\n".join(lines[opens:]).lstrip())[0]
    except json.JSONDecodeError as exc:
        raise _unreadable(name, f"holds JSON that does not parse ({exc})") \
            from exc


def extract_atom_metadata(text: str) -> Optional[Dict[str, Any]]:
    """The ATOM-METADATA block's payload -- the structure's metadata but its
    box, the order record among it (`job-contracts.md` § 3.4) -- or ``None``
    for a deck that carries none.  Read by the run door (``runs.declared``)
    and by the finish beside a job (:func:`order_record_of`)."""
    return read_json_block(text, BLOCK_ATOM_METADATA)


def extract_engine_offset(text: str) -> Optional[Dict[str, Any]]:
    """The ENGINE-OFFSET block's payload -- ``{applied_offset, cell,
    axis_kind, stated}`` -- or ``None`` for a deck that carries none.  The deck's
    own coordinates carry no offset; ``applied_offset`` is the correction that
    was added to the design's to produce them."""
    return read_json_block(text, BLOCK_ENGINE_OFFSET)


def extract_vibration_record(text: str) -> Optional[Dict[str, Any]]:
    """The VIBRATION block's payload -- ``{stage, force_criterion_ev_ang,
    temperature_K, already_relaxed, relaxation, relaxation_stage,
    masses_amu, raman, title, identity, molbuilder_version}`` (format
    ``molbuilder-vibration/v5``), built by
    `spectra.siesta_vibration.vibration_record` and written by
    `script_emit.emit_vibration_record` -- or ``None`` for a deck that
    carries none: every deck but a SIESTA force-constant one."""
    return read_json_block(text, BLOCK_VIBRATION)


def box_of_run(engine_offset: Optional[Dict[str, Any]],
               lattice=None) -> Optional[Dict[str, Any]]:
    """The box the engine had, for coordinates that came from it
    (`model/structure-periodicity.md` § 6.0): the ``cell`` -- ``lattice``,
    the output's own, else the one the deck placed the atoms in -- the
    ``axis_kind`` the deck recorded (:func:`extract_engine_offset`'s
    payload), and the engine's origin, an ``engine_offset`` of 0 whose
    owner is the engine, once a cell is known.  ``None`` when nothing says.
    Asked by the run door (``runs.RunView.frame``) and by the SIESTA
    vibration finish for the structure its result states -- one rule."""
    out: Dict[str, Any] = {}
    if isinstance(lattice, list) and len(lattice) == 3:
        out["cell"] = lattice
    rec = engine_offset or {}
    if rec.get("axis_kind"):
        out["axis_kind"] = [str(k) for k in rec["axis_kind"]]
    if "cell" not in out and rec.get("cell") is not None:
        out["cell"] = rec["cell"]
    if "cell" in out:
        out["engine_offset"] = [0.0, 0.0, 0.0]
        # ...THE ENGINE'S origin, recorded as such: a new calculation made
        # from these coordinates does not carry it (§ 6.0, *Entering a new
        # calculation*).
        out["offset_by"] = "engine"
    return out or None


def order_record_of(block: Optional[Dict[str, Any]]):
    """THE DECK'S ORDER RECORD (`model/overview.md` § 2.2a, I5): an
    ATOM-METADATA payload's ``atom_order``, read by the order record's one
    reader (``atom_permutation.Permutation.from_dict``) over the block's own
    atom count -- ``None`` for a deck in the user order, or one that carries no
    block.  Asked by the run door (``runs.declared``) and by the finish
    beside a job, which reads its own deck's record (`model/overview.md`
    § 2.2a, I5)."""
    record = (block or {}).get("atom_order")
    if record is None:
        return None
    try:                                    # inside molbuilder
        from .atom_permutation import Permutation
    except ImportError:                     # beside a job
        from atom_permutation import Permutation  # type: ignore
    return Permutation.from_dict(record, source="atom_order",
                                 n_atoms=(block or {}).get("n_atoms_total"))


def same_calculation(engine: str, a: str, b: str) -> List[str]:
    """THE ONE DOOR OF *THE SAME CALCULATION* (`execution/job-request.md`
    § 6): every difference between what ``engine`` computes from deck ``a``
    and from deck ``b``, by name -- none when they are the same, the run
    settings aside.  Asked by the transport gather, by prep of a launched
    stage, and by the run record (`current`) and the run door
    (`Run.report`); answered by the engine's own reading, the part its
    record names (``same_calculation``, floor 1), resolved here -- so this
    module, travelling with a job, imports no engine at load."""
    from .engine_table import engine as _engine
    from .kind_table import part
    rec = _engine(engine)
    reading = part(rec.same_calculation, record=f"engine {rec.name}",
                   key="same_calculation")
    return list(reading(a, b))


# --------------------------------------------------------------------- #
#  The effective-parameters block's rows (`model/parse.md` § 5d.3a)     #
# --------------------------------------------------------------------- #

#: What opens a row inside the block -- a comment in every engine's log.
PARAMETER_ROW_PREFIX = "#   "
_ROW_FIELDS = ("item", "default", "asked", "used")


def read_parameters_fence(text: str) -> List[Dict[str, Any]]:
    """Every row of every effective-parameters block in ``text``, in order:
    ``{item, default, asked, used}``.  Lines in the block that are not rows --
    SIESTA's wrapper echoes the deck there too -- are the block's prose and
    are skipped."""
    begin = begin_marker(BLOCK_PARAMETERS)
    end = end_marker(BLOCK_PARAMETERS)
    rows: List[Dict[str, Any]] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == begin:
            inside = True
            continue
        if stripped == end:
            inside = False
            continue
        if not inside or not line.startswith(PARAMETER_ROW_PREFIX + "["):
            continue
        try:
            arr = json.loads(line[len(PARAMETER_ROW_PREFIX):])
        except ValueError:
            continue
        if (isinstance(arr, list) and len(arr) == len(_ROW_FIELDS)
                and isinstance(arr[0], str)):
            rows.append(dict(zip(_ROW_FIELDS, arr)))
    return rows


# --------------------------------------------------------------------- #
#  The provenance block                                                 #
# --------------------------------------------------------------------- #

_PROVENANCE_KV_RE = re.compile(
    r"^#\s+(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")

#: THE PROVENANCE BLOCK'S SUB-BLOCKS -- a ``<name>:`` header, then indented
#: rows read as ``<name>.<key>``: what the auto-policy chose
#: (``resolved-defaults``) and who set each value (``parameters``).  The
#: header's own words, where it carries any, read as ``<name>``.
PARAMETERS_KEY = "parameters"
_PROVENANCE_DEFAULTS_HDR = re.compile(
    r"^#\s+(?P<sub>resolved-defaults|" + PARAMETERS_KEY
    + r")\s*:\s*(?P<note>.*?)\s*$")

_PROVENANCE_DEFAULTS_KV_RE = re.compile(
    r"^#\s{4,}(?P<key>[A-Za-z][A-Za-z0-9._-]*)\s{2,}(?P<val>.+?)\s*$")

def read_provenance_block(text: str) -> Optional[Dict[str, str]]:
    """Find the PROVENANCE block and return its k/v payload as a flat
    dict.  Returns ``None`` when no well-formed PROVENANCE block is
    present.  Empty-but-present block returns ``{}`` (distinct from
    None — `model/parse.md`'s absent-vs-empty rule)."""
    inner = _first_pair(text, BLOCK_PROVENANCE)[0]
    if inner is None:
        return None
    out: Dict[str, str] = {}
    sub = None
    for raw in inner:
        hm = _PROVENANCE_DEFAULTS_HDR.match(raw)
        if hm:
            sub = hm.group("sub")
            if hm.group("note"):
                out[sub] = hm.group("note")
            continue
        if sub is not None:
            dm = _PROVENANCE_DEFAULTS_KV_RE.match(raw)
            if dm:
                out[f"{sub}.{dm.group('key')}"] = dm.group("val")
                continue
            # A line not shaped as a row ends the sub-block; the
            # top-level k/v matcher below may still pick it up.
            sub = None
        m = _PROVENANCE_KV_RE.match(raw)
        if m:
            out[m.group("key")] = m.group("val")
    return out


#: The version a ``generator-version`` value names -- its first word, three
#: numbers -- absent from a file written before 2.0.0, which named only its
#: git commit.
_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)\b")


def written_by(text: str) -> Optional[str]:
    """The molbuilder version that wrote a deck or run script, read from
    its PROVENANCE block -- ``None`` for one written before 2.0.0, which
    recorded only its git commit, or for a file with no block."""
    block = read_provenance_block(text) or {}
    m = _VERSION_RE.match(block.get("generator-version", ""))
    return m.group(1) if m else None


def molbuilder_words(version: Optional[str]) -> str:
    """The molbuilder that wrote a file, as a person reads it --
    ``molbuilder 2.9.6``, or ``a molbuilder from before 2.0.0`` for one that
    named no version (:func:`written_by`).  Every door that names who wrote
    a deck says it so."""
    return f"molbuilder {version}" if version else "a molbuilder from before 2.0.0"


# --------------------------------------------------------------------- #
#  The bench-marks block                                                #
# --------------------------------------------------------------------- #


def _coerce_scalar(s: str) -> Any:
    """Best-effort numeric coercion for BENCH-MARKS scalar values.
    Returns the original string when neither int nor float parses."""
    s = s.strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def read_bench_marks_block(text: str) -> Optional[Dict[str, Any]]:
    """Find the BENCH-MARKS block and return its structured payload.

    Returns ``None`` when no BENCH-MARKS block is present.
    """
    inner = _first_pair(text, BLOCK_BENCH_MARKS)[0]
    if inner is None:
        return None
    out: Dict[str, Any] = {"fields": []}
    for raw in inner:
        # Strip "#   " comment prefix.
        s = raw.lstrip("#").strip()
        if not s:
            continue
        # `field <name> anchor=<x> type=<y> ...` rows.
        if s.startswith("field "):
            tokens = s.split()
            field: Dict[str, Any] = {"name": tokens[1]}
            for tok in tokens[2:]:
                if "=" not in tok:
                    continue
                k, _, v = tok.partition("=")
                v = v.strip()
                # Coerce numeric range / default values.
                if k == "range" and v.startswith("[") and v.endswith("]"):
                    try:
                        a, b = v[1:-1].split(",")
                        field["range"] = [_coerce_scalar(a), _coerce_scalar(b)]
                    except ValueError:
                        field["range"] = v
                elif k == "default":
                    field["default"] = _coerce_scalar(v)
                else:
                    field[k] = v
            out["fields"].append(field)
            continue
        # Top-level `key value` scalars.
        if " " in s and not s.startswith("field "):
            k, _, v = s.partition(" ")
            v = v.strip()
            if k == "version":
                out["version"] = v
            elif v.lower() in ("true", ".true."):
                out[k] = True
            elif v.lower() in ("false", ".false."):
                out[k] = False
            else:
                out[k] = _coerce_scalar(v)
    return out

#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle (`runwrap._members`).
TRAVELS_WITH = ("molbuilder.trajectory_log.keys",)


__all__ = ["BUILD_NAME", "BLOCK_PROVENANCE", "BLOCK_BENCH_MARKS",
           "BLOCK_ATOM_METADATA", "BLOCK_ENGINE_OFFSET", "BLOCK_VIBRATION",
           "BLOCK_USER_CUSTOM", "BLOCK_PARAMETERS", "BLOCK_RUN_SETTINGS",
           "MARKER_RE", "same_calculation",
           "begin_marker", "end_marker", "order_record_of", "read_json_block",
           "DeckRecordUnreadable", "extract_atom_metadata",
           "extract_engine_offset", "extract_vibration_record",
           "PARAMETER_ROW_PREFIX", "PARAMETERS_KEY", "read_parameters_fence",
           "read_provenance_block", "read_bench_marks_block", "written_by",
           "molbuilder_words", "box_of_run", "BLOCK_SHARED", "without_stamps"]
