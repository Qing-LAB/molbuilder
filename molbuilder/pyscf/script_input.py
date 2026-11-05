"""THE SAME CALCULATION, PySCF's reading -- whether two PySCF scripts
compute the same thing, the run settings aside (`execution/job-request.md`
§ 6): the reading `deck_record.same_calculation` asks for a PySCF deck.

A PySCF deck is a Python script, so what it computes is its lines.  Three
kinds of line say nothing of it, and are set aside: its RUN-SETTINGS block
(`execution/job-contracts.md` § 3.1) -- the names the script gives its run
settings, which every other line reads by name; a comment, molbuilder's
blocks among them; and molbuilder's stamps (`deck_record.without_stamps`).
The run's structure line (:data:`SIDECAR_NAME`) is compared by what it
holds: a field the structure reader retires
(`sidecars.molstruct.RETIRED_ENVELOPE_KEYS`), which nothing reads, and the
record's version number, which says which fields a record may hold and
nothing of the structure, are no difference (`execution/job-request.md`
§ 5: *the test is the engine's input*).
Every other line is compared as written: an edit that would compute the
same thing written another way is called different, which is the safe
direction.

It reads with the deck record's grammar (`deck_record`), and the structure
line by the structure reader's own list of retired keys
(`sidecars.molstruct`); it is the part PySCF's engine record names
(``same_calculation``), which the door resolves.
"""
from __future__ import annotations

from typing import List

from ..deck_record import (BLOCK_RUN_SETTINGS, begin_marker, end_marker,
                           without_stamps)


def script_lines(text: str) -> List[str]:
    """The lines of ``text`` that say what the script computes: its
    RUN-SETTINGS block, its comment lines and its blank lines aside, its
    stamps masked."""
    begin, end = begin_marker(BLOCK_RUN_SETTINGS), end_marker(BLOCK_RUN_SETTINGS)
    masked = without_stamps(text.encode("utf-8")).decode("utf-8")
    out: List[str] = []
    in_block = False
    for line in masked.splitlines():
        bare = line.strip()
        if bare == begin:
            in_block = True
            continue
        if bare == end:
            in_block = False
            continue
        if in_block or not bare or bare.startswith("#"):
            continue
        out.append(_as_read(line.rstrip()))
    return out


#: THE NAME the run's structure line binds in a PySCF deck -- written by
#: `pyscf.input.emit_sidecar_constant`, read back by the same-calculation reading
#: (:func:`_as_read`).
SIDECAR_NAME = "_MB_SIDECAR"


def _as_read(line: str) -> str:
    """``line`` as compared: the run's structure line without the envelope's
    retired keys and its version number; any other line, and a structure
    line that does not read as the literal molbuilder writes, as written."""
    import ast
    from ..sidecars.molstruct import RETIRED_ENVELOPE_KEYS
    head = f"{SIDECAR_NAME} = "
    if not line.startswith(head):
        return line
    try:
        payload = ast.literal_eval(line[len(head):])
    except (ValueError, SyntaxError):
        return line
    if not isinstance(payload, dict):
        return line
    aside = {*RETIRED_ENVELOPE_KEYS, "schema_version"}
    return head + repr({k: v for k, v in payload.items() if k not in aside})


def script_differences(ran: str, now: str) -> List[str]:
    """Every line where the script a run ran (``ran``) and the one its stage
    renders now (``now``) compute differently, in words -- ``[]`` when they
    are the same calculation."""
    import difflib
    a, b = script_lines(ran), script_lines(now)
    out: List[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        had = " | ".join(x.strip() for x in a[i1:i2]) or "nothing"
        has = " | ".join(x.strip() for x in b[j1:j2]) or "nothing"
        out.append(f"the run had {had!r}, it renders {has!r} now")
    return out


__all__ = ["script_differences", "script_lines"]
