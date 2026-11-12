"""THE SAME CALCULATION, PySCF's reading -- whether two PySCF scripts
compute the same thing, the run settings aside (`execution/job-request.md`
§ 6): the reading `deck_record.same_calculation` asks for a PySCF deck.

A PySCF deck is a Python script, so what it computes is its statements, AS
PYTHON READS THEM (:func:`script_lines`): its comments, its layout and the
way a value is spelled say nothing of it.  Set aside besides, each saying
nothing of it either: its RUN-SETTINGS block (`execution/job-contracts.md`
§ 3.1) -- the names the script gives its run settings, which every other
statement reads by name; molbuilder's stamps (`deck_record.without_stamps`);
and WHAT IT ONLY REPORTS (the user, 2026-11-12: "ignore the report-only
text"): a statement filling one of the names it reports under
(:data:`REPORTS`) -- its parameters record, its runtime facts, the
convergence targets its progress log shows, a vibration's settings copy,
Methods text, bibliography and result-record version, and its words: each
message it records in its result, hands a call or writes on a saved
geometry -- a message it prints, a string standing alone (its
docstrings), and the message a ``raise`` carries, the raise itself
compared.  A later release may word or
record any of them otherwise (2.69.43 recorded another default), and nothing
PySCF computes reads them; every value that changes what PySCF computes is
stated by statements that are compared (`test_every_form_field_reaches_
the_deck` holds it of every field a form offers).  A statement filling a
report is set aside whole, whatever it calls: molbuilder's decks only read
there, and a value a person's edit sets inside one is not seen.  A script
that does not read as Python is compared line by line, as it is written
(:func:`_as_written`), only its comment lines, run settings and stamps
aside.
The run's structure line (:data:`SIDECAR_NAME`) is compared by what it
holds: a field the structure reader retires
(`sidecars.molstruct.RETIRED_ENVELOPE_KEYS`), which nothing reads, and the
record's version number, which says which fields a record may hold and
nothing of the structure, are no difference (`execution/job-request.md`
§ 5: *the test is the engine's input*).
Every other statement is compared as Python reads it: one that would
compute the same thing written another way is called different, which is
the safe direction.

It reads with the deck record's grammar (`deck_record`), and the structure
line by the structure reader's own list of retired keys
(`sidecars.molstruct`); it is the part PySCF's engine record names
(``same_calculation``), which the door resolves.
"""
from __future__ import annotations

import ast
from typing import List, Optional, Set

from ..deck_record import (BLOCK_RUN_SETTINGS, begin_marker, end_marker,
                           without_stamps)


#: THE NAMES A PySCF DECK ONLY REPORTS UNDER, which nothing PySCF computes
#: reads -- one spelling, which the deck's writers write with
#: (`pyscf.input`, `pyscf.vibration_emitters`, `pyscf.vibration_deck`) and
#: the same-calculation
#: reading sets aside (:func:`script_lines`): the parameters record, a line
#: per setting (`pyscf.input._emit_effective_parameters`); the runtime facts,
#: filled for the record and the progress log; the convergence targets the
#: progress log shows; and, for a vibration's result
#: (`engines/vibration.md` § 6.2), its settings copy, Methods text,
#: bibliography and the result record's version number; and its WORDS --
#: each message it records in its result, hands a call or writes on a saved
#: geometry, put there where it is composed and read from there where it is
#: used (`pyscf.input.emit_words`), so a release that words it otherwise
#: changes no statement that is compared.
PARAMS_NAME = "_MB_PARAMS"
RUNTIME_NAME = "_RUNTIME_INFO"
TARGETS_NAME = "_CONVERGENCE_TARGETS"
CONFIG_NAME = "CONFIG"
METHODS_NAME = "METHODS_TEXT"
BIBLIOGRAPHY_NAME = "BIBLIOGRAPHY_KEYS"
RESULT_VERSION_NAME = "SCHEMA_VERSION"
WORDS_NAME = "_MB_WORDS"
REPORTS = (PARAMS_NAME, RUNTIME_NAME, TARGETS_NAME, CONFIG_NAME,
           METHODS_NAME, BIBLIOGRAPHY_NAME, RESULT_VERSION_NAME, WORDS_NAME)

#: THE NAME the run's structure line binds in a PySCF deck -- written by
#: `pyscf.input.emit_sidecar_constant`, read back by the same-calculation
#: reading by what it holds (:func:`_by_what_it_holds`).
SIDECAR_NAME = "_MB_SIDECAR"

#: What the message of a ``raise`` is compared as: never its words.
_A_MESSAGE = "<message>"


def script_lines(text: str) -> List[str]:
    """The lines of ``text`` that say what the script computes: its
    statements as Python reads them, what it only reports and its run
    settings aside, written out again one statement to a line and its stamps
    masked; for a script that does not read as Python, its lines as written
    (:func:`_as_written`)."""
    # ONE LINE BREAK, Python's: the run-settings block is found by its
    # markers' line numbers, which must be the parser's.
    plain = text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(plain)
    except SyntaxError:
        return _as_written(text)
    begin, end = begin_marker(BLOCK_RUN_SETTINGS), end_marker(BLOCK_RUN_SETTINGS)
    settings: Set[int] = set()
    inside = False
    for number, line in enumerate(plain.split("\n"), start=1):
        bare = line.strip()
        if bare in (begin, end):
            inside = bare == begin
        elif inside:
            settings.add(number)
    read = ast.unparse(_WhatItComputes(settings).visit(tree))
    masked = without_stamps(read.encode("utf-8")).decode("utf-8")
    return [line.rstrip() for line in masked.split("\n") if line.strip()]


class _WhatItComputes(ast.NodeTransformer):
    """A parsed script with what it only reports and its run settings taken
    out (:func:`script_lines`), the message of each ``raise`` made one, and
    its structure line read by what it holds."""

    def __init__(self, settings: Set[int]):
        self._settings = settings

    def generic_visit(self, node):
        if isinstance(node, ast.stmt) and (node.lineno in self._settings
                                           or _only_reports(node)):
            return None
        node = super().generic_visit(node)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            node.exc.args = [_message_aside(a) for a in node.exc.args]
            for keyword in node.exc.keywords:
                keyword.value = _message_aside(keyword.value)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == SIDECAR_NAME):
            node.value = _by_what_it_holds(node.value)
        return node


def _only_reports(node) -> bool:
    """Whether the statement ``node`` only reports: an assignment to or into
    a name of :data:`REPORTS`, a call of such a name's own method, a
    ``print``, or a string standing alone."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target])
        return all(_reported_under(t) in REPORTS for t in targets)
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    if not isinstance(value, ast.Call):
        return False
    if isinstance(value.func, ast.Name):
        return value.func.id == "print"
    return (isinstance(value.func, ast.Attribute)
            and _reported_under(value.func.value) in REPORTS)


def _reported_under(node) -> Optional[str]:
    """The name an assignment target or a method's object stands on --
    ``_RUNTIME_INFO`` for ``_RUNTIME_INFO['x']`` -- or ``None``."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _message_aside(node):
    """An argument of a raised exception as compared: its words, a string or
    an f-string, as one message; any other argument as it is."""
    if isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return ast.Constant(_A_MESSAGE)
    return node


def _by_what_it_holds(value):
    """The structure line's value as compared: the record without the
    envelope's retired keys and its version number; one that does not read
    as the literal molbuilder writes, as it is."""
    from ..sidecars.molstruct import RETIRED_ENVELOPE_KEYS
    try:
        payload = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError):
        return value
    if not isinstance(payload, dict):
        return value
    aside = {*RETIRED_ENVELOPE_KEYS, "schema_version"}
    return ast.parse(repr({k: v for k, v in payload.items()
                           if k not in aside}), mode="eval").body


def _as_written(text: str) -> List[str]:
    """The lines of a script that does not read as Python, as compared: its
    run-settings block, its comment lines and its blank lines aside, its
    stamps masked, its structure line by what it holds."""
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
        out.append(_line_by_what_it_holds(line.rstrip()))
    return out


def _line_by_what_it_holds(line: str) -> str:
    """A line of a script that does not read as Python: its structure line by
    what it holds (:func:`_by_what_it_holds`), any other line as written."""
    head = f"{SIDECAR_NAME} = "
    if not line.startswith(head):
        return line
    try:
        value = ast.parse(line[len(head):], mode="eval").body
    except SyntaxError:
        return line
    return head + ast.unparse(_by_what_it_holds(value))


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


__all__ = ["BIBLIOGRAPHY_NAME", "CONFIG_NAME", "METHODS_NAME", "PARAMS_NAME",
           "REPORTS", "RESULT_VERSION_NAME", "RUNTIME_NAME", "SIDECAR_NAME",
           "TARGETS_NAME", "WORDS_NAME",
           "script_differences", "script_lines"]
