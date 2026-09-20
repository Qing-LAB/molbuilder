"""The benchmark sweep's plan, read back — ``job-set.json`` with ``kind: sweep``.

WHY THIS EXISTS.  `/api/results/dir` asks the registry *what reads this
file*, and the Results picker drops anything the answer is ``None`` for.
The answer for a sweep's plan was nothing, so the whole bench-summary
viewer became unreachable from the tab on 2026-09-18 — measured on
``projects/AuSlab/optimization/slab_op1/01_coarse/bench``, whose only two
files are ``job-set.json`` and ``STAGE-PLAN.md`` and which therefore
listed as *"no result files yet"*.  This is the same gap
``TransportRecordFileParser`` closed for ``.transport.json`` on
2026-09-17, and for the same stated reason: a result kind whose format
was understood only in JavaScript.

WHY IT CLAIMS ONLY SWEEPS, AND WHY THAT IS NOT A HEURISTIC.  Two
different documents share the name ``job-set.json``: a benchmark sweep
and an ordinary calculation's stage ladder.  They are the same schema and
differ in one declared field, ``kind`` — so the discriminator is read,
never guessed.  That distinction is the whole of the objection that
justified dropping the file in the first place (*"a `job-set.json` whose
own route then answers 400"*): the 400 belongs to the LADDER, whose
`/api/bench/summary` correctly refuses, and this parser declines it for
that reason rather than by accident.  `TransportRecordFileParser` states
the rule: **the schema, not the suffix alone.**

THE IMPORTS ARE LAZY ON PURPOSE.  ``jobset`` and ``parse`` are both L2
and ``jobset.model`` is pure stdlib, so the dependency is legal
sideways — but ``jobset/__init__`` pulls in ``prep`` / ``submit`` /
``runstatus``, and every jobset-to-parse import is already function-level
to keep that from closing into a cycle.  Importing at module scope here
would both reopen it from the other side and make ``parse`` drag in the
whole framework, against ``jobset``'s own *"import-light by design"*.
"""

from __future__ import annotations

import json
from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import UnknownFormatError
from molbuilder.parse.types import SidecarResult

from ._helpers import build_sidecar_result

#: The discriminator consumers type-narrow on, in the sidecar convention
#: (``"molstruct/v3"``, ``"spectra/v4"``, ``"transport/v1"``).  Derived
#: from nothing here: the on-disk schema is `jobset.model.SCHEMA`, and
#: this is the parse-side name for what came out of it.
RESULT_SCHEMA = "job-set/v1"


class JobSetReadError(ValueError):
    """Why this file is not a sweep's plan, in words -- surfaced
    verbatim, so a refusal names the actual cause."""


def _load_sweep(path: Path) -> dict:
    """The payload, or :class:`JobSetReadError` saying why not.

    One reader for both halves of the parser, so ``can_parse`` and
    ``parse`` cannot come to disagree about what they are looking at --
    the failure `sidecars/__init__` records for the deleted
    ``TransportSidecarFileParser``, which claimed a shape nothing wrote.

    IT RETURNED ``None`` FOR EVERYTHING until 2026-09-19, and `parse`
    turned every one of those into the same sentence: *"an ordinary
    calculation's ladder says `kind: ladder`"*.  So a sweep plan
    truncated by a killed write, or one that could not be opened,
    reported as a perfectly healthy ladder -- and the bench directory
    listed as *"no result files yet"*, the exact symptom this parser
    exists to remove, with nothing saying the file was damaged.
    `sidecars/transport.py`, the model for this file, carries the real
    cause in a `TransportRecordError`; that half was not copied.

    WHAT THIS DOES NOT FIX.  `detect()` fans a boolean `can_parse` over
    every registered parser, so it cannot attribute a refusal to one and
    answers its own generic sentence.  The picker asks `detect`, so a
    damaged plan STILL lists as an absence there -- the reason reaches a
    caller that asks this parser directly, and nothing else.  Carrying a
    reason through the fan-out is a registry change, not a parser one.
    """
    from molbuilder.jobset.model import FILENAME, KIND_SWEEP, SCHEMA

    p = Path(path)
    if p.name != FILENAME:
        raise JobSetReadError(f"{p.name} is not {FILENAME}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise JobSetReadError(f"{p.name} could not be read: {exc}") from exc
    try:
        said = json.loads(text)
    except ValueError as exc:
        raise JobSetReadError(
            f"{p.name} is not valid JSON -- a killed write leaves this "
            f"shape: {exc}") from exc
    if not isinstance(said, dict):
        raise JobSetReadError(
            f"{p.name} holds a {type(said).__name__}, not an object")
    if said.get("schema") != SCHEMA:
        raise JobSetReadError(
            f"{p.name} says schema {said.get('schema')!r}, not {SCHEMA!r}")
    if said.get("kind") != KIND_SWEEP:
        raise JobSetReadError(
            f"{p.name} says kind {said.get('kind')!r}, not {KIND_SWEEP!r} -- "
            f"an ordinary calculation's stage ladder is also called "
            f"{FILENAME}, and its result is the runs below it, not this file")
    return said


class JobSetSweepFileParser(FileParser):
    """``job-set.json`` — a benchmark sweep's plan."""

    name = "job-set-sweep"
    label = "molbuilder benchmark sweep plan"
    hint = ("a benchmark sweep's job set -- job-set.json with "
            "`kind: sweep`, written by `prep bench <stage>`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        try:
            _load_sweep(Path(path))
        except JobSetReadError:
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        try:
            payload = _load_sweep(Path(path))
        except JobSetReadError as exc:
            # The canonical error `base.FileParser.parse` requires,
            # carrying the reason rather than a guess about it.
            raise UnknownFormatError(str(exc)) from exc
        return build_sidecar_result(
            payload=payload,
            schema=RESULT_SCHEMA,
            parser_name=cls.name,
            source=Path(path),
        )
