"""The benchmark sweep's plan, read back — ``job-set.json`` with ``kind: sweep``.

WHY THIS EXISTS.  `/api/results/dir` asks the registry *what reads this
file*, and the Results picker drops anything the answer is ``None`` for,
so a sweep's plan needs a reader to be offered.

WHY IT CLAIMS ONLY SWEEPS, AND WHY THAT IS NOT A HEURISTIC.  Two
different documents share the name ``job-set.json``: a benchmark sweep
and an ordinary calculation's stage ladder.  They are the same schema and
differ in one declared field, ``kind`` — so the discriminator is read,
never guessed.  A LADDER's `/api/bench/summary` answers 400, and this
parser declines it for that reason.  `TransportRecordFileParser` states
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
    ``parse`` cannot come to disagree about what they are looking at.

    WHAT THIS DOES NOT FIX.  `detect()` fans a boolean `can_parse` over
    every registered parser, so it cannot attribute a refusal to one and
    answers its own generic sentence.  The picker asks `detect`, so a
    damaged plan STILL lists as an absence there -- the reason reaches a
    caller that asks this parser directly, and nothing else.  Carrying a
    reason through the fan-out is a registry change, not a parser one.
    """
    from molbuilder.jobset.model import FILENAME, KIND_SWEEP, JobSet

    p = Path(path)
    if p.name != FILENAME:
        raise JobSetReadError(f"{p.name} is not {FILENAME}")
    # THROUGH THE ONE DOOR, `JobSet.load` (`persist`, the schema by name and
    # major -- `execution/architecture.md` § 3.2).
    try:
        js = JobSet.load(p)
    except OSError as exc:
        raise JobSetReadError(f"{p.name} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JobSetReadError(
            f"{p.name} is not valid JSON -- a killed write leaves this "
            f"shape: {exc}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise JobSetReadError(
            f"{p.name} does not read as a job set: {exc}") from exc
    if js.kind != KIND_SWEEP:
        raise JobSetReadError(
            f"{p.name} says kind {js.kind!r}, not {KIND_SWEEP!r} -- "
            f"an ordinary calculation's stage ladder is also called "
            f"{FILENAME}, and its result is the runs below it, not this file")
    return js.to_dict()


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
