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
from molbuilder.parse.types import SidecarResult

from ._helpers import build_sidecar_result

#: The discriminator consumers type-narrow on, in the sidecar convention
#: (``"molstruct/v3"``, ``"spectra/v4"``, ``"transport/v1"``).  Derived
#: from nothing here: the on-disk schema is `jobset.model.SCHEMA`, and
#: this is the parse-side name for what came out of it.
RESULT_SCHEMA = "job-set/v1"


def _load_sweep(path: Path) -> "dict | None":
    """The payload when this file is a SWEEP's plan, else ``None``.

    One reader for both halves of the parser, so ``can_parse`` and
    ``parse`` cannot come to disagree about what they are looking at —
    the failure `sidecars/__init__` records for the deleted
    ``TransportSidecarFileParser``, which claimed a shape nothing wrote.
    """
    from molbuilder.jobset.model import FILENAME, KIND_SWEEP, SCHEMA

    if Path(path).name != FILENAME:
        return None
    try:
        said = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(said, dict):
        return None
    if said.get("schema") != SCHEMA or said.get("kind") != KIND_SWEEP:
        return None
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
        return _load_sweep(Path(path)) is not None

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        payload = _load_sweep(Path(path))
        if payload is None:
            from molbuilder.parse.errors import UnknownFormatError
            raise UnknownFormatError(
                f"{Path(path).name} is not a benchmark sweep's job set "
                f"(a sweep says `kind: sweep`; an ordinary calculation's "
                f"ladder says `kind: ladder` and its result is the runs "
                f"below it, not this file)")
        return build_sidecar_result(
            payload=payload,
            schema=RESULT_SCHEMA,
            parser_name=cls.name,
            source=Path(path),
        )
