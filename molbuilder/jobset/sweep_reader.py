"""The benchmark sweep's plan, read back — ``job-set.json`` in a benchmark's folder.

WHY THIS EXISTS.  `/api/results/dir` asks the registry *what reads this
file*, and the Results picker drops anything the answer is ``None`` for,
so a sweep's plan needs a reader to be offered.

WHY IT CLAIMS A BENCHMARK'S PLAN BY ITS NAME AND ITS PLACE.  Two
documents share the name ``job-set.json``: a benchmark's plan, in its
stage's ``bench/`` folder, and an ordinary calculation's stage ladder, at
its root.  Where the file lies says which (`calcdirs.where`, the one answer
to *where am I*, `project-layout.md` § 1.4a) -- never its content: a
parser claims its file by its name and refuses on parse (`model/parse.md`
§ 6), so a plan of a version this molbuilder does not read is listed as a
benchmark's plan and, opened, refused in its reader's words, never hidden
as no parser's.  A LADDER's `/api/bench/summary` answers 400; no ladder
lies in a benchmark's folder.

ON THE PLAN'S FLOOR (3), beside `jobset.model`, whose door it reads
through (`execution/architecture.md` § 2.1): the parse registry imports
none of it, and the surfaces register it at start
(`calculation.register_readers`).  A module of its own, so a reader of the
plan (`jobset.model`, import-light by design) loads no parser.
"""

from __future__ import annotations

from pathlib import Path

from .. import calcdirs
from ..parse.base import FileParser
from ..parse.errors import FileUnreadable
from ..parse.sidecars._helpers import build_sidecar_result
from ..parse.types import SidecarResult
from .model import FILENAME, KIND_SWEEP, JobSet


class JobSetReadError(FileUnreadable):
    """Why this file is not a benchmark's plan, in words -- its reader's
    (`JobSet.of`), the file named in them; a parser's refusal of its file,
    carrying the mark (`parse.errors.FileUnreadable`)."""


def _load_sweep(path: Path) -> dict:
    """The payload, or :class:`JobSetReadError` saying why not: read
    through the one door, `JobSet.of` (`persist`, the schema by name and
    major -- `execution/architecture.md` § 3.2, A16), in its words."""
    p = Path(path)
    try:
        js = JobSet.of(p.parent)
    except OSError as exc:
        raise JobSetReadError(f"{p} could not be read: {exc}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        # THE PLAN'S READER'S OWN WORDS, which name the file.
        raise JobSetReadError(str(exc)) from exc
    if js is None:
        raise JobSetReadError(f"{p} is not there")
    if js.kind != KIND_SWEEP:
        raise JobSetReadError(
            f"{p.name} says kind {js.kind!r}, not {KIND_SWEEP!r} -- "
            f"an ordinary calculation's stage ladder is also called "
            f"{FILENAME}, and its result is the runs below it, not this file")
    return js.to_dict()


class JobSetSweepFileParser(FileParser):
    """``job-set.json`` in a benchmark's folder -- a benchmark's plan.
    Returns a :class:`SidecarResult` with ``schema = "job-set/v<N>"``,
    ``N`` the major of the plan's own ``schema`` (`model/parse.md` § 6)."""

    name = "job-set-sweep"
    label = "molbuilder benchmark sweep plan"
    hint = (f"a benchmark sweep's job set -- {FILENAME} in a stage's "
            "`bench/` folder, written by `prep bench <stage>`")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # BY ITS NAME AND ITS PLACE: the version is the parse's to refuse.
        p = Path(path)
        return (p.name == FILENAME and p.is_file()
                and calcdirs.where(p.parent).what == calcdirs.BENCHMARK)

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        return build_sidecar_result(
            payload=_load_sweep(Path(path)),
            kind="job-set",
            parser_name=cls.name,
            source=Path(path),
        )
