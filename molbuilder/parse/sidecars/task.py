"""A transport calculation's description, read back — ``task.json`` at its
root: the report's handle (`web/results.md` § 0.1, § 2.5).

WHY THIS EXISTS.  A transport calculation's report is composed on read
from the calculation's folder (`engines/transport.md` § 2a.12), so a
ladder in progress has one before any `summarize task`; and
`/api/results/dir` asks the registry *what reads this file*, dropping
anything the answer is ``None`` for.  The file that makes the folder a
transport calculation -- its description -- is what the Results panel
opens at the root, as a benchmark's root opens its ``job-set.json``
(`job_set.py`); ``<label>.transport.json`` is `summarize task`'s copy for
the command line.

WHY IT CLAIMS ONLY A TRANSPORT CALCULATION'S.  Every calculation has a
``task.json``; only a transport one has a report composed at its root.
The discriminator is the description's own ``calculation``, read through
its one door (`task.read_task`) -- never the file's place or name alone.
"""
from __future__ import annotations

import json
from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import UnknownFormatError
from molbuilder.parse.types import SidecarResult

from ._helpers import build_sidecar_result

#: The parse-side name of what came out: the description of a transport
#: calculation, the report's handle.
RESULT_SCHEMA = "transport-task/v1"


class TaskReadError(ValueError):
    """Why this file is not a transport calculation's description."""


def _load_transport_task(path: Path) -> dict:
    """The description's JSON, or :class:`TaskReadError` saying why not --
    one reader for both halves of the parser."""
    from molbuilder.runfiles import TASK_FILE
    from molbuilder.task import read_task
    p = Path(path)
    if p.name != TASK_FILE:
        raise TaskReadError(f"{p.name} is not {TASK_FILE}")
    try:
        task = read_task(p)
    except OSError as exc:
        raise TaskReadError(f"{p.name} could not be read: {exc}") from exc
    except Exception as exc:                                  # noqa: BLE001
        raise TaskReadError(
            f"{p.name} does not read as a description: "
            f"{type(exc).__name__}: {exc}") from exc
    if task.calculation != "transport":
        raise TaskReadError(
            f"{p.name} describes a {task.calculation} calculation, whose "
            f"root opens no report")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskReadError(f"{p.name}: {exc}") from exc


class TransportTaskFileParser(FileParser):
    """A transport calculation's ``task.json`` -- the handle of its report,
    composed on read (`engines/transport.md` § 2a.12)."""

    name   = "transport-task"
    label  = "a transport calculation's description (its report's handle)"
    hint   = ("task.json at a transport calculation's root -- the Results "
              "panel opens the calculation's report through it")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        try:
            _load_transport_task(Path(path))
        except TaskReadError:
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        try:
            payload = _load_transport_task(Path(path))
        except TaskReadError as exc:
            raise UnknownFormatError(str(exc)) from exc
        return build_sidecar_result(
            payload=payload,
            schema=RESULT_SCHEMA,
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["TaskReadError", "TransportTaskFileParser", "RESULT_SCHEMA"]
