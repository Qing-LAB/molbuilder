"""A transport calculation's description, read back — ``task.json`` at its
root: the report's handle (`web/results.md` § 0.1, § 2.5).

ON THE DESCRIPTION'S FLOOR (2), beside `task`, whose door it reads through
(`execution/architecture.md` § 2.1): the parse registry imports none of it,
and the surfaces register it at start (`calculation.register_readers`).

WHY THIS EXISTS.  A transport calculation's report is composed on read
from the calculation's folder (`engines/transport.md` § 2a.12), so a
ladder in progress has one before any `summarize task`; and
`/api/results/dir` asks the registry *what reads this file*, dropping
anything the answer is ``None`` for.  The file that makes the folder a
transport calculation -- its description -- is what the Results panel
opens at the root, as a benchmark's folder opens its ``job-set.json``
(`jobset.sweep_reader`); ``<label>.transport.json`` is `summarize task`'s copy for
the command line.

WHY IT CLAIMS ONLY A TRANSPORT CALCULATION'S.  Every calculation has a
``task.json``; only a transport one has a report composed at its root.
The discriminator is the description's own ``calculation``, read through
its one door (`task.of`) -- never the file's place or name alone.
"""
from __future__ import annotations

from pathlib import Path

from .parse.base import FileParser
from .parse.errors import UnknownFormatError
from .parse.sidecars._helpers import build_sidecar_result
from .parse.types import SidecarResult

#: The parse-side name of what came out: the description of a transport
#: calculation, the report's handle.
RESULT_SCHEMA = "transport-task/v1"


class TaskReadError(ValueError):
    """Why this file is not a transport calculation's description."""


def _load_transport_task(path: Path) -> dict:
    """The description's JSON, or :class:`TaskReadError` saying why not --
    one reader for both halves of the parser."""
    from .runfiles import TASK_FILE
    from .task import of as task_of
    p = Path(path)
    if p.name != TASK_FILE:
        raise TaskReadError(f"{p.name} is not {TASK_FILE}")
    try:
        # THE DESCRIPTION'S ONE DOOR (`task.of`, A16): what every other
        # door reads it with, a description written before a rule refused
        # naming the command that rewrites it; its record is the
        # description's own encoding (`Task.to_dict`).
        task = task_of(p.parent)
        if task is None:
            raise FileNotFoundError(p)
        obj = task.to_dict()
    except Exception as exc:                                  # noqa: BLE001
        raise TaskReadError(
            f"{p.name} does not read as a description: "
            f"{type(exc).__name__}: {exc}") from exc
    if task.calculation != "transport":
        raise TaskReadError(
            f"{p.name} describes a {task.calculation} calculation, whose "
            f"root opens no report")
    return obj


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
