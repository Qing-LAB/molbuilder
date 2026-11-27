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
``task.json``; only a kind whose root opens its description has a report
composed at its root -- the kind table's ``root_opens``, a transport
calculation's.  The discriminator is the description's own ``calculation``,
read as the file holds it -- never the file's place or name alone, and
never its version: a parser claims its file by what it is and refuses on
parse (`model/parse.md` § 6), so a description of a version this
molbuilder does not read is listed as this parser's and, opened, refused in
its one door's words (`task.of`), never hidden as no parser's.
"""
from __future__ import annotations

from pathlib import Path

from .parse.base import FileParser
from .parse.errors import FileUnreadable
from .parse.sidecars._helpers import build_sidecar_result
from .parse.types import SidecarResult
from .runfiles import TASK_FILE


class TaskReadError(FileUnreadable):
    """Why this file is not a transport calculation's description -- a
    parser's refusal of its file, carrying the mark
    (`parse.errors.FileUnreadable`)."""


def _load_transport_task(path: Path) -> dict:
    """The description's JSON, read through its one door (`task.of`), or
    :class:`TaskReadError` saying why not, in that door's words."""
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
        # THE READER'S OWN WORDS, never a Python class name before them.
        raise TaskReadError(
            f"{p.name} does not read as a description: {exc}") from exc
    from .kind_table import ROOT_OPENS, kind
    if kind(task.calculation).root_opens != ROOT_OPENS[0]:
        raise TaskReadError(
            f"{p.name} describes {kind(task.calculation).with_article}, "
            f"whose root opens no report")
    return obj


class TransportTaskFileParser(FileParser):
    """A transport calculation's ``task.json`` -- the handle of its report,
    composed on read (`engines/transport.md` § 2a.12).  Returns a
    :class:`SidecarResult` with ``schema = "transport-task/v<N>"``, ``N`` the
    major of the description's own ``schema`` (`model/parse.md` § 6)."""

    name   = "transport-task"
    label  = "a transport calculation's description (its report's handle)"
    hint   = (f"{TASK_FILE} at a transport calculation's root -- the "
              f"Results panel opens the calculation's report through it")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # BY ITS NAME AND ITS KIND, the description's own ``calculation`` as
        # the file holds it (`persist.read_object`, no schema asked): the
        # version is the parse's to refuse.
        from .kind_table import ROOT_OPENS, kind
        from .persist import read_object
        p = Path(path)
        if p.name != TASK_FILE or not p.is_file():
            return False
        try:
            calculation = read_object(p).get("calculation")
            return kind(str(calculation)).root_opens == ROOT_OPENS[0]
        except (OSError, ValueError):     # no object, or a kind none holds
            return False

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        return build_sidecar_result(
            payload=_load_transport_task(Path(path)),
            kind="transport-task",
            parser_name=cls.name,
            source=Path(path),
        )


__all__ = ["TaskReadError", "TransportTaskFileParser"]
