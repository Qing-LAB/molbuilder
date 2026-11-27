"""A run's notification record, read back — ``notified.json`` at a run's
root (`execution/run-reports.md` § 2.5): every event the run sent, in
order, each with the epoch second it was sent at.

Its one reader is `runrecord.read_notified_file`; this parser is the
registry's door to the same file, so the Results tab lists it as an aspect
of the run (`web/results.md` § 0.1).  It claims the file by its name and
refuses on parse (`model/parse.md` § 6): one that does not read is listed
and, opened, refused in its reader's words.
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..errors import FileUnreadable
from ..types import SidecarResult
from ...runfiles import NOTIFIED_RECORD_FILE
from ._helpers import build_sidecar_result


class NotifiedRecordError(FileUnreadable):
    """A ``notified.json`` that is not one, or is not this version."""


class NotifiedFileParser(FileParser):
    """Parse a run's notification record — the events sent for the run.
    Returns a :class:`SidecarResult` with ``schema = "run-notified/v<N>"``,
    ``N`` the major of the schema its reader read it at."""

    name   = "run-notified-json"
    label  = f"molbuilder {NOTIFIED_RECORD_FILE} record"
    hint   = (f"a run's notification record -- {NOTIFIED_RECORD_FILE} at a "
              f"run's root, written by the monitor's notify verb")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # BY ITS NAME: whether it reads is the parse's to say.
        p = Path(path)
        return p.name == NOTIFIED_RECORD_FILE and p.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        from molbuilder.runrecord import (NOTIFIED_SCHEMA, LaunchRecordError,
                                          read_notified_file)
        try:
            sent = read_notified_file(Path(path))
        except LaunchRecordError as exc:
            raise NotifiedRecordError(str(exc)) from exc
        payload = {"schema": NOTIFIED_SCHEMA, "sent": sent}
        return build_sidecar_result(
            payload=payload,
            kind="run-notified",
            parser_name=cls.name, source=Path(path))
