"""A run's notification record, read back — ``notified.json`` at a run's
root (`execution/run-reports.md` § 2.5): every event the run sent, in
order, each with the epoch second it was sent at.

Its one reader is `runrecord.read_notified_file`; this parser is the
registry's door to the same file, so the Results tab lists it as an aspect
of the run (`web/results.md` § 0.1).
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ._helpers import build_sidecar_result


class NotifiedRecordError(ParseError):
    """A ``notified.json`` that is not one, or is not this version."""


class NotifiedFileParser(FileParser):
    """Parse a run's notification record — the events sent for the run.
    Returns a :class:`SidecarResult` with ``schema = "run-notified/v1"``."""

    name   = "run-notified-json"
    label  = "molbuilder notified.json record"
    hint   = ("a run's notification record -- notified.json at a run's "
              "root, written by the monitor's notify verb")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        from molbuilder.runfiles import NOTIFIED_RECORD_FILE
        from molbuilder.runrecord import LaunchRecordError, read_notified_file
        p = Path(path)
        if p.name != NOTIFIED_RECORD_FILE:
            return False
        try:
            read_notified_file(p)
        except (LaunchRecordError, OSError):
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        from molbuilder.runrecord import (NOTIFIED_SCHEMA, LaunchRecordError,
                                          read_notified_file)
        try:
            sent = read_notified_file(Path(path))
        except LaunchRecordError as exc:
            raise NotifiedRecordError(str(exc)) from exc
        return build_sidecar_result(
            payload={"schema": NOTIFIED_SCHEMA, "sent": sent},
            schema="run-notified/v1", parser_name=cls.name,
            source=Path(path))
