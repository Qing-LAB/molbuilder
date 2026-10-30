"""A run of points' index, read back — ``<label>_<stage>-run0.points.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the index's own shape is `runpoints`'s, which writes and reads
it (`execution/runs-of-points.md` § 2.3).  `/api/results/dir` asks the
registry what reads each file and offers only what something reads, so
without this reader the index would sit in the run unoffered.  It claims a
file by its role in the catalogue AND by the schema its writer stamps,
never by the suffix alone -- one reader for every kind's run of points.
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..errors import ParseError
from ..types import SidecarResult
from ._helpers import build_sidecar_result


class RunPointsError(ParseError):
    """A ``.points.json`` that is not one, or is not this version."""


class RunPointsFileParser(FileParser):
    """Parse a run of points' index — every point of the run, what it
    needs and must hold, how it ended as the job saw it; the kind's facts
    about the run; the finish step's entry.  Returns a
    :class:`SidecarResult` with ``schema = "run-points/v1"``."""

    name   = "run-points-json"
    label  = "molbuilder .points.json index"
    hint   = ("a run of points' index -- <label>_<stage>-run<N>.points.json, "
              "written by launch and the run's report steps")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        from molbuilder.runfiles import ROLE_POINTS, role_of
        from molbuilder.runpoints import read_index
        if role_of(Path(path).name) != ROLE_POINTS:
            return False
        try:
            read_index(Path(path))
        except (ValueError, OSError):
            return False
        return True

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        from molbuilder.runpoints import read_index
        try:
            payload = read_index(Path(path))
        except ValueError as exc:
            raise RunPointsError(str(exc)) from exc
        return build_sidecar_result(
            payload=payload, schema="run-points/v1",
            parser_name=cls.name, source=Path(path))


__all__ = ["RunPointsError", "RunPointsFileParser"]
