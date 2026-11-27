"""A run of points' index, read back — ``<label>_<stage>.points.json``.

Contract: `model/parse.md` § 5.5 (what a person can open is the REGISTRY's
question); the index's own shape is `runpoints`'s, which writes and reads
it (`execution/runs-of-points.md` § 2.3).  `/api/results/dir` asks the
registry what reads each file and offers only what something reads, so
without this reader the index would sit in the run unoffered.  It claims a
file by its role in the catalogue and refuses on parse (`model/parse.md`
§ 6) -- an index of another version listed as this parser's and, opened,
refused in its reader's words -- one reader for every kind's run of points.
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..errors import FileUnreadable
from ..types import SidecarResult
from ...runfiles import ROLE_POINTS, role_of
from ._helpers import build_sidecar_result


class RunPointsError(FileUnreadable):
    """A ``.points.json`` that is not one, or is not this version."""


class RunPointsFileParser(FileParser):
    """Parse a run of points' index — every point of the run, what it
    needs and must hold, how it ended as the job saw it; the kind's facts
    about the run; the finish step's entry.  Returns a
    :class:`SidecarResult` with ``schema = "run-points/v<N>"``, ``N`` the
    major of the index's own ``schema``."""

    name   = "run-points-json"
    label  = f"molbuilder {ROLE_POINTS} index"
    hint   = (f"a run of points' index -- <label>_<stage>{ROLE_POINTS}, "
              f"written by launch and the run's report steps")
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # BY ITS ROLE: the version is the parse's to refuse.
        return role_of(Path(path).name) == ROLE_POINTS and Path(path).is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        from molbuilder.runpoints import read_index
        try:
            payload = read_index(Path(path))
        except ValueError as exc:
            raise RunPointsError(str(exc)) from exc
        return build_sidecar_result(
            payload=payload,
            kind="run-points",
            parser_name=cls.name, source=Path(path))


__all__ = ["RunPointsError", "RunPointsFileParser"]
