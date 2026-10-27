"""SIESTA's optical spectrum, ``<label>.EPSIMG`` -- the registered parser
(`model/parse.md` § 5f).

MODULE  parse.engines.siesta_epsimg (floor 1)
ROLE    :class:`SiestaEpsimgFileParser`, ``siesta-epsimg`` →
        :class:`~molbuilder.parse.types.DielectricResult`: what the registry
        answers for an ``.EPSIMG`` -- a person's script through
        `molbuilder.postprocess.parse`, the Results tab's file card -- read
        by the file's one reader (`siesta_epsimg_reader.read_epsimg`), the
        same function an optical call's record step reads it with beside the
        job
USED-BY the registry (`parse.engines.__init__`)

**Claimed by its first line, never by its name alone**: SIESTA writes
``<label>.contrib.EPSIMG`` beside it -- each transition's share, the
engine's scratch (`warm-files.toml` [optical]) -- under the same ending.
"""
from __future__ import annotations

from pathlib import Path

from ..base import FileParser
from ..types import DielectricResult, ParseResult
from .siesta_epsimg_reader import is_epsimg, read_epsimg


class SiestaEpsimgFileParser(FileParser):
    """SIESTA's optical calculation's spectrum, ε₂(ħω) per spin channel."""

    name = "siesta-epsimg"
    label = "SIESTA's optical spectrum (.EPSIMG)"
    hint = ("the <label>.EPSIMG an optical calculation writes beside its "
            ".out (OpticalCalculation true)")
    output = DielectricResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        p = Path(path)
        return (p.name.endswith(".EPSIMG") and p.is_file()
                and is_epsimg(p))

    @classmethod
    def parse(cls, path: Path) -> DielectricResult:
        e = read_epsimg(path)
        return DielectricResult(
            **ParseResult.envelope(cls.name, str(path)),
            energy_ev=e.energy_ev, eps2=e.eps2, spin=e.spin,
            energy_range_ev=e.energy_range_ev, f_sum=e.f_sum,
            drude_wp2_ha2=e.drude_wp2_ha2)


__all__ = ["SiestaEpsimgFileParser"]
