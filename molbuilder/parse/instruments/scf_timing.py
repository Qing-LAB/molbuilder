"""``<base>-runN.scf-timing.log`` — seconds per SCF iteration.

The wrapper tees every SCF row of either phase -- SIESTA's ``scf:`` and
TranSIESTA's ``ts-scf:`` -- into this file with an epoch stamp in front
(`running-a-job.md` § 4.1), so consecutive deltas of one phase ARE its
per-iteration durations.  Nothing else in the run states that number.
"""
from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result
from .scf_timing_rows import scf_timing_metrics


class ScfTimingFileParser(FileParser):
    """The wrapper's SCF-timing tee."""

    name   = "scf-timing"
    label  = "wrapper SCF-timing log (.scf-timing.log)"
    hint   = "files ending in .scf-timing.log"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".scf-timing.log") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=scf_timing_metrics(text),
                                       parser_name=cls.name, source=path)
