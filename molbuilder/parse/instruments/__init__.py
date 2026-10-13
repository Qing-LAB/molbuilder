"""Instrument parsers — what the WRAPPER measured (`parse.md` § 5c).

Registered at import, like `engines/` and `sidecars/` beside them.
"""
from molbuilder.parse.registry import register

from .monitor import MonitorLogFileParser   # re-export
from .scf_timing import ScfTimingFileParser   # re-export
from .util_csv import UtilCsvFileParser   # re-export
from .utilisation import utilisation   # re-export

register(ScfTimingFileParser)
register(MonitorLogFileParser)
register(UtilCsvFileParser)

__all__ = ["ScfTimingFileParser", "MonitorLogFileParser", "UtilCsvFileParser", "utilisation"]
