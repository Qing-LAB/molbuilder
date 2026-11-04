"""Instrument parsers — what the WRAPPER measured (`parse.md` § 5c).

Registered at import, like `engines/` and `sidecars/` beside them.
"""
from molbuilder.parse.registry import register

from .monitor import MonitorLogFileParser   # re-export
from .reports import ReportsFileParser      # re-export
from .scf_timing import ScfTimingFileParser   # re-export
from .util_csv import UtilCsvFileParser   # re-export
from .utilisation import utilisation   # re-export

register(ScfTimingFileParser)
register(MonitorLogFileParser)
register(ReportsFileParser)
register(UtilCsvFileParser)


def metrics_of(path):
    """AN INSTRUMENT FILE'S METRICS, through its registered parser, or ``{}``
    -- the one read the run record and the bench summary ask.  Both report:
    a file they cannot read tells them less (a trial's file truncated, none
    written).  A defect in a reader is not caught (`parse.md` § 5d.1b)."""
    if path is None:
        return {}
    from pathlib import Path

    from molbuilder.parse.errors import ParseError
    from molbuilder.parse.registry import parse
    try:
        return dict(parse(Path(path)).metrics)
    except (OSError, ValueError, ParseError):
        return {}


__all__ = ["ScfTimingFileParser", "MonitorLogFileParser", "ReportsFileParser",
           "UtilCsvFileParser", "metrics_of", "utilisation"]
