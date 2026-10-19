"""The listener's record of the reports it received -- ``<state
dir>/reports/<user>.jsonl``, one JSON object a line, as `web/blueprints/
notify.py` writes it (`run-reports.md` § 4.1a) -- read back for the This
machine tab (`this-machine.md` § 4.1).

MODULE  parse.instruments.reports (L3; the parse registry)
ROLE    the one reader of that file: newest first, each line the report's own
        fields, a line that does not read skipped and counted
USED-BY `GET /api/notify/reports` (`notify_setup.py`)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result


def read_reports(path, *, since: Optional[float] = None,
                 limit: int = 200) -> Dict[str, Any]:
    """``{reports, total, skipped}`` -- the reports in the file newest first
    (by ``sent_at``, the line's order where it states none), at most
    ``limit`` of them, only those sent after ``since`` (an epoch second)
    when given; ``total`` every line that read, ``skipped`` the lines that
    did not.  A file that is not there is ``{reports: [], total: 0}``."""
    p = Path(path)
    if not p.is_file():
        return {"reports": [], "total": 0, "skipped": 0}
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with open(p, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(d, dict):
                skipped += 1
                continue
            rows.append(d)
    rows.reverse()                                   # newest first
    rows.sort(key=lambda d: -float(d.get("sent_at") or 0))
    if since is not None:
        rows = [d for d in rows if float(d.get("sent_at") or 0) > since]
    return {"reports": rows[:max(0, int(limit))], "total": len(rows),
            "skipped": skipped}


class ReportsFileParser(FileParser):
    """The listener's reports file, through the registry."""

    name   = "reports-log"
    label  = "the listener's reports (reports/<user>.jsonl)"
    hint   = "a .jsonl file in a reports/ folder"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        p = Path(path)
        return (p.name.endswith(".jsonl") and p.parent.name == "reports"
                and p.is_file())

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        return build_instrument_result(metrics=read_reports(path),
                                       parser_name=cls.name, source=path)
