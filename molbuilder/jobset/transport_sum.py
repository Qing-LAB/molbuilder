"""WHAT ``summarize task`` ADDS UP FOR A TRANSPORT CALCULATION -- its
transmission points read into the calculation's record and data file,
current against voltage (`engines/transport.md` § 2a.12): the summary the
kind table names for the pair (``sum``, `execution/architecture.md` § 3.3).

MODULE  jobset.transport_sum (floor 6, beside `jobset.summarize`): the
        points, the record and the data file are the transport kind's
        readers and writers (`transport.tbtnc`, `transport.record`,
        `transport.datafile`); this declares the summary from them.
"""
from __future__ import annotations

from molbuilder.jobset.summarize import Summed, SumError, TaskSum


def _run(calc, base, *, say) -> Summed:
    """Every point's DOS per orbital read where it is not kept -- the record
    and the data file hold each region's (`model/parse.md` § 5e), said as it
    goes -- then the record, and the same results as data on their grid
    beside it, the data file written first as the one of the two that can
    refuse, so a refusal leaves the pair the last summarize wrote.  A point
    that has not run yet reads as pending, never a failure."""
    from molbuilder.transport.datafile import DataFileError, write_data_file
    from molbuilder.transport.record import (
        RecordError, collect_record, iv_table_text, write_record)
    from molbuilder.transport.tbtnc import read_points
    readings = read_points(calc, say=say)
    try:
        rec = collect_record(calc)
    except RecordError as e:
        raise SumError(str(e)) from None
    try:
        data = write_data_file(base, calc.task, rec, readings=readings)
    except DataFileError as e:
        raise SumError(str(e)) from None
    out = write_record(base, rec)
    return Summed(text=iv_table_text(rec), written=(out, data),
                  ledger={"out": str(out), "data": str(data),
                          "points": len(rec["points"]),
                          "pending": len(rec["pending"]),
                          "not_run": len(rec["not_run"]),
                          "failed": len(rec["failed"]),
                          "readings": readings})


SUM = TaskSum(event="transport-record", run=_run)
