"""WHAT ``summarize task`` ADDS UP FOR A VIBRATION -- its force-constant
stages compared, a displacement sweep (`engines/vibration.md` § 5.9): the
summary the kind table names for the kind on both engines (``sum``,
`execution/architecture.md` § 3.3).  Only a SIESTA vibration with two or
more force-constant stages has one; any other says why
(`spectra.displacement_sweep.why_no_sweep`).

MODULE  jobset.vibration_sum (floor 6, beside `jobset.summarize`): the
        sweep's reader and writer are the vibration kind's
        (`spectra.displacement_sweep`); this declares the summary from them.
"""
from __future__ import annotations

from molbuilder.spectra.displacement_sweep import why_no_sweep
from molbuilder.jobset.summarize import Summed, SumError, TaskSum


def _run(calc, base, *, say, tolerance_cm1=None) -> Summed:
    """The force-constant stages' results, which their jobs wrote, read where
    they are and compared -- nothing derived that a run did not already
    write, nothing moved; ``tolerance_cm1`` flags a mode whose frequency
    spreads by more than it.  ``say`` is unused: the sweep says nothing as
    it reads."""
    from molbuilder.spectra.displacement_sweep import (
        SweepError, collect_sweep, sweep_table_text, write_sweep)
    try:
        rec = collect_sweep(calc, tolerance_cm1=tolerance_cm1)
    except SweepError as e:
        raise SumError(str(e)) from None
    out = write_sweep(base, rec)
    return Summed(text=sweep_table_text(rec), written=(out,),
                  ledger={"out": str(out),
                          "stages": [x["name"] for x in rec["stages"]],
                          "pending": len(rec["pending"]),
                          "failed": len(rec["failed"]),
                          "tolerance_cm1": tolerance_cm1})


SUM = TaskSum(event="displacement-sweep", run=_run,
              options=("tolerance_cm1",), why_none=why_no_sweep)
