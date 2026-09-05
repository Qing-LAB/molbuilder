"""``<base>.monitor.log`` — what the wrapper's monitor saw.

Three facts, all stated by the monitor and by nothing else:

* the **`[MACHINE]`** line — what kind of node was under this run
  (`scheduler.md` R12).  Written once, first line, at start; the FIRST
  match wins so a corrupted or concatenated log still reads the start
  record.  ``gpu`` is everything after ``gpu=`` because device models
  contain spaces, which is why the monitor puts it last.
* the **`[UTIL-SUMMARY]`** verdict — ``gpu`` | ``host`` | ``mixed``.  It
  encodes the monitor's own thresholds, so it is read here and nowhere
  else.  The LAST line wins.
* the summary's **stated means**, which a caller prefers over the csv's
  reconstruction (`parse.md` § 5c, § 5a's sibling upgrade).

**The summary line can be missing entirely** — it is written only in the
monitor's terminal branch, so a trial the scheduler KILLED leaves a csv
and no summary.  That is the trial a benchmark most needs to read, which
is why nothing here raises on its absence.

*(Moved from `bench/result.py` on 2026-09-04 -- `parse.md` § 5c.)*
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult

from ._helpers import build_instrument_result

_MACHINE = re.compile(
    r"\[MACHINE\]\s+node=(\S+)\s+cores=(\S+)\s+mem_gb=(\S+)\s+gpu=(.+)$")
_VERDICT = re.compile(r"->\s*(GPU-bound|host/CPU-bound|mixed)")
_BOUND_MAP = {"GPU-bound": "gpu", "host/CPU-bound": "host", "mixed": "mixed"}
#: ``cpu mean=NN% (mm-MM); gpu0 sm mean=NN% (mm-MM) -> verdict``
_SUMMARY_CPU = re.compile(r"cpu mean=(\d+(?:\.\d+)?)%")
_SUMMARY_GPU = re.compile(r"gpu\d+ sm mean=(\d+(?:\.\d+)?)%")


def monitor_metrics(text: str) -> Dict[str, Any]:
    """``{machine, bound, stated_cpu_mean_pct, stated_gpu_sm_mean_pct}``.

    Absent facts are absent: ``machine`` is ``{}`` for a log predating the
    line, ``bound`` is ``None`` without a summary, and the stated means
    are omitted rather than defaulted -- a caller must be able to tell
    "the monitor did not say" from "the monitor said zero".
    """
    text = text or ""
    out: Dict[str, Any] = {"machine": {}, "bound": None}
    for ln in text.splitlines():
        m = _MACHINE.search(ln)
        if m:
            out["machine"] = {"node": m.group(1), "cores": m.group(2),
                              "mem_gb": m.group(3), "gpu": m.group(4).strip()}
            break
    line = ""
    for ln in text.splitlines():
        if "[UTIL-SUMMARY]" in ln:
            line = ln                                # last one wins
    v = _VERDICT.search(line)
    out["bound"] = _BOUND_MAP.get(v.group(1)) if v else None
    cpu = _SUMMARY_CPU.search(line)
    if cpu:
        out["stated_cpu_mean_pct"] = float(cpu.group(1))
    gpu = [float(x) for x in _SUMMARY_GPU.findall(line)]
    if gpu:
        # MAX across devices -- the same rule the csv path uses, or the two
        # sources would answer differently on a multi-GPU node.
        out["stated_gpu_sm_mean_pct"] = max(gpu)
    return out


class MonitorLogFileParser(FileParser):
    """The wrapper's utilisation monitor log."""

    name   = "monitor-log"
    label  = "wrapper monitor log (.monitor.log)"
    hint   = "files ending in .monitor.log"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".monitor.log") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=monitor_metrics(text),
                                       parser_name=cls.name, source=path)
