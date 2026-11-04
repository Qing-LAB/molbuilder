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
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from molbuilder import monitor_log as _ML
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import InstrumentResult
from molbuilder.runfiles import ROLE_MONITOR_LOG, role_of

from ._helpers import build_instrument_result


def monitor_metrics(text: str) -> Dict[str, Any]:
    """``{machine, bound, stated_cpu_mean_pct, stated_gpu_sm_mean_pct,
    mem_basis, mem_peak_kernel_gb, mem_limit_gb, ended_at, ended_why}``.
    ``ended_at`` is the closing record's time as the log wrote it (ISO 8601
    with its offset), ``ended_why`` its words; both absent for a run whose
    monitor never closed.

    Absent facts are absent: ``machine`` is ``{}`` for a log predating the
    line, ``bound`` is ``None`` without a summary, and the stated means
    are omitted rather than defaulted -- a caller must be able to tell
    "the monitor did not say" from "the monitor said zero".
    """
    text = text or ""
    out: Dict[str, Any] = {"machine": {}, "bound": None}
    for ln in text.splitlines():
        m = _ML.MACHINE_LINE.search(ln)
        if m:
            out["machine"] = dict(zip(_ML.MACHINE_FIELDS,
                                      (g.strip() for g in m.groups())))
            break
    line = ""
    for ln in text.splitlines():
        m = _ML.MONITOR_ENDED_LINE.match(ln)
        if m:
            out["ended_at"], out["ended_why"] = m.group(1), m.group(2)
    for ln in text.splitlines():
        if _ML.UTIL_SUMMARY_TAG in ln:
            line = ln                                # last one wins
    v = _ML.VERDICT.search(line)
    out["bound"] = _ML.BOUND_OF.get(v.group(1)) if v else None
    cpu = _ML.SUMMARY_CPU.search(line)
    if cpu:
        out["stated_cpu_mean_pct"] = float(cpu.group(1))
    gpu = [float(x) for x in _ML.SUMMARY_GPU.findall(line)]
    if gpu:
        # MAX across devices -- the same rule the csv path uses, or the two
        # sources would answer differently on a multi-GPU node.
        out["stated_gpu_sm_mean_pct"] = max(gpu)
    basis = ""
    for ln in text.splitlines():
        if _ML.UTIL_BASIS_TAG in ln:
            basis = ln                               # last one wins
    for key, pat, cast in (("mem_basis", _ML.BASIS_MEM_READ, str),
                           ("mem_peak_kernel_gb", _ML.BASIS_PEAK_READ, float),
                           ("mem_limit_gb", _ML.BASIS_LIMIT_READ, float)):
        m = pat.search(basis)
        if m:
            out[key] = cast(m.group(1))
    return out


class MonitorLogFileParser(FileParser):
    """The wrapper's utilisation monitor log."""

    name   = "monitor-log"
    label  = f"wrapper monitor log ({ROLE_MONITOR_LOG})"
    hint   = f"files ending in {ROLE_MONITOR_LOG}"
    output = InstrumentResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return role_of(path.name) == ROLE_MONITOR_LOG and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> InstrumentResult:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return build_instrument_result(metrics=monitor_metrics(text),
                                       parser_name=cls.name, source=path)
