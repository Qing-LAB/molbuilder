"""The monitor's two files -- the lines of ``<base>.monitor.log`` a reader
reads, and the columns of ``<base>.util.csv`` -- their words, one spelling.

`monitor.py` writes them, and the patterns below are built on the same text:
`parse/instruments/monitor.py` and `util_csv.py` read with them,
`parse/dirs/job.py` reads the closing record, and `parse/instruments/utilisation.py`
and `jobset/memory.py` say where a peak was read in its words, so a word reworded
is reworded for its reader too (`model/parse.md` § 5c; `wrapper_log` is the same
for the session log).

**Stdlib only, and it travels beside every job** (`runwrap.
monitor_companions`), since the monitor writes with it there.
"""
from __future__ import annotations

import re

#: ``[MACHINE] node=… cores=… mem_gb=… gpu=…`` -- what kind of node was under
#: the run (`scheduler.md` R12).  ``gpu`` LAST because device models contain
#: spaces: a reader takes the rest of the line as the model list.
MACHINE_TAG = "[MACHINE]"
MACHINE_FIELDS = ("node", "cores", "mem_gb", "gpu")
MACHINE_LINE = re.compile(
    re.escape(MACHINE_TAG) + r"\s+"
    + r"\s+".join(f"{k}=(\\S+)" for k in MACHINE_FIELDS[:-1])
    + rf"\s+{MACHINE_FIELDS[-1]}=(.+)$")

#: ``[NOTIFY] (stub) <event>: <summary>`` -- what the log's notifier records of
#: an event, whether or not a channel is set up (`run-reports.md` § 2.5).
NOTIFY_STUB_TAG = "[NOTIFY] (stub) "
NOTIFY_STUB = NOTIFY_STUB_TAG + "{}: {}"
NOTIFY_STUB_LINE = re.compile(re.escape(NOTIFY_STUB_TAG) + r"(\w+): (.*)$")

#: ``[UTIL-SUMMARY] cpu mean=NN% (mm-MM); gpu0 sm mean=NN% (mm-MM) -> verdict``
UTIL_SUMMARY_TAG = "[UTIL-SUMMARY]"
CPU_MEAN = "cpu mean="
GPU_MEAN = " sm mean="
_PCT = r"(\d+(?:\.\d+)?)%"
SUMMARY_CPU = re.compile(re.escape(CPU_MEAN) + _PCT)
SUMMARY_GPU = re.compile(r"gpu\d+" + re.escape(GPU_MEAN) + _PCT)
#: Where the summary's verdict comes from and what it says: each row the word
#: the line says it in, the key a reader states it by (``bound``), and why.
GPU_BOUND = ("GPU-bound", "gpu", "GPU saturated")
HOST_BOUND = ("host/CPU-bound", "host", "GPU starved")
MIXED_BOUND = ("mixed", "mixed", "GPU not saturated")
_VERDICTS = (GPU_BOUND, HOST_BOUND, MIXED_BOUND)
BOUND_OF = {word: bound for word, bound, _why in _VERDICTS}
VERDICT = re.compile(r"->\s*(" + "|".join(re.escape(row[0]) for row in _VERDICTS)
                     + ")")


def verdict_said(row) -> str:
    """A verdict's row as the summary line ends with it."""
    return f" -> {row[0]} ({row[2]})"


#: Where the job's memory peak was read -- the kernel's own counter, stated
#: on the closing ``[UTIL-BASIS]`` line, or the largest sample of `util.csv`
#: -- the words `utilisation` states it in (``mem_peak_from``).
PEAK_KERNEL = "kernel counter"
PEAK_SAMPLED = "largest sample"

#: ``[UTIL-BASIS] …; mem [<source>]; peak <X> GB (kernel counter); limit <Y>
#: GB`` -- what the memory figures are fractions of, the kernel's own peak
#: where it keeps one, and the limit (`model/parse.md` § 5c.1).  ``{}`` is
#: the value.
UTIL_BASIS_TAG = "[UTIL-BASIS]"
BASIS_MEM = "mem [{}]"
BASIS_PEAK = f"peak {{}} GB ({PEAK_KERNEL})"
BASIS_LIMIT = "limit {} GB"


def _with_value(template: str, group: str) -> "re.Pattern":
    return re.compile(re.escape(template).replace(r"\{\}", group))


BASIS_MEM_READ = _with_value(BASIS_MEM, r"([^\]]+)")
BASIS_PEAK_READ = _with_value(BASIS_PEAK, r"([0-9.]+)")
BASIS_LIMIT_READ = _with_value(BASIS_LIMIT, r"([0-9.]+)")

#: The closing record of a run whose process has gone
#: (`execution/run-reports.md` § 2.5) -- ``[<time>] [MONITOR] job ended (<why>);
#: ...``: its time, the run's end, and its words.
MONITOR_ENDED = "[MONITOR] job ended"
MONITOR_ENDED_LINE = re.compile(r"^\[([^\]]+)\]\s+" + re.escape(MONITOR_ENDED)
                                + r"\s*\(([^)]*)\)")


def ended_record(why: str) -> str:
    """The closing record's text, after its time: ``why`` the run's process
    went, in the record's brackets."""
    return f"{MONITOR_ENDED} ({why}); final notify + exit"


#: ``util.csv``'s columns; ``{}`` is the GPU's index.
CSV_EPOCH = "epoch"
CSV_ISO = "iso"
CSV_CPU = "cpu_pct"
CSV_MEM = "mem_gb"
CSV_GPU_SM = "gpu{}_sm"
CSV_GPU_MEMUTIL = "gpu{}_memutil"
CSV_GPU_VRAM = "gpu{}_vram_gb"
CSV_GPU_SM_READ = _with_value(CSV_GPU_SM, r"\d+")
CSV_GPU_VRAM_READ = _with_value(CSV_GPU_VRAM, r"\d+")
