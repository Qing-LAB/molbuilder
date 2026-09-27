"""The wrapper's session log -- ``<base>.runwrap-<stamp>.log`` -- its lines,
and ONE reader of them.

`runwrap` writes the log: its own lines -- the section banner, the host and
environment, the run index, the launch -- and, tee'd in, everything the
engine said on stdout and stderr.  This module is the one spelling of the
wrapper's lines, which the writer renders and every reader reads with: the
run record (`parse/dirs/record.py`), the bench's trial reader
(`bench/result.py`), and `run_status` (`parse/dirs/job.py`), which reads a
SIESTA run's stderr here (`model/parse.md` § 2b).

**Stdlib only, and it travels beside every job** (`runwrap.
MONITOR_COMPANIONS`), so the monitor pairs a run with its log exactly as the
Results tab does.  Split out of `runwrap.py` on 2026-09-27 for that reason,
as the SIESTA family's grammar was split from its parser.

ONE SECTION PER RUN.  A warm retry re-execs the wrapper with its output
still going to the first log, and opens a log of its own -- measured on the
2026-09-25 device: its first log holds run 0 and then run 1, its second run
1 alone -- so a log is read as sections, each opening with the start banner,
and the run a log is FOR is its first section's (:func:`log_of_run`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                        # inside molbuilder
    from . import runfiles as _rf
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf

#: The line every section opens with (``_log STAGE`` in ``env_activation``).
WRAPPER_LOG_START = "===== molbuilder wrapper start ====="
#: ``(key, pattern)`` for the header lines -- the ``_log INFO`` lines of
#: ``env_activation``, the run-index line, the banner's program lines.
_WRAPPER_LOG_LINES = (
    ("hostname",       re.compile(r"\] hostname:\s+(\S+)")),
    ("user",           re.compile(r"\] user:\s+(\S+)")),
    ("cwd",            re.compile(r"\] cwd:\s+(.+?)\s*$")),
    ("conda_env",      re.compile(r"\] CONDA_DEFAULT_ENV=(\S+)")),
    ("python",         re.compile(r"\] which python:\s+(\S+)")),
    ("binary",         re.compile(r"^\s+(?:SIESTA|TBtrans) binary\s*:\s*(\S+)")),
    ("engine_version", re.compile(r"^\s+(?:SIESTA|TBtrans) version\s*:\s*(\S+)")),
)
#: ``[molbuilder] run index: <N>  ->  <out>`` -- which run the section is,
#: written by the wrapper's run-index resolver (`runwrap`).
RUN_INDEX_LINE = "[molbuilder] run index:"
_WRAP_RUN_INDEX = re.compile("^" + re.escape(RUN_INDEX_LINE) + r"\s*(\d+)")
#: ``molbuilder: detected phys_cores=48, n_sockets=2, cores_per_socket=24`` --
#: the NODE's physical cores (`lscpu -p=Core,Socket` ignores the affinity mask,
#: verified), not the allocation's.
_WRAP_NODE = re.compile(
    r"detected\s+phys_cores=(\d+),\s*n_sockets=(\d+),\s*"
    r"cores_per_socket=(\d+)")
#: ``ranks / omp : <N> ranks x <M> OMP threads`` -- written BEFORE the launch,
#: so its ranks are what was ASKED; the thread count is written nowhere else.
_WRAP_RANKS_OMP = re.compile(
    r"ranks\s*/\s*omp\s*:\s*(\d+)\s+ranks\s+x\s+(\d+)\s+OMP threads")
#: ``benchmark: <Program> wall <s>s`` -- the engine's own wall, launch to exit.
_WRAP_WALL = re.compile(r"benchmark:\s+\S+\s+wall\s+([0-9.]+)s")


def read_wrapper_log(text: str) -> List[Dict[str, Any]]:
    """One dict per run section of a wrapper log, in order -- ``run_index``,
    the host lines, ``binary`` / ``engine_version``, ``node_phys_cores`` (and
    its sockets), ``ranks_asked`` / ``threads``, ``engine_elapsed_s``.  A key
    is absent when its section does not state it."""
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for line in (text or "").splitlines():
        if WRAPPER_LOG_START in line:
            cur = {}
            sections.append(cur)
            continue
        if cur is None:            # before any banner: a log from before it
            cur = {}
            sections.append(cur)
        m = _WRAP_RUN_INDEX.match(line)
        if m:
            cur.setdefault("run_index", int(m.group(1)))
            continue
        m = _WRAP_NODE.search(line)
        if m:
            for key, g in (("node_phys_cores", 1), ("node_sockets", 2),
                           ("node_cores_per_socket", 3)):
                cur.setdefault(key, int(m.group(g)))
            continue
        m = _WRAP_RANKS_OMP.search(line)
        if m:
            cur.setdefault("ranks_asked", int(m.group(1)))
            cur.setdefault("threads", int(m.group(2)))
            continue
        m = _WRAP_WALL.search(line)
        if m:
            cur.setdefault("engine_elapsed_s", float(m.group(1)))
            continue
        for key, pat in _WRAPPER_LOG_LINES:
            m = pat.search(line)
            if m:
                cur.setdefault(key, m.group(1))
                break
    # A section with nothing in it -- the `launched-by:` line the launcher
    # writes ahead of the first banner -- is no run.
    return [sec for sec in sections if sec]



def first_run_index(path) -> Optional[int]:
    """The run a session log is FOR -- its first section's run index -- read
    from the head down to that line; ``None`` when it states none (a launch
    refused before the run index was resolved)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _WRAP_RUN_INDEX.match(line)
                if m:
                    return int(m.group(1))
    except OSError:
        return None
    return None


def logs_by_run(directory, label: str
                ) -> Dict[Tuple[Optional[str], int], Path]:
    """Every session log of ``label`` here, by the run it is FOR --
    ``(stage, run) -> path``, the run being the log's FIRST section's (a
    retry's section in the first log is not its log -- see the module's
    note), the newest log where several are that run's.  The stage is the
    name's: the log and the run's output are named from the same deck, so
    they carry the same token, and in a flat directory every stage's logs
    sit side by side."""
    out: Dict[Tuple[Optional[str], int], Path] = {}
    # `find` orders a counterless name by name, and a stamp's order is the
    # clock's, so the newest of a run's logs is the one kept.
    for path, got in _rf.find(directory, label, role=".runwrap-{stamp}.log"):
        run = first_run_index(path)
        if run is not None:
            out[(got.stage, run)] = path
    return out


def log_of_run(directory, label: str, run: int,
               stage: Optional[str] = None) -> Optional[Path]:
    """The session log of the run ``run`` of ``label`` at ``stage`` (``None``
    matching only ``None``), or ``None`` when no log is that run's
    (:func:`logs_by_run`)."""
    return logs_by_run(directory, label).get((stage, run))
