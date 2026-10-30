"""The wrapper's session log -- ``<base>.runwrap-<stamp>.log`` -- its lines,
and ONE reader of them.

`runwrap` writes the log: its own lines -- the section banner, the host and
environment, the try's number, the launch -- and, tee'd in, everything the
engine said on stdout and stderr; the monitor adds its start, *starting* and
then *started* or the error that stopped it (`run-reports.md` § 2.6).  This module is the one spelling of the
wrapper's lines, which the writer renders and every reader reads with: the
run record (`parse/dirs/record.py`), the bench's trial reader
(`bench/result.py`), and `run_status` (`parse/dirs/job.py`), which reads a
SIESTA run's stderr here (`model/parse.md` § 2b) and whether the job's
finish began (`running-a-job.md` § 4.2).

**Stdlib only, and it travels beside every job** (`runwrap.
MONITOR_COMPANIONS`), so the monitor pairs a run with its log exactly as the
Results tab does.

ONE LOG PER START.  Each start of the wrapper opens a log of its own, and a
warm retry hands back the output it was started with before it starts again
(`running-a-job.md` § 3.5), so a log holds its start alone.  A log is read as
sections, each opening with the start banner, and the try a log is FOR is its
first section's (:func:`log_of_run`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                        # inside molbuilder
    from . import runfiles as _rf
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf

#: THE WRAPPER'S OWN LINE -- ``[clock] [LEVEL] message`` -- as `runwrap`'s ``_log``
#: writes it in bash (``printf`` and ``date``) and the monitor bundle's entry
#: writes it in Python (``%`` and ``time.strftime``): the two read the same
#: format and the same clock, so one spelling renders both, and the readers
#: below anchor on the ``] `` it leaves before the message.
LOG_LINE = "[%s] [%-5s] %s"
LOG_CLOCK = "%H:%M:%S%z"

#: The line every section opens with (``_log STAGE`` in ``env_activation``).
WRAPPER_LOG_START = "===== molbuilder wrapper start ====="
#: The message logged as a point's after-step begins -- the kind's own
#: step after the engine, which leaves the result the engine alone does not
#: (`monitor._point_main`, `execution/runs-of-points.md` § 3, op. 10;
#: `engines/vibration.md` § 5.5).  `run_status` reads it: an
#: output that ended, beside this line and no conclusion yet, is a job still
#: deriving its result, not a finished one (`running-a-job.md` § 4.2).
FINISH_STARTED = "finish started:"
#: THE WRAPPER'S LABELLED LINES -- each fact's one spelling: `runwrap`
#: writes the label (or fills the template) and the patterns below are built
#: on the same text, so a label reworded is reworded for its reader too.
HOSTNAME_LABEL = "hostname:"
USER_LABEL = "user:"
CWD_LABEL = "cwd:"
CONDA_ENV_LABEL = "CONDA_DEFAULT_ENV="
PYTHON_LABEL = "job python:"
#: The run-script banner's program lines -- ``<Program> binary  : <path>``.
BINARY_LABEL = "binary"
VERSION_LABEL = "version"
#: ``molbuilder: detected phys_cores=48, n_sockets=2, cores_per_socket=24`` --
#: the NODE's physical cores (`lscpu -p=Core,Socket` ignores the affinity mask,
#: verified), not the allocation's.
NODE_LINE = ("molbuilder: detected phys_cores={phys}, n_sockets={sockets}, "
             "cores_per_socket={cps}")
#: Written BEFORE the launch, so its ranks are what was ASKED; the thread
#: count is written nowhere else.
RANKS_OMP_LINE = "ranks / omp     : {ranks} ranks x {threads} OMP threads"
#: The engine's own wall, launch to exit.
WALL_LINE = "benchmark: {program} wall {seconds}s"


def _pattern(template: str, **groups: str) -> "re.Pattern":
    """``template``'s pattern: its text as written, each ``{name}`` the
    group ``groups[name]``."""
    out = re.escape(template)
    for name, group in groups.items():
        out = out.replace(re.escape("{" + name + "}"), group)
    return re.compile(out)


#: ``(key, pattern)`` for the header lines -- the ``_log INFO`` lines of
#: ``env_activation``, the banner's program lines.
_WRAPPER_LOG_LINES = (
    ("hostname",       re.compile(r"\] " + re.escape(HOSTNAME_LABEL)
                                  + r"\s+(\S+)")),
    ("user",           re.compile(r"\] " + re.escape(USER_LABEL)
                                  + r"\s+(\S+)")),
    ("cwd",            re.compile(r"\] " + re.escape(CWD_LABEL)
                                  + r"\s+(.+?)\s*$")),
    ("conda_env",      re.compile(r"\] " + re.escape(CONDA_ENV_LABEL)
                                  + r"(\S+)")),
    ("python",         re.compile(r"\] " + re.escape(PYTHON_LABEL)
                                  + r"\s+(\S+)")),
    ("binary",         re.compile(r"^\s+(?:SIESTA|TBtrans) "
                                  + re.escape(BINARY_LABEL)
                                  + r"\s*:\s*(\S+)")),
    ("engine_version", re.compile(r"^\s+(?:SIESTA|TBtrans) "
                                  + re.escape(VERSION_LABEL)
                                  + r"\s*:\s*(\S+)")),
)
#: ``[molbuilder] run index: <N>  ->  <out>`` -- which try the section is,
#: written by the run script (`runwrap`).
RUN_INDEX_LINE = "[molbuilder] run index:"
_WRAP_RUN_INDEX = re.compile("^" + re.escape(RUN_INDEX_LINE) + r"\s*(\d+)")
_WRAP_NODE = _pattern(NODE_LINE, phys=r"(\d+)", sockets=r"(\d+)",
                      cps=r"(\d+)")
_WRAP_RANKS_OMP = _pattern(RANKS_OMP_LINE, ranks=r"(\d+)",
                           threads=r"(\d+)")
_WRAP_WALL = _pattern(WALL_LINE, program=r"\S+", seconds=r"([0-9.]+)")


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
        if cur is None:
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



def finish_started(path) -> bool:
    """Whether the session log at ``path`` records the job's finish
    beginning (:data:`FINISH_STARTED`, the message after ``_log``'s level
    bracket) -- ``False`` for an unreadable log."""
    anchor = "] " + FINISH_STARTED
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return any(anchor in line for line in fh)
    except OSError:
        return False


def first_run_index(path) -> Optional[int]:
    """The try a session log is FOR -- its first section's run index -- read
    from the head down to that line; ``None`` when it states none (a start
    refused before the try's number was checked)."""
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
    """Every session log of ``label`` here, by the try it is FOR --
    ``(stage, run) -> path``, ``run`` the log's first section's try (the
    module's note), the newest log where several are that try's.  The stage
    is the name's: the log and the try's output are named from the same
    deck, so they carry the same token."""
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
    """The session log of try ``run`` of ``label`` at ``stage`` (``None``
    matching only ``None``), or ``None`` when no log is that try's
    (:func:`logs_by_run`)."""
    return logs_by_run(directory, label).get((stage, run))
