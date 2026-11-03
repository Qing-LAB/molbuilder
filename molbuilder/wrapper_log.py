"""The wrapper's session log -- ``<base>.runwrap.log`` -- its lines, and ONE
reader of them.

`runwrap` writes the log: its own lines -- the start's banner, the host and
environment, the launch -- and, tee'd in, everything the
engine said on stdout and stderr; the monitor adds its start, *starting* and
then *started* or the error that stopped it (`run-reports.md` § 2.6).  This module is the one spelling of the
wrapper's lines, which the writer renders and every reader reads with: the
run record (`parse/dirs/record.py`), the bench's trial reader
(`bench/result.py`), and `run_status` (`parse/dirs/job.py`), which reads a
SIESTA run's stderr here (`model/parse.md` § 2b) and whether the job's
finish began (`running-a-job.md` § 4.2).

**Stdlib only, and it travels beside every job** (`runwrap.
monitor_companions`), so the monitor pairs a run with its log exactly as the
Results tab does.

ONE LOG PER RUN.  A run folder holds one start (`project-layout.md`
§ 1.6.1), and the start opens the log once the launch-door gate lets it
through, so the log's being there IS the run's start (:func:`log_of_run`);
what starts nothing -- a help request, a dry run, a refused start -- writes
none (`job-contracts.md` § 2.6).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

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

#: The line the run's start opens with (``_log STAGE`` in ``env_activation``).
WRAPPER_LOG_START = "===== molbuilder run script start ====="
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
#: The run-script banner's program lines -- ``<Program> binary  : <path>``,
#: the program named in its engine record's words (`engine_table`,
#: ``programs``), whichever it is.
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
    ("binary",         re.compile(r"^\s+\S+ " + re.escape(BINARY_LABEL)
                                  + r"\s*:\s*(\S+)")),
)
_WRAP_NODE = _pattern(NODE_LINE, phys=r"(\d+)", sockets=r"(\d+)",
                      cps=r"(\d+)")
_WRAP_RANKS_OMP = _pattern(RANKS_OMP_LINE, ranks=r"(\d+)",
                           threads=r"(\d+)")
_WRAP_WALL = _pattern(WALL_LINE, program=r"\S+", seconds=r"([0-9.]+)")


def read_wrapper_log(text: str) -> Dict[str, Any]:
    """The run's start, as its session log states it -- the host lines,
    ``binary``, ``node_phys_cores`` (and its sockets), ``ranks_asked`` /
    ``threads``, ``engine_elapsed_s``.  A key is absent when the log does
    not state it; ``{}`` for a log that states none of them.  The log holds
    one start (the module's note), so its lines are read as one."""
    start: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        m = _WRAP_NODE.search(line)
        if m:
            for key, g in (("node_phys_cores", 1), ("node_sockets", 2),
                           ("node_cores_per_socket", 3)):
                start.setdefault(key, int(m.group(g)))
            continue
        m = _WRAP_RANKS_OMP.search(line)
        if m:
            start.setdefault("ranks_asked", int(m.group(1)))
            start.setdefault("threads", int(m.group(2)))
            continue
        m = _WRAP_WALL.search(line)
        if m:
            start.setdefault("engine_elapsed_s", float(m.group(1)))
            continue
        for key, pat in _WRAPPER_LOG_LINES:
            m = pat.search(line)
            if m:
                start.setdefault(key, m.group(1))
                break
    return start


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


def log_of_run(directory, label: str, stage: Optional[str]
               ) -> Optional[Path]:
    """The session log of ``label``'s run at ``stage`` in ``directory`` --
    every run's files carry its stage; a name with none asks for a log with
    none -- or ``None`` where the folder holds no start (the module's
    note)."""
    found = [p for p, rec in _rf.find(directory, label,
                                      role=_rf.ROLE_SESSION_LOG)
             if rec.stage == stage]
    return found[0] if found else None
