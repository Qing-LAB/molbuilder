"""`run_record` — one run's record: what ran, with what, and how it went.

Contract: `model/parse.md` § 5d — and § 5d.1b for the shape of this module:
**a declared table, one reader per file.**  :func:`contributors` names, for
each file a run can leave, the record fields it answers and the reader that
answers them -- a run output's from its catalogue row, the reader stating its
fields; the composer walks the table and merges what each row returns.
Engines differ only in which files exist, so a run of any kind gets every
fact its files state, and a new engine or a new file is a row, not a path
through this module.  No row reads a format itself: each asks the reader that
lives with that format (§ 1a) — the SIESTA family's table, `runwrap`'s reader
of its own log, the instruments, `materialize` and `prep` for their files,
`deck_record` for its block, the registry for SIESTA's `fdf` log.

**The table is configuration, not reconciliation** (§ 5d.1b).  A run's files
are one calculation reporting itself, one aspect per file -- the `.out` the
ranks the engine ran on, the wrapper's log the ranks it asked for -- so a row
says where a fact is READ; no two files are weighed against each other.

**One run's record only.**  A ladder, a benchmark and a transport calculation
are other reports, built by their own modules (`model/parse.md`, "How results
are organized"); none is composed from this one.

**Cheap reads only** (§ 5d.1): this runs on every folder scan, so it never
builds a trajectory.  The Results tab's Run panel shows it
(`web/results.md` § 3a).

**Which run** (§ 5d.1): the one its caller names (:func:`run_files`) -- its
one set of files, a run folder holding one start; the `fdf` log paired with
the ``.out`` by its stamp; the run's session log.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from ..errors import ParseError

#: How much of a ``.out`` the launch, build, solver and pseudopotential lines
#: can sit in -- the solver is printed after the basis report, ~49 KB into a
#: 42-atom run, and further for a bigger system -- the record's and the bench
#: summary's one window.
SETUP_HEAD = 512 * 1024
#: The tail, for ``>> End of run``.
_TAIL = 16 * 1024


# --------------------------------------------------------------------------- #
#  Which files are the run's                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunFiles:
    """The files of the run a record describes -- :func:`run_files`' answer.
    A file the run does not hold is ``None``."""
    directory: Path
    engine: str
    #: The folder the run's launch record lies in, the run door's
    #: (`runs.Run.launched_in`): the run's own ``run-<n>`` folder -- for a
    #: point of a run of points, the one above it.
    launched_in: Path
    status: Any = None                     # the directory's `RunStatus`
    deck: Optional[Path] = None
    #: The names of the run's stage's files (`runfiles.RunNames`), the
    #: caller's -- the run door's.
    names: Any = None
    #: THE ENGINE'S OUTPUT -- its stdout, the role its record's engine
    #: writes it in (`runfiles.stdout_roles`).
    stdout: Optional[Path] = None
    #: THE RUN'S FILES BY ROLE -- the first by name, the run door's rule
    #: (`runs.Run.file`): what a run output's own reader asks for the file
    #: of its role (:meth:`of`), never a slot this floor names per engine.
    files: Dict[str, Path] = field(default_factory=dict)
    #: THE ENGINE'S LOG OF THE SETTINGS IT READ, where it writes one --
    #: SIESTA's ``fdf.<stamp>.log``, paired by its output's start: found by
    #: the reader its output's catalogue row names (``used_log``).
    used_log: Optional[Path] = None
    #: The run's start, as its session log states it
    #: (`wrapper_log.read_wrapper_log`), and the log's text, for the fence.
    wrapper_start: Dict[str, Any] = field(default_factory=dict)
    wrapper_text: str = ""
    #: THE RUN'S TIMING FILE -- its run script's tee, else a PySCF progress
    #: log (`scf_timing_rows.timing_file`, the one choice).
    timing: Optional[Path] = None
    monitor_log: Optional[Path] = None
    util_csv: Optional[Path] = None
    #: The output's head, read once here for every row that reads it.
    stdout_head: str = ""
    #: The calculation's kind, from its description -- the caller's
    #: (`runs.run_of`), never read here: this floor knows no calculation.
    calculation: Optional[str] = None
    #: Whether the run's deck is still its stage's -- the run door's one
    #: answer (`runs.Run.current`); ``None`` with nothing to compare.
    current: Optional[bool] = None

    @property
    def over(self) -> bool:
        """Whether the run's state is final (`job.FINAL_STATES`) -- an
        ``unreadable`` run's is not: its state cannot be told."""
        from .job import FINAL_STATES
        return self.status is not None and self.status.state in FINAL_STATES

    def of(self, role: str) -> Optional[Path]:
        """The run's file in ``role`` (:attr:`files`), or ``None``."""
        return self.files.get(role)

    def head(self, path: Optional[Path]) -> str:
        """The head of one of the run's files, as far as its launch, build,
        solver and pseudopotential lines can sit (:data:`SETUP_HEAD`)."""
        return read_part(path, head=SETUP_HEAD)

    def tail(self, path: Optional[Path]) -> str:
        """The tail of one of the run's files, for its end lines
        (:data:`_TAIL`)."""
        return read_part(path, tail=_TAIL)


def read_part(path: Optional[Path], *, head: Optional[int] = None,
              tail: Optional[int] = None) -> str:
    """Text of ``path`` -- all of it, or its head, or its tail -- or ``""``."""
    if path is None:
        return ""
    try:
        with open(path, "rb") as fh:
            if tail is not None:
                size = path.stat().st_size
                if size > tail:
                    fh.seek(-tail, 2)
            data = fh.read(head) if head is not None else fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_files(directory, *, names, launched_in, status, engine: str,
              calculation: Optional[str] = None,
              current: Optional[bool] = None) -> Optional[RunFiles]:
    """What the run of the stage ``names`` names (`runfiles.RunNames`) left
    in ``directory``, found through the doors that own each name -- or
    ``None`` when no deck of it is here.

    **The caller says which run** -- the run door (`runs.run_of`,
    `execution/architecture.md` § 3.2), which reads the label from the
    description and the stage from where the folder is.  This floor knows no
    calculation (§ 2.1).

    ``status`` and ``engine`` are the run's `run_status` and engine, which
    the run door has already read.
    ``launched_in`` is where the run's launch record lies, the run door's
    (`runs.Run.launched_in`).
    """
    from ... import runfiles as _rf
    from ...runfiles import deck_roles, find, role_of, stdout_roles
    from ..instruments.scf_timing_rows import timing_file

    d = Path(directory)
    label, stage = names.label, names.stage
    # ONE LISTING of this run's files, filtered below -- `find` lists and
    # reads every name in the directory on each call.
    listing = find(d, label, stage=stage)
    decks = [p for p, rf in listing if rf.role in deck_roles(engine)]
    if not decks:
        return None
    deck = decks[0]

    # THE RUN'S FILES BY ROLE -- the first by name, the run door's rule
    # (`runs.Run.file`).
    files: Dict[str, Path] = {}
    for p, rf in listing:
        files.setdefault(rf.role, p)

    def this_run(role: str) -> Optional[Path]:
        return files.get(role)

    def first_of(roles) -> Optional[Path]:
        return next((p for p, rf in listing if rf.role in roles), None)

    out = first_of(stdout_roles(engine))
    out_head = read_part(out, head=SETUP_HEAD)
    # THE ENGINE'S LOG OF WHAT IT READ, by the reader its output's row names.
    find_used = (_rf.reader(role_of(out.name), "used_log")
                 if out is not None else None)
    used_log = find_used(d, out_head) if find_used is not None else None

    # THE RUN'S SESSION LOG -- its start, the log `run_status` reads
    # SIESTA's stderr in (`wrapper_log.log_of_run`) -- from the one
    # listing, which holds the stage's files alone, as that door matches.
    from ...wrapper_log import read_wrapper_log
    log_text = read_part(this_run(_rf.ROLE_SESSION_LOG))

    return RunFiles(
        directory=d, engine=engine, status=status,
        deck=deck, names=names, stdout=out, stdout_head=out_head,
        files=files, used_log=used_log,
        wrapper_start=read_wrapper_log(log_text), wrapper_text=log_text,
        timing=timing_file(files),
        monitor_log=this_run(_rf.ROLE_MONITOR_LOG),
        util_csv=this_run(_rf.ROLE_UTIL_CSV),
        calculation=calculation, current=current,
        launched_in=Path(launched_in))


# --------------------------------------------------------------------------- #
#  The contributors -- one row per file, each asking the reader that owns it  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Contributor:
    """One row of the record (:func:`contributors`): the record fields it
    answers, the file it reads by its role (``None``: a file every run has),
    and the read.  The engines whose runs have its file are that role's
    (:attr:`engines`) -- never typed."""
    name: str
    fields: Tuple[str, ...]
    role: Optional[str]
    read: Callable[[RunFiles], Dict[str, Any]]

    @property
    def engines(self) -> Tuple[str, ...]:
        """The engines whose runs have this row's file -- its role's engine
        (`runfiles.engine_of_role`) -- or ``()``: every run's."""
        from ...runfiles import engine_of_role
        engine = engine_of_role(self.role) if self.role else None
        return (engine,) if engine else ()


def _ending(f: RunFiles) -> Dict[str, Any]:
    """How the run is doing and whether each of its phases converged -- from
    the ONE scan `run_status` made to judge the state
    (``RunStatus.endings``).  Nothing is read a second time."""
    from ..engines._run_ending import CONCLUDED
    st = f.status
    if st is None:
        return {}
    verdict: Dict[str, Any] = {"state": st.state, "detail": st.detail}
    endings = getattr(st, "endings", {}) or {}
    # THE OUTPUT THAT SPEAKS is the status's own choice (`run_status`'s
    # `active_source`, § 5.1), never a second rule here.
    source = getattr(st, "active_source", None)
    end = endings.get(source) if source else None
    if end is not None:
        # HOW IT ENDED, once its output states an ending (§ 2b): an output
        # with none yet is running, not ended, and a field is stated only
        # where a file states it (§ 5d.1a).
        if end.run_state in CONCLUDED:
            verdict["ended"] = end.run_state
        # PER PHASE (§ 5d.6): a device's periodic initialization converging
        # does not speak for its NEGF loop, so there is no one flag to give.
        if end.phases:
            verdict["converged"] = dict(end.phases)
        verdict.update(_forces_of(end))
    return {"verdict": verdict}


def _forces_of(end) -> Dict[str, Any]:
    """What the run's last step stated of its forces and stress, as the
    verdict says it (`model/parse.md` § 5d.6): the free atoms' largest
    force; every atom's, only when it is the larger -- a held atom carries
    it; the stress tensor raw -- or, for a run that computes none, the
    sentence in their place.  Each stated only when its output states it."""
    if end.forces_not_computed:
        return {"forces_not_computed": end.forces_not_computed}
    out: Dict[str, Any] = {}
    if end.max_force is not None:
        out["max_force_free_ev_ang"] = end.max_force
        if end.max_force_all is not None and end.max_force_all > end.max_force:
            out["max_force_held_ev_ang"] = end.max_force_all
    if end.stress_voigt_kbar:
        out["stress_voigt_kbar"] = list(end.stress_voigt_kbar)
    return out


def _wrapper(f: RunFiles) -> Dict[str, Any]:
    """The run's start, as its session log states it, through the log's
    one reader (`wrapper_log.read_wrapper_log`)."""
    sec = f.wrapper_start
    if not sec:
        return {}
    host = {k: sec[k] for k in ("hostname", "user", "cwd", "conda_env",
                                "python", "node_phys_cores", "node_sockets",
                                "node_cores_per_socket") if k in sec}
    launch = {k: sec[k] for k in ("ranks_asked", "threads") if k in sec}
    comp: Dict[str, Any] = {}
    if host:
        comp["host"] = host
    if launch:
        comp["launch"] = launch
    if "binary" in sec:
        comp["engine"] = {"binary": sec["binary"]}
    if "engine_elapsed_s" in sec:
        comp["time"] = {"engine_elapsed_s": sec["engine_elapsed_s"]}
    return {"computation": comp} if comp else {}


def _instruments(f: RunFiles) -> Dict[str, Any]:
    """What the wrapper measured (§ 5c): seconds per iteration by phase, the
    utilisation and memory through the one door, the node."""
    from ..instruments import metrics_of, utilisation
    mon = metrics_of(f.monitor_log)
    util = utilisation(mon, metrics_of(f.util_csv)) if f.util_csv else {}
    comp: Dict[str, Any] = {}
    # THE RUN'S STAMPED SCF ROWS (§ 5c): its timing file, the one choice
    # (`scf_timing_rows.timing_file`), read by the one rule.
    time = _timing_figures(f.timing)
    if time:
        comp["time"] = time
    memory = {k: util[k] for k in ("mem_peak_gb", "mem_peak_from",
                                   "util_basis", "cpu_mean_pct",
                                   "gpu_sm_mean_pct")
              if util.get(k) is not None}
    memory.update({k: mon[k] for k in ("mem_basis", "mem_limit_gb")
                   if k in mon})
    if memory:
        comp["memory"] = memory
    if mon.get("machine"):
        comp["host"] = {"machine": mon["machine"]}
    return {"computation": comp} if comp else {}


def _timing_figures(path: Optional[Path]) -> Dict[str, Any]:
    """The SCF-timing figures (§ 5c) from one timing file -- the SIESTA
    family's tee or a PySCF progress log -- by the one rule
    (`scf_timing_rows.timing_of`): seconds per iteration, the iterations
    timed and, for a run with both phases, each phase's own and its row
    count.  The run record's and the trajectory viewer's
    (:func:`scf_timing_of`), so the page states one number."""
    from ..instruments.scf_timing_rows import timing_of
    return {k: v for k, v in timing_of(path).items()
            if k.startswith(("s_per_iter", "iters_measured", "rows_"))
            and v is not None}


def scf_timing_of(run) -> Dict[str, Any]:
    """The SCF-timing figures of ``run``, the run door's `Run`: its timing
    file, by the one choice (`scf_timing_rows.timing_file`), read as the
    record reads it.  ``{}`` when the run has none."""
    from ..instruments.scf_timing_rows import timing_file, timing_roles
    path = timing_file({r: run.file(r) for r in timing_roles()})
    return _timing_figures(path) if path is not None else {}


def _launch_record(f: RunFiles) -> Dict[str, Any]:
    """``run.json`` -- how the run was launched: its own record
    (`runrecord.launch_record`), read where it lies -- a point's in its
    run's folder above it (``launched_in``), as the run's state reads it
    (`runs.concluded`)."""
    from ...runrecord import launch_record, read_continued_from
    rec = launch_record(f.launched_in) or {}
    # `continued_from`: the run this one continued from, as its launch took it
    # (`job-system.md` § 5.4, plan W37) -- the launch's own hand-over,
    # written with the record.  EVERY KEY AS THE RECORD STATES IT: its
    # nulls are answers -- no job id for a run here, no queue, started from
    # the structure (`job-contracts.md` § 6.1) -- not "could not check".
    if not rec:
        return {}
    # ``sequence``: the sequence of groups whose job sent it (`job-system.md`
    # § 6.0), null for a run its launch sent itself -- a record written
    # before sequences states none, and its launch sent it.
    keys = ["mode", "command", "job_id", "launched_at", "placed_on",
            "sequence"]
    # `run.json` is the run's (`project-layout.md` § 1.6.3): where the run
    # continued from is said of it.  Not of a point, which starts from
    # the point before it: `started_from`, its own `.continued-from`
    # (`runs-of-points.md` § 1.2 *start*) -- stated only where the marker
    # is, since a planned point's start is copied with no marker.
    launch = {k: rec.get(k) for k in keys}
    if f.launched_in.resolve() == f.directory.resolve():
        launch["continued_from"] = rec.get("continued_from")
    else:
        source = read_continued_from(f.directory)
        if source is not None:
            launch["started_from"] = str(source)
    return {"computation": {"launch": launch}}


def _concluded(f: RunFiles) -> Dict[str, Any]:
    """What the run's process said on its way out: the marker `run_status`
    already found for the run (``RunStatus.concluded``), read by its one
    parser, `runrecord.read_concluded` -- never the file a second time."""
    from ...runrecord import read_concluded
    said = getattr(f.status, "concluded", None)
    if not said:
        return {}
    got = read_concluded(said)
    if got is None:
        return {"computation": {"exit": {"said": said.strip()}}}
    return {"computation": {"exit": got}}


def _deck(f: RunFiles) -> Dict[str, Any]:
    """The deck as run: its path, whole (the page opens it as it is,
    `web/results.md` § 0.3), and hash, whether it is still the stage's
    deck -- the run door's answer (`runs.Run.current`) -- and what it was
    gathered from: its run's gather, at ``run-<n>/`` where the run's points
    share it, then what a point took alone (§ 5d.1; `engines/transport.md`
    § 2a.11)."""
    from ...persist import sha256_of
    from ...runrecord import read_gathered_from
    if f.deck is None:
        return {}
    deck: Dict[str, Any] = {"path": str(f.deck), "sha256": sha256_of(f.deck)}
    if f.current is not None:
        deck["current"] = f.current
    gathered = list(read_gathered_from(f.launched_in))
    if f.directory.resolve() != f.launched_in.resolve():
        gathered += [g for g in read_gathered_from(f.directory)
                     if g not in gathered]
    if gathered:
        deck["gathered_from"] = [{"file": fn, "from": str(src)}
                                 for src, fn in gathered]
    return {"deck": deck}


def _setup(f: RunFiles) -> Dict[str, Any]:
    """The three columns (§ 5d.3), from the run's own records: the wrapper's
    or the deck's block for the default, the deck or the block for what was
    asked, SIESTA's fdf log or the block for what was used."""
    from .setup import setup_rows
    return setup_rows(f)


from ...engine_table import engines as _engines  # noqa: E402
from ..instruments.scf_timing_rows import PHASES as _PHASES  # noqa: E402

#: The timing instrument's figures (§ 5c): the headline phase's, and each
#: phase's own under its name.
_TIMING = tuple(f"computation.time.{k}" for k in ("s_per_iter",
                                                  "iters_measured")) + tuple(
    f"computation.time.{k}_{phase}"
    for k in ("s_per_iter", "iters_measured", "rows")
    for phase in _PHASES)

#: THE FILES EVERY RUN HAS (§ 5d.1b).  ``fields`` are the record paths a row
#: answers, and the declaration is the mechanism: a row contributes those
#: fields and no others (:func:`_declared`), and the record refuses two rows
#: answering one field for one engine (:func:`_one_source`) -- § 5c.1's "one
#: source per quantity".
_EVERY_RUN: Tuple[Contributor, ...] = (
    Contributor("ending", ("verdict.ended", "verdict.state",
                           "verdict.detail", "verdict.converged",
                           "verdict.max_force_free_ev_ang",
                           "verdict.max_force_held_ev_ang",
                           "verdict.stress_voigt_kbar",
                           "verdict.forces_not_computed"),
                None, _ending),
    Contributor("wrapper-log",
                ("computation.host.hostname", "computation.host.user",
                 "computation.host.cwd", "computation.host.conda_env",
                 "computation.host.python", "computation.host.node_phys_cores",
                 "computation.host.node_sockets",
                 "computation.host.node_cores_per_socket",
                 "computation.launch.ranks_asked", "computation.launch.threads",
                 "computation.engine.binary",
                 "computation.time.engine_elapsed_s"),
                None, _wrapper),
    Contributor("instruments",
                _TIMING + ("computation.memory", "computation.host.machine"),
                None, _instruments),
    Contributor("run-json",
                ("computation.launch.mode", "computation.launch.command",
                 "computation.launch.job_id", "computation.launch.launched_at",
                 "computation.launch.placed_on",
                 "computation.launch.sequence",
                 "computation.launch.continued_from",
                 "computation.launch.started_from"),
                None, _launch_record),
    Contributor("concluded", ("computation.exit",), None, _concluded),
    Contributor("deck", ("deck",), None, _deck),
    Contributor("setup", ("setup.rows", "setup.engine_only",
                          "verdict.findings"), None, _setup),
)


@functools.lru_cache(maxsize=1)
def contributors() -> Tuple[Contributor, ...]:
    """THE RECORD'S ROWS: a run output's, by the reader its catalogue row
    names for ``record`` -- the reader stating the fields it answers (its
    ``fields``) -- then the files every run has.  A new engine's output is
    its row (`runfiles.WRITTEN`) and nothing here *(the user, 2026-11-03:
    "One column on the catalogue row")*."""
    from ... import runfiles as _rf
    rows = tuple(Contributor(role, read.fields, role, read)
                 for role in _rf.read_roles("record")
                 for read in [_rf.reader(role, "record")]) + _EVERY_RUN
    _one_source(rows)
    return rows


def _one_source(rows: Tuple[Contributor, ...]) -> None:
    """Refuse a table in which two rows answer one field -- or one answers a
    field inside another's -- for an engine both serve."""
    owner: Dict[Tuple[str, str], str] = {}
    for row in rows:
        # A row whose file every run has answers for every engine the
        # engine table holds (`execution/architecture.md` § 3.3).
        for engine in (row.engines or _engines()):
            for field_ in row.fields:
                for (other, eng), name in owner.items():
                    if eng == engine and name != row.name and (
                            other == field_ or other.startswith(field_ + ".")
                            or field_.startswith(other + ".")):
                        raise ValueError(
                            f"run record: {field_!r} ({row.name}) and "
                            f"{other!r} ({name}) answer one field for "
                            f"{engine} -- one source per quantity (§ 5d.1b)")
                owner[(field_, engine)] = row.name


def _declared(part: Dict[str, Any], fields: Tuple[str, ...],
              at: str = "") -> Dict[str, Any]:
    """``part`` cut to the paths ``fields`` declare: a declared field whole,
    and the dicts on the way to one."""
    out: Dict[str, Any] = {}
    for key, val in part.items():
        path = f"{at}.{key}" if at else key
        if any(path == f or path.startswith(f + ".") for f in fields):
            out[key] = val
        elif isinstance(val, dict) and any(f.startswith(path + ".")
                                           for f in fields):
            sub = _declared(val, fields, path)
            if sub:
                out[key] = sub
    return out


def _merge(into: Dict[str, Any], part: Dict[str, Any]) -> None:
    for key, val in part.items():
        if isinstance(val, dict) and isinstance(into.get(key), dict):
            _merge(into[key], val)
        elif isinstance(val, list) and isinstance(into.get(key), list):
            into[key] = into[key] + val
        else:
            into[key] = val


def run_record(directory, *, names, launched_in, status, engine: str,
               calculation: Optional[str] = None,
               current: Optional[bool] = None,
               fields: Optional[Tuple[str, ...]] = None
               ) -> Optional[Dict[str, Any]]:
    """The record of the run of the stage ``names`` names in ``directory``
    -- the one its caller names, through the run door -- or ``None`` when
    no deck of it is here (§ 5d).  The rest as :func:`run_files` takes
    them.

    ``fields`` -- the record's paths a caller asks for (a path names
    everything under it): only the rows of :func:`contributors` that answer
    one of them are read, and only those paths kept -- the memory a stage
    needs reads a run's peak and ranks, never its whole record
    (`execution/job-request.md` § 3.1a)."""
    f = run_files(directory, names=names, status=status,
                  engine=engine, calculation=calculation,
                  current=current, launched_in=launched_in)
    if f is None:
        return None

    def kept_of(declared: Tuple[str, ...]) -> Tuple[str, ...]:
        # WHAT A ROW ANSWERS OF THE ASK: its fields under an asked path, and
        # an asked path under one of its fields.
        if fields is None:
            return declared
        return tuple(dict.fromkeys(
            [fd for fd in declared
             if any(fd == a or fd.startswith(a + ".") for a in fields)]
            + [a for a in fields
               if any(a.startswith(fd + ".") for fd in declared)]))
    record: Dict[str, Any] = {}
    for row in contributors():
        if row.engines and f.engine not in row.engines:
            continue
        kept = kept_of(row.fields)
        if not kept:
            continue
        try:
            part = _declared(row.read(f), kept)
        except (OSError, ValueError, ParseError):
            # A reporter degrades: one unreadable file costs its own fields,
            # never the record.  A defect in a reader is not caught.
            continue
        _merge(record, part)
    return record or None
