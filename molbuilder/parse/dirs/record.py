"""`run_record` — one attempt's record: what ran, with what, and how it went.

Contract: `model/parse.md` § 5d — and § 5d.1b for the shape of this module:
**a declared table, one reader per file.**  :data:`CONTRIBUTORS` names, for
each file a run can leave, the record fields it answers and the reader that
answers them; the composer walks the table and merges what each row returns.
Engines differ only in which files exist, so a run of any kind gets every
fact its files state, and a new engine or a new file is a row, not a path
through this module.  No row reads a format itself: each asks the reader that
lives with that format (§ 1a) — the SIESTA family's table, `runwrap`'s reader
of its own log, the instruments, `materialize` and `prep` for their files,
`script_emit` for its block, the registry for SIESTA's `fdf` log.

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

**Which run** (§ 5d.1): the one the directory's status speaks for --
`run_status`'s ``active_source``, picked by stage then time (§ 5.1), so the
record and the status can never describe two different runs -- at its latest
``-runN``, with each earlier one and how it ended; the `fdf` log paired with
the ``.out`` by its stamp; the wrapper log whose FIRST section is run N.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: How much of a ``.out`` the launch, build, solver and pseudopotential lines
#: can sit in -- the solver is printed after the basis report, ~49 KB into a
#: 42-atom run (`jobset/summarize.py`'s ``_SETUP_WINDOW``, the same measure).
_HEAD = 512 * 1024
#: The tail, for ``>> End of run``.
_TAIL = 16 * 1024


# --------------------------------------------------------------------------- #
#  Which files are the run's                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunFiles:
    """The files of the run a record describes -- :func:`run_files`' answer.
    A file the attempt does not hold is ``None``."""
    directory: Path
    engine: str
    status: Any = None                     # the directory's `RunStatus`
    deck: Optional[Path] = None
    label: Optional[str] = None
    stage: Optional[str] = None
    run: Optional[int] = None
    out: Optional[Path] = None             # SIESTA family's stdout, run N
    pyscf_log: Optional[Path] = None       # PySCF's stdout, run N
    engine_log: Optional[Path] = None      # PySCF's own logger's file
    fdf_log: Optional[Path] = None         # SIESTA's, paired by stamp
    wrapper_section: Dict[str, Any] = field(default_factory=dict)
    wrapper_text: str = ""                 # run N's section, for the fence
    timing: Optional[Path] = None          # the SIESTA family's tee, run N
    progress_log: Optional[Path] = None    # the stage's progress log
    monitor_log: Optional[Path] = None
    util_csv: Optional[Path] = None
    earlier: Tuple[Tuple[int, Optional[Path]], ...] = ()
    #: The ``.out``'s head, read once here for every row that reads it.
    out_head: str = ""


def _read(path: Optional[Path], *, head: Optional[int] = None,
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




def _deck_roles(engine: str) -> List[str]:
    """The deck role(s) the catalogue gives this engine -- both when the
    engine is not known."""
    from ...runfiles import WRITTEN
    from .rundir import _DECK_ROLES
    own = [r for r in _DECK_ROLES
           if any(a.role == r and a.engine == engine for a in WRITTEN)]
    return own or list(_DECK_ROLES)


def run_files(directory, *, status=None,
              engine: Optional[str] = None) -> Optional[RunFiles]:
    """What the run this directory's status speaks for left, found through
    the doors that own each name -- or ``None`` when neither a deck nor an
    output says whose run it is.

    **A FOLDER READ ALONE** -- its output named by hand, or copied without
    its deck -- is still a run: the output the status speaks for is the run,
    taken as it is, and the folder's one deck is its deck
    (:func:`_alone_files`, `model/parse.md` § 5d.1).  Until 2026-09-27 such
    a folder had a status and a viewer and a record of nothing but its
    state, because every file was looked up by molbuilder's names.

    ``status`` and ``engine`` are the directory's `run_status` and
    `engine_of`, passed by a caller that already has them (the directory
    door does); asked here otherwise.

    **THE RUN IS THE ONE THE STATUS SPEAKS FOR** (§ 5.1: stage, then time).
    A flat calculation keeps every stage in one directory, and taking the
    first deck by name described stage 1 while the status reported stage 3
    -- measured 2026-09-26 on four real flat directories.  Before anything
    has run, the directory's one deck; several decks and no output is no run
    to describe.
    """
    from ..contract import engine_of
    from ...runfiles import find, find_by_role
    from .job import run_status
    from .rundir import labels_in, read_back

    d = Path(directory)
    if status is None:
        status = run_status(d)
    engine = engine or engine_of(str(d))
    labels = labels_in(str(d))
    # THE ENGINE'S OWN DECK ROLE, from the catalogue's `engine` column -- and
    # only files a label reads back: every attempt also holds the modules its
    # monitor runs on, which carry the PySCF deck's suffix, name no `JOB`,
    # and so are nobody's deck (`labels_in`).
    decks = [(p, r) for role in _deck_roles(engine)
             for p in find_by_role(d, role)
             for r in [read_back(p.name, labels)] if r is not None]
    speaks = (read_back(status.active_source, labels)
              if status.active_source else None)
    if speaks is not None:
        mine = [(p, r) for p, r in decks
                if (r.label, r.stage) == (speaks.label, speaks.stage)]
    else:
        mine = decks if len({(r.label, r.stage) for _p, r in decks}) == 1 \
            else []
    alone = _output_read_alone(d, status, speaks, engine)
    if alone is not None:
        return _alone_files(d, engine, status, alone,
                            decks=[p for p, _r in decks])
    if not mine:
        return None
    deck, rec = mine[0]
    label, stage = rec.label, rec.stage

    # ONE LISTING of this run's files, filtered below -- `find` lists and
    # reads every name in the directory on each call, and a flat directory
    # holds hundreds.
    listing = find(d, label, stage=stage)

    def one(role: str, run: Optional[int]) -> Optional[Path]:
        hits = [p for p, rf in listing
                if rf.role == role and (run is None or rf.run == run)]
        return hits[-1] if hits else None

    runs = sorted({rf.run for _p, rf in listing if rf.run is not None})
    n = runs[-1] if runs else None
    out = one(".out", n) if n is not None else one(".out", None)
    pyscf_log = one(".pyscf.log", n) if n is not None else None

    out_head = _read(out, head=_HEAD)
    fdf_log = _fdf_log_of(d, out_head) if out is not None else None

    # THE WRAPPER LOG WHOSE FIRST SECTION IS RUN N: a retry appends its own
    # section to the first log and opens one of its own -- the pairing
    # `run_status` reads SIESTA's stderr by (`wrapper_log.log_of_run`).
    from ...wrapper_log import log_of_run, read_wrapper_log
    section: Dict[str, Any] = {}
    section_text = ""
    log = log_of_run(d, label, n, stage) if n is not None else None
    if log is not None:
        text = _read(log)
        sections = read_wrapper_log(text)
        if sections:
            section = sections[0]
            section_text = _first_section_text(text)

    earlier = tuple((k, one(".out", k) or one(".pyscf.log", k))
                    for k in runs[:-1])
    return RunFiles(
        directory=d, engine=engine, status=status,
        deck=deck, label=label,
        stage=stage, run=n, out=out, out_head=out_head, pyscf_log=pyscf_log,
        engine_log=one(".log", None), fdf_log=fdf_log,
        wrapper_section=section, wrapper_text=section_text,
        timing=one(".scf-timing.log", n) if n is not None else None,
        progress_log=one(".molwatch.log", None),
        monitor_log=one(".monitor.log", n) if n is not None else None,
        util_csv=one(".util.csv", n) if n is not None else None,
        earlier=earlier)


def _fdf_log_of(d: Path, out_head: str) -> Optional[Path]:
    """SIESTA's ``fdf.<stamp>.log`` for the ``.out`` whose head this is.

    THE FDF LOG IS PAIRED BY ITS STAMP: SIESTA opens it in the second it
    prints ``>> Start of run`` -- exact on 122 of 122 real outputs, and
    SIESTA's own naming, so it holds in a folder molbuilder did not write.
    """
    from ..engines import siesta_grammar as _G
    facts: Dict[str, Any] = {}
    for line in out_head.splitlines():
        if _G.read_launch_line(line, facts) == "run_start_local":
            break
    start = facts.get("run_start_local")
    if not start:
        return None
    from ..engines.siesta_fdflog import stamp_of as _fdf_stamp
    return next((p for p in sorted(d.glob("fdf.*.log"))
                 if _fdf_stamp(p) == start), None)


def _output_read_alone(d: Path, status, speaks, engine: str) -> Optional[Path]:
    """The output the status speaks for, when no deck in the folder names it
    -- or ``None``.

    A FOLDER READ ALONE (§ 5d.1): a SIESTA run made by hand (``input.fdf``,
    ``siesta.out``), or an output copied without its deck.  Its name carries
    no label this folder's decks state, so molbuilder's naming cannot find
    its files -- and the output itself says what ran.
    """
    from ...runfiles import role_of, stdout_roles
    if speaks is not None or not status.active_source:
        return None
    path = d / status.active_source
    return (path if path.is_file()
            and role_of(path.name) in stdout_roles(engine) else None)


def _alone_files(d: Path, engine: str, status, output: Path, *,
                 decks: List[Path]) -> RunFiles:
    """A folder read alone: the run is its output, taken as it is.  SIESTA's
    ``fdf`` log is paired by the output's start stamp, as always, and the
    deck is the folder's one deck -- of several, the one whose
    ``SystemLabel`` is the one SIESTA's log says it read; none when that
    does not single one out.  molbuilder's own files -- the wrapper log, the
    instruments, ``run.json`` -- are not there, and their fields are simply
    not stated."""
    from ...runfiles import role_of
    out = output if role_of(output.name) == ".out" else None
    head = _read(out, head=_HEAD)
    fdf_log = _fdf_log_of(d, head) if out is not None else None
    return RunFiles(directory=d, engine=engine, status=status,
                    deck=_deck_read_alone(decks, fdf_log),
                    out=out, out_head=head,
                    pyscf_log=None if out is not None else output,
                    fdf_log=fdf_log)


def _deck_read_alone(decks: List[Path],
                     fdf_log: Optional[Path]) -> Optional[Path]:
    """The deck of a folder read alone (:func:`_alone_files`)."""
    if len(decks) <= 1:
        return decks[0] if decks else None
    if fdf_log is None:
        return None
    from ..fdf import system_label
    from ..registry import parse as _parse
    try:
        read = (_parse(fdf_log).params.get("systemlabel") or {}).get("value")
    except Exception:                                      # noqa: BLE001
        return None
    same = [p for p in decks if read and system_label(_read(p)) == read]
    return same[0] if len(same) == 1 else None


def _first_section_text(text: str) -> str:
    """The first run section of a wrapper log, as text -- where its
    effective-parameters block sits."""
    from ...wrapper_log import WRAPPER_LOG_START
    lines, seen = [], 0
    for line in text.splitlines():
        if WRAPPER_LOG_START in line:
            seen += 1
            if seen > 1:
                break
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  The contributors -- one row per file, each asking the reader that owns it  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Contributor:
    """One row of :data:`CONTRIBUTORS`: the record fields it answers, which
    engines' runs have its file (``()`` = any), and the read."""
    name: str
    fields: Tuple[str, ...]
    engines: Tuple[str, ...]
    read: Callable[[RunFiles], Dict[str, Any]]


def _siesta_out(f: RunFiles) -> Dict[str, Any]:
    """The ``.out``'s build, launch, solver and pseudopotential lines, through
    the SIESTA family's table (`siesta_grammar`)."""
    from ..engines import siesta_grammar as _G
    head = f.out_head
    if not head:
        return {}
    build: Dict[str, Any] = {}
    launch: Dict[str, Any] = {}
    solver: Dict[str, Any] = {}
    lines = head.splitlines()
    for line in lines:
        (_G.read_build_line(line, build) or _G.read_launch_line(line, launch)
         or _G.read_diag_line(line, solver))
    for line in _read(f.out, tail=_TAIL).splitlines():
        _G.read_launch_line(line, launch)
    out: Dict[str, Any] = {"computation": {}}
    comp = out["computation"]
    if build:
        exe = str(build.get("executable", "")).rsplit("/", 1)[-1]
        comp["engine"] = {**({"program": exe} if exe else {}),
                          **({"version": build["version"]}
                             if "version" in build else {}),
                          "build": {k: v for k, v in build.items()
                                    if k not in ("executable", "version")}}
    if solver:
        comp["solver"] = solver
    if "n_mpi_processes" in launch:
        comp["launch"] = {"ranks": launch["n_mpi_processes"]}
    time = {k: launch[k] for k in ("run_start_local", "run_end_local")
            if k in launch}
    if time:
        comp["time"] = time
    pseudos = []
    for sp in _G.read_psml_lines(lines):
        entry = dict(sp)
        path = f.directory / sp["file"]
        if path.is_file():
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            from ...pseudos import parse_psml_header
            info = parse_psml_header(path)
            for key in ("xc_family", "xc_authors", "relativistic",
                        "generator"):
                val = getattr(info, key, None)
                if val and val != "unknown":
                    entry[key] = val
        pseudos.append(entry)
    if pseudos:
        out["setup"] = {"pseudopotentials": pseudos}
    return out


def _ending(f: RunFiles) -> Dict[str, Any]:
    """How the latest run is doing, whether each of its phases converged, and
    how each earlier run ended -- all from the ONE scan `run_status` made to
    judge the state (``RunStatus.endings``).  Nothing is read a second
    time."""
    from ..engines._run_ending import CONCLUDED
    st = f.status
    if st is None:
        return {}
    verdict: Dict[str, Any] = {"state": st.state, "detail": st.detail}
    endings = getattr(st, "endings", {}) or {}
    main = f.out or f.pyscf_log
    end = endings.get(main.name) if main is not None else None
    if end is not None:
        verdict["ended"] = end.run_state
        # PER PHASE (§ 5d.6): a device's periodic initialization converging
        # does not speak for its NEGF loop, so there is no one flag to give.
        if end.phases:
            verdict["converged"] = dict(end.phases)
    # AN EARLIER RUN STATES AN ENDING OR NOTHING.  A later run exists, so
    # one with no ending marker is not running -- it was cut off, which its
    # own file cannot say (§ 2b) -- and "running" is what the scan answers
    # for a file with no marker.  Stated only when the file states it
    # (§ 5d.1a).
    earlier = []
    for k, path in f.earlier:
        row: Dict[str, Any] = {"run": k}
        said = endings.get(path.name) if path is not None else None
        if said is not None and said.run_state in CONCLUDED:
            row["ended"] = said.run_state
        earlier.append(row)
    out: Dict[str, Any] = {"verdict": verdict}
    if earlier:
        out["earlier"] = earlier
    return out


def _wrapper(f: RunFiles) -> Dict[str, Any]:
    """Run N's section of the wrapper log, through `runwrap`'s own reader."""
    sec = f.wrapper_section
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
    from ..registry import parse as _parse
    from ..instruments import utilisation

    def metrics(path):
        if path is None:
            return {}
        try:
            return dict(_parse(Path(path)).metrics)
        except Exception:                                  # noqa: BLE001
            return {}

    mon = metrics(f.monitor_log)
    util = utilisation(mon, metrics(f.util_csv)) if f.util_csv else {}
    comp: Dict[str, Any] = {}
    # THE RUN'S STAMPED SCF ROWS (§ 5c): the SIESTA family's tee, PySCF's
    # progress log -- the one a run has, read by the one rule.
    time = _timing_figures(f.timing if f.timing is not None
                           else f.progress_log)
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


def scf_timing_of(output) -> Dict[str, Any]:
    """The SCF-timing figures of the run ``output`` is an output of: a PySCF
    progress log stamps its own rows; a SIESTA-family ``<base>-runN.out``'s
    are its run's tee, ``<base>-runN.scf-timing.log``, found through the
    run-file door by the output's own label, stage and run.  Read as the
    record reads them.  ``{}`` when the run has none -- an output read
    alone, a log from before its rows were stamped."""
    from ...runfiles import find, role_of
    from .rundir import labels_in, read_back
    p = Path(output)
    if role_of(p.name) == ".molwatch.log":
        return _timing_figures(p)
    rec = read_back(p.name, labels_in(str(p.parent)))
    if rec is None or rec.run is None:
        return {}
    logs = [q for q, _rf in find(p.parent, rec.label, stage=rec.stage,
                                 run=rec.run, role=".scf-timing.log")]
    return _timing_figures(logs[-1]) if logs else {}


def _pyscf_sys(f: RunFiles) -> Dict[str, Any]:
    """PySCF's own report of itself, from its logger's file."""
    from ..engines.pyscf import read_pyscf_sys_info
    info = read_pyscf_sys_info(_read(f.engine_log, head=_HEAD))
    if not info:
        return {}
    engine = {"program": "pyscf"}
    for key in ("version", "python"):
        if key in info:
            engine[key] = info[key]
    comp: Dict[str, Any] = {"engine": engine}
    if "threads" in info:
        comp["launch"] = {"threads_engine": info["threads"]}
    return {"computation": comp}


def _launch_record(f: RunFiles) -> Dict[str, Any]:
    """``run.json`` -- how the attempt was launched (`materialize`)."""
    from ...runrecord import read_run_launch
    rec = read_run_launch(f.directory,
                          basename=f.deck.stem if f.deck else None) or {}
    # `continued_from`: the run this attempt continued from, as prep took it
    # (`job-system.md` § 5.4, plan W37) -- written by the launch from the
    # attempt's `.continued-from`.
    launch = {k: rec[k] for k in ("mode", "command", "job_id", "launched_at",
                                  "placed_on", "continued_from")
              if rec.get(k) is not None}
    return {"computation": {"launch": launch}} if launch else {}


def _concluded(f: RunFiles) -> Dict[str, Any]:
    """What the run's process said on its way out: the marker `run_status`
    already found at the latest run (``RunStatus.concluded``), read by its one
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
    """The deck as run: its path and hash, whether it is still the stage's
    deck, and what it was gathered from."""
    from ...runrecord import read_gathered_from
    from ...script_emit import same_calculation
    if f.deck is None:
        return {}
    text = _read(f.deck)
    deck: Dict[str, Any] = {
        "path": f.deck.name,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    stage_deck = f.directory.parent / f.deck.name
    if stage_deck.is_file() and stage_deck.resolve() != f.deck.resolve():
        deck["current"] = bool(same_calculation(text, _read(stage_deck)))
    gathered = read_gathered_from(f.directory)
    if gathered:
        deck["gathered_from"] = gathered
    return {"deck": deck}


def _setup(f: RunFiles) -> Dict[str, Any]:
    """The three columns (§ 5d.3), from the run's own records: the wrapper's
    or the deck's block for the default, the deck or the block for what was
    asked, SIESTA's fdf log or the block for what was used."""
    from .setup import setup_rows
    return setup_rows(f)


from ..engines.siesta_grammar import (PHASE_NEGF as _PHASE_NEGF,  # noqa: E402
                                      PHASE_PERIODIC as _PHASE_PERIODIC)

#: The engines a run record is composed for; a row whose ``engines`` is
#: ``()`` answers for each.
_ENGINES = ("siesta", "pyscf")

#: The timing instrument's figures (§ 5c): the headline phase's, and each
#: phase's own under its name.
_TIMING = tuple(f"computation.time.{k}" for k in ("s_per_iter",
                                                  "iters_measured")) + tuple(
    f"computation.time.{k}_{phase}"
    for k in ("s_per_iter", "iters_measured", "rows")
    for phase in (_PHASE_PERIODIC, _PHASE_NEGF))

#: THE TABLE (§ 5d.1b).  ``fields`` are the record paths a row answers, and
#: the declaration is the mechanism: a row contributes those fields and no
#: others (:func:`_declared`), and this module refuses to load when two rows
#: answer one field for one engine (:func:`_one_source`) -- § 5c.1's "one
#: source per quantity".
CONTRIBUTORS: Tuple[Contributor, ...] = (
    Contributor("siesta-out",
                ("computation.engine.program", "computation.engine.version",
                 "computation.engine.build", "computation.solver",
                 "computation.launch.ranks", "computation.time.run_start_local",
                 "computation.time.run_end_local", "setup.pseudopotentials"),
                ("siesta",), _siesta_out),
    Contributor("ending", ("verdict.ended", "verdict.state",
                           "verdict.detail", "verdict.converged", "earlier"),
                (), _ending),
    Contributor("wrapper-log",
                ("computation.host.hostname", "computation.host.user",
                 "computation.host.cwd", "computation.host.conda_env",
                 "computation.host.python", "computation.host.node_phys_cores",
                 "computation.host.node_sockets",
                 "computation.host.node_cores_per_socket",
                 "computation.launch.ranks_asked", "computation.launch.threads",
                 "computation.engine.binary",
                 "computation.time.engine_elapsed_s"),
                (), _wrapper),
    Contributor("instruments",
                _TIMING + ("computation.memory", "computation.host.machine"),
                (), _instruments),
    Contributor("pyscf-log",
                ("computation.engine.program", "computation.engine.version",
                 "computation.engine.python",
                 "computation.launch.threads_engine"),
                ("pyscf",), _pyscf_sys),
    Contributor("run-json",
                ("computation.launch.mode", "computation.launch.command",
                 "computation.launch.job_id", "computation.launch.launched_at",
                 "computation.launch.placed_on",
                 "computation.launch.continued_from"),
                (), _launch_record),
    Contributor("concluded", ("computation.exit",), (), _concluded),
    Contributor("deck", ("deck",), (), _deck),
    Contributor("setup", ("setup.rows", "setup.engine_only",
                          "verdict.findings"), (), _setup),
)


def _one_source(rows: Tuple[Contributor, ...]) -> None:
    """Refuse a table in which two rows answer one field -- or one answers a
    field inside another's -- for an engine both serve."""
    owner: Dict[Tuple[str, str], str] = {}
    for row in rows:
        for engine in (row.engines or _ENGINES):
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


_one_source(CONTRIBUTORS)


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


def run_record(directory, *, status=None,
               engine: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The record of the run this directory's status speaks for, or ``None``
    when no deck says whose run it is (§ 5d).  ``status`` and ``engine`` as
    :func:`run_files` takes them."""
    f = run_files(directory, status=status, engine=engine)
    if f is None:
        return None
    record: Dict[str, Any] = {}
    if f.run is not None:
        record["run"] = f.run
    for row in CONTRIBUTORS:
        if row.engines and f.engine not in row.engines:
            continue
        try:
            part = _declared(row.read(f), row.fields)
        except Exception:                                  # noqa: BLE001
            # A reporter degrades: one unreadable file costs its own fields,
            # never the record.
            continue
        _merge(record, part)
    return record or None
