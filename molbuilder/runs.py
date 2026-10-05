"""The run a file belongs to, and its files -- one door.

Contract: `execution/architecture.md` § 3 (`Run`), § 3.2 (*the run a file
belongs to, and its files*; *the description*); `execution/project-layout.md`
§ 1.4a (what a folder is), § 5 (what each file is); `model/parse.md` § 5.1
(the run a folder speaks for); `web/results.md` § 3b (the file card).

Floor 2 (`architecture.md` § 2.1): it reads the description, through its one
reader (`task.read_task`), and composes floor 1's doors -- `calcdirs` for a
folder's own record, `paths` and `identity` for the names a layout gives its
folders, `runfiles` for the names a run gives its files, `runrecord` and
`parse.dirs` for what a run's records and outputs say.  So nothing below it
knows a calculation, which is what `parse/` and `calcdirs` must not
(`architecture.md` § 2.1).

WHY ONE DOOR (plan B11, user 2026-10-04: *"agree to your B11"*).  The run a
file belongs to was worked out by every reader that needed it, each its own
way: labels guessed from the decks in a folder (`rundir.labels_in`), a stage
cut from a name by a second grammar (`identity.parse_stage_token`), the
speaking output picked by the newest file time in one reader and the newest
run index in four others, the description read raw where its kind or shape
was wanted.  The label a run names its files on is the DESCRIPTION's -- a
benchmark trial's is the description's with its point -- and every name is
read back with it, through `runfiles`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from molbuilder import calcdirs
from molbuilder import runfiles as _rf
from molbuilder.identity import parse_token
from molbuilder.paths import trial_label, trial_point


# --------------------------------------------------------------------- #
#  What a folder is, and the description it belongs to (B14)            #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Place:
    """What a folder is in its calculation (`project-layout.md` § 1.4a):
    ``role`` -- ``container`` or ``run``, ``None`` when nothing says -- the
    calculation's ``root`` and its description, ``task``, read once.  A
    folder no calculation claims has none of them: it holds no run of ours
    and is read alone.  ``problem`` is a description that does not read, in
    its reader's words."""
    role: Optional[str] = None
    root: Optional[Path] = None
    task: Any = None
    problem: Optional[str] = None

    @property
    def ours(self) -> bool:
        """The folder belongs to a calculation whose description reads."""
        return self.task is not None


def place_of(directory) -> Place:
    """What ``directory`` is (`project-layout.md` § 1.4a): the calculation
    root, which its description's ``shape`` says is a run (flat) or a
    container (hierarchical), or a folder below it, which its own
    `calcdir.json` says.  **The description is read through its one door,
    `read_task`** (`architecture.md` § 3.2, plan B14: `calcdirs` read its
    ``shape`` raw until 2026-10-04)."""
    from molbuilder.task import read_task
    d = Path(directory)
    root = calcdirs.root_of(d)
    if root is None:
        return Place()
    where = root / _rf.TASK_FILE
    try:
        task = read_task(where)
    except Exception as exc:                               # noqa: BLE001
        return Place(root=root, problem=f"{where} does not read: {exc}")
    if d.resolve() == root.resolve():
        role = calcdirs.RUN if task.shape == "flat" else calcdirs.CONTAINER
    else:
        said = calcdirs.read(d)
        role = said.role if said is not None else None
    return Place(role=role, root=root, task=task)


def _position(folder: Path, place: Place) -> Tuple[str, Optional[str]]:
    """The label a folder's files are named on, and the stage its path
    names -- read off the folders the layout made (`paths`, `identity`):
    ``<NN>_<stage>/`` in the hierarchy, ``bench_<NN>_<stage>/`` at a flat
    root, a trial's ``bench-<point>/`` relabelling its files
    (`paths.trial_label`)."""
    label, stage = place.task.label, None
    try:
        parts = folder.resolve().relative_to(place.root.resolve()).parts
    except ValueError:
        return label, None
    for part in parts:
        if part.startswith("bench_") and parse_token(part[len("bench_"):]):
            stage = part[len("bench_"):]
        elif parse_token(part):
            stage = part
        point = trial_point(part)
        if point:
            label = trial_label(place.task.label, point)
    return label, stage


def _ordinal(stage: Optional[str]) -> int:
    got = parse_token(stage) if stage else None
    return got[0] if got else -1


def speaking(folder, label: str) -> Tuple[Optional[str], Optional[int]]:
    """THE RUN A FOLDER SPEAKS FOR -- ``(stage, run index)`` -- by the one
    rule (`architecture.md` § 3.2, `model/parse.md` § 5.1): the highest
    stage that was launched -- its launch record, or files carrying a run
    index -- then that stage's highest run index; with none launched yet,
    the folder's one stage with a deck.  A file's time decides nothing: a
    copied or restored folder reorders times, never runs."""
    ran: Dict[Optional[str], int] = {}
    decks = set()
    deck_roles = set(_rf.deck_roles())
    for _p, rec in _rf.find(Path(folder), label):
        if rec.run is not None or rec.role == ".run.json":
            ran[rec.stage] = max(ran.get(rec.stage, -1),
                                 rec.run if rec.run is not None else -1)
        if rec.role in deck_roles:
            decks.add(rec.stage)
    if ran:
        stage = max(ran, key=_ordinal)
        return stage, (ran[stage] if ran[stage] >= 0 else None)
    if len(decks) == 1:
        return next(iter(decks)), None
    return None, None


# --------------------------------------------------------------------- #
#  The run, and its files (B11)                                         #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """One run of ours: the calculation it belongs to (``place``), the
    ``folder`` its files are in, the ``label`` they are named on, its
    ``stage`` token and its ``run`` index -- the newest its files carry, or
    the one a file named; ``None`` before it has run."""
    place: Place
    folder: Path
    label: str
    stage: Optional[str] = None
    run: Optional[int] = None

    @property
    def basename(self) -> str:
        """``<label>[_<stage>]`` -- the stem every file of the run carries."""
        return _rf.stem(self.label, self.stage)

    def files(self):
        """``(path, RunFile)`` -- every file of this run's label and stage
        here, read back through `runfiles`, sorted by run index."""
        return _rf.find(self.folder, self.label, stage=self.stage)

    def file(self, role: str, run: Optional[int] = None) -> Optional[Path]:
        """This run's file in ``role`` -- at ``run``, else the newest run
        index that has one (a file with no run index when none does)."""
        hits = [(p, rec) for p, rec in self.files() if rec.role == role
                and (run is None or rec.run == run)]
        return hits[-1][0] if hits else None

    @property
    def deck(self) -> Optional[Path]:
        """The deck this run read, the one its engine runs
        (`runfiles.deck_roles`) -- in its own folder."""
        for role in _rf.deck_roles(self.place.task.engine):
            got = self.file(role)
            if got is not None:
                return got
        return None

    @property
    def stdout(self) -> Optional[Path]:
        """The engine's own output of this run -- its stdout role
        (`runfiles.stdout_roles`) at the run's index -- or ``None``."""
        for role in _rf.stdout_roles(self.place.task.engine):
            got = self.file(role, self.run)
            if got is not None:
                return got
        return None

    @property
    def outputs(self) -> List[Path]:
        """Every engine output of this run's stage here, newest run index
        first -- by the number each carries, never a file's time."""
        roles = set(_rf.stdout_roles(self.place.task.engine))
        return [p for p, rec in reversed(self.files()) if rec.role in roles]

    @property
    def session_log(self) -> Optional[Path]:
        """The run script's session log of THIS run -- the one whose first
        section is its run index (`wrapper_log.log_of_run`), never the
        newest by its stamp -- or ``None``."""
        if self.run is None:
            return None
        from molbuilder.wrapper_log import log_of_run
        return log_of_run(self.folder, self.label, self.run, self.stage)

    @property
    def stage_deck(self) -> Optional[Path]:
        """The stage's own deck, the copy prep writes (`project-layout.md`
        § 5.2): in the stage's folder in the hierarchy, the run's own in the
        flat shape, where one folder is both."""
        deck = self.deck
        if deck is None:
            return None
        if self.place.task.shape == "flat" or self.stage is None:
            return deck
        return self.place.root / self.stage / deck.name


def run_of(path, stage: Optional[str] = None) -> Optional[Run]:
    """THE RUN ``path`` BELONGS TO -- a file of a run, or a run's folder --
    or ``None``: a container, a folder no calculation claims, or one whose
    description does not read (`architecture.md` § 3.2).

    The label is the description's (a trial's, `paths.trial_label`); the
    stage is the one the folder's path names, else ``stage`` -- the one the
    caller asks about, in a folder several stages share (a flat
    calculation's) -- else the one the file's own name reads back to, else
    the run the folder speaks for (:func:`speaking`); the run index is the
    file's own, else the newest."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    place = place_of(folder)
    if not place.ours or place.role != calcdirs.RUN:
        return None
    label, named = _position(folder, place)
    stage = named or stage
    run = None
    if not p.is_dir():
        rec = _rf.parse(p.name, label)
        if rec is not None and rec.stage is not None:
            stage = stage or rec.stage
            run = rec.run
    if stage is None:
        stage, newest = speaking(folder, label)
        run = run if run is not None else newest
    elif run is None:
        runs = [rec.run for _p, rec in _rf.find(folder, label, stage=stage)
                if rec.run is not None]
        run = max(runs) if runs else None
    return Run(place=place, folder=folder, label=label, stage=stage,
               run=run)


# --------------------------------------------------------------------- #
#  What a run declared about its atoms (B12)                            #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Declared:
    """What a run of ours declared about its atoms -- its own deck's records
    (`architecture.md` § 3.2): the ``deck`` that says it, its
    ``atom_metadata`` block (labels, held atoms, annotations) and its
    ``engine_offset`` record (the axis kinds, the cell prep placed the atoms
    in); ``None`` for a record the deck does not carry."""
    deck: Optional[Path] = None
    atom_metadata: Optional[Dict[str, Any]] = None
    engine_offset: Optional[Dict[str, Any]] = None


def declared(run: Run) -> Declared:
    """WHAT ``run`` DECLARED ABOUT ITS ATOMS -- read from its own deck
    (:attr:`Run.deck`) by the readers of molbuilder's blocks, never from a
    file looked for beside an output by its name (`model/parse.md` § 5.3,
    plan B12).  A run with no deck here, or one whose deck carries neither
    record, declares nothing."""
    deck = run.deck
    if deck is None:
        return Declared()
    try:
        text = deck.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return Declared(deck=deck)
    from molbuilder.deck_record import extract_engine_offset
    from molbuilder.script_emit import _extract_atom_metadata_dict
    return Declared(deck=deck,
                    atom_metadata=_extract_atom_metadata_dict(text),
                    engine_offset=extract_engine_offset(text))


def about(path) -> Dict[str, Any]:
    """WHAT A FILE IS -- its row of the catalogue, read back with its run's
    label (`runfiles.row_for`): ``{ours: True, what, writer, when, kind}``;
    or ``{ours: False}``, a file molbuilder does not write -- the engine's,
    SLURM's own, a person's -- and every file of a folder no calculation
    claims (`web/results.md` § 3b, `job-contracts.md` § 2.2)."""
    p = Path(path)
    folder = p.parent
    place = place_of(folder)
    if not place.ours:
        return {"ours": False}
    label, _stage = _position(folder, place)
    return _row_about(p, label)


def run_answer(path) -> Optional[Dict[str, Any]]:
    """``{state, detail, live}`` -- how the run the file at ``path`` belongs
    to is doing, the one door's answer (`parse.dirs.run_state_of`, asked
    with its launch record), with ``problem`` when that record does not
    read.  ``live`` -- `job.LIVE_STATES` -- is what a Results viewer follows
    (`web/results.md` § 4.1).  ``None`` for a file of no run of ours, as
    for an upload: nothing is asked of a folder no calculation claims
    (`model/parse.md` § 5).  *(It answered for such a file's folder, with
    no run named, until 2026-10-04: plan B11, 3b.2.)*"""
    from molbuilder.parse.dirs.job import LIVE_STATES
    from molbuilder.parse.dirs.rundir import run_state_of
    run = run_of(path)
    if run is None:
        return None
    st, problem = run_state_of(run.folder, run.basename)
    said: Dict[str, Any] = {"state": st.state, "detail": st.detail,
                            "live": st.state in LIVE_STATES}
    if problem:
        said["problem"] = problem
    return said


# --------------------------------------------------------------------- #
#  A folder, as a viewer asks about it -- the directory door            #
# --------------------------------------------------------------------- #

def _in_words(writer: str) -> str:
    """A row's ``writer`` as a person reads it: the parenthesised function
    names the contract's manifest carries (`` (`runwrap.write_run_wrapper`)``)
    left out -- they are the contract's, for whoever changes the writer; the
    verb and the moment are the person's."""
    import re
    return re.sub(r"\s*\([^()]*`[^()]*\)", "", writer).strip()


def _row_about(path: Path, label: Optional[str]) -> Dict[str, Any]:
    """:func:`about`'s answer for a file whose folder's label is known."""
    row = _rf.row_for(path, label) if label else None
    if row is None:
        return {"ours": False}
    return {"ours": True, "what": row.what, "writer": _in_words(row.writer),
            "when": row.when, "kind": row.kind}


def _openable(folder: Path, label: str, stage: Optional[str],
              calculation: Optional[str], attempts: List[str]
              ) -> Optional[str]:
    """What a viewer opens in a folder of ours: the first role the
    calculation produces (`runfiles.result_roles`) in which the folder's run
    has a file -- the run it speaks for, at its newest run index -- that a
    parser claims (`model/parse.md` § 5.2)."""
    from molbuilder.parse.dirs.rundir import _claimed
    staged = {a.role: a.staged for a in _rf.ON_THE_LABEL}
    for role in _rf.result_roles(calculation):
        hits = [p for p, _rec in _rf.find(
            folder, label, role=role,
            stage=stage if staged.get(role, True) else None)]
        if not hits:
            continue
        attempts.append(f"*{role} -> {len(hits)} of this run's")
        for cand in reversed(hits):
            if _claimed(str(cand)):
                attempts.append(f"*{role} -> {cand.name}")
                return str(cand)
            attempts.append(f"*{role} -> {cand.name}: no parser claims it, "
                            f"not offered")
    return None


def openable(directory) -> Tuple[Optional[str], List[str]]:
    """``(path, trail)`` -- the file a viewer should load in ``directory``:
    the run it speaks for's result in a folder of ours, the calculation's
    product at a container's root, and in a folder no calculation claims the
    newest file of a role a run writes that a parser claims
    (`parse.dirs.openable_in`)."""
    from molbuilder.parse.dirs.rundir import openable_in
    d = Path(directory)
    place = place_of(d)
    if not place.ours:
        return openable_in(str(d))
    attempts: List[str] = [f"calculation: {place.task.calculation}"]
    label, stage = _position(d, place)
    if place.role == calcdirs.RUN and stage is None:
        stage, _n = speaking(d, label)
    return _openable(d, label, stage, place.task.calculation,
                     attempts), attempts


def folder_answer(directory) -> Dict[str, Any]:
    """WHAT A FOLDER IS AND HOLDS, for a viewer -- the directory door the
    Results tab serves (`/api/results/dir`; `model/parse.md` § 5.0,
    `web/results.md` § 2.3): ``place``, ``engine``, ``openable``,
    ``attempts``, ``status`` and ``record``, and per file its role, label,
    stage, whether a parser reads it and what it is (``files``).

    **What the folder IS decides what is asked of it** (`project-layout.md`
    § 1.4a).  A RUN of ours is asked everything, about the run it speaks for
    (:func:`speaking`): its result, its state, its record.  A CONTAINER has
    no state and no record, though its calculation's product opens at its
    root.  A folder no calculation claims holds no run of ours: its files
    are listed, the registry says what it can open, and none of them is
    claimed -- no label read off a deck, no state asked.  *(This was
    `parse.dirs.rundir.JobDirParser` until 2026-10-04, below the floor it
    must not cross: it read the description raw and guessed labels from the
    decks -- plan B11, B14.)*
    """
    from molbuilder.parse import detect
    from molbuilder.parse.contract import engine_of
    from molbuilder.parse.dirs.record import run_record
    from molbuilder.parse.dirs.rundir import openable_in, run_state_of
    from molbuilder.parse.errors import ParseError

    d = Path(directory)
    place = place_of(d)
    attempts: List[str] = []
    if place.problem:
        attempts.append(place.problem)
    status = record = None
    label: Optional[str] = None
    if not place.ours:
        engine = engine_of(str(d))
        chosen, trail = openable_in(str(d))
        attempts += trail
        attempts.append(
            "this folder is not marked as part of a calculation, so only what "
            "is in it is shown (project-layout.md § 1.4a)")
    else:
        engine = place.task.engine
        calc = place.task.calculation
        label, stage = _position(d, place)
        attempts.append(f"calculation: {calc}")
        if place.role == calcdirs.RUN:
            run = run_of(d)
            chosen = _openable(d, run.label, run.stage, calc, attempts)
            if run.stage is not None:
                st, problem = run_state_of(run.folder, run.basename)
                if problem:
                    attempts.append(problem)
                status = {"state": st.state, "detail": st.detail,
                          "last_change_at": st.last_change_at,
                          "active_source": st.active_source}
                record = run_record(run.folder, label=run.label,
                                    stage=run.stage, status=st, engine=engine,
                                    calculation=calc,
                                    stage_deck=run.stage_deck)
            else:
                attempts.append("no run here yet: no stage has run in this "
                                "folder, and more than one is prepped")
        elif place.role == calcdirs.CONTAINER:
            chosen = _openable(d, label, stage, calc, attempts)
            attempts.append(
                "this directory is a container, not a run -- it has no run "
                "state; its runs are the directories below it "
                "(project-layout.md § 1.4)")
        else:
            # INSIDE A CALCULATION, AND NOTHING SAYS WHAT IT IS -- no
            # `calcdir.json` (a `pseudos/`, a folder a person made): its files
            # are read with the calculation's label, and no run is asked of it.
            chosen = _openable(d, label, stage, calc, attempts)
            attempts.append(
                "nothing marks this folder in its calculation (no "
                "calcdir.json), so only what is in it is shown "
                "(project-layout.md § 1.4a)")
    declared = set(_rf.roles())
    files = []
    for entry in sorted(d.iterdir(), key=lambda e: e.name):
        if not entry.is_file():
            continue
        # READ BACK WITH THE RUN'S LABEL, a declared role or nothing
        # (`job-contracts.md` § 2.2a): a name read without its label is not
        # taken for one of ours -- SIESTA's `fdf.<stamp>.log` is not PySCF's
        # log (plan D24: the listing fell back to `role_of` until 2026-10-04).
        rec = _rf.parse(entry.name, label) if label else None
        if rec is not None and rec.role not in declared:
            rec = None
        try:
            kind = detect(str(entry))
            parser, opens = kind.name, getattr(
                getattr(kind, "output", None), "__name__", None)
        except (ParseError, OSError, ValueError, LookupError):
            parser, opens = None, None
        st_ = entry.stat()
        files.append({
            "name": entry.name,
            "role": rec.role if rec is not None else None,
            "label": rec.label if rec is not None else None,
            "stage": rec.stage if rec is not None else None,
            "parser": parser, "opens": opens,
            "size": st_.st_size, "mtime": st_.st_mtime,
            "about": _row_about(entry, label),
        })
    return {"place": {"role": place.role,
                      "calculation": str(place.root) if place.root else None},
            "engine": engine,
            "openable": chosen,
            "attempts": attempts,
            "status": status,
            "record": record,
            "files": files}


__all__ = ["Place", "place_of", "speaking", "Run", "run_of", "about",
           "run_answer", "openable", "folder_answer"]
