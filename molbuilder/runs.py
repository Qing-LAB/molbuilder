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

WHY ONE DOOR (plan B11, user 2026-10-04: *"agree to your B11"*).  The label a
run names its files on is the DESCRIPTION's -- a
benchmark trial's is the description's with its point -- and every name is
read back with it, through `runfiles`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from molbuilder import calcdirs
from molbuilder import runfiles as _rf
from molbuilder.identity import parse_token
from molbuilder.paths import (attempt_index, flat_bench_token, trial_label,
                              trial_point)

if TYPE_CHECKING:
    from molbuilder.transport.stages import Point


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
    `read_task`** (`architecture.md` § 3.2, plan B14)."""
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


def _position(folder: Path, place: Place
              ) -> Tuple[str, Optional[str], bool]:
    """The label a folder's files are named on, the stage its path names,
    and whether it is a benchmark trial's -- read off the folders the
    layout made (`paths`, `identity`): ``<NN>_<stage>/`` in the hierarchy,
    ``bench_<NN>_<stage>/`` at a flat root, a trial's ``bench-<point>/``
    relabelling its files (`paths.trial_label`)."""
    label, stage, trial = place.task.label, None, False
    try:
        parts = folder.resolve().relative_to(place.root.resolve()).parts
    except ValueError:
        return label, None, False
    for part in parts:
        bench_of = flat_bench_token(part)
        if bench_of:
            stage = bench_of
        elif parse_token(part):
            stage = part
        point = trial_point(part)
        if point:
            label, trial = trial_label(place.task.label, point), True
    return label, stage, trial


def _ordinal(stage: Optional[str]) -> int:
    got = parse_token(stage) if stage else None
    return got[0] if got else -1


def _last_here(folder, label: str) -> Optional[Tuple[str, int]]:
    """``(stage, run)`` of the last run in a folder a calculation's stages
    share -- the run its record names (`runrecord.last_run`), whose restart
    files and results lie there -- or ``None`` while none has started."""
    from molbuilder.runrecord import last_run
    ref = last_run(folder, label)
    if ref is None:
        return None
    rec = _rf.parse_run_name(ref.name, label)
    return rec.stage, rec.run


def speaking(folder, label: str) -> Optional[str]:
    """THE STAGE A FOLDER SPEAKS FOR, by the one rule (`architecture.md`
    § 3.2, `model/parse.md` § 5.1): where the calculation's stages share it,
    its last run's -- the run whose results lie there, by its record
    (`runrecord.last_run`); else the highest stage that was launched -- one
    with a launch record -- and with none launched yet, the folder's one
    stage with a deck; its run is that stage's newest
    (`runrecord.newest_run`, asked by :func:`run_of`).  A file's time
    decides nothing: a copied or restored folder reorders times, never
    runs."""
    last = _last_here(folder, label)
    if last is not None:
        return last[0]
    ran = set()
    decks = set()
    deck_roles = set(_rf.deck_roles())
    for _p, rec in _rf.find(Path(folder), label):
        if rec.role == ".run.json" and rec.run is not None:
            ran.add(rec.stage)
        if rec.role in deck_roles:
            decks.add(rec.stage)
    if ran:
        return max(ran, key=_ordinal)
    if len(decks) == 1:
        return next(iter(decks))
    return None


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
    #: A benchmark trial's run, in its own folder in either shape.
    trial: bool = False

    @property
    def basename(self) -> str:
        """``<label>[_<stage>]`` -- the stem every file of the run carries."""
        return _rf.stem(self.label, self.stage)

    @property
    def launched_in(self) -> Optional[Path]:
        """THE FOLDER THIS RUN'S LAUNCH RECORD LIES IN -- its own ``run-<n>``
        folder, which for a run of points is the one above the point's
        folder (`paths.run_folder_of`, the one reading); ``None`` where no
        part of the folder is one (a flat root's stage).  What every reader
        of the run's state passes to `rundir.run_state_of`
        (`running-a-job.md` § 4.2); two readers passed nothing until
        2026-10-18, so a queued point read *not launched* and a point the
        scheduler cut read *running* for ever."""
        from .paths import run_folder_of
        return run_folder_of(self.folder)

    @property
    def names(self) -> "_rf.RunNames":
        """The names of this run's stage's files, in its calculation's
        shape -- the one composer every writer names them with
        (`runfiles.RunNames`); refused for a run with no stage."""
        return _rf.RunNames.of(self.label, self.stage,
                               self.place.task.shape, trial=self.trial)

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

    def on_label(self, role: str) -> Optional[Path]:
        """This run's file in ``role`` named on its label alone, with no
        stage token -- as an engine names its own (SIESTA's ``<label>.FC``,
        ``.ORB_INDX``) and a run its result (``<label>.spectra.json``) -- in
        its own folder, or ``None``."""
        hits = _rf.find_on_label(self.folder, self.label, role)
        return hits[-1] if hits else None

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
    def prepared(self) -> Path:
        """THE FOLDER PREP WROTE THIS RUN'S FILES IN -- its deck, the
        `.validation.txt` beside it, its run script (`project-layout.md`
        § 5.2) -- which launch copies into the run: the run's folder with
        its run level taken out.  A stage's ``run-<n>/`` is the stage's
        folder, a bias point's ``run-<n>/f001/v0/`` the stage's
        ``f001/v0/``; the flat shape has no run level, and a benchmark
        trial's deck is rendered into its run (`materialize.trial_work_dir`),
        so for both the run's own folder is it."""
        if self.trial:
            return self.folder
        try:
            parts = self.folder.resolve().relative_to(
                self.place.root.resolve()).parts
        except ValueError:
            return self.folder
        return self.place.root.joinpath(
            *(p for p in parts if attempt_index(p) is None))

    @property
    def stage_deck(self) -> Optional[Path]:
        """The stage's own deck, the copy prep writes (`project-layout.md`
        § 5.2), in :attr:`prepared` -- the stage's folder in the hierarchy,
        a bias point's within it, the run's own in the flat shape, where one
        folder is both."""
        deck = self.deck
        if deck is None or self.stage is None:
            return deck
        return self.prepared / deck.name

    @property
    def report(self) -> Optional[Path]:
        """What the generator checked before it wrote this run's deck -- the
        `.validation.txt` prep writes beside it, in :attr:`prepared`
        (`runfiles.WRITTEN`) -- or ``None``: also when prep has written
        another deck there since this run's (a re-prep), whose report the
        file now is, when that one is another calculation
        (`script_emit.same_calculation`, as the record's ``current`` asks)."""
        deck, prepared = self.deck, self.stage_deck
        if deck is None or prepared is None or not prepared.is_file():
            return None
        if prepared.resolve() != deck.resolve():
            from molbuilder.script_emit import same_calculation
            # Read as the record reads a deck (`record._read`): one file
            # that will not decode costs its own row, never the ladder's.
            if not same_calculation(
                    deck.read_text(encoding="utf-8", errors="replace"),
                    prepared.read_text(encoding="utf-8", errors="replace")):
                return None
        hits = _rf.find(self.prepared, self.label, role=".validation.txt",
                        stage=self.stage)
        return hits[-1][0] if hits else None


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
    label, named, trial = _position(folder, place)
    stage = named or stage
    run = None
    if not p.is_dir():
        rec = _rf.parse(p.name, label)
        if rec is not None and rec.stage is not None:
            stage = stage or rec.stage
            run = rec.run
    if stage is None:
        # THE RUN THE FOLDER SPEAKS FOR: where the stages share it, the
        # last run there, by its record -- its stage and its number.
        last = _last_here(folder, label)
        stage, run = last if last is not None else (speaking(folder, label),
                                                     None)
    if stage is not None and run is None:
        # THE STAGE'S NEWEST RUN, by the one rule (`runrecord.newest_run`):
        # its newest launched where its runs share the folder, its newest
        # try in a folder of the run's own.
        from molbuilder.runrecord import newest_run
        run = newest_run(folder, _rf.RunNames.of(
            label, stage, place.task.shape, trial=trial))
    return Run(place=place, folder=folder, label=label, stage=stage,
               run=run, trial=trial)


# --------------------------------------------------------------------- #
#  What a run declared about its atoms (B12)                            #
# --------------------------------------------------------------------- #

class OrderUnstated(ValueError):
    """A RUN WHOSE ORDER NOTHING STATES, refused by a door that builds from
    it (`model/overview.md` § 2.2a, I9): its deck's atom-metadata block was
    written before the order record, for a kind that reorders.  The message is
    :attr:`Declared.order_unstated`'s sentence; a door that only shows the
    run says that sentence instead, a notice
    (`workingcopy_structure.ORDER_UNSTATED`)."""


def _reorders(run: "Run") -> bool:
    """Whether ``run``'s decks are written from a sorted copy (I9) -- the one
    table's answer for its stage (`transport.sort.deck_sort_key`)."""
    from molbuilder.transport.sort import deck_sort_key
    task = run.place.task
    return deck_sort_key(task.engine, task.calculation, run.stage) is not None


@dataclass(frozen=True)
class Declared:
    """What a run of ours declared about its atoms -- its own deck's records
    (`architecture.md` § 3.2): the ``deck`` that says it, its
    ``atom_metadata`` block (labels, held atoms, annotations, the order
    record) and its ``engine_offset`` record (the axis kinds, the cell prep
    placed the atoms in); ``None`` for a record the deck does not carry.
    ``order_unstated``, for a deck written before the order was recorded,
    the sentence every door says of it (I9); ``None`` otherwise."""
    deck: Optional[Path] = None
    atom_metadata: Optional[Dict[str, Any]] = None
    engine_offset: Optional[Dict[str, Any]] = None
    order_unstated: Optional[str] = None

    def atom_metadata_for(self, n_atoms: Optional[int] = None
                          ) -> Optional[Dict[str, Any]]:
        """The atom-metadata block for a structure of ``n_atoms`` -- or
        ``None`` when it says nothing (the block's one predicate,
        ``script_emit.atom_metadata_has_something``), or was written for
        another atom count: its labels are indices, and would name the
        wrong atoms."""
        from molbuilder.script_emit import atom_metadata_has_something
        md = self.atom_metadata
        if not atom_metadata_has_something(md):
            return None
        if n_atoms is not None and md.get("n_atoms_total") != n_atoms:
            return None
        return md

    @property
    def atom_order(self):
        """THE RUN'S ORDER RECORD (`model/overview.md` § 2.2a, I5): its
        deck's, read by the order record's one reader through
        ``deck_record.order_record_of`` -- ``None`` for a run in your order,
        or a deck that states none."""
        from molbuilder.deck_record import order_record_of
        return order_record_of(self.atom_metadata)

    def require_order(self) -> "Declared":
        """This record, refused when nothing states the order its run's atoms
        are in (I9, :attr:`order_unstated`): :class:`OrderUnstated`, the
        sentence every door says.  THE ONE REFUSAL of every door that builds
        from the run (:meth:`RunView.in_your_order`) or would write its file
        as it stands (a bare ``xv2xyz``)."""
        if self.order_unstated:
            raise OrderUnstated(self.order_unstated)
        return self

    def frame(self, lattice=None) -> Optional[Dict[str, Any]]:
        """The box the engine had, for coordinates that came from it
        (`model/structure-periodicity.md` § 6.0): the ``cell`` -- the
        output's own ``lattice``, else the one the deck placed the atoms
        in -- the ``axis_kind`` the deck recorded, and the engine's
        origin, an ``engine_offset`` of 0, once a cell is known.  ``None``
        when nothing says.  Both records are this run's deck's, read in
        one place, so the box and the labels never come from two decks."""
        from .deck_record import box_of_run
        return box_of_run(self.engine_offset, lattice)


def declared(run: Run) -> Declared:
    """WHAT ``run`` DECLARED ABOUT ITS ATOMS -- read from its own deck
    (:attr:`Run.deck`) by the readers of molbuilder's blocks, never from a
    file looked for beside an output by its name (`model/parse.md` § 5.3,
    plan B12).  A run with no deck here, or one whose deck carries neither
    record, declares nothing; a block its deck carries and that cannot be
    read is refused, :class:`deck_record.DeckRecordUnreadable` naming the
    deck."""
    deck = run.deck
    if deck is None:
        return Declared()
    try:
        text = deck.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return Declared(deck=deck)
    from molbuilder.deck_record import (DeckRecordUnreadable,
                                        extract_atom_metadata,
                                        extract_engine_offset)
    # A BLOCK THAT CANNOT BE READ IS REFUSED, naming the deck -- read as a
    # deck that records nothing, the run would be shown with no labels and
    # the engine's box guessed, and nothing said.
    try:
        said = Declared(deck=deck,
                        atom_metadata=extract_atom_metadata(text),
                        engine_offset=extract_engine_offset(text))
        order = said.atom_order
    except DeckRecordUnreadable as exc:
        raise DeckRecordUnreadable(f"{deck}: {exc}") from exc
    # A DECK WRITTEN BEFORE THE RULE (I9), for a kind that reorders: known by
    # what its block states, never by the molbuilder that wrote it (the user,
    # 2026-10-15: "feature/format detection") nor by a file found beside it.
    # A block written since states `atom_order` for every structure, null in
    # your order; a deck with no block wrote no label, so its sort had
    # nothing to move.
    block = said.atom_metadata
    if (order is None and block is not None and "atom_order" not in block
            and _reorders(run)):
        # THE WAY ON, in the words a text read later says -- the run door
        # does not ask whether the stage was launched (`identity`).
        from molbuilder.identity import way_back_words
        said = Declared(
            deck=deck, atom_metadata=block, engine_offset=said.engine_offset,
            order_unstated=(
                f"{deck.name}'s atom-metadata block states no order record "
                f"-- it was written before the field -- so this run's atoms "
                f"may be in the engine's order, and nothing states it; to "
                f"have it recorded, {way_back_words(None)}"))
    return said


@dataclass(frozen=True)
class RunView:
    """What a run says about the structure it ran on -- the one view every
    door that builds a structure from a run builds (`model/parse.md` § 5.3,
    § 5b): the Results tab's load, `xv2xyz --from-run`, the transport
    citation.  ``atom_metadata``, its deck's label block for these atoms
    (:meth:`Declared.atom_metadata_for`); ``periodicity``, the box the engine
    had (:meth:`Declared.frame`); ``info``, what the run says about itself
    (`parse.dirs.run_info`: ``calculation``, ``relaxation``).  Each ``None``
    when the run says nothing of it."""
    atom_metadata: Optional[Dict[str, Any]] = None
    periodicity: Optional[Dict[str, Any]] = None
    info: Optional[Dict[str, Any]] = None
    #: What the run's deck declared (:func:`declared`), read once for the
    #: view -- its records whole, its order record, and that nothing states
    #: it (I9).
    declared: "Declared" = field(default_factory=lambda: Declared())

    @property
    def order_unstated(self) -> Optional[str]:
        """A deck written before the order was recorded: what every door
        says of it (:attr:`Declared.order_unstated`, I9)."""
        return self.declared.order_unstated

    def in_your_order(self, struct, what: str):
        """``(structure, said)``: ``struct`` -- the run's atoms as a file of
        the run states them, in its engine's order -- with what the run says
        about it (:meth:`onto`), put back in your order, and the sentence a
        door says of it (``Structure.in_your_order``, naming ``what``).  THE
        ONE WAY a door that builds from a run's file crosses back: the run
        door's geometry (:func:`steps_of`), the codec's read of an editor or
        a builder, ``xv2xyz --from-run``.  A run whose order nothing states
        is refused (:meth:`Declared.require_order`, I9)."""
        self.declared.require_order()
        return self.onto(struct).in_your_order(what)

    def structure(self, elements, positions):
        """One geometry step of the run as a ``Structure``, with what the
        run says about it (:meth:`onto`)."""
        from molbuilder.structure import Structure
        return self.onto(Structure(
            elements=[str(e) for e in elements],
            positions=[[float(v) for v in p] for p in positions]))

    def onto(self, struct):
        """``struct`` -- the run's atoms, in its engine's order, as a file
        of the run states them -- with what the run says about it: the box,
        under the structure's own field names (plan § 5q D15), when the run
        says one; the labels and the rest of its deck's ATOM-METADATA block,
        the order record among them (`model/overview.md` § 2.2a, I5); the
        ``info`` record."""
        from molbuilder.script_emit import apply_atom_metadata
        per = self.periodicity or {}
        if per.get("cell") is not None:
            # The axis kinds the deck recorded, else the cell's own default.
            struct = struct.replace(
                cell=per["cell"], engine_offset=per.get("engine_offset"),
                offset_by=per.get("offset_by"),
                axis_kind=(tuple(per["axis_kind"]) if per.get("axis_kind")
                           else None))
        else:
            struct = struct.copy()
        if self.atom_metadata:
            apply_atom_metadata(struct, self.atom_metadata)
        if self.info:
            struct.apply_info_dict(self.info)
        return struct


def view_of(run: Optional[Run], *, output=None, traj: Any = None,
            n_atoms: Optional[int] = None, lattice=None) -> RunView:
    """THE VIEW OF A RUN (:class:`RunView`) -- three sources, one question:
    *what does this run say about the structure it ran on?*  The labels and
    the axis kinds from what its deck declared (:func:`declared`, for a
    structure of ``n_atoms``), the box from ``lattice`` -- its output's -- or
    its deck's record, and what it says about itself from ``run_info`` of
    its deck and ``output``, the file read (``traj`` its parse, so it is not
    parsed again).  ``run`` ``None`` -- a file no run of ours holds -- is a
    view of the output alone: its lattice and its own record."""
    from molbuilder.parse.dirs.run_info import run_info
    said = declared(run) if run is not None else Declared()
    return RunView(atom_metadata=said.atom_metadata_for(n_atoms),
                   periodicity=said.frame(lattice),
                   info=run_info(deck=said.deck, output=output, traj=traj),
                   declared=said)


class RunStepsError(ValueError):
    """A run's output that cannot give its geometry: it does not parse, or
    it holds no geometry step.  The message names the file; the door that
    asked adds its way on."""


@dataclass(frozen=True)
class RunSteps:
    """A run's geometry, IN YOUR ORDER (:func:`steps_of`): the ``first`` and
    ``last`` geometry steps of its output -- where it started and where it
    ended -- each with what the run says about it (its labels, its box, its
    ``info``: the deck's ``calculation``, the output's ``relaxation``),
    restored through the run's own order record; and the ``lattice`` its
    last step had."""
    first: Any
    last: Any
    lattice: Optional[List[List[float]]]


def steps_of(run: Run, output) -> RunSteps:
    """THE ONE READ OF A RUN'S GEOMETRY BY A DOOR THAT BUILDS FROM IT
    (`model/overview.md` § 2.2a, I3-I5, I9): ``output``, the run's file the
    door chose, parsed by the registry; its first and last geometry steps
    through the run's view (:func:`view_of` -- its deck's labels, order
    record and box, the output's own lattice); each crossed back to your
    order (``Structure.restored``).  Asked by the force-constant prep
    (`relax`'s geometry and its relaxation record) and the transport
    citation (the relaxed junction and where it started).  Raises
    :class:`RunStepsError` for an output with no geometry to give,
    :class:`OrderUnstated` for a run whose order nothing states, and
    `deck_record.DeckRecordUnreadable` for a deck block that cannot be read
    (:func:`declared`)."""
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError
    output = Path(output)
    try:
        traj = detect(output).parse(str(output))
    except (ParseError, OSError, ValueError) as exc:
        raise RunStepsError(f"{output.name} does not read as the run's "
                            f"output: {exc}") from exc
    steps = [fr for fr in traj.frames if fr.structure is not None]
    if not steps:
        raise RunStepsError(f"{output.name} holds no geometry step: the run "
                            f"never reached its first geometry")
    first, last = steps[0], steps[-1]
    lattice = last.lattice if last.lattice is not None else traj.lattice
    lattice = (None if lattice is None
               else [[float(v) for v in row] for row in lattice])
    view = view_of(run, output=output, traj=traj,
                   n_atoms=len(last.structure.elements), lattice=lattice)
    return RunSteps(
        lattice=lattice,
        first=view.in_your_order(first.structure, output.name)[0],
        last=view.in_your_order(last.structure, output.name)[0])


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
        return _row_about(p, None, unread=place.problem)
    label, _stage, _trial = _position(folder, place)
    return _row_about(p, label)


def run_answer(path) -> Optional[Dict[str, Any]]:
    """``{state, detail, live}`` -- how the run the file at ``path`` belongs
    to is doing, the one door's answer (`parse.dirs.run_state_of`, asked
    with its launch record: one that does not read is ``unreadable``, its
    file named).  ``live`` -- `job.LIVE_STATES` -- is what a Results viewer follows
    (`web/results.md` § 4.1).  ``None`` for a file of no run of ours, as
    for an upload: nothing is asked of a folder no calculation claims
    (`model/parse.md` § 5).  ``None`` too for a file of no run YET: a
    folder where nothing has been launched and no one deck names the stage
    it is for (:func:`speaking`)."""
    from molbuilder.parse.dirs.job import LIVE_STATES
    from molbuilder.parse.dirs.rundir import run_state_of
    run = run_of(path)
    if run is None or run.stage is None:
        return None
    st = run_state_of(run.folder, run.names, run.run,
                      launched_in=run.launched_in)
    return {"state": st.state, "detail": st.detail,
            "live": st.state in LIVE_STATES}


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


def _row_about(path: Path, label: Optional[str], *,
               unread: Optional[str] = None) -> Dict[str, Any]:
    """:func:`about`'s answer for a file whose folder's label is known -- or,
    with no label, a folder no calculation claims, where a saved structure
    pair is molbuilder's (`runfiles.saved_pair_row`).  ``unread``: the folder
    IS a calculation's and its description does not read
    (`place_of`'s ``problem``) -- what the file is cannot be said, and the
    reader's words say why: ``{ours: None, unread}``, never ``ours: False``
    (`web/results.md` § 3b)."""
    if unread:
        return {"ours": None, "unread": unread}
    row = _rf.row_for(path, label) if label else None
    if row is None:
        saved = None if label else _rf.saved_pair_row(path)
        if saved is None:
            return {"ours": False}
        return {"ours": True, "what": saved[0], "writer": _in_words(saved[1]),
                "when": "saved", "kind": "structure"}
    return {"ours": True, "what": row.what, "writer": _in_words(row.writer),
            "when": row.when, "kind": row.kind}


def _openable(folder: Path, label: str, stage: Optional[str],
              calculation: Optional[str], attempts: List[str]
              ) -> Optional[str]:
    """What a viewer opens in a folder of ours: the first role the
    calculation produces (`runfiles.result_roles`) in which the folder's run
    has a file -- the run it speaks for, at its newest run index -- that a
    parser claims (`model/parse.md` § 5.2)."""
    from molbuilder.parse.dirs.rundir import claimed
    staged = {a.role: a.staged for a in _rf.ON_THE_LABEL}
    for role in _rf.result_roles(calculation):
        hits = [p for p, _rec in _rf.find(
            folder, label, role=role,
            stage=stage if staged.get(role, True) else None)]
        if not hits:
            continue
        attempts.append(f"*{role} -> {len(hits)} of this run's")
        for cand in reversed(hits):
            if claimed(str(cand)):
                attempts.append(f"*{role} -> {cand.name}")
                return str(cand)
            attempts.append(f"*{role} -> {cand.name}: no parser claims it, "
                            f"not offered")
    return None


def transmission_point(folder) -> Optional["Point"]:
    """The point a transport calculation's TRANSMISSION run folder runs at
    -- a frame, a voltage, or the one point of a rung that runs once -- or
    ``None`` for any other folder.  TBtrans writes no file a viewer reads on
    its own, so such a folder's result is its calculation's report at that
    point (`web/results.md` § 0.1, § 2): the one rule :func:`openable` and
    the ladder's run rows answer by.  Read off the folder's own tokens --
    ``<NN>_transmission/run-<n>/<point>`` (`engines/transport.md` § 2a.11)
    -- against the rung's points (`transport.stages.points_in`)."""
    from .transport.stages import Point, points_in
    d = Path(folder)
    place = place_of(d)
    if (not place.ours or place.root is None
            or place.task.calculation != "transport"):
        return None
    root = Path(place.root)
    try:
        parts = d.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None
    if (len(parts) < 2 or (parse_token(parts[0]) or (None, None))[1]
            != "transmission" or attempt_index(parts[1]) is None):
        return None
    rel = parts[2:]
    points = points_in(root / parts[0] / parts[1], place.task,
                       "transmission", base=root)
    if not points:
        return Point() if not rel else None
    return next((pt for _at, pt in points if Path(pt.rel).parts == rel),
                None)


def _the_report(place: Place, attempts: List[str]) -> str:
    """A transmission run's result: its calculation's report -- the
    description at the root, the file the root opens
    (:func:`_container_product`)."""
    attempts.append("a transmission run's result is its calculation's "
                    "report at its point -- TBtrans writes no file a viewer "
                    "reads on its own (web/results.md 0.1)")
    return str(Path(place.root) / _rf.TASK_FILE)


def openable(directory) -> Tuple[Optional[str], List[str]]:
    """``(path, trail)`` -- the file a viewer should load in ``directory``:
    the run it speaks for's result in a folder of ours -- a transmission
    run's, its calculation's report at its point (:func:`transmission_point`,
    `web/results.md` § 2) -- the calculation's product at a container's root,
    and in a folder no calculation claims the newest file of a role a run
    writes that a parser claims (`parse.dirs.openable_in`)."""
    from molbuilder.parse.dirs.rundir import openable_in
    d = Path(directory)
    place = place_of(d)
    if not place.ours:
        return openable_in(str(d))
    attempts: List[str] = [f"calculation: {place.task.calculation}"]
    label, stage, _trial = _position(d, place)
    if place.role == calcdirs.CONTAINER:
        return _container_product(d, place, label, stage, attempts), attempts
    if place.role == calcdirs.RUN and transmission_point(d) is not None:
        return _the_report(place, attempts), attempts
    if place.role == calcdirs.RUN and stage is None:
        stage = speaking(d, label)
    return _openable(d, label, stage, place.task.calculation,
                     attempts), attempts


def _container_product(folder: Path, place: Place, label: str,
                       stage: Optional[str], attempts: List[str]
                       ) -> Optional[str]:
    """What a CONTAINER opens.  A transport calculation's root opens its
    description, ``task.json`` -- the report is composed on read from the
    folder (`engines/transport.md` § 2a.12), and the description is the file
    that makes it one (`parse.sidecars.task`), as a benchmark's root opens
    its ``job-set.json``; ``<label>.transport.json`` is `summarize task`'s
    copy for the command line.  Any other container: the first result role
    the calculation produces that is present (`_openable`)."""
    from molbuilder.parse.dirs.rundir import claimed
    is_root = place.root is not None \
        and folder.resolve() == Path(place.root).resolve()
    if is_root and place.task.calculation == "transport":
        handle = folder / _rf.TASK_FILE
        if claimed(str(handle)):
            attempts.append(f"*{_rf.TASK_FILE}: a transport calculation's "
                            f"report, composed on read from this folder")
            return str(handle)
    return _openable(folder, label, stage, place.task.calculation, attempts)


def _file_row(entry: Path, label: Optional[str], place: Place
              ) -> Dict[str, Any]:
    """What one file IS, for a viewer: its role, label and stage read back
    with the run's label, the parser that claims it and what it opens as,
    its size and time, and its catalogue line (`web/results.md` § 3b)."""
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError
    # READ BACK WITH THE RUN'S LABEL, a declared role or nothing
    # (`job-contracts.md` § 2.2a): a name read without its label is not taken
    # for one of ours -- SIESTA's `fdf.<stamp>.log` is not PySCF's log.
    rec = _rf.parse(entry.name, label) if label else None
    if rec is not None and rec.role not in set(_rf.roles()):
        rec = None
    try:
        kind = detect(str(entry))
        parser, opens = kind.name, getattr(
            getattr(kind, "output", None), "__name__", None)
    except (ParseError, OSError, ValueError, LookupError):
        parser, opens = None, None
    st_ = entry.stat()
    return {"name": entry.name,
            "role": rec.role if rec is not None else None,
            "label": rec.label if rec is not None else None,
            "stage": rec.stage if rec is not None else None,
            "parser": parser, "opens": opens,
            "size": st_.st_size, "mtime": st_.st_mtime,
            "about": _row_about(entry, label, unread=place.problem)}


def folder_answer(directory) -> Dict[str, Any]:
    """WHAT A FOLDER IS AND HOLDS, for a viewer -- the directory door the
    Results tab serves (`/api/results/dir`; `model/parse.md` § 5.0,
    `web/results.md` § 2.3): ``place``, ``engine``, ``openable``,
    ``attempts``, ``status`` and ``record``, and per file its role, label,
    stage, whether a parser reads it and what it is (``files``) -- with
    ``report``, the same row for the file it opens when that lies outside
    it, and ``point``, the point a transmission run's folder opens it at.

    **What the folder IS decides what is asked of it** (`project-layout.md`
    § 1.4a).  A RUN of ours is asked everything, about the run it speaks for
    (:func:`speaking`): its result, its state, its record.  A CONTAINER has
    no state and no record, though its calculation's product opens at its
    root.  A folder no calculation claims holds no run of ours: its files
    are listed, the registry says what it can open, and none of them is
    claimed -- no label read off a deck, no state asked.
    """
    from molbuilder.parse.contract import engine_of
    from molbuilder.parse.dirs.record import run_record
    from molbuilder.parse.dirs.rundir import openable_in, run_state_of

    d = Path(directory)
    place = place_of(d)
    attempts: List[str] = []
    if place.problem:
        attempts.append(place.problem)
    status = record = point = None
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
        label, stage, _trial = _position(d, place)
        attempts.append(f"calculation: {calc}")
        if place.role == calcdirs.RUN:
            from molbuilder.runrecord import LaunchRecordError
            try:
                run = run_of(d)
                point = transmission_point(d)
                # THE ONE RULE for what a folder opens (:func:`openable`).
                chosen, trail = openable(d)
            except LaunchRecordError as exc:
                # A RECORD THAT DOES NOT READ is the state `unreadable`, its
                # file named, never a crash (`running-a-job.md` § 4.2): the
                # folder's files are still listed below.
                run, chosen, trail = None, None, []
                status = {"state": "unreadable", "detail": str(exc),
                          "last_change_at": None, "active_source": None}
                attempts.append(str(exc))
            attempts += trail[1:]
            if run is not None and run.stage is not None:
                st = run_state_of(run.folder, run.names, run.run,
                                  launched_in=run.launched_in)
                status = {"state": st.state, "detail": st.detail,
                          "last_change_at": st.last_change_at,
                          "active_source": st.active_source}
                record = run_record(run.folder, names=run.names, run=run.run,
                                    status=st,
                                    engine=engine, calculation=calc,
                                    stage_deck=run.stage_deck)
            elif run is not None:
                attempts.append("no run here yet: nothing has been launched "
                                "in this folder")
        elif place.role == calcdirs.CONTAINER:
            chosen = _container_product(d, place, label, stage, attempts)
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
    files = [_file_row(entry, label, place)
             for entry in sorted(d.iterdir(), key=lambda e: e.name)
             if entry.is_file()]
    # THE FILE IT OPENS, WHEN IT LIES OUTSIDE IT -- a transmission run's
    # calculation report (`web/results.md` § 2): its row, with its path, so
    # the menu lists it first.
    report = None
    if chosen is not None and Path(chosen).parent != d:
        report = dict(_file_row(Path(chosen), label, place), path=chosen)
    return {"place": {"role": place.role,
                      "calculation": str(place.root) if place.root else None},
            "engine": engine,
            "openable": chosen,
            "report": report,
            # THE POINT the opened report is at, for a transmission run's
            # folder (:func:`transmission_point`); ``None`` elsewhere.
            "point": point.to_wire() if point is not None else None,
            "attempts": attempts,
            "status": status,
            "record": record,
            "files": files}


__all__ = ["Place", "place_of", "speaking", "Run", "run_of", "Declared",
           "declared", "OrderUnstated", "RunView",
           "view_of",
           "RunSteps", "RunStepsError", "steps_of", "about", "run_answer",
           "transmission_point", "openable", "folder_answer"]
