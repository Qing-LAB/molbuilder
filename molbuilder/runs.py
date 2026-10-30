"""The run a file belongs to, and its files -- one door.

Contract: `execution/architecture.md` § 3 (`Run`), § 3.2 (*the run a file
belongs to, and its files*; *the description*); `execution/project-layout.md`
§ 1.4a (what a folder is), § 5 (what each file is); `model/parse.md` § 5.1
(the run a folder speaks for); `web/results.md` § 3b (the file card).

Floor 2 (`architecture.md` § 2.1): it reads the description, through its one
door (`task.of`), and composes floor 1's doors -- `calcdirs` for where a
folder is (`calcdirs.where`), `paths` and `identity` for the names a layout
gives its folders, `runfiles` for the names a run gives its files, `runrecord` and
`parse.dirs` for what a run's records and outputs say.  So nothing below it
knows a calculation, which is what `parse/` and `calcdirs` must not
(`architecture.md` § 2.1).

WHY ONE DOOR (plan B11, user 2026-10-04: *"agree to your B11"*).  The label a
run names its files on is the DESCRIPTION's -- a
benchmark trial's is the description's with its point -- and every name is
read back with it, through `runfiles`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from molbuilder import calcdirs
from molbuilder import runfiles as _rf
from molbuilder.identity import parse_token
from molbuilder.paths import trial_label

if TYPE_CHECKING:
    from molbuilder.transport.stages import Point


# --------------------------------------------------------------------- #
#  What a folder is, and the description it belongs to (B14)            #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Place:
    """What a folder is in its calculation (`project-layout.md` § 1.4a):
    ``where`` -- the ground floor's answer (`calcdirs.where`): the
    calculation's root, what the folder is, the stage, run, trial and point
    it sits in -- and the description, ``task``, read once.  A folder no
    calculation claims has neither: it holds no run of ours and is read
    alone.  ``problem`` is a description that does not read, in its
    reader's words."""
    where: calcdirs.Where = field(default_factory=calcdirs.Where)
    task: Any = None
    problem: Optional[str] = None

    @property
    def root(self) -> Optional[Path]:
        """The calculation's folder, ``None`` for a folder none claims."""
        return self.where.root

    @property
    def role(self) -> Optional[str]:
        """``container``, ``run``, ``scratch`` or ``shared``
        (`project-layout.md` § 1.4a, `calcdirs.ROLES`): the root of a
        calculation whose description reads is a
        container; below it, the folder's own `calcdir.json` says;
        ``None`` when nothing says."""
        if self.where.what == calcdirs.CALCULATION:
            return None if self.task is None else calcdirs.CONTAINER
        return self.where.role

    @property
    def ours(self) -> bool:
        """The folder belongs to a calculation whose description reads."""
        return self.task is not None


def place_of(directory) -> Place:
    """What ``directory`` is (`project-layout.md` § 1.4a): where it is
    (`calcdirs.where`), with its calculation's description.  **The
    description is read through its one door, `task.of`**
    (`architecture.md` § 3.2, A16)."""
    from molbuilder.task import of as task_of
    at = calcdirs.where(directory)
    if at.root is None:
        return Place(where=at)
    try:
        task = task_of(at.root)
    except Exception as exc:                               # noqa: BLE001
        return Place(where=at, problem=str(exc))
    return Place(where=at, task=task)


def _position(place: Place) -> Tuple[str, Optional[str], bool]:
    """The label a folder's files are named on, the stage it sits in, and
    whether it is a benchmark trial's -- read off where it is
    (`calcdirs.where`): a stage's files on the rung's label, a trial's
    relabelled with its point (`paths.trial_label`)."""
    from molbuilder.kind_table import rung_label
    at, label = place.where, place.task.label
    if at.stage is not None:
        # A LEAD'S FILES ARE ON ITS OWN STEM (`rung_label`, the one door
        # prep named them by).
        label = rung_label(label, parse_token(at.stage)[1])
    if at.trial is not None:
        label = trial_label(label, at.trial)
    return label, at.stage, at.trial is not None


# --------------------------------------------------------------------- #
#  The run, and its files (B11)                                         #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """One run of ours: the calculation it belongs to (``place``), the
    ``folder`` its files are in, the ``label`` they are named on, its
    ``stage`` token and its ``run`` -- the try: the newest its files carry,
    or the one a file named; ``None`` before it has run."""
    place: Place
    folder: Path
    label: str
    stage: str
    run: Optional[int] = None
    #: A benchmark trial's run, its deck rendered into its own folder.
    trial: bool = False

    @property
    def launched_in(self) -> Path:
        """THE FOLDER THIS RUN'S LAUNCH RECORD LIES IN -- its own ``run-<n>``
        folder, which for a point of a run of points is the one above the
        point's folder: where the folder is (`calcdirs.where`, its
        ``run_folder``; every run of ours lies in one, :func:`run_of`).
        What every reader of the run's state and record passes on
        (`rundir.run_state_of`, `record.run_record`; `running-a-job.md`
        § 4.2)."""
        return self.place.where.run_folder

    @property
    def names(self) -> "_rf.RunNames":
        """The names of this run's stage's files -- the one composer every
        writer names them with (`runfiles.RunNames`)."""
        return _rf.RunNames(self.label, self.stage)

    def files(self):
        """``(path, RunFile)`` -- every file of this run's label and stage
        here, read back through `runfiles`, sorted by try."""
        return _rf.find(self.folder, self.label, stage=self.stage)

    def file(self, role: str, run: Optional[int] = None) -> Optional[Path]:
        """This run's file in ``role`` -- at try ``run``, else the newest
        try that has one (a file with no try's number when none does)."""
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
        (`runfiles.stdout_roles`) at the run's try -- or ``None``."""
        for role in _rf.stdout_roles(self.place.task.engine):
            got = self.file(role, self.run)
            if got is not None:
                return got
        return None

    @property
    def outputs(self) -> List[Path]:
        """Every engine output of this run's stage here, newest try first
        -- by the number each carries, never a file's time."""
        roles = set(_rf.stdout_roles(self.place.task.engine))
        return [p for p, rec in reversed(self.files()) if rec.role in roles]

    @property
    def session_log(self) -> Optional[Path]:
        """The run script's session log of THIS try -- the one whose first
        section is its try (`wrapper_log.log_of_run`), never the newest by
        its stamp -- or ``None``."""
        if self.run is None:
            return None
        from molbuilder.wrapper_log import log_of_run
        return log_of_run(self.folder, self.label, self.run, self.stage)

    @property
    def prepared(self) -> Path:
        """THE FOLDER PREP WROTE THIS RUN'S FILES IN -- its deck, the
        `.validation.txt` beside it, its run script (`project-layout.md`
        § 5.2) -- which launch copies into the run.  A stage's ``run-<n>/``
        is the stage's folder; a point of a run of points, its prepared
        folder (the point's ``prepared``, by its kind's declaration,
        `PointsOf.points_in`): a bias point's ``run-<n>/f001/v0/`` the
        stage's ``f001/v0/``, a call the job plans,
        ``run-<n>/optical/f003/``, its phase's ``optical/``
        (`engines/vibration.md` § 5.11 § 2) -- each composed by the folder
        door (`calcdirs.RunPlace`, `execution/runs-of-points.md` § 2.4); a
        benchmark trial's deck is rendered into its run, ``run-0``, so its
        run's own folder is it."""
        if self.trial:
            return self.folder
        at = self.place.where
        if at.run_folder is None:
            return self.folder
        place = calcdirs.RunPlace.of(at.run_folder)
        if at.what != calcdirs.POINT:
            return place.stage_dir
        from .jobset.points import points_of
        from .runrecord import LaunchRecordError
        declared = points_of(self.place.task)
        try:
            point = (next((pt for _f, pt in declared.points_in(
                at.run_folder, self.place.task, parse_token(self.stage)[1],
                Path(self.place.root)) if pt.rel == at.point), None)
                if declared is not None else None)
        except LaunchRecordError:
            # ITS RUN'S INDEX DOES NOT READ -- an older molbuilder's: the
            # stage's folder, the run's row saying why
            # (`runstatus._sweep_status`).
            point = None
        return (place.prepared(point.prepared) if point is not None
                else place.stage_dir)

    @property
    def stage_deck(self) -> Optional[Path]:
        """The stage's own deck, the copy prep writes (`project-layout.md`
        § 5.2), in :attr:`prepared` -- the stage's folder, a bias point's
        within it, a trial's run's own."""
        deck = self.deck
        if deck is None:
            return deck
        return self.prepared / deck.name

    @property
    def current(self) -> Optional[bool]:
        """WHETHER THIS RUN'S DECK IS STILL ITS STAGE'S -- the one answer
        the record's ``current`` and :attr:`report` give
        (`web/results.md` § 3a): the same calculation as the deck prep
        wrote (:attr:`stage_deck`, `deck_record.same_calculation`).  The
        stage's deck is read with the files beside THIS run's deck: what a
        deck reads from a file beside it -- a call's coordinates, which its
        job writes by design (`engines/vibration.md` § 5.11 § 1, § 4) -- is
        the run's, so only what prep fixes is compared.  ``None`` with
        nothing to compare: no deck, none prep wrote, or the run's deck is
        that one file (a benchmark trial's)."""
        deck, prepared = self.deck, self.stage_deck
        if (deck is None or prepared is None or not prepared.is_file()
                or prepared.resolve() == deck.resolve()):
            return None
        from molbuilder.deck_record import same_calculation
        from molbuilder.parse.fdf import inline_includes, read_deck
        stage_text = inline_includes(
            prepared.read_text(encoding="utf-8-sig", errors="replace"),
            beside=deck.parent)
        return not same_calculation(self.place.task.engine, read_deck(deck),
                                    stage_text)

    @property
    def report(self) -> Optional[Path]:
        """What the generator checked before it wrote this run's deck -- the
        `.validation.txt` prep writes beside it, in :attr:`prepared`
        (`runfiles.WRITTEN`) -- or ``None``: also when prep has written
        another calculation's deck there since this run's (a re-prep,
        :attr:`current`), whose report the file now is."""
        if self.deck is None or self.current is False:
            return None
        hits = _rf.find(self.prepared, self.label, role=".validation.txt",
                        stage=self.stage)
        return hits[-1][0] if hits else None


def run_of(path) -> Optional[Run]:
    """THE RUN ``path`` BELONGS TO -- a file of a run, or a run's folder --
    or ``None``: a container, a folder no calculation claims, or one whose
    description does not read (`architecture.md` § 3.2).

    The label is the description's (a trial's, `paths.trial_label`); the
    stage is the one the folder's path names (`calcdirs.where`) -- a folder
    stamped a run that lies in no stage's ``run-<n>/`` is no run of ours;
    the try is the file's own, else the run's newest."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    place = place_of(folder)
    if (not place.ours or place.role != calcdirs.RUN
            or place.where.run_folder is None):
        return None
    label, stage, trial = _position(place)
    run = None
    if not p.is_dir():
        rec = _rf.parse(p.name, label)
        if rec is not None and rec.stage is not None:
            run = rec.run
    if run is None:
        # THE RUN'S NEWEST TRY, by the one rule (`runrecord.newest_run`).
        from molbuilder.runrecord import newest_run
        run = newest_run(folder, _rf.RunNames(label, stage))
    return Run(place=place, folder=folder, label=label, stage=stage,
               run=run, trial=trial)


def on_label_at(folder, role: str) -> Optional[Path]:
    """THE FILE IN ``role`` NAMED ON THE LABEL ALONE, in ``folder`` -- a
    run's own folder (:func:`run_of`, `Run.on_label`), or the root of a run
    of points (a container at a stage's attempt, `calcdirs.where`), where
    the run's own steps write its result (`engines/vibration.md` § 5.11
    § 2: the spectrum at the run's root); ``None`` for a folder that is
    neither, or one that holds no such file.  The one door a reader of a
    stage's result takes, whether it runs once or per point."""
    folder = Path(folder)
    run = run_of(folder)
    if run is not None:
        return run.on_label(role)
    place = place_of(folder)
    if (not place.ours or place.role != calcdirs.CONTAINER
            or place.where.what != calcdirs.ATTEMPT):
        return None
    label, _stage, _trial = _position(place)
    hits = _rf.find_on_label(folder, label, role)
    return hits[-1] if hits else None


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


def _reorders(task, stage: Optional[str], kind: Optional[str] = None
              ) -> bool:
    """Whether a stage's decks are written from a sorted copy (I9) -- the one
    table's answer for it (`transport.sort.deck_sort_key`), for ``kind``
    where the deck is another kind's (the cited relaxation's deck) and the
    calculation's otherwise."""
    from molbuilder.transport.sort import deck_sort_key
    return deck_sort_key(task.engine, kind or task.calculation,
                         stage) is not None


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
        ``deck_record.order_record_of`` -- ``None`` for a run in the user order,
        or a deck that states none."""
        from molbuilder.deck_record import order_record_of
        return order_record_of(self.atom_metadata)

    def require_order(self) -> "Declared":
        """This record, refused when nothing states the order its run's atoms
        are in (I9, :attr:`order_unstated`): :class:`OrderUnstated`, the
        sentence every door says.  THE ONE REFUSAL of every door that builds
        from the run (:meth:`RunView.in_user_order`) or would write its file
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
    plan B12): :func:`declared_of_deck` of its deck, for its stage."""
    return declared_of_deck(run.deck, run.place.task, run.stage)


def declared_of_deck(deck: Optional[Path], task, stage: Optional[str], *,
                     kind: Optional[str] = None) -> Declared:
    """WHAT A DECK OF ``task``'s DECLARES ABOUT ITS ATOMS, for ``stage``
    (``None`` at the calculation's root) -- a run's (:func:`declared`), a
    stage's prepared deck, the cited deck at a transport calculation's root
    (`model/overview.md` § 2.2b), whose ``kind`` -- a relaxation's -- is the
    one its order is judged by.  No deck, or one that carries neither
    record, declares nothing; a deck that does not read, or a block it
    carries that cannot be read, is refused,
    :class:`deck_record.DeckRecordUnreadable` naming the deck -- read as
    one that declares nothing, a sorted run would be shown unmarked."""
    from molbuilder.deck_record import DeckRecordUnreadable
    if deck is None:
        return Declared()
    from molbuilder.parse.fdf import read_deck
    try:
        text = read_deck(deck)
    except OSError as exc:
        raise DeckRecordUnreadable(f"{deck}: does not read ({exc})") from exc
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
    # the user order; a deck with no block wrote no label, so its sort had
    # nothing to move.
    block = said.atom_metadata
    if (order is None and block is not None and "atom_order" not in block
            and _reorders(task, stage, kind)):
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


def atom_index_map_of(run: Run):
    """THE ATOM INDEX MAP OF A RUN'S FILES (`model/overview.md` § 2.2b, F2,
    F3) -- every file of ``run`` that states none, an engine's and an older
    one of molbuilder's, is listed by engine index and answered here, from
    its deck (I5): both columns from the deck's order record, the identity
    where it states none; the engine column alone for a deck of a structure
    of its own (`transport.sort.deck_own_structure` -- an electrode cell).
    A deck written before the order was recorded is refused by name (I9);
    one that states no atom count names it."""
    return deck_atom_index_map(declared(run), run.place.task, run.stage)


def deck_atom_index_map(said: Declared, task, stage: Optional[str]):
    """THE ATOM INDEX MAP A DECK STATES (`model/overview.md` § 2.2b, F2,
    F3) -- what ``said`` (:func:`declared_of_deck`) holds, for ``task``'s
    ``stage``: both columns from its order record, the identity where it
    states none; the engine column alone for a stage of a structure of its
    own.  Its run's files, and the deck itself wherever it lies."""
    from molbuilder.atom_permutation import AtomIndexMap
    from molbuilder.transport.sort import deck_own_structure
    said = said.require_order()
    record = said.atom_order
    n = (said.atom_metadata or {}).get("n_atoms_total")
    if n is None and record is not None:
        n = record.n_atoms
    if n is None and said.deck is not None and said.deck.suffix == ".fdf":
        # A DECK WITH NOTHING ELSE TO SAY -- no block -- states its count in
        # the engine's own keyword, read through the one fdf reader.
        from molbuilder.parse.fdf import read_deck, read_fdf
        got = read_fdf(read_deck(said.deck))[0].get("numberofatoms")
        n = int(got[0]) if got else None
    if n is None:
        raise ValueError(f"{said.deck.name if said.deck else 'the run'} "
                         f"states no atom count -- a deck written before "
                         f"2.32.0 with no label, no named value and no "
                         f"order record -- so its atoms' indices cannot be "
                         f"stated")
    own = deck_own_structure(task.engine, task.calculation, stage)
    atoms = (AtomIndexMap.of_own_structure(int(n), own) if own
             else AtomIndexMap.of_engine_record(record, int(n)))
    from dataclasses import replace
    return replace(atoms, stated_by=str(said.deck or ""))


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

    def in_user_order(self, struct, what: str):
        """``(structure, said)``: ``struct`` -- the run's atoms as a file of
        the run states them, in its engine's order -- with what the run says
        about it (:meth:`onto`), put back in the user order, and the sentence a
        door says of it (``Structure.in_user_order``, naming ``what``).  THE
        ONE WAY a door that builds from a run's file crosses back: the run
        door's geometry (:func:`steps_of`), the codec's read of an editor or
        a builder, ``xv2xyz --from-run``.  A run whose order nothing states
        is refused (:meth:`Declared.require_order`, I9)."""
        self.declared.require_order()
        return self.onto(struct).in_user_order(what)

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
        ``info`` record; and, for a run whose deck states no order (I9),
        the view's mark (``Structure.order_unstated``), so a file written
        from it says so (§ 2.2b, F4)."""
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
        if self.order_unstated:
            struct.order_unstated = True
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
    """A run's geometry, IN THE USER ORDER (:func:`steps_of`): the ``first`` and
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
    record and box, the output's own lattice); each crossed back to the user
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
        first=view.in_user_order(first.structure, output.name)[0],
        last=view.in_user_order(last.structure, output.name)[0])


def saved_pair_row(path) -> Optional["tuple[str, str]"]:
    """``(what, writer)`` for one half of a saved structure pair -- a
    structure file with its sidecar beside it, or that sidecar with its
    structure -- or ``None``.  The pairing is the codec's
    (`sidecars.molstruct.sidecar_path_for`); a structure it opens is `.xyz`
    or `.pdb`."""
    from molbuilder.sidecars.molstruct import (GEOMETRY_SUFFIXES,
                                               geometry_for, is_sidecar,
                                               sidecar_path_for)
    p = Path(path)
    if is_sidecar(p):
        if geometry_for(p) is not None:
            return _rf.SAVED_PAIR_WHAT["sidecar"], _rf.SAVED_PAIR_WRITER
        return None
    if p.suffix in GEOMETRY_SUFFIXES and sidecar_path_for(p).is_file():
        return _rf.SAVED_PAIR_WHAT["structure"], _rf.SAVED_PAIR_WRITER
    return None


def about(path) -> Dict[str, Any]:
    """WHAT A FILE IS -- its row of the catalogue, read back with its run's
    label (`runfiles.row_for`): ``{ours: True, what, writer, when, kind}``;
    or ``{ours: False}``, a file molbuilder does not write -- the engine's,
    SLURM's own, a person's -- and every file of a folder no calculation
    claims (`web/results.md` § 3b, `job-contracts.md` § 2.2)."""
    p = Path(path)
    place = place_of(p.parent)
    return _row_about(p, _label_of(p, place), unread=place.problem)


def _labels_at(place: Place) -> List[str]:
    """The labels a folder's files are named on: its run's (:func:`
    _position`) -- and AT THE ROOT, where each rung's prep log lies
    (`prep.Resolved.log_path`), every rung's, a lead's on its own stem
    (`rung_label`, the one rule prep names them by).  None for a folder no
    calculation claims."""
    if not place.ours:
        return []
    label, stage, _trial = _position(place)
    if stage is not None:
        return [label]
    from molbuilder.kind_table import rung_label
    return list(dict.fromkeys([rung_label(label, st.name)
                               for st in place.task.stages] + [label]))


def _label_of(path: Path, place: Place) -> Optional[str]:
    """The label ``path`` is named on, of those its folder's files are
    (:func:`_labels_at`): the first the catalogue reads it back with
    (`runfiles.row_for`), else the folder's run's; ``None`` for a folder no
    calculation claims."""
    if not place.ours:
        return None
    for named in _labels_at(place):
        if _rf.row_for(path, named) is not None:
            return named
    return _position(place)[0]


def _copy_of_its_machine(base):
    """The calculation's copy of its machine's record, through its door
    (`record.calculation_environment`) -- ``None`` when there is none or it
    does not read here: an answer about a run reads on without it."""
    from molbuilder.scheduler.record import (UnknownTarget,
                                             calculation_environment)
    try:
        return calculation_environment(base)
    except UnknownTarget:
        return None


def concluded(folder, names, *, base, run=None, launched_in=None):
    """THE STATE OF A RUN, ASKED ONCE OF ITS SCHEDULER AND WRITTEN ONCE
    (`running-a-job.md` § 4.2, *Asked once, written once*) -- the run
    door's answer (`rundir.run_state_of`), and for a sent run that still
    reads live, where this machine sent its job -- its address on the jobs
    list, asked on the cluster it was sent to (`joblist.address_of`,
    `asked_here`: the one rule `molbuilder jobs` asks by) -- the scheduler
    asked whether it still lists the job (the kind's `listed` query): one
    it no longer lists, or lists ended, is written into the run's launch
    record (`runrecord.record_ended`) and the state read again, ``failed``
    from then on wherever the folder goes.  Asked only
    while the record holds no answer.  ``base`` is the calculation's
    folder (the record copy's); ``run`` and ``launched_in`` as
    `run_state_of` takes them, the folder the record lies in settled once
    here for both."""
    from molbuilder.parse.dirs.job import LIVE_STATES
    from molbuilder.parse.dirs.rundir import run_state_of
    from molbuilder.runrecord import (LaunchRecordError, launch_record,
                                      record_ended)
    where = launched_in if launched_in is not None else folder
    st = run_state_of(folder, names, run, launched_in=where)
    if st.state not in LIVE_STATES and not st.by_wall:
        return st
    try:
        launch = launch_record(where)
    except LaunchRecordError:
        return st
    if (not launch or launch.get("mode") != "submit"
            or not launch.get("job_id") or launch.get("ended")):
        return st
    from molbuilder.joblist import address_of, asked_lately
    job = address_of(launch["job_id"], base)
    got = asked_lately(job, "listed") if job is not None else None
    if got is None:
        return st
    going, said = got
    if going:
        if st.by_wall:
            # REQUEUED (a node lost; SLURM's JobRequeue): the job keeps its
            # id and waits again with its first start's files in the
            # folder, past the wall rule's bound -- the scheduler's word
            # outranks the folder's here.
            return replace(
                st, state="running", by_wall=False,
                detail=f"the scheduler lists it: {said}" if said
                else "the scheduler lists it")
        return st
    try:
        record_ended(where, asked_on=job.cluster, said=said)
    except OSError as exc:
        # NOT WRITTEN -- a full disk, a folder not ours: said, the run read
        # as its folder says until a look can write it.
        return replace(st, detail=(
            f"{st.detail} -- the scheduler no longer lists job "
            f"{launch['job_id']}, and that could not be written: {exc}"))
    return run_state_of(folder, names, run, launched_in=where)


def state_of(run: "Run"):
    """:func:`concluded` for a run handle (`run_of`)."""
    return concluded(run.folder, run.names, base=run.place.root, run=run.run,
                     launched_in=run.launched_in)


def record_of(run: "Run", *, status=None, fields=None):
    """THE RUN'S RECORD (`model/parse.md` § 5d) -- the one place a run handle
    (:func:`run_of`) is read with its state (:func:`state_of`, or the one a
    caller already holds) and its calculation's engine, kind and stage deck:
    the Results tab's folder answer and the memory a stage needs
    (`jobset.memory`) ask it alike.  ``fields`` -- the record's paths a
    caller asks for, the rest left unread (`parse.dirs.record.run_record`)."""
    from molbuilder.parse.dirs.record import run_record
    task = run.place.task
    return run_record(run.folder, names=run.names, run=run.run,
                      status=status if status is not None else state_of(run),
                      engine=task.engine, calculation=task.calculation,
                      current=run.current, fields=fields,
                      launched_in=run.launched_in)


def standing_of(launch, base) -> Optional[Dict[str, Any]]:
    """A QUEUED RUN'S STANDING, as its scheduler states it here
    (`running-a-job.md` § 4.2; `scheduler.queue.standing`): ``{asked_on,
    listed, state, reason, start, priority, say}`` -- ``say`` the one line
    a row shows -- or ``{asked_on: None, say: "ask on <machine>"}`` where
    the scheduler cannot be asked from this server: it is asked only where
    this machine sent the job, by the one rule `molbuilder jobs` asks
    (`joblist.address_of`, `asked_here`); anywhere else the row says where
    to ask.  ``None`` for a run
    with no job id (a run here).  ``launch`` is the run's launch record,
    ``base`` its calculation's folder (the record copy's)."""
    jid = (launch or {}).get("job_id")
    if not jid:
        return None
    from molbuilder.joblist import Asked, address_of, asked_here, asked_lately
    from molbuilder.scheduler.record import LOCAL_TARGET
    job = address_of(jid, base)
    got = asked_lately(job, "standing") if job is not None else None
    if got is None:
        if job is not None:
            # SENT FROM HERE, NOT ASKED: why, in the jobs list's words.
            return {"asked_on": None, "say": asked_here(job, Asked())[1]
                    or "the scheduler cannot be asked here"}
        rec = _copy_of_its_machine(base)
        machine = rec.machine if rec is not None else None
        return {"asked_on": None, "say": (
            f"ask on {machine}" if machine and machine != LOCAL_TARGET
            else "ask on the machine that sent it")}
    asked_on = job.cluster
    if not got.get("listed"):
        return {"asked_on": asked_on, "listed": False,
                "say": "the scheduler here no longer lists it"}
    words = [got["state"] + (f" ({got['reason']})" if got.get("reason")
                             else "")]
    if got.get("start"):
        words.append(f"expected to start {got['start']}")
    else:
        words.append("no start estimate yet")
    if got.get("priority"):
        words.append(f"priority {got['priority']}")
    return {"asked_on": asked_on, "listed": True, "state": got["state"],
            "reason": got.get("reason"), "start": got.get("start"),
            "priority": got.get("priority"), "say": " · ".join(words)}


def run_answer(path) -> Optional[Dict[str, Any]]:
    """``{state, detail, live}`` -- how the run the file at ``path`` belongs
    to is doing, the one door's answer (`parse.dirs.run_state_of`, asked
    with its launch record: one that does not read is ``unreadable``, its
    file named).  ``live`` -- `job.LIVE_STATES` -- is what a Results viewer follows
    (`web/results.md` § 4.1).  ``None`` for a file of no run of ours, as
    for an upload: nothing is asked of a folder no calculation claims
    (`model/parse.md` § 5)."""
    from molbuilder.parse.dirs.job import LIVE_STATES
    run = run_of(path)
    if run is None:
        # A FILE AT A RUN OF POINTS' ROOT -- the result its own steps wrote
        # there (`engines/vibration.md` § 5.11 § 7): the run's state is its
        # points'.
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        got = points_root_state(place_of(folder), folder)
        if got is None:
            return None
        return {"state": got["state"], "detail": got["detail"],
                "live": got["state"] in LIVE_STATES}
    st = _as_its_point(run, state_of(run))
    return {"state": st.state, "detail": st.detail,
            "live": st.state in LIVE_STATES}


def _as_its_point(run: Run, st):
    """A POINT OF A RUN OF POINTS IS READ AS ITS RUN'S POINTS ARE
    (`engines/transport.md` § 2a.11; `engines/vibration.md` § 5.11 § 7):
    one still to come by its own folder -- ``pending`` or ``queued`` -- or
    ``failed`` by its job's end alone is its row of the run's points
    (`continuation.points_state`), so a point the ended walk never began
    reads *not run* (`continuation.never_reached`), never still to come and
    never failed.
    ``st``, the run door's own answer, for any other run and any other
    state."""
    at = run.place.where
    if (at.what != calcdirs.POINT
            or st.state not in ("pending", "queued", "failed")):
        return st
    from molbuilder.jobset.continuation import points_state
    from molbuilder.jobset.points import points_of
    from molbuilder.runrecord import LaunchRecordError, launch_record
    if points_of(run.place.task) is None:
        return st
    try:
        launch = launch_record(run.launched_in)
        if launch is None:
            # A RUN PREP OPENED AND NOBODY LAUNCHED: its points are still
            # to come, never reached by a walk that never ran.
            return st
        _state, _detail, rows = points_state(
            run.launched_in, run.names, run.place.task,
            parse_token(run.stage)[1], launch=launch, base=run.place.root)
    except LaunchRecordError:
        # ITS RUN'S RECORDS DO NOT READ: the point answers by its own
        # folder, the run's row saying why (`runstatus._sweep_status`).
        return st
    row = next((r for r in rows if r["rel"] == at.point), None)
    if row is None or not row.get("not_run"):
        return st
    from molbuilder.runpoints import NOT_RUN
    return replace(st, state=NOT_RUN, detail=row["not_run"])


def points_root_state(place: Place, folder) -> Optional[Dict[str, Any]]:
    """A RUN OF POINTS' ROOT -- a container at a stage's attempt
    (`calcdirs.where`), of a kind whose rung runs per point
    (`jobset.points`) -- answers the run's state in the folder answer's
    shape, by the one rule for a run of points (`continuation.
    stage_run_state`) -- one whose records do not read is ``unreadable``,
    its file named; ``None`` for any other place.  The calculation is
    opened once for it (`calculation.open`, A16)."""
    if not place.ours or place.role != calcdirs.CONTAINER:
        return None
    at = place.where
    if at.what != calcdirs.ATTEMPT or at.stage is None or at.trial is not None:
        return None
    from molbuilder.jobset.points import points_of
    if points_of(place.task) is None:
        return None
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.continuation import stage_run_state
    from molbuilder.jobset.points import rung_points_of
    from molbuilder.runrecord import LaunchRecordError
    calc = open_calculation(place.root)
    stage = parse_token(at.stage)[1]
    if not rung_points_of(calc, stage):
        return None
    try:
        state, detail = stage_run_state(calc, stage, Path(folder))
    except LaunchRecordError as exc:
        # ITS RECORDS DO NOT READ: ``unreadable``, the file named
        # (`running-a-job.md` § 4.2).
        state, detail = "unreadable", str(exc)
    return {"state": state, "detail": detail,
            "last_change_at": None, "active_source": None}


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
    pair is molbuilder's (:func:`saved_pair_row`).  ``unread``: the folder
    IS a calculation's and its description does not read
    (`place_of`'s ``problem``) -- what the file is cannot be said, and the
    reader's words say why: ``{ours: None, unread}``, never ``ours: False``
    (`web/results.md` § 3b)."""
    if unread:
        return {"ours": None, "unread": unread}
    row = _rf.row_for(path, label) if label else None
    if row is None:
        saved = None if label else saved_pair_row(path)
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
    has a file -- the run it speaks for, at its newest try -- that a parser
    claims (`model/parse.md` § 5.2)."""
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
    """The point a run folder of a rung whose role opens its calculation's
    report (the kind table's role ``opens`` -- a transport calculation's
    transmission) runs at -- a frame, a voltage, or the one point of a rung
    that runs once -- or ``None`` for any other folder.  TBtrans writes no
    file a viewer reads on its own, so such a folder's result is its
    calculation's report at that point (`web/results.md` § 0.1, § 2): the
    one rule :func:`openable` and the ladder's run rows answer by.  Read
    off where the folder is (`calcdirs.where`) -- a stage's run,
    ``<NN>_transmission/run-<n>/``, or a point's folder in it
    (`engines/transport.md` § 2a.11), never a benchmark trial's -- against
    the points its run holds (`transport.stages.points_in`)."""
    from .kind_table import kind
    from .transport.stages import Point, points_in
    place = place_of(folder)
    if not place.ours:
        return None
    at = place.where
    if (at.what not in (calcdirs.ATTEMPT, calcdirs.POINT)
            or at.run_folder is None or at.trial is not None):
        return None
    # A RUNG WHOSE ROLE OPENS THE REPORT (the kind table's role ``opens``):
    # its stages are its rungs, each its role's name.
    rung = parse_token(at.stage)[1]
    role = kind(place.task.calculation).role(rung)
    if role is None or role.opens != "report":
        return None
    rel = Path(at.point).parts if at.what == calcdirs.POINT else ()
    points = points_in(at.run_folder, place.task, rung,
                       base=Path(place.root))
    if not points:
        return Point() if not rel else None
    return next((pt for _folder, pt in points
                 if Path(pt.rel).parts == rel), None)


def _the_report(place: Place, attempts: List[str]) -> str:
    """A transmission run's result: its calculation's report -- the
    description at the root, the file the root opens
    (:func:`_container_product`)."""
    attempts.append("a transmission run's result is its calculation's "
                    "report at its point -- TBtrans writes no file a viewer "
                    "reads on its own (web/results.md 0.1)")
    return str(calcdirs.description_path(place.root))


#: What a folder stamped a run that lies in no stage's ``run-<n>/`` is
#: said to be (:func:`openable`).
NOT_IN_A_RUN = ("this folder says it is a run, but it lies in no run-<n>/ of "
                "a stage: no run of molbuilder's, so only what is in it is "
                "shown (project-layout.md § 1.4a)")


def openable(directory) -> Tuple[Optional[str], List[str]]:
    """``(path, trail)`` -- the file a viewer should load in ``directory``:
    the run it speaks for's result in a folder of ours -- a transmission
    run's, its calculation's report at its point (:func:`transmission_point`,
    `web/results.md` § 2) -- the calculation's product at a container's root,
    and in a folder no calculation claims the newest file of a role a run
    writes that a parser claims (`parse.dirs.openable_in`)."""
    d = Path(directory)
    place = place_of(d)
    return _opens(d, place, transmission_point(d)
                  if place.ours and place.role == calcdirs.RUN else None)


def _opens(d: Path, place: Place, point) -> Tuple[Optional[str], List[str]]:
    """:func:`openable` for a folder whose place and transmission point
    (:func:`transmission_point`, ``None`` for any but a run folder) the
    caller already holds -- the folder answer reads each once."""
    from molbuilder.parse.dirs.rundir import openable_in
    if not place.ours:
        return openable_in(str(d))
    attempts: List[str] = [f"calculation: {place.task.calculation}"]
    if place.role == calcdirs.RUN and place.where.run_folder is None:
        # A FOLDER STAMPED A RUN THAT LIES IN NO STAGE'S ``run-<n>/`` is no
        # run of ours (:func:`run_of`) -- said, and read alone
        # (`project-layout.md` § 1.4a: the missing relation is stated).
        attempts.append(NOT_IN_A_RUN)
        chosen, trail = openable_in(str(d))
        return chosen, attempts + trail
    label, stage, _trial = _position(place)
    if place.role == calcdirs.CONTAINER:
        return _container_product(d, place, label, stage, attempts), attempts
    if place.role == calcdirs.RUN and point is not None:
        return _the_report(place, attempts), attempts
    return _openable(d, label, stage, place.task.calculation,
                     attempts), attempts


def _container_product(folder: Path, place: Place, label: str,
                       stage: Optional[str], attempts: List[str]
                       ) -> Optional[str]:
    """What a CONTAINER opens (`web/results.md` § 0.1).  The root of a kind
    whose root opens its description (the kind table's ``root_opens`` -- a
    transport calculation's) opens ``task.json`` -- the report is composed
    on read from the folder (`engines/transport.md` § 2a.12), and the
    description is the file that makes it one (`task_reader`);
    ``<label>.transport.json`` is `summarize task`'s copy for the command
    line.  A benchmark's folder opens its sweep's ``job-set.json``, the
    bench summary's handle -- the registry claims a job set only as a
    sweep's (`jobset.sweep_reader`), so a ladder's is never opened.  Any
    other container: the first result role the calculation produces that is
    present (`_openable`)."""
    from molbuilder.kind_table import kind
    from molbuilder.parse.dirs.rundir import claimed
    rec = kind(place.task.calculation)
    if (place.where.what == calcdirs.CALCULATION
            and rec.root_opens == "description"):
        handle = calcdirs.description_path(folder)
        if claimed(str(handle)):
            attempts.append(f"*{_rf.TASK_FILE}: {rec.with_article}'s "
                            f"report, composed on read from this folder")
            return str(handle)
    sweep = folder / _rf.JOBSET_FILE
    if claimed(str(sweep)):
        attempts.append(f"*{_rf.JOBSET_FILE}: a benchmark's sweep, its "
                        f"trials compared")
        return str(sweep)
    return _openable(folder, label, stage, place.task.calculation, attempts)


def _file_row(entry: Path, place: Place) -> Dict[str, Any]:
    """What one file IS, for a viewer: its role, label and stage read back
    with the label it is named on (:func:`_label_of`), the parser that
    claims it and what it opens as, its size and time, and its catalogue
    line (`web/results.md` § 3b) -- what :func:`about` answers."""
    from molbuilder.parse import detect
    from molbuilder.parse.errors import ParseError
    # READ BACK WITH ITS LABEL, a declared role or nothing
    # (`job-contracts.md` § 2.2a): a name read without its label is not taken
    # for one of ours -- SIESTA's `fdf.<stamp>.log` is not PySCF's log.
    label = _label_of(entry, place)
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
    § 1.4a).  A RUN of ours is asked everything, about its own newest try:
    its result, its state, its record.  A CONTAINER has
    no state and no record, though its calculation's product opens at its
    root.  A folder no calculation claims holds no run of ours: its files
    are listed, the registry says what it can open, and none of them is
    claimed -- no label read off a deck, no state asked.
    """
    from molbuilder.parse.contract import engine_of
    from molbuilder.parse.dirs.rundir import openable_in

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
        if place.root is None:
            # A CALCULATION'S FOLDER WHOSE DESCRIPTION DOES NOT READ is read
            # alone too, its reader's words said above -- never *not marked*.
            attempts.append(
                "this folder is not marked as part of a calculation, so only "
                "what is in it is shown (project-layout.md § 1.4a)")
    else:
        engine = place.task.engine
        calc = place.task.calculation
        label, stage, _trial = _position(place)
        attempts.append(f"calculation: {calc}")
        if place.role == calcdirs.RUN:
            run = run_of(d)
            point = transmission_point(d)
            # THE ONE RULE for what a folder opens (:func:`openable`), with
            # the place and the point read once.
            chosen, trail = _opens(d, place, point)
            attempts += trail[1:]
            if run is not None:
                # A LAUNCH RECORD THAT DOES NOT READ is the state
                # `unreadable`, its file named (`running-a-job.md` § 4.2);
                # a point is read as its run's points are.
                st = _as_its_point(run, state_of(run))
                status = {"state": st.state, "detail": st.detail,
                          "last_change_at": st.last_change_at,
                          "active_source": st.active_source}
                record = record_of(run, status=st)
        elif place.role == calcdirs.SCRATCH:
            # THE ENGINE'S SCRATCH OF THE RUN ABOVE -- a device run's
            # `tsgf/` (`engines/transport.md` § 2a.11): read alone, no run
            # asked of it, nothing in it opens.
            chosen = None
            attempts.append(
                "the engine's scratch of the run above (its leads' surface "
                "Green's function files, one voltage's pair at a time): "
                "nothing in it opens (engines/transport.md § 2a.11)")
        elif place.role == calcdirs.SHARED:
            # THE ENERGIES A TRANSMISSION POINT SHARES with a finished
            # calculation's point (`engines/transport.md` § 2a.16): read with
            # the point they lie in -- past every `shared/` it holds, one in
            # another -- its calculation's report at that point, which it
            # opens -- and named by what it is a copy of.
            from molbuilder.runrecord import LaunchRecordError, read_copied_from
            at = d.parent
            while calcdirs.where(at).role == calcdirs.SHARED:
                at = at.parent
            chosen, _trail = openable(at)
            point = transmission_point(at)
            try:
                said = read_copied_from(d)
            except LaunchRecordError as exc:
                said = str(exc)
            attempts.append(
                "the energies the point above shares with "
                + (said or "a point it names nowhere")
                + ", copied from it: read with that point, whose report "
                  "opens (engines/transport.md § 2a.16)")
        else:
            # A CONTAINER: a folder of a calculation of ours is a run, a run's
            # scratch or a container -- its own record says, the root a
            # container (`Place.role`); one that says nothing is no
            # calculation's (`calcdirs.where`).
            # A RUN OF POINTS' ROOT among them: no state, no record
            # (`model/parse.md` § 5.2) -- its points are below it.
            chosen = _container_product(d, place, label, stage, attempts)
            attempts.append(
                "this directory is a container, not a run -- it has no "
                "run state; its runs are the directories below it "
                "(project-layout.md § 1.4)")
    files = [_file_row(entry, place)
             for entry in sorted(d.iterdir(), key=lambda e: e.name)
             if entry.is_file()]
    # THE FILE IT OPENS, WHEN IT LIES OUTSIDE IT -- a transmission run's
    # calculation report (`web/results.md` § 2): its row, with its path, so
    # the menu lists it first.
    report = None
    if chosen is not None and Path(chosen).parent != d:
        report = dict(_file_row(Path(chosen), place_of(Path(chosen).parent)),
                      path=chosen)
    # WHERE THE FOLDER IS, whole (`calcdirs.where`): its role, its
    # calculation, what it is, and the stage, run, trial and point it sits in.
    at = place.where
    return {"place": {"role": place.role,
                      "calculation": str(place.root) if place.root else None,
                      "what": at.what, "stage": at.stage, "run": at.run,
                      "trial": at.trial, "point": at.point},
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


__all__ = ["Place", "place_of", "Run", "run_of", "Declared",
           "declared", "declared_of_deck", "deck_atom_index_map",
           "OrderUnstated", "atom_index_map_of", "RunView",
           "view_of",
           "RunSteps", "RunStepsError", "steps_of", "about", "run_answer",
           "transmission_point", "openable", "folder_answer", "state_of",
           "record_of"]
