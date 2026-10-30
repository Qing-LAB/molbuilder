"""P4a — prep COMPOSES the transport calculation from its one citation.

`engines/transport.md` § 3.1 and § 6.2: resolve the junction citation on
the machine where prep runs (strict composition, a missing
or unconcluded relaxation is a refusal naming what to run first, never a
trigger to run it); read the junction it holds — a cited run's folder
through the run door, as the Results tab reads a run (`runs.run_of`,
`runs.openable`, and the one read of a run's geometry, `runs.steps_of`, in
the user order), or a cited pair through the codec, in the user order; sort it for
TranSIESTA (`sort.sort_by`, P2); apply the lead gate;
extract the two electrode models from the sorted blocks (the wizard's
move, § 3); and record the provenance — the citation, its kind, and the
content hashes of the files it was composed from, a cited run's **own
deck** among them (the fdf that actually ran is the truth about a result;
user ruling 2026-08-28).

Pure composition: everything here reads the tree and returns objects;
the caller (prep, `jobset.gather.composed_for_prep`) owns what lands on
disk where.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..cell import entering
from ..parse.fdf import read_deck
from ..persist import sha256_of
from ..structure import (SAME_PLACE_ANG, Structure, farthest_apart,
                         frame_words)
from .sort import (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE, SortError,
                   deck_sort_key, sort_by)
from .wizard import ElectrodeModel, extract_electrode_model

if TYPE_CHECKING:                                    # pragma: no cover
    from ..runs import Run


class ComposeError(Exception):
    """A citation the composition cannot honour — the message names
    exactly what to run (or fix) first, ready to surface verbatim."""


#: The composed record's on-disk names, beside the transport
#: calculation's ``task.json`` (§ 4.1: the cited structure is COPIED in
#: with provenance, and the folder then travels like any other).
#: The catalogue's names (`runfiles.WRITTEN`, `job-contracts.md` § 2.2).
from ..runfiles import JUNCTION_FILE as JUNCTION_GEOMETRY  # noqa: E402 -- the SORTED junction (codec pair)
from ..runfiles import JUNCTION_CITED_FILE as JUNCTION_DECK  # noqa: E402 -- the attempt's own deck, verbatim
from ..runfiles import SLOT_PROVENANCE_FILE as PROVENANCE_FILE  # noqa: E402

#: The citation kinds a record names (`engines/transport.md` § 3.1).
CITATION_KINDS = ("run", "pair")

#: The composition record's schema (`execution/job-contracts.md` § 6.1),
#: stamped by its one writer (:func:`write_compose_record`).
#: ``@2``: the junction's sidecar carries its order record, and no file of it
#: lies beside (`model/overview.md` § 2.2a, I7).  What a record holds is
#: asked of the files themselves, never of this name
#: (:func:`recorded_junction`).
PROVENANCE_SCHEMA = "molbuilder/slot-provenance@2"


def record_files(kind: str) -> Tuple[str, ...]:
    """WHAT THE TRAVELLED RECORD CONSISTS OF for a citation of ``kind`` —
    one answer, read by the write and by the load, so they cannot disagree
    about whether a copy is complete: the sorted geometry and its label file
    (every frame, its order record, and a cited pair's ``info.calculation``),
    the provenance, and a cited run's deck (the electronic contract IS the
    file).

    The geometry's label file is not named literally: it is whatever
    the codec pairs with :data:`JUNCTION_GEOMETRY`, asked of the codec's
    own rule (`sidecars.molstruct.sidecar_path_for`).
    """
    from ..sidecars.molstruct import sidecar_path_for
    if kind not in CITATION_KINDS:
        raise ValueError(f"record_files: a citation is one of "
                         f"{', '.join(CITATION_KINDS)}, not {kind!r}")
    return ((JUNCTION_GEOMETRY,
             sidecar_path_for(Path(JUNCTION_GEOMETRY)).name,
             PROVENANCE_FILE)
            + ((JUNCTION_DECK,) if kind == "run" else ()))


def _unusable_cell(struct) -> Optional[str]:
    """Why this structure's cell cannot carry a junction — or ``None``.

    ONE QUESTION, BOTH KINDS.  ``Structure`` reads a zero-volume lattice
    without refusing it (§ 8.2, *"reading does not judge"*), so a pair's
    sidecar whose ``cell`` is a row of zeros loads and passes an ``is None``
    guard.

    What this guard is worth is WHERE and IN WHOSE WORDS the refusal
    lands: at the citation door, naming the cited pair -- rather than
    surviving to the deck writer to be described as a problem with a box,
    several steps from the file that holds it.

    A junction needs a real box (`science/junction-cell.md`): the transverse
    vectors set the k-mesh and the image separation, and the transport vector
    is the device length.  None of those exist in a degenerate cell.
    """
    if struct.cell is None:
        return "states no cell"
    c = np.asarray(struct.cell, dtype=float)
    if c.shape != (3, 3) or not np.all(np.isfinite(c)):
        return "states a cell that is not three finite vectors"
    # The ONE threshold, from the module that owns the question.
    from ..cell import ZERO_VOLUME_TOL
    if abs(float(np.linalg.det(c))) < ZERO_VOLUME_TOL:
        return ("states a cell with no volume (its vectors are not "
                "linearly independent)")
    return None


@dataclass(frozen=True)
class ComposedJunction:
    """Everything the transport stages render from."""
    #: the relaxed, LABELED junction in TranSIESTA's order, carrying its
    #: order record (`model/overview.md` § 2.2a)
    junction: Structure
    #: what the sort said of it on the compose that made it (`sort_by`'s
    #: notes) -- none on a record read back, which no sort made
    notes: Tuple[str, ...]
    #: the same structure before the sort (relaxed positions, source
    #: order).  ``None`` on a record loaded back from the travelled
    #: copy: the original order lives in the citation and the sorted
    #: junction's own order record, and no stage renders from it.
    relaxed: Optional[Structure]
    electrode_left: ElectrodeModel
    electrode_right: ElectrodeModel
    #: a cited run's deck TEXT, verbatim — the fdf that actually ran is the
    #: truth about a result, so the copy that travels is the file itself,
    #: re-parseable anywhere (user ruling 2026-08-28).  ``None`` for a cited
    #: pair, whose record of the run is its ``info.calculation``, carried by
    #: the junction's own sidecar.
    deck_text: Optional[str]
    #: the citation's kind · its path · content hashes · how a cited run
    #: ended and what it converged, or a cited pair's frame count — written
    #: beside the copies so a result can always say which junction built it
    provenance: Dict[str, object]
    #: THE ORDER RECORD FOUND, NOT READ (`engines/transport.md` § 1.1, *The
    #: first prep*): a recorded junction whose sidecar states none, checked
    #: atom for atom against its citation composed again in memory and given
    #: that composition's record -- which the prep that read it writes
    #: (:func:`write_compose_record`).  ``False`` for every other junction.
    order_found: bool = False



@dataclass(frozen=True)
class CitedRun:
    """The cited relaxation run (`engines/transport.md` § 3.1): the folder
    and the run it speaks for (`runs.run_of`), and how that run ended.
    Layout, names and tree position play no part (user ruling 2026-08-29);
    being a run of ours -- launched by `jobset`, with the wrapper's record
    -- does (decision 7, 2026-10-08).  What it left is read by the run
    door, as the Results tab reads it (:func:`_run_geometry`)."""
    path: Path
    run: "Run"                    # the run the folder speaks for
    #: the run record's concluded line -- ``rc=0 at <date>`` -- or ``None``
    #: for a run still going (or force-stopped: no file tells those apart);
    #: classification RECORDS that, compose refuses it (§ 3.1).
    concluded: Optional[str] = None
    #: the exit code that line states, when it concluded.
    exit_code: Optional[int] = None
    #: what the run's output says it converged -- ``"geometry yes"`` /
    #: ``"geometry NO"`` -- or ``None`` when it says neither
    #: (`RunStatus.converged_words`), for showing; and the verdict a
    #: reader decides on, ``("geometry", ok)`` (`RunStatus.verdict`).
    converged: Optional[str] = None
    verdict: Optional[Tuple[str, bool]] = None


@dataclass(frozen=True)
class CitedPair:
    """The cited structure pair (`engines/transport.md` § 3.1, § 2a.9): its
    ``.xyz`` and its sidecar, the structure they hold -- every frame, frame 0
    the base -- and ``calculation``, the record of the optimization frame 0
    came from (``info.calculation``, `parse.contract.contract_of`'s
    shape)."""
    path: Path                    # the .xyz
    sidecar: Path                 # its .molstruct.json
    structure: Structure          # every frame (`StructureCodec.load(frames=True)`)
    calculation: Dict[str, object]
    #: WHAT ITS READ SAID (`workingcopy_structure.READ_NOTICES`): a pair that
    #: states no atom order, read as listed by user index (`model/overview.md`
    #: § 2.2b, F4); one in an engine's order, put back in the user order.
    said: Tuple = ()

    @property
    def n_frames(self) -> int:
        return self.structure.n_frames


#: The citation condition, in one sentence — used verbatim by every
#: refusal so the user always learns the WHOLE condition, not just the
#: half they tripped on.
CITATION_CONDITION = (
    "a citation is a relaxation run of molbuilder's own that has finished "
    "-- its folder, launched by `jobset launch` so its run record says how "
    "it ended, its region labels in its deck's block -- or a structure pair "
    "that records the optimization it came from -- its .xyz, its "
    ".molstruct.json carrying info.calculation: one geometry, as the "
    "Results tab or `molbuilder xv2xyz --from-run` saves a finished "
    "relaxation, or a frame set, the junction moved along one vibration "
    "mode, as the Results tab's Write frame set saves it from a vibration's "
    "result (engines/transport.md 3.1).  Relax the junction through "
    "`molbuilder describe` -> `prep task` -> `launch task`, then cite that "
    "run or the pair saved from it; to move it along a vibration, open the "
    "vibration's result on the Results tab, select the mode, press Write "
    "frame set and cite the .xyz it saves -- a vibration's own folder is "
    "not cited")


def classify_citation(cited: Path) -> "CitedRun | CitedPair":
    """Classify a citation against § 3.1's condition by what the path is: a
    folder is a relaxation run of molbuilder's own (:func:`_classify_run`), a
    file a structure pair (:func:`_classify_pair`).  Raises
    :class:`ComposeError` naming what is missing.

    CLASSIFYING IS NOT COMPOSING: a run still going, or one that ended with
    an error, is RECORDED here -- describing a transport calculation ahead
    of its relaxation is legal -- and refused by :func:`compose_junction`;
    the describe doors refuse one that ended with an error first
    (:func:`refuse_failed_citation`).
    """
    cited = Path(cited)
    if cited.is_file():
        return _classify_pair(cited)
    return _classify_run(cited)


def _classify_run(cite_dir: Path) -> CitedRun:
    """A cited folder, read through the run door (`runs.run_of`,
    `execution/architecture.md` § 3.2): the run in it, its one stage; its
    launch record and how it ended, the run record's doors."""
    from molbuilder.runrecord import (LaunchRecordError, ending,
                                      launch_record)
    from .. import calcdirs
    from ..runs import place_of, run_of
    place = place_of(cite_dir)
    if place.problem:
        raise ComposeError(place.problem)
    run = run_of(cite_dir)
    if run is None:
        where = ("no calculation claims this folder"
                 if not place.ours else
                 "it is a container, not a run -- cite one of the runs below "
                 "it" if place.role == calcdirs.CONTAINER else
                 "it says it is a run, but lies in no stage's run-<n>/"
                 if place.role == calcdirs.RUN else
                 "it is a run's own scratch or shared folder, not a run -- "
                 "cite the run above it")
        raise ComposeError(
            f"{cite_dir} holds no run of molbuilder's: {where}.  "
            f"{CITATION_CONDITION}.")
    # ONE RUN, the one the run door settled (`run.run`), asked of each door:
    # its launch record -- in the folder the run door says it lies in
    # (`Run.launched_in`: a point's is its run's) -- how it ended, its state.
    try:
        launched = launch_record(run.launched_in) is not None
    except LaunchRecordError as e:
        raise ComposeError(str(e)) from e
    # HOW THE CITED RUN ENDED, asked of the one door (`runrecord.ending`,
    # `execution/architecture.md` § 3.2) about THIS run's newest try.
    end = ending(cite_dir, run.names.stem, run.run)
    if not launched and not end.concluded:
        raise ComposeError(
            f"{cite_dir} holds the run {run.names.stem}, but nothing launched "
            f"it and nothing recorded how it ended.  {CITATION_CONDITION}.")
    # WHAT IT CONVERGED, the run door's reading of its output (the same the
    # status verb prints): carried into the provenance and said by the tab,
    # so a geometry that did not converge is cited knowingly.
    from ..parse.dirs.rundir import run_state_of
    st = run_state_of(cite_dir, run.names, run.run,
                      launched_in=run.launched_in)
    # A RUN THAT RELAXED NO GEOMETRY is not what transport cites (§ 3.1): a
    # vibration run, whose last geometry is a displacement, or a transport
    # rung -- its output reports an SCF alone.
    if st.verdict is not None and st.verdict[0] != "geometry":
        raise ComposeError(
            f"{cite_dir} holds the run {run.names.stem}, which relaxed no "
            f"geometry -- its output reports a self-consistent field alone, "
            f"so its last geometry is not a relaxed junction.  "
            f"{CITATION_CONDITION}.")
    return CitedRun(path=cite_dir, run=run, concluded=end.line,
                    exit_code=end.code, converged=st.converged_words,
                    verdict=st.verdict)


def _classify_pair(path: Path) -> CitedPair:
    """A cited structure pair: every frame through the codec
    (`StructureCodec.load(frames=True)`), its ``info.calculation`` -- what
    the settings default from -- and the per-frame promise only the
    citation can check (`engines/transport.md` § 2a.9): every frame keeps
    the electrode atoms where frame 0 has them, within
    `wizard.FROZEN_TOL_ANG`, because every frame shares the leads'
    calculation.  The frames' rows are their maker's, carried with the
    structure and never read here (`model/structure.md` § 2.2f).  The
    same atoms, species and cell are the structure's own invariant, refused
    by its reader."""
    from ..sidecars.molstruct import sidecar_path_for
    from ..workingcopy_structure import StructureCodec
    sidecar = sidecar_path_for(path)
    # A PAIR IS ITS .xyz WITH ITS .molstruct.json -- an engine's own file in
    # a run, read with what its run states, is not one: cite the run.
    if not sidecar.is_file():
        raise ComposeError(
            f"{path} carries no info.calculation -- it has no .molstruct.json "
            f"beside it -- so nothing says which settings its geometry was "
            f"optimized with.  {CITATION_CONDITION}.")
    # A BUILDER'S READ (`model/overview.md` § 2.2a, I3, I8): the pair in the
    # user order -- an engine's record put back, a pair whose order nothing
    # states refused -- and what the read says kept for the doors that tell
    # a person (prep, the Transport tab).
    said: list = []
    try:
        struct = StructureCodec().load(path, frames=True, in_user_order=True,
                                       said_out=said)
    except (ValueError, OSError) as exc:
        raise ComposeError(f"{path} does not read as a structure pair "
                           f"({exc}).  {CITATION_CONDITION}.") from exc
    calc = (struct.info or {}).get("calculation")
    if not isinstance(calc, dict) or not isinstance(calc.get("contract"),
                                                    dict):
        raise ComposeError(
            f"{path} carries no info.calculation -- {sidecar.name} records no "
            f"optimization -- so nothing says which settings its geometry was "
            f"optimized with.  {CITATION_CONDITION}.")
    _frames_keep_the_leads(struct, path)
    return CitedPair(path=path, sidecar=sidecar, structure=struct,
                     calculation=calc, said=tuple(said))


def _frames_keep_the_leads(struct: Structure, path: Path) -> None:
    """Every frame's electrode atoms where frame 0 has them (§ 2a.9: the
    lead Hamiltonian is truth for every frame), measured as the lead gate
    measures an unmoved atom -- the distance, against
    `wizard.FROZEN_TOL_ANG` -- refused naming each frame and its atoms, in
    the pair's own atom order."""
    from ..structure import frame_words
    from .wizard import FROZEN_TOL_ANG
    regions = struct.regions or {}
    leads = sorted(set(regions.get(REGION_LEFT_ELECTRODE, ()))
                   | set(regions.get(REGION_RIGHT_ELECTRODE, ())))
    if struct.n_frames < 2 or not leads:
        return
    frames = np.asarray(struct.frames, dtype=float)
    broken = []
    for f in range(1, len(frames)):
        dist = np.linalg.norm(frames[f][leads] - frames[0][leads], axis=1)
        moved = [(i, float(d)) for i, d in zip(leads, dist)
                 if d > FROZEN_TOL_ANG]
        if moved:
            shown = ", ".join(f"{struct.atoms_named([i])} "
                              f"by {d:.4f} A"
                              for i, d in moved[:3])
            more = f" and {len(moved) - 3} more" if len(moved) > 3 else ""
            broken.append(f"{frame_words(f, len(frames))} moves "
                          f"{shown}{more}")
    if broken:
        raise ComposeError(
            f"{path.name}: {'; '.join(broken)} -- every frame of a set "
            f"shares the leads' calculation, so an electrode atom stays where "
            f"the base, {frame_words(0, len(frames))}, has it, within "
            f"{FROZEN_TOL_ANG:g} A "
            f"(engines/transport.md 2a.9).  Write the frames with the "
            f"electrodes held.")


def resolve_citation(citation: str, tree_root: Path
                     ) -> "Tuple[Path, CitedRun | CitedPair]":
    """The citation's path, fenced to the tree and classified against § 3.1's
    condition (`engines/transport.md` § 3.1).  Public: the web hand-over
    validates a citation through the SAME door prep composes through."""
    from ..projects import OutsideRoot, contain
    try:
        cited = contain(tree_root / citation, tree_root)
    except OutsideRoot as exc:
        raise ComposeError(
            f"the junction citation {citation!r} leaves the projects "
            f"tree: {exc}")
    if not cited.exists():
        raise ComposeError(
            f"the junction citation {citation!r} names nothing under the "
            f"projects tree.  {CITATION_CONDITION}.")
    return cited, classify_citation(cited)


def refuse_failed_citation(cited, citation: str) -> None:
    """A CITATION NO PREP COULD EVER TAKE, refused when the calculation is
    described, before anything is written (`engines/transport.md` § 3.1;
    the user, 2026-10-25): a relaxation run that ended with an error.  Its
    last geometry is whatever the engine left when it failed, and a folder
    citing it could not be pointed at another run -- one calculation is one
    structure.  A run still going is legal: a person may plan against a run
    in flight, and prep refuses it should it fail."""
    if (isinstance(cited, CitedRun) and cited.concluded is not None
            and cited.exit_code not in (None, 0)):
        raise ComposeError(
            f"the cited relaxation {citation!r} ended with exit code "
            f"{cited.exit_code} ({cited.concluded}) -- its last geometry is "
            f"whatever the engine left when it failed, and a calculation "
            f"citing it could never be prepared.  Launch the relaxation "
            f"again and cite the run that finishes, or cite another "
            f"relaxation (engines/transport.md 3.1).")


def _with_transport(across, where: str) -> Tuple[str, str, str]:
    """The junction's axis kinds: ``across`` -- the cited structure's kinds
    off the transport axis, each periodic or isolated -- and ``transport``
    along it, the axis `kmesh` states (the cell's third), so the composition
    and every rung's k-point mesh read one statement.  A record that says
    something else across is refused, naming ``where`` it said it: read as
    periodic it would replace the person's declaration without a word."""
    from ..kmesh import TRANSPORT_AXIS
    kinds = [str(k) for k in across]
    if len(kinds) != 3 or not all(
            k in ("periodic", "isolated")
            for i, k in enumerate(kinds) if i != TRANSPORT_AXIS):
        raise ComposeError(
            f"{where} records the axis kinds {kinds!r}; across the transport "
            f"axis a junction is periodic or isolated, so this record does "
            f"not say which (engines/transport.md 6.1c)")
    kinds[TRANSPORT_AXIS] = "transport"
    return tuple(kinds)


def junction_axis_kind(cited: "CitedRun | CitedPair") -> Tuple[str, str, str]:
    """The composed junction's axis kinds: across the transport axis, what
    the cited structure declared; along it, ``transport``.

    A cited run's deck records its structure's kinds in its ENGINE-OFFSET
    block (`script_emit.emit_engine_offset`), so a person who built a wire or
    chain junction -- isolated across, `engines/transport.md` § 6.1c -- keeps
    that declaration through the composition; a deck that records none is
    read periodic across.  A cited pair states its own (its sidecar's).
    """
    if isinstance(cited, CitedPair):
        return _with_transport(cited.structure.axis_kind, cited.sidecar.name)
    from ..atom_permutation import PermutationError
    from ..deck_record import DeckRecordUnreadable
    from ..runs import declared
    try:
        said = declared(cited.run)
    except (DeckRecordUnreadable, PermutationError) as exc:
        raise ComposeError(str(exc)) from exc
    recorded = (said.engine_offset or {}).get("axis_kind")
    return _with_transport(recorded or ("periodic",) * 3,
                           f"the cited deck {said.deck.name}"
                           if said.deck is not None else str(cited.path))


def labeled_citation_structure(cited: "CitedRun | CitedPair"):
    """The citation's LABELED structure, and the file its labels live in --
    ``(structure, source)``: a cited run's relaxed geometry with its deck's
    labels (``source`` the deck), or a cited pair's frames, its sidecar the
    source.

    ONE door, because its callers must agree about which labels are
    real: the composition itself (and the swap it applies to its own
    copy) and the orientation question the tab asks before composing.
    A second reading with its own precedence is how a tab offers to
    swap the labels of a file that is not the one being read.
    """
    kinds = junction_axis_kind(cited)
    if isinstance(cited, CitedPair):
        # THE PAIR'S OWN BOX AND LABELS, z transport: stated at the one
        # door, so `__post_init__` validates the box against the kinds.
        try:
            return (cited.structure.replace(axis_kind=kinds), cited.sidecar)
        except ValueError as exc:
            raise ComposeError(
                f"{cited.path.name} states a cell transport cannot use: "
                f"{exc}") from exc
    struct, _start, _output = _run_geometry(cited, kinds)
    from ..runs import declared
    return struct, declared(cited.run).deck


def _run_geometry(cited: CitedRun, kinds):
    """``(structure, start, output)`` -- what a cited run left, read through
    the run door as the Results tab reads a run (`execution/architecture.md`
    § 3.2; `model/parse.md` § 5.3, § 5b): the file that holds its result
    (`runs.openable`), and the run door's one read of its geometry
    (`runs.steps_of`, IN THE USER ORDER): its LAST geometry step, the relaxed
    junction, in the run's box, with what its deck declared -- the labels,
    the held atoms -- and what the run says about itself; and its FIRST
    step's coordinates, in the user order too, where the relaxation started, for
    the lead gate's *frozen means unmoved*.  The axis kinds are ``kinds``
    (:func:`junction_axis_kind`: z transport), stated at construction so
    ``__post_init__`` validates the box."""
    from ..runs import OrderUnstated, RunStepsError, openable, steps_of
    output, _trail = openable(cited.path)
    if output is None:
        raise ComposeError(
            f"{cited.path} holds no output of the run {cited.run.names.stem} "
            f"that a parser reads, so nothing says what geometry the "
            f"relaxation left.  {CITATION_CONDITION}.")
    # THE RUN'S GEOMETRY IN THE USER ORDER, through the run door's one read
    # (`runs.steps_of`): the relaxed junction and where it started, each
    # restored through the run's own order record -- and the composer sorts
    # it for TranSIESTA from there (`model/overview.md` § 2.2a, I3, I4).  A
    # run whose order nothing states is refused by name (I9).
    from ..atom_permutation import PermutationError
    from ..deck_record import DeckRecordUnreadable
    try:
        steps = steps_of(cited.run, output)
    except (RunStepsError, OrderUnstated, DeckRecordUnreadable,
            PermutationError) as exc:
        raise ComposeError(str(exc)) from exc
    try:
        struct = steps.last.replace(axis_kind=kinds)
    except ValueError as exc:
        # `prep` catches only ComposeError/SortError, so a bare ValueError
        # would surface as a traceback.
        raise ComposeError(f"{Path(output).name} states a cell transport "
                           f"cannot use: {exc}") from exc
    if not struct.regions:
        from ..runs import declared
        md = declared(cited.run).atom_metadata or {}
        why = (f"its deck's label block is for {md.get('n_atoms_total')} "
               f"atoms" if md.get("regions") else "its deck declares none")
        raise ComposeError(
            f"the cited relaxation in {cited.path} carries no region labels "
            f"for its {struct.n_atoms} atoms: {why}.  "
            f"Transport derives the electrodes FROM the labels (L-electrode / "
            f"R-electrode; engines/transport.md 3.1, 4) -- label the junction "
            f"on the Molbuilder tab and relax it through `molbuilder describe` -> "
            f"`prep task` -> `launch task`, then cite that run.")
    return (struct, np.asarray(steps.first.positions, dtype=float),
            Path(output))


def _extract_and_gate_electrodes(dev: Structure, *, prior_positions=None,
                                 held_by_relaxation=None):
    """Extract both leads from the sorted device.

    What a lead must BE is `wizard.extract_electrode_model`'s question
    -- every atom frozen, unmoved if a starting geometry is given,
    evenly spaced -- and a block that is none of those never becomes a
    model.  This passes the inputs through and turns the refusal into a
    `ComposeError`.

    Whether a lead is thick enough for its Hamiltonian's reach -- the
    principal-layer condition -- is TranSIESTA's own check, at the
    device's start (`ts_electrode.F90` ``check_connectivity``); its stop
    is said with what it means (`siesta_grammar.STOP_MEANINGS`), and
    nothing here judges it (user, 2026-10-20: "don't do the job of the
    engine").

    *prior_positions* is the geometry the cited relaxation STARTED from,
    in the user order, or ``None`` when there is none to compare against — the
    recompose-from-record path, where the comparison was made when the
    record was written.  *held_by_relaxation* is the atoms the cited
    junction's relaxation record held, in the user order
    (:func:`_held_by_its_relaxation`), or ``None`` the same way.

    A refusal names atoms in the user order, the device's own
    answer (`Structure.map_to_user_index`), never by their place in TranSIESTA's
    deck order (`engine_atom_index`).
    """
    # BOTH BLOCKS, THEN RAISE, so a junction with two broken blocks is
    # fixed and re-relaxed once, not once per block.
    models, refusals = [], []
    for region in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
        try:
            models.append(extract_electrode_model(
                dev, region, prior_positions=prior_positions,
                held_by_relaxation=held_by_relaxation))
        except ValueError as exc:
            refusals.append(str(exc))
    if len(refusals) == 1:
        raise ComposeError(
            f"the labeled electrode block cannot serve as a lead: "
            f"{refusals[0]}")
    if refusals:
        raise ComposeError(
            "neither labeled electrode block can serve as a lead. "
            + "  ".join(f"({i}) {r}." for i, r in enumerate(refusals, 1)))
    return tuple(models)


def _with_swapped_leads(struct, source_name: str):
    """THIS CALCULATION'S COPY with ``L-electrode`` ↔ ``R-electrode`` traded
    (`engines/transport.md` § 4; user, 2026-10-04): the description's
    ``swap_electrodes: true``, applied to the junction as it is composed --
    two arrays of indices in molbuilder's own metadata, no coordinate, no
    keyword, no result -- and never to the cited run's files.  NO GEOMETRY
    IS CONSULTED (user ruling, 2026-08-29): a swap is a rename, and whether
    the labels should be the other way round is the author's judgement;
    the only condition is that both labels exist."""
    regions = dict(struct.regions or {})
    for lab in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
        if lab not in regions:
            raise ComposeError(
                f"{source_name} does not carry {lab}, so there is no pair "
                f"to swap -- the description says swap_electrodes, and "
                f"the cited junction has one lead label.")
    regions[REGION_LEFT_ELECTRODE], regions[REGION_RIGHT_ELECTRODE] = (
        list(regions[REGION_RIGHT_ELECTRODE]),
        list(regions[REGION_LEFT_ELECTRODE]))
    return struct.replace(regions=regions)


#: THE TRANSPORT AXIS, lattice vector ``c`` -- z, as every junction states
#: its axis kinds (:func:`junction_axis_kind`): `kmesh.TRANSPORT_AXIS`, the
#: one home (a third copy stood here until 2026-10-18, K3).
from ..kmesh import TRANSPORT_AXIS as _TRANSPORT_AXIS  # noqa: E402


def _refuse_outside(struct: Structure) -> None:
    """An atom outside the cell along z is refused as the junction is
    composed (`engines/transport.md` § 4): TranSIESTA requires every atom
    inside, and the hand-off would refuse it (`cell.require_placed`) in
    words that cannot name the way out a transport calculation has.  The
    junction enters centred (`cell.entering`), so only an origin a person
    assigned on a cited pair can leave one there -- or a cell shorter than
    the junction's period, the leads' end layers meeting through the
    periodic image: I12's collision, said by its own check
    (`validation.sidecar.check_junction_boundary`).
    Asked as every rung's deck will place it, in any frame: the one
    placement rule (`cell.to_engine`) and the hand-off's own containment
    check (`cell.atoms_outside`)."""
    from ..cell import atoms_outside, to_engine
    outside = set()
    for i in range(struct.n_frames):
        outside |= set(atoms_outside(to_engine(struct.frame_at(i)))
                       .get(_TRANSPORT_AXIS, ()))
    outside = sorted(outside)
    if not outside:
        return
    where = struct.atoms_named(outside, limit=8)
    if struct.engine_offset is None:
        # CENTRED AND STILL OUTSIDE: the atoms span more than the cell along
        # z, so the leads' end layers meet through the image -- I12's
        # collision, in its own words: the room measured from the lead and
        # the c that closes the boundary to one layer spacing.
        from ..validation.sidecar import check_junction_boundary
        said = [i.message for i in check_junction_boundary(
                    struct, severity="error")
                if i.where == "cell.transport_collision"]
        raise ComposeError(
            f"the junction, centred in its cell, has {where} outside it "
            f"along z, the transport direction: "
            + (said[0] if said else
               "its atoms span more than the cell's length there "
               "(engines/transport.md 6.1c, I12)."))
    raise ComposeError(
        f"the cited junction has {where} outside its cell along z, the "
        f"transport direction, where TranSIESTA requires every atom inside "
        f"-- the origin assigned on the cited pair puts them there.  Move "
        f"that origin on the pair's Cell page, or set it back to Automatic, "
        f"and cite it again (engines/transport.md 4).")


def refuse_inside_citation(dest, cited, *, what: str = "the cited junction",
                           shown: Optional[str] = None) -> None:
    """THE CALCULATION NEVER LIVES IN ITS CITATION (`engines/transport.md`
    § 3.1): describing into the cited run's folder, or below it, would put
    ``task.json`` -- and later every prep attempt -- inside the finished
    relaxation it cites; or inside the finished transport calculation it
    reuses (§ 2a.16), ``what`` naming it.  ``cited`` is the cited folder.
    Asked by both doors that describe; ``shown`` is the folder as the door's
    reader names it, as `describe.refuse_described` takes it."""
    home = Path(cited).resolve()
    there = Path(dest).resolve()
    if there == home or home in there.parents:
        raise ComposeError(
            f"{shown or there} {'is' if there == home else 'is inside'} "
            f"{what} {home.name} -- the calculation never lives inside its "
            f"citation.  Make a folder for this transport calculation (under "
            f"the project's transport topic) and describe it there "
            f"(engines/transport.md 3.1).")


def _held_by_its_relaxation(relaxed: Structure):
    """The atoms the cited junction's relaxation held, in the user order, as its
    record says (`parse.contract.held_atoms_of`) -- read from the junction
    as cited, before it enters the calculation, because the record is about
    those coordinates; ``None`` when it carries no record about them."""
    from ..parse.contract import held_atoms_of
    rec = (getattr(relaxed, "info", None) or {}).get("relaxation")
    if (not isinstance(rec, dict)
            or rec.get("geometry_sha256") != relaxed.geometry_fingerprint()):
        return None
    return held_atoms_of(relaxed, rec)


def compose_junction(citation: str, *, tree_root,
                     swap_electrodes: bool = False) -> ComposedJunction:
    """The whole § 4.1–4.2 compose: citation → sorted, gated, extracted --
    from either kind (`engines/transport.md` § 3.1): a cited run's relaxed
    geometry (:func:`_run_parts`) or a cited pair's frames
    (:func:`_pair_parts`), then one sort, one extraction, one record.

    Raises :class:`ComposeError` (refusals naming what to run first) or
    :class:`~molbuilder.transport.sort.SortError` (the § 4.1a label
    gates) — the caller surfaces either verbatim.
    """
    tree_root = Path(tree_root)
    _path, cited = resolve_citation(citation, tree_root)
    parts = (_pair_parts(cited, swap_electrodes)
             if isinstance(cited, CitedPair)
             else _run_parts(cited, citation, swap_electrodes))
    relaxed, start, deck_text, files, said = parts
    held = _held_by_its_relaxation(relaxed)
    # THE JUNCTION ENTERS THIS CALCULATION (`structure-periodicity.md` § 6.0):
    # the engine's origin of a cited run's coordinates is not carried into
    # it, so the one rule centres it for every rung; a person's assigned
    # origin on a cited pair is kept.
    relaxed = entering(relaxed)
    _refuse_outside(relaxed)

    # THE JUNCTION'S SORTED COPY, by the key its decks are written by (the
    # one table, `sort.deck_sort_key`).
    sorted_res = sort_by(relaxed, deck_sort_key("siesta", "transport"))
    dev = sorted_res.structure

    # The electrode models, extracted from the SORTED blocks -- and the
    # § 3 lead gates, which is the same step: a block that is not frozen
    # bulk does not become a model (`wizard.extract_electrode_model`).  The
    # starting geometry stays in the user order: the gate compares each atom of
    # the sorted device with itself there, through the device's own answer
    # (`Structure.map_to_user_index`).
    elec_l, elec_r = _extract_and_gate_electrodes(
        dev, prior_positions=start, held_by_relaxation=held)

    provenance = {
        # WHICH KIND was cited (§ 3.1): it decides which files the record
        # holds (`record_files`).
        "kind": "pair" if isinstance(cited, CitedPair) else "run",
        # THE RENAME THIS COPY CARRIES (§ 4): the record answers for the
        # choice it was composed with, as it answers for its citation.
        "swap_electrodes": bool(swap_electrodes),
        "slot": "junction",
        "citation": citation,
        # The files this junction was composed from, with hashes -- a
        # result can always say which bytes built it, and a cited pair
        # edited since is refused by name (`load_compose_record`).
        "files": files,
        **said,
    }
    return ComposedJunction(
        junction=dev,
        notes=sorted_res.notes,
        relaxed=relaxed,
        electrode_left=elec_l,
        electrode_right=elec_r,
        deck_text=deck_text,
        provenance=provenance,
    )


def _pair_parts(cited: CitedPair, swap_electrodes: bool):
    """A cited pair, ready to sort: ``(junction, start, deck_text, files,
    said)`` -- every frame with its labels (the swap applied to this
    calculation's copy), no starting geometry (a pair holds the one it was
    saved with), no deck (none beside a pair), its two files' hashes, and
    its frame count for the record."""
    struct, source = labeled_citation_structure(cited)
    if swap_electrodes:
        struct = _with_swapped_leads(struct, Path(source).name)
    _bad = _unusable_cell(struct)
    if _bad:
        raise ComposeError(
            f"the cited pair {cited.path.name} {_bad} -- a junction needs its "
            f"lattice (science/junction-cell.md).")
    files = {f.name: sha256_of(f) for f in (cited.path, cited.sidecar)}
    return struct, None, None, files, {"frames": cited.n_frames}


def _run_parts(cited: CitedRun, citation: str, swap_electrodes: bool):
    """A cited run, ready to sort: ``(junction, start, deck_text, files,
    said)`` -- what it left, read through the run door
    (:func:`_run_geometry`): its last geometry step with its labels and
    record, and its first step's coordinates for the lead gate's *frozen
    means unmoved*; its deck verbatim, the record's copy; the hashes of the
    output read and of the deck; and how the run ended."""
    if cited.concluded is None:
        raise ComposeError(
            f"the cited relaxation {citation!r} has not CONCLUDED -- it is "
            f"still running, or it was force-stopped (the two look "
            f"identical on disk; project-layout.md 1.6).  Let it finish; "
            f"transport never decides this over you (engines/transport.md "
            f"3.1).")
    if cited.exit_code != 0:
        raise ComposeError(
            f"the cited relaxation {citation!r} ended with exit code "
            f"{cited.exit_code} ({cited.concluded}) -- its last geometry is "
            f"whatever the engine left when it failed, not a relaxed "
            f"junction.  Run the relaxation to its end, then cite it "
            f"(engines/transport.md 3.1).")
    struct, start, output = _run_geometry(cited, junction_axis_kind(cited))
    # A swap renames the labels on this calculation's copy
    # (`_with_swapped_leads`) and never writes the cited run.
    deck = cited.run.deck
    if swap_electrodes:
        struct = _with_swapped_leads(struct, deck.name)
    # `Structure.cell`'s setter is permissive, so an unusable box would
    # travel on and raise a bare ValueError later -- and `prep` catches only
    # ComposeError/SortError, so it would reach the person as a traceback.
    _bad = _unusable_cell(struct)
    if _bad:
        raise ComposeError(
            f"the cited relaxation in {cited.path} {_bad} -- a junction "
            f"needs its lattice (science/junction-cell.md).")
    files = {f.name: sha256_of(f) for f in (output, deck)}
    said = {
        # HOW THE CITED RUN ENDED -- its record's line -- and what it
        # converged, the run door's reading of its output (§ 3.1): a
        # geometry cited with `geometry NO` is cited knowingly, and the
        # geometry it hands over is then the last one SIESTA wrote.
        "evidence": cited.concluded,
        "relaxation": {"converged": cited.converged,
                       "exit_code": cited.exit_code},
    }
    return struct, start, read_deck(deck), files, said


def write_compose_record(base_dir, composed: ComposedJunction) -> List[str]:
    """The composed junction, ON DISK beside the transport calculation's
    ``task.json`` — § 4.1's *"the cited structure is COPIED in with
    provenance"*, whole: the SORTED structure PAIR (the geometry every
    stage renders from -- every frame -- AND the file carrying its region
    labels), a cited run's own deck verbatim (the electronic contract,
    re-parseable anywhere), and the two sidecars.  With these the folder travels:
    :func:`load_compose_record` rebuilds the junction on a machine where
    the cited tree does not exist.

    Answers what it wrote, checked against :func:`record_files` — the
    list is not a hand-kept second copy of that set, and a codec that
    somehow skipped the label file is caught HERE, where the record is
    made, rather than at a load on another machine."""
    from ..persist import write_bytes, write_json
    from ..workingcopy_structure import StructureCodec
    base_dir = Path(base_dir)
    # THE ENGINE'S RECORD, in the engine's order with its order record in
    # the sidecar (`model/overview.md` § 2.2a) -- the codec's one door for it.
    StructureCodec().write(composed.junction,
                           base_dir / JUNCTION_GEOMETRY, engine_record=True)
    if composed.deck_text is not None:
        write_bytes(base_dir / JUNCTION_DECK,
                    composed.deck_text.encode("utf-8"))
    # AT THIS SCHEMA, whoever composed it: a record read back and written
    # again holds what this molbuilder writes.
    write_json(base_dir / PROVENANCE_FILE,
               {**composed.provenance, "schema": PROVENANCE_SCHEMA})

    expected = record_files(composed.provenance["kind"])
    missing = [n for n in expected if not (base_dir / n).is_file()]
    if missing:
        raise ComposeError(
            f"the composed record in {base_dir} is missing "
            f"{', '.join(missing)} right after being written -- it "
            f"would not rebuild on another machine, so nothing should "
            f"rely on it.  (The geometry travels as a PAIR; its label "
            f"file is what carries the electrode regions.)")
    return list(expected)


def _pair_unchanged(cited: CitedPair, provenance, citation: str) -> None:
    """A cited pair's two files, against the hashes its record pinned
    (`engines/transport.md` § 3.1) -- refused by name when either changed:
    the seed and the leads ran on what the pair said then, and a frame set
    is a data set, made once by the script that writes it."""
    pinned = provenance.get("files") or {}
    changed = [f.name for f in (cited.path, cited.sidecar)
               if pinned.get(f.name) != sha256_of(f)]
    if changed:
        raise ComposeError(
            f"the cited pair {citation!r} has changed since this calculation "
            f"was composed from it: {', '.join(changed)} no longer "
            f"{'holds' if len(changed) == 1 else 'hold'} the bytes "
            f"{PROVENANCE_FILE} pinned.  This calculation's stages ran on "
            f"the pair as it was; cite the pair as it is now in a new "
            f"transport calculation (`molbuilder describe`), or restore it "
            f"(engines/transport.md 3.1).")


def read_provenance(base_dir, *, read=None) -> Optional[Dict]:
    """The calculation's ``slot-provenance.json`` -- where each part of the
    junction came from, written at its first prep
    (:func:`write_compose_record`) -- through `persist`, or ``None`` before
    anything was composed.  The one reader of that file: the record's own
    check below, the transport record, and the frame count
    (`transport.stages.frames_of`).  ``read`` reads a path's text where the
    record is a prep's plan's, not yet on disk (`prepared_with.record`)."""
    import json
    from ..persist import read_json
    path = Path(base_dir) / PROVENANCE_FILE
    if read is not None:
        try:
            return json.loads(read(path))
        except FileNotFoundError:
            return None
    return read_json(path) if path.is_file() else None


@dataclass(frozen=True)
class RecordedJunction:
    """The composed junction as its record holds it
    (:func:`recorded_junction`)."""
    #: every frame, in TranSIESTA's order, its order record from its sidecar
    structure: Structure
    #: WHY ITS ORDER IS NOT STATED -- its sidecar written before the field
    #: (`model/overview.md` § 2.2a, I9) -- in one sentence, the fact alone:
    #: each door that names the junction's atoms in the user order says its own
    #: way on.  ``None`` for a sidecar that states it (null for an order the
    #: sort left as it was).
    order_unstated: Optional[str]


def recorded_junction(base_dir) -> Optional[RecordedJunction]:
    """THE ONE READ OF THE COMPOSED JUNCTION beside ``task.json``, through the
    codec (every frame, its order record from its sidecar), and whether that
    sidecar states the order record at all -- known by what the file states,
    never by the molbuilder that wrote it (the user, 2026-10-15:
    "feature/format detection").  ``None`` before anything was composed;
    a pair that does not read is refused in its reader's words,
    :class:`ComposeError`.  Asked by the record's read for prep and the
    gather (:func:`load_compose_record`), the transport record, its data
    file and a selection's PDOS, and `prep task` with no group named."""
    from ..sidecars.molstruct import sidecar_path_for
    from ..workingcopy_structure import (ORDER_UNSTATED, StructureCodec,
                                         pair_order_unstated_said)
    path = Path(base_dir) / JUNCTION_GEOMETRY
    if not path.is_file():
        return None
    stated: set = set()
    try:
        # ITS ORDER IS THIS READ'S TO JUDGE, below -- opened whatever its
        # sidecar states, as a door that shows it does (I9).
        struct = StructureCodec().load(path, frames=True, stated_out=stated,
                                       opened=(ORDER_UNSTATED,),
                                       said_out=[])
    except ValueError as exc:
        # THE CODEC'S REFUSALS are ValueErrors -- a sidecar that does not
        # parse, one whose geometry pin is not the file's, an atom list that
        # does not read (`structure_pair.MolstructJsonError` among them).
        raise ComposeError(f"the composed junction, {path.name}, does not "
                           f"read: {exc}") from exc
    return RecordedJunction(
        structure=struct,
        order_unstated=(
            None if "atom_order" in stated else
            "the composed junction's "
            + pair_order_unstated_said(sidecar_path_for(path).name)))


def junction_difference(recorded: Structure,
                        composed: Structure) -> Optional[str]:
    """``None`` when ``composed`` is the junction ``recorded`` holds, both
    in one order -- TranSIESTA's for a record read back, the user order for
    a frame set's frame 0 (`transport.reuse.frame_set_refusal`) -- the same
    number of atoms and frames; atom *i*
    the same element at the same place in every frame, and the same cell
    vectors, by the one rule two geometries are compared by
    (`structure.farthest_apart`, `structure.SAME_PLACE_ANG`); the same kind
    on each axis; every label the same set of atoms -- else the first
    difference, in words, an atom named in the user order
    (`Structure.atoms_named`)."""
    if recorded.n_atoms != composed.n_atoms:
        return (f"it holds {composed.n_atoms} atoms, the record "
                f"{recorded.n_atoms}")
    if recorded.n_frames != composed.n_frames:
        return (f"it holds {composed.n_frames} frame(s), the record "
                f"{recorded.n_frames}")
    for i, (a, b) in enumerate(zip(recorded.elements, composed.elements)):
        if a != b:
            return (f"{composed.atoms_named([i])} is {b} where the record "
                    f"holds {a}")

    def frames_of(s):
        return s.frames if s.frames is not None else [s.positions]
    for f, (a, b) in enumerate(zip(frames_of(recorded), frames_of(composed))):
        i, d = farthest_apart(a, b)
        if d > SAME_PLACE_ANG:
            return (f"{composed.atoms_named([i])} stands {d:.2e} Å from its "
                    f"recorded place"
                    + (f" in {frame_words(f, recorded.n_frames)}"
                       if recorded.n_frames > 1 else ""))
    if (recorded.cell is None) != (composed.cell is None):
        return "its cell is not the recorded one"
    if recorded.cell is not None:
        i, d = farthest_apart(recorded.cell, composed.cell)
        if d > SAME_PLACE_ANG:
            return (f"its lattice vector index {i} stands {d:.2e} Å from "
                    f"the recorded one")
    axes = [tuple(s.axis_kind or ()) for s in (recorded, composed)]
    if axes[0] != axes[1]:
        return f"its axes are {axes[1]}, the record's {axes[0]}"
    labels_r = {k: sorted(v) for k, v in (recorded.regions or {}).items()}
    labels_c = {k: sorted(v) for k, v in (composed.regions or {}).items()}
    if labels_r != labels_c:
        differ = sorted(k for k in set(labels_r) | set(labels_c)
                        if labels_r.get(k) != labels_c.get(k))
        return f"its label {differ[0]!r} holds other atoms than the record's"
    return None


def load_compose_record(base_dir, *, citation: str, tree_root=None,
                        why: "Optional[list]" = None,
                        swap_electrodes: bool = False
                        ) -> Optional[ComposedJunction]:
    """The travelled copy, loaded back — or ``None`` when there is no
    complete record for THIS citation (prep then composes fresh).

    **`None` HAS SEVERAL CAUSES AND THEY ARE NOT THE SAME NEWS** -- no
    record, a record of another citation or another rename, a record that
    names no kind, an incomplete one.  Pass a list as *why* and the reason
    is appended to it, in words for a person.
    The `retired_out` pattern (`workingcopy_structure.StructureCodec.load`):
    the happy path is unchanged and the caller that needs more asks for it.

    Without it, all of them read as *"there is no record"* -- while the
    likeliest cause, on a folder that has travelled somewhere the citation
    cannot be re-resolved, is that the record is sitting right there and
    was composed from a DIFFERENT attempt.

    The record answers for the citation it was made from: a
    ``task.json`` re-pointed at a different attempt must NOT keep
    serving the old copy, so a citation mismatch reads as *no record*.
    The § 3 lead gates re-run on the loaded structure (cheap, pure) —
    frozen-declared and evenly-spaced both need only the structure.  The
    UNMOVED comparison does not re-run: it measures against the geometry
    the relaxation started from, which is exactly what a travelled folder
    no longer carries, and the provenance records that it passed when the
    copy was made.

    *tree_root* holds a cited PAIR to the bytes it was composed from
    (§ 3.1): one edited since is refused by name, because the seed and the
    leads already ran on what it said then.
    """
    base_dir = Path(base_dir)
    # WHICH FORM decides which files the record needs, and the form is
    # in the provenance -- so read that first, then ask `record_files`
    # for the set.  Incomplete in any of them = NO RECORD, which is the
    # answer that makes prep compose fresh instead of failing.
    def _no(reason: str):
        if why is not None:
            why.append(reason)
        return None

    provenance = read_provenance(base_dir)
    if provenance is None:
        return _no(f"there is no {PROVENANCE_FILE} beside it, so nothing "
                   f"here has been composed yet")
    if provenance.get("citation") != citation:
        return _no(f"the record beside it was composed from "
                   f"{provenance.get('citation')!r}, and this task now "
                   f"cites {citation!r}")
    # THE RENAME IS PART OF THE RECORD'S IDENTITY (§ 4): a copy composed
    # with the leads one way round does not serve a description that now
    # says the other.
    if bool(provenance.get("swap_electrodes", False)) != bool(swap_electrodes):
        return _no(f"the record beside it was composed with "
                   f"swap_electrodes {bool(provenance.get('swap_electrodes', False))}, "
                   f"and this task now says {bool(swap_electrodes)}")
    kind = provenance.get("kind")
    if kind not in CITATION_KINDS:
        return _no(f"the record beside it names no citation kind "
                   f"({PROVENANCE_FILE} says {kind!r}), so it does not say "
                   f"which files it holds")
    missing = [n for n in record_files(kind)
               if not (base_dir / n).is_file()]
    if missing:
        return _no(f"the record beside it is incomplete -- "
                   f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                   f"missing")
    deck_path = base_dir / JUNCTION_DECK
    # THE JUNCTION, IN THE ENGINE'S ORDER, CARRIES ITS OWN ORDER RECORD in its
    # sidecar (`model/overview.md` § 2.2a, I7) -- none for an order the sort
    # left as it was.
    try:
        recorded = recorded_junction(base_dir)
    except ComposeError as exc:
        return _no(str(exc))
    dev = recorded.structure
    # ONE WHOSE SIDECAR STATES NO ORDER -- written before the field -- is
    # usable when its citation composed again, in memory, is the junction
    # recorded (`engines/transport.md` § 1.1, *The first prep*): it is given
    # that composition's order record, which the prep that read it writes
    # (`order_found`); another, or a citation that does not compose here, is
    # no record.
    order_found = False
    if recorded.order_unstated:
        if tree_root is None:
            return _no(f"{recorded.order_unstated}; it is checked against its "
                       f"citation where the citation resolves, inside its "
                       f"projects tree")
        try:
            composed = compose_junction(citation, tree_root=tree_root,
                                        swap_electrodes=swap_electrodes)
        except (ComposeError, SortError) as exc:
            return _no(f"{recorded.order_unstated}; its citation does not "
                       f"compose again here -- {exc}")
        differs = junction_difference(dev, composed.junction)
        if differs:
            return _no(f"{recorded.order_unstated}; composed again from its "
                       f"citation, it is not the junction recorded -- "
                       f"{differs}")
        dev = dev.replace(atom_order=composed.junction.atom_order)
        order_found = True
    if tree_root is not None:
        try:
            _cited_path, cited = resolve_citation(citation, Path(tree_root))
        except ComposeError:
            cited = None        # the citation moved: the record stands alone
        if isinstance(cited, CitedPair):
            _pair_unchanged(cited, provenance, citation)
    # NO `prior_positions` HERE, and that is not an omission: this
    # rebuilds from a junction that was already composed, and the
    # geometry the relaxation started from is not part of the record --
    # the unmoved comparison ran when the record was written.  The frozen
    # DECLARATION is asked separately and does re-run here, as does
    # even-spacing.  Both need only the structure.
    elec_l, elec_r = _extract_and_gate_electrodes(dev)
    deck_text = read_deck(deck_path) if kind == "run" else None
    return ComposedJunction(
        junction=dev,
        notes=(),
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        deck_text=deck_text,
        provenance=provenance,
        order_found=order_found,
    )
