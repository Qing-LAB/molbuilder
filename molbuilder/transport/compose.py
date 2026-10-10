"""P4a — prep COMPOSES the transport calculation from its one citation.

`engines/transport.md` § 3.1 and § 6.2: resolve the junction citation on
the machine where prep runs (strict composition, ruling Q2 — a missing
or unconcluded relaxation is a refusal naming what to run first, never a
trigger to run it); read the junction it holds — a cited run's folder
through the run door, as the Results tab reads a run (`runs.run_of`,
`runs.openable`, the parse registry, `runs.view_of`), or a cited pair
through the codec; run the categorical sort (P2); apply the lead gate;
extract the two electrode models from the sorted blocks (the wizard's
move, § 3); and record the provenance — the citation, its kind, and the
content hashes of the files it was composed from, a cited run's **own
deck** among them (the fdf that actually ran is the truth about a result;
user ruling 2026-08-28).

Pure composition: everything here reads the tree and returns objects;
the caller (prep, `jobset.prep._composed_for_prep`) owns what lands on
disk where.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..atom_permutation import PERMUTATION_FILE, read_permutation
from ..structure import Structure
from .sort import (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE,
                   SortResult, sort_by, write_permutation)
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
# (`PERMUTATION_FILE`, the record beside them, is `atom_permutation`'s.)

#: The citation kinds a record names (`engines/transport.md` § 3.1).
CITATION_KINDS = ("run", "pair")


def record_files(kind: str) -> Tuple[str, ...]:
    """WHAT THE TRAVELLED RECORD CONSISTS OF for a citation of ``kind`` —
    one answer, read by the write and by the load, so they cannot disagree
    about whether a copy is complete: the sorted geometry and its label file
    (every frame, and a cited pair's ``info.calculation``), the provenance,
    the permutation, and a cited run's deck (the electronic contract IS the
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
             PROVENANCE_FILE, PERMUTATION_FILE)
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
    #: the relaxed, LABELED junction in canonical transport order
    sorted: SortResult
    #: the same structure before the sort (relaxed positions, source
    #: order).  ``None`` on a record loaded back from the travelled
    #: copy: the original order lives in the citation and the
    #: permutation sidecar, and no stage renders from it.
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    #: (`jobset.runstatus.converged_of`).
    converged: Optional[str] = None


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
    ".molstruct.json carrying info.calculation, as the Results tab or "
    "`molbuilder xv2xyz --from-run` saves a finished relaxation "
    "(engines/transport.md 3.1).  Relax the junction through `jobset init` "
    "-> `prep task` -> `launch task`, then cite that run or the pair saved "
    "from it")


def classify_citation(cited: Path) -> "CitedRun | CitedPair":
    """Classify a citation against § 3.1's condition by what the path is: a
    folder is a relaxation run of molbuilder's own (:func:`_classify_run`), a
    file a structure pair (:func:`_classify_pair`).  Raises
    :class:`ComposeError` naming what is missing.

    CLASSIFYING IS NOT COMPOSING: a run still going, or one that ended with
    an error, is RECORDED here -- describing a transport calculation ahead
    of its relaxation is legal -- and refused by :func:`compose_junction`.
    """
    cited = Path(cited)
    if cited.is_file():
        return _classify_pair(cited)
    return _classify_run(cited)


def _classify_run(cite_dir: Path) -> CitedRun:
    """A cited folder, read through the run door (`runs.run_of`,
    `execution/architecture.md` § 3.2): the run it speaks for -- in a run's
    own folder its one stage, at a flat calculation's root the stage the
    root speaks for; its launch record and how it ended, the run record's
    doors."""
    from molbuilder.runrecord import (LaunchRecordError, ending,
                                      launch_record)
    from .. import calcdirs
    from ..runs import place_of, run_of
    place = place_of(cite_dir)
    if place.problem:
        raise ComposeError(place.problem)
    run = run_of(cite_dir)
    if run is None or run.stage is None:
        where = ("no calculation claims this folder"
                 if not place.ours else
                 "it is a container, not a run -- cite one of the runs below "
                 "it" if place.role == calcdirs.CONTAINER else
                 "nothing was launched here")
        raise ComposeError(
            f"{cite_dir} holds no run of molbuilder's: {where}.  "
            f"{CITATION_CONDITION}.")
    try:
        launched = launch_record(cite_dir, run.names) is not None
    except LaunchRecordError as e:
        raise ComposeError(str(e)) from e
    # HOW THE CITED RUN ENDED, asked of the one door (`runrecord.ending`,
    # `execution/architecture.md` § 3.2) about THIS run -- never the
    # directory's `run_status`, which can answer for a neighbour stage's run.
    end = ending(cite_dir, run.basename)
    if not launched and not end.concluded:
        raise ComposeError(
            f"{cite_dir} holds the run {run.basename}, but nothing launched "
            f"it and nothing recorded how it ended.  {CITATION_CONDITION}.")
    # WHAT IT CONVERGED, the run door's reading of its output (the same the
    # status verb prints): carried into the provenance and said by the tab,
    # so a geometry that did not converge is cited knowingly.
    from ..jobset.runstatus import converged_of
    from ..parse.dirs.rundir import run_state_of
    try:
        converged = converged_of(run_state_of(cite_dir, run.names))
    except Exception:                                      # noqa: BLE001
        converged = None
    return CitedRun(path=cite_dir, run=run, concluded=end.line,
                    exit_code=end.code, converged=converged)


def _classify_pair(path: Path) -> CitedPair:
    """A cited structure pair: every frame through the codec
    (`StructureCodec.load(frames=True)`), its ``info.calculation`` -- what
    the settings default from -- and the per-frame promise only the
    citation can check (`engines/transport.md` § 2a.9): every frame keeps
    the electrode atoms where frame 0 has them, within
    `wizard.FROZEN_TOL_ANG`, because every frame shares the leads'
    calculation.  The same atoms, species and cell are the structure's own
    invariant, refused by its reader."""
    from ..sidecars.molstruct import sidecar_path_for
    from ..workingcopy_structure import StructureCodec
    sidecar = sidecar_path_for(path)
    try:
        struct = StructureCodec().load(path, frames=True)
    except (ValueError, OSError) as exc:
        raise ComposeError(f"{path} does not read as a structure pair: "
                           f"{exc}") from exc
    calc = (struct.info or {}).get("calculation")
    if not isinstance(calc, dict) or not isinstance(calc.get("contract"),
                                                    dict):
        why = ("it has no .molstruct.json beside it" if not sidecar.is_file()
               else f"{sidecar.name} records no optimization")
        raise ComposeError(
            f"{path} carries no info.calculation -- {why} -- so nothing says "
            f"which settings its geometry was optimized with.  "
            f"{CITATION_CONDITION}.")
    _frames_keep_the_leads(struct, path)
    return CitedPair(path=path, sidecar=sidecar, structure=struct,
                     calculation=calc)


def _frames_keep_the_leads(struct: Structure, path: Path) -> None:
    """Every frame's electrode atoms where frame 0 has them (§ 2a.9: the
    lead Hamiltonian is truth for every frame), measured as the lead gate
    measures an unmoved atom -- the distance, against
    `wizard.FROZEN_TOL_ANG` -- refused naming each frame and its atoms, in
    the pair's own atom order."""
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
            shown = ", ".join(f"atom {i} ({struct.elements[i]}) by {d:.4f} A"
                              for i, d in moved[:3])
            more = f" and {len(moved) - 3} more" if len(moved) > 3 else ""
            broken.append(f"frame {f} moves {shown}{more}")
    if broken:
        raise ComposeError(
            f"{path.name}: {'; '.join(broken)} -- every frame of a set "
            f"shares the leads' calculation, so an electrode atom stays where "
            f"frame 0 has it, within {FROZEN_TOL_ANG:g} A "
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
    from ..runs import declared
    said = declared(cited.run)
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
    (`runs.openable`), parsed by the registry; its LAST geometry step, the
    relaxed junction, in the run's box, with what its deck declared -- the
    labels, the held atoms -- and what the run says about itself
    (`runs.view_of`); and its FIRST step's coordinates, where the
    relaxation started, for the lead gate's *frozen means unmoved*.  The
    axis kinds are ``kinds`` (:func:`junction_axis_kind`: z transport),
    stated at construction so ``__post_init__`` validates the box."""
    from ..parse import detect
    from ..parse.errors import ParseError
    from ..runs import openable, view_of
    output, _trail = openable(cited.path)
    if output is None:
        raise ComposeError(
            f"{cited.path} holds no output of the run {cited.run.basename} "
            f"that a parser reads, so nothing says what geometry the "
            f"relaxation left.  {CITATION_CONDITION}.")
    try:
        traj = detect(output).parse(str(output))
    except (ParseError, OSError, ValueError) as exc:
        raise ComposeError(f"{Path(output).name} does not read as the "
                           f"run's output: {exc}") from exc
    steps = [fr for fr in traj.frames if fr.structure is not None]
    if not steps:
        raise ComposeError(
            f"{Path(output).name} holds no geometry step: the run "
            f"{cited.run.basename} never reached its first geometry.")
    first, last = steps[0], steps[-1]
    lattice = last.lattice if last.lattice is not None else traj.lattice
    view = view_of(cited.run, output=output, traj=traj,
                   n_atoms=len(last.structure.elements),
                   lattice=(None if lattice is None
                            else [[float(v) for v in row] for row in lattice]))
    try:
        struct = view.structure(last.structure.elements,
                                last.structure.positions).replace(
                                    axis_kind=kinds)
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
            f"on the Molbuilder tab and relax it through `jobset init` -> "
            f"`prep task` -> `launch task`, then cite that run.")
    return (struct, np.asarray(first.structure.positions, dtype=float),
            Path(output))


def _extract_and_gate_electrodes(dev: Structure, *, prior_positions=None,
                                 atom_ids=None, ion_dir=None):
    """Extract both leads from the sorted device, then measure the one
    § 3 condition that needs files from outside the structure.

    What a lead must BE is `wizard.extract_electrode_model`'s question
    -- every atom frozen, unmoved if a starting geometry is given,
    evenly spaced -- and a block that is none of those never becomes a
    model.  This passes the inputs through and turns the refusal into a
    `ComposeError`.

    What stays is the principal-layer condition, because it is the one
    that cannot be answered from the structure: it compares the orbital
    INTERACTION RANGE against the lead's PERIOD (§ 3's own wording) and
    the ranges are READ, never guessed — SIESTA leaves ``<El>.ion``
    beside every run, and *ion_dir* (the cited directory) is where they
    are looked for.  No readable ``.ion`` for an element -> the condition
    is honestly UNVERIFIED (a note on the model; TranSIESTA verifies lead
    connectivity itself at run time), never a refusal on a number nobody
    measured.

    *prior_positions* is the geometry the cited relaxation STARTED from,
    already permuted into *dev*'s order, or ``None`` when there is none to
    compare against — the recompose-from-record path, where the comparison
    was made when the record was written.

    *atom_ids* is the sort's ``sorted_to_original``, so a refusal names
    atoms by the identity in the person's own file rather than by their
    place in TranSIESTA's deck order (`engine_atom_index`).
    """
    from ..parse.ion import max_orbital_rc_ang
    # BOTH BLOCKS, THEN RAISE, so a junction with two broken blocks is
    # fixed and re-relaxed once, not once per block.
    models, refusals = [], []
    for region in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
        try:
            models.append(extract_electrode_model(
                dev, region, prior_positions=prior_positions,
                atom_ids=atom_ids))
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
    models = tuple(models)
    for model in models:
        # Tiling is the extraction's: `cell.bulk_z_period` refuses an
        # unevenly spaced block, so a model that exists has tiled.  The
        # principal-layer condition below needs its period, which is why
        # that order is structural rather than a choice made here.
        # Thick enough -- the principal-layer condition.  Two orbitals
        # couple within rc_i + rc_j of each other.  The nearest atoms
        # of NEXT-NEAREST lead cells are separated along transport by
        # 2*period - span (the top of cell n to the bottom of cell
        # n+2); their true 3-D distance is at least that, so gating on
        # the axial separation is the conservative side.  The
        # self-energy stays adjacent-cell-only iff the range fits.
        # (Written as 2*period - span, NOT period + interlayer: the
        # two agree only while the period is DERIVED as span +
        # interlayer, and an explicitly overridden z-period breaks
        # that identity.)
        elems = sorted(set(model.elements))
        rc = {el: (max_orbital_rc_ang(Path(ion_dir) / f"{el}.ion")
                   if ion_dir is not None else None)
              for el in elems}
        unread = sorted(el for el, r in rc.items() if r is None)
        if unread:
            model.notes.append(
                f"principal-layer condition UNVERIFIED for the "
                f"{model.label} block: no readable "
                f"{', '.join(el + '.ion' for el in unread)} beside the "
                f"citation to read the orbital ranges from.  TranSIESTA "
                f"verifies lead connectivity itself at run time.")
        else:
            reach = 2.0 * max(rc.values())
            gap = 2.0 * model.z_period - model.z_span
            # The wizard's ~12 A floor is a GUESS made before anything
            # was read (wizard.py), and it is wrong about exactly the
            # leads this measurement exists to judge -- a 3-layer Au
            # block is 4.8 A and passes.  A measured verdict retires
            # it: carrying both would leave one model saying "may be
            # too thin" beside the numbers proving it is not.
            model.notes[:] = [n for n in model.notes
                              if "principal layer" not in n]
            if reach > gap:
                raise ComposeError(
                    f"the orbital interaction range {reach:.2f} A "
                    f"(2 x max orbital cutoff, read from "
                    f"{', '.join(el + '.ion' for el in elems)}) exceeds "
                    f"the {gap:.2f} A between next-nearest "
                    f"{model.label} cells (period "
                    f"{model.z_period:.2f} A, block span "
                    f"{model.z_span:.2f} A) -- the self-energy would "
                    f"couple beyond adjacent cells "
                    f"(engines/transport.md 3).  Label more electrode "
                    f"layers on the junction and re-relax, or re-label "
                    f"and re-cite.")
            model.notes.append(
                f"principal-layer condition MEASURED: orbital reach "
                f"{reach:.2f} A fits the {gap:.2f} A between "
                f"next-nearest cells (from "
                f"{', '.join(el + '.ion' for el in elems)}).")
    return models


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
    relaxed, start, ion_dir, deck_text, files, said = parts

    sorted_res = sort_by(relaxed, "transport")
    dev = sorted_res.structure

    # The electrode models, extracted from the SORTED blocks -- and the
    # § 3 lead gates, which is the same step: a block that is not frozen
    # bulk does not become a model (`wizard.extract_electrode_model`).
    #
    # The starting geometry goes in the SORTED device's index order, so
    # the gate compares atom i against atom i; `sorted_to_original[new] =
    # old` is exactly the indexing `categorical_sort` used to build the
    # sorted positions.  The same tuple travels as `atom_ids` so a
    # refusal names atoms the way the person's own file does.
    prior = (None if start is None
             else start[list(sorted_res.sorted_to_original)])
    elec_l, elec_r = _extract_and_gate_electrodes(
        dev, prior_positions=prior,
        atom_ids=sorted_res.sorted_to_original, ion_dir=ion_dir)

    provenance = {
        "schema": "molbuilder/slot-provenance@1",
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
        sorted=sorted_res,
        relaxed=relaxed,
        electrode_left=elec_l,
        electrode_right=elec_r,
        deck_text=deck_text,
        provenance=provenance,
    )


def _pair_parts(cited: CitedPair, swap_electrodes: bool):
    """A cited pair, ready to sort: ``(junction, start, ion_dir, deck_text,
    files, said)`` -- every frame with its labels (the swap applied to this
    calculation's copy), no starting geometry (a pair holds the one it was
    saved with), no ``.ion`` directory and no deck (none beside a pair), its
    two files' hashes, and its frame count for the record."""
    struct, source = labeled_citation_structure(cited)
    if swap_electrodes:
        struct = _with_swapped_leads(struct, Path(source).name)
    _bad = _unusable_cell(struct)
    if _bad:
        raise ComposeError(
            f"the cited pair {cited.path.name} {_bad} -- a junction needs its "
            f"lattice (science/junction-cell.md).")
    files = {f.name: _sha256(f) for f in (cited.path, cited.sidecar)}
    return struct, None, None, None, files, {"frames": cited.n_frames}


def _run_parts(cited: CitedRun, citation: str, swap_electrodes: bool):
    """A cited run, ready to sort: ``(junction, start, ion_dir, deck_text,
    files, said)`` -- what it left, read through the run door
    (:func:`_run_geometry`): its last geometry step with its labels and
    record, and its first step's coordinates for the lead gate's *frozen
    means unmoved*; the run's folder for the ``.ion`` files; its deck
    verbatim, the record's copy; the hashes of the output read and of the
    deck; and how the run ended."""
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
    files = {f.name: _sha256(f) for f in (output, deck)}
    said = {
        # HOW THE CITED RUN ENDED -- its record's line -- and what it
        # converged, the run door's reading of its output (§ 3.1): a
        # geometry cited with `geometry NO` is cited knowingly, and the
        # geometry it hands over is then the last one SIESTA wrote.
        "evidence": cited.concluded,
        "relaxation": {"converged": cited.converged,
                       "exit_code": cited.exit_code},
    }
    return struct, start, cited.path, deck.read_text(), files, said


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
    from ..persist import write_json
    from ..workingcopy_structure import StructureCodec
    base_dir = Path(base_dir)
    StructureCodec().write(composed.sorted.structure,
                           base_dir / JUNCTION_GEOMETRY)
    if composed.deck_text is not None:
        (base_dir / JUNCTION_DECK).write_text(composed.deck_text)
    write_json(base_dir / PROVENANCE_FILE, composed.provenance)
    write_permutation(base_dir, composed.sorted)

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
               if pinned.get(f.name) != _sha256(f)]
    if changed:
        raise ComposeError(
            f"the cited pair {citation!r} has changed since this calculation "
            f"was composed from it: {', '.join(changed)} no longer "
            f"{'holds' if len(changed) == 1 else 'hold'} the bytes "
            f"{PROVENANCE_FILE} pinned.  This calculation's stages ran on "
            f"the pair as it was; cite the pair as it is now in a new "
            f"transport calculation (`jobset init`), or restore it "
            f"(engines/transport.md 3.1).")


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

    *tree_root* is what keeps the principal-layer half of those gates
    working here: the orbital ranges live in a CITED run's ``.ion`` files,
    which the travelled folder does not carry.  Without it the condition
    degrades honestly to UNVERIFIED (a note on the model) rather than
    silently passing.  With it, a cited PAIR is held to the bytes it was
    composed from (§ 3.1): one edited since is refused by name, because the
    seed and the leads already ran on what it said then.
    """
    base_dir = Path(base_dir)
    prov_path = base_dir / PROVENANCE_FILE
    # WHICH FORM decides which files the record needs, and the form is
    # in the provenance -- so read that first, then ask `record_files`
    # for the set.  Incomplete in any of them = NO RECORD, which is the
    # answer that makes prep compose fresh instead of failing.
    def _no(reason: str):
        if why is not None:
            why.append(reason)
        return None

    if not prov_path.is_file():
        return _no(f"there is no {PROVENANCE_FILE} beside it, so nothing "
                   f"here has been composed yet")
    provenance = json.loads(prov_path.read_text())
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
    perm = read_permutation(base_dir)
    from ..workingcopy_structure import StructureCodec
    dev = StructureCodec().load(base_dir / JUNCTION_GEOMETRY, frames=True)
    sorted_res = SortResult(
        structure=dev,
        original_to_sorted=perm.original_to_sorted,
        sorted_to_original=perm.sorted_to_original,
        key=perm.key)
    ion_dir = None
    if tree_root is not None:
        try:
            cited_path, cited = resolve_citation(citation, Path(tree_root))
        except ComposeError:
            cited = None        # the citation moved: UNVERIFIED, honestly
        if isinstance(cited, CitedPair):
            _pair_unchanged(cited, provenance, citation)
        elif cited is not None:
            ion_dir = cited_path
    # NO `prior_positions` HERE, and that is not an omission: this
    # rebuilds from a junction that was already composed, and the
    # geometry the relaxation started from is not part of the record --
    # the unmoved comparison ran when the record was written.  The frozen
    # DECLARATION is asked separately and does re-run here, as does
    # even-spacing.  Both need only the structure.
    elec_l, elec_r = _extract_and_gate_electrodes(
        dev, atom_ids=sorted_res.sorted_to_original, ion_dir=ion_dir)
    deck_text = deck_path.read_text() if kind == "run" else None
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        deck_text=deck_text,
        provenance=provenance,
    )
