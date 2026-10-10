"""P4a — prep COMPOSES the transport calculation from its one citation.

`engines/transport.md` § 3.1 and § 6.2: resolve the junction citation on
the machine where prep runs (strict composition, ruling Q2 — a missing
or unconcluded attempt is a refusal naming what to run first, never a
trigger to run it); PARSE the relaxed geometry from the attempt's own
``.XV`` (Bohr → Å — never file-copied: an old-order ``.XV`` is exactly
what the § 4.1a fence forbids crossing the sort); overlay it on the
cited calculation's labeled source structure; run the categorical sort
(P2); apply the frozen-unmoved gate; extract the two electrode models
from the sorted blocks (the wizard's move, § 3); and record the
provenance — citation, attempt, and the content hashes of the files it
was composed from, **the attempt's own deck** among them (the fdf that
actually ran is the truth about a result; user ruling 2026-08-28).

Pure composition: everything here reads the tree and returns objects;
the caller (prep, `jobset.prep._composed_for_prep`) owns what lands on
disk where.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..atom_permutation import PERMUTATION_FILE, read_permutation
from ..structure import Structure
from ..parse.fdf import parse_fdf_params
from .sort import (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE,
                   SortResult, sort_by, write_permutation)
from ..units import UnknownUnit
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

#: How far two statements of the SAME cell may differ before the citation
#: is refused.  The cell travels deck -> SIESTA -> ``.XV``: this project
#: writes ``LatticeVectors`` at twelve decimals in Angstrom, SIESTA works
#: in Bohr and writes the ``.XV`` in Bohr, and the two Bohr constants
#: differ in their last digits -- a relative error around 1e-11, so under
#: 1e-9 A on any cell a junction has.  1e-6 A is three orders above that
#: floor and far below any difference that means a different box.
CELL_AGREEMENT_TOL_ANG = 1e-6


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

    ONE QUESTION, BOTH FORMS.  ``Structure`` reads a zero-volume lattice
    without refusing it (§ 8.2, *"reading does not judge"*), so a form-B
    sidecar whose ``cell`` is a row of zeros loads and passes an
    ``is None`` guard.

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


def _cells_agree_or_refuse(a_name: str, a, b_name: str, b) -> None:
    """Two statements of one cell must be the same cell.

    A junction's box is stated more than once -- the deck's
    ``LatticeVectors``, the ``.XV`` SIESTA wrote back, and a sidecar's
    ``cell`` when the labels come from one.  They describe the SAME
    relaxation, so a disagreement is not a value to choose between: one
    of the files is not from this calculation, and silently taking
    either would put a box the person never set into the transport deck,
    where it sets the transverse k-mesh and the image separation.

    ``None`` on either side is not a disagreement -- it means that file
    states no cell, which the caller handles.
    """
    if a is None or b is None:
        return
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ComposeError(
            f"{a_name} and {b_name} state cells of different shape "
            f"({aa.shape} vs {bb.shape}) -- they do not describe the same "
            f"relaxation.")
    worst = float(np.abs(aa - bb).max())
    if worst > CELL_AGREEMENT_TOL_ANG:
        raise ComposeError(
            f"{a_name} and {b_name} disagree about the cell by "
            f"{worst:.4g} A -- they do not describe the same relaxation. "
            f"The cell sets the transverse k-mesh and the image "
            f"separation, so transport will not guess which one you "
            f"meant.  Cite a directory whose files belong to one run, or "
            f"remove the file that does not.\n"
            f"  {a_name}: {np.diag(aa).round(6).tolist()} (diagonal)\n"
            f"  {b_name}: {np.diag(bb).round(6).tolist()} (diagonal)")


def read_xv(path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """SIESTA's ``.XV`` → ``(cell_ang (3,3), elements, positions_ang)``.

    THE PARSE MODULE'S READER, reshaped: `parse/coords/siesta_xv.py` is the
    only `.XV` reader.

    The TUPLE SHAPE stays, because `compose_junction`'s overlay wants the
    three arrays positionally and the atom ORDER is the deck's order -- that
    identity is why the overlay is a plain positional replacement.
    """
    from ..parse.coords.siesta_xv import SiestaXVError, read_xv_with_cell
    try:
        struct, cell = read_xv_with_cell(Path(path))
    except SiestaXVError as exc:
        raise ComposeError(f"{path} does not parse as a .XV file: {exc}")
    except OSError as exc:
        raise ComposeError(f"{path} could not be read: {exc}")
    if cell is None:                      # defensive: the strict door raises
        raise ComposeError(f"{path} carries no cell")
    return (np.asarray(cell, dtype=float),
            list(struct.elements),
            np.asarray(struct.positions, dtype=float))


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
    """The cited relaxation run (`engines/transport.md` § 3.1): the run the
    cited folder speaks for (`runs.run_of`), its deck, the ``.XV`` SIESTA
    wrote under its label, and how that run ended.  Layout, names and tree
    position play no part (user ruling 2026-08-29); being a run of ours --
    launched by `jobset`, with the wrapper's record -- does (decision 7,
    2026-10-08)."""
    path: Path
    run: "Run"                    # the run the folder speaks for
    deck: Path                    # that run's deck (`Run.deck`)
    xv: Path                      # its carried .XV (`Run.carried`)
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
    root speaks for -- its deck and its carried ``.XV``; its launch record
    and how it ended, the run record's doors."""
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
                 "it is a stage's folder, not a run -- cite one of its runs"
                 if place.role == calcdirs.CONTAINER else
                 "nothing was launched here")
        raise ComposeError(
            f"{cite_dir} holds no run of molbuilder's: {where}.  "
            f"{CITATION_CONDITION}.")
    deck = run.deck
    xv = run.carried(".XV")
    if deck is None or xv is None:
        missing = " and ".join(
            what for what, got in (("no deck", deck), ("no .XV", xv))
            if got is None)
        raise ComposeError(
            f"{cite_dir} holds the run {run.basename} with {missing}: SIESTA "
            f"writes {run.label}.XV at the relaxation's first geometry step, "
            f"beside the deck it ran.  {CITATION_CONDITION}.")
    try:
        launched = launch_record(cite_dir, run.names) is not None
    except LaunchRecordError as e:
        raise ComposeError(str(e)) from e
    # HOW THE CITED RUN ENDED, asked of the one door (`runrecord.ending`,
    # `execution/architecture.md` § 3.2) about THIS run's deck -- never the
    # directory's `run_status`, which can answer for a neighbour stage's run.
    end = ending(cite_dir, deck.stem)
    if not launched and not end.concluded:
        raise ComposeError(
            f"{cite_dir} holds {deck.name} and {xv.name}, but no run of "
            f"molbuilder's: nothing launched it and nothing recorded how it "
            f"ended.  {CITATION_CONDITION}.")
    # WHAT IT CONVERGED, the run door's reading of its output (the same the
    # status verb prints): carried into the provenance and said by the tab,
    # so a geometry that did not converge is cited knowingly.
    from ..jobset.runstatus import converged_of
    from ..parse.dirs.rundir import run_state_of
    try:
        converged = converged_of(run_state_of(cite_dir, run.names))
    except Exception:                                      # noqa: BLE001
        converged = None
    return CitedRun(path=cite_dir, run=run, deck=deck, xv=xv,
                    concluded=end.line, exit_code=end.code,
                    converged=converged)


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
    from ..deck_record import extract_engine_offset
    recorded = (extract_engine_offset(
        cited.deck.read_text(errors="replace")) or {}).get("axis_kind")
    return _with_transport(recorded or ("periodic",) * 3,
                           f"the cited deck {cited.deck.name}")


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
    from ..runs import declared
    from ..script_emit import apply_atom_metadata
    from ..sidecars.molstruct import MolstructPairingError

    cell, xv_elements, xv_pos = read_xv(cited.xv)
    # STATED AT CONSTRUCTION, AND Z IS TRANSPORT (`junction_axis_kind`):
    # along z every transport run is open (`engines/transport.md` § 5 I8);
    # across it the relaxation's deck recorded the person's answer -- a slab
    # junction periodic, a wire or chain junction isolated (§ 6.1c).  Stated
    # at construction, so `__post_init__` validates the box and reconciles
    # the axes.
    said = declared(cited.run)
    # ...and the `.XV` is the engine's frame, so it states an offset of 0 on
    # either label lane -- WHEN THE DECK RECORDED ITS PLACEMENT (§ 6.0):
    # every rung composed from it then applies nothing.  A deck with no
    # `engine-offset` record states none, and the rule centres its junction
    # -- a rigid shift, and the relaxation stays citable.
    stated = np.zeros(3) if said.engine_offset is not None else None
    try:
        struct = Structure(elements=list(xv_elements), positions=xv_pos.copy(),
                           cell=cell, axis_kind=kinds,
                           engine_offset=stated)
    except ValueError as exc:
        # `prep` catches only ComposeError/SortError, so a bare ValueError would surface as
        # a traceback.
        raise ComposeError(
            f"{cited.xv.name} states a cell transport cannot use: {exc}")

    # THE DECK SET THE BOX AND THE .XV CAME BACK WITH IT.  A fixed-cell
    # relaxation cannot move it, so a disagreement means these two files
    # are not from one run.  Checked before the labels, because a label
    # applied to the wrong geometry is the worse failure.
    from ..parse.fdf import parse_fdf_params as _parse_params
    from ..units import UnknownUnit as _UnknownUnit
    try:
        _deck_cell = _parse_params(cited.deck.read_text(),
                                   source=cited.deck.name).cell_ang
    except _UnknownUnit:
        _deck_cell = None      # the unit refusal is compose_junction's to raise
    _cells_agree_or_refuse(f"the deck {cited.deck.name}", _deck_cell,
                           cited.xv.name, cell)
    # THE LABELS ARE WHAT THE RUN DECLARED: its deck's block, read by the
    # run door (`runs.declared`) and applied by the block's one reader.
    try:
        if said.atom_metadata is not None and apply_atom_metadata(
                struct, said.atom_metadata) and struct.regions:
            return struct, cited.deck
    except MolstructPairingError as exc:
        # The deck's label block states the atom count too, so a deck and a
        # `.XV` from two relaxations disagree here first.
        raise ComposeError(
            f"the deck {cited.deck.name} and {cited.xv.name} do not describe "
            f"the same relaxation: {exc}")
    raise ComposeError(
        f"the cited relaxation in {cited.path} carries no region labels: its "
        f"deck {cited.deck.name} has no ATOM-METADATA block.  Transport "
        f"derives the electrodes FROM the labels (L-electrode / R-electrode; "
        f"engines/transport.md 3.1, 4) -- label the junction on the "
        f"Molbuilder tab and relax it through `jobset init` -> `prep task` -> "
        f"`launch task`, then cite that run.")


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
    files, said)`` -- its relaxed geometry with its deck's labels, the
    geometry it started from (the deck's coordinate block, for the lead
    gate's *frozen means unmoved*), the run's folder for the ``.ion`` files,
    the deck verbatim, the files' hashes, and how the run ended."""
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
            f"{cited.exit_code} ({cited.concluded}) -- its .XV is whatever "
            f"the engine left when it failed, not a relaxed junction.  Run "
            f"the relaxation to its end, then cite it (engines/transport.md "
            f"3.1).")

    deck, xv_path = cited.deck, cited.xv
    deck_text = deck.read_text()
    # The cited deck's own reading, for the gates below.
    try:
        params = parse_fdf_params(deck_text, source=deck.name)
    except UnknownUnit as exc:
        # A unit this build cannot convert is a citation it cannot
        # honour: every number taken from the deck would be wrong by
        # a fixed ratio.  The reader's own sentence names the field.
        raise ComposeError(f"the cited deck cannot be read: {exc}")

    # The labeled source structure, through the one door the tab's
    # orientation question also reads, so both agree on which labels are
    # real.  It is also the ONLY reader of the .XV on this path: a second
    # parse here would be a second answer to "what does this relaxation
    # say", free to drift.  A swap renames the labels on this calculation's
    # copy (`_with_swapped_leads`) and never writes the deck.
    struct, label_source = labeled_citation_structure(cited)
    if swap_electrodes:
        struct = _with_swapped_leads(struct, Path(label_source).name)
    # `Structure.cell`'s setter is permissive, so an unusable box would
    # travel to the construction below and raise a bare ValueError there --
    # and `prep` catches only ComposeError/SortError, so it would reach the
    # person as a traceback instead of a sentence.
    _bad = _unusable_cell(struct)
    if _bad:
        raise ComposeError(
            f"the cited relaxation in {cited.path} {_bad} -- a junction "
            f"needs its lattice (science/junction-cell.md).  The deck's "
            f"%block LatticeVectors and the .XV both carry one.")
    cell = np.asarray(struct.cell, dtype=float)
    xv_pos = np.asarray(struct.positions, dtype=float)

    if params.n_atoms is not None and params.n_atoms != len(xv_pos):
        raise ComposeError(
            f"the deck {deck.name} declares {params.n_atoms} atoms "
            f"but {xv_path.name} carries {len(xv_pos)} -- the "
            f"two files do not describe the same relaxation.")

    # THE GEOMETRY THE RELAXATION STARTED FROM (§ 3.1): the deck's own
    # coordinate block.  The extraction compares it against the .XV to
    # decide "frozen means unmoved" (§ 3, ruling Q3).
    if params.coords_ang is None:
        raise ComposeError(
            f"the deck {deck.name} carries no convertible "
            f"coordinate block (AtomicCoordinatesAndAtomicSpecies "
            f"in Ang/Bohr/Fractional), so the frozen gate cannot "
            f"compare start against end -- a deck molbuilder wrote "
            f"carries them; relax the junction through jobset and cite "
            f"that run (engines/transport.md 3.1).")
    src_pos = np.asarray(params.coords_ang, dtype=float)
    if len(src_pos) != len(xv_pos):
        raise ComposeError(
            f"the deck {deck.name}'s coordinate block ({len(src_pos)} "
            f"atoms) does not match {xv_path.name} ({len(xv_pos)}) -- "
            f"the two files do not describe the same relaxation.")

    # THE RELAXED COORDINATES AND THE BOX THEY CAME BACK IN, and nothing
    # else stated by hand: ``replace`` carries every other field the cited
    # junction states (`model/structure.md` § 2.2a).
    relaxed = struct.replace(positions=xv_pos.copy(), cell=cell)
    files = {f.name: _sha256(f) for f in (deck, xv_path)}
    said = {
        # HOW THE CITED RUN ENDED -- its record's line -- and what it
        # converged, the run door's reading of its output (§ 3.1): a
        # geometry cited with `geometry NO` is cited knowingly, and the
        # `.XV` it hands over is then the last geometry SIESTA wrote.
        "evidence": cited.concluded,
        "relaxation": {"converged": cited.converged,
                       "exit_code": cited.exit_code},
    }
    return relaxed, src_pos, cited.path, deck_text, files, said


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

    **`None` HAS THREE CAUSES AND THEY ARE NOT THE SAME NEWS.**  Pass a
    list as *why* and the reason is appended to it, in words for a person.
    The `retired_out` pattern (`workingcopy_structure.StructureCodec.load`):
    the happy path is unchanged and the caller that needs more asks for it.

    Without it, all three read as *"there is no record"* -- while the
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
    deck_text = deck_path.read_text() if deck_path.is_file() else None
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        deck_text=deck_text,
        provenance=provenance,
    )
