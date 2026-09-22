"""P4a — prep COMPOSES the transport calculation from its one citation.

`archive/2026-09-01-transport-design.md` § 4.1–4.2: resolve the junction citation on
the machine where prep runs (strict composition, ruling Q2 — a missing
or unconcluded attempt is a refusal naming what to run first, never a
trigger to run it); PARSE the relaxed geometry from the attempt's own
``.XV`` (Bohr → Å — never file-copied: an old-order ``.XV`` is exactly
what the § 4.1a fence forbids crossing the sort); overlay it on the
cited calculation's labeled source structure; run the categorical sort
(P2); apply the frozen-unmoved gate; extract the two electrode models
from the sorted blocks (the wizard's move, § 4.2); and record the
provenance — citation, attempt, content hashes, and the parameter
snapshot read from **the attempt's own deck** (the fdf that actually
ran is the truth about a result; user ruling 2026-08-28).

Pure composition: everything here reads the tree and returns objects;
the caller (prep's transport arm, P4b) owns what lands on disk where.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..chemistry import symbol_for_z
from ..config.transport import (REGION_LEFT_ELECTRODE,
                                REGION_RIGHT_ELECTRODE)
from ..structure import Structure
from ..parse.fdf import _BOHR_ANG, parse_fdf_params
from .sort import SortResult, categorical_sort
from ..units import UnknownUnit
from .wizard import ElectrodeModel, extract_electrode_model


class ComposeError(Exception):
    """A citation the composition cannot honour — the message names
    exactly what to run (or fix) first, ready to surface verbatim."""


#: The composed record's on-disk names, beside the transport
#: calculation's ``task.json`` (§ 4.1: the cited structure is COPIED in
#: with provenance, and the folder then travels like any other).
JUNCTION_GEOMETRY = "junction.xyz"          # the SORTED junction (codec pair)
JUNCTION_DECK = "junction.cited.fdf"        # the attempt's own deck, verbatim
PROVENANCE_FILE = "slot-provenance.json"
PERMUTATION_FILE = "atom-permutation.json"

#: How far two statements of the SAME cell may differ before the citation
#: is refused.  The cell travels deck -> SIESTA -> ``.XV``: this project
#: writes ``LatticeVectors`` at twelve decimals in Angstrom, SIESTA works
#: in Bohr and writes the ``.XV`` in Bohr, and the two Bohr constants
#: differ in their last digits -- a relative error around 1e-11, so under
#: 1e-9 A on any cell a junction has.  1e-6 A is three orders above that
#: floor and far below any difference that means a different box.
CELL_AGREEMENT_TOL_ANG = 1e-6


def record_files(form: str = "relaxation") -> Tuple[str, ...]:
    """WHAT THE TRAVELLED RECORD CONSISTS OF — one answer, read by the
    write and by the load, so they cannot disagree about whether a copy
    is complete.

    It used to be spelled three times (what ``write_compose_record``
    puts down, what it *says* it put down, and what
    ``load_compose_record`` requires), and the three disagreed: the
    codec writes the geometry as a PAIR, and the file carrying the
    region labels was in none of the lists.  A record whose labels had
    been deleted therefore passed the completeness check and loaded a
    junction with no electrodes -- dying inside the lead gates instead
    of answering "incomplete, compose again".

    The geometry's label file is not named literally: it is whatever
    the codec pairs with :data:`JUNCTION_GEOMETRY`, asked of the codec's
    own rule (`sidecars.molstruct.sidecar_path_for`).
    """
    from ..sidecars.molstruct import sidecar_path_for
    always = (JUNCTION_GEOMETRY,
              sidecar_path_for(Path(JUNCTION_GEOMETRY)).name,
              PROVENANCE_FILE, PERMUTATION_FILE)
    # A form-A record travels with the deck (the contract IS the file);
    # a form-B citation never had one.
    return always + ((JUNCTION_DECK,) if form == "relaxation" else ())


def _unusable_cell(struct) -> Optional[str]:
    """Why this structure's cell cannot carry a junction — or ``None``.

    ONE QUESTION, BOTH FORMS.  Form A spelled this out and form B asked only
    *"is the cell None?"*, which was enough for as long as
    ``Structure.__post_init__`` refused a zero-volume lattice outright.  That
    refusal was removed on 2026-09-21 so a pair holding a bad box could be
    OPENED and corrected on the Cell page (§ 8.2, *"reading does not judge"*)
    -- and form B had been relying on it without saying so.  Measured: a
    form-B sidecar whose ``cell`` is a row of zeros loads and passes the
    ``is None`` guard.

    **It does not crash, and the first telling of this said it did.**
    ``transiesta.axis_vacuum`` inverts the cell unguarded and does raise
    ``LinAlgError`` when called directly -- but nothing reaches it with a
    bad box: ``_emit_geometry`` is only ever a ``Block`` in a deck layout,
    and ``script_emit.render_deck`` runs ``report(validate(...))`` before the
    first block renders, so the deck path answers *"[cell.no_volume] This box
    is flat (8 x 8 x 0 A)"*.  Both the original review and its cross-check
    asserted the traceback from the function in isolation without tracing the
    call, which is the § 1d step-0 mistake in miniature.

    What this guard is actually worth is WHERE and IN WHOSE WORDS the refusal
    lands: at the citation door, naming the cited pair, the way form A has
    always done -- rather than surviving to the deck writer to be described
    as a problem with a box, several steps from the file that holds it.

    A junction needs a real box (`science/junction-cell.md`): the transverse
    vectors set the k-mesh and the image separation, and the transport vector
    is the device length.  None of those exist in a degenerate cell.
    """
    if struct.cell is None:
        return "states no cell"
    c = np.asarray(struct.cell, dtype=float)
    if c.shape != (3, 3) or not np.all(np.isfinite(c)):
        return "states a cell that is not three finite vectors"
    # The ONE threshold, from the module that owns the question -- not a
    # fourth literal (`cell.py`'s header records the era of four).
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

    THE PARSE MODULE'S READER, reshaped.  This used to be a second, complete
    `.XV` parser -- same name, different return type, sitting beside
    `parse/coords/siesta_xv.py`, whose own first paragraph says "this is the
    only `.XV` reader".  `constants.py` records what the pair already cost:
    the two carried different Bohr radii, so "the same file gave coordinates
    4e-7 apart depending on which reader was asked".  Unifying the constant
    fixed that number and left both parsers standing; this removes the
    second one.

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
    #: the parameter snapshot read from the cited deck — ``None`` for a
    #: form-B citation (§ 4.1b: a labeled structure carries no contract)
    fdf_params: Optional[object]
    #: the cited deck TEXT, verbatim — the fdf that actually ran is the
    #: truth about a result, so the copy that travels is the file itself,
    #: re-parseable anywhere (user ruling 2026-08-28).  ``None`` for a
    #: form-B citation: the electronic contract is then the description's
    #: own (its contract fields are OPEN, § 4.1b)
    deck_text: Optional[str]
    #: citation · resolved paths · content hashes — written beside the
    #: copies so a result can always say which junction built it
    provenance: Dict[str, object]
    #: which § 4.1b form the citation satisfied — "relaxation" (A) or
    #: "structure" (B)
    form: str = "relaxation"
    #: a form-B pair's RECORDED contract (`info.calculation` in its
    #: sidecar — the Results tab wrote it from the finished run's own
    #: deck; structure-info-plan.md I5/I6).  When present the contract
    #: fields seal exactly as form A's do; ``None`` = the open lane.
    recorded_contract: Optional[Dict[str, object]] = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CitedDir:
    """The § 4.1b classification of a cited directory — WHICH form its
    files satisfy, and with which files.  Layout, names and tree
    position play no part (user ruling 2026-08-29)."""
    path: Path
    form: str                     # "relaxation" (A) | "structure" (B)
    deck: Optional[Path] = None   # form A: the one .fdf
    xv: Optional[Path] = None     # form A: the one .XV
    xyz: Optional[Path] = None    # form B: the one .xyz
    sidecar: Optional[Path] = None    # form B: its .molstruct.json
    #: the run record's concluded line, when a record exists in the
    #: directory; ``None`` with ``has_record=False`` means "no record —
    #: the .XV is taken as the final geometry" (said honestly, § 4.1b)
    concluded: Optional[str] = None
    has_record: bool = False


#: The § 4.1b condition, in one sentence — used verbatim by every
#: refusal so the user always learns the WHOLE condition, not just the
#: half they tripped on.
CITATION_CONDITION = (
    "a citable directory holds EITHER a finished relaxation -- exactly "
    "one .fdf and exactly one .XV together -- OR a labeled structure -- "
    "exactly one .xyz with its .molstruct.json beside it "
    "(transport-design.md 4.1b)")


def classify_citation(cite_dir: Path) -> CitedDir:
    """Classify a directory against the § 4.1b file condition.

    Raises :class:`ComposeError` naming exactly which file is missing
    (or ambiguous) when the directory satisfies neither form.  Form A
    wins when both are present — the deck carries the contract, and
    more information never loses to less.
    """
    from ..jobset.materialize import RUN_LAUNCH_FILE, attempt_concluded
    from ..runfiles import find_by_role
    cite_dir = Path(cite_dir)
    # THE DECK IS OURS AND THE REST IS NOT, and the two halves of this
    # condition ask accordingly (`project-layout.md` § 4.5).  `.fdf` is a role
    # `runfiles.WRITTEN` declares, so the catalogue searches for it.  `.XV` is
    # SIESTA's own restart file and a bare `.xyz` is a person's structure --
    # neither is a name molbuilder composes, so neither has a door here, and
    # `WRITTEN` deliberately does not enumerate what an engine writes.
    decks = find_by_role(cite_dir, ".fdf")
    xvs = sorted(p for p in cite_dir.glob("*.XV") if p.is_file())
    xyzs = sorted(p for p in cite_dir.glob("*.xyz") if p.is_file())
    # THE PAIR IS COMPOSED BY THE MODULE THAT OWNS THE SUFFIX
    # (`sidecars.molstruct.sidecar_path_for`), not by slicing `.xyz` off a
    # name here -- and the slice was subtly its own rule: it stripped exactly
    # ``.xyz`` where the composer strips the LAST suffix, so the two agreed
    # only for names ending in `.xyz`, which is the only case reached.
    from ..sidecars.molstruct import sidecar_path_for
    pairs = [x for x in xyzs if sidecar_path_for(x).is_file()]

    if decks and xvs:
        if len(decks) > 1:
            raise ComposeError(
                f"{cite_dir} holds {len(decks)} .fdf files "
                f"({', '.join(d.name for d in decks)}) -- the citation "
                f"names a directory, so the directory must answer "
                f"unambiguously.  Keep one deck, or cite a directory "
                f"holding one.")
        if len(xvs) > 1:
            raise ComposeError(
                f"{cite_dir} holds {len(xvs)} .XV files "
                f"({', '.join(x.name for x in xvs)}) -- ambiguous; keep "
                f"the relaxation's own one.")
        deck = decks[0]
        # DECK-SCOPED, and that is the whole question.  `attempt_concluded`
        # is asked about THIS deck; `run_status` answers about the DIRECTORY.
        # They look like one question and are not -- measured 2026-09-18: in a
        # directory holding a neighbour rung at a higher attempt index, the
        # directory answer reports that rung's `rc=1 (walltime)` for a
        # citation whose own run concluded `rc=0`, and the tab renders
        # CONCLUDED (rc=1) for a clean relaxation.  It also cost 3,000x the
        # runtime (a full parse of every `.out`, discarded) and appended a
        # `.parse.log` into the person's finished directory on every browse.
        concluded = attempt_concluded(cite_dir, deck.stem)
        # THE ENGINE'S OWN GOODBYE COUNTS.  `transport.md`: *"evidence is
        # FILES, never a marker spelling of ours"* -- SIESTA writes
        # `0_NORMAL_EXIT` as its last act on a clean exit, so a run carrying
        # it ran to its own end whatever launched it, or nothing did.
        # molbuilder's own marker answers first because it carries the rc.
        #
        # `attempt_concluded` CANNOT answer this: the marker has no label.
        # It belongs here rather than there because form A has already
        # required exactly one `.fdf` and one `.XV` above -- which is the
        # unambiguous directory an unlabelled marker needs.
        #
        # RESTORED 2026-09-18.  `dada356a` deleted these lines when it moved
        # this to `run_status`; the revert put the call back and not the
        # fallback, so `attempt_concluded` answered None for every
        # SIESTA-only run.  Measured: 5 of 5 citable directories in the
        # checkout refused, the tab said "NOT CONCLUDED -- still running, or
        # force-stopped" for a finished relaxation, and prep declined the
        # citation.  A shipped test was failing on main.
        if concluded is None and (cite_dir / "0_NORMAL_EXIT").is_file():
            concluded = "0_NORMAL_EXIT"
        # A molbuilder attempt mid-run HAS record files that do not
        # conclude; `attempt_concluded` answers None for both that and
        # no-record-at-all.  This third clause is LOAD-BEARING, not
        # decoration: it is what separates "still running" from "no run
        # record", and `compose_junction` gates on the difference.  Tell them apart by the files themselves --
        # classification only RECORDS the state (describing ahead of a
        # running relax is legal); COMPOSING from it refuses (strict
        # composition, ruling Q2 -- compose_junction).
        # `run.json` is ONE NAME, so it is asked as one -- it globbed a
        # literal with no wildcard in it.  `.concluded` is the catalogue's, so
        # the catalogue finds it.
        has_record = (concluded is not None
                      or (cite_dir / RUN_LAUNCH_FILE).is_file()
                      or bool(find_by_role(cite_dir, ".concluded")))
        return CitedDir(path=cite_dir, form="relaxation", deck=deck,
                        xv=xvs[0], concluded=concluded,
                        has_record=has_record)

    if pairs:
        if len(pairs) > 1:
            raise ComposeError(
                f"{cite_dir} holds {len(pairs)} .xyz+.molstruct.json "
                f"pairs ({', '.join(x.name for x in pairs)}) -- "
                f"ambiguous; keep one, or cite a directory holding one.")
        xyz = pairs[0]
        return CitedDir(
            path=cite_dir, form="structure", xyz=xyz,
            sidecar=sidecar_path_for(xyz))

    # Neither form: name what IS there and what the condition wants.
    held = []
    if decks:
        held.append(f"{len(decks)} .fdf but no .XV")
    if xvs and not decks:
        held.append(f"{len(xvs)} .XV but no .fdf")
    if xyzs and not pairs:
        held.append(f"{len(xyzs)} .xyz but no stem-matched "
                    f".molstruct.json")
    what = "; ".join(held) if held else "none of the required files"
    raise ComposeError(
        f"{cite_dir} is not citable: it holds {what}.  "
        f"{CITATION_CONDITION}.")


def recorded_contract_of(cited: CitedDir) -> Optional[Dict[str, object]]:
    """A form-B pair's ``info.calculation`` block, when its sidecar
    carries one with a usable ``contract`` dict — else ``None``.
    ONE reader for compose and both web doors, so the lanes cannot
    disagree about what counts as recorded."""
    if cited.form != "structure" or cited.sidecar is None:
        return None
    # THROUGH THE DOOR.  `labeled_structure_from` in this same module reads
    # sidecars with `molstruct.load`; this one hand-parsed, twenty lines away,
    # under a docstring promising "ONE reader".  `load` validates the envelope
    # and reads `utf-8-sig`.  `MolstructJsonError` is a ValueError, so "else
    # None" is unchanged for a sidecar that is missing or malformed.
    from ..sidecars import molstruct as _molstruct
    try:
        raw = _molstruct.load(cited.sidecar)
    except (OSError, ValueError):
        return None
    block = (raw.get("info") or {}).get("calculation")         if isinstance(raw.get("info"), dict) else None
    if (isinstance(block, dict)
            and isinstance(block.get("contract"), dict)
            and block["contract"]):
        return block
    return None


def _warn_about_edits_since_the_contract_was_recorded(
        recorded: Optional[Dict[str, object]], citation: str) -> None:
    """Say what an edit since the recording invalidated — one line each.

    **The settings are inherited; what they were chosen for may not be.**
    A form-B citation fills the new calculation's mesh cutoff, transverse
    k-mesh, functional and temperature from `info.calculation` — the
    finished run's own deck, copied into the pair by the Results tab. Two
    flags say what an edit since then touched (`molview.md` § 8.4a), and
    they invalidate different things, which is why they are two:

    * ``structure_modified`` — a geometry or cell op. Mesh cutoff is a grid
      density over the CELL and the transverse k-mesh samples the reciprocal
      cell, so both were converged for a geometry that is no longer there.
    * ``labels_modified`` — a label write. No setting is a function of a
      name, so the settings stand. But on a junction the electrode/device
      partition IS labels, so which atoms were the left electrode, the
      device and the frozen set may now differ from what was relaxed — and
      the categorical sort downstream reads exactly those labels.

    Warn, not refuse, for both: trimming a stray solvent molecule and
    renaming a region are each legitimate things to do to a relaxed
    structure, and refusing would block them. The person is told and
    decides.

    **Neither can fire where nothing is inherited.** A flag is written only
    onto a structure that already carries an `info.calculation` block, and
    `recorded_contract_of` answers ``None`` unless that block holds a
    non-empty ``contract`` — so a structure that never came from a run
    (SMILES, a plain `.xyz`, anything built in Modify) reaches neither end
    of this. Nothing to inherit, nothing to be stale about.

    *(Written 2026-09-07. The flag had been set since 2026-08-29 and read by
    NOTHING -- its only consumer in the tree was a test grepping the JS
    source for its own name. Splitting it in two was the precondition: one
    flag covered label writes as well, so acting on it meant warning that a
    mesh cutoff might not apply because someone renamed a region.)*
    """
    if not recorded:
        return
    from ..issues import Issue
    from ..validation import report
    engine = recorded.get("engine", "?")
    source = recorded.get("source", "?")
    found = []
    if recorded.get("structure_modified"):
        found.append(Issue(
            "warn",
            f"the geometry or cell of {citation} was edited after its "
            f"settings were recorded, so the mesh cutoff and transverse "
            f"k-mesh below come from the {engine} deck {source} and were "
            f"converged for a cell that is no longer there -- re-check them "
            f"against the structure you are citing",
            where="citation.structure_modified"))
    if recorded.get("labels_modified"):
        found.append(Issue(
            "warn",
            f"the labels of {citation} were edited after its settings were "
            f"recorded -- the settings still stand, but the electrode and "
            f"device regions this calculation sorts on are labels, so check "
            f"they are still the partition the {engine} deck {source} "
            f"relaxed",
            where="citation.labels_modified"))
    if found:
        report(found)


def resolve_citation(citation: str, tree_root: Path
                     ) -> Tuple[Path, CitedDir]:
    """The citation's directory, fenced to the tree and classified
    against the § 4.1b file condition.  Public since P7b: the web
    hand-over validates a citation through the SAME door prep composes
    through."""
    from ..projects import OutsideRoot, contain
    try:
        cite_dir = contain(tree_root / citation, tree_root)
    except OutsideRoot as exc:
        raise ComposeError(
            f"the junction citation {citation!r} leaves the projects "
            f"tree: {exc}")
    if not cite_dir.is_dir():
        raise ComposeError(
            f"the junction citation {citation!r} is not a directory "
            f"under the projects tree.  The citation names a directory "
            f"whose FILES satisfy the condition: {CITATION_CONDITION}.")
    return cite_dir, classify_citation(cite_dir)


def labeled_citation_structure(cited: CitedDir):
    """The cited directory's LABELED structure, and where its labels
    live -- ``(structure, source)`` with *source* the deck (in-body
    block) or the ``.molstruct.json`` that carries them.

    ONE door, because three callers must agree about which labels are
    real: the composition itself, the orientation question the tab
    asks before composing, and the swap that rewrites them.  A second
    reading with its own precedence is how a tab offers to fix a file
    that is not the one being read.

    Form A's precedence is the deck's own block FIRST, then exactly one
    sidecar beside it (4.1b); form B is the pair.
    """
    from ..script_emit import (_extract_atom_metadata_dict,
                               apply_atom_metadata)
    from ..sidecars.molstruct import (MolstructPairingError,
                                      apply_to_structure)
    from ..sidecars.molstruct import load as load_sidecar

    if cited.form == "structure":
        from ..workingcopy_structure import StructureCodec
        return StructureCodec().load(cited.xyz), cited.sidecar

    cell, xv_elements, xv_pos = read_xv(cited.xv)
    # STATED AT CONSTRUCTION, AND Z IS TRANSPORT.  No cited file records
    # `axis_kind`: the ATOM-METADATA block carries `regions` + `annotations`
    # only, the `.XV` carries the cell, and SIESTA has no such concept.  It
    # does not need recording -- `engines/transport.md` § 5 I8 settles it for
    # every transport run: z is open (kz = 1, the leads enter as self-energies
    # Σ, and the engine preflight refuses kz != 1) while x and y are the
    # transverse periodic mesh.
    #
    # Assigning `.cell` afterwards instead skipped `__post_init__`, so the
    # box arrived unvalidated and every axis stayed `isolated`: the emitted
    # electrode deck then read `pbc` and printed "the transport axis (c) has
    # vacuum / is not periodic; the electrode .TSHS cannot attach seamlessly"
    # on a junction that is periodic in-plane and open along z by design.
    try:
        struct = Structure(elements=list(xv_elements), positions=xv_pos.copy(),
                           cell=cell,
                           axis_kind=("periodic", "periodic", "transport"))
    except ValueError as exc:
        # Live now that the cell goes through the constructor: `prep` catches
        # only ComposeError/SortError, so a bare ValueError would surface as
        # a traceback.
        raise ComposeError(
            f"{cited.xv.name} states a cell transport cannot use: {exc}")
    deck_text = cited.deck.read_text()

    # THE DECK SET THE BOX AND THE .XV CAME BACK WITH IT.  A fixed-cell
    # relaxation cannot move it, so a disagreement means these two files
    # are not from one run -- the case a stray `.XV` left in the
    # directory produces.  Checked before the labels, because a label
    # applied to the wrong geometry is the worse failure.
    from ..parse.fdf import parse_fdf_params as _parse_params
    from ..units import UnknownUnit as _UnknownUnit
    try:
        _deck_cell = _parse_params(deck_text, source=cited.deck.name).cell_ang
    except _UnknownUnit:
        _deck_cell = None      # the unit refusal is compose_junction's to raise
    _cells_agree_or_refuse(f"the deck {cited.deck.name}", _deck_cell,
                           cited.xv.name, cell)
    block = _extract_atom_metadata_dict(deck_text)
    try:
        if block is not None and apply_atom_metadata(struct, block):
            return struct, cited.deck
    except MolstructPairingError as exc:
        # THE SAME DISAGREEMENT the deck-vs-.XV gate below names, reaching us
        # one step earlier because the deck's label block states the count
        # too.  One guard, in the reader; the file names are the context only
        # this caller has, so they are added here rather than duplicated
        # there.
        raise ComposeError(
            f"the deck {cited.deck.name} and {cited.xv.name} do not describe "
            f"the same relaxation: {exc}")

    # Through the finder (§ 4.5); this globbed the suffix a second time.
    from ..sidecars.molstruct import sidecars_in
    sidecars = sidecars_in(cited.path)
    if len(sidecars) > 1:
        raise ComposeError(
            f"the cited deck {cited.deck.name} carries no ATOM-METADATA "
            f"block and {cited.path} holds {len(sidecars)} "
            f".molstruct.json files -- ambiguous; keep the one "
            f"that labels this relaxation.")
    if len(sidecars) == 1:
        _side = load_sidecar(sidecars[0])
        # `apply_to_structure` is a full REPLACE of the metadata block --
        # cell included -- so the sidecar's box would silently displace
        # the relaxation's.  It is the same junction, so the two must
        # already agree; if they do not, one file is not from this run.
        _cells_agree_or_refuse(sidecars[0].name, _side.get("cell"),
                               cited.xv.name, cell)
        # COMPLETE THE BLOCK, DO NOT PATCH THE RESULT.  `apply_metadata_dict`
        # is a full replace and `model/structure.md` § 2.2 says what an absent
        # key means: absent `cell` -> non-periodic.  So a sidecar that records
        # labels but no box was applied as "no box", `__post_init__` reconciled
        # every axis to isolated, and setting `.cell` back afterwards restored
        # the box but not the periodicity -- a junction emitted with an
        # explicit cell and `pbc = (False, False, False)`.
        #
        # The relaxation's own box is the box, and z is transport
        # (`engines/transport.md` § 5 I8), so both are stated here and the one
        # authority applies them together.
        #
        # AND THE CORNER IS THIS FRAME'S, NOT THE AUTHORING PAIR'S.  These
        # coordinates came from the `.XV` -- SIESTA's own frame, which anchors
        # the cell at (0,0,0) -- while the sidecar's `cell_origin` is the
        # low corner of the box the
        # author drew around DIFFERENT coordinates.  A full replace adopts it,
        # and `render_fdf` then shifts these atoms by `-cell_origin` a second
        # time: a junction saved from `add_slab` came out translated by its
        # whole authoring corner, far-face atoms wrapping into the leads.
        # The cell above is a SHAPE and survives the change of frame; the
        # origin does not, so it is stripped here -- explicitly, which is what
        # `model/structure.md` § 2.2a asks of a field that does not travel.
        #
        # STRIPPED, NOT SET TO ZERO.  `null` does not mean "the corner is
        # (0,0,0)" -- it means DERIVE it (`structure-periodicity.md` § 6
        # clause 2a), and for a `.XV`, whose atoms are already inside
        # [0, cell), the derivation answers "no shift".  If a relaxation
        # drifted an atom outside the box the derivation wraps it back in,
        # which is right and is what an explicit zero would have prevented.
        apply_to_structure(struct, {
            **_side,
            "cell": _side.get("cell") or [[float(x) for x in row]
                                          for row in cell],
            "cell_origin": None,
            "axis_kind": _side.get("axis_kind")
                         or ["periodic", "periodic", "transport"],
        })
        if struct.regions:
            return struct, sidecars[0]
    raise ComposeError(
        f"the cited relaxation in {cited.path} carries no region "
        f"labels: the deck {cited.deck.name} has no in-body "
        f"ATOM-METADATA block and no .molstruct.json sits beside "
        f"it.  Transport derives the electrodes FROM the labels "
        f"(L-electrode / R-electrode; transport-design.md 4.1b) "
        f"-- relabel and re-relax through molbuilder, or put the "
        f"structure's .molstruct.json in the same directory.")


def swap_electrode_labels(cited: CitedDir) -> str:
    """Rename ``L-electrode`` ↔ ``R-electrode`` on the CITED files.
    Answers the name of the file that changed.

    The person agrees to this in the tab -- it edits their finished
    run's label block, and nothing else.  What moves is two arrays of
    indices in molbuilder's own metadata; no coordinate, no engine
    keyword, no result is touched, which is why a relabel does not
    invalidate the relaxation it annotates.  Renaming at the SOURCE
    (rather than compensating inside the composite) is what makes every
    later citation of the same directory read the same way.

    NO GEOMETRY IS CONSULTED (user ruling, 2026-08-29).  A swap is a
    rename, and whether the labels *should* be the other way round is
    the author's judgement about their own experiment -- the tab warns
    and offers, this performs.  The only condition is that both labels
    exist, because otherwise there is no pair to rename.
    """
    from ..script_emit import (BLOCK_ATOM_METADATA, begin_marker,
                               emit_atom_metadata, end_marker)

    _struct, source = labeled_citation_structure(cited)

    def _swapped(regions):
        out = dict(regions or {})
        for lab in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
            if lab not in out:
                raise ComposeError(
                    f"{source.name} does not carry {lab}, so there is "
                    f"no pair here to swap.")
        out[REGION_LEFT_ELECTRODE], out[REGION_RIGHT_ELECTRODE] = (
            list(out[REGION_RIGHT_ELECTRODE]),
            list(out[REGION_LEFT_ELECTRODE]))
        return out

    # THE FILE THAT CARRIES THE LABELS IS THE FILE THAT CHANGES.  Which
    # one that is came from the same door that read them, so the swap
    # can never rewrite a block the composition does not read (form A
    # accepts either an in-body block OR a sidecar beside the deck).
    from ..sidecars.molstruct import is_sidecar
    if is_sidecar(source):
        # THE SIDECAR'S OWN READER AND WRITER, AND ITS LOCK.  This was a raw
        # `json.loads` / `json.dumps` pair: no envelope validation, no
        # `encoding=` on the read, and -- the one that bites silently --
        # `json.dumps` without `ensure_ascii=False`, which `molstruct.dumps`
        # documents as "what keeps a non-ASCII region label a literal instead
        # of an escape, so a second writer without it produces a different
        # file for the same structure".  Swapping the electrodes of a junction
        # labelled `α-helix` rewrote that label escaped.
        #
        # And it is a read-modify-write, which `save` says must hold the lock:
        # "if you're doing a read-modify-write cycle, wrap the entire cycle in
        # `with_lock`".
        from ..sidecars import molstruct as _molstruct
        with _molstruct.with_lock(source):
            data = _molstruct.load(source)
            data["regions"] = _swapped(data.get("regions"))
            _molstruct.save(source, data)
        return source.name

    from ..script_emit import _extract_atom_metadata_dict
    text = source.read_text(encoding="utf-8")
    payload = _extract_atom_metadata_dict(text)
    if payload is None:
        raise ComposeError(
            f"{source.name} carries no atom-metadata block, so "
            f"there are no labels here to swap.")
    n_atoms = payload.get("n_atoms_total")
    if not isinstance(n_atoms, int) or n_atoms <= 0:
        raise ComposeError(
            f"the atom-metadata block in {source.name} states no atom "
            f"count, so a rewrite would lose it -- refusing to touch "
            f"the file.")
    # Everything the block carried that the swap did not come to change
    # rides through verbatim: the selection rules, the extensible
    # channels, and WHEN the labels were made.  Only `created_by`
    # gains a line, because that field is the block's own record of
    # who wrote it and this write is part of that history.
    block = emit_atom_metadata(
        regions=_swapped(payload.get("regions")),
        n_atoms_total=n_atoms,
        created_by=(str(payload.get("created_by") or "molbuilder")
                    + " (L/R swapped by molbuilder transport relabel)"),
        created_at=payload.get("created_at"),
        selection_rules=payload.get("selection_rules") or None,
        annotations=payload.get("annotations") or None)
    if not block:
        raise ComposeError(
            f"the rewritten atom-metadata block came out empty -- "
            f"{source.name} was NOT changed.")
    begin, end = (begin_marker(BLOCK_ATOM_METADATA),
                  end_marker(BLOCK_ATOM_METADATA))
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ComposeError(
            f"the atom-metadata fence in {source.name} is not "
            f"where its own markers say -- refusing to rewrite it.")
    _write_atomically(source,
                      text[:i] + block.rstrip("\n") + "\n"
                      + text[j + len(end):].lstrip("\n"))
    return source.name


def _write_atomically(path: Path, text: str) -> None:
    """Same-directory temp + replace: a half-written label block would
    make a finished run unreadable to the tool that wrote it."""
    tmp = path.with_suffix(path.suffix + ".mb-tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
    compare against — form B, and the recompose-from-record path, where
    the comparison was made when the record was written.

    *atom_ids* is the sort's ``sorted_to_original``, so a refusal names
    atoms by the identity in the person's own file rather than by their
    place in TranSIESTA's deck order (`engine_atom_index`).
    """
    from ..parse.ion import max_orbital_rc_ang
    # BOTH BLOCKS, THEN RAISE.  A generator here short-circuited on the
    # first bad lead, so a junction with two broken blocks was fixed and
    # re-relaxed once per block.  The loop this consolidated accumulated
    # across both regions before raising, and that is worth keeping.
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
                    f"(transport-design.md 3).  Label more electrode "
                    f"layers on the junction and re-relax, or re-label "
                    f"and re-cite.")
            model.notes.append(
                f"principal-layer condition MEASURED: orbital reach "
                f"{reach:.2f} A fits the {gap:.2f} A between "
                f"next-nearest cells (from "
                f"{', '.join(el + '.ion' for el in elems)}).")
    return models


def compose_junction(citation: str, *, tree_root) -> ComposedJunction:
    """The whole § 4.1–4.2 compose: citation → sorted, gated, extracted.

    Raises :class:`ComposeError` (refusals naming what to run first) or
    :class:`~molbuilder.transport.sort.SortError` (the § 4.1a label
    gates) — the caller surfaces either verbatim.
    """
    from ..workingcopy_structure import StructureCodec

    tree_root = Path(tree_root)
    cite_dir, cited = resolve_citation(citation, tree_root)

    if cited.form == "relaxation" and cited.has_record \
            and cited.concluded is None:
        raise ComposeError(
            f"the cited relaxation {citation!r} has a run record but "
            f"has not CONCLUDED -- it is still running, or it was "
            f"force-stopped (the two look identical on disk; "
            f"project-layout.md 1.6).  Let it finish; transport never "
            f"decides this over you (ruling Q2).")

    if cited.form == "structure":
        # ---- form B: the labeled pair IS the final structure ---------
        struct = StructureCodec().load(cited.xyz)
        _bad = _unusable_cell(struct)
        if _bad:
            raise ComposeError(
                f"the cited pair {cited.xyz.name} + "
                f"{cited.sidecar.name} {_bad} -- a junction "
                f"needs its lattice (science/junction-cell.md).  Set "
                f"the cell in the sidecar (the Modify tab's Cell page "
                f"writes it), then cite again.")
        cell = np.asarray(struct.cell, dtype=float)
        xv_pos = np.asarray(struct.positions, dtype=float)
        deck = xv_path = None
        src_pos = None
        deck_text = None
        params = None
        concluded = None
        # Form B's labels are the pair's own sidecar, which is already
        # in `cited` -- listed in the provenance below like every other
        # file the citation consists of.
        label_source = None
        recorded = recorded_contract_of(cited)
        _warn_about_edits_since_the_contract_was_recorded(recorded, citation)
    else:
        # ---- form A: deck + .XV, everything from the same directory --
        deck, xv_path, concluded = cited.deck, cited.xv, cited.concluded
        deck_text = deck.read_text()
        recorded = None
        # PARSED ONCE.  The gates below and the returned snapshot are
        # the same reading of the same bytes; a second parse in the
        # return was a second answer free to drift from the one the
        # gates ran on.
        try:
            params = parse_fdf_params(deck_text, source=deck.name)
        except UnknownUnit as exc:
            # A unit this build cannot convert is a citation it cannot
            # honour: every number taken from the deck would be wrong by
            # a fixed ratio.  The reader's own sentence names the field.
            raise ComposeError(f"the cited deck cannot be read: {exc}")

        # The labeled source structure, from THIS directory (4.1b) --
        # through the one door the swap and the tab's orientation
        # question also read, so all three agree on which labels are
        # real and where they live.  It is also the ONLY reader of the
        # .XV on this path: a second parse here would be a second
        # answer to "what does this relaxation say", free to drift.
        #
        # KEEP THE SOURCE.  Form A's labels may come from the deck's own
        # block or from a .molstruct.json beside it, and which one it
        # was belongs in the provenance: it is a file this junction was
        # composed from, and the one the rename endpoint rewrites.
        struct, label_source = labeled_citation_structure(cited)
        # THE GUARD FORM B HAS HAD ALL ALONG.  `Structure.cell`'s setter
        # is permissive, so an unusable box travels to the construction
        # below and raises a bare ValueError there -- and `prep` catches
        # only ComposeError/SortError, so it reaches the person as a
        # traceback instead of a sentence.
        _bad = _unusable_cell(struct)
        if _bad:
            raise ComposeError(
                f"the cited relaxation in {cited.path} {_bad} -- a junction "
                f"needs its lattice (science/junction-cell.md).  The deck's "
                f"%block LatticeVectors and the .XV both carry one; if the "
                f"labels come from a .molstruct.json, its `cell` must not "
                f"be null.")
        cell = np.asarray(struct.cell, dtype=float)
        xv_elements = list(struct.elements)
        xv_pos = np.asarray(struct.positions, dtype=float)

        if params.n_atoms is not None and params.n_atoms != len(xv_elements):
            raise ComposeError(
                f"the deck {deck.name} declares {params.n_atoms} atoms "
                f"but {xv_path.name} carries {len(xv_elements)} -- the "
                f"two files do not describe the same relaxation.")

        # THE GEOMETRY THE RELAXATION STARTED FROM (4.1b), read here
        # because only form A has one: the deck's own coordinate block.
        # The extraction compares it against the .XV to decide "frozen
        # means unmoved" (§ 3, ruling Q3); without it that question has
        # no start to measure from and the refusal below says so.
        if params.coords_ang is None:
            raise ComposeError(
                f"the deck {deck.name} carries no convertible "
                f"coordinate block (AtomicCoordinatesAndAtomicSpecies "
                f"in Ang/Bohr/Fractional), so the frozen gate cannot "
                f"compare start against end.  Include coordinates in "
                f"the deck, or cite an .xyz+.molstruct.json pair.")
        src_pos = np.asarray(params.coords_ang, dtype=float)
        if len(src_pos) != len(xv_pos):
            raise ComposeError(
                f"the deck {deck.name}'s coordinate block ({len(src_pos)} "
                f"atoms) does not match {xv_path.name} ({len(xv_pos)}) -- "
                f"the two files do not describe the same relaxation.")
        # `src_pos` goes to the extraction, which asks whether these
        # atoms moved as part of deciding whether the block is a lead.

    # THE RELAXED COORDINATES AND THE BOX THEY CAME BACK IN, and nothing
    # else stated by hand.  This was a fourteen-field list that did not
    # name `cell_origin` or `info`, so the cited junction lost its stored
    # corner and its recorded contract on the way in -- then lost them
    # again in `categorical_sort` below (`model/structure.md` § 2.2a).
    relaxed = struct.replace(positions=xv_pos.copy(), cell=cell)

    sorted_res = categorical_sort(relaxed)
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
    prior = (None if src_pos is None
             else src_pos[list(sorted_res.sorted_to_original)])
    elec_l, elec_r = _extract_and_gate_electrodes(
        dev, prior_positions=prior,
        atom_ids=sorted_res.sorted_to_original, ion_dir=cite_dir)

    provenance = {
        "schema": "molbuilder/slot-provenance@1",
        "slot": "junction",
        "citation": citation,
        "form": cited.form,
        # The 4.1b files this junction was composed from, with hashes --
        # a result can always say which bytes built it.  `label_source`
        # is here because the electrode REGIONS are a fact about this
        # junction as much as its coordinates are, and on form A they
        # may live in a .molstruct.json that is in none of the other
        # slots.  (When they live in the deck, the dict keys dedupe.)
        "files": {f.name: _sha256(f)
                  for f in (deck, xv_path, cited.xyz, cited.sidecar,
                            label_source)
                  if f is not None},
        # Honest convergence evidence (4.1b): the record line when one
        # exists; "no-record" when the .XV is taken as final; "given"
        # for a cited structure pair.
        "evidence": (concluded if concluded is not None
                     else ("no-record" if cited.form == "relaxation"
                           else "given")),
    }
    if recorded is not None:
        provenance["recorded_contract"] = recorded
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=relaxed,
        electrode_left=elec_l,
        electrode_right=elec_r,
        fdf_params=params,
        deck_text=deck_text,
        provenance=provenance,
        form=cited.form,
        recorded_contract=recorded,
    )


def write_compose_record(base_dir, composed: ComposedJunction) -> List[str]:
    """The composed junction, ON DISK beside the transport calculation's
    ``task.json`` — § 4.1's *"the cited structure is COPIED in with
    provenance"*, whole: the SORTED structure PAIR (the geometry every
    stage renders from AND the file carrying its region labels), the
    attempt's own deck verbatim (the electronic contract, re-parseable
    anywhere), and the two sidecars.  With these the folder travels:
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
    write_json(base_dir / PERMUTATION_FILE, composed.sorted.sidecar())

    expected = record_files(composed.form)
    missing = [n for n in expected if not (base_dir / n).is_file()]
    if missing:
        raise ComposeError(
            f"the composed record in {base_dir} is missing "
            f"{', '.join(missing)} right after being written -- it "
            f"would not rebuild on another machine, so nothing should "
            f"rely on it.  (The geometry travels as a PAIR; its label "
            f"file is what carries the electrode regions.)")
    return list(expected)


def _params_or_none(deck_text):
    """The recorded deck's parameters, or None when it states a unit this
    build cannot convert.

    A travelled record is REBUILT rather than re-gated, so an unreadable
    unit here must not take the whole folder down -- the fields simply go
    unanswered, which is what `None` already means to every consumer.
    """
    if not deck_text:
        return None
    try:
        return parse_fdf_params(deck_text)
    except UnknownUnit:
        return None


def load_compose_record(base_dir, *, citation: str, tree_root=None
                        ) -> Optional[ComposedJunction]:
    """The travelled copy, loaded back — or ``None`` when there is no
    complete record for THIS citation (prep then composes fresh).

    The record answers for the citation it was made from: a
    ``task.json`` re-pointed at a different attempt must NOT keep
    serving the old copy, so a citation mismatch reads as *no record*.
    The § 3 lead gates re-run on the loaded structure (cheap, pure) —
    frozen-declared and evenly-spaced both need only the structure.  The
    UNMOVED comparison does not re-run: it measures against the geometry
    the relaxation started from, which is exactly what a travelled folder
    no longer carries, and the provenance records that it passed when the
    copy was made.  *(This said "the frozen gate does not", before
    2026-09-20 split the declaration from the movement and gave them
    separate names; only the second half was ever meant.)*

    *tree_root* is what keeps the principal-layer half of those gates
    working here: the orbital ranges live in the CITED directory's
    ``.ion`` files, which the travelled folder does not carry.  Without
    it the condition degrades honestly to UNVERIFIED (a note on the
    model) rather than silently passing.
    """
    base_dir = Path(base_dir)
    prov_path = base_dir / PROVENANCE_FILE
    # WHICH FORM decides which files the record needs, and the form is
    # in the provenance -- so read that first, then ask `record_files`
    # for the set.  Incomplete in any of them = NO RECORD, which is the
    # answer that makes prep compose fresh instead of failing.
    if not prov_path.is_file():
        return None
    provenance = json.loads(prov_path.read_text())
    if provenance.get("citation") != citation:
        return None
    form = provenance.get("form", "relaxation")
    if any(not (base_dir / n).is_file() for n in record_files(form)):
        return None
    deck_path = base_dir / JUNCTION_DECK
    perm = json.loads((base_dir / PERMUTATION_FILE).read_text())
    from ..workingcopy_structure import StructureCodec
    dev = StructureCodec().load(base_dir / JUNCTION_GEOMETRY)
    sorted_res = SortResult(
        structure=dev,
        original_to_sorted=tuple(perm["original_to_sorted"]),
        sorted_to_original=tuple(perm["sorted_to_original"]))
    ion_dir = None
    if tree_root is not None:
        try:
            ion_dir, _cited = resolve_citation(citation, Path(tree_root))
        except ComposeError:
            ion_dir = None      # the citation moved: UNVERIFIED, honestly
    # NO `prior_positions` HERE, and that is not an omission: this
    # rebuilds from a junction that was already composed, and the
    # geometry the relaxation started from is not part of the record.
    # For a form-A record the unmoved comparison ran when the record was
    # written; for a form B one it never ran at all, because form B has
    # no starting geometry anywhere -- which is exactly why the frozen
    # DECLARATION is asked separately, and it does re-run here, as does
    # even-spacing.  Both need only the structure.
    elec_l, elec_r = _extract_and_gate_electrodes(
        dev, atom_ids=sorted_res.sorted_to_original, ion_dir=ion_dir)
    deck_text = deck_path.read_text() if deck_path.is_file() else None
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        fdf_params=(_params_or_none(deck_text)),
        deck_text=deck_text,
        provenance=provenance,
        form=form,
        recorded_contract=provenance.get("recorded_contract"),
    )
