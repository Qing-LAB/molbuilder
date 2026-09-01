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
from .preflight import _BOHR_ANG, parse_fdf_params
from .sort import SortResult, categorical_sort
from .wizard import ElectrodeModel, extract_electrode_model


class ComposeError(Exception):
    """A citation the composition cannot honour — the message names
    exactly what to run (or fix) first, ready to surface verbatim."""


#: How far a FROZEN atom may sit from its source position before the
#: constraint is judged broken (§ 3: *frozen means unmoved*).  Real
#: constrained relaxations reproduce fixed atoms to writing precision;
#: this absorbs unit round-trips (Å → Bohr → Å through the ``.XV``),
#: never a physical drift.
FROZEN_TOL_ANG = 1e-3

#: The composed record's on-disk names, beside the transport
#: calculation's ``task.json`` (§ 4.1: the cited structure is COPIED in
#: with provenance, and the folder then travels like any other).
JUNCTION_GEOMETRY = "junction.xyz"          # the SORTED junction (codec pair)
JUNCTION_DECK = "junction.cited.fdf"        # the attempt's own deck, verbatim
PROVENANCE_FILE = "slot-provenance.json"
PERMUTATION_FILE = "atom-permutation.json"


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


def read_xv(path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """SIESTA's ``.XV`` → ``(cell_ang (3,3), elements, positions_ang)``.

    The format is SIESTA's own: three cell rows (vector + velocity, in
    Bohr), an atom count, then one row per atom —
    ``species_index  Z  x y z  vx vy vz`` (Bohr).  The atom ORDER is the
    deck's order, which is the source structure's order — that identity
    is why the overlay in :func:`compose_junction` is a plain
    positional replacement.
    """
    path = Path(path)
    lines = [ln.split() for ln in path.read_text().split("\n") if ln.strip()]
    if len(lines) < 4:
        raise ComposeError(f"{path} is not a .XV file (too short)")
    try:
        cell = np.array([[float(x) for x in lines[i][:3]]
                         for i in range(3)]) * _BOHR_ANG
        n = int(lines[3][0])
        rows = lines[4:4 + n]
        if len(rows) != n:
            raise ValueError(f"declares {n} atoms, carries {len(rows)}")
        elements = [symbol_for_z(int(r[1])) for r in rows]
        pos = np.array([[float(x) for x in r[2:5]]
                        for r in rows]) * _BOHR_ANG
    except (ValueError, IndexError) as exc:
        raise ComposeError(f"{path} does not parse as a .XV file: {exc}")
    return cell, elements, pos


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
    from ..jobset.materialize import attempt_concluded
    cite_dir = Path(cite_dir)
    decks = sorted(cite_dir.glob("*.fdf"))
    xvs = sorted(cite_dir.glob("*.XV"))
    xyzs = sorted(cite_dir.glob("*.xyz"))
    pairs = [x for x in xyzs
             if (cite_dir / (x.name[: -len(".xyz")] + ".molstruct.json")
                 ).is_file()]

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
        concluded = attempt_concluded(cite_dir, deck.stem)
        # THE ENGINE'S OWN GOODBYE COUNTS (4.1b: evidence is FILES,
        # never a marker spelling of ours).  SIESTA writes 0_NORMAL_EXIT
        # as its last act on a clean exit -- a run that carries it RAN
        # TO ITS OWN END whatever wrapper (or no wrapper) launched it.
        # molbuilder's own marker still answers first because it carries
        # the rc line.
        if concluded is None and (cite_dir / "0_NORMAL_EXIT").is_file():
            concluded = "0_NORMAL_EXIT (SIESTA's own clean-exit marker)"
        # A molbuilder attempt mid-run HAS record files that do not
        # conclude; attempt_concluded answers None for both that and
        # no-record-at-all.  Tell them apart by the files themselves --
        # classification only RECORDS the state (describing ahead of a
        # running relax is legal); COMPOSING from it refuses (strict
        # composition, ruling Q2 -- compose_junction).
        has_record = (concluded is not None
                      or bool(list(cite_dir.glob("run.json"))
                              + list(cite_dir.glob("*.concluded"))))
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
            sidecar=cite_dir / (xyz.name[: -len(".xyz")]
                                + ".molstruct.json"))

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
    try:
        raw = json.loads(cited.sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = (raw.get("info") or {}).get("calculation")         if isinstance(raw.get("info"), dict) else None
    if (isinstance(block, dict)
            and isinstance(block.get("contract"), dict)
            and block["contract"]):
        return block
    return None


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
    from ..script_emit import apply_inbody_atom_metadata
    from ..sidecars.molstruct import apply_to_structure
    from ..sidecars.molstruct import load as load_sidecar

    if cited.form == "structure":
        from ..workingcopy_structure import StructureCodec
        return StructureCodec().load(cited.xyz), cited.sidecar

    cell, xv_elements, xv_pos = read_xv(cited.xv)
    struct = Structure(elements=list(xv_elements), positions=xv_pos.copy())
    struct.cell = cell
    deck_text = cited.deck.read_text()
    if apply_inbody_atom_metadata(struct, deck_text):
        return struct, cited.deck

    sidecars = sorted(cited.path.glob("*.molstruct.json"))
    if len(sidecars) > 1:
        raise ComposeError(
            f"the cited deck {cited.deck.name} carries no ATOM-METADATA "
            f"block and {cited.path} holds {len(sidecars)} "
            f".molstruct.json files -- ambiguous; keep the one "
            f"that labels this relaxation.")
    if len(sidecars) == 1:
        apply_to_structure(struct, load_sidecar(sidecars[0]))
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
    if source.name.endswith(".molstruct.json"):
        data = json.loads(source.read_text())
        data["regions"] = _swapped(data.get("regions"))
        _write_atomically(source, json.dumps(data, indent=2) + "\n")
        return source.name

    from ..parse.scripts.atom_metadata import _extract_atom_metadata_dict
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


def _extract_and_gate_electrodes(dev: Structure, *, ion_dir=None):
    """Extract both leads from the sorted device and run the § 3 gates
    that live at the LEAD level, **tiling first and then the
    principal-layer condition** — the second is computed from numbers
    the first validates (the frozen gate ran before the overlay; label
    completeness is the sort's).  Refusals name the block and the
    numbers (user ruling Q3).

    The principal-layer condition compares the orbital INTERACTION
    RANGE against the lead's PERIOD (§ 3's own wording) -- and the
    ranges are READ, never guessed: SIESTA leaves ``<El>.ion`` beside
    every run, and *ion_dir* (the cited directory) is where they are
    looked for.  No readable ``.ion`` for an element -> the condition
    is honestly UNVERIFIED (a note on the model; TranSIESTA verifies
    lead connectivity itself at run time), never a refusal on a number
    nobody measured."""
    from ..cell import LAYER_TOL_ANG, detect_layers
    from ..parse.ion import max_orbital_rc_ang
    try:
        models = (extract_electrode_model(dev, REGION_LEFT_ELECTRODE),
                  extract_electrode_model(dev, REGION_RIGHT_ELECTRODE))
    except ValueError as exc:
        raise ComposeError(
            f"the labeled electrode block cannot serve as a lead: {exc}")
    for model in models:
        # ---- TILING FIRST, and the order is a DEPENDENCY, not taste.
        # Every number the principal-layer condition below uses is
        # derived from the layer spacing: the period is span + median
        # interlayer, so a block whose label boundary cuts a partial
        # layer has a meaningless median, a meaningless period, and a
        # meaningless gap.  Checked second, that block was refused for
        # "orbital range exceeds the gap" -- a true statement about
        # invented numbers, and the wrong thing to go and fix.
        layer_z = detect_layers(model.positions[:, 2], LAYER_TOL_ANG)
        gaps = [layer_z[i + 1] - layer_z[i]
                for i in range(len(layer_z) - 1)]
        odd = [g for g in gaps
               if abs(g - model.d_interlayer) > LAYER_TOL_ANG]
        if odd:
            raise ComposeError(
                f"the {model.label} block does not TILE along the "
                f"transport axis: its layers sit at spacings "
                f"{', '.join(f'{g:.3f}' for g in gaps)} A (median "
                f"{model.d_interlayer:.3f} A), so repeating the block "
                f"does not reproduce a bulk lead "
                f"(transport-design.md 3).  The label boundary likely "
                f"cuts a partial layer -- re-label the block on whole "
                f"bulk layers.")

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
        if struct.cell is None:
            raise ComposeError(
                f"the cited pair {cited.xyz.name} + "
                f"{cited.sidecar.name} carries no cell -- a junction "
                f"needs its lattice (science/junction-cell.md).  Set "
                f"the cell in the sidecar (the Modify tab's Cell page "
                f"writes it), then cite again.")
        cell = np.asarray(struct.cell, dtype=float)
        xv_pos = np.asarray(struct.positions, dtype=float)
        deck = xv_path = None
        deck_text = None
        params = None
        concluded = None
        # Form B's labels are the pair's own sidecar, which is already
        # in `cited` -- listed in the provenance below like every other
        # file the citation consists of.
        label_source = None
        recorded = recorded_contract_of(cited)
    else:
        # ---- form A: deck + .XV, everything from the same directory --
        deck, xv_path, concluded = cited.deck, cited.xv, cited.concluded
        deck_text = deck.read_text()
        recorded = None
        # PARSED ONCE.  The gates below and the returned snapshot are
        # the same reading of the same bytes; a second parse in the
        # return was a second answer free to drift from the one the
        # gates ran on.
        params = parse_fdf_params(deck_text)

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
        cell = np.asarray(struct.cell, dtype=float)
        xv_elements = list(struct.elements)
        xv_pos = np.asarray(struct.positions, dtype=float)

        if params.n_atoms is not None and params.n_atoms != len(xv_elements):
            raise ComposeError(
                f"the deck {deck.name} declares {params.n_atoms} atoms "
                f"but {xv_path.name} carries {len(xv_elements)} -- the "
                f"two files do not describe the same relaxation.")

        # frozen means unmoved (§ 3, ruling Q3) -- start = the deck's
        # own coordinates, end = the .XV (4.1b: the gate is form A's).
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
        moved = []
        for label in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
            for i in struct.regions.get(label, ()):
                d = float(np.linalg.norm(xv_pos[i] - src_pos[i]))
                if d > FROZEN_TOL_ANG:
                    moved.append((i, struct.elements[i], label, d))
        if moved:
            shown = "; ".join(f"atom {i} ({el}, {lab}) moved {d:.4f} A"
                              for i, el, lab, d in moved[:6])
            more = f" and {len(moved) - 6} more" if len(moved) > 6 else ""
            raise ComposeError(
                f"{len(moved)} electrode atom(s) MOVED during the cited "
                f"relaxation: {shown}{more}.  Frozen means unmoved "
                f"(transport-design.md 3, ruling Q3): the electrode "
                f"blocks are the seam the self-energies attach to.  "
                f"Re-relax the junction with the electrode atoms "
                f"constrained, or fix the labels.")

    relaxed = Structure(
        elements=list(struct.elements),
        positions=xv_pos.copy(),
        atom_names=struct.atom_names,
        residue_ids=struct.residue_ids,
        residue_names=struct.residue_names,
        chain_ids=struct.chain_ids,
        title=struct.title,
        regions={k: list(v) for k, v in struct.regions.items()},
        frozen_atoms=(None if struct.frozen_atoms is None
                      else list(struct.frozen_atoms)),
        cell=cell,
        pbc=struct.pbc,
        axis_kind=struct.axis_kind,
        vacuum=struct.vacuum,
        annotations=dict(struct.annotations),
    )

    sorted_res = categorical_sort(relaxed)
    dev = sorted_res.structure

    # the electrode models, extracted from the SORTED blocks -- the
    # wizard's analysis (layer period, thickness floor, lateral cell
    # from the device) is the § 3 tiling/thickness gate in code form.
    elec_l, elec_r = _extract_and_gate_electrodes(dev, ion_dir=cite_dir)

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


def load_compose_record(base_dir, *, citation: str, tree_root=None
                        ) -> Optional[ComposedJunction]:
    """The travelled copy, loaded back — or ``None`` when there is no
    complete record for THIS citation (prep then composes fresh).

    The record answers for the citation it was made from: a
    ``task.json`` re-pointed at a different attempt must NOT keep
    serving the old copy, so a citation mismatch reads as *no record*.
    The § 3 lead gates re-run on the loaded structure (cheap, pure);
    the frozen gate does not — it compared against the CITED source,
    which is exactly what a travelled folder no longer has, and the
    provenance records that it passed when the copy was made.

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
    elec_l, elec_r = _extract_and_gate_electrodes(dev, ion_dir=ion_dir)
    deck_text = deck_path.read_text() if deck_path.is_file() else None
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        fdf_params=(parse_fdf_params(deck_text) if deck_text else None),
        deck_text=deck_text,
        provenance=provenance,
        form=form,
        recorded_contract=provenance.get("recorded_contract"),
    )
