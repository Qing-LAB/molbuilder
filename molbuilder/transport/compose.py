"""P4a — prep COMPOSES the transport calculation from its one citation.

`plans/transport-design.md` § 4.1–4.2: resolve the junction citation on
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
from .wizard import (MIN_ELECTRODE_THICKNESS_ANG, ElectrodeModel,
                     extract_electrode_model)


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


def _extract_and_gate_electrodes(dev: Structure):
    """Extract both leads from the sorted device and run the § 3 gates
    that live at the LEAD level: **tiling** and **principal-layer
    thickness** (the frozen gate ran before the overlay; label
    completeness is the sort's).  Refusals name the block and the
    numbers (user ruling Q3)."""
    from ..cell import LAYER_TOL_ANG, detect_layers
    try:
        models = (extract_electrode_model(dev, REGION_LEFT_ELECTRODE),
                  extract_electrode_model(dev, REGION_RIGHT_ELECTRODE))
    except ValueError as exc:
        raise ComposeError(
            f"the labeled electrode block cannot serve as a lead: {exc}")
    for model in models:
        # Thick enough -- the principal-layer condition.
        if model.z_span < MIN_ELECTRODE_THICKNESS_ANG:
            raise ComposeError(
                f"the {model.label} block spans {model.z_span:.2f} A "
                f"along transport -- thinner than the "
                f"~{MIN_ELECTRODE_THICKNESS_ANG:.0f} A principal-layer "
                f"floor, so the lead's self-energy would couple beyond "
                f"adjacent cells (transport-design.md 3).  Label more "
                f"electrode layers on the junction and re-relax, or "
                f"re-label and re-cite.")
        # The block must tile: repeating it along transport must
        # reproduce a bulk lead, i.e. the layers sit at ONE spacing.
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
    return models


def compose_junction(citation: str, *, tree_root) -> ComposedJunction:
    """The whole § 4.1–4.2 compose: citation → sorted, gated, extracted.

    Raises :class:`ComposeError` (refusals naming what to run first) or
    :class:`~molbuilder.transport.sort.SortError` (the § 4.1a label
    gates) — the caller surfaces either verbatim.
    """
    from ..sidecars.molstruct import apply_to_structure, load as load_sidecar
    from ..script_emit import apply_inbody_atom_metadata
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
        concluded = None
        recorded = recorded_contract_of(cited)
    else:
        # ---- form A: deck + .XV, everything from the same directory --
        deck, xv_path, concluded = cited.deck, cited.xv, cited.concluded
        deck_text = deck.read_text()
        recorded = None
        params = parse_fdf_params(deck_text)
        cell, xv_elements, xv_pos = read_xv(xv_path)

        # The labeled source structure, from THIS directory (4.1b): the
        # deck's own in-body ATOM-METADATA block first; else exactly one
        # .molstruct.json beside it.
        struct = Structure(elements=list(xv_elements),
                           positions=xv_pos.copy())
        struct.cell = cell
        labeled = apply_inbody_atom_metadata(struct, deck_text)
        if not labeled:
            sidecars = sorted(cite_dir.glob("*.molstruct.json"))
            if len(sidecars) == 1:
                apply_to_structure(struct, load_sidecar(sidecars[0]))
                labeled = bool(struct.regions)
            elif len(sidecars) > 1:
                raise ComposeError(
                    f"the cited deck {deck.name} carries no ATOM-METADATA "
                    f"block and {cite_dir} holds {len(sidecars)} "
                    f".molstruct.json files -- ambiguous; keep the one "
                    f"that labels this relaxation.")
        if not labeled or not struct.regions:
            raise ComposeError(
                f"the cited relaxation in {cite_dir} carries no region "
                f"labels: the deck {deck.name} has no in-body "
                f"ATOM-METADATA block and no .molstruct.json sits beside "
                f"it.  Transport derives the electrodes FROM the labels "
                f"(L-electrode / R-electrode; transport-design.md 4.1b) "
                f"-- relabel and re-relax through molbuilder, or put the "
                f"structure's .molstruct.json in the same directory.")

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
    elec_l, elec_r = _extract_and_gate_electrodes(dev)

    provenance = {
        "schema": "molbuilder/slot-provenance@1",
        "slot": "junction",
        "citation": citation,
        "form": cited.form,
        # The 4.1b files this junction was composed from, with hashes --
        # a result can always say which bytes built it.
        "files": {f.name: _sha256(f)
                  for f in (deck, xv_path, cited.xyz, cited.sidecar)
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
        fdf_params=(parse_fdf_params(deck_text) if deck_text else None),
        deck_text=deck_text,
        provenance=provenance,
        form=cited.form,
        recorded_contract=recorded,
    )


def write_compose_record(base_dir, composed: ComposedJunction) -> List[str]:
    """The composed junction, ON DISK beside the transport calculation's
    ``task.json`` — § 4.1's *"the cited structure is COPIED in with
    provenance"*, whole: the SORTED structure pair (the geometry every
    stage renders from), the attempt's own deck verbatim (the electronic
    contract, re-parseable anywhere), and the two sidecars.  With these
    four the folder travels: :func:`load_compose_record` rebuilds the
    junction on a machine where the cited tree does not exist."""
    from ..persist import write_json
    from ..workingcopy_structure import StructureCodec
    base_dir = Path(base_dir)
    StructureCodec().write(composed.sorted.structure,
                           base_dir / JUNCTION_GEOMETRY)
    written = [JUNCTION_GEOMETRY]
    if composed.deck_text is not None:
        (base_dir / JUNCTION_DECK).write_text(composed.deck_text)
        written.append(JUNCTION_DECK)
    write_json(base_dir / PROVENANCE_FILE, composed.provenance)
    write_json(base_dir / PERMUTATION_FILE, composed.sorted.sidecar())
    return written + [PROVENANCE_FILE, PERMUTATION_FILE]


def load_compose_record(base_dir, *, citation: str
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
    """
    base_dir = Path(base_dir)
    paths = {name: base_dir / name
             for name in (JUNCTION_GEOMETRY, JUNCTION_DECK,
                          PROVENANCE_FILE, PERMUTATION_FILE)}
    required = [n for n in paths if n != JUNCTION_DECK]
    if not all(paths[n].is_file() for n in required):
        return None
    provenance = json.loads(paths[PROVENANCE_FILE].read_text())
    if provenance.get("citation") != citation:
        return None
    # A form-A record without its travelled deck is incomplete (the
    # contract travels as the file itself); a form-B record never had
    # one.
    form = provenance.get("form", "relaxation")
    if form == "relaxation" and not paths[JUNCTION_DECK].is_file():
        return None
    perm = json.loads(paths[PERMUTATION_FILE].read_text())
    from ..workingcopy_structure import StructureCodec
    dev = StructureCodec().load(paths[JUNCTION_GEOMETRY])
    sorted_res = SortResult(
        structure=dev,
        original_to_sorted=tuple(perm["original_to_sorted"]),
        sorted_to_original=tuple(perm["sorted_to_original"]))
    elec_l, elec_r = _extract_and_gate_electrodes(dev)
    deck_text = (paths[JUNCTION_DECK].read_text()
                 if paths[JUNCTION_DECK].is_file() else None)
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
