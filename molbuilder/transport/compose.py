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
    #: the parameter snapshot read from the attempt's own deck
    fdf_params: object
    #: the attempt's deck TEXT, verbatim — the fdf that actually ran is
    #: the truth about a result, so the copy that travels is the file
    #: itself, re-parseable anywhere (user ruling 2026-08-28)
    deck_text: str
    #: citation · resolved paths · content hashes — written beside the
    #: copies so a result can always say which junction built it
    provenance: Dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_citation(citation: str, tree_root: Path) -> Tuple[Path, Path]:
    from ..projects import OutsideRoot, contain
    from ..task import FILENAME as TASK_FILENAME
    calc_rel, _, attempt_rel = citation.partition("@")
    try:
        calc_dir = contain(tree_root / calc_rel, tree_root)
    except OutsideRoot as exc:
        raise ComposeError(
            f"the junction citation {citation!r} leaves the projects tree: "
            f"{exc}")
    if not (calc_dir / TASK_FILENAME).is_file():
        raise ComposeError(
            f"the junction citation names {calc_rel!r}, but "
            f"{calc_dir / TASK_FILENAME} does not exist -- there is no "
            f"calculation there to compose from.  Run `jobset init` for "
            f"the junction first, or fix the citation.")
    attempt_dir = calc_dir / attempt_rel
    if not attempt_dir.is_dir():
        raise ComposeError(
            f"the cited attempt {attempt_rel!r} does not exist under "
            f"{calc_rel}.  Strict composition (transport-design.md, "
            f"ruling Q2): run the junction first --\n"
            f"  cd {calc_dir} && molbuilder jobset prep run <stage> "
            f"&& molbuilder jobset launch run <stage>")
    return calc_dir, attempt_dir


def _attempt_artifacts(calc_dir: Path, attempt_dir: Path,
                       citation: str) -> Tuple[Path, Path, str]:
    """The attempt's deck, its ``.XV``, and its concluded rc line."""
    from ..jobset.materialize import attempt_concluded
    decks = sorted(attempt_dir.glob("*.fdf"))
    if not decks:
        raise ComposeError(
            f"the cited attempt {attempt_dir} holds no .fdf -- it was "
            f"never prepped; run the junction first (ruling Q2).")
    deck = decks[0]
    concluded = attempt_concluded(attempt_dir, deck.stem)
    if concluded is None:
        raise ComposeError(
            f"the cited attempt {citation!r} has not CONCLUDED -- it is "
            f"still running, or it was force-stopped (the two look "
            f"identical on disk; project-layout.md 1.6).  Let it finish, "
            f"or re-run the junction; transport never decides this over "
            f"you (ruling Q2).")
    xvs = sorted(attempt_dir.glob("*.XV"))
    if not xvs:
        raise ComposeError(
            f"the cited attempt concluded but left no .XV under "
            f"{attempt_dir} -- no relaxed geometry to compose from.  "
            f"(A SIESTA relaxation writes <SystemLabel>.XV; a run that "
            f"died before its first step may not.)  Re-run the junction "
            f"or cite a different attempt.")
    return deck, xvs[0], concluded


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
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..workingcopy_structure import StructureCodec

    tree_root = Path(tree_root)
    calc_dir, attempt_dir = _resolve_citation(citation, tree_root)
    deck, xv_path, concluded = _attempt_artifacts(calc_dir, attempt_dir,
                                                  citation)

    cited_task = read_task(calc_dir / TASK_FILENAME)
    source = calc_dir / cited_task.structure.source
    if not source.is_file():
        raise ComposeError(
            f"the cited calculation's source structure "
            f"{cited_task.structure.source!r} is missing from {calc_dir} "
            f"-- the portable folder is incomplete.")
    struct = StructureCodec().load(source)

    cell, xv_elements, xv_pos = read_xv(xv_path)
    if xv_elements != list(struct.elements):
        raise ComposeError(
            f"the attempt's .XV does not describe the cited source "
            f"structure: elements differ ({xv_path.name} vs "
            f"{source.name}).  The .XV order is the deck's order, which "
            f"is the source's -- a mismatch means the attempt belongs to "
            f"a different structure.")

    # frozen means unmoved (§ 3) -- checked BEFORE the overlay, against
    # the source the labels were drawn on.
    src_pos = np.asarray(struct.positions, dtype=float)
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
            f"(transport-design.md 3, ruling Q3): the electrode blocks "
            f"are the seam the self-energies attach to.  Re-relax the "
            f"junction with the electrode atoms constrained, or fix the "
            f"labels.")

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
        "calculation": str(calc_dir.relative_to(tree_root)),
        "attempt": str(attempt_dir.relative_to(calc_dir)),
        "concluded": concluded,
        "deck": deck.name,
        "deck_sha256": _sha256(deck),
        "xv": xv_path.name,
        "xv_sha256": _sha256(xv_path),
    }
    deck_text = deck.read_text()
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=relaxed,
        electrode_left=elec_l,
        electrode_right=elec_r,
        fdf_params=parse_fdf_params(deck_text),
        deck_text=deck_text,
        provenance=provenance,
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
    (base_dir / JUNCTION_DECK).write_text(composed.deck_text)
    write_json(base_dir / PROVENANCE_FILE, composed.provenance)
    write_json(base_dir / PERMUTATION_FILE, composed.sorted.sidecar())
    return [JUNCTION_GEOMETRY, JUNCTION_DECK,
            PROVENANCE_FILE, PERMUTATION_FILE]


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
    if not all(p.is_file() for p in paths.values()):
        return None
    provenance = json.loads(paths[PROVENANCE_FILE].read_text())
    if provenance.get("citation") != citation:
        return None
    perm = json.loads(paths[PERMUTATION_FILE].read_text())
    from ..workingcopy_structure import StructureCodec
    dev = StructureCodec().load(paths[JUNCTION_GEOMETRY])
    sorted_res = SortResult(
        structure=dev,
        original_to_sorted=tuple(perm["original_to_sorted"]),
        sorted_to_original=tuple(perm["sorted_to_original"]))
    elec_l, elec_r = _extract_and_gate_electrodes(dev)
    deck_text = paths[JUNCTION_DECK].read_text()
    return ComposedJunction(
        sorted=sorted_res,
        relaxed=None,
        electrode_left=elec_l,
        electrode_right=elec_r,
        fdf_params=parse_fdf_params(deck_text),
        deck_text=deck_text,
        provenance=provenance,
    )
