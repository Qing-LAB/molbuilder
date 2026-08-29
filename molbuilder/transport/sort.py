"""The categorical sort — `plans/transport-design.md` § 4.1a.

TranSIESTA identifies electrode atoms by POSITION in the device's atom
list (Brandbyge et al., PRB 65, 165401 (2002) § III; the 4.1+
``elec-pos`` form still demands consecutive atoms in the electrode
calculation's order).  Relaxation order stays free — the source-file
order is the atom IDENTITY everywhere else — so this module produces,
at transport prep, a sorted COPY in the canonical layout:

    [ buffer ][ L-electrode ][ bridge ][ R-electrode ][ buffer ]

Pure by design: ``(Structure) -> (sorted Structure, permutation)``, no
I/O, no engine knowledge.  The physics is invariant under the
permutation (a similarity transformation — § 4.1a records the proof
sketch); what this module must get right is bookkeeping, so every
index-carrying field (``regions``, ``frozen_atoms``, ``annotations``,
the per-atom parallel arrays) is remapped through one map, and the
result is checked to be a bijection before it is returned.

Refusals name atoms (user ruling Q3): an atom with no partition label,
or with two, has no place in the canonical order and TranSIESTA would
misassign it silently — exactly the failure this module exists to make
impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.transport import (REGION_BRIDGE, REGION_BUFFER,
                                REGION_LEFT_ELECTRODE,
                                REGION_RIGHT_ELECTRODE)
from ..structure import Structure, remap_annotations

#: The four PARTITION labels — exactly one per atom.  Everything else
#: (``interface``, ``frozen_atoms``, user labels) is partition-neutral
#: bookkeeping that rides along untouched.
PARTITION_LABELS = (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE,
                    REGION_BRIDGE, REGION_BUFFER)

#: The transport axis is the THIRD cell axis (z) — the same convention
#: the emitter already uses (electrode regions sorted by z-centroid,
#: ``semi-inf-direction ±A3``).
TRANSPORT_AXIS = 2

#: The permutation sidecar's schema name (registered: job-contracts.md
#: § 6.1's Atom-permutation row, with P6's record work).
PERMUTATION_SCHEMA = "molbuilder/atom-permutation@1"


class SortError(ValueError):
    """A structure the canonical order cannot be built from — the
    message names the atoms and the rule, ready to surface verbatim."""


@dataclass(frozen=True)
class SortResult:
    structure: Structure
    #: ``original_to_sorted[i]`` = where original atom ``i`` now sits.
    original_to_sorted: Tuple[int, ...]
    #: ``sorted_to_original[j]`` = which original atom sits at ``j``.
    sorted_to_original: Tuple[int, ...]

    def sidecar(self) -> dict:
        """The recorded permutation — what maps every downstream index
        (forces, Mulliken, PDOS, the 1-based numbers in the files) back
        to the relaxation's identities."""
        return {
            "schema": PERMUTATION_SCHEMA,
            "original_to_sorted": list(self.original_to_sorted),
            "sorted_to_original": list(self.sorted_to_original),
        }


def _atom_word(struct: Structure, i: int) -> str:
    return f"atom {i} ({struct.elements[i]})"


def _partition_of(struct: Structure) -> List[str]:
    """One partition label per atom, or the two named refusals."""
    n = len(struct.elements)
    membership: List[List[str]] = [[] for _ in range(n)]
    for label in PARTITION_LABELS:
        for i in struct.regions.get(label, ()):
            membership[i].append(label)

    unlabeled = [i for i in range(n) if not membership[i]]
    if unlabeled:
        shown = ", ".join(_atom_word(struct, i) for i in unlabeled[:8])
        more = f" and {len(unlabeled) - 8} more" if len(unlabeled) > 8 else ""
        raise SortError(
            f"{len(unlabeled)} atom(s) carry no partition label: {shown}"
            f"{more}.  Every atom must be exactly one of "
            f"{', '.join(PARTITION_LABELS)} (`interface` and other "
            f"labels ride on top and do not count) -- an unlabeled atom "
            f"has no place in TranSIESTA's atom order and would be "
            f"misassigned silently.")

    doubled = [i for i in range(n) if len(membership[i]) > 1]
    if doubled:
        shown = "; ".join(
            f"{_atom_word(struct, i)}: {' + '.join(membership[i])}"
            for i in doubled[:8])
        more = f" and {len(doubled) - 8} more" if len(doubled) > 8 else ""
        raise SortError(
            f"{len(doubled)} atom(s) carry more than one partition "
            f"label: {shown}{more}.  The partition must be disjoint -- "
            f"TranSIESTA puts each atom in exactly one block.")

    return [membership[i][0] for i in range(n)]


def categorical_sort(struct: Structure) -> SortResult:
    """The canonical transport order, from labels — § 4.1a.

    * electrode blocks: layer-major along the transport axis, then the
      transverse coordinates — deterministic, and the extracted
      electrode cell inherits exactly this order, which is what makes
      the device-block ↔ electrode-calculation correspondence hold by
      construction;
    * bridge: the ORIGINAL relative order (stable) — no physics reads
      it, and stability keeps the user's mental map of their molecule;
    * buffer: outermost, each atom to the end it is nearest along the
      transport axis.
    """
    part = _partition_of(struct)
    pos = np.asarray(struct.positions, dtype=float)
    z = pos[:, TRANSPORT_AXIS]

    by = {label: [i for i, p in enumerate(part) if p == label]
          for label in PARTITION_LABELS}
    left, right = by[REGION_LEFT_ELECTRODE], by[REGION_RIGHT_ELECTRODE]
    if not left or not right or not by[REGION_BRIDGE]:
        missing = [lab for lab in (REGION_LEFT_ELECTRODE, REGION_BRIDGE,
                                   REGION_RIGHT_ELECTRODE) if not by[lab]]
        raise SortError(
            f"the partition is missing {', '.join(missing)} -- a "
            f"2-terminal junction needs all three (buffer is optional).")

    # Labels must agree with geometry: L below R along the transport
    # axis, or the extracted leads face the wrong way.
    if float(np.mean(z[left])) >= float(np.mean(z[right])):
        raise SortError(
            f"the {REGION_LEFT_ELECTRODE} block sits ABOVE the "
            f"{REGION_RIGHT_ELECTRODE} block along the transport axis "
            f"(z centroids {np.mean(z[left]):.3f} vs "
            f"{np.mean(z[right]):.3f} A) -- the labels disagree with "
            f"the geometry.  Swap the labels, or reorient the cell.")

    def _layer_major(idx: List[int]) -> List[int]:
        return sorted(idx, key=lambda i: (z[i], pos[i, 0], pos[i, 1]))

    mid = 0.5 * (float(z.min()) + float(z.max()))
    buf_lo = _layer_major([i for i in by[REGION_BUFFER] if z[i] <= mid])
    buf_hi = _layer_major([i for i in by[REGION_BUFFER] if z[i] > mid])

    # Buffer means OUTSIDE (§ 3, buffer sanity): padding beyond the
    # electrode blocks, excluded from the NEGF region.  A buffer atom
    # sitting at or inside its electrode block would be sorted to an
    # outer end its geometry contradicts -- the same labels-vs-geometry
    # class as the L-above-R refusal above.
    misplaced = ([i for i in buf_lo
                  if z[i] >= min(float(z[j]) for j in left) - 1e-6]
                 + [i for i in buf_hi
                    if z[i] <= max(float(z[j]) for j in right) + 1e-6])
    if misplaced:
        shown = "; ".join(
            f"{_atom_word(struct, i)} at z={z[i]:.3f} A"
            for i in misplaced[:6])
        more = f" and {len(misplaced) - 6} more" if len(misplaced) > 6 else ""
        raise SortError(
            f"{len(misplaced)} buffer atom(s) sit AT or INSIDE the "
            f"electrode blocks along the transport axis: {shown}{more}.  "
            f"Buffer means padding OUTSIDE the electrodes (excluded from "
            f"the NEGF region; transport-design.md 3) -- relabel the "
            f"atoms, or move them beyond the electrode blocks.")

    order = (buf_lo + _layer_major(left) + by[REGION_BRIDGE]
             + _layer_major(right) + buf_hi)

    # The bijection check (user: "make sure nothing is missed") -- done
    # mechanically even though the construction above cannot fail it:
    # this is the guard that outlives refactors.
    n = len(struct.elements)
    if sorted(order) != list(range(n)):
        raise SortError(
            f"internal error: the sort is not a bijection over {n} "
            f"atoms -- nothing was written.")

    old_to_new = {old: new for new, old in enumerate(order)}

    def _take(seq):
        return None if seq is None else [seq[i] for i in order]

    sorted_struct = Structure(
        elements=[struct.elements[i] for i in order],
        positions=pos[order].copy(),
        atom_names=_take(struct.atom_names),
        residue_ids=_take(struct.residue_ids),
        residue_names=_take(struct.residue_names),
        chain_ids=_take(struct.chain_ids),
        title=struct.title,
        regions={label: sorted(old_to_new[i] for i in idx)
                 for label, idx in struct.regions.items()},
        frozen_atoms=(None if struct.frozen_atoms is None else
                      sorted(old_to_new[i] for i in struct.frozen_atoms)),
        cell=None if struct.cell is None else struct.cell.copy(),
        pbc=struct.pbc,
        axis_kind=struct.axis_kind,
        vacuum=struct.vacuum,
        annotations=remap_annotations(struct.annotations, old_to_new),
    )
    return SortResult(
        structure=sorted_struct,
        original_to_sorted=tuple(old_to_new[i] for i in range(n)),
        sorted_to_original=tuple(order),
    )
