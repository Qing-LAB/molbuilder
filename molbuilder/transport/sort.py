"""The categorical sort — `plans/transport-design.md` § 4.1a.

TranSIESTA identifies electrode atoms by POSITION in the device's atom
list (Brandbyge et al., PRB 65, 165401 (2002) § III; the 4.1+
``elec-pos`` form still demands consecutive atoms in the electrode
calculation's order).  Relaxation order stays free — the source-file
order is the atom IDENTITY everywhere else — so this module produces,
at transport prep, a sorted COPY in the canonical layout:

    [ buffer ][ lower electrode ][ bridge ][ upper electrode ][ buffer ]

Which block is which is GEOMETRY: the lower one leads, because it is
the one whose self-energy extends to ``-A3``.  The ``L-electrode`` /
``R-electrode`` names do a different job — they carry the chemical
potential (`transiesta.py`) — so a junction labeled the other way
round is not an error, it is one that biases the other end, and the
sort says so in a note rather than refusing it (user, 2026-08-29).

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


#: How the two electrode blocks sit along transport, as one word.
#: ONE door for the fact, because four places need it and they must not
#: disagree: the sort's note, the engine preflight's warning, the web
#: describe (which offers the swap when the answer is ``inverted``),
#: and the swap itself.
ORDER_CANONICAL = "canonical"        # L below R -- the usual convention
ORDER_INVERTED = "inverted"          # L above R -- runs; a swap flips it
ORDER_INTERLEAVED = "interleaved"    # no two ends -- nothing to build


def electrode_orientation(struct: Structure) -> Optional[str]:
    """Which of the three shapes this structure's electrodes are in, or
    ``None`` when it has no two electrode blocks to compare.

    The usual convention (transport-design.md § 4.1a): ``L-electrode``
    is the LOW-z lead.  ``inverted`` is NOT an error — it is a valid
    junction whose author biased the other end, and it is answered as
    its own word so every surface can say so without matching prose.
    """
    regions = struct.regions or {}
    left = list(regions.get(REGION_LEFT_ELECTRODE, ()))
    right = list(regions.get(REGION_RIGHT_ELECTRODE, ()))
    if not left or not right:
        return None
    z = np.asarray(struct.positions, dtype=float)[:, TRANSPORT_AXIS]
    # INTERLEAVING IS DECIDED FIRST, and on the geometrically ordered
    # pair.  Both halves matter: a junction that both interleaves AND
    # is named the wrong way round is not fixable by a swap, and
    # calling it `inverted` would offer a relabel that leaves it just
    # as unusable; and the overlap test only means anything once the
    # lower block is known, since an inverted-but-separated pair
    # trivially has max(z[left]) > min(z[right]).
    lo, hi = ((left, right)
              if float(np.mean(z[left])) <= float(np.mean(z[right]))
              else (right, left))
    if max(float(z[i]) for i in lo) >= min(float(z[i]) for i in hi):
        return ORDER_INTERLEAVED
    return ORDER_CANONICAL if lo is left else ORDER_INVERTED


def inverted_note(struct: Structure) -> str:
    """The one sentence every surface says about an inverted pair —
    the sort's note, the engine preflight's warning, the tab's offer.

    It states the measurement and the CONSEQUENCE, and stops: which
    end carries +V/2 is the author's call about their own experiment
    (user ruling, 2026-08-29 — check z, warn, never refuse).
    """
    regions = struct.regions or {}
    z = np.asarray(struct.positions, dtype=float)[:, TRANSPORT_AXIS]
    zl = float(np.mean(z[list(regions.get(REGION_LEFT_ELECTRODE, ()))]))
    zr = float(np.mean(z[list(regions.get(REGION_RIGHT_ELECTRODE, ()))]))
    return (
        f"{REGION_LEFT_ELECTRODE} is the HIGH-z block (z centroids "
        f"{zl:.3f} vs {zr:.3f} A), the reverse of the usual convention "
        f"({REGION_LEFT_ELECTRODE} low, {REGION_RIGHT_ELECTRODE} high "
        f"-- TranSIESTA's own: 'left' is the first atoms, the -A3 end).  "
        f"This runs as labeled: the atom order and the semi-infinite "
        f"directions follow the GEOMETRY, and mu = +V/2 follows the "
        f"NAME, so the HIGH-z lead is the positively biased one.  Swap "
        f"the two labels if you meant the low-z lead to carry +V/2.")


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
    #: what the sort observed but did not refuse — today the one
    #: convention note.  Carried, never raised: the caller decides
    #: where a warning belongs on its surface.
    notes: Tuple[str, ...] = ()

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

    * electrode blocks: the LOWER one first (geometry, not the label —
      it is the block that extends to ``-A3``), then layer-major along
      the transport axis and the transverse coordinates —
      deterministic, and the extracted electrode cell inherits exactly
      this order, which is what makes the device-block ↔
      electrode-calculation correspondence hold by construction;
    * bridge: the ORIGINAL relative order (stable) — no physics reads
      it, and stability keeps the user's mental map of their molecule;
    * buffer: outermost, each atom at the end it lies OUTSIDE of —
      below the lower electrode block or above the upper one.  That is
      the same rule that decides whether it is a legal buffer atom at
      all, asked once, so the placement and the check cannot disagree.
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

    # THE ONLY REFUSAL LEFT HERE.  Two distinguishable ends is not a
    # convention, it is what a 2-terminal junction IS: with the two
    # blocks' z-ranges overlapping there is no lower and no upper, so
    # there is no order to build and no relabel that would help.
    shape = electrode_orientation(struct)
    if shape == ORDER_INTERLEAVED:
        raise SortError(
            f"the {REGION_LEFT_ELECTRODE} and {REGION_RIGHT_ELECTRODE} "
            f"blocks INTERLEAVE along the transport axis (z ranges "
            f"{min(float(z[i]) for i in left):.3f}.."
            f"{max(float(z[i]) for i in left):.3f} and "
            f"{min(float(z[i]) for i in right):.3f}.."
            f"{max(float(z[i]) for i in right):.3f} A) -- a 2-terminal "
            f"junction needs two distinguishable ends.  Fix the labels "
            f"so each electrode is one block of layers.")

    # THE ATOM ORDER FOLLOWS GEOMETRY, NEVER THE LABEL (user ruling,
    # 2026-08-29: check z, warn, let the author decide).  The lower
    # block goes first because it is the one that extends to -A3;
    # putting the L-LABELED block first when it sits on top would aim
    # that lead down INTO the bridge, which is not a naming preference
    # but a broken self-energy.  The label keeps its own job: it
    # carries the chemical potential (transiesta.py), so an inverted
    # junction runs exactly as labeled, with the high-z lead at +V/2.
    # Taken from the ONE predicate's answer, not re-derived: a second
    # z comparison here would be a second place that decides which
    # block is lower, free to drift from the one every other surface
    # asks.
    lo, hi = (left, right) if shape == ORDER_CANONICAL else (right, left)
    notes: List[str] = []
    if shape == ORDER_INVERTED:
        notes.append(inverted_note(struct))

    def _layer_major(idx: List[int]) -> List[int]:
        return sorted(idx, key=lambda i: (z[i], pos[i, 0], pos[i, 1]))

    # WHICH END A BUFFER ATOM BELONGS TO, and whether it is a legal
    # buffer atom at all, are ONE question asked once.  Buffer means
    # OUTSIDE (§ 3, buffer sanity): padding beyond the electrode
    # blocks, excluded from the NEGF region -- so "below the lower
    # block" IS the low end, "above the upper block" IS the high end,
    # and anything else is not outside anything.
    #
    # It used to be two rules: the side came from the midpoint of the
    # WHOLE structure and the legality was then checked against the
    # electrodes.  They agree on ordinary junctions and part company on
    # lopsided ones -- padding at one end taller than everything below
    # it puts the midpoint above the upper electrode, files real
    # top-buffer atoms at the bottom, and refuses a correctly labeled
    # structure for "buffer inside the electrode".  One rule cannot
    # disagree with itself.
    z_lo_min = min(float(z[j]) for j in lo)
    z_hi_max = max(float(z[j]) for j in hi)
    buf_lo, buf_hi, misplaced = [], [], []
    for i in by[REGION_BUFFER]:
        if float(z[i]) < z_lo_min - 1e-6:
            buf_lo.append(i)
        elif float(z[i]) > z_hi_max + 1e-6:
            buf_hi.append(i)
        else:
            misplaced.append(i)
    buf_lo, buf_hi = _layer_major(buf_lo), _layer_major(buf_hi)
    if misplaced:
        shown = "; ".join(
            f"{_atom_word(struct, i)} at z={z[i]:.3f} A"
            for i in misplaced[:6])
        more = f" and {len(misplaced) - 6} more" if len(misplaced) > 6 else ""
        raise SortError(
            f"{len(misplaced)} buffer atom(s) sit AT or INSIDE the "
            f"electrode blocks along the transport axis: {shown}{more}.  "
            f"Buffer means padding OUTSIDE the electrodes -- below "
            f"z={z_lo_min:.3f} or above z={z_hi_max:.3f} A -- and is "
            f"excluded from the NEGF region (transport-design.md 3).  "
            f"Relabel the atoms, or move them beyond the electrode "
            f"blocks.")

    order = (buf_lo + _layer_major(lo) + by[REGION_BRIDGE]
             + _layer_major(hi) + buf_hi)

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
        notes=tuple(notes),
    )
