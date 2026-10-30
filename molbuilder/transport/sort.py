"""The categorical sort — `engines/transport.md` § 4.

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

Pure by design: ``sort_by(structure, key) -> SortResult`` -- the sorted
copy, carrying its order record (``atom_order``), and the sort's notes -- no
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
from typing import List, Optional, Tuple

import numpy as np

from ..atom_permutation import Permutation, SORT_HELD_FIRST, SORT_TRANSPORT
from ..structure import REMAPPED_NOT_EDITED, Structure

# --------------------------------------------------------------------- #
#  The region-label vocabulary -- `model/structure-annotations.md` § 5   #
# --------------------------------------------------------------------- #
#
# The labels a transport junction's atoms carry in the `.molstruct.json`
# sidecar's `regions`.  Here because this module owns the partition they
# make; everything that reads a region label -- the emitter, compose,
# validation -- asks here.  The two electrode labels are the kind table's:
# each is the region a lead is cut from, named on the lead's role
# (`execution/architecture.md` § 3.3), so a rename happens there, once.

from ..kind_table import kind as _kind  # noqa: E402

_LEADS = _kind("transport").leads
REGION_LEFT_ELECTRODE  = _LEADS["electrode_L"]
REGION_RIGHT_ELECTRODE = _LEADS["electrode_R"]
REGION_BRIDGE          = "bridge"
#: Atoms excluded from the NEGF region entirely (``TS.Atoms.Buffer``):
#: padding at the OUTER ends of the device, beyond the electrode blocks.
#: Optional -- most 2-terminal junctions need none.
REGION_BUFFER          = "buffer"

#: The two leads, by EXACT name -- the list every reader of "which regions
#: are leads" asks (`structure-annotations.md` § 5; user, 2026-10-02:
#: "these are just two matching names").  A transport run is 2-terminal,
#: so any other label, ``tip-electrode`` included, is not a lead.
ELECTRODE_LABELS = (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE)

#: The four PARTITION labels — exactly one per atom.  Everything else
#: (``interface``, ``frozen_atoms``, user labels) is partition-neutral
#: bookkeeping that rides along untouched.
PARTITION_LABELS = (*ELECTRODE_LABELS, REGION_BRIDGE, REGION_BUFFER)

#: The transport axis is the THIRD cell axis (z) -- the one home is
#: `kmesh.TRANSPORT_AXIS` (a copy stood here until 2026-10-18, K3).
from ..kmesh import TRANSPORT_AXIS  # noqa: E402

#: How the two electrode blocks sit along transport, as one word.
#: ONE door for the fact, because three places need it and they must not
#: disagree: the sort's note, the web describe (which offers the swap when
#: the answer is ``inverted``), and the swap itself.
ORDER_CANONICAL = "canonical"        # L below R -- the usual convention
ORDER_INVERTED = "inverted"          # L above R -- runs; a swap flips it
ORDER_INTERLEAVED = "interleaved"    # no two ends -- nothing to build


def electrode_orientation(struct: Structure) -> Optional[str]:
    """Which of the three shapes this structure's electrodes are in, or
    ``None`` when it has no two electrode blocks to compare.

    The usual convention (engines/transport.md § 4): ``L-electrode``
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
    the sort's note, the tab's offer.

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
    """A sorted copy and what the sort said.  The copy carries its own order
    record (``structure.atom_order``, `model/overview.md` § 2.2a) -- none
    when the sort moved nothing."""
    structure: Structure
    #: what the sort observed but did not refuse — today the one
    #: convention note.  Carried, never raised: the caller decides
    #: where a warning belongs on its surface.
    notes: Tuple[str, ...] = ()


def _partition_of(struct: Structure) -> List[str]:
    """One partition label per atom, or the two named refusals."""
    n = len(struct.elements)
    membership: List[List[str]] = [[] for _ in range(n)]
    for label in PARTITION_LABELS:
        for i in struct.regions.get(label, ()):
            membership[i].append(label)

    unlabeled = [i for i in range(n) if not membership[i]]
    if unlabeled:
        shown = struct.atoms_named(unlabeled, limit=8)
        raise SortError(
            f"{len(unlabeled)} atom(s) carry no partition label: {shown}.  "
            f"Every atom must be exactly one of "
            f"{', '.join(PARTITION_LABELS)} (`interface` and other "
            f"labels ride on top and do not count) -- an unlabeled atom "
            f"has no place in TranSIESTA's atom order and would be "
            f"misassigned silently.")

    doubled = [i for i in range(n) if len(membership[i]) > 1]
    if doubled:
        shown = "; ".join(
            f"{struct.atoms_named([i])}: {' + '.join(membership[i])}"
            for i in doubled[:8])
        more = f" and {len(doubled) - 8} more" if len(doubled) > 8 else ""
        raise SortError(
            f"{len(doubled)} atom(s) carry more than one partition "
            f"label: {shown}{more}.  The partition must be disjoint -- "
            f"TranSIESTA puts each atom in exactly one block.")

    return [membership[i][0] for i in range(n)]


def _transport_order(struct: Structure) -> Tuple[List[int], Tuple[str, ...]]:
    """The first key's order -- the canonical transport order, from labels,
    § 4.1a -- and what it says: ``(order, notes)``, for :func:`sort_by`.

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
            f"{struct.atoms_named([i])} at z={z[i]:.3f} A"
            for i in misplaced[:6])
        more = f" and {len(misplaced) - 6} more" if len(misplaced) > 6 else ""
        raise SortError(
            f"{len(misplaced)} buffer atom(s) sit AT or INSIDE the "
            f"electrode blocks along the transport axis: {shown}{more}.  "
            f"Buffer means padding OUTSIDE the electrodes -- below "
            f"z={z_lo_min:.3f} or above z={z_hi_max:.3f} A -- and is "
            f"excluded from the NEGF region (engines/transport.md 4).  "
            f"Relabel the atoms, or move them beyond the electrode "
            f"blocks.")

    order = (buf_lo + _layer_major(lo) + by[REGION_BRIDGE]
             + _layer_major(hi) + buf_hi)
    return order, tuple(notes)


#: THE KEYS, by name -- what a calculation needs done to the atom order.
#: A kind asks ``sort_by(struct, deck_sort_key(...))``, the key its decks
#: are written by (:data:`SORTED_COPY`); the name travels into the order
#: record so the return leg knows which order it is undoing.  The names are
#: the order record's (``atom_permutation.SORT_KEY_NAMES``); each maps to the
#: function that works its order out, and :func:`sort_by` hands that order to
#: the one machinery, ``apply_order``, with the key -- a new need is a new
#: name there and an entry here, never a second machine.
SORT_KEYS = {
    SORT_TRANSPORT:  lambda s: _transport_order(s),
    SORT_HELD_FIRST: lambda s: _held_first_order(s),
}


#: WHICH DECKS ARE WRITTEN FROM A SORTED COPY, AND BY WHICH KEY, keyed by
#: ``(engine, kind)`` (`model/overview.md` § 2.2a): a SIESTA vibration's
#: every stage, held atoms first (`engines/vibration.md` § 5.2), and a
#: transport calculation's junction rungs, TranSIESTA's electrode blocks
#: (`engines/transport.md` § 4).  Every other calculation's decks are in
#: the user order.  THE ONE TABLE (:func:`deck_sort_key`): prep and the composer
#: sort by it, and the run door reads from it which runs' decks may be in
#: an engine's order (I9).
SORTED_COPY = {("siesta", "vibration"): SORT_HELD_FIRST,
               ("siesta", "transport"): SORT_TRANSPORT}


#: What a deck whose atoms are a structure of its own is, said where its
#: map has no user index (`model/overview.md` § 2.2b, F2).
OWN_STRUCTURE_WORDS = "a structure of its own (the electrode cell)"


def deck_own_structure(engine: str, kind: str,
                       stage: Optional[str] = None) -> Optional[str]:
    """What a deck of ``stage`` of an ``(engine, kind)`` calculation is when
    its atoms are A STRUCTURE OF ITS OWN -- not the user's, reordered or
    not: a transport lead rung's electrode cell (a lead role,
    the kind's own ``leads``; `model/overview.md` § 2.2a, *a structure
    derived from the user's*) -- in words; ``None`` for every other deck."""
    if stage is not None:
        from .deck import rung_of
        if rung_of(stage) in _kind(kind).leads:
            return OWN_STRUCTURE_WORDS
    return None


def deck_sort_key(engine: str, kind: str,
                  stage: Optional[str] = None) -> Optional[str]:
    """The key a deck of ``stage`` (its name or its token) of an
    ``(engine, kind)`` calculation is written from a sorted copy by
    (:data:`SORTED_COPY`) -- or ``None``, a deck in the user order: every
    calculation the table does not name, and a deck of a structure of its
    own (:func:`deck_own_structure`)."""
    key = SORTED_COPY.get((str(engine), str(kind)))
    if key is not None and deck_own_structure(engine, kind, stage):
        return None
    return key


def sort_by(struct: Structure, key: str) -> SortResult:
    """THE ONE WAY INTO AN ENGINE'S ORDER (`model/overview.md` § 2.2a): the
    named key's order, the shared machinery, the copy carrying its order
    record.  A structure already remapped is refused -- an engine order is
    made only from the user order, so two are never composed (I4)."""
    try:
        fn = SORT_KEYS[key]
    except KeyError:
        raise SortError(f"no sort key named {key!r}; the keys are "
                        f"{', '.join(sorted(SORT_KEYS))}") from None
    if struct.atom_order is not None:
        raise SortError(REMAPPED_NOT_EDITED)
    order, notes = fn(struct)
    return apply_order(struct, order, notes, key=key)


def _held_first_order(struct: Structure
                      ) -> Tuple[List[int], Tuple[str, ...]]:
    """The second key's order: held atoms first, free atoms last, each in the
    user order among themselves -- `model/overview.md` § 2.2 -- and nothing to
    say: ``(order, ())``, for :func:`sort_by`.

    SIESTA's force-constant run nudges atoms *A through B* (`FC.First` /
    `FC.Last`) and cannot take a scattered free set; molbuilder's held set
    is whatever a person ticked.  So the copy the deck renders from puts
    every held atom before every free one, and the free range is the
    tail: ``FC.First = n_held + 1``, ``FC.Last = n``.  Nothing held means
    the identity order, which the structure carries no record of
    (`model/overview.md` § 2.2a, I2).
    """
    n = len(struct.elements)
    held = sorted(int(i) for i in (struct.frozen_atoms or []))
    held_set = set(held)
    free = [i for i in range(n) if i not in held_set]
    return held + free, ()


def apply_order(struct: Structure, order: List[int],
                notes: Tuple[str, ...] = (), *, key: str) -> SortResult:
    """The machinery every sort key shares: check the order is a bijection,
    move every per-atom field with its atom, record both directions.

    ``order[j]`` is the ORIGINAL index of the atom that sits at ``j`` in
    the copy.  A key computes an order; this is what turns it into a
    structure and a record, so two keys cannot remap differently.
    """
    # The bijection check (user: "make sure nothing is missed") -- done
    # mechanically even though a key's construction cannot fail it:
    # this is the guard that outlives refactors.
    n = len(struct.elements)
    if sorted(order) != list(range(n)):
        raise SortError(
            f"internal error: the sort is not a bijection over {n} "
            f"atoms -- nothing was written.")

    old_to_new = {old: new for new, old in enumerate(order)}
    # THE STRUCTURE'S OWN REORDER (`model/structure.md` § 2.2e): every
    # per-atom field moves with its atom, and so does every frame of a frame
    # set; every other field rides whole.  The copy then carries its order
    # record (``Permutation.recorded``: none when nothing moved).
    perm = Permutation(original_to_sorted=tuple(old_to_new[i] for i in range(n)),
                       sorted_to_original=tuple(int(i) for i in order),
                       key=key)
    sorted_struct = struct.take(order).replace(atom_order=perm.recorded)
    return SortResult(structure=sorted_struct, notes=tuple(notes))
