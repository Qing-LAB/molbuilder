"""The atom-permutation record -- a sorted deck's order, and the way back to the person's.

MODULE  atom_permutation (L1, the job as described; stdlib + numpy only)
ROLE    the order record -- the class every reader crosses with, its one dict
        reader, the sort keys' names, ``same_order``; and
        ``atom-permutation.json``'s schema and reader, until the file retires
        (`model/overview.md` § 2.2a, plan § 5ab R5)
USED-BY structure.py (the type of ``Structure.atom_order``), transport/sort.py
        (the keys, and the record it sorts by), transport/compose.py (the
        composite's record), spectra/vibrational_analysis.py and
        spectra/siesta_vibration.py (read it back, beside the job)
TRAVELS in ``mb_vibration.pyz`` beside a SIESTA force-constant job
        (`runwrap.VIBRATION_COMPANIONS`), and in ``mb_pyscf.pyz``
        (`runwrap.PYSCF_COMPANIONS`) as the type ``structure.py`` holds

A deck written from a sorted copy -- held atoms first for a SIESTA
force-constant run (`engines/vibration.md` § 5.2), the electrode blocks for
TranSIESTA (`transport/sort.py`) -- speaks the sorted order in every file the
engine writes.  This record is the one way back (`model/overview.md` § 2.2:
recorded once and inverted once, through one pair): a reader asks
:meth:`Permutation.rows_to_input_order` and :meth:`Permutation.original_of`,
and never inverts by hand.

ITS OWN MODULE, because the record travels: the SIESTA vibration's finish
runs beside a job, where the sort's imports do not exist, and reads the
record.  The sort WRITES the record (`sort.write_permutation`), from the
result it produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

#: The record's schema name (registered: `job-contracts.md` § 6.1).
PERMUTATION_SCHEMA = "molbuilder/atom-permutation@1"

#: THE SORT KEYS, by name -- what a calculation needs done to the atom order:
#: TranSIESTA's electrode blocks, or a SIESTA force-constant run's held atoms
#: first.  Named HERE, where the order record lives, so a reader beside a job
#: can check a record's key; ``transport.sort`` maps each name to its order.
SORT_TRANSPORT = "transport"
SORT_HELD_FIRST = "held-first"
SORT_KEY_NAMES = (SORT_TRANSPORT, SORT_HELD_FIRST)

#: WHAT EACH ORDER IS CALLED where a person reads it -- the viewer's mark and
#: every door that says it restored a structure (`model/overview.md`
#: § 2.2a; `web/results.md` § 4): one table, beside the names.
SORT_KEY_WORDS = {SORT_TRANSPORT: "TranSIESTA's order",
                  SORT_HELD_FIRST: "held atoms first"}


class PermutationError(ValueError):
    """A record that cannot put a run's rows back in the input order: absent,
    of another schema, or whose two directions are not inverse bijections.
    The message names the file, ready to surface verbatim."""


def _two_lists(o2s, s2o, source: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """THE ONE CHECK of an order record's two lists, for both its readers
    (:meth:`Permutation.from_dict`, :func:`read_permutation`): each a list of
    whole atom indices, each naming every one of its atoms once, each undoing
    the other.  Refused naming ``source``."""
    def indices(v):
        if not isinstance(v, (list, tuple)) or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in v):
            raise PermutationError(
                f"{source}: an order record states its key and both "
                f"directions as lists of whole atom indices")
        return tuple(v)
    o2s, s2o = indices(o2s), indices(s2o)
    n, every = len(s2o), list(range(len(s2o)))
    if sorted(s2o) != every or sorted(o2s) != every or any(
            s2o[o2s[i]] != i for i in range(n)):
        raise PermutationError(
            f"{source}: the order record's two lists must each name every "
            f"one of the {n} atoms once, and each undo the other")
    return o2s, s2o


@dataclass(frozen=True)
class Permutation:
    """A recorded permutation, read back -- the return leg of § 2.2.

    ``sorted_to_original[j]`` is the input index of the atom at sorted
    position ``j``; ``original_to_sorted[i]`` where input atom ``i`` went.
    Everything a reader of a sorted run needs goes through these two
    methods, so no reader inverts by hand.
    """
    original_to_sorted: Tuple[int, ...]
    sorted_to_original: Tuple[int, ...]
    #: the key the record names (``""`` when it names none)
    key: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.sorted_to_original)

    @property
    def words(self) -> str:
        """What this order is called where a person reads it
        (:data:`SORT_KEY_WORDS`)."""
        return SORT_KEY_WORDS.get(self.key, self.key)

    @property
    def said(self) -> str:
        """The viewer's mark of a structure in this order: *in the engine's
        order (held atoms first)*."""
        return f"in the engine's order ({self.words})"

    @property
    def is_identity(self) -> bool:
        """Nothing moved -- an order that is never recorded
        (`model/overview.md` § 2.2a, I2)."""
        return list(self.sorted_to_original) == list(range(self.n_atoms))

    def original_of(self, sorted_indices) -> List[int]:
        """The input indices of atoms named by sorted position, in the
        order given."""
        return [int(self.sorted_to_original[int(j)]) for j in sorted_indices]

    def sorted_of(self, original_indices) -> List[int]:
        """The sorted positions of atoms named by their input index, in the
        order given -- the way a selection made in the person's order is
        asked of data in the engine's."""
        return [int(self.original_to_sorted[int(i)]) for i in original_indices]

    def to_dict(self) -> dict:
        """The order record as the structure's metadata states it -- the key
        and both directions (`model/structure-molstruct.md` § 1)."""
        return {"key": self.key,
                "sorted_to_original": list(self.sorted_to_original),
                "original_to_sorted": list(self.original_to_sorted)}

    @classmethod
    def from_dict(cls, d, *, source: str = "atom_order",
                  n_atoms: Optional[int] = None) -> "Permutation":
        """THE ONE READER of an order record's dict -- the structure's
        ``atom_order`` field, a sidecar's, a deck's block's.  Refuses, naming
        ``source``: a value that is not an object, a key that is not one of
        :data:`SORT_KEY_NAMES`, two lists that do not each name every atom
        once and undo each other (over ``n_atoms`` atoms when given), and the
        identity, which is never recorded (`model/overview.md` § 2.2a, I2)."""
        if isinstance(d, cls):
            perm = d
        else:
            if not isinstance(d, dict):
                raise PermutationError(
                    f"{source}: an order record is an object {{key, "
                    f"sorted_to_original, original_to_sorted}}; got "
                    f"{type(d).__name__}")
            stray = sorted(k for k in d
                           if k not in ("key", "sorted_to_original",
                                        "original_to_sorted"))
            if stray:
                raise PermutationError(
                    f"{source}: an order record holds key, "
                    f"sorted_to_original and original_to_sorted; it also "
                    f"carries {stray!r}")
            o2s, s2o = _two_lists(d.get("original_to_sorted"),
                                  d.get("sorted_to_original"), source)
            perm = cls(original_to_sorted=o2s, sorted_to_original=s2o,
                       key=d.get("key"))
        if perm.key not in SORT_KEY_NAMES:
            raise PermutationError(
                f"{source}: the order record's key {perm.key!r} is not one of "
                f"{', '.join(SORT_KEY_NAMES)}")
        n = perm.n_atoms
        _two_lists(perm.original_to_sorted, perm.sorted_to_original, source)
        if n_atoms is not None and n != n_atoms:
            raise PermutationError(
                f"{source}: the order record is over {n} atoms; the structure "
                f"holds {n_atoms}")
        if perm.is_identity:
            raise PermutationError(
                f"{source}: the order record states the identity order, which "
                f"is never recorded -- a structure in its own order carries "
                f"none (model/overview.md § 2.2a)")
        return perm

    def rows_to_input_order(self, rows, sorted_indices):
        """Per-atom rows that stand in ``sorted_indices`` order (a subset
        of the sorted copy), reordered so they follow the INPUT order of
        those same atoms.  Returns ``(rows_in_input_order,
        input_indices_ascending)``."""
        orig = self.original_of(sorted_indices)
        order = np.argsort(orig)
        arr = np.asarray(rows)
        return arr[..., order, :] if arr.ndim >= 2 else arr[order], \
            [orig[k] for k in order]


def same_order(a: "Optional[Permutation]", b: "Optional[Permutation]") -> bool:
    """Whether two decks' atoms stand in the same order -- what decides that a
    restart file one wrote may be read by the other (`model/overview.md`
    § 2.2a, I6).  ``None`` is the person's own order, which the identity is
    too: an order is compared by where each atom sits, not by its key."""
    def where(p):
        return None if p is None or p.is_identity else tuple(
            p.sorted_to_original)
    return where(a) == where(b)


def read_permutation(directory) -> Permutation:
    """The permutation recorded in ``directory``, or a refusal naming the
    file: a sorted run with no record is a run whose numbers cannot be
    returned."""
    # THE CATALOGUE'S NAME (`runfiles.WRITTEN`), imported both ways and only
    # here: the module travels in two bundles, one without `runfiles`.
    try:                                    # inside molbuilder
        from .runfiles import PERMUTATION_FILE
    except ImportError:                     # beside a job
        from runfiles import PERMUTATION_FILE  # type: ignore[no-redef]
    p = Path(directory) / PERMUTATION_FILE
    if not p.is_file():
        raise PermutationError(
            f"no {PERMUTATION_FILE} beside {Path(directory)}: the run was "
            f"written from a sorted copy and its per-atom results cannot "
            f"be put back in the input order without the record")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") != PERMUTATION_SCHEMA:
        raise PermutationError(f"{p}: schema {d.get('schema')!r} is not "
                               f"{PERMUTATION_SCHEMA!r}")
    o2s, s2o = _two_lists(d.get("original_to_sorted"),
                          d.get("sorted_to_original"), str(p))
    return Permutation(original_to_sorted=o2s, sorted_to_original=s2o,
                       key=str(d.get("key", "") or ""))


__all__ = ["PERMUTATION_SCHEMA", "Permutation", "PermutationError",
           "SORT_HELD_FIRST", "SORT_KEY_NAMES", "SORT_KEY_WORDS",
           "SORT_TRANSPORT",
           "read_permutation", "same_order"]
