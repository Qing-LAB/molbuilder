"""The atom-permutation record -- a sorted deck's order, and the way back to the person's.

MODULE  atom_permutation (L1, the job as described; stdlib + numpy only)
ROLE    ``atom-permutation.json``: its file name, its schema, the class every
        reader inverts it with, and its one reader
USED-BY transport/sort.py (writes the record it sorted by), transport/compose.py
        (the composite's record), spectra/vibrational_analysis.py and
        spectra/siesta_vibration.py (read it back, beside the job)
TRAVELS in ``mb_vibration.pyz`` beside a SIESTA force-constant job
        (`runwrap.VIBRATION_COMPANIONS`)

A deck written from a sorted copy -- held atoms first for a SIESTA
force-constant run (`engines/vibration.md` § 5.2), the electrode blocks for
TranSIESTA (`transport/sort.py`) -- speaks the sorted order in every file the
engine writes.  This record is the one way back (`model/overview.md` § 2.2:
recorded once and inverted once, through one pair): a reader asks
:meth:`Permutation.rows_to_input_order` and :meth:`Permutation.original_of`,
and never inverts by hand.

ITS OWN MODULE, because the record travels.  It sat in `transport/sort.py`
until 2026-09-28, whose imports -- the structure, the transport config -- do
not exist beside a job; the SIESTA vibration's finish runs there and reads the
record, and a reader that cannot import the record's class would have to
invert by hand, which is what this module exists to end.  The sort still
WRITES the record (`sort.write_permutation`), from the result it produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

#: The record's file name beside a calculation -- and, as part of the
#: calculation's shared package, in each attempt that runs from its sorted
#: copy (`execution/job-contracts.md` § 6.1).
PERMUTATION_FILE = "atom-permutation.json"

#: The record's schema name (registered: `job-contracts.md` § 6.1).
PERMUTATION_SCHEMA = "molbuilder/atom-permutation@1"


class PermutationError(ValueError):
    """A record that cannot put a run's rows back in the input order: absent,
    of another schema, or whose two directions are not inverse bijections.
    The message names the file, ready to surface verbatim."""


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
    #: the key the record names (``""`` on a record written before keys
    #: were recorded)
    key: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.sorted_to_original)

    def original_of(self, sorted_indices) -> List[int]:
        """The input indices of atoms named by sorted position, in the
        order given."""
        return [int(self.sorted_to_original[int(j)]) for j in sorted_indices]

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


def read_permutation(directory) -> Permutation:
    """The permutation recorded in ``directory``, or a refusal naming the
    file: a sorted run with no record is a run whose numbers cannot be
    returned."""
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
    o2s = tuple(int(i) for i in d["original_to_sorted"])
    s2o = tuple(int(i) for i in d["sorted_to_original"])
    n = len(s2o)
    if sorted(s2o) != list(range(n)) or len(o2s) != n or any(
            s2o[o2s[i]] != i for i in range(n)):
        raise PermutationError(f"{p}: the two directions are not inverse "
                               f"bijections over {n} atoms")
    return Permutation(original_to_sorted=o2s, sorted_to_original=s2o,
                       key=str(d.get("key", "") or ""))


__all__ = ["PERMUTATION_FILE", "PERMUTATION_SCHEMA", "Permutation",
           "PermutationError", "read_permutation"]
