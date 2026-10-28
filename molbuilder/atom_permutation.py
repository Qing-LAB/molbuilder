"""The order record -- a sorted deck's order, and the way back to the user order.

MODULE  atom_permutation (L1, the job as described; stdlib + numpy only)
ROLE    the order record -- the class every reader crosses with, its one dict
        reader, the sort keys' names (`model/overview.md` § 2.2a)
USED-BY structure.py (the type of ``Structure.atom_order``), transport/sort.py
        (the keys, and the record it sorts by), deck_record.py (a deck's
        record, read by ``order_record_of``), jobset/transport_rung.py (its
        refusal), spectra/vibrational_analysis.py and
        spectra/siesta_vibration.py (read it back, beside the job)
TRAVELS in ``mb_vibration.pyz`` beside each call of a SIESTA vibration
        (`runwrap.VIBRATION_COMPANIONS`), and in ``mb_pyscf.pyz``
        (`runwrap.PYSCF_COMPANIONS`) as the type ``structure.py`` holds

A deck written from a sorted copy -- held atoms first for a SIESTA
force-constant run (`engines/vibration.md` § 5.2), the electrode blocks for
TranSIESTA (`transport/sort.py`) -- speaks the sorted order in every file the
engine writes.  This record is the one way back (`model/overview.md` § 2.2:
recorded once and inverted once, through one pair): a reader asks
:meth:`Permutation.map_rows_to_user_order` and :meth:`Permutation.map_to_user_index`,
and never inverts by hand.

ITS OWN MODULE, because the record travels: the SIESTA vibration's finish
runs beside a job, where the sort's imports do not exist, and reads the
record from its deck.  The sort MAKES the record (`sort.apply_order`) and the
sorted copy carries it; the copy's writers state it -- the deck's block, the
geometry's sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

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
    """A record that cannot put a run's rows back in the user order: not an
    object, a key that is not a sort's, or two directions that are not inverse
    bijections.  The message names its source, ready to surface verbatim."""


def _two_lists(o2s, s2o, source: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """THE ONE CHECK of an order record's two lists, for its reader
    (:meth:`Permutation.from_dict`): each a list of whole atom indices, each naming every one of its atoms once, each undoing
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
            f"{source}: the order record's two lists must be inverse "
            f"bijections over its {n} atoms -- each naming every atom once, "
            f"each undoing the other")
    return o2s, s2o


@dataclass(frozen=True)
class Permutation:
    """A recorded permutation, read back -- the return leg of § 2.2.

    ``sorted_to_original[j]`` is the user index of the atom at engine index
    ``j`` (its place in the sorted copy); ``original_to_sorted[i]`` the
    engine index of user atom ``i``.
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
    def recorded(self) -> "Optional[Permutation]":
        """What a structure in this order carries as its ``atom_order``:
        this record, or ``None`` for the identity, which is never recorded
        (`model/overview.md` § 2.2a, I2)."""
        return None if self.is_identity else self

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

    def map_to_user_index(self, sorted_indices) -> List[int]:
        """The user indices of atoms named by their engine index (their
        place in the sorted copy), in the order given; an index that is not
        one of the atoms refused (:func:`checked_indices`)."""
        return [int(self.sorted_to_original[j]) for j in
                checked_indices(sorted_indices, self.n_atoms)]

    def map_to_engine_index(self, original_indices) -> List[int]:
        """The engine indices of atoms named by their user index, in the
        order given -- the way a selection made in the user order is
        asked of data in the engine's; an index that is not one of the
        atoms refused (:func:`checked_indices`)."""
        return [int(self.original_to_sorted[i]) for i in
                checked_indices(original_indices, self.n_atoms)]

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

    def map_rows_to_user_order(self, rows, sorted_indices):
        """Per-atom rows that stand in ``sorted_indices`` order (a subset
        of the sorted copy), reordered so they follow the user order of
        those same atoms.  Returns ``(rows_in_the_user_order,
        user_indices_ascending)``."""
        orig = self.map_to_user_index(sorted_indices)
        order = np.argsort(orig)
        arr = np.asarray(rows)
        return arr[..., order, :] if arr.ndim >= 2 else arr[order], \
            [orig[k] for k in order]


# ----------------------------------------------------------------------- #
#  The atom index map -- what a file says of its atoms' indices            #
#  (`model/overview.md` § 2.2b)                                            #
# ----------------------------------------------------------------------- #

#: WHICH INDEX A FILE'S ATOMS ARE LISTED BY -- the values of its
#: ``atoms_listed_by``: row i of its atom axis is the atom of that index i.
LISTED_BY_USER = "user_index"
LISTED_BY_ENGINE = "engine_index"
LISTED_NOT_STATED = "not_stated"
LISTED_BY = (LISTED_BY_USER, LISTED_BY_ENGINE, LISTED_NOT_STATED)

#: Every atom index a file molbuilder writes counts from this (§ 2, F5).
ATOM_INDEX_BASE = 0

#: THE FIELDS, by name -- one table every writer states them through and the
#: reader reads (F6): which index the atoms are listed by, where counting
#: starts, and per atom of the file its user and its engine index.
FIELD_LISTED_BY = "atoms_listed_by"
FIELD_BASE = "atom_index_base"
FIELD_USER = "user_atom_index"
FIELD_ENGINE = "engine_atom_index"
#: Why a column is not written (F1) -- where one is missing, the sentence
#: the reader says.
FIELD_WHY = "atom_index_why"
#: The key a JSON record carries the map under.
MAP_KEY = "atom_index_map"


def map_rows_to_user_order(record: "Optional[Permutation]", rows,
                       engine_indices, n_atoms: int):
    """Per-atom rows standing in ``engine_indices`` order, put in the user
    order -- through ``record``'s own method (:meth:`Permutation.
    map_rows_to_user_order`), or, for a run that recorded none (it ran the
    user order, I2), by those same indices: never an identity record built
    by hand (`model/overview.md` § 2.2a, *Forbidden*).  Returns ``(rows,
    user_indices_ascending)``."""
    if record is not None:
        return record.map_rows_to_user_order(rows, engine_indices)
    idx = checked_indices(engine_indices, n_atoms)
    order = np.argsort(idx)
    arr = np.asarray(rows)
    return (arr[..., order, :] if arr.ndim >= 2 else arr[order],
            [idx[k] for k in order])


def map_to_user_index(record: "Optional[Permutation]", engine_indices,
                  n_atoms: int) -> List[int]:
    """Engine indices as user indices -- ``record``'s own method, or the
    same indices for a run that recorded none (I2); each one of the
    ``n_atoms`` atoms (:func:`checked_indices`)."""
    if record is not None:
        return record.map_to_user_index(engine_indices)
    return checked_indices(engine_indices, n_atoms)


class AtomIndexError(ValueError):
    """A map that does not read, or a mapping its file cannot answer -- the
    message says which column is not known and why, ready to surface."""


def checked_indices(indices, n_atoms: int, what: str = "atom index"
                    ) -> List[int]:
    """THE ONE BOUNDS CHECK of every map door (`model/overview.md` § 2.2a,
    *Forbidden*): ``indices`` as whole numbers, each one of the ``n_atoms``
    atoms, counted from 0 -- refused by name otherwise, never wrapped (``-1``
    is no atom) and never passed through.  A :class:`AtomIndexError`, so a
    door answers it as the refusal it is."""
    out = []
    for i in indices:
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise AtomIndexError(f"{what} {i!r} is not a whole number")
        if not 0 <= int(i) < int(n_atoms):
            raise AtomIndexError(
                f"{what} {int(i)} is not one of the {int(n_atoms)} atoms "
                f"(0 to {int(n_atoms) - 1}, counted from 0)")
        out.append(int(i))
    return out


def _column(values, n: int, what: str, source: str) -> Tuple[int, ...]:
    """A column of the map: ``n`` whole atom indices naming every atom once
    (each a permutation of ``0 … n−1``), refused naming ``source``."""
    if not isinstance(values, (list, tuple)) or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in values):
        raise AtomIndexError(f"{source}: {what} is a list of whole atom "
                             f"indices")
    # ITS LENGTH FIRST: a file claiming more atoms than it lists is refused
    # without a range that size being built.
    if len(values) != n or sorted(values) != list(range(n)):
        raise AtomIndexError(f"{source}: {what} must name each of the "
                             f"file's {n} atoms once, counted from 0")
    return tuple(values)


@dataclass(frozen=True)
class AtomIndexMap:
    """WHICH INDEX A FILE'S ATOMS ARE LISTED BY, and both indices of each
    (`model/overview.md` § 2.2b): ``listed_by`` (:data:`LISTED_BY`),
    ``n_atoms``, and per atom of the file ``user`` -- its user index -- and
    ``engine`` -- its engine index -- each ``None`` where it is not known;
    ``said`` is the reader's sentence: why a column is not known, or the
    warning a file that states no order is read with (F4); ``stated_by``
    the file a reader found it in.  **Made only
    from where the atoms came from** -- :meth:`of_user_structure`,
    :meth:`of_engine_record`, :meth:`of_own_structure`, :meth:`from_dict`
    -- never put together from parts.  The column the file is listed by is
    the identity, checked."""
    listed_by: str
    n_atoms: int
    user: Optional[Tuple[int, ...]]
    engine: Optional[Tuple[int, ...]]
    said: str = ""
    #: The file that states it -- the file itself, its sidecar, its run's
    #: deck -- where a reader found it (`atom_index.atom_index_map`).
    stated_by: str = ""

    def __post_init__(self):
        if self.listed_by not in LISTED_BY:
            raise AtomIndexError(f"atoms_listed_by {self.listed_by!r} is not "
                                 f"one of {', '.join(LISTED_BY)}")
        for name, col in (("user", self.user), ("engine", self.engine)):
            if col is not None:
                _column(list(col), self.n_atoms, f"the {name} column",
                        "an atom index map")
        own = {LISTED_BY_USER: self.user, LISTED_NOT_STATED: self.user,
               LISTED_BY_ENGINE: self.engine}[self.listed_by]
        if own is not None and tuple(own) != tuple(range(len(own))):
            raise AtomIndexError(
                f"a file listed by {self.listed_by} lists its atoms 0, 1, "
                f"2 … in that column")

    # -- made from where the atoms came from ------------------------------

    @classmethod
    def of_user_structure(cls, n_atoms: int, said: str = (
            "given to no engine: its engine order is each calculation's, "
            "stated by that calculation's files")) -> "AtomIndexMap":
        """The user's structure, given to no engine (F2): listed by user
        index, the user column alone -- one structure may be given to
        several calculations, each sorting it its own way; ``said`` why
        there is no engine index."""
        return cls(LISTED_BY_USER, int(n_atoms), tuple(range(n_atoms)), None,
                   said)

    @classmethod
    def of_engine_record(cls, record: "Optional[Permutation]",
                         n_atoms: int) -> "AtomIndexMap":
        """A run's files and a calculation's composed junction (F2): listed
        by engine index, both columns from the order record -- the identity
        where it is none (the engine ran the user order, or the sort moved
        nothing)."""
        n = int(n_atoms)
        if record is not None and record.n_atoms != n:
            raise AtomIndexError(f"the order record is over {record.n_atoms} "
                                 f"atoms; the file holds {n}")
        user = (tuple(int(i) for i in record.sorted_to_original)
                if record is not None else tuple(range(n)))
        return cls(LISTED_BY_ENGINE, n, user, tuple(range(n)))

    @classmethod
    def of_own_structure(cls, n_atoms: int, said: str) -> "AtomIndexMap":
        """A run of a structure of its own -- an electrode cell, not a
        reordering of the user's atoms -- or a file no run of ours holds:
        listed by engine index, the engine column alone, ``said`` saying
        why there is no user index."""
        return cls(LISTED_BY_ENGINE, int(n_atoms), None,
                   tuple(range(n_atoms)), said)

    @classmethod
    def assumed_user(cls, n_atoms: int, said: str) -> "AtomIndexMap":
        """A structure file that states no order, read as listed by user
        index (F4) -- ``said`` the warning it is read with."""
        return cls(LISTED_NOT_STATED, int(n_atoms), tuple(range(n_atoms)),
                   None, said)

    # -- the file's statement ------------------------------------------------

    def to_dict(self) -> dict:
        """The fields as a file states them (F1): each column where it is
        known; a map ``not_stated`` states no column -- its user column is
        the reading's assumption, never the file's; and where a column is
        not written, why (:data:`FIELD_WHY`, the map's ``said``), so the
        reader says it."""
        out = {FIELD_LISTED_BY: self.listed_by, FIELD_BASE: ATOM_INDEX_BASE}
        if self.user is not None and self.listed_by != LISTED_NOT_STATED:
            out[FIELD_USER] = list(self.user)
        if self.engine is not None:
            out[FIELD_ENGINE] = list(self.engine)
        if self.said and (FIELD_USER not in out or FIELD_ENGINE not in out):
            out[FIELD_WHY] = self.said
        return out

    @classmethod
    def from_dict(cls, d, *, n_atoms: int, source: str = MAP_KEY,
                  said: str = "") -> "AtomIndexMap":
        """THE ONE READER of the fields -- a JSON record's block, a
        sidecar's, a NetCDF group's as ``ncdata`` reads them -- checked as
        it is read: the listing one of :data:`LISTED_BY`, counting from 0,
        each column naming the file's ``n_atoms`` atoms once, the column it
        is listed by the identity.  A map ``not_stated`` reads as listed by
        user index (F4), ``said`` its warning."""
        if not isinstance(d, dict):
            raise AtomIndexError(f"{source}: the atom index map is an "
                                 f"object; got {type(d).__name__}")
        fields = (FIELD_LISTED_BY, FIELD_BASE, FIELD_USER, FIELD_ENGINE,
                  FIELD_WHY)
        stray = sorted(set(d) - set(fields))
        if stray:
            raise AtomIndexError(f"{source}: the atom index map holds "
                                 f"{', '.join(fields)}; it also carries "
                                 f"{stray!r}")
        # WHY A COLUMN IS NOT WRITTEN, as the file states it -- unless the
        # reader has a sentence of its own (F4's warning names the file).
        said = said or str(d.get(FIELD_WHY) or "")
        if d.get(FIELD_BASE) != ATOM_INDEX_BASE:
            raise AtomIndexError(f"{source}: {FIELD_BASE} is "
                                 f"{d.get(FIELD_BASE)!r}; every atom index a "
                                 f"file molbuilder writes counts from 0")
        n = int(n_atoms)
        user = (_column(d[FIELD_USER], n, FIELD_USER, source)
                if FIELD_USER in d else None)
        engine = (_column(d[FIELD_ENGINE], n, FIELD_ENGINE, source)
                  if FIELD_ENGINE in d else None)
        listed = d.get(FIELD_LISTED_BY)
        if listed == LISTED_NOT_STATED:
            return cls.assumed_user(n, said)
        try:
            return cls(listed, n, user, engine, said)
        except AtomIndexError as exc:
            raise AtomIndexError(f"{source}: {exc}") from exc

    # -- mapping, the map's -----------------------------------------------

    def _needs(self, user: bool, engine: bool) -> None:
        missing = [name for name, want, col in
                   (("user", user, self.user), ("engine", engine, self.engine))
                   if want and col is None]
        if missing:
            raise AtomIndexError(
                f"this file states no {missing[0]} index"
                + (f": {self.said}" if self.said else ""))

    def map_to_engine_index(self, user_indices) -> List[int]:
        """User indices → engine indices, in the order given -- to ask sisl
        about an engine's file; an index that is not one of the file's atoms
        refused (:func:`checked_indices`)."""
        self._needs(user=True, engine=True)
        at = {u: k for k, u in enumerate(self.user)}
        return [int(self.engine[at[u]]) for u in
                checked_indices(user_indices, self.n_atoms, "user index")]

    def map_to_user_index(self, engine_indices) -> List[int]:
        """Engine indices → user indices, in the order given -- to name
        what an engine's file says; an index that is not one of the file's
        atoms refused (:func:`checked_indices`)."""
        self._needs(user=True, engine=True)
        at = {e: k for k, e in enumerate(self.engine)}
        return [int(self.user[at[e]]) for e in
                checked_indices(engine_indices, self.n_atoms, "engine index")]

    def _relisted(self, by: str) -> Tuple[np.ndarray, "AtomIndexMap"]:
        col = self.user if by == LISTED_BY_USER else self.engine
        order = np.argsort(np.asarray(col))
        pick = (lambda c: None if c is None
                else tuple(int(c[k]) for k in order))
        return order, AtomIndexMap(by, self.n_atoms, pick(self.user),
                                   pick(self.engine), self.said,
                                   self.stated_by)

    def map_rows_to_user_order(self, rows):
        """Per-atom rows in this file's order (its first axis), reordered so
        row k is user atom k -- ``(rows, the map they are now listed by)``."""
        self._needs(user=True, engine=False)
        order, listed = self._relisted(LISTED_BY_USER)
        return np.asarray(rows)[order], listed

    def map_rows_to_engine_order(self, rows):
        """Per-atom rows in this file's order, reordered so row k is engine
        atom k -- ``(rows, the map they are now listed by)``."""
        self._needs(user=False, engine=True)
        order, listed = self._relisted(LISTED_BY_ENGINE)
        return np.asarray(rows)[order], listed

    def as_listed_by_user(self) -> "AtomIndexMap":
        """The same atoms' map, listed by user index -- a result for the
        user, written from the record its calculation ran on (F2)."""
        self._needs(user=True, engine=False)
        return self._relisted(LISTED_BY_USER)[1]

    def table(self) -> List[dict]:
        """Every atom of the file: its user index and its engine index, each
        counted from 0 -- the correspondence whole.  The number a view shows
        is the browser's (`model/overview.md` § 2.1), never made here."""
        return [{"user_index": None if self.user is None else int(self.user[k]),
                 "engine_index": (None if self.engine is None
                                  else int(self.engine[k]))}
                for k in range(self.n_atoms)]


__all__ = ["ATOM_INDEX_BASE", "AtomIndexError", "AtomIndexMap",
           "FIELD_BASE", "FIELD_ENGINE", "FIELD_LISTED_BY", "FIELD_USER",
           "FIELD_WHY", "checked_indices", "map_rows_to_user_order",
           "map_to_user_index",
           "LISTED_BY", "LISTED_BY_ENGINE", "LISTED_BY_USER",
           "LISTED_NOT_STATED", "MAP_KEY", "Permutation", "PermutationError",
           "SORT_HELD_FIRST", "SORT_KEY_NAMES", "SORT_KEY_WORDS",
           "SORT_TRANSPORT"]
