"""THE SAME ENGINE INPUT -- whether two SIESTA decks give the engine the same
input (`engines/transport.md` § 6.1): the question the transport gather asks
of a run before it takes its files, never which molbuilder wrote the deck.

The SIESTA engine's own door, because every part of the answer is SIESTA's:
what its fdf reader reads (`parse.fdf`), the value it takes for a keyword a
deck leaves out (the catalogue's ``engine_default``, as a deck writes it --
`siesta.layout.unit_word`), and where it places the atoms (the reader's
geometry in Å, compared by the structure's one rule,
`structure.farthest_apart`).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..parse.fdf import (FDF_FALSE_WORDS, FDF_TRUE_WORDS, GEOMETRY_KEYS,
                         deck_atoms, fdf_key, fdf_true, parse_fdf_params,
                         read_fdf)
from ..structure import SAME_PLACE_ANG, farthest_apart

_LOGICAL = FDF_TRUE_WORDS | FDF_FALSE_WORDS


def engine_input_difference(ran: str, now: str) -> Optional[str]:
    """``None`` when the deck a run ran (``ran``) and the deck its stage
    renders now (``now``) give SIESTA the same input, else the first
    difference, in words (`engines/transport.md` § 6.1).

    Both are read as SIESTA reads them: comments are not input --
    molbuilder's blocks among them; a keyword as fdf names it; a value as a
    value -- a number as a number, a logical as fdf reads it, a unit as
    written; a block row by row.  The geometry is compared where SIESTA
    places it, in Å (:func:`_geometry_difference`).  A keyword one deck
    leaves out is the same input as the other's statement of the engine's
    own default (:func:`engine_defaults`)."""
    spelled: Dict[str, str] = {}
    ran_keys, ran_blocks = read_fdf(ran, spelled_out=spelled)
    now_keys, now_blocks = read_fdf(now, spelled_out=spelled)
    defaults = engine_defaults()
    for key in sorted((set(ran_keys) | set(now_keys)) - GEOMETRY_KEYS):
        a, b = ran_keys.get(key), now_keys.get(key)
        if a is not None and b is not None:
            if not _same(a, b):
                return (f"{spelled[key]}: the run had {' '.join(a)}, it "
                        f"renders {' '.join(b)} now")
            continue
        stated = a if a is not None else b
        if key in defaults and _same(stated, defaults[key]):
            continue                 # the engine's own default, stated
        return (f"{spelled[key]} {' '.join(stated)}: "
                + ("the run stated it and the deck now leaves it out"
                   if a is not None else
                   "the deck states it now and the run left it out"))
    for key in sorted((set(ran_blocks) | set(now_blocks)) - GEOMETRY_KEYS):
        a, b = ran_blocks.get(key), now_blocks.get(key)
        if a is None or b is None:
            return (f"block {spelled[key]}: "
                    + ("the run stated it and the deck now does not"
                       if a is not None else
                       "the deck states it now and the run did not"))
        if len(a) != len(b):
            return (f"block {spelled[key]}: the run had {len(a)} rows, it "
                    f"renders {len(b)}")
        for i, (ra, rb) in enumerate(zip(a, b), start=1):
            if not _same(ra, rb):
                return (f"block {spelled[key]}, row {i}: the run had "
                        f"{' '.join(ra)}, it renders {' '.join(rb)} now")
    return _geometry_difference(ran, now)


def _same(a: List[str], b: List[str]) -> bool:
    """Two value lists, word for word as SIESTA reads them: a number as a
    number, a logical as fdf's (``T`` is ``true``), any other word
    case-blind."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if float(x) != float(y):
                return False
            continue
        except ValueError:
            pass
        if x.lower() in _LOGICAL and y.lower() in _LOGICAL:
            if fdf_true([x]) != fdf_true([y]):
                return False
        elif x.lower() != y.lower():
            return False
    return True


def _geometry_difference(ran: str, now: str) -> Optional[str]:
    """The two decks' geometry, where SIESTA places it: each atom the same
    species, at the same place, and each lattice vector the same, within
    :data:`~molbuilder.structure.SAME_PLACE_ANG` -- read in Å by the deck
    reader (:attr:`~molbuilder.parse.fdf.FdfParams.coords_ang`,
    ``cell_ang``), the coordinates' format, ``LatticeConstant`` and units
    folded in.  An atom is named by its row of the coordinates block, the
    deck's own numbering."""
    block = "block AtomicCoordinatesAndAtomicSpecies"
    try:
        had, has = deck_atoms(ran), deck_atoms(now)
    except ValueError as exc:
        return f"{block}: {exc}"
    if len(had.labels) != len(has.labels):
        return (f"{block}: the run had {len(had.labels)} atoms, it renders "
                f"{len(has.labels)} now")
    for i, (za, la, zb, lb) in enumerate(zip(
            had.atomic_numbers, had.labels, has.atomic_numbers, has.labels),
            start=1):
        if (za, la) != (zb, lb):
            return (f"{block}, row {i}: the run had {la} (Z {za}), it "
                    f"renders {lb} (Z {zb}) now")
    params = [parse_fdf_params(text) for text in (ran, now)]
    states = [read_fdf(text)[1] for text in (ran, now)]
    for what, key, a, b in (
            (block, "atomiccoordinatesandatomicspecies",
             params[0].coords_ang, params[1].coords_ang),
            ("block LatticeVectors", "latticevectors",
             params[0].cell_ang, params[1].cell_ang)):
        if not any(key in s for s in states):
            continue                    # neither deck states it
        if a is None or b is None:
            return (f"{what}: "
                    + ("the run's" if a is None else "the deck's now")
                    + " does not read in Å" + (
                        "" if all(key in s for s in states)
                        else " -- one deck does not state it"))
        i, d = farthest_apart(a, b)
        if d > SAME_PLACE_ANG:
            return (f"{what}, row {i + 1}: {d:.2e} Å from where the run had "
                    f"it")
    return None


def engine_defaults() -> Dict[str, List[str]]:
    """``{fdf keyword's key: [value, unit]}`` -- every SIESTA item whose
    catalogue row states the engine's own default (``engine_default``), as
    a deck states it: the value and the unit word fdf reads beside it."""
    from ..template import catalogue
    from .layout import unit_word
    out: Dict[str, List[str]] = {}
    for it in catalogue().items:
        if not it.engine_default or not it.anchor or (
                it.engines and "siesta" not in it.engines):
            continue
        word = unit_word(it.name)
        out[fdf_key(it.anchor)] = [str(it.engine_default[0])] + (
            [word] if word else [])
    return out


__all__ = ["engine_input_difference", "engine_defaults"]
