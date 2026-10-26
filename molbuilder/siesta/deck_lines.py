"""THE ONE WRITER OF A SIESTA DECK'S COORDINATE ROWS -- the body of
``%block AtomicCoordinatesAndAtomicSpecies`` (`engines/vibration.md`
§ 5.11 § 3; `engines/siesta.md` § 3).

MODULE  siesta.deck_lines (floor 1: the standard library alone)
ROLE    one spelling of a coordinate row, ``x y z species``, for every
        writer of one: the optimization and vibration deck (`siesta.input`),
        the TranSIESTA decks (`transport.transiesta`), and the file a run's
        own Python writes beside a phase's deck for one SIESTA call
        (`spectra.siesta_vibration`, the ``plan`` verb) -- the rows SIESTA
        reads through ``%block AtomicCoordinatesAndAtomicSpecies <
        coordinates.fdf`` (`parse.fdf.read_deck`)
TRAVELS in ``mb_vibration.pyz`` (`runwrap.VIBRATION_COMPANIONS`)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def coordinates_rows(elements: Sequence[str], positions: Iterable,
                     species_index: Dict[str, int]) -> List[str]:
    """One row per atom, in the deck's order: its position in Å to ten
    decimals and its species' 1-based index in ``ChemicalSpeciesLabel``
    (``species_index``, element -> index)."""
    return [f"{x:.10f} {y:.10f} {z:.10f} {species_index[el]}"
            for el, (x, y, z) in zip(elements, positions)]


def coordinates_text(elements: Sequence[str], positions: Iterable,
                     species_index: Dict[str, int]) -> str:
    """The rows as the file beside a deck holds them, one a line."""
    return "".join(f"{row}\n" for row in coordinates_rows(
        elements, positions, species_index))


__all__ = ["coordinates_rows", "coordinates_text"]
