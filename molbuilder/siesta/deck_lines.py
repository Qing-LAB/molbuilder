"""THE ONE WRITER OF A SIESTA DECK'S GEOMETRY -- the counts, and each
block whole: ``%block AtomicCoordinatesAndAtomicSpecies`` (`engines/
vibration.md` § 5.11 § 3; `engines/siesta.md` § 3), ``%block
LatticeVectors`` and ``%block ChemicalSpeciesLabel``, and their rows.

MODULE  siesta.deck_lines (floor 1: the standard library alone)
ROLE    one spelling of each geometry row -- a coordinate row,
        ``x y z species``; a lattice vector; a species, ``index Z label``
        -- for every writer of one: the optimization and vibration deck
        (`siesta.input`),
        the TranSIESTA decks (`transport.transiesta`), and the file a run's
        own Python writes beside a phase's deck for one SIESTA call
        (`spectra.siesta_vibration`, the ``plan`` verb) -- the rows SIESTA
        reads through ``%block AtomicCoordinatesAndAtomicSpecies <
        coordinates.fdf`` (`parse.fdf.read_deck`)
TRAVELS in ``mb_vibration.pyz`` (`runwrap.VIBRATION_COMPANIONS`)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


def species_index(species: Sequence[str]) -> Dict[str, int]:
    """Each species' 1-based index in ``ChemicalSpeciesLabel``, in the
    deck's species order (``species``, `chemistry.species_order`) -- the
    index a coordinate row names its species by."""
    return {label: i for i, label in enumerate(species, 1)}


def species_rows(species: Sequence[str], numbers: Sequence[int]
                 ) -> List[str]:
    """The body of ``%block ChemicalSpeciesLabel``: ``index Z label`` per
    species, in the deck's species order (``species``), ``numbers`` each
    label's atomic number -- the label the person's, Z derived
    (`model/chemistry.md` § 3)."""
    index = species_index(species)
    return [f"{index[label]} {z} {label}"
            for label, z in zip(species, numbers)]


def lattice_rows(cell: Iterable) -> List[str]:
    """The body of ``%block LatticeVectors``: one row per vector, in Å to
    twelve decimals (under ``LatticeConstant 1.0 Ang``)."""
    return [f"{a:.12f} {b:.12f} {c:.12f}" for a, b, c in cell]


def coordinates_rows(elements: Sequence[str], positions: Iterable,
                     species_index: Dict[str, int]) -> List[str]:
    """One row per atom, in the deck's order: its position in Å to ten
    decimals and its species' 1-based index in ``ChemicalSpeciesLabel``
    (``species_index``, element -> index)."""
    return [f"{x:.10f} {y:.10f} {z:.10f} {species_index[el]}"
            for el, (x, y, z) in zip(elements, positions)]


def count_lines(n_atoms: int, n_species: int) -> List[str]:
    """``NumberOfAtoms`` and ``NumberOfSpecies``."""
    return [f"NumberOfAtoms     {n_atoms}", f"NumberOfSpecies   {n_species}"]


def lattice_block(cell: Iterable) -> List[str]:
    """The lattice, whole: ``LatticeConstant 1.0 Ang`` and the vectors'
    block (:func:`lattice_rows`)."""
    return ["LatticeConstant 1.0 Ang", "%block LatticeVectors",
            *lattice_rows(cell), "%endblock LatticeVectors"]


def species_block(species: Sequence[str], numbers: Sequence[int]
                  ) -> List[str]:
    """The species' block, whole (:func:`species_rows`)."""
    return ["%block ChemicalSpeciesLabel", *species_rows(species, numbers),
            "%endblock ChemicalSpeciesLabel"]


def coordinates_block(rows: Sequence[str] = (), *,
                      beside: Optional[str] = None) -> List[str]:
    """The coordinates, whole: their format and their block -- the rows
    (:func:`coordinates_rows`), or, for a deck whose rows lie in the file
    ``beside`` it, libfdf's include of that file (`engines/vibration.md`
    § 5.11 § 4)."""
    if beside is not None:
        return ["AtomicCoordinatesFormat Ang",
                f"%block AtomicCoordinatesAndAtomicSpecies < {beside}"]
    return ["AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies", *rows,
            "%endblock AtomicCoordinatesAndAtomicSpecies"]


def coordinates_text(elements: Sequence[str], positions: Iterable,
                     species_index: Dict[str, int]) -> str:
    """The rows as the file beside a deck holds them, one a line."""
    return "".join(f"{row}\n" for row in coordinates_rows(
        elements, positions, species_index))


__all__ = ["coordinates_block", "coordinates_rows", "coordinates_text",
           "count_lines", "lattice_block", "lattice_rows", "species_block",
           "species_index", "species_rows"]
