"""Every way a physical quantity is written in an engine's files.

MODULE  units (floor 0; imports ``constants`` and nothing else)
ROLE    the dialects — which words an energy, a length or a temperature may
        be spelled in, and the ONE door that turns one into a number
USED-BY every reader of an engine file, and the tests that read a deck

**One object, one module.**  These are not a handful of helpers that happen
to take a string and a unit word; they are the complete set of dialects a
physical quantity is spoken in, and they belong together because the whole
cost of getting this wrong is paid when two of them are used on the same
file.  `scheduler/quantities.py` makes the same argument for a wall time and
a memory size, for the same reason and after the same kind of bug.

**The dialects disagree, which is what makes the distinction load-bearing.**
SIESTA writes ``MeshCutoff`` in Ry, ``ElectronicTemperature`` in K *or* in an
energy, and accepts eV for either.  A reader that knows only some of those
words does not fail — it returns the number it was given, in the wrong unit,
and every value downstream is wrong by a fixed ratio.  The ratio between a
Rydberg and an electronvolt is 13.6.

**The vocabulary is shared; the POLICY is the caller's.**  What a missing
unit means is a fact about the format being read, not about the quantity: a
bare energy in an ``.fdf`` really is Ry, and one in a tbtrans contour block
really is eV.  So each reader passes its own *default* and decides whether an
unknown word is a refusal — but no reader keeps its own word list, because
that is where a word goes missing.

**Refusing beats assuming.**  A wrong factor is invisible in the result and
wrong by a fixed ratio in every number downstream, so an unknown unit raises
and names what it did not understand.

L1, for `quantities.py`'s reason: this is a codec on a basic unit.  Nothing
here touches a filesystem, a scheduler or a workflow, which is what lets the
parsers, the emitters and the transport composer all reach it.
"""

from __future__ import annotations

from typing import Mapping, Optional

from molbuilder.constants import (BOHR_ANGSTROM, BOLTZMANN_EV_K, HARTREE_EV,
                                  RYDBERG_EV)

__all__ = [
    "ENERGY_EV", "LENGTH_ANGSTROM", "TEMPERATURE_K",
    "UnknownUnit", "convert", "energy_ev", "energy_ry", "length_ang",
    "temperature_k",
]


class UnknownUnit(ValueError):
    """A unit word no table in this module knows."""


#: Energy unit word -> the value in eV.  The words are what SIESTA, TranSIESTA
#: and tbtrans accept in the files this project reads.
ENERGY_EV: Mapping[str, float] = {
    "ev": 1.0,
    "mev": 1e-3,
    "ry": RYDBERG_EV,
    "ryd": RYDBERG_EV,
    "rydberg": RYDBERG_EV,
    "ha": HARTREE_EV,
    "hartree": HARTREE_EV,
}

#: Length unit word -> the value in Ångström.
LENGTH_ANGSTROM: Mapping[str, float] = {
    "ang": 1.0,
    "angstrom": 1.0,
    "bohr": BOHR_ANGSTROM,
    "nm": 10.0,
}

#: Temperature unit word -> the value in kelvin.  An ENERGY written here is
#: converted through Boltzmann's constant, because SIESTA's
#: ``ElectronicTemperature`` accepts either and means the same physics by
#: both — which is the one place the two vocabularies meet.
TEMPERATURE_K: Mapping[str, float] = dict(
    {"k": 1.0, "kelvin": 1.0},
    **{word: ev / BOLTZMANN_EV_K for word, ev in ENERGY_EV.items()},
)


def convert(value: float, unit: Optional[str], table: Mapping[str, float], *,
            what: str, source: str, default: Optional[str] = None) -> float:
    """*value* written in *unit*, as the number *table* is keyed to.

    *default* is the unit a value with no unit word is written in — the
    format's rule, which is why the caller states it.  ``None`` means a
    bare value is refused rather than guessed at.

    *what* and *source* only appear in the refusal, so a person reading it
    knows which field of which file to look at.
    """
    key = (unit or "").strip().lower()
    if not key:
        if default is None:
            raise UnknownUnit(
                f"{source}: {what} states no unit, and this format has no "
                f"default for it -- refusing rather than picking one, "
                f"because a wrong factor is invisible in the result and "
                f"wrong by a fixed ratio in every number after it")
        key = default.strip().lower()
    factor = table.get(key)
    if factor is None:
        raise UnknownUnit(
            f"{source}: {what} carries unit {unit!r}, which this reader does "
            f"not know how to convert (known: {', '.join(sorted(table))}).  "
            f"Refusing rather than assuming a factor -- a wrong one is "
            f"invisible in the result and wrong by a fixed ratio in every "
            f"number downstream.")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise UnknownUnit(
                f"{source}: {what} is {value!r}, which is not a "
                f"number") from None
    try:
        # Not `float(value)`: an array of coordinates converts elementwise,
        # and a netCDF reader hands one straight through.
        return value * factor
    except TypeError:
        raise UnknownUnit(
            f"{source}: {what} is {value!r}, which is not a number") from None


def energy_ev(value, unit=None, *, what="an energy", source="the deck",
              default=None) -> float:
    """An energy in eV."""
    return convert(value, unit, ENERGY_EV, what=what, source=source,
                   default=default)


def energy_ry(value, unit=None, *, what="an energy", source="the deck",
              default=None) -> float:
    """An energy in Ry — what SIESTA's own scalars are stored as."""
    return energy_ev(value, unit, what=what, source=source,
                     default=default) / RYDBERG_EV


def length_ang(value, unit=None, *, what="a length", source="the deck",
               default=None) -> float:
    """A length in Ångström."""
    return convert(value, unit, LENGTH_ANGSTROM, what=what, source=source,
                   default=default)


def temperature_k(value, unit=None, *, what="a temperature",
                  source="the deck", default=None) -> float:
    """A temperature in kelvin, from a temperature OR an energy."""
    return convert(value, unit, TEMPERATURE_K, what=what, source=source,
                   default=default)
