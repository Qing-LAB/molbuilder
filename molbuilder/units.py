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
unit means is a fact about the FORMAT being read, and the format's own engine
is the only authority on it: libfdf refuses a physical value with no unit, so
`parse/fdf.py` passes no default and leaves a bare value unanswered, while a
tbtrans contour block is written in eV.  Each reader states its own rule and
decides whether an unknown word is a refusal — but no reader keeps its own
word list, because that is where a word goes missing.

A default here is never a judgement call.  `tests/test_siesta_keyword_smoke.py`
asks the shipped binary what the engine does and asserts this side follows.
The two defaults that were invented instead were both wrong, and one of them
refused a correct junction.

**Refusing beats assuming.**  A wrong factor is invisible in the result and
wrong by a fixed ratio in every number downstream, so an unknown unit raises
and names what it did not understand.

L1, for `quantities.py`'s reason: this is a codec on a basic unit.  Nothing
here touches a filesystem, a scheduler or a workflow, which is what lets the
parsers, the emitters and the transport composer all reach it.
"""

from __future__ import annotations

from typing import Mapping, Optional

from molbuilder.constants import (AVOGADRO as _AVOGADRO, BOHR_ANGSTROM,
                                  BOLTZMANN_EV_K, HARTREE_EV,
                                  JOULE_EV as _J_EV,
                                  KCAL_MOL_EV as _KCAL_MOL_EV,
                                  HZ_EV as _HZ_EV, CM1_EV as _CM1_EV,
                                  RYDBERG_EV)

__all__ = [
    "ENERGY_EV", "ENERGY_RY", "LENGTH_ANGSTROM", "TEMPERATURE_K",
    "UnknownUnit", "convert", "energy_ev", "energy_ry", "length_ang",
    "temperature_k",
]


class UnknownUnit(ValueError):
    """A unit word no table in this module knows."""


#: Energy unit word -> the value in eV.
#:
#: The words are SIESTA 5.4.2's own energy dimension, read out of the
#: shipped binary's libfdf table, plus `ryd`/`rydberg` as spellings this
#: project accepts on the way in.  `K`/`Kelvin` are here because SIESTA
#: files them under ENERGY, not temperature -- k_B is the pivot.
ENERGY_EV: Mapping[str, float] = {
    "ev": 1.0,
    "mev": 1e-3,
    "ry": RYDBERG_EV,
    "mry": RYDBERG_EV * 1e-3,
    "ryd": RYDBERG_EV,
    "rydberg": RYDBERG_EV,
    "ha": HARTREE_EV,
    "mha": HARTREE_EV * 1e-3,
    "hartree": HARTREE_EV,
    "mhartree": HARTREE_EV * 1e-3,
    "k": BOLTZMANN_EV_K,
    "kelvin": BOLTZMANN_EV_K,
    "j": _J_EV,
    "kj": _J_EV * 1e3,
    "erg": _J_EV * 1e-7,
    "kcal/mol": _KCAL_MOL_EV,
    "kj/mol": _J_EV * 1e3 / _AVOGADRO,
    "hz": _HZ_EV,
    "thz": _HZ_EV * 1e12,
    "cm-1": _CM1_EV,
    "cm^-1": _CM1_EV,
    "cm**-1": _CM1_EV,
}

#: Energy unit word -> the value in Ry.  DERIVED from the eV table, so a
#: word cannot be in one and missing from the other -- but `ry` lands on
#: exactly 1.0, which is why this exists rather than dividing the eV
#: answer: `200 Ry` through `v * RYDBERG_EV / RYDBERG_EV` comes back
#: 200.00000000000003, and SIESTA's own emitter writes that straight into
#: a deck.
ENERGY_RY: Mapping[str, float] = {w: ev / RYDBERG_EV
                                  for w, ev in ENERGY_EV.items()}

#: Length unit word -> the value in Ångström.  SIESTA's own length
#: dimension, plus `angstrom` as a spelling this project accepts.
LENGTH_ANGSTROM: Mapping[str, float] = {
    "ang": 1.0,
    "angstrom": 1.0,
    "bohr": BOHR_ANGSTROM,
    "m": 1e10,
    "cm": 1e8,
    "nm": 10.0,
    "pm": 1e-2,
}

#: Temperature unit word -> the value in kelvin.  DERIVED from the energy
#: table through k_B, because SIESTA files kelvin as an energy word and a
#: keyword like ``ElectronicTemperature`` takes either.
TEMPERATURE_K: Mapping[str, float] = {
    w: ev / BOLTZMANN_EV_K for w, ev in ENERGY_EV.items()
}


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
    return convert(value, unit, ENERGY_RY, what=what, source=source,
                   default=default)


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
