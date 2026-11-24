"""Every way a physical quantity is written in an engine's files.

MODULE  units (floor 0; imports ``constants`` and nothing else)
ROLE    the dialects — which words an energy, a length or a temperature may
        be spelled in, and the ONE door that turns one into a number; and
        molbuilder's one list of the units it writes (``_TABLE``: each
        spelling, its meaning, how a name spells it, how a page shows it --
        its columns :data:`UNITS`, :data:`UNIT_IN_NAME`,
        :data:`UNIT_ON_A_PAGE`), the unit a name carries
        (:func:`unit_of_name`) and a unit's spelling from its look
        (:func:`unit_spelled`)
USED-BY every reader of an engine file; the one data-file writer
        (`ncdata.check_said`); a structure's row validator
        (`structure._customized_row`, `model/structure.md` § 2.2d); the
        vibration frame sets' vocabulary (`vibration_frames.Row`); the tests
        that read a deck

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

A default here is never a judgement call.  `tests/test_siesta_keyword_smoke_e2e.py`
asks the shipped binary what the engine does and asserts this side follows.

**Refusing beats assuming.**  A wrong factor is invisible in the result and
wrong by a fixed ratio in every number downstream, so an unknown unit raises
and names what it did not understand.

L1, for `quantities.py`'s reason: this is a codec on a basic unit.  Nothing
here touches a filesystem, a scheduler or a workflow, which is what lets the
parsers, the emitters and the transport composer all reach it.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

# TWO WAYS, because the fdf reader that reads unit words through this module
# travels beside a SIESTA vibration job (`runwrap.members`).
try:                                        # inside molbuilder
    from .constants import (AVOGADRO as _AVOGADRO, BOHR_ANGSTROM,
                            BOLTZMANN_EV_K, HARTREE_EV,
                            JOULE_EV as _J_EV,
                            KCAL_MOL_EV as _KCAL_MOL_EV,
                            HZ_EV as _HZ_EV, CM1_EV as _CM1_EV,
                            RYDBERG_EV)
    from . import constants as _C
except ImportError:                         # beside a job, in mb_vibration.pyz
    from constants import (AVOGADRO as _AVOGADRO, BOHR_ANGSTROM,
                           BOLTZMANN_EV_K, HARTREE_EV,
                           JOULE_EV as _J_EV,
                           KCAL_MOL_EV as _KCAL_MOL_EV,
                           HZ_EV as _HZ_EV, CM1_EV as _CM1_EV,
                           RYDBERG_EV)
    import constants as _C  # type: ignore[no-redef]

__all__ = [
    "ENERGY_EV", "ENERGY_RY", "LENGTH_ANGSTROM", "TEMPERATURE_K",
    "UNITS", "UNIT_IN_NAME", "UNIT_ON_A_PAGE",
    "UnknownUnit", "convert", "energy_ev", "energy_ry", "length_ang",
    "temperature_k", "unit_of_name", "unit_spelled",
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


def size_words(n_bytes: float) -> str:
    """A size in bytes as a person reads it -- ``49.2 GiB``, ``245.0 KiB``
    -- in the largest binary unit it reaches, one decimal: the one spelling
    every line about a file's size uses."""
    for unit, power in (("PiB", 5), ("TiB", 4), ("GiB", 3), ("MiB", 2),
                        ("KiB", 1)):
        if n_bytes >= 1024 ** power:
            return f"{n_bytes / 1024 ** power:.1f} {unit}"
    return f"{int(n_bytes)} B"


def volts_words(v: float) -> str:
    """A voltage as a person reads it -- *0 V*, *0.2 V*, *-0.5 V* -- its
    number spelled as a bias point's folder spells it (`task.bias_token`):
    the one spelling every sentence about a voltage uses; a table's aligned
    column keeps its own width (`frame_average.summary_lines`)."""
    return f"{float(v):g} V"


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


def _scientific(value: float, digits: int) -> str:
    """``value`` as a unit's meaning states it -- ``7.748092e-5``, its
    exponent unpadded."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


#: MOLBUILDER'S ONE LIST OF UNITS (`model/datafiles.md` § 2c), a row per
#: unit: its ONE SPELLING -- every unit a data file states a number in --
#: what it means in words, how a NAME spells it (§ 2a rule 2: a number's name
#: ends with it -- ``force_ev_per_ang`` -- and a number of no unit, ``1``,
#: carries none), and how a PAGE shows it (none for a number of no unit).
#: A new unit is a row here and in § 2c, in the change that first writes it.
#: :data:`UNITS`, :data:`UNIT_IN_NAME` and :data:`UNIT_ON_A_PAGE` are its
#: columns.
_TABLE: Tuple[Tuple[str, str, str, str], ...] = (
    ("1", "no unit -- a count, an index, a fraction or a flag", "", ""),
    ("%", "per cent", "percent", "%"),
    ("s", "second", "s", "s"),
    ("K", "kelvin", "k", "K"),
    ("V", "volt", "v", "V"),
    ("A", "ampere", "a", "A"),
    ("e", f"the elementary charge, "
          f"{_scientific(_C.ELEMENTARY_CHARGE_C, 9)} C", "e", "e"),
    ("eV", f"electronvolt, {_scientific(_C.ELEMENTARY_CHARGE_C, 9)} J",
     "ev", "eV"),
    ("eV/angstrom", "electronvolt per angstrom", "ev_per_ang", "eV/Å"),
    ("kbar", "kilobar, 1e8 Pa", "kbar", "kbar"),
    ("1/eV", "per electronvolt", "per_ev", "1/eV"),
    ("Eh", f"hartree, the atomic unit of energy, {_C.HARTREE_EV:.6f} eV",
     "eh", "Eₕ"),
    ("Eh^2", "hartree squared", "eh2", "Eₕ²"),
    ("Eh/K", "hartree per kelvin", "eh_per_k", "Eₕ/K"),
    ("Eh/bohr", f"hartree per bohr, the atomic unit of force (1 bohr = "
                f"{_C.BOHR_ANGSTROM:.6f} angstrom)", "eh_per_bohr", "Eₕ/bohr"),
    ("atm", f"standard atmosphere, {_C.ATM_PA:g} Pa", "atm", "atm"),
    ("G0", f"the conductance quantum, 2e^2/h = "
           f"{_scientific(_C.CONDUCTANCE_QUANTUM_S, 6)} S", "g0", "G₀"),
    ("nm", "nanometre", "nm", "nm"),
    ("cm-1", "wavenumber, per centimetre", "cm1", "cm⁻¹"),
    ("km/mol", "kilometre per mole", "km_per_mol", "km/mol"),
    ("angstrom", "angstrom, 1e-10 m", "ang", "Å"),
    ("angstrom^3", "cubic angstrom", "ang3", "Å³"),
    ("amu", f"the unified atomic mass unit (dalton), "
            f"{_scientific(_C.AMU_KG, 8)} kg", "amu", "amu"),
    ("amu^1/2 angstrom", "square-root amu times angstrom", "amu12_ang",
     "amu½·Å"),
    ("amu^-1/2", "per square-root amu", "per_amu12", "amu⁻½"),
    ("angstrom^2/amu^1/2", "square angstrom per square-root amu",
     "ang2_per_amu12", "Å²/amu½"),
    ("D/(angstrom amu^1/2)", f"debye ({_scientific(_C.DEBYE_C_M, 5)} C m) "
                             f"per angstrom per square-root amu",
     "d_per_ang_amu12", "D/(Å·amu½)"),
    ("angstrom^4/amu", "angstrom to the fourth per amu", "ang4_per_amu",
     "Å⁴/amu"),
    ("mdyn/angstrom", "millidyne per angstrom, 100 N/m -- a force constant "
                      "as Gaussian prints it", "mdyn_per_ang", "mdyn/Å"),
    ("kcal/mol", f"kilocalorie per mole, {_C.CALORIE_J:g} kJ/mol",
     "kcal_per_mol", "kcal/mol"),
    ("GiB", "gibibyte, 2^30 bytes", "gib", "GiB"),
)
#: EACH UNIT'S ONE SPELLING, with what it means in words (:data:`_TABLE`):
#: what a data file states a number in (`ncdata.check_said`).  A
#: structure's row may state any spelling (`model/structure.md` § 2.2d).
UNITS: Dict[str, str] = {unit: means for unit, means, _, _ in _TABLE}
#: EACH UNIT AS A NAME SPELLS IT (:data:`_TABLE`): asked by the one data-file
#: writer (`ncdata.check_said`) and read by :func:`unit_of_name`.
UNIT_IN_NAME: Dict[str, str] = {unit: in_name
                                for unit, _, in_name, _ in _TABLE}
#: EACH UNIT AS A PAGE SHOWS IT (:data:`_TABLE`): the look the vibration
#: frame sets' words give a row's unit (`vibration_frames.Row.page_unit`),
#: and the look of the unit a name carries that a row's ``unit`` may repeat
#: (`model/structure.md` § 2.2d); empty for a number of no unit.
UNIT_ON_A_PAGE: Dict[str, str] = {unit: on_page
                                  for unit, _, _, on_page in _TABLE}


def unit_of_name(name: str) -> Optional[str]:
    """THE UNIT A NAME CARRIES (`model/datafiles.md` § 2c): the unit whose
    name spelling (:data:`UNIT_IN_NAME`) the name ends with after an
    underscore -- the longest such, so ``force_ev_per_ang`` is eV/angstrom,
    not angstrom -- or ``None`` where it ends with none, or with its inverse
    (``x_per_ang`` carries no unit of the table's).  A bare name is a name,
    never a unit: a lattice constant ``a`` carries no ampere."""
    best: Optional[str] = None
    for unit, spelled in UNIT_IN_NAME.items():
        if spelled and name.endswith("_" + spelled) \
                and (best is None or len(spelled) > len(UNIT_IN_NAME[best])):
            best = unit
    if best is None:
        return None
    spelled = UNIT_IN_NAME[best]
    if not spelled.startswith("per_") and name.endswith("_per_" + spelled):
        return None
    return best


def unit_spelled(word: Optional[str]) -> Optional[str]:
    """THE TABLE'S SPELLING of a unit written in it or as a page shows it
    (:data:`UNIT_ON_A_PAGE`: ``Å`` is ``angstrom``), or ``None`` for a word
    the table does not hold -- a unit a person spells otherwise (``deg``,
    ``fs``), which a structure's row may state (`model/structure.md`
    § 2.2d)."""
    if word in UNITS:
        return word
    for spelled, look in UNIT_ON_A_PAGE.items():
        if look and look == word:
            return spelled
    return None


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants",)
