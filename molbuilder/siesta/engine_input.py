"""THE SAME ENGINE INPUT -- whether two SIESTA decks give the engine the same
input, the run settings aside (`engines/transport.md` § 6.1): SIESTA's
reading of *the same calculation* (`deck_record.same_calculation`,
`execution/job-request.md` § 6), never which molbuilder wrote the deck.

The SIESTA engine's own door, because every part of the answer is SIESTA's:
what its fdf reader reads (`parse.fdf`), the value it takes for a keyword a
deck leaves out (the catalogue's ``engine_default``, as a deck writes it --
`siesta.layout.unit_word`), what it reads only under a condition, the
blocks it builds itself, what only steers its scratch -- each a fact of its
source, named with its line below -- and where it places the atoms (the
reader's geometry in Å, compared by the structure's one rule,
`structure.farthest_apart`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..parse.fdf import (FDF_FALSE_WORDS, FDF_TRUE_WORDS, GEOMETRY_KEYS,
                         deck_atoms, fdf_key, fdf_true, parse_fdf_params,
                         read_fdf)
from ..engine_atom_index import siesta_numbering
from ..structure import SAME_PLACE_ANG, farthest_apart

_LOGICAL = FDF_TRUE_WORDS | FDF_FALSE_WORDS

#: THE ENGINE'S SCRATCH: where TranSIESTA keeps an electrode's surface
#: Green's function file and whether it reads an existing one back -- the
#: keywords (`m_ts_options.F90`:472,476) and the electrode block's rows
#: (`ts_electrode.F90`:606-644).  A file read back is checked against the
#: electrode it is for -- cell, k-points, weights, energy points, atoms --
#: and the run stopped on a mismatch (`m_ts_gf.F90`:139-162, `check_Green`),
#: never against the electrode's Hamiltonian: where it lies changes no
#: number, and a file read back is the same input only as one its own run
#: wrote, which a molbuilder run does, fresh for each run and each voltage
#: (`engines/transport.md` § 2a.11).  Left out of the comparison.
SCRATCH_KEYWORDS = frozenset({fdf_key("TS.Elecs.GF.ReUse"),
                              fdf_key("TS.Elecs.Out-of-core")})
ELECTRODE_BLOCK = fdf_key("TS.Elec.")
SCRATCH_ROWS = frozenset(fdf_key(w) for w in ("GF", "GF-file", "GF.ReUse",
                                               "out-of-core"))

#: READ UNDER BIAS ALONE: TranSIESTA reads its non-equilibrium contours --
#: every `TS.Contours.nEq...` keyword and block, every `TS.Contour.nEq.`
#: block -- only when |TS.Voltage| exceeds 1e-5 eV (`m_ts_options.F90`:267,
#: `IsVolt`; `m_ts_contour.f90`:74-76).
UNDER_BIAS = (fdf_key("TS.Contours.nEq"), fdf_key("TS.Contour.nEq."))
IS_VOLT_EV = 1e-5

#: THE CONTOUR THE ENGINE BUILDS ITSELF: under bias, with no
#: `TS.Contours.nEq` block, TranSIESTA builds one line, `line`, from
#: `- c kT - |V|/2` to `|V|/2 + c kT`, `delta 0.01 eV`, `method mid-rule`,
#: ``c`` the deck's `TS.Contours.nEq.Fermi.Cutoff` (`m_ts_contour_neq.f90`:
#: 150-187).  A deck that states exactly that is the same input.
NEQ_LIST, NEQ_LINE = fdf_key("TS.Contours.nEq"), fdf_key("TS.Contour.nEq.line")
NEQ_CUTOFF = fdf_key("TS.Contours.nEq.Fermi.Cutoff")
#: That line's spacing, as the engine writes it (`m_ts_contour_neq.f90`).
NEQ_LINE_DELTA = ("0.01", "eV")


def engine_input_differences(ran: str, now: str) -> List[str]:
    """Every difference between the input SIESTA reads from the deck a run
    ran (``ran``) and from the deck its stage renders now (``now``), in
    words -- ``[]`` when they are the same engine input
    (`engines/transport.md` § 6.1).

    Both are read as SIESTA reads them (:func:`_as_siesta_reads`): comments
    are not input -- molbuilder's blocks among them; the engine's scratch is
    not input; what it reads only under bias counts only under bias; a block
    it builds itself is what a deck that leaves it out gives it.  A keyword
    as fdf names it; a value as a value -- a number as a number, a logical as
    fdf reads it, a unit as written; a block row by row.  A keyword one deck
    leaves out is the same input as the other's statement of the engine's
    own default (:func:`engine_defaults`).  The geometry is compared where
    SIESTA places it, in Å (:func:`_geometry_difference`).  A RUN SETTING'S
    KEYWORD is no part of it (:func:`run_setting_keywords`): it says how
    fast and where a run computes, never what (`execution/job-request.md`
    § 2)."""
    spelled: Dict[str, str] = {}
    ran_keys, ran_blocks = _as_siesta_reads(*read_fdf(ran, spelled_out=spelled),
                                            spelled)
    now_keys, now_blocks = _as_siesta_reads(*read_fdf(now, spelled_out=spelled),
                                            spelled)
    aside = run_setting_keywords()
    ran_keys = {k: v for k, v in ran_keys.items() if k not in aside}
    now_keys = {k: v for k, v in now_keys.items() if k not in aside}
    defaults = engine_defaults()
    out: List[str] = []
    for key in sorted((set(ran_keys) | set(now_keys)) - GEOMETRY_KEYS):
        a, b = ran_keys.get(key), now_keys.get(key)
        if a is not None and b is not None:
            if not _same(a, b):
                out.append(f"{spelled[key]}: the run had {' '.join(a)}, it "
                           f"renders {' '.join(b)} now")
            continue
        stated = a if a is not None else b
        if key in defaults and _same(stated, defaults[key]):
            continue                 # the engine's own default, stated
        out.append(f"{spelled[key]} {' '.join(stated)}: "
                   + ("the run stated it and the deck now leaves it out"
                      if a is not None else
                      "the deck states it now and the run left it out"))
    for key in sorted((set(ran_blocks) | set(now_blocks)) - GEOMETRY_KEYS):
        a, b = ran_blocks.get(key), now_blocks.get(key)
        if a is None or b is None:
            out.append(f"block {spelled[key]}: "
                       + ("the run stated it and the deck now does not"
                          if a is not None else
                          "the deck states it now and the run did not"))
            continue
        if len(a) != len(b):
            out.append(f"block {spelled[key]}: the run had {len(a)} rows, "
                       f"it renders {len(b)}")
            continue
        for i, (ra, rb) in enumerate(zip(a, b), start=1):
            if not _same(ra, rb):
                out.append(f"block {spelled[key]}, row {i}: the run had "
                           f"{' '.join(ra)}, it renders {' '.join(rb)} now")
    geometry = _geometry_difference(ran, now)
    return out + ([geometry] if geometry else [])


def _as_siesta_reads(keys: Dict[str, List[str]],
                     blocks: Dict[str, List[List[str]]],
                     spelled: Dict[str, str]
                     ) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """One deck's keywords and blocks as SIESTA reads them: the engine's
    scratch left out (:data:`SCRATCH_KEYWORDS`, :data:`SCRATCH_ROWS`); what
    it reads under bias alone left out at zero bias (:data:`UNDER_BIAS`);
    under bias, the contour it builds itself where the deck states none
    (:data:`NEQ_LIST`)."""
    keys = {k: v for k, v in keys.items() if k not in SCRATCH_KEYWORDS}
    blocks = {k: ([r for r in rows if not (r and fdf_key(r[0]) in SCRATCH_ROWS)]
                  if k.startswith(ELECTRODE_BLOCK) else rows)
              for k, rows in blocks.items()}
    if not _under_bias(keys):
        return ({k: v for k, v in keys.items() if not k.startswith(UNDER_BIAS)},
                {k: v for k, v in blocks.items()
                 if not k.startswith(UNDER_BIAS)})
    if NEQ_LIST not in blocks:
        cutoff = (keys.get(NEQ_CUTOFF)
                  or engine_defaults().get(NEQ_CUTOFF))[0]
        spelled.setdefault(NEQ_LIST, "TS.Contours.nEq")
        spelled.setdefault(NEQ_LINE, "TS.Contour.nEq.line")
        blocks = {**blocks,
                  NEQ_LIST: [["line"]],
                  NEQ_LINE: [["part", "line"],
                             ["from", "-", cutoff, "kT", "-", "|V|/2", "to",
                              "|V|/2", "+", cutoff, "kT"],
                             ["delta", *NEQ_LINE_DELTA],
                             ["method", "mid-rule"]]}
    return keys, blocks


def _under_bias(keys: Dict[str, List[str]]) -> bool:
    """Whether the deck is under bias as TranSIESTA decides it:
    |TS.Voltage| above :data:`IS_VOLT_EV` (`m_ts_options.F90`:267); a deck
    that states none is at zero."""
    from ..constants import RYDBERG_EV
    from ..units import energy_ry
    toks = keys.get(fdf_key("TS.Voltage"))
    if not toks:
        return False
    volts_ry = energy_ry(float(toks[0]), toks[1] if len(toks) > 1 else None,
                         what="TS.Voltage", source="the deck")
    return abs(volts_ry * RYDBERG_EV) > IS_VOLT_EV


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
            had.atomic_numbers, had.labels, has.atomic_numbers, has.labels)):
        if (za, la) != (zb, lb):
            return (f"{block}, SIESTA's atom {siesta_numbering(i)}: the run "
                    f"had {la} (Z {za}), it renders {lb} (Z {zb}) now")
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
            return (f"{what}, " + (f"SIESTA's atom {siesta_numbering(i)}"
                                   if key.startswith("atomic") else
                                   f"lattice vector index {i}")
                    + f": {d:.2e} Å from where the run had it")
    return None


def run_setting_keywords() -> frozenset:
    """Every fdf keyword a SIESTA run setting writes -- the catalogue's
    `execution` items' ``anchor`` and ``expands`` (`template.md` § 6.2),
    asked, never listed."""
    from ..template import catalogue, is_run_setting
    out = set()
    for it in catalogue().items:
        if not is_run_setting(it) or (
                it.engines and "siesta" not in it.engines):
            continue
        for word in ((it.anchor,) if it.anchor else ()) + tuple(
                it.expands or ()):
            out.add(fdf_key(word))
    return frozenset(out)


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


__all__ = ["engine_input_differences", "engine_defaults",
           "run_setting_keywords"]
