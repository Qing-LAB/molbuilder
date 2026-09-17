"""TranSIESTA consistency preflight -- the cross-run contract gates.

A TranSIESTA conductance run is three coupled calculations (relax -> bulk
electrode ``.TSHS`` -> NEGF device) whose CORRECTNESS hinges on the device
and electrode runs sharing one numerical contract + a geometric clone +
commensurate k, with the electrode a dense-``kz`` bulk and the device
``kz=1``.  Humans break exactly these couplings; this module turns the
prose "Golden Rule" (docs/engines/transport.md) into
hard gates over the two ``.fdf`` files that will actually run.

It parses BOTH fdfs and reports OK / WARN / ERROR per check.  It validates
*consistency*, not physics it can't see -- so it is engine-truthful: it
catches the device-vs-electrode mismatches that silently give wrong
transmission, before any compute is spent.

Scientific basis + the parameter rationale: ``engines/transport.md`` § 5 (the
invariant set, each gate traced to its reference) and § 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from molbuilder.constants import BOHR_ANGSTROM as _BOHR_ANG


def _norm(key: str) -> str:
    return key.lower().replace(".", "").replace("-", "").replace("_", "")


def _parse_fdf(text: str) -> Tuple[Dict[str, List[str]],
                                   Dict[str, List[List[str]]]]:
    """``(scalars, blocks)`` from fdf text; first occurrence wins."""
    scalars: Dict[str, List[str]] = {}
    blocks: Dict[str, List[List[str]]] = {}
    in_block: Optional[str] = None
    rows: List[List[str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("%block"):
            in_block = _norm(line[len("%block"):].strip())
            rows = []
            continue
        if low.startswith("%endblock"):
            if in_block is not None:
                blocks.setdefault(in_block, rows)
            in_block = None
            continue
        if in_block is not None:
            rows.append(line.split())
            continue
        toks = line.split()
        if toks:
            scalars.setdefault(_norm(toks[0]), toks[1:])
    return scalars, blocks


def _to_float(s) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _ry(toks: List[str]) -> Optional[float]:
    """A SIESTA energy scalar (value [unit]) in Ry."""
    if not toks:
        return None
    v = _to_float(toks[0])
    if v is None:
        return None
    unit = toks[1].lower() if len(toks) > 1 else "ry"
    return v * 2.0 if unit in ("ha", "hartree") else v


@dataclass
class FdfParams:
    kgrid: Optional[Tuple[int, int, int]] = None
    mesh_cutoff_ry: Optional[float] = None
    energy_shift_ry: Optional[float] = None
    xc: Optional[str] = None
    basis_size: Optional[str] = None
    cell_ang: Optional[List[List[float]]] = None      # 3 lattice vectors
    z_len_ang: Optional[float] = None
    atom_z_span_ang: Optional[float] = None           # max_z - min_z
    solution_method: Optional[str] = None
    saves_ts_hs: bool = False
    n_atoms: Optional[int] = None
    #: The deck's own START positions in Ang, deck order — the frozen
    #: gate's baseline under the § 4.1b citation condition (form A:
    #: start = the deck's coordinates, end = the .XV).  None when the
    #: coordinate block is absent or its format cannot be converted.
    coords_ang: Optional[List[List[float]]] = None
    #: Fermi smearing in K (None when the deck leaves SIESTA's default).
    electronic_temperature_k: Optional[float] = None
    #: XC as the deck SPELLS it (``xc`` above is the normalised
    #: comparison key; these carry the verbatim words for a consumer
    #: that re-emits them -- the transport composite's `config_for`).
    xc_functional: Optional[str] = None
    xc_authors: Optional[str] = None


def parse_fdf_params(text: str) -> FdfParams:
    sc, bl = _parse_fdf(text)
    p = FdfParams()

    if "kgridmonkhorstpack" in bl and len(bl["kgridmonkhorstpack"]) >= 3:
        try:
            rows = bl["kgridmonkhorstpack"][:3]
            p.kgrid = tuple(int(float(rows[i][i])) for i in range(3))
        except (ValueError, IndexError):
            pass

    if "meshcutoff" in sc:
        p.mesh_cutoff_ry = _ry(sc["meshcutoff"])
    if "paoenergyshift" in sc:
        p.energy_shift_ry = _ry(sc["paoenergyshift"])
    if "paobasissize" in sc:
        p.basis_size = sc["paobasissize"][0] if sc["paobasissize"] else None

    # ElectronicTemperature: an energy-or-temperature scalar.  K and
    # meV cover what molbuilder's own emitters write; anything else is
    # left None (deck default) rather than converted wrongly.
    if "electronictemperature" in sc and sc["electronictemperature"]:
        toks = sc["electronictemperature"]
        v = _to_float(toks[0])
        unit = toks[1].lower() if len(toks) > 1 else "k"
        if v is not None:
            if unit == "k":
                p.electronic_temperature_k = v
            elif unit == "mev":
                p.electronic_temperature_k = v * 11.604518            # k_B

    func = (sc.get("xcfunctional") or ["LDA"])[0]
    auth = (sc.get("xcauthors") or ["CA"])[0]
    p.xc = f"{func.upper()}/{auth.upper()}"
    if sc.get("xcfunctional"):
        p.xc_functional = sc["xcfunctional"][0]
    if sc.get("xcauthors"):
        p.xc_authors = sc["xcauthors"][0]

    if "solutionmethod" in sc and sc["solutionmethod"]:
        p.solution_method = sc["solutionmethod"][0].lower()

    # TS.HS.Save / TS.SaveHS / SaveHS  (any truthy => writes .TSHS)
    for k in ("tshssave", "tssavehs", "savehs"):
        if k in sc and sc[k] and sc[k][0].lower() in ("t", "true", ".true.",
                                                      "yes"):
            p.saves_ts_hs = True

    # ----- geometry: cell + atom z-span -----
    lat_const = 1.0
    if "latticeconstant" in sc and sc["latticeconstant"]:
        v = _to_float(sc["latticeconstant"][0])
        if v is not None:
            unit = (sc["latticeconstant"][1].lower()
                    if len(sc["latticeconstant"]) > 1 else "ang")
            lat_const = v * _BOHR_ANG if unit.startswith("bohr") else v
    if "latticevectors" in bl and len(bl["latticevectors"]) >= 3:
        try:
            vecs = [[lat_const * float(x) for x in row[:3]]
                    for row in bl["latticevectors"][:3]]
            p.cell_ang = vecs
            p.z_len_ang = abs(vecs[2][2])
        except (ValueError, IndexError):
            pass

    coords = bl.get("atomiccoordinatesandatomicspecies")
    if coords:
        p.n_atoms = len(coords)
        fmt = (sc.get("atomiccoordinatesformat") or ["ang"])[0].lower()
        # Full positions in Ang (the frozen gate's baseline).  Three
        # convertible formats; fractional needs the cell.  A format this
        # cannot convert leaves coords_ang None — callers say so rather
        # than guessing.
        xyz: List[List[float]] = []
        for row in coords:
            if len(row) < 3:
                xyz = []
                break
            v = [_to_float(row[0]), _to_float(row[1]), _to_float(row[2])]
            if any(x is None for x in v):
                xyz = []
                break
            xyz.append(v)          # type: ignore[arg-type]
        if xyz:
            if fmt.startswith(("frac", "scaledbylatticevectors")):
                if p.cell_ang:
                    c = p.cell_ang
                    xyz = [[sum(f[k] * c[k][d] for k in range(3))
                            for d in range(3)] for f in xyz]
                else:
                    xyz = []
            elif fmt.startswith(("bohr", "notscaledcartesianbohr")):
                xyz = [[x * _BOHR_ANG for x in f] for f in xyz]
            elif not fmt.startswith(("ang", "notscaledcartesianang")):
                xyz = []           # an unknown format is not guessed at
        if xyz:
            p.coords_ang = xyz
            zs = [f[2] for f in xyz]
            p.atom_z_span_ang = max(zs) - min(zs)
        else:
            zs = []
            for row in coords:
                if len(row) >= 3:
                    z = _to_float(row[2])
                    if z is None:
                        continue
                    if fmt.startswith("frac") and p.z_len_ang:
                        z *= p.z_len_ang
                    elif fmt.startswith("bohr"):
                        z *= _BOHR_ANG
                    zs.append(z)
            if zs:
                p.atom_z_span_ang = max(zs) - min(zs)
    return p


# --------------------------------------------------------------------- #
#  the gates                                                            #
# --------------------------------------------------------------------- #


# ===================================================================== #
#  THE CROSS-DECK COMPARISON DELETED 2026-09-17
#
#  `Check`, `PreflightReport`, `preflight`, `preflight_files` and
#  `format_report` compared two FINISHED `.fdf` files -- a device deck against
#  an electrode deck -- and reported `engines/transport.md` 5's thirteen
#  invariants as an error/warn/ok checklist.  They were written 2026-06-27 for
#  the hand-assembly workflow, where a person wrote both decks and nothing else
#  compared them: "Humans break exactly these couplings", as the header said.
#
#  Under the composite there is no second deck to disagree with.  Both are
#  derived from ONE citation and resolved from ONE template, so eleven of the
#  thirteen hold BY CONSTRUCTION or by a live gate, and I11 is held BETTER --
#  `compose.py` reads real orbital interaction ranges from the citation's own
#  `.ion` files, retiring the ~12 A floor this module used, which passes a
#  4.8 A three-layer Au block.  The two that were held here alone, I9 and I12,
#  moved to `_validate_transport_kind` on 2026-09-17, keyed on the calculation
#  KIND so they fire on every prep rather than on a command somebody remembers.
#
#  WHAT SURVIVES ABOVE IS THE READER, and it is the reason this file stays:
#  `parse_fdf_params` has four production callers (`citation_defaults`,
#  `compose`, `parse/contract`, the transport blueprint) and `_BOHR_ANG` one.
#  Reading an fdf and COMPARING two of them are different jobs; only the second
#  one lost its subject.
# ===================================================================== #

__all__ = ["FdfParams", "parse_fdf_params"]
