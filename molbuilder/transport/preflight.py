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

Scientific basis + the parameter rationale: transiesta-workflow.md § 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_BOHR_ANG = 0.529177


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


# NOTE (backend-architecture.md V3): ``Check`` / ``PreflightReport`` are a
# CHECKLIST type, deliberately distinct from the shared ``Issue`` / validation
# pass — NOT accidental duplication.  This validator compares TWO run
# directories (device vs electrode .fdf), which the (struct, cfg) engine-
# validator registry can't express, and it reports PASSING gates too (the
# "ok" severity) so ``format_report`` can print a full device<->electrode
# checklist.  ``Issue`` has no "ok" state (an ok gate is simply the absence of
# an Issue).  The two type systems are BRIDGED (``to_issues`` below) so a
# future web transport-preflight surface can consume the problems uniformly,
# without collapsing the checklist affordance the CLI needs.
@dataclass
class Check:
    id: str
    severity: str          # "error" | "warn" | "ok" | "info"
    message: str

    def to_issue(self) -> "Optional[Issue]":
        """The problem-severity view of this check as a shared ``Issue``
        (``where`` = the check id), or None for an "ok" pass — which
        ``Issue`` has no way to represent."""
        from ..issues import Issue
        if self.severity == "ok":
            return None
        return Issue(severity=self.severity, message=self.message,
                     where=self.id)


@dataclass
class PreflightReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def n_error(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warn")

    def ok(self) -> bool:
        return self.n_error == 0

    def to_issues(self) -> "List[Issue]":
        """The cross-run findings as shared ``Issue`` objects (the
        error/warn/info checks; "ok" passes drop out).  The bridge that
        lets this CLI-only checklist feed an Issue-based surface without
        being a disconnected third type system (V3)."""
        return [i for i in (c.to_issue() for c in self.checks)
                if i is not None]


def _vecs_close(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    return all(abs(a[i][j] - b[i][j]) <= tol for i in range(2)
               for j in range(3))


def preflight(device: FdfParams, electrode: FdfParams, *,
              tol_ang: float = 0.05,
              min_electrode_thickness_ang: float = 12.0,
              electrode_kz_warn: int = 20,
              z_vacuum_warn_ang: float = 3.0) -> PreflightReport:
    """Run the cross-run consistency gates (transiesta-workflow.md § 6.3).

    ``device`` is the NEGF run, ``electrode`` the bulk lead run.
    """
    out: List[Check] = []

    def add(cid, sev, msg):
        out.append(Check(cid, sev, msg))

    # roles
    if device.solution_method != "transiesta":
        add("role.device", "warn",
            f"device SolutionMethod is {device.solution_method!r}, expected "
            f"'transiesta' (is this really the NEGF run?)")
    if electrode.solution_method == "transiesta":
        add("role.electrode", "warn",
            "electrode SolutionMethod is 'transiesta'; the lead run should "
            "be a plain SCF (diagon) that saves the .TSHS")

    # C7: electrode must write the .TSHS
    if not electrode.saves_ts_hs:
        add("electrode.saveHS", "warn",
            "electrode fdf has no TS.HS.Save/SaveHS true -> it won't write "
            "the .TSHS the device needs (transiesta-workflow.md § 4 step 2)")

    # C4: transverse lattice vectors must match (clone prerequisite)
    cells_match = _vecs_close(device.cell_ang, electrode.cell_ang, tol_ang)
    if device.cell_ang and electrode.cell_ang:
        if not cells_match:
            add("cell.transverse", "error",
                "device and electrode transverse lattice vectors (a,b) "
                "differ -> the electrode cannot be projected onto the "
                "device leads (need the same lateral supercell, § 4 'Golden "
                "Rule').")
        else:
            add("cell.transverse", "ok",
                "device/electrode lateral cell (a,b) match")

    # C1: transverse k commensurate (require equal when cells match)
    if device.kgrid and electrode.kgrid:
        if cells_match and device.kgrid[:2] != electrode.kgrid[:2]:
            add("kgrid.transverse", "error",
                f"device transverse k {device.kgrid[:2]} != electrode "
                f"{electrode.kgrid[:2]}; with the same lateral cell they must "
                f"be equal (TBtrans requires commensurate device/electrode k).")
        elif not cells_match:
            add("kgrid.transverse", "warn",
                "lateral cells differ -> verify the device/electrode k are "
                "commensurate (Bloch expansion).")
        else:
            add("kgrid.transverse", "ok",
                f"transverse k match ({device.kgrid[:2]})")

    # C2: device kz must be 1; electrode kz must be dense bulk sampling
    if device.kgrid:
        if device.kgrid[2] != 1:
            add("kgrid.device_kz", "error",
                f"device kz = {device.kgrid[2]} (must be 1 -- NEGF treats z "
                f"as OPEN; kz>1 imposes a fake Bloch periodicity along the "
                f"wire and breaks transport, § 4 / § 1).")
        else:
            add("kgrid.device_kz", "ok", "device kz = 1")
    if electrode.kgrid:
        ekz = electrode.kgrid[2]
        if ekz <= 1:
            add("kgrid.electrode_kz", "error",
                "electrode kz = 1; the lead is a PERIODIC BULK run and needs "
                "a dense, converged kz (a thin cell has a large BZ along z). "
                "kz=1 gives a wrong electrode Hamiltonian (§ 4.2).")
        elif ekz < electrode_kz_warn:
            add("kgrid.electrode_kz", "warn",
                f"electrode kz = {ekz} looks low for a bulk lead; converge it "
                f"(start ~80) -- the result must be kz-converged (§ 4.2).")
        else:
            add("kgrid.electrode_kz", "ok", f"electrode kz = {ekz} (dense)")

    # C3: identical numerical contract
    for cid, label, dv, ev in (
        ("contract.meshcutoff", "MeshCutoff (Ry)",
         device.mesh_cutoff_ry, electrode.mesh_cutoff_ry),
        ("contract.energyshift", "PAO.EnergyShift (Ry)",
         device.energy_shift_ry, electrode.energy_shift_ry),
        ("contract.xc", "XC", device.xc, electrode.xc),
        ("contract.basis", "PAO.BasisSize",
         device.basis_size, electrode.basis_size),
    ):
        if dv is None or ev is None:
            add(cid, "warn", f"{label}: missing in one fdf "
                             f"(device={dv}, electrode={ev}) -- cannot verify")
        elif isinstance(dv, float) and abs(dv - ev) > 1e-9:
            add(cid, "error", f"{label} mismatch: device={dv}, "
                              f"electrode={ev} (must be identical, § 4.3).")
        elif dv != ev:
            add(cid, "error", f"{label} mismatch: device={dv!r}, "
                              f"electrode={ev!r} (must be identical, § 4.3).")
        else:
            add(cid, "ok", f"{label} = {dv} (matches)")

    # C5: electrode thickness vs orbital range (heuristic)
    if electrode.z_len_ang is not None:
        if electrode.z_len_ang < min_electrode_thickness_ang:
            add("electrode.thickness", "warn",
                f"electrode z-length {electrode.z_len_ang:.1f} Ang < "
                f"~{min_electrode_thickness_ang:.0f} Ang: may be thinner than "
                f"the electronic PRINCIPAL LAYER (orbital range ~2x the "
                f"EnergyShift radius spans ~3 Au(111) layers; the principal "
                f"layer is typically ~6 layers). TranSIESTA's electrode "
                f"check will confirm (§ 4.1).")
        else:
            add("electrode.thickness", "ok",
                f"electrode z-length {electrode.z_len_ang:.1f} Ang")

    # C6: device z-vacuum (slab junction wants ~zero)
    if device.z_len_ang is not None and device.atom_z_span_ang is not None:
        gap = device.z_len_ang - device.atom_z_span_ang
        if gap > z_vacuum_warn_ang:
            add("device.z_vacuum", "warn",
                f"device cell has ~{gap:.1f} Ang of z-vacuum (cell z "
                f"{device.z_len_ang:.1f} - atom span {device.atom_z_span_ang:.1f}). "
                f"A slab junction needs ~zero z-vacuum or the frozen leads "
                f"are strained (§ 4 / § 1). Add the bulk interlayer gap so z "
                f"wraps seamlessly.")
        else:
            add("device.z_vacuum", "ok",
                f"device z-vacuum {gap:.1f} Ang (~seamless)")

    return PreflightReport(checks=out)


def preflight_files(device_path, electrode_path, **opts) -> PreflightReport:
    dev = parse_fdf_params(Path(device_path).read_text(encoding="utf-8",
                                                        errors="replace"))
    ele = parse_fdf_params(Path(electrode_path).read_text(encoding="utf-8",
                                                          errors="replace"))
    return preflight(dev, ele, **opts)


_SEV_TAG = {"error": "ERROR", "warn": "WARN ", "ok": "ok   ", "info": "info "}


def format_report(report: PreflightReport) -> str:
    lines = ["transport preflight -- device <-> electrode consistency"]
    for c in report.checks:
        lines.append(f"  [{_SEV_TAG.get(c.severity, c.severity)}] "
                     f"{c.id:<24} {c.message}")
    lines.append(f"  => {report.n_error} error(s), {report.n_warn} warning(s)")
    if report.ok():
        lines.append("  PASS (no blocking inconsistency)")
    else:
        lines.append("  FAIL -- fix the ERROR(s) before running TranSIESTA")
    return "\n".join(lines)


__all__ = [
    "FdfParams", "Check", "PreflightReport", "parse_fdf_params",
    "preflight", "preflight_files", "format_report",
]
