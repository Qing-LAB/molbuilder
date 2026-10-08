"""What a transmission run's ``<label>.TBT.nc`` says -- the device's DOS, its
parts, the leads' DOS and the eigenchannels -- read for the transport report
(`web/results.md` § 2.5; `engines/transport.md` § 2a.12).

**sisl reads the file** (its TBtrans reader, `sisl.io.tbtrans.
tbtncSileTBtrans`); nothing here parses NetCDF by hand.  sisl is in the
`molbuilder` env (`envs/host-env.txt`), which the server runs in.

What the transmission deck asks TBtrans for (`transport/deck.py`
``TBT_SECTION``) is what is here: ``TBT.DOS.Gf`` -- the device Green-function
DOS per orbital, energy and k; ``TBT.DOS.A`` -- the spectral DOS of the FIRST
electrode only (TBtrans skips the last without ``TBT.DOS.A.All``,
``m_tbt_save.F90``); ``TBT.DOS.Elecs`` -- each lead's bulk DOS; ``TBT.T.Eig``
-- the transmission eigenvalues.  Every quantity is k-averaged, on the
file's own energy grid, which is the grid of ``.TBT.AVTRANS_*``: TBtrans
writes that file from the same variable (``state_cdf2ascii``).

**An orbital's type is the device run's own ``<label>.ORB_INDX``**, which
SIESTA writes by default (``WriteOrbitalIndex``, ``read_options.F90``): its
``l`` and ``m``, named as SIESTA names them (``atmfuncs.f``
``symfio``: for ``l = 1``, ``m = -1, 0, 1`` are p_y, p_z, p_x).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: The orbital types a selection may be narrowed to, as ``(l, m)`` --
#: ``m`` ``None`` for every orbital of that ``l``.  SIESTA's own naming
#: (``atmfuncs.f`` ``symfio``); a polarization orbital is of its ``l``.
ORBITAL_TYPES: Dict[str, tuple] = {
    "s": (0, None), "p": (1, None), "d": (2, None),
    "px": (1, 1), "py": (1, -1), "pz": (1, 0),
}

#: What the tab says under the orbital menu (`web/results.md` § 2.5).
ORBITAL_NOTE = ("TBtrans writes each orbital's own DOS, not the terms "
                "between orbitals: a p orbital is exact along x, y or z, "
                "so a ring's π orbitals are p_x, p_y or p_z only when the "
                "ring lies in a coordinate plane.")


class TbtError(Exception):
    """The file cannot answer what was asked -- the message says why."""


def tbt_file(run_dir: Path, label: str) -> Optional[Path]:
    """The transmission run's ``<label>.TBT.nc`` in ``run_dir``, or
    ``None`` -- the spin-polarized run's two files are plan K21's."""
    p = Path(run_dir) / f"{label}.TBT.nc"
    return p if p.is_file() else None


def _open(nc: Path):
    import sisl
    try:
        return sisl.get_sile(str(nc))
    except Exception as exc:                                  # noqa: BLE001
        raise TbtError(f"{Path(nc).name} does not read: {exc}") from exc


def _floats(a) -> List[float]:
    return [float(x) for x in a]


def _written(nc: Path) -> set:
    """What TBtrans wrote, as ``"DOS"`` / ``"<elec>/ADOS"`` -- the file's
    own variables, read with netCDF4: a quantity it was not asked for is
    absent, and absence is decided here, never by a read that failed."""
    import netCDF4
    with netCDF4.Dataset(str(nc)) as ds:
        out = set(ds.variables)
        for g, grp in ds.groups.items():
            out |= {f"{g}/{v}" for v in grp.variables}
    return out


def point_dos(nc: Path, regions: Dict[str, Sequence[int]]) -> Dict:
    """The DOS block of one bias point (`web/results.md` § 2.5):
    ``{energy_ev, total, by_label, lead_spectral, lead_bulk, eigenchannels}``
    -- each k-averaged on the file's energy grid, ``by_label`` the device DOS
    summed over each region's atoms (0-based, the device run's order) that
    are in the device.  A quantity TBtrans did not write is absent, never
    zero."""
    tbt = _open(nc)
    have = _written(nc)
    dev = set(int(a) for a in tbt.a_dev)
    out: Dict = {"energy_ev": _floats(tbt.E)}
    if "DOS" in have:
        out["total"] = _floats(tbt.DOS())
        by_label = {}
        for label, atoms in regions.items():
            mine = sorted(int(a) for a in atoms if int(a) in dev)
            if mine:
                by_label[label] = _floats(tbt.DOS(atoms=mine))
        out["by_label"] = by_label
    spectral = {e: _floats(tbt.ADOS(e)) for e in tbt.elecs
                if f"{e}/ADOS" in have}
    bulk = {e: _floats(tbt.BDOS(e)) for e in tbt.elecs
            if f"{e}/DOS" in have}
    if spectral:
        out["lead_spectral"] = spectral
    if bulk:
        out["lead_bulk"] = bulk
    if len(tbt.elecs) >= 2:
        a, b = tbt.elecs[0], tbt.elecs[1]
        if f"{a}/{b}.T.Eig" in have:
            eig = tbt.transmission_eig(a, b)
            out["eigenchannels"] = [_floats(eig[:, i])
                                    for i in range(eig.shape[1])]
    return out


def selection_pdos(nc: Path, orb_indx: Optional[Path],
                   atoms: Sequence[int], orbitals: str = "all") -> Dict:
    """The PDOS of ``atoms`` (0-based, the device run's order), narrowed to
    one orbital type (:data:`ORBITAL_TYPES`, or ``"all"``):
    ``{energy_ev, pdos, atoms, outside_device}``.  Atoms outside the device
    region (a buffer) have no DOS and are named, not counted."""
    tbt = _open(nc)
    dev = set(int(a) for a in tbt.a_dev)
    asked = sorted({int(a) for a in atoms})
    if not asked:
        raise TbtError("no atoms selected")
    inside = [a for a in asked if a in dev]
    outside = [a for a in asked if a not in dev]
    if not inside:
        raise TbtError("none of the selected atoms is in the device "
                       "region, where TBtrans computes the DOS")
    if "DOS" not in _written(nc):
        raise TbtError(f"{Path(nc).name} holds no device DOS -- the run "
                       f"was not asked for it (TBT.DOS.Gf)")
    if orbitals == "all":
        pdos = tbt.DOS(atoms=inside)
    else:
        if orbitals not in ORBITAL_TYPES:
            raise TbtError(f"orbital type {orbitals!r}: one of "
                           f"all, {', '.join(ORBITAL_TYPES)}")
        if orb_indx is None or not Path(orb_indx).is_file():
            raise TbtError("the device run's .ORB_INDX is not there, so "
                           "its orbitals' types are not known")
        orbs = _orbitals_of(tbt, Path(orb_indx), inside, orbitals)
        if not orbs:
            raise TbtError(f"the selected atoms have no {orbitals} "
                           f"orbital in their basis")
        pdos = tbt.DOS(orbitals=orbs)
    return {"energy_ev": _floats(tbt.E), "pdos": _floats(pdos),
            "atoms": inside, "outside_device": outside,
            "orbitals": orbitals}


def _orbitals_of(tbt, orb_indx: Path, atoms: Sequence[int],
                 kind: str) -> List[int]:
    """The unit-cell orbital indices of ``atoms`` of type ``kind`` -- each
    atom's basis as the device run's ``.ORB_INDX`` lists it (sisl's reader,
    one entry per atom, in the run's order).  Its orbital counts must be the
    ``.TBT.nc``'s, atom by atom, or the two files are not of one run and the
    answer is refused."""
    import sisl
    l_want, m_want = ORBITAL_TYPES[kind]
    try:
        basis = sisl.get_sile(str(orb_indx)).read_basis()
    except Exception as exc:                                  # noqa: BLE001
        raise TbtError(f"{orb_indx.name} does not read: {exc}") from exc
    firsto = [int(x) for x in tbt.geometry.firsto]
    counts = [len(basis[i].orbitals) for i in range(len(basis))]
    if counts != [firsto[i + 1] - firsto[i] for i in range(len(firsto) - 1)]:
        raise TbtError(f"{orb_indx.name} and {Path(tbt.file).name} list "
                       f"different orbitals per atom -- not one run's files")
    out: List[int] = []
    for a in atoms:
        for j, orb in enumerate(basis[a].orbitals):
            if orb.l == l_want and (m_want is None or orb.m == m_want):
                out.append(firsto[a] + j)
    return out
