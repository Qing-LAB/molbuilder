"""What a transmission point says of its DOS -- its total, its regions', the
leads', the eigenchannels and a selection's PDOS -- for the transport report
(`web/results.md` § 2.5; `engines/transport.md` § 2a.12).

**Read through the SIESTA family's parser** (`model/parse.md` § 5e): the
averages from TBtrans's k-averaged text files, on the data file's own
energies (`tbtrans_nc.point_averages`); the DOS per orbital from the data
file's kept reading (`tbtrans_nc.kept`), summed here over a region's atoms or
a selection's, narrowed to an orbital type by the device run's own
``<label>.ORB_INDX``.  No file's format is read here: the parser reads them.
**A reading not kept is no region's DOS and no PDOS** -- said, never read
here: a data file is read on the person's word
(`tbtrans_nc.TBTransNcFileParser.parse`).

**An orbital's type is the device run's ``.ORB_INDX``**, which SIESTA writes
by default (``WriteOrbitalIndex``, ``read_options.F90``): its ``l`` and
``m``, named as SIESTA names them (``atmfuncs.f`` ``symfio``: for ``l = 1``,
``m = -1, 0, 1`` are p_y, p_z, p_x).
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

#: The words a point without its kept reading is said in.
NOT_READ = ("its DOS per orbital is not read yet -- read it from the "
            "report's DOS tab, or `jobset summarize task`")


class TbtError(Exception):
    """The files cannot answer what was asked -- the message says why."""


def _channels(spin: str) -> tuple:
    return ("up", "down") if spin == "polarized" else ("unpolarized",)


def energies_of(folder, label: str) -> List[float]:
    """The energies a point's run computed, ascending, at full precision
    -- its first data file's, through the parser
    (`tbtrans_nc.energies`; a polarized run's two are one grid), never the
    five decimals its text files print (`engines/transport.md` § 2a.16).
    :class:`TbtError` when the folder holds none."""
    from ..parse.engines.tbtrans_nc import data_files, energies
    files = data_files(folder, label)
    if not files:
        raise TbtError(f"{Path(folder).name} holds no data file of {label}")
    return list(energies(next(iter(files.values()))))


def point_data_files(folder, label: str) -> List[Path]:
    """Every data file a point's DOS by region asks: the point's own and
    those in its ``shared/``, recursively (`engines/transport.md`
    § 2a.16)."""
    from ..parse.engines.tbtrans_nc import data_files
    from .energies import SHARED_DIR
    folder = Path(folder)
    out = list(data_files(folder, label).values())
    if (folder / SHARED_DIR).is_dir():
        out += point_data_files(folder / SHARED_DIR, label)
    return out


def readings(folder, label: str) -> List[object]:
    """What is kept of each of the point's data files' readings
    (:func:`point_data_files`, `tbtrans_nc.kept_state`); a run of ours holds
    each, so a kept one stating no user index is out of date
    (`model/overview.md` § 2.2b)."""
    from ..parse.engines.tbtrans_nc import kept_state
    return [kept_state(p, need_user=True)
            for p in point_data_files(folder, label)]


def _sum(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def point_atoms(folder):
    """THE ATOM INDEX MAP of a transmission point's files -- its run's deck's
    (`runs.atom_index_map_of`, `model/overview.md` § 2.2b), and its
    ``shared/`` files' too: the finished point it shares with ran the same
    structure (`engines/transport.md` § 2a.16)."""
    from ..runs import atom_index_map_of, run_of
    run = run_of(folder)
    if run is None:
        raise TbtError(f"{Path(folder).name} is no run of ours: nothing "
                       f"states its atoms' indices")
    return atom_index_map_of(run)


def _checked(reading, expected, name: str) -> None:
    """THE DIGEST AGAINST ITS CALCULATION'S RECORD (`model/overview.md`
    § 2.2b, F7): the map a reading states must be ``expected`` -- the
    composed junction's, every junction rung's deck being written from it
    -- or the regions and a selection would name other atoms; refused by
    name."""
    got = reading.atom_index_map
    if got is None or (got.user, got.engine) != (expected.user,
                                                 expected.engine):
        raise TbtError(
            f"{name}: its atoms' index map is not the composed junction's "
            f"-- a reading of another structure, or of none of ours; read "
            f"it again (model/overview.md 2.2b)")


def _by_label(reading, regions: Dict[str, Sequence[int]]
              ) -> Dict[str, List[float]]:
    """Each region's DOS: the reading's columns of its atoms (0-based, the
    device run's order) that are in the device, summed."""
    import numpy as np
    atom = np.asarray(reading.atom)
    device = set(reading.atom)
    out = {}
    for name, atoms in regions.items():
        mine = sorted(int(a) for a in atoms if int(a) in device)
        if mine:
            out[name] = [float(x) for x in
                         reading.dos[:, np.isin(atom, mine)].sum(axis=1)]
    return out


#: THE CURVES A DOS BLOCK MAY HOLD (:func:`point_dos`) -- a block holding
#: none is no DOS: the record says why, never a blank plot (`web/results.md`
#: § 2.5).
DOS_CURVES = ("total", "by_label", "lead_spectral", "lead_bulk",
              "eigenchannels", "by_spin")

#: Why a point's block holds no curve: TBtrans wrote none, or, a point that
#: shares energies, not every part of its grid holds one (`energies.joined`
#: keeps no value on part of a grid).
NO_CURVE = ("its files hold no DOS over its whole grid -- TBtrans wrote none "
            "(TBT.DOS.Gf, TBT.DOS.Elecs and TBT.T.Eig off), or, for a point "
            "that shares energies, not every part of its grid holds one")


def point_dos(folder, label: str, regions: Dict[str, Sequence[int]],
              spin: str, expected, *, av: Optional[Dict] = None) -> Dict:
    """THE DOS BLOCK of a point's folder (`web/results.md` § 2.5):
    ``{energy_ev, total, by_label, lead_spectral, lead_bulk, eigenchannels}``
    -- the averages TBtrans wrote, on the data file's energies; ``by_label``
    each region's, from the data file's kept reading, present only when it
    is kept.  A polarized point's two channels summed, and ``by_spin`` each
    channel's own ``total`` and ``eigenchannels`` beside.  A quantity
    TBtrans did not write is absent, never zero.  ``regions`` count by
    engine index; ``expected`` is the composed junction's atom index map,
    which each kept reading's must be (F7).  ``av`` is the point's averages
    where the caller read them already (`record.point_averages_of`, one read
    for the transmission and the DOS)."""
    from ..parse.engines.tbtrans_nc import data_files, kept
    if av is None:
        from .record import point_averages_of
        try:
            av = point_averages_of(folder, label)
        except (OSError, ValueError) as exc:
            raise TbtError(str(exc)) from exc
    channels = _channels(spin)
    missing = [c for c in channels if c not in av]
    if missing:
        raise TbtError(f"{Path(folder).name} holds no k-averaged file of "
                       f"the {' or '.join(missing)} channel")
    data = data_files(folder, label)
    per: Dict[str, Dict] = {}
    for c in channels:
        a = av[c]
        d: Dict = {"energy_ev": a["energy_ev"]}
        for src, dst in (("dos", "total"), ("lead_spectral", "lead_spectral"),
                         ("lead_bulk", "lead_bulk"),
                         ("eigenchannels", "eigenchannels")):
            if src in a:
                d[dst] = a[src]
        reading = kept(data[c], need_user=True) if c in data else None
        if reading is not None and reading.dos is not None and regions:
            _checked(reading, expected, data[c].name)
            d["by_label"] = _by_label(reading, regions)
        per[c] = d
    if len(channels) == 1:
        return per[channels[0]]
    up, dn = per["up"], per["down"]
    if len(up["energy_ev"]) != len(dn["energy_ev"]) or any(
            abs(x - y) > 1e-9 for x, y in zip(up["energy_ev"],
                                              dn["energy_ev"])):
        raise TbtError(f"{Path(folder).name}: its two spin channels are on "
                       f"different energy grids -- not one run's two spins")
    out: Dict = {"energy_ev": up["energy_ev"]}
    if "total" in up and "total" in dn:
        out["total"] = _sum(up["total"], dn["total"])
    if "by_label" in up and "by_label" in dn:
        out["by_label"] = {k: _sum(v, dn["by_label"][k])
                           for k, v in up["by_label"].items()
                           if k in dn["by_label"]}
    for key in ("lead_spectral", "lead_bulk"):
        both = {e: _sum(v, dn[key][e]) for e, v in up.get(key, {}).items()
                if e in dn.get(key, {})}
        if both:
            out[key] = both
    out["by_spin"] = {
        s: {k: d[k] for k in ("total", "eigenchannels") if k in d}
        for s, d in (("up", up), ("down", dn))}
    return out


def selection_pdos(folder, label: str, orb_indx: Optional[Path],
                   atoms: Sequence[int], orbitals: str = "all", *,
                   expected) -> Dict:
    """The PDOS of ``atoms`` (0-based, the device run's order) in a point's
    folder, narrowed to one orbital type (:data:`ORBITAL_TYPES`, or
    ``"all"``): ``{energy_ev, pdos, atoms, outside_device, orbitals}``, from
    each data file's kept reading -- a polarized point's two channels
    summed.  Atoms outside the device region (a buffer) have no DOS and are
    named, not counted.  :class:`TbtError` names what is missing: a reading
    not kept (:data:`NOT_READ`), a device DOS not written, a reading whose
    atom index map is not ``expected``, the composed junction's (F7)."""
    import numpy as np
    from ..parse.engines.tbtrans_nc import data_files, kept
    asked = sorted({int(a) for a in atoms})
    if not asked:
        raise TbtError("no atoms selected")
    if orbitals != "all" and orbitals not in ORBITAL_TYPES:
        raise TbtError(f"orbital type {orbitals!r}: one of all, "
                       f"{', '.join(ORBITAL_TYPES)}")
    files = data_files(folder, label)
    if not files:
        raise TbtError(f"{Path(folder).name} holds no data file of {label}")
    total = None
    inside: List[int] = []
    outside: List[int] = []
    energy: List[float] = []
    for path in files.values():
        reading = kept(path, need_user=True)
        if reading is None:
            raise TbtError(f"{path.name}: {NOT_READ}")
        _checked(reading, expected, path.name)
        if reading.dos is None:
            raise TbtError(f"{path.name} holds no device DOS -- the run was "
                           f"not asked for it (TBT.DOS.Gf)")
        device = set(reading.atom)
        inside = [a for a in asked if a in device]
        outside = [a for a in asked if a not in device]
        if not inside:
            raise TbtError("none of the selected atoms is in the device "
                           "region, where TBtrans computes the DOS")
        if orbitals == "all":
            mask = np.isin(np.asarray(reading.atom), inside)
        else:
            if orb_indx is None or not Path(orb_indx).is_file():
                raise TbtError("the device run's .ORB_INDX is not there, so "
                               "its orbitals' types are not known")
            orbs = _orbitals_of(reading.firsto, Path(orb_indx), inside,
                                orbitals, path.name)
            if not orbs:
                raise TbtError(f"the selected atoms have no {orbitals} "
                               f"orbital in their basis")
            mask = np.isin(np.asarray(reading.orbital), orbs)
        mine = reading.dos[:, mask].sum(axis=1)
        total = mine if total is None else total + mine
        energy = list(reading.energy_ev)
    return {"energy_ev": energy, "pdos": [float(x) for x in total],
            "atoms": inside, "outside_device": outside,
            "orbitals": orbitals}


def _orbitals_of(firsto: Sequence[int], orb_indx: Path,
                 atoms: Sequence[int], kind: str, name: str) -> List[int]:
    """The geometry's orbital indices of ``atoms`` of type ``kind`` -- each
    atom's basis as the device run's ``.ORB_INDX`` lists it (sisl's reader,
    one entry per atom, in the run's order).  Its orbital counts must be the
    data file's, atom by atom (its reading's ``firsto``), or the two are not
    of one run and the answer is refused."""
    import sisl
    l_want, m_want = ORBITAL_TYPES[kind]
    try:
        basis = sisl.get_sile(str(orb_indx)).read_basis()
    except Exception as exc:                                  # noqa: BLE001
        raise TbtError(f"{orb_indx.name} does not read: {exc}") from exc
    firsto = [int(x) for x in firsto]
    counts = [len(basis[i].orbitals) for i in range(len(basis))]
    if counts != [firsto[i + 1] - firsto[i] for i in range(len(firsto) - 1)]:
        raise TbtError(f"{orb_indx.name} and {name} list different orbitals "
                       f"per atom -- not one run's files")
    out: List[int] = []
    for a in atoms:
        for j, orb in enumerate(basis[a].orbitals):
            if orb.l == l_want and (m_want is None or orb.m == m_want):
                out.append(firsto[a] + j)
    return out


def point_readings(calc) -> List[Dict]:
    """EACH FINISHED TRANSMISSION POINT'S READINGS (`model/parse.md` § 5e):
    the transmission's newest run, as the record reads it
    (`record.result_folders`) -- per point its run finished, as the status
    door's rows say it, read once for the run (`continuation.points_state`,
    `architecture.md` § 3.2), its folder and what is kept of each data
    file's reading, its own and those in its ``shared/`` (:func:`readings`).
    A point still running, or stopped, is none: its data file is TBtrans's
    while it writes, and a partial one after; a run not launched has no
    row, and no point finished."""
    from ..calcdirs import RunPlace
    from ..jobset.continuation import points_state
    from ..jobset.naming import stage_home
    from ..runrecord import LaunchRecordError, launch_record
    from .record import result_folders
    base, task = Path(calc.base), calc.task
    names = stage_home(base, task, "transmission").names
    folders = result_folders(base, task, "transmission")
    rows: Dict[str, Dict] = {}
    for run in {r for _, w, r in folders if w is not None}:
        try:
            launch = launch_record(run)
        except LaunchRecordError:
            continue
        if launch is None:
            continue
        try:
            got = points_state(run, names, task, "transmission",
                               launch=launch, base=base)[2]
        except LaunchRecordError:
            # ITS INDEX DOES NOT READ -- an older molbuilder's: no point of
            # it is read, the stage's row saying why.
            continue
        rows.update({p["folder"]: p for p in got})
    return [{"point": pt, "folder": where,
             "readings": readings(where, task.label)}
            for pt, where, run in folders
            if where is not None
            and (rows.get(RunPlace.of(run).walked(pt.rel)) or {}).get("state")
            == "finished"]


def read_points(calc, *, fresh: bool = False, take_over: bool = False,
                say=None, progress=None) -> List[Dict]:
    """EVERY FINISHED POINT'S DATA FILES READ whose reading is not kept --
    each one, with ``fresh`` -- one at a time, through the parser, which
    claims each and keeps its reading (`tbtrans_nc.TBTransNcFileParser.
    parse`; `model/parse.md` § 5e).  A file a reading on this host holds is
    not read; one whose reader is gone is (the parser removes its stale
    claim); one a reading on another host holds, only with ``take_over``,
    the person's word.  ``say(words)`` is told as each begins and ends,
    ``progress(path, k, n)`` as each k-point is read.  Answers each file's
    outcome: ``{"file", "state", "said"}`` -- ``kept``, ``being read`` or
    ``failed``."""
    import time
    from ..parse.engines.tbtrans_nc import (ReadingClaimed,
                                            TBTransNcFileParser,
                                            claim_words)
    from ..report_fields import span_words
    from ..units import size_words
    base = Path(calc.base)
    tell = say or (lambda _w: None)
    out: List[Dict] = []
    for entry in point_readings(calc):
        # THE POINT'S ATOM INDEX MAP, its run's deck's -- each digest states
        # it (`model/overview.md` § 2.2b), its `shared/` ones too.
        atoms = None
        for k in entry["readings"]:
            rel = str(k.path.relative_to(base))
            if k.state == "kept" and not fresh:
                continue
            alive = (k.claim or {}).get("alive")
            if k.state == "being read" and (
                    alive is True or (alive is None and not take_over)):
                said = f"{rel} is being read -- {claim_words(k.claim or {})}"
                tell(said)
                out.append({"file": rel, "state": "being read",
                            "said": said})
                continue
            tell(f"reading {rel} ({size_words(k.size)}) -- its DOS per "
                 f"orbital, kept beside it")
            began = time.time()
            try:
                atoms = atoms or point_atoms(entry["folder"])
                got = TBTransNcFileParser.parse(
                    k.path, fresh=fresh,
                    # THE PERSON'S WORD covers the claim they were shown
                    # -- one on another host when the list was taken --
                    # never one made meanwhile (`model/parse.md` § 5e).
                    take_over=(take_over and k.state == "being read"
                               and alive is None),
                    progress=(None if progress is None else
                              (lambda i, n, p=k.path: progress(p, i, n))),
                    atoms=atoms)
            except ReadingClaimed as exc:
                said = str(exc)
                out.append({"file": rel, "state": "being read",
                            "said": said})
            except (OSError, RuntimeError, ValueError, TbtError) as exc:
                said = f"{rel} does not read: {exc}"
                out.append({"file": rel, "state": "failed", "said": said})
            else:
                # KEPT ALREADY -- read by another meanwhile, or beside the
                # claim of a reading that is gone, removed: nothing read.
                said = (f"read {rel} in {span_words(time.time() - began)}"
                        if not got.kept else f"{rel}: kept" + (
                            " -- the claim of a reading that is gone "
                            "removed" if k.state == "being read" else ""))
                out.append({"file": rel, "state": "kept", "said": said})
            tell(said)
    return out
