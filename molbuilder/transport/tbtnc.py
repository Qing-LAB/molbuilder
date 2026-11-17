"""What a transmission point says of its DOS -- its total, its regions', the
leads', the eigenchannels and a selection's PDOS -- for the transport report,
its data file and a PDOS from the files alike (`web/results.md` § 2.5;
`engines/transport.md` § 2a.12): **one sum** (:func:`summed_pdos`) of a
region's atoms and of a selection's.

**Read through the SIESTA family's parser** (`model/parse.md` § 5e): the
averages from TBtrans's k-averaged text files, on the data file's own
energies (`tbtrans_nc.point_averages`); the DOS per orbital from the data
file's kept reading (`tbtrans_nc.kept`), summed here over a region's atoms or
a selection's, narrowed to an orbital type by the device run's own
``<label>.ORB_INDX``; from the files (:func:`pdos_from_files`), the digests a
data file names (`tbtrans_nc.kept_at`), narrowed by its orbital table.  No
file's format is read here: the parser reads them.

USED-BY the transport record (`record`), its data file (`datafile`), its
        sum (`sum`), its export (`export`), its energies (`energies`), the
        transport routes (`web/blueprints/transport.py`), and a person's
        script's PDOS from the files (`postprocess.pdos`, through
        :func:`pdos_from_files`)

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

#: THE WAY TO A POINT'S DOS PER ORBITAL, said wherever one is missing --
#: the report's line, the export's, a PDOS from the files.
READ_WAY = ("read it from the report's DOS tab (Read the DOS per orbital), "
            "or `jobset summarize task`")

#: The words a point without its kept reading is said in.
NOT_READ = f"its DOS per orbital is not read yet -- {READ_WAY}"


class TbtError(Exception):
    """The files cannot answer what was asked -- the message says why."""


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
            f"it again")


def _by_label(readings: Sequence[Tuple[str, Any]],
              regions: Dict[str, Sequence[int]], expected
              ) -> Dict[str, List[float]]:
    """Each region's DOS: its atoms (0-based, the device run's order) summed
    by :func:`summed_pdos` over ``readings`` -- the PDOS of the region, the
    channels summed; a region with no atom in the device has none."""
    return {name: summed_pdos(readings, None, atoms, "all",
                              expected=expected)["pdos"]
            for name, atoms in regions.items()
            if all(covered(r, atoms)[0] for _name, r in readings)}


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
    ``{energy_ev, total, by_label, lead_spectral, lead_bulk, eigenchannels,
    device_atoms, per_orbital}`` -- the averages TBtrans wrote, on the data
    file's energies; ``by_label`` each region's, summed by
    :func:`summed_pdos` from the data file's kept readings, present only
    when every channel's is kept; ``device_atoms`` the atoms a kept reading
    covers, and ``per_orbital`` each kept reading's digest and the data
    file's version it states.  A polarized point's two channels summed,
    and ``by_spin`` each channel's own ``total`` and ``eigenchannels``
    beside.  A quantity
    TBtrans did not write is absent, never zero.  ``regions`` count by
    engine index; ``expected`` is the composed junction's atom index map,
    which each kept reading's must be (F7).  ``av`` is the point's averages
    where the caller read them already (`record.point_averages_of`, one read
    for the transmission and the DOS)."""
    from ..parse.engines.tbtrans_nc import data_files, digest_of, kept
    if av is None:
        from .record import point_averages_of
        try:
            av = point_averages_of(folder, label)
        except (OSError, ValueError) as exc:
            raise TbtError(str(exc)) from exc
    from .conductance import SPIN_CHANNELS
    channels = SPIN_CHANNELS[spin]
    missing = [c for c in channels if c not in av]
    if missing:
        raise TbtError(f"{Path(folder).name} holds no k-averaged file of "
                       f"the {' or '.join(missing)} channel")
    data = data_files(folder, label)
    per: Dict[str, Dict] = {}
    read: List[Tuple[str, Any]] = []
    for c in channels:
        a = av[c]
        d: Dict = {"energy_ev": a["energy_ev"]}
        for src, dst in (("dos", "total"), ("lead_spectral", "lead_spectral"),
                         ("lead_bulk", "lead_bulk"),
                         ("eigenchannels", "eigenchannels")):
            if src in a:
                d[dst] = a[src]
        reading = kept(data[c], need_user=True) if c in data else None
        if reading is not None:
            # EVERY KEPT READING IS THE JUNCTION'S (F7) -- a reading of
            # another structure is refused here, its DOS no point's.
            if expected is not None:
                _checked(reading, expected, data[c].name)
            # THE ATOMS ITS DOS COVERS -- TBtrans's device region, the only
            # atoms it computes a DOS for (`engines/transport.md` § 2a.12,
            # *How each DOS is made*), by engine index.
            d["device_atoms"] = device_atoms(reading)
            # THE READING IT IS, named: its digest and the data file's
            # version it states, which a PDOS from the files reads back
            # (*A PDOS of the atoms you choose, from the files*).
            d["per_orbital"] = [{"digest": str(digest_of(data[c])),
                                 "channel": c,
                                 "version": [int(v) for v in reading.version]}]
        if reading is not None and reading.dos is not None:
            read.append((data[c].name, reading))
        per[c] = d
    # EACH REGION'S DOS, every channel's reading kept: the one sum a PDOS of
    # chosen atoms is (:func:`summed_pdos`), its channels summed there.
    by_label = (_by_label(read, regions, expected)
                if regions and len(read) == len(channels) else None)
    if len(channels) == 1:
        if by_label is not None:
            per[channels[0]]["by_label"] = by_label
        return per[channels[0]]
    up, dn = per["up"], per["down"]
    from .energies import same_grid
    if not same_grid(up["energy_ev"], dn["energy_ev"]):
        raise TbtError(f"{Path(folder).name}: its two spin channels are on "
                       f"different energy grids -- not one run's two spins")
    out: Dict = {"energy_ev": up["energy_ev"]}
    # THE ATOMS ITS DOS COVERS, from either channel's kept reading -- one
    # run's two spins, one device region.
    atoms_covered = up.get("device_atoms") or dn.get("device_atoms")
    if atoms_covered is not None:
        out["device_atoms"] = atoms_covered
    if "per_orbital" in up or "per_orbital" in dn:
        out["per_orbital"] = (list(up.get("per_orbital") or ())
                              + list(dn.get("per_orbital") or ()))
    if "total" in up and "total" in dn:
        out["total"] = _sum(up["total"], dn["total"])
    if by_label is not None:
        out["by_label"] = by_label
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
    atom index map is not ``expected``, the composed junction's (F7).  The
    sum is :func:`summed_pdos`'s, which a PDOS from the files sums with too
    (:func:`pdos_from_files`)."""
    from ..parse.engines.tbtrans_nc import data_files, kept
    from ..runfiles import SIESTA_ORB_INDX
    files = data_files(folder, label)
    if not files:
        raise TbtError(f"{Path(folder).name} holds no data file of {label}")
    readings = []
    for path in files.values():
        reading = kept(path, need_user=True)
        if reading is None:
            raise TbtError(f"{path.name}: {NOT_READ}")
        readings.append((path.name, reading))
    basis = None
    if orbitals != "all":
        if orb_indx is None or not Path(orb_indx).is_file():
            raise TbtError(f"the device run's {SIESTA_ORB_INDX} is not "
                           f"there, so its orbitals' types are not known")
        basis = per_atom(basis_table(Path(orb_indx)))
    return summed_pdos(readings, basis, atoms, orbitals, expected=expected,
                       basis_said=Path(orb_indx).name if orb_indx else "")


def basis_table(orb_indx: Path) -> List[Tuple[int, int, int, int, int]]:
    """THE DEVICE RUN'S BASIS, every orbital in its order as ``(atom, n, l,
    m, zeta)`` -- the atom by engine index -- from its ``<label>.ORB_INDX``
    (sisl's reader, `tbtrans_nc.read_orbitals`): the one read of it, which
    the report's PDOS and the data file's orbital table ask alike."""
    from ..parse.engines.tbtrans_nc import read_orbitals
    try:
        basis = read_orbitals(orb_indx)
    except Exception as exc:                                  # noqa: BLE001
        raise TbtError(f"{orb_indx.name} does not read: {exc}") from exc
    return [(i, int(o.n), int(o.l), int(o.m), int(o.zeta))
            for i in range(len(basis)) for o in basis[i].orbitals]


def per_atom(rows: Sequence[Tuple[int, int, int, int, int]]
             ) -> List[List[Tuple[int, int]]]:
    """A basis table's rows -- ``(atom, n, l, m, zeta)``, the atom by engine
    index, in the run's order -- as each atom's orbitals' ``(l, m)``: what
    a selection's orbital type is matched on (:func:`summed_pdos`), read
    from the ``.ORB_INDX`` or from a data file's orbital table alike."""
    out: List[List[Tuple[int, int]]] = [
        [] for _ in range(1 + max((r[0] for r in rows), default=-1))]
    for atom, _n, l, m, _z in rows:
        out[atom].append((l, m))
    return out


def orbital_word(l: int, m: int) -> str:
    """An orbital's type in the orbital menu's words (:data:`ORBITAL_TYPES`):
    the one naming its ``l`` and ``m`` (``px``), else its ``l`` alone
    (``d``); ``""`` for an ``l`` the menu does not offer."""
    exact = [k for k, (ll, mm) in ORBITAL_TYPES.items()
             if ll == l and mm == m]
    whole = [k for k, (ll, mm) in ORBITAL_TYPES.items()
             if ll == l and mm is None]
    return (exact or whole or [""])[0]


def summed_pdos(readings: Sequence[Tuple[str, Any]],
                basis: Optional[List[List[Tuple[int, int]]]],
                atoms: Sequence[int], orbitals: str = "all", *, expected,
                basis_said: str = "") -> Dict:
    """THE ONE SUM of a selection's PDOS, and so of a region's
    (:func:`_by_label`; `web/results.md` § 2.5;
    `engines/transport.md` § 2a.12, *A PDOS of the atoms you choose, from
    the files*): over ``readings`` -- each a channel's kept reading, named --
    the DOS columns of ``atoms`` (0-based, the device run's order) that the
    reading covers, narrowed to the orbitals of type ``orbitals`` by
    ``basis`` (each atom's ``(l, m)``, :func:`per_atom`; ``None`` for
    ``"all"``), the channels summed.  ``{energy_ev, pdos, atoms,
    outside_device, orbitals}``; :class:`TbtError` names what cannot
    answer."""
    import numpy as np
    asked = sorted({int(a) for a in atoms})
    if not asked:
        raise TbtError("no atoms selected")
    if not readings:
        raise TbtError("no kept reading of the point's data file to sum")
    if orbitals != "all" and orbitals not in ORBITAL_TYPES:
        raise TbtError(f"orbital type {orbitals!r}: one of all, "
                       f"{', '.join(ORBITAL_TYPES)}")
    total = None
    inside: List[int] = []
    outside: List[int] = []
    energy: List[float] = []
    for name, reading in readings:
        _checked(reading, expected, name)
        if reading.dos is None:
            raise TbtError(f"{name} holds no device DOS -- the run was "
                           f"not asked for it (TBT.DOS.Gf)")
        inside, outside = covered(reading, asked)
        if not inside:
            raise TbtError("none of the selected atoms is in the device "
                           "region, where TBtrans computes the DOS")
        if orbitals == "all":
            mask = np.isin(np.asarray(reading.atom), inside)
        else:
            orbs = _orbitals_of(reading.firsto, basis, inside, orbitals,
                                name, basis_said)
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


def device_atoms(reading) -> List[int]:
    """The atoms a reading's DOS covers -- TBtrans's device region, the only
    atoms it computes a DOS for -- by engine index, in order."""
    return sorted({int(a) for a in reading.atom})


def covered(reading, atoms: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Which of ``atoms`` (0-based, the device run's order) a reading's DOS
    covers -- TBtrans's device region -- and which it does not: the one rule
    a region's sum and a selection's both ask."""
    device = set(device_atoms(reading))
    inside = [int(a) for a in atoms if int(a) in device]
    outside = [int(a) for a in atoms if int(a) not in device]
    return inside, outside


def _orbitals_of(firsto: Sequence[int], basis: List[List[Tuple[int, int]]],
                 atoms: Sequence[int], kind: str, name: str,
                 basis_said: str) -> List[int]:
    """The geometry's orbital indices of ``atoms`` of type ``kind`` -- each
    atom's basis, ``(l, m)`` per orbital in the device run's order
    (:func:`per_atom` of the ``.ORB_INDX``'s :func:`basis_table`, or of a
    data file's orbital table, :func:`pdos_from_files`).
    Its orbital counts must be the reading's, atom by atom (its ``firsto``),
    or the two are not of one run and the answer is refused."""
    l_want, m_want = ORBITAL_TYPES[kind]
    firsto = [int(x) for x in firsto]
    counts = [len(orbs) for orbs in basis]
    if counts != [firsto[i + 1] - firsto[i] for i in range(len(firsto) - 1)]:
        raise TbtError(f"{basis_said} and {name} list different orbitals "
                       f"per atom -- not one run's files")
    out: List[int] = []
    for a in atoms:
        for j, (ll, mm) in enumerate(basis[a]):
            if ll == l_want and (m_want is None or mm == m_want):
                out.append(firsto[a] + j)
    return out


def point_readings(calc) -> List[Dict]:
    """EACH FINISHED TRANSMISSION POINT'S READINGS (`model/parse.md` § 5e):
    the transmission's newest run, as the record reads it
    (`record.result_folders`) -- per point whose run finished, its folder
    and what is kept of each data file's reading, its own and those in its
    ``shared/`` (:func:`readings`).  ONE ANSWER TO *this point finished*: a
    run of points' rows, read once for the run (`continuation.read_points`,
    `architecture.md` § 3.2); a transmission run once, its one point, by
    the run door's state of its run (`runs.state_of`).  A point still
    running, or stopped, is none: its data file is TBtrans's while it
    writes, and a partial one after; a run not launched -- a copy cut short
    before its launch record -- holds no point finished."""
    from ..calcdirs import RunPlace, where
    from ..jobset.continuation import read_points as points_reading
    from ..runs import run_of, state_of
    from .record import result_folders
    folders = result_folders(calc, "transmission")
    finished = set()
    for run in {r for _, w, r in folders if w is not None}:
        if not where(run).of_points:
            # RUN ONCE: its one point is the run, by the run door.
            ran = run_of(run)
            if ran is not None and state_of(ran).state == "finished":
                finished.add(Path(run))
            continue
        # ITS ONE READING (`continuation.read_points`): a run not launched,
        # or one whose records do not read, holds no point finished -- the
        # stage's row says why.
        got = points_reading(calc, "transmission", run)
        if got.launch is None:
            continue
        place = RunPlace.of(run)
        finished |= {place.point(r["rel"]) for r in got.rows
                     if r["state"] == "finished"}
    return [{"point": pt, "folder": folder,
             "readings": readings(folder, calc.task.label)}
            for pt, folder, _run in folders
            if folder is not None and Path(folder) in finished]


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


def pdos_from_files(data_file, point, *, atoms=(), labels=(), regions=(),
                    orbitals="all") -> Dict[str, Any]:
    """THE PDOS OF A MIXED SELECTION FROM THE FILES -- ``postprocess.pdos``'s
    question, answered here (its words there; `engines/transport.md`
    § 2a.12, *A PDOS of the atoms you choose, from the files*): the point
    ``point`` of the data file ``data_file``, the union of ``atoms`` (user
    indices), every atom carrying one of ``labels`` and every atom of
    ``regions``, narrowed to ``orbitals``; each digest the file names read as
    the reading the file was composed from (`tbtrans_nc.kept_at`) and summed
    by the report's one sum (:func:`summed_pdos`), a point that shares
    energies joined as the report joins it.  :class:`TbtError` names what
    cannot answer."""
    from ..atom_index import atom_index_map
    from ..atom_permutation import AtomIndexError, checked_indices
    from ..parse.engines.tbtrans_nc import kept_at
    from ..structure import Structure
    from .datafile import read_data_file
    from .energies import joined, restricted
    path = Path(data_file)
    here = path.parent
    groups, amap = read_data_file(path)["groups"], atom_index_map(path)
    # A POINT IS A GROUP THAT SAYS WHETHER IT IS DONE -- not a frame's group
    # of points, nor a point's device run.
    held = sorted(k[len("points/"):] for k in groups
                  if k.startswith("points/")
                  and "done" in groups[k]["variables"])
    if point not in held:
        raise TbtError(f"{point} is not one of {path.name}'s points: "
                       f"{', '.join(held)}")
    J = {k: v["data"] for k, v in groups["structure"]["variables"].items()}
    P = {k: v["data"] for k, v in
         groups[f"points/{point}"]["variables"].items()}
    if int(P["done"]) != 1:
        why = str(P["why"]) or "no transmission yet"
        raise TbtError(f"{point} is not done: {why}")
    n = len(J["atom"])
    # THE SELECTION, a union -- each part naming at least one atom; an
    # index through the map's one bounds check (`checked_indices`).
    try:
        chosen = set(checked_indices(atoms, n, "user index"))
    except AtomIndexError as exc:
        raise TbtError(str(exc)) from exc
    # THE JUNCTION'S LABELS, the structure's own (`structure_json`, read
    # through its door): matched as written, never read for meaning.
    carried = Structure.from_dict(
        groups["structure"]["attrs"]["structure_json"]).regions or {}
    for label in labels:
        members = [int(a) for a in carried.get(label, ())]
        if not members:
            raise TbtError(f"no atom of the junction carries the label "
                           f"{label!r}")
        chosen.update(members)
    owned = sorted({str(r) for r in J["atom_region"] if str(r)})
    for region in regions:
        if region not in owned:
            raise TbtError(f"region {region!r}: one of {', '.join(owned)}")
        chosen.update(a for a in range(n) if str(J["atom_region"][a]) == region)
    if not chosen:
        raise TbtError("no atoms selected")
    # THE DIGESTS THE FILE NAMES -- only a point every reading of which was
    # kept and named when it was written (`datafile`): none named, its why.
    sources = P["dos_per_orbital_files"]
    if not sources:
        why = (str(P["dos_why"]) or str(P["dos_per_orbital_reading"])
               or "not read")
        raise TbtError(f"{point}: {path.name} names no reading of its DOS "
                       f"per orbital ({why}) -- {READ_WAY}, then summarize "
                       f"or export again")
    if orbitals != "all" and not len(J["orbital"]):
        why = groups["structure"]["attrs"].get("orbitals_why", "")
        raise TbtError(f"{path.name} holds no orbital table ({why}), so "
                       f"its orbitals' types are not known")
    # THE FILE'S ATOMS AS ITS DIGESTS LIST THEM, by engine index (F7), and
    # its orbital table as each atom's (l, m) -- the page's own shape.
    expected = amap.as_listed_by_engine()
    engine = dict(zip(range(n), amap.map_to_engine_index(list(range(n)))))
    basis = None if orbitals == "all" else per_atom(
        [(engine[int(J["orbital_atom"][o])], int(J["orbital_n"][o]),
          int(J["orbital_l"][o]), int(J["orbital_m"][o]),
          int(J["orbital_zeta"][o])) for o in range(len(J["orbital"]))])

    def read(folder):
        mine = [s for s in sources if (here / s["digest"]).parent == folder]
        readings = []
        for s in mine:
            got = kept_at(here / s["digest"], s["version"], need_user=True)
            if got is None:
                raise TbtError(
                    f"{s['digest']} is not the reading {path.name} was "
                    f"composed from -- not there, read again from another "
                    f"run of its data file or by another molbuilder, "
                    f"damaged, or stating no atom's user index: summarize or "
                    f"export again")
            readings.append((Path(s["digest"]).name, got))
        return summed_pdos(readings, basis,
                           [engine[a] for a in sorted(chosen)], orbitals,
                           expected=expected,
                           basis_said=f"{path.name}'s orbital table")
    own = here / str(P["folder"])
    # JOINED AS THE REPORT JOINS IT, on the point's own grid -- the file's.
    got = restricted(joined(own, read, window_of=lambda _f: None),
                     [float(e) for e in P["energy_ev"]])
    return {"point": point, "energy_ev": got["energy_ev"],
            "pdos": got["pdos"], "units": "1/eV",
            "selection": {"atoms": sorted({int(a) for a in atoms}),
                          "labels": list(labels), "regions": list(regions)},
            "atoms": sorted(amap.map_to_user_index(got["atoms"])),
            "outside_device": sorted(
                amap.map_to_user_index(got["outside_device"])),
            "orbitals": orbitals,
            "digests": [s["digest"] for s in sources],
            "said": ("the DOS per orbital of the digests named, summed over "
                     "the orbitals of the atoms selected -- their union -- "
                     "that the point's DOS covers; a polarized point's two "
                     "channels summed.  The selection, and so the sum, is "
                     "the person's.")}
