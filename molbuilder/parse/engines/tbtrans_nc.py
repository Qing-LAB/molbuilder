"""TBtrans's data file -- ``<label>.TBT.nc`` -> :class:`TransmissionResult`,
its reading kept beside it (`model/parse.md` § 5e).

**What the file holds, and what is read.**  Every quantity per k-point,
energy and orbital: on a 3000-orbital junction 1.6 GB a point, each table
stored as one small piece per k-point and energy.  What no k-averaged text
file TBtrans writes holds is the device's DOS **per orbital** -- a region's
DOS, a selection's PDOS -- so that is what this parser reads: averaged over
the k-points with TBtrans's own weights, one k-point at a time (one pass
over the table, its progress told), each column a device orbital as sisl
orders them (``o_dev``, sorted), with each column's atom.  sisl reads the
file (its TBtrans reader); nothing here parses NetCDF by hand.

**Its energies ascending.**  TBtrans keeps a data file's energies contour by
contour, in the order its deck lists them, and prints every k-averaged text
file sorted by energy (`m_tbt_save.F90` `state_cdf2ascii`, its pivot); a
point that shares energies computes interleaved contours, so the two orders
differ.  The reading holds them ascending -- its DOS rows with them -- and
so does :func:`energies`: one order for every reader.

**Its reading is kept, per version of the file, beside it** (the
catalogue's ``.{tbt}.digest.nc``, :func:`digest_of`): § 2b reads a run's
output once per version -- its size and modification time -- in memory; a
data file's reading costs minutes on a cluster's file system, so it is kept
on disk, this module its one writer and its one reader, written through the
variable record every NetCDF file molbuilder writes shares (`ncdata`).  A
digest of another version is out of date and read by nothing.

**One reading of a file at a time** (:func:`_claimed`): a reading claims the
file by creating its claim (``.{tbt}.digest.nc.reading``) exclusively,
through the one writer (`persist.write_bytes`), stating who reads -- the
host, the process and when the kernel started it -- and how far; another
asker finds the claim and does not read.  A claim on this host whose
process is gone -- or is another process now, or is this one, which reads
no such file -- is stale and removed by the next asker; one on another host
stands until the person takes it over.

**One NetCDF access at a time in the process** (`ncdata.NC_LOCK`): every
open, read and write here holds it -- the data file one k-point at a time
-- so a reading in one request thread lets the page's other requests in.

Not stdlib (sisl, netCDF4, numpy): it does not travel beside a job, and
nothing beside a job reads a data file.
"""
from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import ParseError
from molbuilder.parse.types import ParseResult, TransmissionResult
from . import siesta_grammar as _G
from .tbtrans import PRINTED_HALF_DECIMAL_EV, averaged_files, read_averaged

#: The digest's schema, stated in it.
DIGEST_SCHEMA = "molbuilder/tbt-digest@4"
#: The claims THIS PROCESS holds -- the readings its threads run, each by
#: its claim's path and when it began: a claim naming this process and
#: absent here is no reading of ours (a copy, a restored state).
_READING_HERE: "set[Tuple[str, float]]" = set()
#: netCDF-4 is HDF5.
_MAGIC = b"\x89HDF"


class ReadingClaimed(ParseError):
    """Another reading holds the data file's claim -- ``claim`` says whose,
    since when and how far (`model/parse.md` § 5e)."""

    def __init__(self, path, claim: Dict[str, Any]):
        self.path = Path(path)
        self.claim = claim
        super().__init__(f"{self.path.name} is being read -- "
                         f"{claim_words(claim)}")


@dataclass(frozen=True)
class Kept:
    """A data file's reading, as kept (:func:`kept_state`): ``state`` --
    ``kept``, ``missing``, ``out of date`` (a digest that is not of the file
    as it is) or ``being read`` (``claim``, with ``alive``: ``True`` or
    ``False`` for a reader on this host, ``None`` for one on another) -- and
    the data file's size in bytes."""
    path: Path
    state: str
    size: int
    claim: Optional[Dict[str, Any]] = None

    def to_wire(self, base=None) -> Dict[str, Any]:
        """The state on the wire, with the words every surface says it in
        -- its size, and what is kept or who is reading; ``file`` the data
        file's path from ``base`` (a calculation's folder), its name
        without one."""
        from molbuilder.units import size_words
        said = {"kept": "kept",
                "missing": "not read yet",
                "out of date": "out of date -- not a reading of its data "
                               "file as it is now"}.get(self.state)
        if self.state == "being read":
            said = "being read " + claim_words(self.claim or {})
            if (self.claim or {}).get("alive") is False:
                said += " -- that process is gone; a reading may begin"
        return {"file": (str(self.path.relative_to(base)) if base is not None
                         else self.path.name),
                "state": self.state, "size": self.size,
                "size_words": size_words(self.size), "claim": self.claim,
                "said": said}


def channel_of(path) -> Optional[str]:
    """The channel a data file's name says -- ``unpolarized``, ``up``,
    ``down`` -- or ``None`` for a name that is not one's."""
    name = Path(path).name
    for channel, tag in _G.TBT_CHANNELS:
        if name.endswith(_G.tbt_nc(tag)):
            return channel
    return None


def _label_of(path: Path) -> str:
    """The label a data file is named on: its name before its channel's
    role."""
    tag = dict(_G.TBT_CHANNELS)[channel_of(path)]
    return path.name[:-len(_G.tbt_nc(tag))]


def _beside(path, role: str) -> Path:
    """The file of catalogue ``role`` beside data file ``path``, on its
    label and its channel's tag (`runfiles.compose`)."""
    from molbuilder.runfiles import compose
    path = Path(path)
    tag = dict(_G.TBT_CHANNELS)[channel_of(path)].strip(".")
    return path.with_name(compose(_label_of(path), role, tbt=tag))


def digest_of(path) -> Path:
    """The digest beside data file ``path``: ``<label>.TBT.nc`` ->
    ``<label>.TBT.digest.nc``, a channel's tag kept."""
    from molbuilder.runfiles import TBT_DIGEST
    return _beside(path, TBT_DIGEST)


def data_files(directory, label: str) -> Dict[str, Path]:
    """The data files in ``directory``, by channel -- ``{"unpolarized":
    …}`` or ``{"up": …, "down": …}`` -- each that is there, found on the
    label through the catalogue (`runfiles.find_on_label`)."""
    from molbuilder.runfiles import find_on_label
    out: Dict[str, Path] = {}
    for channel, tag in _G.TBT_CHANNELS:
        hits = find_on_label(directory, label, _G.tbt_nc(tag))
        if hits:
            out[channel] = hits[-1]
    return out


def _version(path) -> Tuple[int, int]:
    st = Path(path).stat()
    return (st.st_size, st.st_mtime_ns)


# ---- the claim ------------------------------------------------------------ #

def _claim_path(path) -> Path:
    from molbuilder.runfiles import TBT_DIGEST_CLAIM
    return _beside(path, TBT_DIGEST_CLAIM)


def _read_claim(cp: Path) -> Optional[Dict[str, Any]]:
    """The claim at ``cp`` -- ``{host, pid, started, started_at,
    updated_at, k, n}``, and ``alive`` -- or ``None`` when there is none.
    One that does not read is a claim nonetheless, its fields unknown."""
    try:
        text = cp.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        text = ""
    try:
        claim = json.loads(text)
        if not isinstance(claim, dict):
            raise ValueError
    except ValueError:
        claim = {}
    claim["alive"] = _alive(claim, cp)
    return claim


def _alive(claim: Dict[str, Any], cp: Path) -> Optional[bool]:
    """Whether a claim's reader is still there -- asked only on its own
    host, ``None`` elsewhere, where it cannot be: its process, by its id and
    when the kernel started it (`joblist.started_of`), so a later process
    given the same id is not it; and where it names this process, one of
    its own readings (:data:`_READING_HERE`)."""
    if claim.get("host") != socket.gethostname():
        return None
    from molbuilder.joblist import started_of
    try:
        pid = int(claim["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    now = started_of(pid)
    if now is None or now != claim.get("started"):
        return False
    if pid == os.getpid():
        return (str(cp), claim.get("started_at")) in _READING_HERE
    return True


def claim_words(claim: Dict[str, Any]) -> str:
    """A claim in words: who reads, since when, how far."""
    if not claim.get("host"):
        return "a claim that does not say whose"
    since = time.strftime("%Y-%m-%d %H:%M:%S",
                          time.localtime(claim.get("started_at") or 0))
    far = (f", k-point {claim['k']} of {claim['n']}" if claim.get("n")
           else ", not begun")
    moved = time.strftime("%H:%M:%S",
                          time.localtime(claim.get("updated_at") or 0))
    return (f"by {claim['host']}, process {claim.get('pid')}, since {since}"
            f"{far}, last moved at {moved}")


def _clear_claim(cp: Path, *, take_over: bool) -> None:
    """Another reading's claim removed where the asker may: one whose reader
    on this host is gone (stale), and any on the person's word
    (``take_over``)."""
    held = _read_claim(cp)
    if held is not None and (take_over or held.get("alive") is False):
        cp.unlink(missing_ok=True)


@contextmanager
def _claimed(path, *, take_over: bool) -> Iterator[Callable[[int, int],
                                                            None]]:
    """THE DATA FILE CLAIMED for one reading (`model/parse.md` § 5e): its
    claim created exclusively, stating this host, this process, when and
    how far -- the updater yielded rewrites how far; the claim removed when
    the reading ends, if it is still this one's.  Held by another:
    :class:`ReadingClaimed` -- but a stale one is removed first, and
    ``take_over`` removes any (:func:`_clear_claim`)."""
    cp = _claim_path(path)
    _clear_claim(cp, take_over=take_over)
    from molbuilder.joblist import started_of
    from molbuilder.persist import write_bytes
    mine = {"host": socket.gethostname(), "pid": os.getpid(),
            "started": started_of(os.getpid()),
            "started_at": time.time(), "updated_at": time.time(),
            "k": 0, "n": 0}
    # OURS BEFORE IT IS THERE, so no thread of this process takes it for a
    # copy in between.
    token = (str(cp), mine["started_at"])
    _READING_HERE.add(token)
    try:
        # THE ONE WRITER, exclusive: the name created whole, or refused.
        write_bytes(cp, json.dumps(mine).encode("utf-8"), exclusive=True)
    except FileExistsError:
        _READING_HERE.discard(token)
        raise ReadingClaimed(path, _read_claim(cp) or {}) from None

    def ours(claim: Optional[Dict[str, Any]]) -> bool:
        return bool(claim) and claim.get("host") == mine["host"] \
            and claim.get("pid") == mine["pid"] \
            and claim.get("started_at") == mine["started_at"]

    def update(k: int, n: int) -> None:
        now = _read_claim(cp)
        if not ours(now):
            # TAKEN OVER by another reading: this one stops.
            raise ReadingClaimed(path, now or {})
        mine.update(k=k, n=n, updated_at=time.time())
        write_bytes(cp, json.dumps(mine).encode("utf-8"))
    try:
        yield update
    finally:
        if ours(_read_claim(cp)):
            cp.unlink(missing_ok=True)
        _READING_HERE.discard(token)


# ---- the state, and the reading ------------------------------------------- #

def kept_state(path, *, need_user: bool = False) -> Kept:
    """What is kept of data file ``path``'s reading, without reading it or
    the digest's tables (`model/parse.md` § 5e) -- ``need_user``: a run of
    ours holds the file, so a digest stating no user index is out of date
    (`model/overview.md` § 2.2b)."""
    path = Path(path)
    size = path.stat().st_size
    claim = _read_claim(_claim_path(path))
    if claim is not None:
        return Kept(path, "being read", size, claim)
    if not digest_of(path).is_file():
        return Kept(path, "missing", size)
    return Kept(path, "kept" if _digest(path, part=(), need_user=need_user)
                is not None else "out of date", size)


def kept(path, *, need_user: bool = False) -> Optional[TransmissionResult]:
    """Data file ``path``'s kept reading when it is of the file as it is --
    never a read of the data file; ``None`` otherwise (``need_user`` as
    :func:`kept_state`)."""
    return _from_digest(Path(path), need_user=need_user)


#: What a kept reading is read back from: the digest's variables -- those
#: every reading holds, then the rest (a data file with no device DOS has
#: no columns).
_HELD = ("energy_ev", "firsto")
_READING = _HELD + ("DOS", "device_orbital", "device_orbital_atom")


def _digest(path: Path, part: Tuple[str, ...], *, need_user: bool = False
            ) -> Optional[Dict[str, Any]]:
    """The digest beside data file ``path`` when it is a reading of the file
    as it is -- its schema and the version it states -- with the variables
    ``part`` names that it holds and its atom index map; ``None`` otherwise:
    none there, another version's, one stating no user index where
    ``need_user``, or one that does not read (read again, and written
    anew)."""
    import numpy as np
    from molbuilder.atom_permutation import FIELD_USER
    from molbuilder.ncdata import read_atom_index_map, reading
    digest = digest_of(path)
    if not digest.is_file():
        return None
    version = _version(path)
    try:
        with reading(digest) as ds:
            if (getattr(ds, "schema", "") != DIGEST_SCHEMA
                    or (int(ds.data_file_size), int(ds.data_file_mtime_ns))
                    != version
                    or not all(v in ds.variables for v in _HELD)
                    or (need_user and FIELD_USER not in ds.variables)):
                return None
            ds.set_auto_mask(False)
            return {"channel": str(ds.channel), "version": version,
                    "atom_index_map": read_atom_index_map(
                        ds, source=digest.name),
                    **{v: np.asarray(ds[v][...]) for v in part
                       if v in ds.variables}}
    except (OSError, RuntimeError, KeyError, AttributeError, ValueError):
        return None


def _from_digest(path: Path, *, need_user: bool = False
                 ) -> Optional[TransmissionResult]:
    got = _digest(path, part=_READING, need_user=need_user)
    if got is None:
        return None
    return TransmissionResult(
        **ParseResult.envelope(TBTransNcFileParser.name, path),
        channel=got["channel"],
        energy_ev=tuple(float(e) for e in got["energy_ev"]),
        dos=got["DOS"].astype(float) if "DOS" in got else None,
        orbital=tuple(int(o) for o in got.get("device_orbital", ())),
        atom=tuple(int(a) for a in got.get("device_orbital_atom", ())),
        firsto=tuple(int(o) for o in got["firsto"]),
        version=got["version"], kept=True,
        atom_index_map=got["atom_index_map"])


def _read(path: Path, progress: Callable[[int, int], None]
          ) -> Tuple[TransmissionResult, Dict[str, Any]]:
    """The data file read: its energies, ascending, and the device DOS
    averaged over the k-points per device orbital, one k-point at a time --
    and, for the digest to describe itself (:func:`_write_digest`), what
    else the file states and TBtrans's k-averaged text files beside it."""
    import numpy as np
    from molbuilder.ncdata import reading
    version = _version(path)
    # WHAT THE FILE STATES OF ITSELF -- one access, the plain reader's.
    with reading(path) as ds:
        has_dos = "DOS" in ds.variables
        stated = {k: str(ds.getncattr(k)) for k in ds.ncattrs()}
    # ITS GEOMETRY AND ELECTRODES -- one access, sisl's.
    with reading(path, sile=True) as tbt:
        raw = np.asarray(tbt.E, dtype=float)
        geometry = tbt.geometry
        wk = [float(w) for w in tbt.wk]
        o_dev = np.asarray(tbt.o_dev)
        facts = {"device": [int(a) for a in tbt.a_dev],
                 "kpt": np.asarray(tbt.kpt), "wkpt": wk,
                 "electrodes": [str(e) for e in tbt.elecs]}
        facts["mu_ev"] = [float(tbt.chemical_potential(e))
                          for e in facts["electrodes"]]
        facts["kT_ev"] = [float(tbt.kT(e)) for e in facts["electrodes"]]
        facts["eta_ev"] = float(tbt.eta())
    order = np.argsort(raw, kind="stable")
    # TBTRANS'S OWN AVERAGES BESIDE IT, on its energies -- before the loop:
    # files that are not one run's are refused before minutes of reading,
    # not after.
    elecs = facts["electrodes"]
    averages = (point_averages(
        path.parent, _label_of(path), pair="-".join(elecs[:2]),
        leads=tuple(elecs)).get(channel_of(path), {})
        if len(elecs) >= 2 else {})
    dos = None
    if has_dos:
        acc = np.zeros((len(raw), len(o_dev)))
        progress(0, len(wk))
        for ik, w in enumerate(wk):
            # ONE K-POINT AN ACCESS (`ncdata.reading`): the page's other
            # requests between, and no handle held across them -- the
            # file's version the one the reading began with, or the
            # reading is refused rather than mixed.
            if _version(path) != version:
                raise ValueError(f"{path.name} changed during its reading "
                                 f"(at k-point {ik + 1} of {len(wk)}): "
                                 f"read it again")
            with reading(path, sile=True) as tbt:
                acc += np.asarray(tbt.DOS(kavg=ik, sum=False),
                                  dtype=float) * w
            progress(ik + 1, len(wk))
        dos = acc[order]
    result = TransmissionResult(
        **ParseResult.envelope(TBTransNcFileParser.name, path),
        channel=channel_of(path),
        energy_ev=tuple(float(e) for e in raw[order]), dos=dos,
        orbital=tuple(int(o) for o in o_dev) if has_dos else (),
        atom=(tuple(int(a) for a in geometry.o2a(o_dev)) if has_dos
              else ()),
        firsto=tuple(int(o) for o in geometry.firsto),
        version=version, kept=False)
    meta: Dict[str, Any] = {
        "stated": stated, **facts,
        "xyz": np.asarray(geometry.xyz), "cell": np.asarray(geometry.cell),
        "elements": _elements_of(path, geometry), "averages": averages}
    return result, meta


def _elements_of(path: Path, geometry) -> Optional[List[str]]:
    """Each atom's element -- which the data file does not state -- from
    the Hamiltonian TBtrans read beside it, ``<label>.TS.HSX``, when it is
    there and is the data file's geometry, orbital for orbital; ``None``
    otherwise (a sharing point's ``shared/`` holds none)."""
    import numpy as np
    import sisl
    from molbuilder.ncdata import NC_LOCK
    from molbuilder.runfiles import compose
    hsx = path.with_name(compose(_label_of(path), _G.TS_HSX))
    if not hsx.is_file():
        return None
    try:
        with NC_LOCK:
            g = sisl.get_sile(str(hsx)).read_geometry()
    except Exception:                                         # noqa: BLE001
        return None
    if (g.na != geometry.na
            or not np.array_equal(g.firsto, geometry.firsto)
            or not np.allclose(g.xyz, geometry.xyz, atol=1e-4)):
        return None
    return [str(a.symbol) for a in g.atoms]


def _digest_variables(path: Path, result: TransmissionResult,
                      meta: Dict[str, Any], atoms):
    """THE DIGEST'S VARIABLES, each stating itself (`ncdata.Var`): the
    reading, what the data file states of its geometry and electrodes, and
    TBtrans's own k-averages from their text files -- ``(dims, variables,
    values)``; the atom axis as ``atoms``, its atom index map
    (`model/overview.md` § 2.2b)."""
    import numpy as np
    from molbuilder.ncdata import Var
    from molbuilder.runfiles import compose
    data = path.name
    av = meta["averages"]
    files = av.get("files", {})
    hsx = compose(_label_of(path), _G.TS_HSX)
    n_atoms = len(result.firsto) - 1
    if atoms.n_atoms != n_atoms:
        raise ValueError(f"{data}: its geometry holds {n_atoms} atoms; the "
                         f"map of its run states {atoms.n_atoms}")
    # EVERY AXIS A ROW OF `ncdata.AXES`, its coordinate beside it
    # (`model/datafiles.md` § 2a).
    dims = {"energy_ev": len(result.energy_ev),
            "device_orbital": len(result.orbital), "atom": atoms,
            "atom_bound": len(result.firsto), "xyz": 3, "cell_vector": 3,
            "reciprocal_vector": 3, "kpoint": len(meta["wkpt"]),
            "lead": len(meta["electrodes"]),
            "device_atom": len(meta["device"])}
    rows = [
        (Var("energy_ev", ("energy_ev",), "coordinate", "eV", "energy",
             "relative to the leads' Fermi level (TBtrans's E - E_F), at the "
             "data file's full precision, ascending", source=data),
         np.asarray(result.energy_ev, dtype=float)),
        (Var("device_orbital", ("device_orbital",), "coordinate", "1",
             "the device orbital", "its index in the geometry, from 0 -- "
             "one DOS column each, in sisl's sorted order", dtype="i4"),
         np.asarray(result.orbital, dtype="i4")),
        (Var("device_orbital_atom", ("device_orbital",), "coordinate", "1",
             "each device orbital's atom", "its engine index, from 0",
             dtype="i4", auxiliary=True, atom_index=True),
         np.asarray(result.atom, dtype="i4")),
        (Var("atom", ("atom",), "coordinate", "1", "atom index",
             "the atom's engine index, from 0 -- the digest is listed by "
             "engine index (atoms_listed_by); user_atom_index beside it "
             "(model/overview.md 2.2b)", dtype="i4"),
         np.arange(n_atoms, dtype="i4")),
        (Var("atom_bound", ("atom_bound",), "coordinate", "1",
             "an atom's orbital bound", "from 0 to the atom count: bound i "
             "is atom i's first orbital, the last one past the last "
             "orbital", dtype="i4"),
         np.arange(n_atoms + 1, dtype="i4")),
        (Var("firsto", ("atom_bound",), "raw", "1",
             "the geometry's first orbital of each atom",
             "0-based, and one past the last orbital", source=data,
             dtype="i4"),
         np.asarray(result.firsto, dtype="i4")),
        (Var("xyz", ("xyz",), "coordinate", "", "Cartesian component",
             "x, y, z", dtype=str),
         np.asarray(["x", "y", "z"], dtype=object)),
        (Var("cell_vector", ("cell_vector",), "coordinate", "",
             "lattice vector", "a, b, c", dtype=str),
         np.asarray(["a", "b", "c"], dtype=object)),
        (Var("reciprocal_vector", ("reciprocal_vector",), "coordinate", "",
             "reciprocal lattice vector", "b1, b2, b3", dtype=str),
         np.asarray(["b1", "b2", "b3"], dtype=object)),
        (Var("position", ("atom", "xyz"), "raw", "angstrom",
             "each atom's position", "x, y, z", source=data),
         np.asarray(meta["xyz"], dtype=float)),
        (Var("cell", ("cell_vector", "xyz"), "raw", "angstrom",
             "the cell's three vectors", "one a row", source=data),
         np.asarray(meta["cell"], dtype=float)),
        (Var("device_atom", ("device_atom",), "coordinate", "1",
             "the atoms of the device region", "their engine indices, "
             "0-based in the geometry", source=data, dtype="i4",
             atom_index=True),
         np.asarray(meta["device"], dtype="i4")),
        (Var("kpoint", ("kpoint",), "coordinate", "1", "k-point index",
             "from 0, in the data file's order", dtype="i4"),
         np.arange(len(meta["wkpt"]), dtype="i4")),
        (Var("kpt", ("kpoint", "reciprocal_vector"), "raw", "1",
             "the k-points averaged over", "fractions of the cell's "
             "reciprocal vectors b1, b2, b3", source=data),
         np.asarray(meta["kpt"], dtype=float)),
        (Var("wkpt", ("kpoint",), "raw", "1", "each k-point's weight",
             "the data file's own; they sum to 1", source=data),
         np.asarray(meta["wkpt"], dtype=float)),
        (Var("lead", ("lead",), "coordinate", "",
             "each lead's name", "as its TS.Elecs block names it",
             source=data, dtype=str),
         np.asarray(meta["electrodes"], dtype=object)),
        (Var("lead_mu", ("lead",), "raw", "eV",
             "each lead's chemical potential", "on the energy axis's "
             "scale, as the data file states it", source=data),
         np.asarray(meta["mu_ev"], dtype=float)),
        (Var("lead_kT", ("lead",), "raw", "eV",
             "each lead's electronic temperature",
             "k_B T of the lead's Fermi distribution, as the data file "
             "states it", source=data),
         np.asarray(meta["kT_ev"], dtype=float)),
        (Var("device_eta", (), "raw", "eV",
             "the device region's eta",
             "the imaginary part added to the energy of the device's "
             "Green's function, as the data file states it", source=data),
         np.asarray(meta["eta_ev"], dtype=float)),
    ]
    if meta["elements"]:
        rows.append((Var("element", ("atom",), "coordinate", "",
                         "each atom's element", "not stated by the data "
                         "file: read from the Hamiltonian TBtrans read, of "
                         "this geometry", source=hsx, dtype=str,
                         auxiliary=True),
                     np.asarray(meta["elements"], dtype=object)))
    if result.dos is not None:
        rows.append((Var("DOS", ("energy_ev", "device_orbital"), "derived",
                         "1/eV",
                         "the device's DOS per device orbital",
                         "averaged over the k-points: each k-point's DOS "
                         "times its weight (wkpt), summed in double "
                         "precision, kept in TBtrans's single", source=data,
                         dtype="f4"),
                     np.asarray(result.dos, dtype="f4")))
    # TBTRANS'S OWN K-AVERAGES, on the energies above, each from its text
    # file (`tbtrans.read_averaged`, § 5e).
    for key, name, what in (
            ("transmission", "transmission", "the k-averaged transmission"),
            ("dos", "dos_total", "the device's k-averaged DOS")):
        if key in av:
            rows.append((Var(name, ("energy_ev",), "raw",
                             "1/eV" if key == "dos" else "1", what,
                             "TBtrans's own average, as it printed it",
                             source=files[key]),
                         np.asarray(av[key], dtype=float)))
    if "eigenchannels" in av:
        dims["eigenchannel"] = len(av["eigenchannels"])
        rows.append((Var("eigenchannel", ("eigenchannel",), "coordinate",
                         "1", "eigenchannel", "counted from 0, as TBtrans "
                         "orders them", dtype="i4"),
                     np.arange(len(av["eigenchannels"]), dtype="i4")))
        rows.append((Var("eigenchannel_transmission",
                         ("eigenchannel", "energy_ev"), "raw", "1",
                         "each eigenchannel's k-averaged transmission",
                         "TBtrans's own average, as it printed it",
                         source=files["eigenchannels"]),
                     np.asarray(av["eigenchannels"], dtype=float)))
    for key, what in (("lead_spectral", "spectral DOS"),
                      ("lead_bulk", "bulk DOS")):
        for lead, values in av.get(key, {}).items():
            rows.append((Var(f"{key}_dos_{lead}", ("energy_ev",), "raw",
                             "1/eV",
                             f"the {lead} lead's k-averaged {what}",
                             "TBtrans's own average, as it printed it",
                             source=files[key][lead]),
                         np.asarray(values, dtype=float)))
    return dims, [v for v, _ in rows], {v.name: x for v, x in rows}


def _write_digest(path: Path, result: TransmissionResult,
                  meta: Dict[str, Any], atoms) -> None:
    """The reading kept beside the data file -- self-describing, so a
    person's own script may read it alone (`model/parse.md` § 5e): the file
    stating itself, its source the files its variables were read from
    (`model/datafiles.md` § 2b), every variable stating itself, its atoms'
    index map among them, through the variable record's one writing loop
    (`ncdata`), handed to the one writer whole (`persist.write_bytes`)."""
    from molbuilder import __version__
    from molbuilder.ncdata import (NC_LOCK, as_bytes, in_memory, state,
                                   write_group)
    from molbuilder.persist import write_bytes
    dims, variables, values = _digest_variables(path, result, meta, atoms)
    read_from = dict.fromkeys(v.source for v in variables if v.source)
    attrs = {
        "data_file": path.name,
        # As text: a size or a time past 2**31 is no NetCDF int.
        "data_file_size": str(result.version[0]),
        "data_file_mtime_ns": str(result.version[1]),
        # WHAT THE DATA FILE STATES OF ITSELF (its program, its date).
        **{f"data_file_{k}": v for k, v in meta["stated"].items()},
        "channel": result.channel,
        "spin": ("TBtrans's T and DOS carry no spin factor: a non-polarized "
                 "run's (channel unpolarized) are each spin's, up = down; a "
                 "polarized run's this file's channel alone"),
        "energies_held": ("the energies this data file holds -- its run's "
                          "own; a molbuilder point that shares energies "
                          "with a finished one holds the others in its "
                          "shared/ folder, beside their own digest"),
        "orbital_types": ("not stated by the data file: the device run's "
                          "<label>.ORB_INDX names each orbital's l and m, "
                          "in the same order"),
    }
    with NC_LOCK:
        ds = in_memory(digest_of(path).name)
        state(ds, title=("a TBtrans data file, read: the device's DOS per "
                         "orbital averaged over its k-points, with what the "
                         "file states of its geometry and its electrodes, "
                         "and TBtrans's own k-averaged quantities"),
              schema=DIGEST_SCHEMA, molbuilder_version=__version__,
              source="read from " + ", ".join(read_from))
        write_group(ds, dims, variables, values)
        ds.setncatts(attrs)
        data = as_bytes(ds)
    write_bytes(digest_of(path), data)


class TBTransNcFileParser(FileParser):
    """``<label>.TBT.nc`` (``_UP`` / ``_DN``) -> :class:`TransmissionResult`
    -- the kept reading when it is of the file as it is, else the file read
    once and its reading kept (`model/parse.md` § 5e)."""

    name = "tbtrans-nc"
    label = "TBtrans data file (.TBT.nc)"
    hint = "a transmission run's <label>.TBT.nc, written by tbtrans"
    output = TransmissionResult
    keeps_atom_index_map = True

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        path = Path(path)
        if channel_of(path) is None or not path.is_file():
            return False
        try:
            with path.open("rb") as fh:
                return fh.read(4) == _MAGIC
        except OSError:
            return False

    @classmethod
    def parse(cls, path: Path, *, fresh: bool = False,
              take_over: bool = False,
              progress: Optional[Callable[[int, int], None]] = None,
              atoms=None) -> TransmissionResult:
        """The reading of ``path``: kept, unless ``fresh`` or ``take_over``
        -- a stale claim beside it removed (:func:`_clear_claim`) -- else
        read: the file claimed (:class:`ReadingClaimed` while another reads
        it; ``take_over`` removes another's claim and reads in its place),
        ``progress(k, n)`` told as each k-point is read, and kept.
        ``atoms`` -- the atom index map of the run that holds the file,
        handed by the caller that knows the run (this parser knows no
        calculation): the digest states it, and a kept one stating no user
        index is read again; none -- a file no run of ours holds -- the
        engine index alone (`model/overview.md` § 2.2b, F3)."""
        path = Path(path)
        need_user = atoms is not None and atoms.user is not None
        if not (fresh or take_over):
            got = _from_digest(path, need_user=need_user)
            if got is not None:
                _clear_claim(_claim_path(path), take_over=False)
                return got
        with _claimed(path, take_over=take_over) as update:
            def told(k: int, n: int) -> None:
                update(k, n)
                if progress is not None:
                    progress(k, n)
            result, meta = _read(path, told)
            if atoms is None:
                from molbuilder.atom_permutation import AtomIndexMap
                atoms = AtomIndexMap.of_own_structure(
                    len(result.firsto) - 1,
                    "no atom index map was handed to this reading (a "
                    "run's file is read with its run's, through "
                    "molbuilder.postprocess.parse or its calculation): "
                    "nothing here states "
                    "its atoms' user index")
            _write_digest(path, result, meta, atoms)
        return replace(result, atom_index_map=atoms)


# ---- a point's averages --------------------------------------------------- #

def energies(path) -> Tuple[float, ...]:
    """Data file ``path``'s energies, at full precision, ascending: its
    kept reading's, or the file's own energy list (a small read)."""
    from molbuilder.ncdata import reading
    got = _digest(Path(path), part=("energy",))
    if got is not None and "energy" in got:
        return tuple(float(e) for e in got["energy"])
    with reading(path, sile=True) as tbt:
        return tuple(sorted(float(e) for e in tbt.E))


def point_averages(folder, label: str, *, pair: str,
                   leads: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """A POINT'S AVERAGES, per channel (`model/parse.md` § 5e): TBtrans's
    k-averaged text files (:func:`tbtrans.averaged_files`) -- the
    transmission, its eigenchannels, the device's DOS, the leads' spectral
    and bulk DOS -- each on the data file's energies, at full precision,
    matched by their order, both ascending (TBtrans prints its text files
    sorted by energy): ``{channel: {"energy_ev", "transmission",
    "eigenchannels", "dos", "lead_spectral": {lead}, "lead_bulk": {lead},
    "files": {quantity: name}}}``, each quantity that is there.  The printed
    energies where the folder holds no data file of the channel.
    :class:`ValueError` names two files that are not one grid."""
    found = averaged_files(folder, label, pair=pair, leads=leads)
    data = data_files(folder, label)
    out: Dict[str, Dict[str, Any]] = {}
    for channel, files in found.items():
        full = energies(data[channel]) if channel in data else None
        got: Dict[str, Any] = {"files": {}}

        def on(path: Path) -> List[List[float]]:
            read = read_averaged(path)
            printed = read["energy_ev"]
            nonlocal full
            if full is None:
                full = tuple(printed)
            if len(printed) != len(full) or any(
                    abs(a - b) > PRINTED_HALF_DECIMAL_EV
                    for a, b in zip(full, printed)):
                raise ValueError(f"{path.name} and the data file of its "
                                 f"run are not one grid -- not one run's "
                                 f"files")
            return read["columns"]
        for quantity, where in files.items():
            if isinstance(where, dict):
                got[quantity] = {lead: on(p)[0] for lead, p in where.items()}
                got["files"][quantity] = {lead: p.name
                                          for lead, p in where.items()}
            else:
                cols = on(where)
                got[quantity] = cols if quantity == "eigenchannels" \
                    else cols[0]
                got["files"][quantity] = where.name
        got["energy_ev"] = list(full)
        out[channel] = got
    return out


__all__ = ["TBTransNcFileParser", "ReadingClaimed", "Kept", "DIGEST_SCHEMA",
           "channel_of", "digest_of", "data_files", "kept_state", "kept",
           "claim_words", "energies", "point_averages"]
