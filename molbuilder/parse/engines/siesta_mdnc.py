"""SIESTA netCDF MD history — ``<label>.MD.nc`` → :class:`TrajectoryResult`.

WHY THIS EXISTS.  Until now every SIESTA coordinate and every frame energy
molbuilder shows was recovered by parsing ``.out`` TEXT: ``outcoor:`` blocks
for geometry, ``siesta: E_KS(eV) = …`` for energy.  Two things are wrong with
that, and both were measured rather than assumed (2026-08-15):

  * **Precision.**  The ``.out`` prints ``E_KS(eV) = -30.4405`` — four
    decimals.  Near the end of a relaxation the step-to-step energy change
    falls below 1e-4 eV, so the text form cannot express the quantity a
    convergence plot is trying to draw.  ``etot`` here is full double
    (``-30.44046192323598``).
  * **Fortran fixed-width output.**  Adjacent columns touch when a value
    fills its field (``-1.929956131.029438``) and overflow to
    ``**********`` when it does not fit.  ``parse/engines/siesta.py``
    carries a separator-inserting regex AND a structural column slicer to
    survive that.  Typed netCDF arrays have no such failure mode.

**This parser does not replace the ``.out`` parser and must not.**  The
``.out`` is the only source of run state, errors, forces and the per-SCF-cycle
history — SIESTA writes NO structured equivalent for any of those.  What this
gives is coordinates and per-step energy that never went through a text
formatter, so a mangled SCF row degrades the SCF detail panel instead of
costing a frame.

THREE FACTS ABOUT THE FILE THAT SHAPE THIS CODE.  All three verified against
SIESTA 5.4.2 (``Src/md_out.F90``, ``Src/write_md_record.F``) and a live run:

  1. **It is written only when ``WriteMDhistory`` is true AND the binary was
     built with ``-DCDF``.**  Both hold for the packaged ``molbuilder-siesta``
     env (``OUTVARS.yml`` reports ``netcdf: yes``), but neither is guaranteed
     elsewhere — so an absent file is ORDINARY, never an error.
  2. **Frame k here is ``.out`` frame k+1.**  The file holds post-move
     geometries: the manual calls them *"the predicted values for the next
     step"*, and a measured relaxation aligned at lag -1 to 4e-9 Ang while
     lag 0 differed by 5e-2.  There is no row for the input geometry.
  3. **It ACCUMULATES across runs** (manual: *"accumulative even for
     different runs"* — ``md_netcdf`` opens the existing file and appends).
     A warm restart therefore appends to the previous run's file, so frame
     index is NOT run-local and fact 2's lag is NOT a constant.

Fact 3 is why :func:`align_to_reference` matches on COORDINATES rather than
doing index arithmetic.  A hardcoded ``-1`` would be right on a fresh run and
silently wrong on every restart — which is the workflow this project is built
around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from molbuilder.chemistry import symbol_for_z
from molbuilder.frame import Frame
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import (
    ParseResult, ParseWarning, TrajectoryResult)
from molbuilder.structure import Structure

#: Unit conversions.  Keyed by the ``unit`` ATTRIBUTE the file carries on each
#: variable, never by position — ``md_out.F90`` writes ``xa`` in Bohr while
#: ``volume`` is already Ang**3, so a blanket assumption is wrong in the same
#: file.  An unrecognised unit REFUSES (see :func:`_to`), because a future
#: SIESTA that writes Angstrom would otherwise be scaled by 1.89 in silence.
from molbuilder.constants import BOHR_ANGSTROM as _BOHR_TO_ANG
from molbuilder.constants import RYDBERG_EV as _RY_TO_EV

_LENGTH: Dict[str, float] = {"bohr": _BOHR_TO_ANG, "ang": 1.0,
                             "angstrom": 1.0}
_ENERGY: Dict[str, float] = {"ry": _RY_TO_EV, "ev": 1.0}

#: netCDF classic files begin with "CDF" + a version byte; netCDF-4 is HDF5
#: ("\x89HDF").
#:
#: A CHEAP EXIT, not the correctness gate -- the schema check below (does it
#: have ``xa`` and ``etot``?) is what actually decides, and it already rejects
#: everything this would.  Mutation testing on 2026-08-15 confirmed that:
#: deleting this check left every test green, because opening a non-netCDF
#: file raises and ``can_parse`` returns False anyway.  It stays because
#: detection runs over every file in a run directory, and reading four bytes
#: beats constructing a netCDF Dataset to learn the same thing.
_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF")


def _to(value, unit: str, table: Dict[str, float], *, what: str,
        source: Path):
    """Convert *value* to molbuilder units, or refuse."""
    key = (unit or "").strip().lower()
    factor = table.get(key)
    if factor is None:
        raise ValueError(
            f"{source.name}: {what} carries unit {unit!r}, which this reader "
            f"does not know how to convert (known: "
            f"{', '.join(sorted(table))}).  Refusing rather than assuming a "
            f"factor -- a wrong one is invisible in the result and wrong by "
            f"a fixed ratio in every number downstream.")
    return value * factor


class SiestaMdNcFileParser(FileParser):
    """``<label>.MD.nc`` → a TrajectoryResult carrying what the file knows.

    **A deliberately PARTIAL result.**  ``run_state`` stays ``"unknown"``,
    forces and scf_history stay ``None``: the file contains none of them, and
    inventing a value would be worse than admitting the gap.  ``TrajectoryResult``
    already defaults ``run_state`` to ``"unknown"``, so this is a shape the
    contract anticipates rather than a special case.
    """

    name = "siesta-mdnc"
    label = "SIESTA netCDF MD history (.MD.nc)"
    hint = "the <label>.MD.nc written next to the .out when WriteMDhistory is on"
    output = TrajectoryResult

    @classmethod
    def footgun_hint_for(cls, filename: str) -> Optional[str]:
        if filename.endswith(".MD"):
            return (f"{filename} is SIESTA's UNFORMATTED Fortran MD history "
                    f"-- its layout depends on the compiler that wrote it. "
                    f"Point molbuilder at the sibling .MD.nc instead, which "
                    f"holds the same data in netCDF.")
        return None

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        path = Path(path)
        if not path.name.endswith(".MD.nc") or not path.is_file():
            return False
        try:
            with path.open("rb") as fh:
                if not fh.read(4).startswith(_MAGIC):
                    return False
        except OSError:
            return False
        # Name and magic agree; confirm it is OUR schema rather than some
        # other netCDF that happens to be called .MD.nc.
        try:
            ds = _open(path)
        except Exception:                                   # noqa: BLE001
            return False
        try:
            return {"xa", "etot"}.issubset(set(ds.variables))
        finally:
            ds.close()

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        path = Path(path)
        ds = _open(path)
        try:
            frames, warnings, lattice, series = _frames_from(ds, path)
        finally:
            ds.close()
        return TrajectoryResult(
            # The SIXTH hand-built envelope, found in the 2026-09-04
            # review: this one bypassed even its own package's helper.
            **ParseResult.envelope(cls.name, path),
            frames=frames,
            lattice=lattice,
            source_format="siesta-mdnc",
            # NOT "finished": the file says nothing about how the run ended,
            # and a restart appends to it, so even a complete-looking file
            # may belong to a job still running.
            run_state="unknown",
            error_message=None,
            runtime_info={
                "mdnc_frame_count": len(frames),
                # Recorded so a composer can say WHY it trusted this file
                # over the .out, and so the offset is visible in the result
                # rather than buried in the code that applied it.
                "mdnc_note": ("post-move geometries; frame k corresponds to "
                              ".out frame k+1 on a fresh run, and the file "
                              "accumulates across restarts"),
                **{f"mdnc_{k}": v for k, v in series.items()},
            },
            parse_warnings=warnings,
        )


def _open(path: Path):
    """Open the dataset, preferring netCDF4 and falling back to scipy.

    Both read the classic format SIESTA writes.  Two backends because the
    read path must not become a reason to add a dependency: ``netCDF4`` is
    present in the molbuilder env today, and ``scipy`` is a hard dependency
    already, so one of them is always there.
    """
    try:
        from netCDF4 import Dataset                        # type: ignore
        return Dataset(str(path))
    except ImportError:
        from scipy.io import netcdf_file                   # type: ignore
        return netcdf_file(str(path), "r", mmap=False)


def _var(ds, name: str):
    v = ds.variables.get(name) if hasattr(ds.variables, "get") \
        else ds.variables[name]
    return v


def _unit_of(var) -> str:
    """The ``unit`` attribute, whichever backend produced *var*."""
    unit = getattr(var, "unit", None)
    if isinstance(unit, bytes):
        return unit.decode("ascii", "replace")
    return unit or ""


def _frames_from(ds, path: Path):
    warnings: List[ParseWarning] = []
    xa_v = _var(ds, "xa")
    xa = _to(np.array(xa_v[:], dtype=float), _unit_of(xa_v), _LENGTH,
             what="xa (atomic coordinates)", source=path)

    iza = np.array(_var(ds, "iza")[:], dtype=int) \
        if "iza" in ds.variables else None
    if iza is None:
        raise ValueError(
            f"{path.name}: no 'iza' variable, so the atoms cannot be named. "
            f"This is not the .MD.nc layout SIESTA 5.4.2 writes.")
    elements = [symbol_for_z(int(z)) for z in iza]

    etot_v = _var(ds, "etot")
    etot = _to(np.array(etot_v[:], dtype=float), _unit_of(etot_v), _ENERGY,
               what="etot (total energy)", source=path)
    # eks and etot differ once there is smearing entropy (F = E - TS): on a
    # metal at raised ElectronicTemperature they are NOT the same number.
    # Both are carried; `energy` takes etot, and eks rides along per frame.
    eks = None
    if "eks" in ds.variables:
        eks_v = _var(ds, "eks")
        eks = _to(np.array(eks_v[:], dtype=float), _unit_of(eks_v), _ENERGY,
                  what="eks (Kohn-Sham energy)", source=path)

    cell_v = _var(ds, "cell") if "cell" in ds.variables else None
    lattice = None
    if cell_v is not None:
        cell = _to(np.array(cell_v[:], dtype=float), _unit_of(cell_v),
                   _LENGTH, what="cell (cell vectors)", source=path)
        if cell.ndim == 3 and len(cell):
            # Trajectory.lattice is the run's cell; per-frame cells are
            # reserved for variable-cell MD no consumer draws yet, so the
            # LAST step's cell is the one that describes the result.
            lattice = cell[-1]

    def _opt(name: str):
        if name not in ds.variables:
            return None
        return np.array(_var(ds, name)[:], dtype=float)

    temp, psol, volume = _opt("temp"), _opt("psol"), _opt("volume")

    # ---- THE ROW IS NOT A STEP -------------------------------------- #
    # A single .MD.nc row mixes TWO steps, and this is the trap the whole
    # module exists to avoid.  Measured on a live H2 relaxation:
    #
    #     row k :  xa   = the geometry AFTER move k+1   (the "predicted"
    #                     positions the manual warns about)
    #              etot = the energy OF geometry k      (the one just
    #                     evaluated, before the move)
    #
    # So ``Frame(structure=xa[k], energy=etot[k])`` -- the obvious
    # pairing, and what this file did in its first draft -- attaches every
    # geometry to the PREVIOUS geometry's energy.  Nothing raises, the
    # frame count looks right, and every energy in the trajectory is off
    # by one move.  On a converging relaxation the numbers are close
    # enough that a plot still looks plausible, which is what makes it
    # dangerous.
    #
    # The pairing that agrees with the .out is ``xa[k]`` with
    # ``etot[k+1]``, verified to the .out's own printing precision.  The
    # final row's energy has not been computed yet, so it is None -- an
    # honest gap rather than a recycled number.
    n_steps = xa.shape[0]
    frames: List[Frame] = []
    for k in range(n_steps):
        try:
            struct = Structure(elements=list(elements),
                               positions=np.asarray(xa[k], dtype=float))
        except Exception as exc:                            # noqa: BLE001
            warnings.append(ParseWarning(
                source=str(path), line_no=None, snippet=f"step {k}",
                error=f"could not build a Structure for step {k}: {exc}",
                category="structure"))
            continue
        e_idx = k + 1                    # see THE ROW IS NOT A STEP above
        frames.append(Frame(
            structure=struct,
            step_index=k,
            energy=(float(etot[e_idx]) if e_idx < len(etot)
                    and np.isfinite(etot[e_idx]) else None),
            forces=None,          # not in this file -- the .out owns them
            max_force=None,
            scf_history=None,     # likewise: no per-cycle data exists here
        ))

    # The remaining per-step series ride on the RESULT envelope, not on
    # Frame.  Frame's field list is a published contract (parse.md § 2) and
    # it has no slot for engine extras; widening it for four SIESTA-only
    # series would push an engine detail into the shape every parser shares.
    # `runtime_info` is already `Dict[str, Any]` and is where per-run extras
    # belong.
    # Shifted the same way as `etot`, and for the same reason: every
    # per-step SCALAR in this file describes the geometry that was just
    # evaluated, while `xa` on the same row is the one about to be tried.
    series: Dict[str, List[Optional[float]]] = {}
    for label, arr in (("eks_eV", eks), ("temperature_K", temp),
                       ("pressure_kBar", psol), ("volume_A3", volume)):
        if arr is None:
            continue
        shifted = list(arr[1:n_steps + 1])
        series[label] = (
            [float(v) if np.isfinite(v) else None for v in shifted]
            + [None] * (len(frames) - len(shifted)))
    return frames, warnings, lattice, series


# --------------------------------------------------------------------- #
#  Alignment                                                            #
# --------------------------------------------------------------------- #

def align_to_reference(reference: Sequence, candidate: Sequence, *,
                       tol: float = 1e-4) -> List[Optional[int]]:
    """Map each *reference* frame to the *candidate* frame that IS it.

    Returns one entry per reference frame: the index into *candidate* whose
    geometry matches, or ``None`` when nothing does.

    **Matching, not arithmetic — and that is the whole point.**  The obvious
    implementation is ``candidate[i - 1]``, which is correct on a fresh run
    and wrong on every warm restart, because ``.MD.nc`` accumulates across
    runs (see the module docstring, fact 3).  It is also wrong whenever the
    ``.out`` repeats its final geometry, which a relaxation does.  Comparing
    coordinates costs a few array subtractions and is immune to both.

    *tol* is in Angstrom and compares the max per-atom displacement.  The
    default is deliberately loose relative to the 4e-9 agreement measured on
    a real run: the ``.out`` values are ROUNDED to ~1e-6 Ang, so the true
    difference between a matching pair is a rounding artefact, not noise —
    but a genuinely different relaxation step is orders of magnitude further
    away than 1e-4.

    Scanning is monotonic: a candidate already claimed cannot be claimed
    again, so two reference frames with identical geometry (the repeated
    final step) map to different candidates or to ``None``, never both to
    the same row.
    """
    out: List[Optional[int]] = []
    next_free = 0
    cand_pos = [_positions_of(c) for c in candidate]
    for ref in reference:
        ref_pos = _positions_of(ref)
        hit: Optional[int] = None
        for j in range(next_free, len(cand_pos)):
            if _same(ref_pos, cand_pos[j], tol):
                hit = j
                next_free = j + 1
                break
        out.append(hit)
    return out


# --------------------------------------------------------------------- #
#  The merge — what the .out parser calls                               #
# --------------------------------------------------------------------- #

def sibling_md_nc(out_path: Path) -> Optional[Path]:
    """The ``.MD.nc`` belonging to *out_path*, or None.

    Two ways to find it, because the .out's NAME and SIESTA's ``SystemLabel``
    need not agree -- a run redirected to ``siesta.out`` with
    ``SystemLabel hemeC`` writes ``hemeC.MD.nc``:

      1. the exact stem (``x.out`` -> ``x.MD.nc``), which is the molbuilder
         convention and the common case;
      2. failing that, the only ``*.MD.nc`` in the same directory.

    **Exactly one, or nothing.**  Two .MD.nc files in a directory means two
    runs, and guessing which belongs to this .out would attach one run's
    geometry to another's energies -- the precise failure this whole module
    exists to prevent, arrived at from a different direction.
    """
    out_path = Path(out_path)
    directory = out_path.parent
    stem = out_path.name
    for suffix in (".out", ".log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    exact = directory / f"{stem}.MD.nc"
    if exact.is_file():
        return exact
    try:
        found = sorted(directory.glob("*.MD.nc"))
    except OSError:
        return None
    return found[0] if len(found) == 1 else None


def upgrade_frames(frames: Sequence, out_path: Path, *,
                   tol: float = 1e-4) -> tuple:
    """Return ``(frames, info)`` with text-derived values replaced by netCDF
    ones wherever a sibling ``.MD.nc`` describes the same geometry.

    **STRICTLY AN UPGRADE — never a regression.**  Every path that cannot
    improve a frame leaves it exactly as the text parser produced it: no
    sibling, an unreadable one, a frame with no match, a matched frame whose
    energy the file has not computed yet.  That is what makes this safe to
    apply unconditionally -- a run whose SIESTA lacked ``-DCDF`` parses
    precisely as it did before.

    **The .out's frame list is the trajectory and is never re-shaped.**  The
    same count comes back, in the same order, because ``.MD.nc`` has no row
    for the INPUT geometry (user, 2026-08-15: *"we don't want to lose the
    first input as the starting frame for the result display"*) and repeats
    nothing at the end.  This function only ever swaps values INTO an
    existing frame.
    """
    import dataclasses

    info: Dict[str, object] = {}
    nc_path = sibling_md_nc(out_path)
    if nc_path is None:
        return list(frames), info
    try:
        nc = SiestaMdNcFileParser.parse(nc_path)
    except Exception as exc:                                # noqa: BLE001
        # An unreadable sibling is not an error in the .out: the run is
        # still perfectly described by its text output.  Recorded, not
        # raised, so a broken netCDF never costs a user their results.
        info["mdnc_error"] = f"{nc_path.name}: {exc}"
        return list(frames), info

    mapping = align_to_reference(frames, nc.frames, tol=tol)
    out: List = []
    n_coords = n_energy = 0
    for frame, j in zip(frames, mapping):
        if j is None:
            out.append(frame)
            continue
        src = nc.frames[j]
        changes: Dict[str, object] = {}
        src_pos, own_pos = _positions_of(src), _positions_of(frame)
        # Shape guard: the alignment matched on these very coordinates, so a
        # mismatch here should be impossible -- which is exactly why it is
        # worth refusing rather than trusting.  Swapping in an array of a
        # different atom count would rebuild the Structure around the wrong
        # element list.
        if (src_pos is not None and own_pos is not None
                and src_pos.shape == own_pos.shape):
            # `replace` re-runs Structure.__post_init__, so the swapped
            # geometry gets the same validation any parsed one does; the
            # annotations, elements and PDB metadata ride along untouched.
            changes["structure"] = dataclasses.replace(
                frame.structure, positions=src_pos)
            n_coords += 1
        if src.energy is not None:
            changes["energy"] = src.energy
            n_energy += 1
        out.append(dataclasses.replace(frame, **changes) if changes else frame)

    info.update({
        "mdnc_source": nc_path.name,
        "mdnc_frames_matched": sum(1 for j in mapping if j is not None),
        "mdnc_coords_upgraded": n_coords,
        "mdnc_energies_upgraded": n_energy,
    })
    for key in ("mdnc_eks_eV", "mdnc_temperature_K", "mdnc_pressure_kBar",
                "mdnc_volume_A3"):
        if key in nc.runtime_info:
            info[key] = nc.runtime_info[key]
    return out, info


def _positions_of(frame) -> Optional[np.ndarray]:
    struct = getattr(frame, "structure", None)
    pos = getattr(struct, "positions", None) if struct is not None else None
    return None if pos is None else np.asarray(pos, dtype=float)


def _same(a: Optional[np.ndarray], b: Optional[np.ndarray],
          tol: float) -> bool:
    if a is None or b is None or a.shape != b.shape or not a.size:
        return False
    return bool(np.max(np.abs(a - b)) <= tol)
