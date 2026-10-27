"""A SIESTA VIBRATION'S OPTICAL CALLS AS STRUCTURED DATA -- ``<label>.optical.nc``
at the run's root, the record of the Raman phase so far (`engines/vibration.md`
§ 5.11 § 8).

MODULE  spectra.optical_data (L2; numpy, netCDF4 and the shared variable
        record; it travels beside the job, `runwrap.VIBRATION_COMPANIONS`)
ROLE    :func:`write_optical_data` -- the optical calls' payload, composed by
        `spectra.siesta_vibration` from the run's index and each call's own
        files, written as the file: every variable stating itself
        (:class:`ncdata.Var`); :func:`read_optical_data` -- the file back as
        the payload, which is the file's test and a person's script's
USED-BY the record step of each optical call and the run's finish
        (`spectra.siesta_vibration`)

**The groups** (`model/datafiles.md`): ``/`` what the calculation is, in
words, the run, SIESTA's version and the molbuilder that prepared it;
``/settings`` every optical keyword as the deck wrote it, the direction, the
selection as asked and as applied, ``δ``, and the optical deck's whole text
and sha256; ``/structure`` the reference geometry -- the result's own
layout (`spectra.datafile.write_structure_group`), its atom axis listed by
user index; ``/frames`` one row per optical call: its mode, frequency, side
and ``δQ``, the geometry it computed at, its state and the run that computed
it, its output's SCF facts, its ``.EPSIMG`` header's, and ε₂ over
``frame × photon_energy_ev × spin``.

**A frame not yet run is a row of fills** beside its state: the file is
rewritten after each call, so it is the Raman phase so far.  Every frame
that holds a spectrum holds it on one grid of photon energies -- the calls
share one deck -- and a frame on another grid is refused by name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job.
try:                                        # inside molbuilder
    from ..atom_permutation import AtomIndexMap
    from ..ncdata import (NC_LOCK, Var, as_bytes, empty, in_memory,
                          json_attr, number_attr, read_variables, reading,
                          unnumber_attr, write_group)
    from ..persist import write_bytes
    from ..structure import Structure as _Structure
    from .datafile import read_structure_group, write_structure_group
except ImportError:                         # beside a job, in mb_vibration.pyz
    from atom_permutation import AtomIndexMap  # type: ignore[no-redef]
    from ncdata import (NC_LOCK, Var, as_bytes, empty,  # type: ignore[no-redef]
                        in_memory, json_attr, number_attr, read_variables,
                        reading, unnumber_attr, write_group)
    from persist import write_bytes  # type: ignore[no-redef]
    from structure import Structure as _Structure  # type: ignore[no-redef]
    from spectra_datafile import (  # type: ignore[no-redef]
        read_structure_group, write_structure_group)

DATA_SCHEMA = "molbuilder/optical-data@1"
READABLE_DATA_SCHEMAS = (DATA_SCHEMA,)

#: WHAT THE CALCULATION IS, in words -- the file's own first sentence.
WHAT = ("SIESTA's optical calculation: the imaginary part of the dielectric "
        "function, eps2, for a field along the direction /settings states -- "
        "first-order perturbation over the self-consistent states, with no "
        "local field (independent states) -- at the force-constant run's "
        "reference geometry and at +dQ and -dQ along each selected mode "
        "(engines/vibration.md 5.11; science/vibrational-optical-response.md "
        "3)")

#: The payload's keys, and its frames' -- a key this file does not know is
#: refused by name rather than dropped.
_PAYLOAD_KEYS = ("label", "run", "siesta_version", "molbuilder_version",
                 "settings", "structure", "atom_index_map", "masses_amu",
                 "photon_energy_ev", "spin", "frames")
_SETTINGS_KEYS = ("keywords", "direction", "selection", "delta_ang",
                  "deck_name", "deck_text", "deck_sha256")
_FRAME_KEYS = ("call", "mode", "frequency_cm1", "side", "dq_amu12_ang",
               "position_ang", "state", "computed_in", "scf_iterations",
               "total_energy_ev", "fermi_level_ev", "engine_elapsed_s",
               "f_sum", "drude_wp2_ha2", "eps2")


class OpticalDataError(ValueError):
    """The payload does not lay out as this file, or the file is not one
    this molbuilder reads -- the message names the key or the schema."""


def _frame_vars(label: str) -> Tuple[Var, ...]:
    epsimg = f"{label}.EPSIMG"
    out = f"the call's output, {label}_<stage>-run<N>.out"
    return (
        Var("frame", ("frame",), "coordinate", "1", "the optical call",
            "from 0: frame 0 the force-constant run's reference geometry, "
            "then each selected mode's +dQ and -dQ sides, in the order the "
            "calls were planned (engines/vibration.md 5.11 3)", dtype="i4"),
        Var("atom", ("atom",), "coordinate", "1", "atom index",
            "the atom's user index, from 0 -- the file is listed by user "
            "index (atoms_listed_by); engine_atom_index beside it in "
            "/structure (model/overview.md 2.2b)", dtype="i4"),
        Var("xyz", ("xyz",), "coordinate", "", "Cartesian component",
            "x, y, z", dtype=str),
        Var("photon_energy_ev", ("photon_energy_ev",), "coordinate", "eV",
            "the photon energy",
            "the light's energy of each row, hbar*omega, as SIESTA wrote it: "
            "Optical.Broaden / 10 apart (Src/optical.F)",
            source=epsimg),
        Var("spin", ("spin",), "coordinate", "", "spin channel",
            "unpolarized, or up and down -- the SIESTA family's names",
            dtype=str),
        Var("call", ("frame",), "given", "", "the call's folder",
            "its folder below the run: optical/f000, optical/f001, ...",
            dtype=str, auxiliary=True),
        Var("mode", ("frame",), "given", "1", "the mode it is displaced along",
            "the mode's 1-based index in the result; -1 at frame 0",
            dtype="i4", fill=-1),
        Var("frequency_cm1", ("frame",), "given", "cm-1",
            "the mode's frequency", "the harmonic frequency; NaN at frame 0"),
        Var("side", ("frame",), "given", "1", "the side of the displacement",
            "+1 or -1 along the mode; 0 at frame 0", dtype="i4"),
        Var("dq_amu12_ang", ("frame",), "given", "amu^1/2 Angstrom",
            "the displacement along the mode",
            "dQ = delta / max_k |L_k| in mass-weighted coordinates: the atom "
            "that moves most moves by delta (engines/vibration.md 5.11 3); "
            "NaN at frame 0"),
        Var("position_ang", ("frame", "atom", "xyz"), "raw", "Angstrom",
            "the atom's position",
            "the geometry the call computed at, in the user order; NaN for a "
            "call that wrote none", source=out),
        Var("state", ("frame",), "state", "", "the call's state",
            "finished, failed or pending, by the run's one done rule "
            "(engines/vibration.md 5.11 6)", dtype=str),
        Var("computed_in", ("frame",), "state", "", "the run that computed it",
            "the run's folder below the calculation -- this run's, or the "
            "one a warm launch took the call over from; empty while pending",
            dtype=str),
        Var("scf_iterations", ("frame",), "raw", "1", "the SCF's iterations",
            "the cycles of the call's SCF; -1 where it wrote none",
            dtype="i4", fill=-1, source=out),
        Var("total_energy_ev", ("frame",), "raw", "eV", "the total energy",
            "E_KS of the call's SCF", source=out),
        Var("fermi_level_ev", ("frame",), "raw", "eV", "the Fermi level",
            "Ef of the call's last SCF cycle", source=out),
        Var("engine_elapsed_s", ("frame",), "raw", "s",
            "the engine's wall time",
            "SIESTA, launch to exit, as the call's session log measured it",
            source="the call's session log, its benchmark line"),
        Var("f_sum", ("frame", "spin"), "raw", "1", "the f-sum rule",
            "the integral of eps2 against the electrons it should count, as "
            "SIESTA states it -- 1 when every transition is in the range",
            source=f"{epsimg}, its header"),
        Var("drude_wp2_ha2", ("frame", "spin"), "raw", "Ha^2",
            "the Drude plasma frequency squared",
            "the intraband term SIESTA computes beside eps2 and leaves out "
            "of it", source=f"{epsimg}, its header"),
        Var("eps2", ("frame", "photon_energy_ev", "spin"), "raw", "1",
            "the imaginary part of the dielectric function",
            "for a field along /settings' direction, each transition spread "
            "as a Gaussian of Optical.Broaden (Src/optical.F); the width a "
            "person reads is composed when it is presented "
            "(engines/vibration.md 5.11 10)", source=epsimg),
    )


def _checked(payload: Mapping[str, Any]) -> None:
    for keys, block, where in (
            (_PAYLOAD_KEYS, payload, "the optical payload"),
            (_SETTINGS_KEYS, payload.get("settings") or {}, "its settings")):
        unknown = sorted(k for k in block if k not in keys)
        if unknown:
            raise OpticalDataError(
                f"{where} carries {unknown}, which this data file does not "
                f"know how to state; refused rather than dropped")
    for i, f in enumerate(payload.get("frames") or ()):
        unknown = sorted(k for k in f if k not in _FRAME_KEYS)
        if unknown:
            raise OpticalDataError(
                f"frame {i} carries {unknown}, which this data file does not "
                f"know how to state; refused rather than dropped")


def write_optical_data(payload: Mapping[str, Any], path) -> Path:
    """``payload`` -- the optical calls as `spectra.siesta_vibration` composes
    them -- written to ``path`` through the one atomic writer
    (`persist.write_bytes`); returns the path.  Refused by name: a key this
    file does not know, and a frame whose spectrum is on another grid of
    photon energies than the payload states."""
    _checked(payload)
    p = Path(path)
    label = str(payload["label"])
    frames = list(payload.get("frames") or ())
    grid = np.asarray(payload.get("photon_energy_ev") or (), dtype=float)
    spins = [str(s) for s in payload.get("spin") or ()]
    structure = payload["structure"]
    # THE ATOM AXIS AS ITS MAP (`model/overview.md` § 2.2b), over the
    # structure's atoms as its own door counts them.
    atoms = AtomIndexMap.from_dict(
        payload["atom_index_map"],
        n_atoms=len(_Structure.from_dict(structure).elements),
        source="the optical payload's atom_index_map")
    n_atoms = atoms.n_atoms
    settings = dict(payload.get("settings") or {})
    with NC_LOCK:
        ds = in_memory(p.name)
        ds.setncatts({
            "schema": DATA_SCHEMA, "label": label, "Conventions": "CF-1.8",
            "what": WHAT, "run": str(payload.get("run") or ""),
            "siesta_version": str(payload.get("siesta_version") or ""),
            "molbuilder_version": str(payload.get("molbuilder_version")
                                      or "")})
        g = ds.createGroup("settings")
        g.setncatts({
            "keywords": json_attr(settings.get("keywords") or {}),
            "direction": json_attr(settings.get("direction")),
            "selection": json_attr(settings.get("selection")),
            "delta_ang": number_attr(settings.get("delta_ang")),
            "deck_name": str(settings.get("deck_name") or ""),
            "deck_text": str(settings.get("deck_text") or ""),
            "deck_sha256": str(settings.get("deck_sha256") or "")})
        write_structure_group(ds, structure, atoms,
                              masses=payload.get("masses_amu"))
        g = ds.createGroup("frames")
        dims = {"frame": len(frames), "atom": atoms, "xyz": 3,
                "photon_energy_ev": int(grid.size), "spin": len(spins)}
        vars_ = _frame_vars(label)
        sizes = {**dims, "atom": n_atoms}
        values = {v.name: empty(v, sizes) for v in vars_}
        values["frame"][:] = np.arange(len(frames))
        values["atom"][:] = np.arange(n_atoms)
        values["xyz"][:] = ["x", "y", "z"]
        values["photon_energy_ev"][:] = grid
        values["spin"][:] = spins
        for k, f in enumerate(frames):
            values["call"][k] = str(f["call"])
            values["mode"][k] = -1 if f.get("mode") is None else int(f["mode"])
            values["frequency_cm1"][k] = number_attr(f.get("frequency_cm1"))
            values["side"][k] = int(f.get("side") or 0)
            values["dq_amu12_ang"][k] = number_attr(f.get("dq_amu12_ang"))
            values["state"][k] = str(f.get("state") or "")
            values["computed_in"][k] = str(f.get("computed_in") or "")
            if f.get("position_ang") is not None:
                values["position_ang"][k] = np.asarray(f["position_ang"],
                                                       dtype=float)
            if f.get("scf_iterations") is not None:
                values["scf_iterations"][k] = int(f["scf_iterations"])
            for name in ("total_energy_ev", "fermi_level_ev",
                         "engine_elapsed_s"):
                values[name][k] = number_attr(f.get(name))
            for name in ("f_sum", "drude_wp2_ha2"):
                if f.get(name) is not None:
                    values[name][k] = np.asarray(f[name], dtype=float)
            if f.get("eps2") is not None:
                e = np.asarray(f["eps2"], dtype=float)
                if e.shape != (len(spins), grid.size):
                    raise OpticalDataError(
                        f"frame {f['call']}: a spectrum of shape {e.shape}; "
                        f"the calls share one grid -- {len(spins)} spin "
                        f"channel(s) by {grid.size} photon energies")
                values["eps2"][k] = e.T
        write_group(g, dims, vars_, values)
        data = as_bytes(ds)
    write_bytes(p, data)
    return p


def read_optical_data(path, *, data=None) -> Dict[str, Any]:
    """``<label>.optical.nc`` back as the payload it was written from -- the
    file's door (`runfiles.WRITTEN`); a schema it does not read refused by
    name.  A frame not run reads its fills as ``None``.  ``data``, the
    file's bytes already in hand, is read from memory (`ncdata.reading`)."""
    out: Dict[str, Any] = {}
    with reading(path, data=data) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") not in READABLE_DATA_SCHEMAS:
            raise OpticalDataError(
                f"{Path(path).name} states the schema "
                f"{attrs.get('schema')!r}; this molbuilder reads "
                f"{', '.join(READABLE_DATA_SCHEMAS)}")
        for key in ("label", "run", "siesta_version", "molbuilder_version"):
            out[key] = str(attrs.get(key, ""))
        g = ds["settings"]
        sa = {k: g.getncattr(k) for k in g.ncattrs()}
        out["settings"] = {
            "keywords": json.loads(str(sa["keywords"])),
            "direction": json.loads(str(sa["direction"])),
            "selection": json.loads(str(sa["selection"])),
            "delta_ang": unnumber_attr(sa["delta_ang"]),
            "deck_name": str(sa["deck_name"]),
            "deck_text": str(sa["deck_text"]),
            "deck_sha256": str(sa["deck_sha256"])}
        st = read_structure_group(ds["structure"],
                                  source=f"{Path(path).name}/structure")
        out["structure"] = st["structure"]
        out["atom_index_map"] = (st["atoms"].to_dict()
                                 if st["atoms"] is not None else None)
        masses = st["masses"]
        out["masses_amu"] = ([float(m) for m in masses]
                             if masses.size and np.isfinite(masses).all()
                             else None)
        fv = read_variables(ds["frames"])
        out["photon_energy_ev"] = [float(x) for x in
                                   fv["photon_energy_ev"]["data"]]
        out["spin"] = [str(s) for s in fv["spin"]["data"]]
        frames: List[Dict[str, Any]] = []
        for k in range(int(fv["frame"]["data"].shape[0])):
            mode = int(fv["mode"]["data"][k])
            pos = np.asarray(fv["position_ang"]["data"][k], dtype=float)
            eps2 = np.asarray(fv["eps2"]["data"][k], dtype=float).T
            f_sum = np.asarray(fv["f_sum"]["data"][k], dtype=float)
            drude = np.asarray(fv["drude_wp2_ha2"]["data"][k], dtype=float)
            iters = int(fv["scf_iterations"]["data"][k])
            frames.append({
                "call": str(fv["call"]["data"][k]),
                "mode": None if mode < 0 else mode,
                "frequency_cm1": unnumber_attr(fv["frequency_cm1"]["data"][k]),
                "side": int(fv["side"]["data"][k]),
                "dq_amu12_ang": unnumber_attr(fv["dq_amu12_ang"]["data"][k]),
                "position_ang": (pos.tolist() if np.isfinite(pos).all()
                                 and pos.size else None),
                "state": str(fv["state"]["data"][k]),
                "computed_in": str(fv["computed_in"]["data"][k]),
                "scf_iterations": None if iters < 0 else iters,
                **{name: unnumber_attr(fv[name]["data"][k])
                   for name in ("total_energy_ev", "fermi_level_ev",
                                "engine_elapsed_s")},
                "f_sum": (f_sum.tolist() if f_sum.size
                          and np.isfinite(f_sum).all() else None),
                "drude_wp2_ha2": (drude.tolist() if drude.size
                                  and np.isfinite(drude).all() else None),
                "eps2": (eps2 if eps2.size and np.isfinite(eps2).all()
                         else None)})
        out["frames"] = frames
    return out


__all__ = ["DATA_SCHEMA", "READABLE_DATA_SCHEMAS", "OpticalDataError", "WHAT",
           "write_optical_data", "read_optical_data"]
