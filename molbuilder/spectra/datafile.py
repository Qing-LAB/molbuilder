"""THE VIBRATION RESULT AS STRUCTURED DATA -- ``<label>.spectra.nc``, the
same result the JSON holds, in NetCDF-4 groups (`engines/vibration.md`
§ 6.10.5).

MODULE  spectra.datafile (L2; numpy, netCDF4 and the shared variable record)
ROLE    :func:`write_spectra_data` -- the payload the JSON is written from, as
        the file: every variable stating itself (:class:`ncdata.Var`), the
        scalar blocks as attributes, the structure whole as JSON text beside
        the arrays an analysis reaches for; :func:`read_spectra_data` -- the
        file back as the payload, which is the file's test
USED-BY `sidecars.spectra.write_spectra_payload`, the one writer of the
        result, at every write on both engines -- beside the job in both
        bundles (`runwrap.VIBRATION_COMPANIONS`, `runwrap.PYSCF_COMPANIONS`)

**One source, two files.**  The ``.nc`` states the sha256 of the JSON bytes
written with it, which ties each to its pair at every phase's rewrite.  The
JSON stays the result a person reads and the Results tab loads; molbuilder
itself reads the ``.nc`` only in its round-trip test.

**What was not computed says so once**: a fill value where the JSON has
``null`` (a number's NaN, a text's empty string, an integer's stated fill,
a flag's -1), and the phase and route attributes that already state what
ran -- no second flag beside them.  The equilibrium energy subgroup travels
whole or not at all (§ 6.2), so its ``scf_energy_eh`` says for all three.

**Nothing is stated twice**: ``/structure`` holds the arrays an analysis
reaches for and ``Structure.to_dict()`` whole as JSON text -- never a second
encoding of the labels, channels or ``info`` by hand (`model/structure.md`
§ 2.1), so a field the structure gains later reaches the ``.nc`` with no
edit here.  A key of a block this file does not know is refused by name --
a fact the writer thinks it saved and this file would drop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job.
try:                                        # inside molbuilder
    from ..ncdata import (NC_LOCK, Var, as_bytes, empty, flag, in_memory,
                          json_attr, number_attr, read_variables, unflag,
                          unnumber_attr, write_group)
    from ..persist import write_bytes
    from ..structure import Structure as _Structure
except ImportError:                         # beside a job, in a bundle
    from structure import Structure as _Structure  # type: ignore[no-redef]
    from ncdata import (NC_LOCK, Var, as_bytes, empty, flag,
                        in_memory, json_attr, number_attr, read_variables,
                        unflag, unnumber_attr, write_group)
    from persist import write_bytes

DATA_SCHEMA = "molbuilder/spectra-data@1"

#: The fill of an integer attribute that can be not stated.
_INT_FILL = -1


class SpectraDataError(ValueError):
    """The payload does not lay out as this file -- the message names the
    key or shape, ready to surface verbatim."""


# --------------------------------------------------------------------- #
#  The scalar blocks -- each key's kind, so a round trip is exact         #
# --------------------------------------------------------------------- #

#: The top-level scalars, as attributes of the root: ``key -> kind`` --
#: ``text``, ``int``, ``int?`` (nullable), ``number?``, ``flag?``, ``json``.
_ROOT = {
    "engine": "text", "engine_version": "text",
    "molbuilder_version": "text", "timestamp": "text",
    "n_atoms_total": "int", "methods_text": "text",
    "bibliography_keys": "json", "selected_mode_idxs_1based": "json",
    "config": "json", "engine_metadata": "json", "runtime_info": "json",
    "phase_frequencies": "text", "phase_relaxation": "text",
    "phase_raman": "text", "phase_ir": "text", "phase_es": "text",
    "ir_route": "text", "ir_fd_step_ang": "number?",
    "raman_route": "text", "raman_fd_step_ang": "number?",
    "hessian_scope": "text", "n_atoms_in_hessian": "int?",
    "hessian_density_fit": "flag?",
}
#: The ``relaxation`` block, as ``relaxation_<key>`` attributes.
_RELAXATION = {
    "enabled": "flag?", "already_relaxed": "flag?", "n_steps": "int",
    "max_force_eh_bohr": "number?", "max_force_all_atoms_eh_bohr": "number?",
    "converged": "flag?", "warning": "text", "record": "json",
}
#: The ``thermo`` block but its grid, as attributes of ``/thermo``.
_THERMO = {
    "regime": "text", "temperature_K": "number?", "pressure_atm": "number?",
    "zpe_eh": "number?", "u_vib_eh": "number?", "h_eh": "number?",
    "s_eh_k": "number?", "g_eh": "number?", "n_modes": "int",
    "n_imag_excluded": "int", "n_rigid_removed": "int", "note": "text",
}
#: The thermo grid's curves (`normal_modes.vibrational_thermo_grid`).
_GRID_CURVES = (
    ("zpe_eh", "Eh", "the zero-point energy", "the harmonic ZPE at this temperature"),
    ("u_vib_eh", "Eh", "the vibrational internal energy", "U_vib at this temperature"),
    ("h_eh", "Eh", "the enthalpy", "H at this temperature, as the regime defines it"),
    ("s_eh_k", "Eh/K", "the entropy", "S at this temperature, as the regime defines it"),
    ("g_eh", "Eh", "the free energy", "G = H - T*S at this temperature"),
)
#: A mode's electronic-structure probe (`engines/vibration.md` § 4.8).
_ES_SCALARS = (
    ("amplitude_ang", "Angstrom", "the displacement amplitude",
     "how far along the mode the probe displaced, each way"),
    ("scf_energy_eq_eh", "Eh", "the SCF energy at equilibrium",
     "E_SCF at the equilibrium geometry"),
    ("scf_energy_minus_eh", "Eh", "the SCF energy at -amplitude",
     "E_SCF displaced by -amplitude along the mode"),
    ("scf_energy_plus_eh", "Eh", "the SCF energy at +amplitude",
     "E_SCF displaced by +amplitude along the mode"),
)
_ES_WINDOWS = (
    ("mo_energies_eq_eh", "the orbital energies at equilibrium"),
    ("mo_energies_minus_eh", "the orbital energies at -amplitude"),
    ("mo_energies_plus_eh", "the orbital energies at +amplitude"),
)


def _encode(kind: str, value):
    if kind == "text":
        return "" if value is None else str(value)
    if kind == "int":
        return int(value)
    if kind == "int?":
        return _INT_FILL if value is None else int(value)
    if kind == "number?":
        return number_attr(value)
    if kind == "flag?":
        return np.int8(flag(value))
    if kind == "json":
        return json_attr(value)
    raise SpectraDataError(f"no encoding for kind {kind!r}")


def _decode(kind: str, value):
    if kind == "text":
        return str(value)
    if kind == "int":
        return int(value)
    if kind == "int?":
        v = int(value)
        return None if v == _INT_FILL else v
    if kind == "number?":
        return unnumber_attr(value)
    if kind == "flag?":
        return unflag(value)
    if kind == "json":
        return json.loads(str(value))
    raise SpectraDataError(f"no encoding for kind {kind!r}")


def _block_attrs(block: Mapping[str, Any], table: Mapping[str, str],
                 prefix: str, where: str) -> Dict[str, Any]:
    """``block``'s keys as typed attributes, ``prefix + key``; a key the
    table does not know is refused by name."""
    unknown = sorted(k for k in block if k not in table)
    if unknown:
        raise SpectraDataError(
            f"{where} carries {unknown}, which this data file does not "
            f"know how to state; refused rather than dropped")
    return {prefix + k: _encode(table[k], block[k]) for k in block}


def _block_back(attrs: Mapping[str, Any], table: Mapping[str, str],
                prefix: str) -> Dict[str, Any]:
    return {k: _decode(kind, attrs[prefix + k])
            for k, kind in table.items() if prefix + k in attrs}


# --------------------------------------------------------------------- #
#  The variables                                                          #
# --------------------------------------------------------------------- #

def _structure_vars() -> Tuple[Var, ...]:
    return (
        Var("element", ("atom",), "given", "", "the atom's element",
            "the chemical symbol, in the input order", dtype=str,
            auxiliary=True),
        Var("position_ang", ("atom", "xyz"), "given", "Angstrom",
            "the atom's position",
            "the geometry the Hessian is taken at (§ 4.2), in the input order"),
        Var("mass_amu", ("atom",), "given", "amu", "the atom's mass",
            "the mass the analysis weighted by (equilibrium.masses_amu, I3)"),
        Var("held", ("atom",), "given", "1", "whether the atom was held",
            "the structure's frozen_atoms label: 1 held, 0 free",
            dtype="i1", flag=True),
        Var("cell_ang", ("xyz", "xyz"), "given", "Angstrom", "the cell",
            "the box the run had, a row per lattice vector; NaN for none"),
        Var("axis_kind", ("xyz",), "given", "", "each axis's kind",
            "periodic or isolated, as the run was given it", dtype=str),
    )


def _equilibrium_vars() -> Tuple[Var, ...]:
    return (Var("mo_energy_eh", ("mo",), "raw", "Eh", "the orbital energies",
                "every MO energy at the equilibrium SCF, ascending",
                source="the equilibrium SCF"),)


def _mode_vars() -> Tuple[Var, ...]:
    return (
        Var("index_1based", ("mode",), "coordinate", "1", "the mode's index",
            "1-based, ascending frequency", dtype="i4"),
        Var("frequency_cm1", ("mode",), "derived", "cm-1", "the frequency",
            "the harmonic frequency; negative for an imaginary mode"),
        Var("free_atom_index", ("free_atom",), "given", "1",
            "which atom each row is",
            "the 0-based atom index, in the input order, of each free-atom "
            "row of the eigenvectors (free_atom_idxs)", dtype="i4",
            auxiliary=True),
        Var("eigenvector_canonical", ("mode", "free_atom", "xyz"), "derived",
            "1", "the mass-weighted eigenvector",
            "sum over atoms of m |L|^2 = 1 (§ 6.3)"),
        Var("eigenvector_display", ("mode", "free_atom", "xyz"), "derived",
            "1", "the eigenvector for drawing", "max |L| = 1 (§ 6.3)"),
        Var("zero_point_amplitude_amu12_ang", ("mode",), "derived",
            "amu^1/2 Angstrom", "the zero-point amplitude",
            "sqrt(hbar / (2 omega)) in mass-weighted coordinates (§ 6.6)"),
        Var("zero_point_displacement_ang", ("mode", "free_atom", "xyz"),
            "derived", "Angstrom", "the zero-point displacement",
            "each atom's displacement at the zero-point amplitude (§ 6.6)"),
        Var("ir_intensity_km_mol", ("mode",), "raw", "km/mol",
            "the IR intensity", "as the engine computed it; NaN where not",
            source="the engine"),
        Var("raman_activity_a4_amu", ("mode",), "raw", "A^4/amu",
            "the Raman activity", "as the engine computed it; NaN where not",
            source="the engine"),
        Var("ir_active", ("mode",), "derived", "1", "whether IR active",
            "the activity class's IR verdict (§ 6.6); -1 where not computed",
            dtype="i1", flag=True),
        Var("raman_active", ("mode",), "derived", "1", "whether Raman active",
            "the activity class's Raman verdict (§ 6.6); -1 where not "
            "computed", dtype="i1", flag=True),
        Var("activity_class", ("mode",), "derived", "", "the activity class",
            "the whole-run judgement of § 6.6", dtype=str),
        Var("has_imag", ("mode",), "derived", "1", "whether imaginary",
            "1 for an imaginary frequency", dtype="i1", flag=True),
        Var("removed_motion", ("removed", "free_atom", "xyz"), "derived", "1",
            "a removed whole-body motion",
            "an orthonormal Cartesian pattern over the free atoms removed "
            "before diagonalising (removed_motions.patterns)"),
    )


def _es_vars() -> Tuple[Var, ...]:
    return (
        Var("mode_index_1based", ("es",), "coordinate", "1",
            "the probed mode", "index_1based of the mode this row is of",
            dtype="i4"),
        Var("homo_index_in_window", ("es",), "given", "1",
            "the HOMO's index in the window",
            "0-based index of the HOMO among the window's orbitals",
            dtype="i4"),
        *(Var(name, ("es",), "raw", units, long_name, definition,
              source="the probe's SCFs")
          for name, units, long_name, definition in _ES_SCALARS),
        *(Var(name, ("es", "window"), "raw", "Eh", long_name,
              "the orbital window's energies, ascending",
              source="the probe's SCFs")
          for name, long_name in _ES_WINDOWS),
    )


def _grid_vars() -> Tuple[Var, ...]:
    return (
        Var("temperatures_K", ("temperature",), "coordinate", "K",
            "the temperature", "each temperature the curves are drawn at"),
        *(Var(name, ("temperature",), "derived", units, long_name,
              definition)
          for name, units, long_name, definition in _GRID_CURVES),
    )


# --------------------------------------------------------------------- #
#  Write                                                                  #
# --------------------------------------------------------------------- #

def _fill(variables: Sequence[Var], dims: Mapping[str, int]
          ) -> Dict[str, np.ndarray]:
    return {v.name: empty(v, dims) for v in variables}


def write_spectra_data(payload: Mapping[str, Any], path, *,
                       json_sha256: str, label: str) -> Path:
    """``payload`` -- the result as the JSON is written from, its modes'
    derived values stated -- written to ``path`` as the data file, through
    the one atomic writer (`persist.write_bytes`); returns the path.
    ``json_sha256`` ties the file to the JSON written with it."""
    p = Path(path)
    unknown = sorted(k for k in payload
                     if k not in _ROOT and k not in (
                         "schema_version", "free_atom_idxs",
                         "frozen_atom_idxs", "structure", "equilibrium",
                         "modes", "relaxation", "thermo", "removed_motions"))
    if unknown:
        raise SpectraDataError(
            f"the result carries {unknown}, which this data file does not "
            f"know how to state; refused rather than dropped")
    structure = payload.get("structure")
    n_atoms = int(payload["n_atoms_total"])
    free = [int(i) for i in payload["free_atom_idxs"]]
    modes = list(payload.get("modes") or ())
    eq = dict(payload.get("equilibrium") or {})
    removed = dict(payload.get("removed_motions") or {})
    patterns = list(removed.get("patterns") or ())
    thermo = dict(payload.get("thermo") or {})
    grid = dict(thermo.pop("grid", None) or {})
    relaxation = dict(payload.get("relaxation") or {})

    # ONE NETCDF ACCESS AT A TIME IN THE PROCESS (`ncdata.NC_LOCK`).
    with NC_LOCK:
        ds = in_memory(p.name)
        attrs: Dict[str, Any] = {
            "schema": DATA_SCHEMA, "json_sha256": str(json_sha256),
            "spectra_schema_version": int(payload["schema_version"]),
            "label": str(label), "Conventions": "CF-1.8",
            **_block_attrs({k: payload[k] for k in _ROOT if k in payload},
                           _ROOT, "", "the result"),
            **_block_attrs(relaxation, _RELAXATION, "relaxation_",
                           "the relaxation block"),
        }
        if "count" in removed:
            attrs["removed_motions_count"] = int(removed["count"])
        ds.setncatts(attrs)
        ds.createDimension("xyz", 3)

        # /structure -- the arrays an analysis reaches for, and the structure
        # whole as its one serialiser states it.
        g = ds.createGroup("structure")
        vars_ = _structure_vars()
        values = _fill(vars_, {"atom": n_atoms, "xyz": 3})
        held: List[int]
        if structure is not None:
            # THROUGH THE STRUCTURE'S OWN DOOR (`model/structure.md` § 2.1):
            # its elements, positions, held set, cell and axis kinds as the
            # instance states them, never a hand-walk of the dict.
            s = _Structure.from_dict(structure)
            values["element"][:] = [str(e) for e in s.elements]
            values["position_ang"][:] = np.asarray(s.positions, dtype=float)
            held = [int(i) for i in (s.frozen_atoms or ())]
            if s.cell is not None:
                values["cell_ang"][:] = np.asarray(s.cell, dtype=float)
            if s.axis_kind is not None:
                values["axis_kind"][:] = [str(k) for k in s.axis_kind]
        else:
            held = [int(i) for i in payload.get("frozen_atom_idxs") or ()]
        values["held"][:] = 0
        for i in held:
            values["held"][i] = 1
        if eq.get("masses_amu") is not None:
            values["mass_amu"][:] = np.asarray(eq["masses_amu"], dtype=float)
        write_group(g, {"atom": n_atoms}, vars_, values)
        g.setncattr("structure_json", json_attr(structure))

        # /equilibrium -- the energy subgroup, whole or not at all (§ 6.2).
        g = ds.createGroup("equilibrium")
        mo = eq.get("mo_energies_eh")
        g.setncatts({"scf_energy_eh": number_attr(eq.get("scf_energy_eh")),
                     "homo_idx": _encode("int?", eq.get("homo_idx"))})
        vars_ = _equilibrium_vars()
        values = _fill(vars_, {"mo": len(mo or ())})
        if mo:
            values["mo_energy_eh"][:] = np.asarray(mo, dtype=float)
        write_group(g, {"mo": len(mo or ())}, vars_, values)

        # /modes
        g = ds.createGroup("modes")
        n_free = len(free)
        dims = {"mode": len(modes), "free_atom": n_free,
                "removed": len(patterns)}
        vars_ = _mode_vars()
        values = _fill(vars_, {**dims, "xyz": 3})
        values["free_atom_index"][:] = free
        for k, m in enumerate(modes):
            values["index_1based"][k] = int(m["index_1based"])
            values["frequency_cm1"][k] = float(m["frequency_cm1"])
            values["eigenvector_canonical"][k] = np.asarray(
                m["eigenvector_canonical"], dtype=float)
            values["eigenvector_display"][k] = np.asarray(
                m["eigenvector_display"], dtype=float)
            values["has_imag"][k] = flag(m.get("has_imag"))
            for name in ("ir_intensity_km_mol", "raman_activity_a4_amu",
                         "zero_point_amplitude_amu12_ang"):
                values[name][k] = number_attr(m.get(name))
            if m.get("zero_point_displacement_ang") is not None:
                values["zero_point_displacement_ang"][k] = np.asarray(
                    m["zero_point_displacement_ang"], dtype=float)
            values["ir_active"][k] = flag(m.get("ir_active"))
            values["raman_active"][k] = flag(m.get("raman_active"))
            values["activity_class"][k] = str(m.get("activity_class") or "")
        for r, pattern in enumerate(patterns):
            values["removed_motion"][r] = np.asarray(pattern, dtype=float)
        write_group(g, dims, vars_, values)

        # /modes/electronic_structure -- one row per probed mode.
        probed = [m for m in modes
                  if m.get("electronic_structure") is not None]
        windows = {len(m["electronic_structure"]["mo_energies_eq_eh"])
                   for m in probed}
        if len(windows) > 1:
            raise SpectraDataError(
                f"the probed modes' orbital windows differ in size "
                f"({sorted(windows)}): one window per result")
        window = windows.pop() if windows else 0
        ge = g.createGroup("electronic_structure")
        vars_ = _es_vars()
        values = _fill(vars_, {"es": len(probed), "window": window})
        for k, m in enumerate(probed):
            es = m["electronic_structure"]
            values["mode_index_1based"][k] = int(m["index_1based"])
            values["homo_index_in_window"][k] = int(es["homo_index_in_window"])
            for name, *_ in _ES_SCALARS:
                values[name][k] = float(es[name])
            for name, _ in _ES_WINDOWS:
                values[name][k] = np.asarray(es[name], dtype=float)
        write_group(ge, {"es": len(probed), "window": window}, vars_, values)

        # /thermo, and its grid.
        g = ds.createGroup("thermo")
        g.setncatts(_block_attrs(thermo, _THERMO, "", "the thermo block"))
        gg = g.createGroup("grid")
        if grid:
            temps = [float(t) for t in grid.get("temperatures_K") or ()]
            vars_ = _grid_vars()
            values = _fill(vars_, {"temperature": len(temps)})
            values["temperatures_K"][:] = temps
            unknown = sorted(k for k in grid if k != "temperatures_K"
                             and k not in {n for n, *_ in _GRID_CURVES})
            if unknown:
                raise SpectraDataError(
                    f"the thermo grid carries {unknown}, which this data file "
                    f"does not know how to state; refused rather than dropped")
            for name, *_ in _GRID_CURVES:
                if name in grid:
                    values[name][:] = np.asarray(grid[name], dtype=float)
            write_group(gg, {"temperature": len(temps)}, vars_, values)
            gg.setncattr("curves", json_attr([n for n, *_ in _GRID_CURVES
                                              if n in grid]))
        data = as_bytes(ds)
    write_bytes(p, data)
    return p


# --------------------------------------------------------------------- #
#  Read                                                                   #
# --------------------------------------------------------------------- #

def read_spectra_data(path) -> Dict[str, Any]:
    """``<label>.spectra.nc`` back as the payload the JSON holds -- the
    file's door (`runfiles.WRITTEN`); refuses another schema by name."""
    import netCDF4
    out: Dict[str, Any] = {}
    with NC_LOCK, netCDF4.Dataset(str(path), "r") as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") != DATA_SCHEMA:
            raise SpectraDataError(
                f"{Path(path).name} states the schema "
                f"{attrs.get('schema')!r}, not {DATA_SCHEMA}")
        out["schema_version"] = int(attrs["spectra_schema_version"])
        out.update(_block_back(attrs, _ROOT, ""))
        relaxation = _block_back(attrs, _RELAXATION, "relaxation_")
        out["relaxation"] = relaxation
        # /structure
        g = ds["structure"]
        sv = read_variables(g)
        structure = json.loads(str(g.getncattr("structure_json")))
        held_flags = sv["held"]["data"]
        masses = sv["mass_amu"]["data"]
        if structure is not None:
            out["structure"] = structure
        else:
            out["frozen_atom_idxs"] = [int(i) for i, h in enumerate(held_flags)
                                       if int(h) == 1]
        # /equilibrium
        g = ds["equilibrium"]
        scf = unnumber_attr(g.getncattr("scf_energy_eh"))
        mo = read_variables(g)["mo_energy_eh"]["data"]
        eq: Dict[str, Any] = {
            "scf_energy_eh": scf,
            "mo_energies_eh": (None if scf is None
                               else [float(x) for x in mo]),
            "homo_idx": _decode("int?", g.getncattr("homo_idx")),
        }
        if masses.size and np.isfinite(masses).all():
            eq["masses_amu"] = [float(m) for m in masses]
        out["equilibrium"] = eq
        # /modes
        g = ds["modes"]
        mv = read_variables(g)
        out["free_atom_idxs"] = [int(i) for i in mv["free_atom_index"]["data"]]
        n_modes = int(mv["index_1based"]["data"].shape[0])
        es_rows: Dict[int, Dict[str, Any]] = {}
        ge = g["electronic_structure"]
        ev = read_variables(ge)
        for k, idx in enumerate(ev["mode_index_1based"]["data"]):
            es_rows[int(idx)] = {
                "amplitude_ang": float(ev["amplitude_ang"]["data"][k]),
                **{name: [float(x) for x in ev[name]["data"][k]]
                   for name, _ in _ES_WINDOWS},
                "homo_index_in_window": int(
                    ev["homo_index_in_window"]["data"][k]),
                **{name: float(ev[name]["data"][k])
                   for name, *_ in _ES_SCALARS[1:]},
            }
        modes = []
        for k in range(n_modes):
            idx = int(mv["index_1based"]["data"][k])
            # THE DERIVATION'S RULE (`spectra.derived`): an imaginary mode
            # has no zero-point amplitude and no displacement.
            amplitude = unnumber_attr(
                mv["zero_point_amplitude_amu12_ang"]["data"][k])
            m: Dict[str, Any] = {
                "index_1based": idx,
                "frequency_cm1": float(mv["frequency_cm1"]["data"][k]),
                "raman_activity_a4_amu": unnumber_attr(
                    mv["raman_activity_a4_amu"]["data"][k]),
                "ir_intensity_km_mol": unnumber_attr(
                    mv["ir_intensity_km_mol"]["data"][k]),
                "eigenvector_canonical": mv["eigenvector_canonical"]["data"][k]
                .tolist(),
                "eigenvector_display": mv["eigenvector_display"]["data"][k]
                .tolist(),
                "has_imag": bool(unflag(mv["has_imag"]["data"][k])),
                "electronic_structure": es_rows.get(idx),
                "ir_active": unflag(mv["ir_active"]["data"][k]),
                "raman_active": unflag(mv["raman_active"]["data"][k]),
                "activity_class": str(mv["activity_class"]["data"][k]),
                "zero_point_amplitude_amu12_ang": amplitude,
                "zero_point_displacement_ang": (
                    None if amplitude is None else
                    mv["zero_point_displacement_ang"]["data"][k].tolist()),
            }
            modes.append(m)
        out["modes"] = modes
        removed: Dict[str, Any] = {}
        if "removed_motions_count" in attrs:
            removed = {"count": int(attrs["removed_motions_count"]),
                       "patterns": mv["removed_motion"]["data"].tolist()}
        out["removed_motions"] = removed
        # /thermo
        g = ds["thermo"]
        thermo = _block_back({k: g.getncattr(k) for k in g.ncattrs()},
                             _THERMO, "")
        gg = g["grid"]
        if "curves" in gg.ncattrs():
            gv = read_variables(gg)
            grid = {"temperatures_K": [float(t) for t in
                                       gv["temperatures_K"]["data"]]}
            for name in json.loads(str(gg.getncattr("curves"))):
                grid[name] = [float(x) for x in gv[name]["data"]]
            thermo["grid"] = grid
        out["thermo"] = thermo
    return out


__all__ = ["DATA_SCHEMA", "SpectraDataError", "write_spectra_data",
           "read_spectra_data"]
