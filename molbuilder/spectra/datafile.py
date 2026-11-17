"""THE VIBRATION RESULT AS STRUCTURED DATA -- ``<label>.spectra.nc``, the
same result the JSON holds, in NetCDF-4 groups (`engines/vibration.md`
§ 6.10.5; every name it holds, `model/datafiles.md` § 9.2).

MODULE  spectra.datafile (L2; numpy, netCDF4 and the shared variable record)
ROLE    :func:`write_spectra_data` -- the payload the JSON is written from, as
        the file: every variable stating itself (:class:`ncdata.Var`), the
        scalar blocks' texts, counts and flags as attributes and their
        numbers as variables of no dimension, the structure whole as JSON
        text beside the arrays an analysis reaches for; :func:`read_spectra_data` -- the
        file back as the payload, which is the file's test
USED-BY `sidecars.spectra.write_spectra_payload`, the one writer of the
        result, at every write on both engines -- beside the job in both
        bundles (`runwrap.VIBRATION_COMPANIONS`, `runwrap.PYSCF_COMPANIONS`)

**The file states itself** (`model/datafiles.md` § 2b): its title, its
schema, the molbuilder that wrote the result, and its source -- the engine
and its version that computed it; every group its title.

**One source, two files.**  The ``.nc`` states the sha256 of the JSON bytes
written with it, which ties each to its pair at every phase's rewrite.  The
JSON stays the result a person reads and the Results tab loads; molbuilder
itself reads the ``.nc`` in its round-trip test, for its atom index map
(`atom_index`), and for a person's script (`postprocess.read_spectra_data`).

**What was not computed says so once**: a fill value where the JSON has
``null`` (a number's NaN, a text's empty string, an integer's stated fill,
a flag's -1), and the phase and route attributes that already state what
ran -- no second flag beside them.  The equilibrium energy subgroup travels
whole or not at all (§ 6.2), so its total energy (the result's
``scf_energy_eh``) says for all three.

**Nothing is stated twice**: ``/structure`` holds the arrays an analysis
reaches for and ``Structure.to_dict()`` whole as JSON text -- never a second
encoding of the labels, channels or ``info`` by hand (`model/structure.md`
§ 2.1), so a field the structure gains later reaches the ``.nc`` with no
edit here.  A key of a block this file does not know is refused by name --
a fact the writer thinks it saved and this file would drop.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job.
try:                                        # inside molbuilder
    from ..atom_permutation import MAP_KEY, AtomIndexMap
    from ..ncdata import (NC_LOCK, Var, as_bytes, axis_values, checked,
                          coordinate, empty, flag, in_memory,
                          json_attr, key_of, named_now,
                          number_attr, read_structure_group, read_variables,
                          state, unflag, unjson_attr, unnumber_attr,
                          write_group, write_structure_group)
    from ..persist import unknown_keys, write_bytes
except ImportError:                         # beside a job, in a bundle
    from atom_permutation import MAP_KEY, AtomIndexMap  # type: ignore[no-redef]
    from ncdata import (NC_LOCK, Var, as_bytes, axis_values, checked,
                        coordinate, empty, flag, in_memory,
                        json_attr, key_of, named_now,
                        number_attr, read_structure_group, read_variables,
                        state, unflag, unjson_attr, unnumber_attr,
                        write_group, write_structure_group)
    from persist import unknown_keys, write_bytes  # type: ignore[no-redef]

#: Why a key a data file does not state is refused at its write -- this
#: one's and the optical calls' (`spectra.optical_data`).
UNSTATED = (".  A key this data file does not know how to state is refused "
            "rather than dropped")

DATA_SCHEMA = "molbuilder/spectra-data@10"
#: The schemas the reader reads: a file written before 2.32.0 (``@1``) holds
#: the same variables without its atom index map, read as it is and never
#: rewritten -- its map made as it is read (`engines/vibration.md`
#: § 6.10.5; `model/overview.md` § 2.2b, F3); one written before 2.33.2
#: (``@1``, ``@2``) names four axes' values otherwise, and before 2.33.3
#: (``@3``) the temperatures' (each axis's ``earlier``); one before 2.36.0
#: (``@1``-``@4``) holds no ``/partial_charges``; one before 2.43.0
#: (``@1``-``@6``) no ``optical`` citation; one before 2.47.0 (``@1``-``@7``)
#: keeps its numbers as attributes (:func:`_block_back`) and states no title
#: or source; one before the framework's one structure layout (``@1``-``@8``)
#: states each axis's kind over ``xyz`` -- the reader takes the structure
#: from ``structure_json``, so each reads as it is (`model/datafiles.md`
#: § 3a); one before 2.72.3 (``@1``-``@9``) names some variables and
#: attributes by the result's keys -- read by them, each variable's
#: ``earlier`` names, each attribute's key (`model/datafiles.md` § 2a
#: rule 5).
READABLE_DATA_SCHEMAS = ("molbuilder/spectra-data@1",
                         "molbuilder/spectra-data@2",
                         "molbuilder/spectra-data@3",
                         "molbuilder/spectra-data@4",
                         "molbuilder/spectra-data@5",
                         "molbuilder/spectra-data@6",
                         "molbuilder/spectra-data@7",
                         "molbuilder/spectra-data@8",
                         "molbuilder/spectra-data@9", DATA_SCHEMA)
#: The fill of an integer attribute that can be not stated.
_INT_FILL = -1


class SpectraDataError(ValueError):
    """The payload does not lay out as this file -- the message names the
    key or shape, ready to surface verbatim."""


# --------------------------------------------------------------------- #
#  The scalar blocks -- each key's kind, so a round trip is exact         #
# --------------------------------------------------------------------- #

def _number(name: str, kind: str, units: str, long_name: str,
            definition: str, source: str = "", earlier=()) -> Var:
    """A block's number: a variable of no dimension, NaN where the result
    states none (`model/datafiles.md` § 2); ``earlier`` the names an older
    schema wrote it under -- its record key is the block table's."""
    return Var(name, (), kind, units, long_name, definition, source=source,
               earlier=tuple(earlier))


#: The top-level scalars, as attributes of the root: ``key -> kind`` --
#: ``text``, ``int``, ``int?`` (nullable), ``flag?``, ``json`` -- or
#: ``(kind, name)`` for one the file names otherwise than the result keys it
#: (`model/datafiles.md` § 2a rule 2; an older file names it by its key),
#: and a number as its variable (:func:`_number`).
_ROOT = {
    "engine": "text", "engine_version": "text",
    "molbuilder_version": "text", "timestamp": "text",
    "n_atoms_total": "int", "methods_text": "text",
    "bibliography_keys": "json",
    "selected_mode_idxs_1based": ("json", "selected_mode_index_1based"),
    "config": "json", "engine_metadata": "json", "runtime_info": "json",
    "phase_frequencies": "text", "phase_relaxation": "text",
    "phase_raman": "text", "phase_ir": "text", "phase_es": "text",
    "ir_route": "text",
    "ir_fd_step_ang": _number(
        "ir_fd_step_ang", "given", "angstrom",
        "the infrared's finite-difference step",
        "each free atom moved by it either way, where the dipole's "
        "derivatives were taken by difference (engines/vibration.md 4.6)"),
    "raman_route": "text",
    "raman_fd_step_ang": _number(
        "raman_fd_step_ang", "given", "angstrom",
        "the Raman's finite-difference step",
        "the step the polarizability's derivatives were taken over, where "
        "they were taken by difference (engines/vibration.md 4.6)"),
    "hessian_scope": "text", "n_atoms_in_hessian": "int?",
    "hessian_density_fit": "flag?",
    # THE OPTICAL CALLS' CITATION (v13): the data file that holds them.
    "optical": "json",
}
#: The ``relaxation`` block, as ``relaxation_<key>`` attributes and
#: variables.
_RELAXATION = {
    "enabled": "flag?", "already_relaxed": "flag?", "n_steps": "int",
    "max_force_eh_bohr": _number(
        "relaxation_max_force_eh_per_bohr", "raw", "Eh/bohr",
        "the largest force component on a free atom",
        "the largest absolute Cartesian component, at the geometry the "
        "Hessian is taken at -- the force the relaxation is judged by "
        "(engines/vibration.md 4.3)",
        source="the engine's forces at that geometry",
        earlier=("relaxation_max_force_eh_bohr",)),
    "max_force_all_atoms_eh_bohr": _number(
        "relaxation_max_force_all_atoms_eh_per_bohr", "raw", "Eh/bohr",
        "the largest force component on any atom",
        "the largest absolute Cartesian component, held atoms included, "
        "at the same geometry", source="the engine's forces at that geometry",
        earlier=("relaxation_max_force_all_atoms_eh_bohr",)),
    "converged": "flag?", "warning": "text?", "record": "json",
}
#: The ``thermo`` block but its grid, as attributes and variables of
#: ``/thermo``.
_THERMO = {
    "regime": "text",
    "temperature_K": _number("temperature_k", "given", "K",
                             "the temperature",
                             "the one the headline values are at",
                             earlier=("temperature_K",)),
    "pressure_atm": _number("pressure_atm", "given", "atm", "the pressure",
                            "of the gas-phase translational term; NaN where "
                            "the regime has none"),
    "zpe_eh": _number("zpe_eh", "derived", "Eh", "the zero-point energy",
                      "the harmonic ZPE"),
    "u_vib_eh": _number("u_vib_eh", "derived", "Eh",
                        "the vibrational thermal energy",
                        "U_vib at the temperature, the zero-point energy "
                        "excluded"),
    "h_eh": _number("h_eh", "derived", "Eh", "the enthalpy, as the regime "
                    "sums it",
                    "rrho: H at the temperature; vibrational-only: the "
                    "vibrational sums above the electronic energy, no kT -- "
                    "not an enthalpy, as the note says (engines/vibration.md "
                    "4.7)"),
    "s_eh_k": _number("s_eh_per_k", "derived", "Eh/K", "the entropy",
                      "S at the temperature, as the regime defines it",
                      earlier=("s_eh_k",)),
    "g_eh": _number("g_eh", "derived", "Eh", "the free energy",
                    "G = H - T*S at the temperature"),
    "n_modes": "int",
    "n_imag_excluded": "int", "n_rigid_removed": "int", "note": "text",
}
#: ``/equilibrium``'s block: its energy, a variable (an attribute before
#: @8), and its HOMO's index, an attribute (:data:`_ROOT`'s form).
_EQUILIBRIUM = {
    "scf_energy_eh": _number(
        "total_energy_eh", "raw", "Eh", "the total energy",
        "the converged SCF's total energy, E_SCF, at the equilibrium "
        "geometry; NaN where the engine reports none -- the orbital "
        "energies with it", source="the equilibrium SCF",
        earlier=("scf_energy_eh",)),
    "homo_idx": ("int?", "homo_index"),
}
#: ``/partial_charges``' block: the force-constant step, a variable (an
#: attribute before @8).
_CHARGES = {
    "displacement_ang": _number(
        "fc_displacement_ang", "given", "angstrom",
        "the force-constant step's displacement",
        "each step but 0 moves its atom by it, either way, along its axis",
        earlier=("displacement_ang",)),
}
#: The geometries the probe computes at: equilibrium and either way along
#: the mode.
_PROBED_AT = (
    ("eq", "equilibrium", "at the equilibrium geometry"),
    ("minus", "-amplitude", "displaced by -amplitude along the mode"),
    ("plus", "+amplitude", "displaced by +amplitude along the mode"),
)
#: The thermo grid's curves (`normal_modes.vibrational_thermo_grid`),
#: each from its key in the grid block.
_GRID_CURVES = (
    Var("zpe_eh", ("temperature_k",), "derived", "Eh",
        "the zero-point energy", "the harmonic ZPE at this temperature"),
    Var("u_vib_eh", ("temperature_k",), "derived", "Eh",
        "the vibrational thermal energy",
        "U_vib at this temperature, the zero-point energy excluded"),
    Var("h_eh", ("temperature_k",), "derived", "Eh",
        "the enthalpy, as the regime sums it",
        "at this temperature: rrho H; vibrational-only the vibrational sums "
        "above the electronic energy, not an enthalpy (engines/vibration.md "
        "4.7)"),
    Var("s_eh_per_k", ("temperature_k",), "derived", "Eh/K", "the entropy",
        "S at this temperature, as the regime defines it", key="s_eh_k",
        earlier=("s_eh_k",)),
    Var("g_eh", ("temperature_k",), "derived", "Eh", "the free energy",
        "G = H - T*S at this temperature"),
)
#: A mode's electronic-structure probe (`engines/vibration.md` § 4.8):
#: its numbers and its orbital windows, each from its key in the probe's
#: block -- a name otherwise than the key the name an older file wrote.
_ES_SCALARS = (
    Var("amplitude_ang", ("probed_mode",), "given", "angstrom",
        "the displacement amplitude",
        "the push each way along the mode's display form: its largest "
        "Cartesian component moves by it (engines/vibration.md 4.8)"),
    *(Var(f"total_energy_{at}_eh", ("probed_mode",), "raw", "Eh",
          f"the total energy at {where}", f"E_SCF {geometry}",
          source="the probe's SCFs", key=f"scf_energy_{at}_eh",
          earlier=(f"scf_energy_{at}_eh",))
      for at, where, geometry in _PROBED_AT),
)
_ES_WINDOWS = tuple(
    Var(f"mo_energy_{at}_eh", ("probed_mode", "window"), "raw", "Eh",
        f"the orbital energies at {where}",
        "the orbital window's energies, ascending",
        source="the probe's SCFs", key=f"mo_energies_{at}_eh",
        earlier=(f"mo_energies_{at}_eh",))
    for at, where, _ in _PROBED_AT)


def _encode(kind: str, value):
    if kind in ("text", "text?"):
        return "" if value is None else str(value)
    if kind == "int":
        return int(value)
    if kind == "int?":
        return _INT_FILL if value is None else int(value)
    if kind == "flag?":
        return np.int8(flag(value))
    if kind == "json":
        return json_attr(value)
    raise SpectraDataError(f"no encoding for kind {kind!r}")


def _decode(kind: str, value):
    if kind == "text":
        return str(value)
    if kind == "text?":
        return str(value) or None
    if kind == "int":
        return int(value)
    if kind == "int?":
        v = int(value)
        return None if v == _INT_FILL else v
    if kind == "flag?":
        return unflag(value)
    if kind == "json":
        return unjson_attr(value)
    raise SpectraDataError(f"no encoding for kind {kind!r}")


def _block_attrs(block: Mapping[str, Any], table: Mapping[str, Any],
                 prefix: str, where: str) -> Dict[str, Any]:
    """``block``'s keys but its numbers as typed attributes, ``prefix +
    key``; a key the table does not know is refused by name."""
    said = unknown_keys(block, table, note=UNSTATED)
    if said:
        raise SpectraDataError(f"{where}: {said}")
    return {prefix + _attr_name(k, table[k]): _encode(_kind(table[k]),
                                                      block[k])
            for k in block if not isinstance(table[k], Var)}


def _kind(entry) -> str:
    """A block table's encoding of an attribute (:data:`_ROOT`)."""
    return entry[0] if isinstance(entry, tuple) else entry


def _attr_name(key: str, entry) -> str:
    """The name the file gives the attribute of result key ``key``."""
    return entry[1] if isinstance(entry, tuple) else key


def _block_numbers(block: Mapping[str, Any], table: Mapping[str, Any]
                   ) -> Tuple[Tuple[Var, ...], Dict[str, np.ndarray]]:
    """``block``'s numbers as their variables and values -- each key the
    block holds."""
    held = [(k, v) for k, v in table.items()
            if isinstance(v, Var) and k in block]
    return (tuple(v for _, v in held),
            {v.name: np.asarray(number_attr(block[k])) for k, v in held})


def _numbers(table: Mapping[str, Any]) -> Tuple[Var, ...]:
    """A block table's numbers -- its variables."""
    return tuple(v for v in table.values() if isinstance(v, Var))


def _block_back(attrs: Mapping[str, Any], held: Mapping[str, Any],
                table: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """The block back from a group's attributes and its variables
    ``held`` -- read with the table's numbers (`ncdata.read_variables`) --
    each under its name now (`ncdata.named_now`): a number from its
    variable; an attribute by its name, one the file names otherwise than
    the result's key named by the key before ``@10``."""
    attrs = named_now(attrs, {prefix + kind[1]: (prefix + k,)
                              for k, kind in table.items()
                              if isinstance(kind, tuple)})
    out: Dict[str, Any] = {}
    for k, kind in table.items():
        if isinstance(kind, Var):
            if kind.name in held:
                out[k] = unnumber_attr(held[kind.name]["data"])
        elif prefix + _attr_name(k, kind) in attrs:
            out[k] = _decode(_kind(kind), attrs[prefix + _attr_name(k, kind)])
    return out


# --------------------------------------------------------------------- #
#  The variables                                                          #
# --------------------------------------------------------------------- #

def _partial_charge_vars(partitions: Sequence[str]) -> Tuple[Var, ...]:
    """``/partial_charges`` -- the force-constant run's charges
    (`engines/vibration.md` § 5.10 ②): the steps, and every atom's charge at
    each, one variable per partition the result states."""
    return (
        Var("fc_step", ("fc_step",), "coordinate", "1",
            "the force-constant step",
            "from 0: 0 the reference geometry, then each displacement in "
            "SIESTA's order (engines/vibration.md 5.4)", dtype="i4"),
        coordinate("atom"),
        Var("displaced_atom", ("fc_step",), "given", "1",
            "the atom the step displaces",
            "its user index, from 0; -1 at step 0, which displaces none",
            dtype="i4", fill=-1, atom_index=True),
        Var("displaced_axis", ("fc_step",), "given", "",
            "the axis the step displaces along",
            "x, y or z; empty at step 0", dtype=str),
        Var("displaced_side", ("fc_step",), "given", "1",
            "the side of the displacement",
            "-1 or +1, the step fc_displacement_ang along the axis; 0 at "
            "step 0",
            dtype="i4"),
        *(Var(f"{name}_charge_e", ("fc_step", "atom"), "raw", "e",
              f"the atom's {name.capitalize()} charge",
              "its valence charge less the electrons the partition gives "
              "it -- positive where electrons left it -- at each step",
              source=f"the run's output, its {name.capitalize()} Atomic "
                     f"Populations blocks")
          for name in partitions),
    )


def _equilibrium_vars() -> Tuple[Var, ...]:
    """``/equilibrium``'s orbital energies -- its energy and HOMO are its
    block's (:data:`_EQUILIBRIUM`)."""
    return (Var("mo", ("mo",), "coordinate", "1", "the orbital's index",
                "from 0, ascending energy", dtype="i4"),
            Var("mo_energy_eh", ("mo",), "raw", "Eh", "the orbital energies",
                "every MO energy at the equilibrium SCF, ascending",
                source="the equilibrium SCF"),)


def _mode_vars() -> Tuple[Var, ...]:
    return (
        replace(coordinate("mode"), earlier=("index_1based",)),
        coordinate("xyz"),
        Var("removed", ("removed",), "coordinate", "1",
            "the removed motion's index", "from 0", dtype="i4"),
        Var("frequency_cm1", ("mode",), "derived", "cm-1", "the frequency",
            "the harmonic frequency; negative for an imaginary mode"),
        Var("free_atom", ("free_atom",), "coordinate", "1",
            "which atom each row is",
            "the 0-based atom index, in the user order, of each free-atom "
            "row of the eigenvectors (free_atom_idxs) -- counting by the "
            "file's atom index map, in /structure", dtype="i4",
            earlier=("free_atom_index",),
            atom_index=True),
        Var("eigenvector_canonical_per_amu12", ("mode", "free_atom", "xyz"),
            "derived", "amu^-1/2", "the Cartesian eigenvector",
            "normalised so that the sum over atoms of m |L|^2 is 1, m in "
            "amu: the form intensities and amplitudes pair with "
            "(engines/vibration.md 6.3)", key="eigenvector_canonical",
            earlier=("eigenvector_canonical",)),
        Var("eigenvector_display", ("mode", "free_atom", "xyz"), "derived",
            "1", "the eigenvector for drawing",
            "scaled so its largest Cartesian component is 1 "
            "(engines/vibration.md 6.3)"),
        Var("zero_point_amplitude_amu12_ang", ("mode",), "derived",
            "amu^1/2 angstrom", "the zero-point amplitude",
            "sqrt(hbar / (2 omega)) in mass-weighted coordinates "
            "(engines/vibration.md 6.6); NaN for an imaginary mode"),
        Var("zero_point_displacement_ang", ("mode", "free_atom", "xyz"),
            "derived", "angstrom", "the zero-point displacement",
            "each atom's displacement at the zero-point amplitude "
            "(engines/vibration.md 6.6); NaN for an imaginary mode"),
        Var("ir_intensity_km_per_mol", ("mode",), "derived", "km/mol",
            "the IR intensity", "the orientation average of |dmu/dQ|^2, "
            "in km/mol (constants.IR_KM_MOL_PER_D2_A2_AMU; "
            "engines/vibration.md 4.6); NaN where not computed",
            key="ir_intensity_km_mol", earlier=("ir_intensity_km_mol",)),
        Var("raman_activity_ang4_per_amu", ("mode",), "derived",
            "angstrom^4/amu", "the Raman activity",
            "Placzek's orientation-averaged activity "
            "45 a'^2 + 7 gamma'^2 from the polarizability's derivatives "
            "along the mode (engines/vibration.md 4.6); NaN where not "
            "computed", key="raman_activity_a4_amu",
            earlier=("raman_activity_a4_amu",)),
        Var("dipole_derivative_d_per_ang_amu12", ("mode", "xyz"), "derived",
            "D/(angstrom amu^1/2)", "the dipole change along the mode",
            "dmu/dQ: the engine's dipole derivatives dmu/dR projected on "
            "the canonical eigenvector -- the vector the IR intensity is "
            "the orientation average of (engines/vibration.md § 4.6); NaN "
            "where none was kept"),
        Var("ir_active", ("mode",), "derived", "1", "whether IR active",
            "the activity class's IR verdict (engines/vibration.md 6.6); "
            "-1 where not computed",
            dtype="i1", flag=True),
        Var("raman_active", ("mode",), "derived", "1", "whether Raman active",
            "the activity class's Raman verdict (engines/vibration.md "
            "6.6); -1 where not "
            "computed", dtype="i1", flag=True),
        Var("activity_class", ("mode",), "derived", "", "the activity class",
            "the whole-run judgement (engines/vibration.md 6.6): both, "
            "ir-only, raman-only, silent, or partial -- a channel not "
            "computed, so the mode cannot be placed", dtype=str),
        Var("has_imag", ("mode",), "derived", "1", "whether imaginary",
            "1 for an imaginary frequency", dtype="i1", flag=True),
        Var("removed_motion", ("removed", "free_atom", "xyz"), "derived", "1",
            "a removed whole-body motion",
            "an orthonormal Cartesian pattern over the free atoms removed "
            "before diagonalising (removed_motions.patterns)"),
    )


def _es_vars() -> Tuple[Var, ...]:
    return (
        Var("probed_mode", ("probed_mode",), "coordinate", "1",
            "the probed mode", "the 1-based index of the mode this row is of",
            dtype="i4", earlier=("es", "mode_index_1based")),
        Var("window", ("window",), "coordinate", "1",
            "the orbital's place in the window", "from 0, ascending energy",
            dtype="i4"),
        Var("homo_index_in_window", ("probed_mode",), "given", "1",
            "the HOMO's index in the window",
            "0-based index of the HOMO among the window's orbitals",
            dtype="i4"),
        *_ES_SCALARS, *_ES_WINDOWS,
    )


#: The thermo grid's axis, from its key in the grid block.
_GRID_TEMPERATURE = Var(
    "temperature_k", ("temperature_k",), "coordinate", "K", "the temperature",
    "each temperature the curves are drawn at", key="temperatures_K",
    earlier=("temperature", "temperatures_K"))


def _grid_vars() -> Tuple[Var, ...]:
    return (_GRID_TEMPERATURE, *_GRID_CURVES)


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
    said = unknown_keys(payload, (
        *_ROOT, "schema_version", "free_atom_idxs", "frozen_atom_idxs",
        "structure", "equilibrium", "modes", "relaxation", "thermo",
        "removed_motions", MAP_KEY, "partial_charges"), note=UNSTATED)
    if said:
        raise SpectraDataError(f"the result: {said}")
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

    top = {k: payload[k] for k in _ROOT if k in payload}

    # ONE NETCDF ACCESS AT A TIME IN THE PROCESS (`ncdata.NC_LOCK`).
    with NC_LOCK:
        ds = in_memory(p.name)
        state(ds, title=f"the vibrational analysis of {label}: its normal "
                        f"modes, their strengths, its thermochemistry and "
                        f"the geometry it is about -- the result the JSON "
                        f"beside it holds, its bytes' sha256 json_sha256",
              schema=DATA_SCHEMA,
              molbuilder_version=str(payload.get("molbuilder_version") or ""),
              engine=str(payload.get("engine") or ""),
              engine_version=str(payload.get("engine_version") or ""))
        attrs: Dict[str, Any] = {
            "json_sha256": str(json_sha256),
            "spectra_schema_version": int(payload["schema_version"]),
            "label": str(label),
            **{k: v for k, v in _block_attrs(top, _ROOT, "",
                                              "the result").items()
               if k not in ds.ncattrs()},
            **_block_attrs(relaxation, _RELAXATION, "relaxation_",
                           "the relaxation block"),
        }
        if "count" in removed:
            attrs["removed_motions_count"] = int(removed["count"])
        ds.setncatts(attrs)
        numbers, values = _block_numbers(top, _ROOT)
        more, more_values = _block_numbers(relaxation, _RELAXATION)
        write_group(ds, {}, numbers + more, {**values, **more_values})

        # /structure -- the arrays an analysis reaches for, and the structure
        # whole as its one serialiser states it; THE ATOM AXIS AS ITS MAP
        # (`model/overview.md` § 2.2b): the result's own, listed by user
        # index.
        if MAP_KEY not in payload:
            raise SpectraDataError(
                "the result states no atom_index_map: its data file says "
                "which index its atoms are listed by")
        atoms = AtomIndexMap.from_dict(payload[MAP_KEY],
                                       n_atoms=n_atoms,
                                       source="the result's atom_index_map")
        write_structure_group(ds, structure, atoms,
                              geometry="the geometry the vibration is "
                                       "about",
                              masses=eq.get("masses_amu"),
                              held=payload.get("frozen_atom_idxs") or ())

        # /equilibrium -- the energy subgroup, whole or not at all (§ 6.2).
        g = ds.createGroup("equilibrium")
        mo = eq.get("mo_energies_eh")
        block = {k: eq.get(k) for k in _EQUILIBRIUM}
        g.setncatts(_block_attrs(block, _EQUILIBRIUM, "", "the equilibrium"))
        numbers, held = _block_numbers(block, _EQUILIBRIUM)
        vars_ = numbers + _equilibrium_vars()
        values = {**_fill(vars_, {"mo": len(mo or ())}), **held}
        values["mo"][:] = np.arange(len(mo or ()))
        if mo:
            values["mo_energy_eh"][:] = np.asarray(mo, dtype=float)
        write_group(g, {"mo": len(mo or ())}, vars_, values,
                    title="the equilibrium SCF: its energy, its orbital "
                          "energies and the HOMO's index (homo_index, from "
                          "0; -1 where not stated)")

        # /modes
        g = ds.createGroup("modes")
        n_free = len(free)
        dims = {"mode": len(modes), "free_atom": n_free,
                "removed": len(patterns), "xyz": 3}
        vars_ = tuple(MODE.values())
        values = _fill(vars_, dims)
        values["free_atom"][:] = free
        values["xyz"][:] = axis_values("xyz", 3)
        values["removed"][:] = np.arange(len(patterns))
        for k, m in enumerate(modes):
            values["mode"][k] = int(m["index_1based"])
            values["frequency_cm1"][k] = float(m["frequency_cm1"])
            values["eigenvector_canonical_per_amu12"][k] = np.asarray(
                m[key_of(MODE["eigenvector_canonical_per_amu12"])],
                dtype=float)
            values["eigenvector_display"][k] = np.asarray(
                m["eigenvector_display"], dtype=float)
            values["has_imag"][k] = flag(m.get("has_imag"))
            for name in ("ir_intensity_km_per_mol",
                         "raman_activity_ang4_per_amu",
                         "zero_point_amplitude_amu12_ang"):
                values[name][k] = number_attr(m.get(key_of(MODE[name])))
            if m.get("zero_point_displacement_ang") is not None:
                values["zero_point_displacement_ang"][k] = np.asarray(
                    m["zero_point_displacement_ang"], dtype=float)
            if m.get("dipole_derivative_d_per_ang_amu12") is not None:
                values["dipole_derivative_d_per_ang_amu12"][k] = np.asarray(
                    m["dipole_derivative_d_per_ang_amu12"], dtype=float)
            values["ir_active"][k] = flag(m.get("ir_active"))
            values["raman_active"][k] = flag(m.get("raman_active"))
            values["activity_class"][k] = str(m.get("activity_class") or "")
        for r, pattern in enumerate(patterns):
            values["removed_motion"][r] = np.asarray(pattern, dtype=float)
        write_group(g, dims, vars_, values,
                    title="the normal modes: each one's frequency, "
                          "eigenvectors, zero-point motion, strengths and "
                          "activity, and the whole-body motions removed "
                          "before diagonalising")

        # /modes/electronic_structure -- one row per probed mode.
        probed = [m for m in modes
                  if m.get("electronic_structure") is not None]
        # ONE WINDOW, its size the equilibrium's.
        windows = {len(m["electronic_structure"][key_of(_ES_WINDOWS[0])])
                   for m in probed}
        if len(windows) > 1:
            raise SpectraDataError(
                f"the probed modes' orbital windows differ in size "
                f"({sorted(windows)}): one window per result")
        window = windows.pop() if windows else 0
        ge = g.createGroup("electronic_structure")
        vars_ = _es_vars()
        values = _fill(vars_, {"probed_mode": len(probed), "window": window})
        values["window"][:] = np.arange(window)
        for k, m in enumerate(probed):
            es = m["electronic_structure"]
            values["probed_mode"][k] = int(m["index_1based"])
            values["homo_index_in_window"][k] = int(es["homo_index_in_window"])
            for v in _ES_SCALARS:
                values[v.name][k] = float(es[key_of(v)])
            for v in _ES_WINDOWS:
                values[v.name][k] = np.asarray(es[key_of(v)], dtype=float)
        write_group(ge, {"probed_mode": len(probed), "window": window}, vars_,
                    values,
                    title="the electronic-structure probe: per probed mode, "
                          "the SCF energy and an orbital window at the "
                          "equilibrium and displaced either way along it")

        # /thermo, and its grid -- an axis of no temperature where the
        # result draws no curves.
        g = ds.createGroup("thermo")
        g.setncatts(_block_attrs(thermo, _THERMO, "", "the thermo block"))
        numbers, values = _block_numbers(thermo, _THERMO)
        write_group(g, {}, numbers, values,
                    title="the harmonic thermochemistry at one temperature, "
                          "as its regime defines it")
        gg = g.createGroup("grid")
        said = unknown_keys(grid, [key_of(v) for v in _grid_vars()],
                            note=UNSTATED)
        if said:
            raise SpectraDataError(f"the thermo grid: {said}")
        temps = [float(t) for t in grid.get(key_of(_GRID_TEMPERATURE))
                 or ()]
        vars_ = _grid_vars()
        values = _fill(vars_, {"temperature_k": len(temps)})
        values["temperature_k"][:] = temps
        for v in _GRID_CURVES:
            if key_of(v) in grid:
                values[v.name][:] = np.asarray(grid[key_of(v)], dtype=float)
        write_group(gg, {"temperature_k": len(temps)}, vars_, values,
                    title="the thermochemistry drawn over temperature; "
                          "curves names the ones the result holds")
        if grid:
            gg.setncattr("curves", json_attr([v.name for v in _GRID_CURVES
                                              if key_of(v) in grid]))

        # /partial_charges -- the force-constant run's, when the result
        # states them (`engines/vibration.md` § 5.10 ②).
        charges = payload.get("partial_charges")
        if charges is not None:
            g = ds.createGroup("partial_charges")
            steps = list(charges["steps"])
            partitions = list(charges["charges_e"])
            numbers, held = _block_numbers(
                {k: charges[k] for k in _CHARGES}, _CHARGES)
            vars_ = numbers + _partial_charge_vars(partitions)
            dims = {"fc_step": len(steps), "atom": atoms}
            values = {**_fill(vars_, {"fc_step": len(steps), "atom": n_atoms}),
                      **held}
            values["fc_step"][:] = np.arange(len(steps))
            values["atom"][:] = axis_values("atom", n_atoms)
            for i, s in enumerate(steps):
                values["displaced_atom"][i] = (-1 if s["atom"] is None
                                               else int(s["atom"]))
                values["displaced_axis"][i] = str(s["axis"] or "")
                values["displaced_side"][i] = int(s["side"])
            for name in partitions:
                values[f"{name}_charge_e"][:] = np.asarray(
                    charges["charges_e"][name], dtype=float)
            write_group(g, dims, vars_, values,
                        title="the force-constant run's atomic charges at "
                              "each of its steps, one variable per "
                              "partition")
            g.setncatts({"partitions": json_attr(partitions)})
        data = as_bytes(ds)
    write_bytes(p, data)
    return p


# --------------------------------------------------------------------- #
#  Read                                                                   #
# --------------------------------------------------------------------- #

def read_spectra_data(path, *, data=None) -> Dict[str, Any]:
    """``<label>.spectra.nc`` back as the payload the JSON holds -- the
    file's door (`runfiles.WRITTEN`); a schema it does not read
    (:data:`READABLE_DATA_SCHEMAS`) refused by name."""
    out: Dict[str, Any] = {}
    with checked(path, schemas=READABLE_DATA_SCHEMAS, error=SpectraDataError,
                 data=data) as (ds, attrs):
        out["schema_version"] = int(attrs["spectra_schema_version"])
        held = read_variables(ds, _numbers(_ROOT) + _numbers(_RELAXATION))
        out.update(_block_back(attrs, held, _ROOT, ""))
        relaxation = _block_back(attrs, held, _RELAXATION, "relaxation_")
        out["relaxation"] = relaxation
        # /structure, and THE ATOM INDEX MAP the file states of it
        # (§ 2.2b), back as the result's own block.
        st = read_structure_group(ds["structure"],
                                  source=f"{Path(path).name}/structure")
        structure, held_flags, masses = (st["structure"], st["held"],
                                         st["masses"])
        if structure is not None:
            out["structure"] = structure
        else:
            out["frozen_atom_idxs"] = [int(i) for i, h in enumerate(held_flags)
                                       if int(h) == 1]
        if st["atoms"] is not None:
            out[MAP_KEY] = st["atoms"].to_dict()
        # /equilibrium
        g = ds["equilibrium"]
        qv = read_variables(g, _numbers(_EQUILIBRIUM))
        eq: Dict[str, Any] = _block_back(
            {k: g.getncattr(k) for k in g.ncattrs()}, qv, _EQUILIBRIUM, "")
        eq["mo_energies_eh"] = (
            None if eq.get("scf_energy_eh") is None
            else [float(x) for x in qv["mo_energy_eh"]["data"]])
        if masses is not None:
            eq["masses_amu"] = masses
        out["equilibrium"] = eq
        # /modes
        g = ds["modes"]
        mv = read_variables(g, tuple(MODE.values()))
        out["free_atom_idxs"] = [int(i) for i in mv["free_atom"]["data"]]
        n_modes = int(mv["mode"]["data"].shape[0])
        es_rows: Dict[int, Dict[str, Any]] = {}
        ev = read_variables(g["electronic_structure"], _es_vars())
        for k, idx in enumerate(ev["probed_mode"]["data"]):
            es_rows[int(idx)] = {
                **{key_of(v): [float(x) for x in ev[v.name]["data"][k]]
                   for v in _ES_WINDOWS},
                "homo_index_in_window": int(
                    ev["homo_index_in_window"]["data"][k]),
                **{key_of(v): float(ev[v.name]["data"][k])
                   for v in _ES_SCALARS},
            }
        modes = []
        for k in range(n_modes):
            idx = int(mv["mode"]["data"][k])
            # THE DERIVATION'S RULE (`spectra.derived`): an imaginary mode
            # has no zero-point amplitude and no displacement.
            amplitude = unnumber_attr(
                mv["zero_point_amplitude_amu12_ang"]["data"][k])
            m: Dict[str, Any] = {
                "index_1based": idx,
                "frequency_cm1": float(mv["frequency_cm1"]["data"][k]),
                **{key_of(MODE[name]): unnumber_attr(
                    mv[name]["data"][k])
                   for name in ("raman_activity_ang4_per_amu",
                                "ir_intensity_km_per_mol")},
                key_of(MODE["eigenvector_canonical_per_amu12"]):
                    mv["eigenvector_canonical_per_amu12"]["data"][k].tolist(),
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
            # THE DIPOLE CHANGE, where the file keeps the variable (@6): the
            # fill value says the engine kept none.
            if "dipole_derivative_d_per_ang_amu12" in mv:
                v = np.asarray(
                    mv["dipole_derivative_d_per_ang_amu12"]["data"][k],
                    dtype=float)
                m["dipole_derivative_d_per_ang_amu12"] = (
                    None if not np.all(np.isfinite(v)) else v.tolist())
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
                             read_variables(g, _numbers(_THERMO)), _THERMO,
                             "")
        gg = g["grid"]
        if "curves" in gg.ncattrs():
            gv = read_variables(gg, _grid_vars())
            grid = {key_of(_GRID_TEMPERATURE): [float(t) for t in
                                       gv["temperature_k"]["data"]]}
            # THE CURVES THE FILE NAMES, each under its name now
            # (`ncdata.named_now`).
            named = named_now(
                dict.fromkeys(unjson_attr(gg.getncattr("curves"))),
                {v.name: v.earlier for v in _GRID_CURVES})
            for v in _GRID_CURVES:
                if v.name in named:
                    grid[key_of(v)] = [float(x) for x in gv[v.name]["data"]]
            thermo["grid"] = grid
        out["thermo"] = thermo
        # /partial_charges, where the file holds them (from @5).
        out["partial_charges"] = None
        if "partial_charges" in ds.groups:
            g = ds["partial_charges"]
            pv = read_variables(g, _numbers(_CHARGES))
            partitions = unjson_attr(g.getncattr("partitions"))
            steps = []
            for i in range(int(pv["fc_step"]["data"].shape[0])):
                atom = int(pv["displaced_atom"]["data"][i])
                steps.append({"atom": None if atom < 0 else atom,
                              "axis": str(pv["displaced_axis"]["data"][i])
                              or None,
                              "side": int(pv["displaced_side"]["data"][i])})
            out["partial_charges"] = {
                **_block_back({k: g.getncattr(k) for k in g.ncattrs()}, pv,
                              _CHARGES, ""),
                "steps": steps,
                "charges_e": {name: pv[f"{name}_charge_e"]["data"].tolist()
                              for name in partitions}}
    return out


__all__ = ["DATA_SCHEMA", "SpectraDataError", "UNSTATED", "write_spectra_data",
           "read_spectra_data"]


#: The modes' variables by name -- what the writer and the reader ask a
#: variable's record key of (`ncdata.key_of`).
MODE = {v.name: v for v in _mode_vars()}
