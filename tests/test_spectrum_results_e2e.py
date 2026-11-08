"""What a vibration's result is, asked of results real runs wrote
(`engines/vibration.md` § 6; `process/testing.md` § 6).

A result is what a run derives, so a test of what its reader, its data file or
the load door make of one reads a result a run wrote, here, in this pass
(`support.real_runs`): every state a real run's file passes through, each a row
of :data:`_PYSCF` or :data:`_SIESTA` -- a run's own file once it ended, and,
while it went, the version of it kept as the run wrote it
(`RealRun.versions`).

The reader's refusals of a file edited by hand are not asked (`process/
testing.md` § 3b, § 6): the one rule shown is a file of another version, what
the writer writes with its version moved
(:func:`test_a_file_of_another_version_is_refused_by_that_version`).  What the
rest of `engines/vibration.md` § 6.7 refuses is asked from the other side: what
the writers write is what those rules call correct.  WHAT NO RUN WRITES GOES
UNSEEN HERE: those refusals themselves; modes with no geometry, which the load
door would serve without their shares; a file older than this release's writers.

What the runs' science says -- frequencies, intensities against their windows,
the bends' degeneracy, the thermochemistry -- is `test_vibration_e2e.py`'s and
`test_siesta_vibration_e2e.py`'s, on these same runs.
"""
from __future__ import annotations

import hashlib
import json
import math
from unittest.mock import ANY

import numpy as np
import pytest

from _road import conda_hook, env_available
from molbuilder.runfiles import (ROLE_SPECTRA, ROLE_SPECTRA_DATA, compose,
                                 stem_before)
from molbuilder.runs import on_label_at
from molbuilder.sidecars.spectra import (SpectraJsonSchemaError,
                                         dump_spectra_json, parse_spectra_json,
                                         write_spectra_payload)
from molbuilder.spectra.datafile import read_spectra_data
from molbuilder.spectra.results import SCHEMA_VERSION
from molbuilder.structure import Structure
from support.pages import spectra_served

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-pySCF")
             and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-pySCF and molbuilder-siesta envs + a "
               "detectable conda hook"),
]


def _read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _final(run):
    """The result the run's own folder holds now it ended, by the one door
    for a result on the label."""
    return on_label_at(run.attempt("freq"), ROLE_SPECTRA)


def _data_file(result):
    """The data file beside ``result``, by the name the catalogue gives it."""
    return result.with_name(compose(stem_before(result.name, ROLE_SPECTRA),
                                    ROLE_SPECTRA_DATA))


def _column(modes, key):
    """A strength's column, as `engines/vibration.md` § 6.5 tells them apart:
    ``null`` on every mode, or a number on every mode; ``None`` with no mode
    yet.  A column mixing the two is neither, and fails its row."""
    if not modes:
        return None
    stated = {m[key] is None for m in modes}
    return {frozenset({True}): "null",
            frozenset({False}): "numbers"}.get(frozenset(stated), "mixed")


def _held(d) -> list:
    """The held atoms where the file states them: its structure's label, or,
    before any geometry exists, a list of their own (§ 6.10.2)."""
    if "structure" in d:
        assert "frozen_atom_idxs" not in d, "the held atoms stated twice"
        return Structure.from_dict(d["structure"]).frozen_atoms or []
    return d["frozen_atom_idxs"]


def _facts(d) -> dict:
    """What a result states of its state, in the terms of
    `engines/vibration.md` § 4.9, § 6.2 and § 6.5."""
    eq, modes = d["equilibrium"], d["modes"]
    energies = {eq[k] is None
                for k in ("scf_energy_eh", "mo_energies_eh", "homo_idx")}
    assert len(energies) == 1, "the energy subgroup travels whole or not at all"
    return {
        "phases": {p: d[f"phase_{p}"]
                   for p in ("relaxation", "frequencies", "raman", "ir", "es")},
        "geometry": "structure" in d,
        "held": _held(d),
        "energies": not energies.pop(),
        "masses": "masses_amu" in eq,
        "modes": len(modes),
        "infrared": _column(modes, "ir_intensity_km_mol"),
        "raman": _column(modes, "raman_activity_a4_amu"),
        "probed": [m["index_1based"] for m in modes
                   if m["electronic_structure"] is not None],
        "thermo": d["thermo"].get("regime"),
        "removed": d["removed_motions"].get("count"),
        "routes": (d["ir_route"], d["raman_route"]),
        "charges": d["partial_charges"] is not None,
        "optical": d["optical"] is not None,
    }


def _phases(relaxation, frequencies, raman, ir, es) -> dict:
    return {"relaxation": relaxation, "frequencies": frequencies,
            "raman": raman, "ir": ir, "es": es}


_DONE, _RUNNING, _EMPTY, _NOT = "complete", "running", "empty", "not requested"

#: A file before its geometry, as § 6.5 says it reads from its first write:
#: `null` where a phase has given no value yet, `none` for a route, empty for
#: the blocks.
_FIRST = {"geometry": False, "energies": False, "masses": False, "modes": 0,
          "infrared": None, "raman": None, "probed": [], "thermo": None,
          "removed": None, "routes": ("none", "none"), "charges": False,
          "optical": False}
#: A PySCF file at its end but for what its row says: the geometry, its
#: energies and masses, a free molecule's modes and thermochemistry.
_PYSCF_END = {"geometry": True, "energies": True, "masses": True, "held": [],
              "modes": 3, "infrared": "null", "raman": "null", "probed": [],
              "thermo": "rrho", "removed": 6, "routes": ("none", "none"),
              "charges": False, "optical": False}
#: The SIESTA run's file, once its force constants are recorded (§ 6.5): no
#: energy subgroup, no strength on any mode, its carbon held -- two turns about
#: it removed -- its charges and its optical citation stated.
_SIESTA_END = {"geometry": True, "energies": False, "masses": True, "held": [2],
               "modes": 4, "infrared": "null", "raman": "null", "probed": [],
               "thermo": "vibrational-only", "removed": 2,
               "routes": ("charges", "optical"), "charges": True,
               "optical": True}


#: EVERY STATE A PYSCF RUN'S FILE PASSES THROUGH, a row each: the run, the
#: version of its file, and what it states there.
_PYSCF = {
    "no geometry yet": (
        "held", lambda d: d["phase_relaxation"] == _RUNNING,
        {**_FIRST, "held": [0],
         "phases": _phases(_RUNNING, _EMPTY, _NOT, _EMPTY, _NOT)}),
    "a geometry and no modes yet": (
        "held", lambda d: d["phase_frequencies"] == _RUNNING,
        {**_PYSCF_END, "held": [0], "modes": 0, "infrared": None, "raman": None,
         "thermo": None, "removed": None,
         "phases": _phases(_DONE, _RUNNING, _NOT, _EMPTY, _NOT)}),
    "modes, and its infrared running": (
        "held", lambda d: d["phase_ir"] == _RUNNING,
        {**_PYSCF_END, "held": [0], "thermo": "vibrational-only", "removed": 3,
         "routes": ("finite-difference", "none"),
         "phases": _phases(_DONE, _DONE, _NOT, _RUNNING, _NOT)}),
    "ended, atoms held": (
        "held", None,
        {**_PYSCF_END, "held": [0], "infrared": "numbers",
         "thermo": "vibrational-only", "removed": 3,
         "routes": ("finite-difference", "none"),
         "phases": _phases(_DONE, _DONE, _NOT, _DONE, _NOT)}),
    "ended, Raman asked and no infrared": (
        "water", None,
        {**_PYSCF_END, "raman": "numbers",
         "routes": ("none", "finite-difference"),
         "phases": _phases(_DONE, _DONE, _DONE, _NOT, _NOT)}),
    "ended, infrared and the probe": (
        # WHICH ROUTE gave its infrared is the env's to say: test_vibration_e2e.
        "probed", None,
        {**_PYSCF_END, "infrared": "numbers", "probed": [1, 3],
         "routes": (ANY, "none"),
         "phases": _phases(_DONE, _DONE, _NOT, _DONE, _DONE)}),
    "ended, two modes imaginary": (
        "linear", None,
        {**_PYSCF_END, "modes": 4, "removed": 5,
         "phases": _phases(_NOT, _DONE, _NOT, _NOT, _NOT)}),
}

#: ...AND A SIESTA RUN'S, from the record step on (§ 5.11 § 3, § 6.2): nothing
#: is written before its force constants are.
_SIESTA = {
    "its optical calls to come": (
        lambda d: d["phase_raman"] == _RUNNING,
        {**_SIESTA_END, "phases": _phases(_DONE, _DONE, _RUNNING, _DONE, _NOT)}),
    "ended": (
        None,
        {**_SIESTA_END, "phases": _phases(_DONE, _DONE, _DONE, _DONE, _NOT)}),
}


class _EveryState:
    """What holds of a result at every state a real run's file passes through
    -- a row of a table each, ``state`` (the file, what it states there)."""

    def test_the_file_states_what_its_state_is(self, state):
        """§ 4.9, § 6.2, § 6.5: which phases are where, the held atoms in their
        one place, the energy subgroup whole or absent, `null` where a strength
        was not computed and a number where it was, the blocks and routes.
        THE SILENT FAILURE: a reader of the live file -- the poller, the
        phase dots -- told something else of a run still going, which nothing
        that reads a finished run can see."""
        path, states = state
        assert _facts(_read(path)) == states

    def test_read_and_written_again_it_is_the_same_file(self, state, tmp_path):
        """`spectra/results.py`: `from_dict(to_dict(x)) == x`, on what a run
        wrote: every key, every number bit for bit, every `null` a `null`.  It
        is what stands for each key read from its neighbour: a run's numbers
        are all different, so an orbital window at -A read as the one at +A,
        the canonical eigenvector read from the display form or a number
        rounded on read does not come back the same.  And the file is what the
        writer promises (§ 6.8) and the registry's own: no byte-order mark, its
        parser found by the one door (`model/parse.md` § 3) -- dropped from the
        registry, a result is listed with `parser: null` and the picker drops
        it, while every read here still reads it."""
        from molbuilder.parse import detect
        path, _states = state
        raw = path.read_bytes()
        assert raw[:1] == b"{", raw[:3]
        written = json.loads(raw.decode("utf-8"))
        again = tmp_path / path.name
        dump_spectra_json(parse_spectra_json(path), again)
        assert _read(again) == written
        assert detect(str(path)).name == "spectra-json"

    def test_the_reader_holds_the_geometry_and_energies_it_was_given(self, state):
        """The typed side (`spectra/results.py`), which every reader but the
        wire takes: the geometry a mode's eigenvectors belong to, under
        `equilibrium_elements`, from the file's one structure -- a displacement
        otherwise drawn on whatever structure is loaded -- and the reference
        energies as arrays, each `None` while the file states none."""
        path, _states = state
        d, got = _read(path), parse_spectra_json(path)
        if "structure" in d:
            assert "elements" not in d["equilibrium"]
            assert got.equilibrium_elements == d["structure"]["elements"]
            np.testing.assert_array_equal(got.equilibrium_positions_ang,
                                          d["structure"]["positions"])
        else:
            assert got.equilibrium_elements is None
            assert got.equilibrium_positions_ang is None
        if d["equilibrium"]["mo_energies_eh"] is None:
            assert got.equilibrium_mo_energies_eh is None
        else:
            assert isinstance(got.equilibrium_mo_energies_eh, np.ndarray)
            np.testing.assert_array_equal(got.equilibrium_mo_energies_eh,
                                          d["equilibrium"]["mo_energies_eh"])

    def test_the_atoms_it_is_about_are_the_structures(self, state):
        """§ 6.2, § 6.10.2: the free atoms are the complement of the held ones,
        ascending -- the order of every mode's rows -- one row each, and
        `len(modes) = 3 n_free - removed`.  THE SILENT FAILURE: the viewer
        scatters each row onto the atom the list names, so a list that does
        not partition the structure animates the wrong atoms."""
        path, _states = state
        d = _read(path)
        free = d["free_atom_idxs"]
        assert free == [i for i in range(d["n_atoms_total"])
                        if i not in _held(d)]
        for m in d["modes"]:
            assert np.shape(m["eigenvector_canonical"]) == (len(free), 3)
            assert np.shape(m["eigenvector_display"]) == (len(free), 3)
        if d["modes"]:
            assert len(d["modes"]) == 3 * len(free) - d["removed_motions"]["count"]

    def test_each_mode_states_its_two_forms_and_the_reader_never_crosses_them(
            self, state):
        """§ 6.3, the two forms: the canonical one `sum m|L|^2 = 1` in the
        file's masses, the science form; the display one the same rescaled so
        its largest absolute component is 1, the animation's -- and the reader
        hands each back under its own name.  THE SILENT FAILURE: every
        intensity and physical amplitude pairs with the canonical form, so a
        reader serving the display form there is wrong by a per-mode factor,
        with nothing on the page to say so."""
        path, _states = state
        d, got = _read(path), parse_spectra_json(path)
        if not d["modes"]:
            return
        masses = np.asarray(d["equilibrium"]["masses_amu"])[d["free_atom_idxs"]]
        for row, mode in zip(d["modes"], got.modes):
            canonical = np.asarray(row["eigenvector_canonical"])
            display = np.asarray(row["eigenvector_display"])
            assert float((masses[:, None] * canonical ** 2).sum()) == \
                pytest.approx(1.0, abs=1e-6), row["index_1based"]
            assert np.abs(display).max() == pytest.approx(1.0, abs=1e-12)
            np.testing.assert_allclose(
                display, canonical / np.abs(canonical).max(), atol=1e-12)
            np.testing.assert_array_equal(mode.eigenvector_canonical, canonical)
            np.testing.assert_array_equal(mode.eigenvector_display, display)

    def test_the_data_file_gives_back_the_payload_the_json_holds(
            self, state, tmp_path):
        """§ 6.10.5, *"Read back, the NetCDF gives the payload the JSON holds,
        and that round trip is the file's test"* -- the file the job wrote
        beside its result, in the job's env, and the one the writer writes from
        any version of the result -- each naming the JSON bytes it was written
        with.  THE SILENT FAILURE: a variable, attribute or encoding of the
        `.nc` that drops or bends a value -- a `null` read back as a number, a
        flag as an int, a block's key lost, the reference step's atom as -1 --
        which reading the writer's code does not show, because the loss is in
        the pairing of the writer's table with the reader's."""
        import netCDF4
        path, _states = state
        payload = _read(path)
        written = tmp_path / path.name
        write_spectra_payload(payload, written)
        pairs = [written]
        if (path.parent / "run.json").is_file():
            pairs.append(path)
        for result in pairs:
            nc = _data_file(result)
            assert nc.is_file(), "the data file is written beside the JSON"
            assert read_spectra_data(nc) == payload, nc
            with netCDF4.Dataset(str(nc)) as ds:
                said = ds.getncattr("json_sha256")
            assert said == hashlib.sha256(result.read_bytes()).hexdigest(), nc

    def test_the_load_door_serves_it_and_adds_what_only_it_can(
            self, state, web_client, monkeypatch):
        """`web/spectra.md` § 6, `engines/vibration.md` § 6.6: `POST
        /api/spectra/load` answers the file's own fields unchanged -- a `null`
        a `null` -- at every state of a run, and adds, never into the file,
        each mode's element shares where the file states a geometry, and the
        frame set's numbers at the temperature its thermochemistry states;
        no temperature stated, none assumed.  THE SILENT FAILURE: the viewer
        drawing what the file does not state, or a computed field leaking into
        the format, which the writer would then write as if the schema declared
        it."""
        from molbuilder.spectra.mode_frames import at_temperature
        path, _states = state
        d = _read(path)
        served = spectra_served(web_client, monkeypatch, path)
        for key in d:
            if key not in ("modes", "methods_text"):
                assert served[key] == d[key], key
        kelvin = d["thermo"].get("temperature_K")
        if kelvin is None:
            assert "frameset_temperature_k" not in served
        else:
            assert served["frameset_temperature_k"] == kelvin
        assert len(served["modes"]) == len(d["modes"])
        for row, mode in zip(d["modes"], served["modes"]):
            assert {k: mode[k] for k in row} == row
            assert "motion_share_by_element" not in row
            assert "motion_share_by_element" in mode
            if kelvin is None:
                assert "frameset" not in mode
            else:
                assert mode["frameset"] == at_temperature(row, kelvin)


@pytest.fixture(scope="module", params=list(_PYSCF))
def pyscf_state(request, real_water_vibration, real_water_held,
                real_water_probed, real_water_linear):
    runs = {"water": real_water_vibration, "held": real_water_held,
            "probed": real_water_probed, "linear": real_water_linear}
    which, holds, states = _PYSCF[request.param]
    run = runs[which]
    return (_final(run) if holds is None else run.version_where(holds)), states


@pytest.fixture(scope="module", params=list(_SIESTA))
def siesta_state(request, real_co2_vibration):
    holds, states = _SIESTA[request.param]
    run = real_co2_vibration
    return (_final(run) if holds is None else run.version_where(holds)), states


class TestOnPySCF(_EveryState):

    @pytest.fixture
    def state(self, pyscf_state):
        return pyscf_state


class TestOnSIESTA(_EveryState):

    @pytest.fixture
    def state(self, siesta_state):
        return siesta_state


# --------------------------------------------------------------------- #
#  The element shares -- by the masses the file states                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", ["water", "held"])
def test_each_elements_share_is_its_part_of_the_mass_weighted_motion(
        fixture_name, real_water_vibration, real_water_held, web_client,
        monkeypatch):
    """`web/spectra.md` § 4.2, on real modes -- free water, and held water whose
    rows are not the atoms' own numbers: an element's share is `sum m|L|^2` over
    its atoms, and each row is the atom the free list names.  THE SILENT
    FAILURE: a row read as the atom of its own position -- with the oxygen held,
    a hydrogen's motion handed to the oxygen.  The trap the share exists for, a
    heavy atom owning a mode while moving less, is `spectra/test_motion_share.py`'s,
    on the numbers of the mode that showed it."""
    run = {"water": real_water_vibration, "held": real_water_held}[fixture_name]
    path = _final(run)
    d = _read(path)
    elements = d["structure"]["elements"]
    masses = d["equilibrium"]["masses_amu"]
    served = spectra_served(web_client, monkeypatch, path)
    for row, mode in zip(d["modes"], served["modes"]):
        weight = {}
        for atom, moved in zip(d["free_atom_idxs"], row["eigenvector_canonical"]):
            weight[elements[atom]] = weight.get(elements[atom], 0.0) + (
                masses[atom] * float(np.dot(moved, moved)))
        total = sum(weight.values())
        assert mode["motion_share_by_element"] == pytest.approx(
            {el: w / total for el, w in weight.items()}, abs=1e-12)


# --------------------------------------------------------------------- #
#  A file of another version -- the one refusal shown                    #
# --------------------------------------------------------------------- #


def test_a_file_of_another_version_is_refused_by_that_version(
        real_water_probed, web_client, tmp_path, monkeypatch):
    """`engines/vibration.md` § 6.7, § 6.1, K2674-3: what the writer writes, its
    version moved, is refused by name -- `SpectraJsonSchemaError`, carrying the
    two versions -- and the load door answers 422 `schema_mismatch` with them,
    which the page reads to say *update molbuilder* without parsing a sentence.
    THE SILENT FAILURE: the reader's own gate gone, so the typed decoder's
    second one answers instead, and the page is told a field is malformed
    (400 `field`) about a file only a newer molbuilder can read.  One rule, one
    example (`process/testing.md` § 3b)."""
    from conftest import point_the_one_door_at
    moved = SCHEMA_VERSION + 1
    final = _final(real_water_probed)
    path = point_the_one_door_at(tmp_path, monkeypatch.setenv) / final.name
    path.write_text(json.dumps({**_read(final), "schema_version": moved}),
                    encoding="utf-8")
    with pytest.raises(SpectraJsonSchemaError) as refused:
        parse_spectra_json(path)
    assert (refused.value.expected, refused.value.actual) == (SCHEMA_VERSION,
                                                              moved)
    r = web_client.post("/api/spectra/load", json={"path": str(path)})
    said = r.get_json()
    assert r.status_code == 422, said
    assert (said["ok"], said["kind"]) == (False, "schema_mismatch"), said
    assert (said["expected_version"], said["actual_version"]) == (SCHEMA_VERSION,
                                                                  moved), said


# --------------------------------------------------------------------- #
#  An imaginary mode (science)                                           #
# --------------------------------------------------------------------- #


def test_an_imaginary_mode_stays_negative_and_has_no_amplitude(
        real_water_linear, web_client, monkeypatch):
    """`engines/vibration.md` § 6.3, § 4.7: SCIENCE.  Water drawn straight is a
    saddle by symmetry, so a run reaches it: its two bends imaginary and equal,
    moving the hydrogens off the axis; its two stretches real, along it; both
    kinds in one file.  Sign and flag travel together -- drop the sign, and a
    saddle reads as an ordinary vibration -- through the writer, the reader and
    the load door; an imaginary mode has no zero-point amplitude, no thermal
    spread, no frame set, and is left out of the thermochemistry and counted.
    `null` here, never a number made of `sqrt` of a negative.

    MUTATION THIS MUST FAIL AGAINST: an `abs()` of the frequency anywhere on
    that road; the has_imag flag not read."""
    path = _final(real_water_linear)
    d = _read(path)
    got = parse_spectra_json(path)
    served = spectra_served(web_client, monkeypatch, path, temperature_k=298.15)
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    assert freqs == sorted(freqs)
    assert freqs[0] < 0 and freqs[1] < 0 and 0 < freqs[2] < freqs[3], freqs
    assert freqs[0] == pytest.approx(freqs[1], rel=1e-3), freqs
    assert d["thermo"]["n_imag_excluded"] == 2
    for k, row in enumerate(d["modes"]):
        imaginary = k < 2
        moved = np.asarray(row["eigenvector_canonical"])
        # THE MOLECULE LIES ALONG X: a bend moves nothing along it, a stretch
        # nothing across it.
        along = float((moved[:, 0] ** 2).sum() / (moved ** 2).sum())
        assert (along < 1e-6 if imaginary else along > 1 - 1e-6), (k, along)
        for said in (row, served["modes"][k]):
            assert said["has_imag"] is imaginary
            assert said["frequency_cm1"] == freqs[k]
        assert got.modes[k].has_imag is imaginary
        assert got.modes[k].frequency_cm1 == freqs[k]
        if imaginary:
            assert row["zero_point_amplitude_amu12_ang"] is None
            assert row["zero_point_displacement_ang"] is None
            assert served["modes"][k]["thermal_spread_amu12_ang"] is None
            assert served["modes"][k]["frameset"] is None
        else:
            assert row["zero_point_amplitude_amu12_ang"] > 0
            assert served["modes"][k]["thermal_spread_amu12_ang"] > 0
            assert served["modes"][k]["frameset"] is not None


# --------------------------------------------------------------------- #
#  The probe (K2674-4)                                                   #
# --------------------------------------------------------------------- #


def test_a_probes_two_sides_are_the_pushes_it_names(real_water_probed):
    """`engines/vibration.md` § 4.8, § 6.3, K2674-4: a probed mode's block on
    what a run wrote, its amplitude the push the person's item states, its
    equilibrium the file's own -- and, since each number is a push of that size
    along the display form, the two sides' mean rise is the harmonic energy of
    that push, `1/2 lambda Q^2` with `lambda` from the mode's own frequency and
    `Q = A / max|L_canonical|`.  An analytic answer, across four keys and the
    frequency: measured here within 0.3 %.  THE SILENT FAILURE: a coupling
    `dE / 2A` whose denominator is not the push its numerator was taken at; the
    energy at -A read as the one at +A gives a shift of nothing with no word.
    The modes' sides differ where the symmetry says they do: the bend breaks no
    symmetry, so its two sides differ, and the asymmetric stretch's are mirror
    images."""
    from molbuilder.constants import (BOHR_ANGSTROM,
                                      CM1_PER_SQRT_HARTREE_BOHR2_AMU)
    path = _final(real_water_probed)
    d = _read(path)
    got = parse_spectra_json(path)
    bend, _symmetric, asymmetric = d["modes"]
    amplitude = d["config"]["displacement_amplitude_ang"]
    for row, mode in zip(d["modes"], got.modes):
        block = row["electronic_structure"]
        if block is None:
            assert mode.electronic_structure is None
            continue
        typed = mode.electronic_structure
        assert block["amplitude_ang"] == amplitude == typed.amplitude_ang
        assert block["scf_energy_eq_eh"] == d["equilibrium"]["scf_energy_eh"]
        assert typed.scf_energy_minus_eh == block["scf_energy_minus_eh"]
        assert typed.scf_energy_plus_eh == block["scf_energy_plus_eh"]
        rise = (block["scf_energy_minus_eh"] + block["scf_energy_plus_eh"]
                ) / 2.0 - block["scf_energy_eq_eh"]
        largest = np.abs(row["eigenvector_canonical"]).max()
        q_bohr = amplitude / largest / BOHR_ANGSTROM
        curvature = (row["frequency_cm1"]
                     / CM1_PER_SQRT_HARTREE_BOHR2_AMU) ** 2
        assert rise == pytest.approx(curvature * q_bohr ** 2 / 2.0, rel=0.01), (
            row["index_1based"], rise)
    sides = (lambda m: abs(m["electronic_structure"]["scf_energy_plus_eh"]
                           - m["electronic_structure"]["scf_energy_minus_eh"]))
    assert sides(bend) > 1e-6, bend["electronic_structure"]
    assert sides(asymmetric) < 1e-6, asymmetric["electronic_structure"]


# --------------------------------------------------------------------- #
#  Numbers as the engine gave them                                       #
# --------------------------------------------------------------------- #


def test_the_engines_numbers_reach_their_keys_unrounded(real_water_probed):
    """`engines/vibration.md` § 6.4: *"what is ours is that it reaches the right
    key in the right unit, unrounded"*.  The equilibrium energy is the number
    the deck printed from the same variable, and a double's digits are all
    there: the ~10⁻⁵ eV shifts the tab reports are differences of numbers of
    order 10, so a `%.6f`, a float32 cast or a `round()` on write destroys the
    quantity while leaving every value looking plausible -- and, since the
    reader reads what was rounded, nothing else in this module can see it."""
    from molbuilder.runs import run_of
    d = _read(_final(real_water_probed))
    eq = d["equilibrium"]
    said = run_of(real_water_probed.attempt("freq")).stdout.read_text(
        errors="replace")
    assert f"E = {eq['scf_energy_eh']:.10f} Ha" in said
    assert round(eq["scf_energy_eh"], 12) != eq["scf_energy_eh"]
    assert any(round(e, 12) != e for e in eq["mo_energies_eh"])
    assert any(float(np.float32(e)) != e for e in eq["mo_energies_eh"])


def test_a_character_beyond_ascii_travels_as_written(real_water_held, tmp_path):
    """§ 6.8: the file is UTF-8 and states cm⁻¹ and Å as they are, for a person
    who opens it: a run's Methods paragraph, read and written again, holds them
    verbatim, no `\\uXXXX` in the file."""
    path = _final(real_water_held)
    methods = _read(path)["methods_text"]
    beyond = sorted({c for c in methods if ord(c) > 127})
    assert beyond, "the paragraph states no such character: a test of nothing"
    again = tmp_path / path.name
    dump_spectra_json(parse_spectra_json(path), again)
    for text in (path.read_text(encoding="utf-8"),
                 again.read_text(encoding="utf-8")):
        assert all(c in text for c in beyond), beyond
        assert "\\u" not in text


def test_the_writer_refuses_a_number_json_cannot_hold_and_keeps_the_file(
        real_water_probed, tmp_path):
    """§ 6.8: *"no `NaN`: a non-finite number is refused before the file is
    written"* -- an SCF that diverged stops the writer, and what was on disk
    stays whole, with no temporary file beside it: a live reader meets the
    prior version or the new one, never a half-written file.  One rule, one
    example (`process/testing.md` § 3b)."""
    result = parse_spectra_json(_final(real_water_probed))
    path = tmp_path / _final(real_water_probed).name
    dump_spectra_json(result, path)
    before = path.read_bytes()
    result.equilibrium_scf_eh = float("nan")
    with pytest.raises(ValueError):
        dump_spectra_json(result, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [path.name, _data_file(path).name])


# --------------------------------------------------------------------- #
#  What the load door serves when asked                                  #
# --------------------------------------------------------------------- #


def test_the_thermal_spread_is_served_at_the_temperature_asked(
        real_water_probed, web_client, monkeypatch):
    """`web/spectra.md` § 4.1: the load door serves each mode's thermal spread
    at the request's `temperature_k`, says the temperature back, and the tab
    computes nothing: `sigma = Q_zp sqrt(coth(hbar omega / 2 k_B T))`
    (`science/vibrational-averaging.md` § 2); asked for no temperature, no mode
    carries the key.  The contract's own table of the formula is
    `spectra/test_derived.py`'s.  The silent failure: a page drawing a thermal
    size the server never derived, or one at another temperature."""
    from molbuilder.constants import CM1_KELVIN
    path = _final(real_water_probed)
    d = _read(path)
    kelvin = 1500.0
    served = spectra_served(web_client, monkeypatch, path, temperature_k=kelvin)
    assert served["temperature_k"] == kelvin
    for row, mode in zip(d["modes"], served["modes"]):
        x = row["frequency_cm1"] * CM1_KELVIN / (2.0 * kelvin)
        assert mode["thermal_spread_amu12_ang"] / \
            row["zero_point_amplitude_amu12_ang"] == pytest.approx(
                math.sqrt(1.0 / math.tanh(x)), rel=1e-12)
    served = spectra_served(web_client, monkeypatch, path)
    assert "temperature_k" not in served
    assert all("thermal_spread_amu12_ang" not in m for m in served["modes"])


def test_what_a_frame_sets_temperature_means_is_served_as_the_script_says_it(
        real_water_probed, web_client, monkeypatch):
    """`web/spectra.md` § 4.3: at the frame-set dialog's temperature the load
    door serves each mode's numbers for each rule, says the temperature back,
    and a notebook's door (`molbuilder.postprocess.outermost_move`) answers the
    same.  The contract's own table of the numbers is
    `spectra/test_derived.py`'s.  The silent failure: a dialog whose numbers are
    not the frames the generator writes, or a script told another."""
    from molbuilder.postprocess import outermost_move
    from molbuilder.spectra.mode_frames import NODES
    path = _final(real_water_probed)
    kelvin = 1000.0
    served = spectra_served(web_client, monkeypatch, path,
                   frameset_temperature_k=kelvin)
    assert served["frameset_temperature_k"] == kelvin
    for mode in served["modes"]:
        said = mode["frameset"]
        assert said["sigma_amu12_ang"] == pytest.approx(
            said["sigma_over_zero_point"]
            * mode["zero_point_amplitude_amu12_ang"], rel=1e-12)
        for nodes in NODES:
            assert said["outermost_move_ang"][str(nodes)] == outermost_move(
                path, mode=mode["index_1based"], temperature_k=kelvin,
                nodes=nodes)


# --------------------------------------------------------------------- #
#  The data file, of a run's own                                         #
# --------------------------------------------------------------------- #


def test_the_element_is_the_atoms_element_never_its_species_label(
        real_water_linear):
    """`/structure/element` says *the chemical symbol* (`model/chemistry.md`
    § 3: a species is a label and an element): the run's hydrogens are labelled
    `H1` and `H2`, the file states `H` there, the labels kept in the structure
    the group holds whole.  THE SILENT FAILURE: the variable was filled from the
    labels, so a reader selecting gold by element missed every labelled gold
    atom."""
    import xarray as xr
    nc = _data_file(_final(real_water_linear))
    group = xr.load_dataset(nc, group="structure")
    assert [str(e) for e in group.element.values] == ["O", "H", "H"]
    back = Structure.from_dict(read_spectra_data(nc)["structure"])
    assert back.elements == ["O", "H1", "H2"]


def test_every_axis_is_a_coordinate_a_reader_selects_by(real_water_probed):
    """`model/datafiles.md` § 2a: every axis a coordinate of its own name, so a
    reader selects by value -- a temperature of the thermochemistry's grid, a
    mode, an atom.  THE SILENT FAILURE (the 2026-10-23 review): the file's axes
    carried their values under other names (`index_1based`, `temperatures_K`),
    so `.sel(temperature_k=298.15)` raised and xarray indexed no axis of any
    group.

    MUTATION THIS MUST FAIL AGAINST: an axis's values named other than the
    axis."""
    import xarray as xr
    final = _final(real_water_probed)
    d = _read(final)
    nc = _data_file(final)
    for group in ("structure", "equilibrium", "modes",
                  "modes/electronic_structure", "thermo/grid"):
        ds = xr.load_dataset(nc, group=group)
        assert set(ds.sizes) <= set(ds.indexes), (group, dict(ds.sizes),
                                                  list(ds.indexes))
    th = d["thermo"]
    grid = xr.load_dataset(nc, group="thermo/grid")
    assert float(grid.g_eh.sel(temperature_k=th["temperature_K"])) == th["g_eh"]
    modes = xr.load_dataset(nc, group="modes")
    assert float(modes.frequency_cm1.sel(mode=2)) == \
        d["modes"][1]["frequency_cm1"]
