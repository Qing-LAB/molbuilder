"""A real H2 vibration's mechanics and its record (`engines/vibration.md`
§ 6.6, § 6.7, § 6.11 § 6, § 7; `science/vibrational-averaging.md` § 2b.5,
its end-to-end row; `engines/vibration-frames.md` § 1, § 4, § 5): a free H2
run on PySCF down the road, its result read by `postprocess.vibration` from
the run's folder and each of its two files and by the result's reader, and a
frame set written from the run's folder and read back reopening its run.
The numbers are held to CODATA's constants and the file's own frequency and
masses, never to molbuilder's own -- but for one cross-door check, a run's
folder's outermost move by `postprocess.outermost_move` against the
toolkit's for the same state (`engines/vibration-frames.md` § 4: one rule,
two doors); the record's rules -- a file its own inputs contradict refused,
the modes' temperatures the file's own -- to the contract, on that run's
result.
The closed forms on typed modes are `test_vibration_mechanics.py`.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import constants as codata

from molbuilder.runfiles import ROLE_SPECTRA, ROLE_SPECTRA_DATA, compose, stem_before
from molbuilder.runs import on_label_at
from support import real_runs as _real_runs

AMU_KG = codata.physical_constants["atomic mass constant"][0]


@_real_runs.makes_a_real_run("module", "molbuilder-pySCF")
def real_h2_vibration(tmp_path_factory):
    """This module's own run (`process/testing.md` tier 3): a free H2."""
    return _real_runs.h2_vibration(tmp_path_factory)


def _files(run):
    result = on_label_at(run.attempt("freq"), ROLE_SPECTRA)
    data = result.with_name(compose(stem_before(result.name, ROLE_SPECTRA),
                                    ROLE_SPECTRA_DATA))
    return result, data


@pytest.mark.parametrize("which", ["json", "data file"])
def test_h2s_bond_moves_by_the_textbooks_zero_point_amplitude(
        real_h2_vibration, which):
    """§ 2b.5, end to end: from the file's frequency and masses, H2's bond
    r.m.s. at 0 K is ``sqrt(hbar / 2 mu omega)``; at ``Q_zp`` each atom moves
    half of it, the two in opposite directions along the bond; the
    projection of those moves gives ``Q_zp`` back; at 0 K the state's spread
    is ``Q_zp`` and each atom's r.m.s. move half the bond's; an open
    thermal start bound to the mode is at the temperature the record states
    (§ 6.11 § 9 item 6; `engines/vibration-frames.md` § 2),
    its spread ``Q_zp sqrt(coth(h c nu / 2 k_B T))`` -- the same from the
    JSON and from its data file; the vibration says the path it was read
    from, as given, and the structure the result states.  THE SILENT FAILURE: a mass, a unit or a
    normalisation off, which every number derived from the mode would carry
    without a sign; or the record's temperature misread from one file."""
    from molbuilder.postprocess import states, vibration
    result, data = _files(real_h2_vibration)
    read = result if which == "json" else data
    v = vibration(read)
    stated_t = json.loads(result.read_text())["thermo"]["temperature_K"]
    assert (v.record_temperature_k, v.read_path) == (stated_t, str(read))
    assert v.structure is not None and len(v.structure.elements) == 2
    assert len(v.modes) == 1
    mode = v.modes[0]
    nu, (m1, m2) = mode.frequency_cm1, mode.masses_amu
    assert m1 == m2 and nu > 3000.0
    mu_kg = m1 * m2 / (m1 + m2) * AMU_KG
    omega = 2.0 * math.pi * codata.c * nu * 100.0
    bond_rms = math.sqrt(codata.hbar / (2.0 * mu_kg * omega)) * 1e10
    u = mode.displacement(zero_point_widths=1.0)
    axis = mode.positions_ang[1] - mode.positions_ang[0]
    axis /= np.linalg.norm(axis)
    along = u @ axis
    assert along[0] - along[1] == pytest.approx(-bond_rms, rel=1e-6) or \
        along[0] - along[1] == pytest.approx(bond_rms, rel=1e-6)
    assert np.abs(along) == pytest.approx([bond_rms / 2] * 2, rel=1e-6)
    assert mode.coordinate(u) == pytest.approx(
        mode.zero_point_amplitude_amu12_ang, rel=1e-9)
    cold = mode.bind(states.thermal(temperature_k=0.0))
    assert cold.spread_amu12_ang == pytest.approx(
        mode.zero_point_amplitude_amu12_ang, rel=1e-12)
    assert cold.atom_rms_ang == pytest.approx([bond_rms / 2] * 2, rel=1e-6)
    taken = mode.bind(states.thermal())
    assert (taken.temperature_k, taken.temperature_from) == (
        stated_t, "the record's")
    q_zp = math.sqrt(codata.hbar / (2.0 * omega) / AMU_KG) * 1e10
    x = codata.h * codata.c * nu * 100.0 / (codata.k * stated_t)
    assert taken.spread_amu12_ang == pytest.approx(
        q_zp * math.sqrt(1.0 / math.tanh(x / 2.0)), rel=1e-6)
    assert mode.motion_shares(by="element") == pytest.approx({"H": 1.0})

    # THE RECORD STATES EACH MODE'S MECHANICS (v15, § 6.11 § 3), in the file
    # itself -- the JSON's keys and the data file's variables alike: each
    # atom half the motion, the largest zero-point move each atom's half of
    # the bond's r.m.s., Gaussian's reduced mass the atom's own mass, the
    # force constant mu omega^2 from CODATA; at the record's temperature the
    # spread Q_zp sqrt(coth(h c nu / 2 k_B T)) and the mean quanta
    # 1 / (exp(h c nu / k_B T) - 1), at 0 K the spread Q_zp and no quanta;
    # and the script door's `m.thermal` the record's row.
    if which == "json":
        stated = json.loads(result.read_text())
    else:
        from molbuilder.spectra.datafile import read_spectra_data
        stated = read_spectra_data(data)
    row = stated["modes"][0]
    assert row["motion_share"] == pytest.approx([0.5, 0.5], rel=1e-9)
    assert row["largest_zero_point_move_ang"] == pytest.approx(bond_rms / 2,
                                                              rel=1e-6)
    assert row["reduced_mass_amu"] == pytest.approx(m1, rel=1e-9)
    assert row["force_constant_mdyn_per_ang"] == pytest.approx(
        m1 * AMU_KG * omega ** 2 / 100.0, rel=1e-6)
    thermal = stated["modes_thermal"]
    k = thermal["temperature_k"].index(stated_t)
    assert thermal["thermal_spread_amu12_ang"][0][k] == pytest.approx(
        q_zp * math.sqrt(1.0 / math.tanh(x / 2.0)), rel=1e-6)
    assert thermal["mean_quanta"][0][k] == pytest.approx(1.0 / math.expm1(x),
                                                         rel=1e-6, abs=0.0)
    cold = thermal["temperature_k"].index(0.0)
    assert (thermal["thermal_spread_amu12_ang"][0][cold],
            thermal["mean_quanta"][0][cold]) == (pytest.approx(q_zp, rel=1e-6),
                                                 0.0)
    assert mode.thermal["temperature_k"] == thermal["temperature_k"]
    for key in ("thermal_spread_amu12_ang", "mean_quanta"):
        assert mode.thermal[key] == pytest.approx(thermal[key][0], rel=1e-12,
                                                  abs=1e-300)


def _rewritten(run, tmp_path, change, name="h2"):
    """The run's own result, its payload changed by ``change``, written by
    the one writer into ``tmp_path`` -- both files, as a writer writes them."""
    from molbuilder.sidecars.spectra import write_spectra_payload
    result, _ = _files(run)
    payload = json.loads(result.read_text())
    change(payload)
    out = tmp_path / compose(name, ROLE_SPECTRA)
    write_spectra_payload(payload, out)
    return out


def test_a_v15_file_whose_stored_values_its_inputs_contradict_is_refused(
        real_h2_vibration, tmp_path):
    """§ 6.6, § 6.7, I16: a reader works each derived value out again and
    refuses, by name, a file whose stored one disagrees -- a file never
    carries a number its own inputs contradict; one example of the one rule,
    a share moved, the refusal naming the mode, the key and the element.
    The run's file edited by hand, the one way to show it: no writer states
    a value its inputs contradict.
    THE SILENT FAILURE: the reader taking a stored v15 value on trust, so
    the file and its own inputs tell a person two different things."""
    from molbuilder.sidecars.spectra import (SpectraJsonFieldError,
                                             parse_spectra_json)
    result, _ = _files(real_h2_vibration)
    payload = json.loads(result.read_text())
    payload["modes"][0]["motion_share"] = [0.6, 0.4]
    path = tmp_path / compose("edited", ROLE_SPECTRA)
    path.write_text(json.dumps(payload))
    with pytest.raises(SpectraJsonFieldError,
                       match=r"mode 1: motion_share\[0\] is 0.6, its "
                             r"derivation"):
        parse_spectra_json(path)


def test_the_modes_temperatures_are_the_files_own(real_h2_vibration,
                                                 tmp_path):
    """§ 6.6, § 6.7: a file states the temperatures its thermal block is
    at, and a reader checks each row at them -- never at a list of its own,
    so a later list changes new files only.  The run's own result, written
    again by the one writer stating 0 K, a temperature where H2's mean
    quanta is near the smallest number a double holds (``h c nu / k_B T``
    = 740, where one unit in the last place is a percent of it),
    77 K and the record's: read back from both files, the script door's
    `m.thermal` is at exactly those, and the result read and written again
    keeps them.  That cold quanta one unit away in its last place -- the
    most two machines' ``exp`` may differ by, which no run here can cause --
    still reads (the floor of § 6.7).  THE SILENT FAILURES: every file
    written before a change to the thermochemistry's grid refused; a result
    the job's node wrote refused on the server for a last bit."""
    from molbuilder.postprocess import vibration
    from molbuilder.sidecars.spectra import parse_spectra_json
    result, data = _files(real_h2_vibration)
    stated = json.loads(result.read_text())
    nu = stated["modes"][0]["frequency_cm1"]
    t_cold = codata.h * codata.c * nu * 100.0 / (codata.k * 740.0)
    own = [0.0, t_cold, 77.0, stated["thermo"]["temperature_K"]]

    def state_own(payload):
        payload["modes_thermal"] = {"temperature_k": own}
    out = _rewritten(real_h2_vibration, tmp_path, state_own)
    for read in (out, out.with_name(compose("h2", ROLE_SPECTRA_DATA))):
        assert vibration(read).modes[0].thermal["temperature_k"] == own
    # READ AND WRITTEN AGAIN, the file keeps its list.
    assert parse_spectra_json(out).to_dict()["modes_thermal"][
        "temperature_k"] == own
    written = json.loads(out.read_text())
    cold = written["modes_thermal"]["mean_quanta"][0][1]
    assert 0.0 < cold < np.finfo(float).tiny

    last_bit = json.loads(out.read_text())
    last_bit["modes_thermal"]["mean_quanta"][0][1] = float(
        np.nextafter(cold, 1.0))
    path = tmp_path / compose("last_bit", ROLE_SPECTRA)
    path.write_text(json.dumps(last_bit))
    parse_spectra_json(path)



def test_a_run_folder_reads_as_its_vibration_and_writes_its_set(
        real_h2_vibration, tmp_path):
    """`engines/vibration-frames.md` § 1, § 5, end to end: the run's folder
    is read as its vibration -- the run, the result JSON's sha256, the same
    whichever of the result's two files is read, and frame 0's record of
    the relaxation its geometry came from, or why there is none -- and a
    set written of it states the run and the sha256 as its provenance rows
    and that record as frame 0's info.calculation, or says why not.  THE
    SILENT FAILURE: a set whose provenance names another result than the
    one its modes came from, or a transport citation stamped with settings
    from nowhere."""
    from molbuilder.frameset import read
    from molbuilder.persist import sha256_of
    from molbuilder.postprocess import (outermost_move, read_frame_set,
                                        states, vibration, write_frame_set)
    from molbuilder.projects import address
    from molbuilder.workingcopy_structure import StructureCodec
    run = real_h2_vibration.attempt("freq")
    result, data = _files(real_h2_vibration)
    v = vibration(run)
    assert v.run == run and v.read_path == str(run)
    assert v.result_sha256 == sha256_of(result)
    assert vibration(result).result_sha256 == v.result_sha256
    assert vibration(data).result_sha256 == v.result_sha256
    # A PySCF DECK RECORDS NO RELAXATION (§ 1): none, and why, said.
    assert v.relaxation_calculation is None
    assert "no transport citation" in v.relaxation_said
    written = write_frame_set(
        tmp_path / "h2.xyz",
        v.mode(1).bind(states.thermal(temperature_k=300.0)).frames(3))
    s = StructureCodec().load(written.path, frames=True)
    rows = read(s)
    assert rows.value("vibration_run") == address(run)
    assert rows.value("vibration_result_sha256") == v.result_sha256
    assert "calculation" not in (s.info or {})
    assert v.relaxation_said in written.said
    # THE SET REOPENS ITS RUN, the one its vibration_run row names (§ 5).
    back = read_frame_set(written.path)
    assert back.mode_distribution is not None
    assert back.mode_distribution.mode.vibration.run == run
    # ANY STATE'S OUTERMOST MOVE, from the run's folder (§ 4): the mode's
    # state's own, one rule at two doors.
    two = states.level(2)
    assert outermost_move(run, mode=1, chain_or_distribution=two,
                          nodes=5) == v.mode(1).bind(two).outermost_move(5)


#: EACH STATE THE DOORS ASK (`engines/vibration-frames.md` § 6), as typed on
#: the command line, the rows § 5.1 says its set states, and those it does
#: not.
_DOORS_STATES = {
    "thermal": (["--temperature", "300"],
                {"temperature_k": 300.0, "sampling_rule": "gauss-hermite",
                 "quantum": 1}, {"quanta"}),
    "n quanta": (["--quanta", "2"],
                 {"quanta": 2, "mean_quanta": 2.0, "sampling_rule": "gauss"},
                 {"temperature_k"}),
    "mean quanta": (["--mean-quanta", "0.5"],
                    {"mean_quanta": 0.5, "sampling_rule": "gauss-hermite"},
                    {"temperature_k", "quanta"}),
    # A MEAN JUST ABOVE ONE QUANTUM, said "1 quantum" as printed.
    "a mean just above one": (["--mean-quanta", "1.00004"],
                              {"mean_quanta": 1.00004,
                               "sampling_rule": "gauss-hermite"},
                              {"temperature_k", "quanta"}),
}


@pytest.fixture(scope="module")
def doors_sets(real_h2_vibration, tmp_path_factory):
    """`molbuilder frameset from-mode` typed on the H2 run's folder in each
    state the doors ask, and once naming none -- what each answered."""
    from support.road import molbuilder
    run = real_h2_vibration
    folder = str(run.attempt("freq").relative_to(run.tree))
    said = {}
    with _real_runs.road_at(run.tree.parent, tmp_path_factory,
                            engine="pyscf"):
        for kind, (flags, _, _) in _DOORS_STATES.items():
            said[kind] = molbuilder(
                "frameset", "from-mode", folder, "--mode", "1", *flags,
                "--out", f"P/structure/{kind.replace(' ', '-')}.xyz")
        # NO STATE AND TWO STATES AT ONCE, ASKED OF A FOLDER THAT IS NO RUN:
        # the state is refused first, before any run is read.
        said["no state"] = molbuilder(
            "frameset", "from-mode", "P/structure", "--mode", "1",
            "--out", "P/structure/none.xyz")
        said["two states"] = molbuilder(
            "frameset", "from-mode", "P/structure", "--mode", "1",
            "--quanta", "2", "--temperature", "300",
            "--out", "P/structure/two.xyz")
    return run, said


@pytest.mark.parametrize("kind", list(_DOORS_STATES))
def test_the_command_line_writes_a_modes_set_in_any_state(doors_sets, kind):
    """`engines/vibration-frames.md` § 6 (the user's decision,
    `engines/vibration.md` § 6.11 § 9 item 2): `frameset from-mode` writes
    the run's mode in the state its flags name -- thermal, mean quanta, n
    quanta (a mean said "1 quantum" as printed) -- its set stating that
    state's rows (§ 5.1) and the command printing the state and each frame
    from 1.  The silent failure: a set sampling another state than the
    person typed."""
    from molbuilder.frameset import read
    from molbuilder.workingcopy_structure import StructureCodec
    run, said = doors_sets
    r = said[kind]
    assert r.exit_code == 0, r.output
    _flags, expected, absent = _DOORS_STATES[kind]
    rows = read(StructureCodec().load(
        run.tree / "P" / "structure" / f"{kind.replace(' ', '-')}.xyz",
        frames=True))
    for name, value in expected.items():
        assert rows.value(name) == pytest.approx(value), (name, rows.value(name))
    for name in absent:
        assert rows.value(name) is None, name
    state_line = r.output.splitlines()[1].strip()
    assert state_line and state_line in rows.value("assumptions"), r.output
    # WHAT IT PRINTS IS THE SET'S OWN: its mean quanta and spread, from its
    # rows (§ 5.1), four figures -- the noun the printed figure's
    # (`web/spectra.md` § 4.3): "1 quantum", "2 quanta" -- in the one
    # wording the page prints too.
    spread = rows.value("spread_amu12_ang") / rows.value(
        "zero_point_amplitude_amu12_ang")
    mean = f"{rows.value('mean_quanta'):.4g}"
    assert (f"it holds {mean} {'quantum' if mean == '1' else 'quanta'} "
            f"on average; its spread is {spread:.4g} times its zero-point "
            f"amplitude") in r.output, r.output
    # MANY QUANTA SAID, never refused (the user's choice, 2026-11-24):
    # beyond 1 eV in the mode, <n> h c nu by CODATA, the note says so.
    from scipy import constants as codata
    energy = (rows.value("mean_quanta") * codata.h * codata.c
              * rows.value("frequency_cm1") * 100.0 / codata.e)
    assert ("far up its ladder, where the harmonic picture the frames are "
            "drawn from may no longer hold" in r.output) == (energy > 1.0), (
        energy, r.output)
    # EACH FRAME AS EVERY MESSAGE NAMES ONE, from 1 (`model/structure.md`
    # § 2.2e).
    n = len(rows.weights)
    assert [line.split(":")[0].strip() for line in r.output.splitlines()
            if line.strip().startswith("frame ")] == [
        f"frame {f} of {n}" for f in range(1, n + 1)], r.output


@pytest.mark.parametrize("kind, words", [
    ("no state", "asked every time"),
    ("two states", "name no one state"),
])
def test_the_command_line_refuses_a_state_before_any_run(doors_sets, kind,
                                                         words):
    """`engines/vibration-frames.md` § 6: a state named nowhere, or two at
    once, is a usage error, exit 2 (`conventions.md` § 3), in the toolkit's
    words and before any run is read -- never the record's temperature
    assumed.  The silent failure: a set sampled at a state the person never
    stated, or a refusal naming the run when the state was the fault."""
    _run, said = doors_sets
    refused = said[kind]
    assert refused.exit_code == 2 and words in refused.output, refused.output
    assert "no run of ours" not in refused.output, refused.output
