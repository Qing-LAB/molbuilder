"""A real H2 vibration's mechanics, through the script door a person uses
(`engines/vibration.md` § 6.11 § 6, § 7; `science/vibrational-averaging.md`
§ 2b.5, its end-to-end row): a free H2 run on PySCF down the road, its
result read by `postprocess.vibration` from each of its two files, and the
answers held to CODATA's constants and the file's own frequency and masses
-- never to molbuilder's own numbers.  The closed forms on typed modes are
`test_vibration_mechanics.py`.
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
    is ``Q_zp`` and each atom's r.m.s. move half the bond's; a state asked
    with no temperature is at the one the record states (§ 6.11 § 9 item 6),
    its spread ``Q_zp sqrt(coth(h c nu / 2 k_B T))`` -- the same from the
    JSON and from its data file.  THE SILENT FAILURE: a mass, a unit or a
    normalisation off, which every number derived from the mode would carry
    without a sign; or the record's temperature misread from one file."""
    from molbuilder.postprocess import vibration
    result, data = _files(real_h2_vibration)
    read = result if which == "json" else data
    v = vibration(read)
    stated_t = json.loads(result.read_text())["thermo"]["temperature_K"]
    assert (v.record_temperature_k, v.source) == (stated_t, read.name)
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
    cold = mode.state(temperature_k=0.0)
    assert cold.spread_amu12_ang == pytest.approx(
        mode.zero_point_amplitude_amu12_ang, rel=1e-12)
    assert cold.atom_rms_ang == pytest.approx([bond_rms / 2] * 2, rel=1e-6)
    taken = mode.state()
    assert (taken.temperature_k, taken.temperature_from) == (
        stated_t, "the record's")
    q_zp = math.sqrt(codata.hbar / (2.0 * omega) / AMU_KG) * 1e10
    x = codata.h * codata.c * nu * 100.0 / (codata.k * stated_t)
    assert taken.spread_amu12_ang == pytest.approx(
        q_zp * math.sqrt(1.0 / math.tanh(x / 2.0)), rel=1e-6)
    assert mode.element_shares == pytest.approx({"H": 1.0})
