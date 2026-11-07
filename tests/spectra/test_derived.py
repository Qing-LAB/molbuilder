"""What a mode's temperature means, by the contract's own tables
(`science/vibrational-averaging.md` § 2, § 5; `web/spectra.md` § 4.1, § 4.3):
mathematics with an answer on paper, on the numbers the tables name.  What the
load door serves of it is asked of a real run's result in
`tests/test_spectrum_results_e2e.py`.
"""
from __future__ import annotations

import pytest

from molbuilder.spectra.derived import (thermal_spread_amu12_ang,
                                        zero_point_amplitude_amu12_ang)
from molbuilder.spectra.mode_frames import at_temperature


def test_a_soft_mode_swings_past_its_zero_point_and_a_stiff_one_does_not():
    """`web/spectra.md` § 4.1, the contract's own table: at 298 K a 100 cm-1
    mode swings 2.1x its zero-point, a 3000 cm-1 mode 1.00x.  The silent
    failure: a thermal size off by the coth's argument, which still reads as a
    plausible number of Angstrom."""
    assert thermal_spread_amu12_ang(1.0, 100.0, 298) == pytest.approx(
        2.1, abs=0.05)
    assert thermal_spread_amu12_ang(1.0, 3000.0, 298) == pytest.approx(
        1.00, abs=0.005)


@pytest.mark.parametrize("kelvin, quanta, spread, moves", [
    (4, 0.0, 1.000, (0.088, 0.145, 0.190)),
    (298, 0.007, 1.007, (0.088, 0.146, 0.191)),
    (1000, 0.29, 1.26, (0.110, 0.182, 0.239)),
    (2159, 1.0, 1.73, (0.152, 0.251, 0.329)),
])
def test_what_a_temperature_means_for_a_mode_is_the_contracts_table(
        kelvin, quanta, spread, moves):
    """`web/spectra.md` § 4.3: at a frame set's temperature, a mode's mean
    number of quanta, its spread and the outermost frames' largest atom move
    for three, five and seven frames.  A row of the contract's own example
    (`science/vibrational-averaging.md` § 5, mode 48 of the user's AuBDTAu:
    1039.9 cm-1, the atom that moves most 0.3977 amu^-1/2 along it), to the
    digits the table prints.  The silent failure: a dialog whose numbers are not
    the frames the generator writes -- another rule's node, a spread without the
    temperature."""
    said = at_temperature(
        {"frequency_cm1": 1039.9,
         "zero_point_amplitude_amu12_ang":
             zero_point_amplitude_amu12_ang(1039.9, False),
         "eigenvector_canonical": [[0.3977, 0.0, 0.0], [0.0, -0.2, 0.0]]},
        kelvin)
    assert said["mean_quanta"] == pytest.approx(quanta, abs=0.005)
    assert said["sigma_over_zero_point"] == pytest.approx(
        spread, abs=0.0005 if spread < 1.01 else 0.005)
    assert [said["outermost_move_ang"][n] for n in ("3", "5", "7")] \
        == pytest.approx(list(moves), abs=0.0005)
