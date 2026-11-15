"""The least pole energy molbuilder says to type (`engines/transport.md`
§ 6.1c: a pole energy giving fewer than 20 poles "is said before the queue
wait, as information, with the least energy that gives twenty").

TranSIESTA counts the equilibrium contour's poles as int(E / (pi kT))
(§ 6.1c; `m_ts_chem_pot.F90`), so the least energy giving twenty is the
smallest hundredth of an eV whose count is twenty -- above 20 pi kT, rounded
up.  Checked against that rule at the temperatures the form offers, from
the room's to the form's top, where rounding to the nearest hundredth would
fall one pole short (300 K: 1.62 eV gives 19).

THE SILENT FAILURE: the energy molbuilder suggests typed as said, and
TranSIESTA stopping on it after the queue wait.

API-LEVEL: the rule's one function, `transport.transiesta.pole_energy_for`,
which the settings gate says the energy by; the real run's prep saying it
is `test_transport_on_a_real_junction_e2e`'s.
"""
from __future__ import annotations

import math

import pytest


@pytest.mark.parametrize("temperature_k", [300.0, 350.0, 1000.0, 1847.0,
                                           5000.0])
def test_the_least_energy_said_gives_twenty_poles_and_a_hundredth_less_not(
        temperature_k):
    from molbuilder.transport.transiesta import pole_energy_for
    from molbuilder.units import ENERGY_EV
    pi_kt = math.pi * temperature_k * ENERGY_EV["k"]
    said = pole_energy_for(20, temperature_k)
    assert round(said, 2) == said, said
    assert int(said / pi_kt) >= 20, (said, said / pi_kt)
    assert int(round(said - 0.01, 2) / pi_kt) < 20, (said, said / pi_kt)
