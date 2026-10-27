"""The Raman view's numbers -- `engines/vibration.md` § 5.11 § 10, § 14.

Three silent failures a SIESTA run cannot show by itself, since every
frame's spectrum goes through the same arithmetic: the principal value or its
units wrong (the Kramers-Kronig real part against a Lorentz oscillator's
closed form); the real part wrong beside a line as narrow as a run stores it,
where a laser near a transition reads it (against a Gaussian's closed form,
Dawson's function); and the broadening composed wrong (a run's spectrum
widened after the run against the same transitions spread at the wider width
directly, which § 12 measured exact on SIESTA's own output).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from molbuilder.constants import RYDBERG_EV
from molbuilder.spectra.raman import (broadened, eps1_minus_one,
                                      photon_energy_ev)

#: SIESTA's grid at the default run width, 0.002 Ry: Optical.Broaden / 10
#: apart up to 10 Ry (`Src/optical.F`; `engines/vibration.md` § 13, R-c).
RUN_EV = 0.002 * RYDBERG_EV
GRID = np.arange(0.0, 10.0 * RYDBERG_EV + RUN_EV / 20, RUN_EV / 10)

#: A Lorentz oscillator, eV: eps = 1 + f / (w0^2 - w^2 - i g w).
W0, G, F = 5.0, 0.5, 10.0


def lorentz_eps2(w):
    return F * G * w / ((W0 ** 2 - w ** 2) ** 2 + (G * w) ** 2)


def lorentz_eps1_minus_one(w):
    return F * (W0 ** 2 - w ** 2) / ((W0 ** 2 - w ** 2) ** 2 + (G * w) ** 2)


@pytest.mark.parametrize("laser_nm", [None, 633.0, 248.0, 200.0],
                         ids=["static", "below-resonance", "at-resonance",
                              "above-resonance"])
def test_kramers_kronig_gives_a_lorentz_oscillators_real_part(laser_nm):
    """The real part at zero and at a laser below, at and above the
    oscillator's resonance equals its closed form -- the principal value's
    pole and the units of the photon energy."""
    omega = 0.0 if laser_nm is None else photon_energy_ev(laser_nm)
    got = eps1_minus_one(GRID, lorentz_eps2(GRID), omega)
    peak = F / (G * W0)          # the real part's scale near resonance
    assert got == pytest.approx(lorentz_eps1_minus_one(omega),
                                abs=1e-4 * peak)


#: One transition at the run's own width -- the narrowest line a stored
#: spectrum holds, ten grid steps wide -- near a laser's energy.
LINE_EV = 2.0


def line_eps1_minus_one(w):
    """Its real part, closed form: a Gaussian's Hilbert transform is
    Dawson's function, ``(2/(pi s)) [D((w + e)/s) - D((w - e)/s)]``."""
    from scipy.special import dawsn
    s = RUN_EV
    return 2.0 / (math.pi * s) * (dawsn((w + LINE_EV) / s)
                                  - dawsn((w - LINE_EV) / s))


#: Where the laser sits against the line and the grid: between two grid
#: points near the line's top, on one, a hair above one, on the flank.
_NODE = GRID[np.argmin(np.abs(GRID - (LINE_EV + 0.3 * RUN_EV)))]


@pytest.mark.parametrize("omega", [LINE_EV + 0.3 * RUN_EV + 0.37 * RUN_EV / 10,
                                   _NODE, _NODE + 1e-9,
                                   LINE_EV - 2.0 * RUN_EV],
                         ids=["between-points", "on-a-point",
                              "a-hair-above-a-point", "on-the-flank"])
def test_kramers_kronig_holds_beside_a_narrow_line(omega):
    """The real part beside one transition as narrow as a run stores it
    equals the closed form wherever the laser sits on the grid -- the pole
    taken out by the spectrum's own value there, never a straight line's
    between two points."""
    eps2 = siesta_spread([(LINE_EV, 1.0)], RUN_EV)
    peak = np.max(np.abs(line_eps1_minus_one(GRID)))
    assert eps1_minus_one(GRID, eps2, omega) == pytest.approx(
        line_eps1_minus_one(omega), abs=1e-4 * peak)


def siesta_spread(sticks, width_ev):
    """Transitions ``(energy, weight)`` spread on the grid as SIESTA spreads
    them at ``Optical.Broaden`` = ``width_ev`` (`Src/optical.F`)."""
    out = np.zeros_like(GRID)
    for e, w in sticks:
        out += (w * np.exp(-((GRID - e) / width_ev) ** 2)
                / (math.sqrt(math.pi) * width_ev))
    return out


@pytest.mark.parametrize("width_ev", [0.03, 0.136, 1.36])
def test_a_run_widened_after_it_equals_the_wider_spread(width_ev):
    """A run's spectrum widened to ``width_ev`` after the run equals the
    same transitions spread at ``width_ev`` directly -- two of SIESTA's
    Gaussians make one."""
    sticks = [(3.0, 1.0), (3.05, 0.4), (7.2, 2.5), (20.0, 0.8)]
    widened = broadened(GRID, siesta_spread(sticks, RUN_EV), RUN_EV, width_ev)
    direct = siesta_spread(sticks, width_ev)
    assert np.max(np.abs(widened - direct)) == pytest.approx(
        0.0, abs=1e-4 * np.max(direct))
