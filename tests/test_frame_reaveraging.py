"""A frame set's frames re-averaged over another motion
(`science/vibrational-averaging.md` § 5.4; `frame_average.motion_weights`).

Each frame is its position along the mode; its weight is the motion's.  The
expectations are the contract's own table -- the contact bond
``T = exp(-a q)`` at ``a sigma_T = 1``, frames written at the junction's
spread ``sigma_T``, a swing ``Q0`` added on top -- and its rules: the set's
own weights reproduced where the motion is the set's own, a swing past the
outermost frame refused, three frames blind to a swing's shape, a weight
below zero said.  The frames' positions are the Gauss-Hermite rule's, from
numpy (``hermgauss``), as the generator places them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from molbuilder.frame_average import ReaverageError, motion_weights

#: A mode whose zero-point amplitude is 1 amu^1/2 Angstrom, so that at 0 K
#: the junction's spread sigma_T is 1 and a swing adding n quanta turns at
#: Q0 = 2 sqrt(n) (science § 2a.2).
_MODE = {"frequency_cm1": 1000.0, "zero_point_amplitude_amu12_ang": 1.0}


def _frames(m: int) -> np.ndarray:
    """The m-point rule's positions at spread 1: q = sqrt(2) x."""
    x, _w = np.polynomial.hermite.hermgauss(m)
    return np.sqrt(2.0) * x


def _swing(q0: float) -> float:
    """The quanta a swing of amplitude q0 adds, at Q_zp = 1."""
    return q0 * q0 / 4.0


@pytest.mark.parametrize("q0, expected", [
    (1.0, (1.9573, 2.0815, 2.0873)),
    (2.0, (None, 3.6405, 3.7494)),
    (3.0, (None, None, 7.9011)),
])
def test_the_frames_reaveraged_over_a_swing_give_the_contracts_table(
        q0, expected):
    """`science/vibrational-averaging.md` § 5.4, its table: the contact bond
    averaged over the junction's bell curve carried along a swing, from 3,
    5 and 7 frames written at the junction's spread -- each number the
    table's, and a swing turning beyond the outermost frame refused, naming
    both.  The silent failure: a re-averaged number that is the bell
    curve's (the shape lost), or one taken from values extrapolated past
    the frames."""
    for m, want in zip((3, 5, 7), expected):
        q = _frames(m)
        if want is None:
            with pytest.raises(ReaverageError) as refused:
                motion_weights(q, temperature_k=0, swing_quanta=_swing(q0),
                               **_MODE)
            assert "beyond the outermost frame" in str(refused.value)
            continue
        got = motion_weights(q, temperature_k=0, swing_quanta=_swing(q0),
                             **_MODE)
        assert float(np.dot(got["weights"], np.exp(-q))) == pytest.approx(
            want, abs=5e-5), m


@pytest.mark.parametrize("m", [3, 5, 7])
def test_the_sets_own_motion_gives_the_sets_own_weights(m):
    """§ 5.4: at the set's own temperature with no swing, the weights are
    the Gauss-Hermite rule's -- the set's own -- so the re-averaged number
    is the record's.  The silent failure: a re-averaging that disagrees
    with the record where nothing was asked to change."""
    _x, w = np.polynomial.hermite.hermgauss(m)
    got = motion_weights(_frames(m), temperature_k=0, swing_quanta=0,
                         **_MODE)
    assert got["weights"] == pytest.approx(list(w / math.sqrt(math.pi)),
                                           abs=1e-12)
    assert got["said"] == []


def test_what_is_said_beside_the_number():
    """§ 5.4, *what it needs*: three frames under a swing are said blind to
    its shape; seven frames under a swing three times the junction's spread
    are said weighted below zero -- the middle frame at -1.0 -- and the
    number is still given.  The motion is said whole: its quanta, spreads
    and turning point, and how far the frames reach."""
    three = motion_weights(_frames(3), temperature_k=0,
                           swing_quanta=_swing(1.0), **_MODE)
    assert any("blind to the swing's shape" in s for s in three["said"])
    seven = motion_weights(_frames(7), temperature_k=0,
                           swing_quanta=_swing(3.0), **_MODE)
    middle = int(np.argmin(np.abs(_frames(7))))
    assert seven["weights"][middle] == pytest.approx(-1.0, abs=0.02)
    assert any("weighted below zero" in s for s in seven["said"])
    motion = seven["motion"]
    assert motion["swing_amplitude_amu12_ang"] == pytest.approx(3.0)
    assert motion["thermal_spread_amu12_ang"] == pytest.approx(1.0)
    assert motion["spread_amu12_ang"] == pytest.approx(math.sqrt(1 + 4.5))
    assert seven["reach"]["outermost_over_swing"] == pytest.approx(
        float(np.abs(_frames(7)).max()) / 3.0)


@pytest.mark.parametrize("q, ask, refused", [
    ([float("nan")] * 3, {}, "no position along a mode"),
    ([0.0, 1.0, 1.0], {}, "at one position"),
    (None, {"temperature_k": -1}, "number of kelvin"),
    (None, {"swing_quanta": -0.5}, "0 or more"),
    (None, {"frequency_cm1": None}, "frequency and zero-point amplitude"),
], ids=["no positions", "two at one position", "below zero kelvin",
        "negative quanta", "no mode"])
def test_a_motion_the_frames_cannot_carry_is_refused(q, ask, refused):
    """§ 5.4: refused by name -- frames stating no position, two frames at
    one position, a temperature below zero, negative swing quanta, a set
    stating no mode's frequency.  The silent failure: weights computed from
    nothing, or a motion the person never stated."""
    kw = {"temperature_k": 0, "swing_quanta": 0, **_MODE, **ask}
    with pytest.raises(ReaverageError) as got:
        motion_weights(_frames(3) if q is None else q, **kw)
    assert refused in str(got.value)
