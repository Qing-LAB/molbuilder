"""P6 — the transport record's parsers (`engines/transport.md` § 6,
`transport/record.py`).

What a TBtrans output says is read off the transport road's own minimal
junction (plan Q5-Q7, `process/testing.md` § 6).
"""
from __future__ import annotations

from molbuilder.transport.record import conductance_g0, linear_response_iv


def test_conductance_needs_the_window_to_straddle_ef():
    assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None


def test_a_perfect_channels_low_bias_current_is_g0_times_v():
    """`engines/transport.md` § 2a.10: under the low-bias approximation the
    record's own I(V) is (2e/h) ∫ T(E, 0) [f_L − f_R] dE with μ = ±V/2.  For
    T ≡ 1 the integral is exactly eV at any temperature, so I = G0·V -- the
    Landauer result, which is the contract's expectation and no code's.  A
    voltage whose window plus the Fermi tails reaches past the slice's energy
    window is not integrated: None, with the reach named.  API-level by the
    science exception (testing.md § 3a): the road reaches no transmission
    output on the stand-in, so the integral is run here on T ≡ 1.

    Silent before this: no test ran the integral, so `np.trapz` -- gone from
    this numpy -- crashed `summarize task` on the road's first low-bias
    calculation (2026-10-08); reading the code shows a valid call.
    """
    from molbuilder.constants import CONDUCTANCE_QUANTUM_S as G0
    import numpy as np

    e = np.linspace(-2.0, 2.0, 4001).tolist()
    got = linear_response_iv(e, [1.0] * len(e), [0.0, 0.2, 0.4, 3.9],
                             kt_ev=0.0259)
    for v, i in zip([0.0, 0.2, 0.4], got["current_a"]):
        assert abs(i - G0 * v) < 1e-3 * G0 * max(v, 0.1), (v, i, G0 * v)
    assert got["current_a"][3] is None and "does not reach" in got["notes"]["3.9"]
