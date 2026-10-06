"""P6 — the transport record's parsers (`engines/transport.md` § 6,
`transport/record.py`).

``tests/data/chain.TBT.AVTRANS_L-R`` is FROZEN FROM A REAL RUN -- the
equilibrium point's k-averaged transmission, verbatim, from the carbon-chain
live walk of 2026-08-29 (SIESTA/TBtrans 5.4.2).  T(E) ≈ 2.0 there is the
textbook two-π-channel answer for a perfect cumulene chain, which is what
makes the file also a physics pin; G(E_F) is interpolated only inside the
window.  The current-line tests read the 0.4 V point's output tail cut from
its run, or lines invented as text, and were retired 2026-10-04
(`process/testing.md` § 6).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.transport.record import conductance_g0, parse_avtrans

_DATA = Path(__file__).parent / "data"


class TestTheParsers:

    def test_avtrans_parses_the_real_file(self):
        e, t = parse_avtrans((_DATA / "chain.TBT.AVTRANS_L-R").read_text())
        assert len(e) == 400 and len(t) == 400
        assert e[0] == pytest.approx(-1.995)
        # the physics pin: a perfect cumulene chain carries two open
        # pi channels -- T(E_F) = 2
        assert conductance_g0(e, t) == pytest.approx(2.0, abs=0.01)

    # Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
    # 4 tests here parsed a TBtrans output tail cut from its run, or
    # current lines and an AVTRANS body invented as text (`process/testing.md` § 6).

    def test_conductance_needs_the_window_to_straddle_ef(self):
        assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None
