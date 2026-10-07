"""P6 — the transport record's parsers (`engines/transport.md` § 6,
`transport/record.py`).

What a TBtrans output says is read off the transport road's own minimal
junction (plan Q5-Q7): the k-averaged transmission saved from the carbon-chain
walk of 2026-08-29, ``tests/data/chain.TBT.AVTRANS_L-R``, was retired with its
test 2026-10-06, and the current-line tests -- the 0.4 V point's output tail
cut from its run, or lines invented as text -- on 2026-10-04
(`process/testing.md` § 6).
"""
from __future__ import annotations

from molbuilder.transport.record import conductance_g0


def test_conductance_needs_the_window_to_straddle_ef():
    assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None
