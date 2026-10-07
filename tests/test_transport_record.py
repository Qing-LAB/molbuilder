"""P6 — the transport record's parsers (`engines/transport.md` § 6,
`transport/record.py`).

What a TBtrans output says is read off the transport road's own minimal
junction (plan Q5-Q7, `process/testing.md` § 6).
"""
from __future__ import annotations

from molbuilder.transport.record import conductance_g0


def test_conductance_needs_the_window_to_straddle_ef():
    assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None
