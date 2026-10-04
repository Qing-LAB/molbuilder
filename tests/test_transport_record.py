"""P6 — the transport record (`engines/transport.md` § 6,
`transport/record.py` + `summarize run`'s transport arm + the
transmission walk).

The parse fixtures are FROZEN FROM A REAL RUN — the carbon-chain live
walk of 2026-08-29 (SIESTA/TBtrans 5.4.2): ``tests/data/
chain.TBT.AVTRANS_L-R`` is the equilibrium point's k-averaged
transmission verbatim, ``chain-tbtrans-v0.4.out`` the 0.4 V point's
output tail with the binary's own current line.  T(E) ≈ 2.0 there is
the textbook two-π-channel answer for a perfect cumulene chain, which
is what makes these fixtures also a physics pin.

Properties under guard, each named for its failure:

* the AVTRANS parse (grid + T), the current parse (Fortran floats,
  negative mantissa form included), G(E_F) by interpolation;
* `collect_record`: per-point walk, a not-yet-run point reads as
  PENDING (never a failure of the set), nothing-ran refuses naming
  what to launch;
* the record file + `summarize run`'s table;
* the transmission walk CONTINUES past a failed point, and a
  not-yet-run point reads as pending rather than broken — the two rules
  of § 6's walk callout: nothing is handed forward, so a bad point says
  nothing about the next, and `summarize` is a reader (the device
  chain's stop rule deliberately does not apply, because ITS points do
  chain a density).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.transport.record import (RecordError, conductance_g0,
                                         parse_avtrans, parse_current_a)
# `prep_calculation` from there too: the five steps handed the run card's
# shape, as the prep entry hands it (`architecture.md` § 5.2).

_DATA = Path(__file__).parent / "data"


class TestTheParsers:

    def test_avtrans_parses_the_real_file(self):
        e, t = parse_avtrans((_DATA / "chain.TBT.AVTRANS_L-R").read_text())
        assert len(e) == 400 and len(t) == 400
        assert e[0] == pytest.approx(-1.995)
        # the physics pin: a perfect cumulene chain carries two open
        # pi channels -- T(E_F) = 2
        assert conductance_g0(e, t) == pytest.approx(2.0, abs=0.01)

    def test_the_current_line_parses(self):
        amps = parse_current_a(
            (_DATA / "chain-tbtrans-v0.4.out").read_text())
        assert amps == pytest.approx(3.09835e-05)

    def test_fortran_negative_mantissa_parses(self):
        assert parse_current_a(
            "L -> R, V [V] / I [A]: 0.400000     V / -.619664E-05 A"
        ) == pytest.approx(-6.19664e-06)

    def test_no_current_line_is_none_not_a_crash(self):
        assert parse_current_a("nothing here") is None

    def test_conductance_needs_the_window_to_straddle_ef(self):
        assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None

    def test_garbage_refuses_by_name(self):
        with pytest.raises(RecordError):
            parse_avtrans("# only comments\n")


