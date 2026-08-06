"""SpectrumChart `_maths.js` — the obligations § 6.3 and § 9 lay on it.

Every assertion here is derived from docs/web/spectrumchart.md, not from the
source: each test names the rule it guards. The maths layer is the one part with
values in and values out and no DOM, which is why it is built and tested first
(contract § 12, first test level).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node

MATHS = (
    Path(__file__).resolve().parents[2]
    / "molbuilder" / "web" / "static" / "lib" / "spectrumchart" / "_maths.js"
)


def call(expr: str, *, modes=None, freqs=None, broadening=None):
    """Evaluate one expression against the module and return its JSON result."""
    setup = []
    if modes is not None:
        setup.append(f"const modes = {json.dumps(modes)};")
    if freqs is not None:
        setup.append(f"const freqs = {json.dumps(freqs)};")
    if broadening is not None:
        setup.append(f"const w = {json.dumps(broadening)};")
    snippet = (
        f"const M = await import({json.dumps(MATHS.resolve().as_uri())});\n"
        + "\n".join(setup)
        + f"\nconsole.log(JSON.stringify({expr}));"
    )
    return run_node([], snippet)


def band(freqs, broadening):
    return call("M.bandHalfWidths(freqs, w)", freqs=freqs, broadening=broadening)


def curve(modes, broadening):
    return call("M.envelope(modes, w)", modes=modes, broadening=broadening)


def mode(index, freq, activity=None, imaginary=False):
    m = {"index": index, "freq": freq, "imaginary": imaginary}
    if activity is not None:
        m["activity"] = activity
    return m


# --- § 6.3  the bands -------------------------------------------------------

class TestBandWidths:
    """§ 6.3 — four steps, in this order, give every band its half-width."""

    @pytest.mark.parametrize("broadening", [0, 0.5, 3, 7.99])
    def test_floor_applies_at_any_width_below_it_not_only_at_zero(self, broadening):
        """§ 6.3 — at ANY broadening below 8 cm-1, zero included, an isolated band
        is 8 cm-1 half-wide."""
        assert band([200.0, 1000.0, 3000.0], broadening) == [8, 8, 8]

    def test_above_the_floor_the_band_is_the_broadening_width(self):
        """§ 6.3, step 1 — the region you aim at is the region you see."""
        assert band([200.0, 1000.0, 3000.0], 20) == [20, 20, 20]

    def test_clamped_to_half_the_gap_so_two_bands_meet_and_stop(self):
        """§ 6.3, step 3 — the picture in the contract: broadening 20, modes 24 apart."""
        assert band([1000.0, 1024.0], 20) == [12, 12]

    def test_each_mode_is_clamped_by_its_own_nearer_neighbour(self):
        """§ 6.3, step 3 — 'the nearer neighbour', so a crowded pair narrows and a
        lone mode does not."""
        assert band([1000.0, 1024.0, 2000.0], 20) == [12, 12, 20]

    def test_modes_at_the_same_frequency_stay_clickable(self):
        """§ 6.3, step 4 — both keep a band rather than clamping each other to nothing."""
        assert band([1131.8, 1131.8], 20) == [0.25, 0.25]

    def test_no_two_bands_overlap_at_any_width(self):
        """§ 6.3 — every point of the plot belongs to at most one band."""
        freqs = [212.0, 218.0, 400.0, 401.5, 402.0, 1130.0, 1131.0, 3100.0]
        for width in (0, 1, 8, 20, 50, 200):
            half = band(freqs, width)
            spans = sorted(
                (f - h, f + h) for f, h in zip(freqs, half)
            )
            # equal frequencies share a span by design (step 4); the rest must not overlap
            for (lo1, hi1), (lo2, hi2) in zip(spans, spans[1:]):
                if (lo1, hi1) == (lo2, hi2):
                    continue
                assert hi1 <= lo2 + 1e-9, f"bands overlap at broadening {width}"

    def test_the_band_you_are_in_is_the_nearest_mode(self):
        """§ 6.3 — a click inside a band always means the nearest mode."""
        freqs = [1000.0, 1024.0, 2000.0]
        half = band(freqs, 20)
        for f, h in zip(freqs, half):
            for probe in (f - h + 1e-6, f, f + h - 1e-6):
                nearest = min(freqs, key=lambda g: abs(probe - g))
                assert nearest == f

    def test_an_empty_spectrum_has_no_bands(self):
        """§ 8.3 — an empty list is a state, not a failure."""
        assert band([], 20) == []


# --- § 9  the envelope ------------------------------------------------------

class TestEnvelope:
    """§ 9 — the curve is a sum of Lorentzians, sampled by accuracy not by constants."""

    def test_no_curve_at_zero_broadening(self):
        """§ 8.3 — zero means no curve: bare sticks."""
        assert curve([mode(1, 1000.0, 5.0)], 0) is None

    def test_the_curve_at_a_mode_is_its_own_peak_plus_the_tails_of_the_others(self):
        """§ 9 — the envelope is the sum, computed here independently of the source."""
        modes = [mode(1, 1000.0, 10.0), mode(2, 1030.0, 4.0)]
        w = 20.0
        got = curve(modes, w)
        gamma = w / 2
        expected = 10.0 + 4.0 * (gamma**2 / ((1000.0 - 1030.0) ** 2 + gamma**2))
        i = min(range(len(got["x"])), key=lambda k: abs(got["x"][k] - 1000.0))
        assert got["x"][i] == pytest.approx(1000.0, abs=w / 8)
        assert got["y"][i] == pytest.approx(expected, rel=0.02)

    def test_with_no_strengths_anywhere_every_mode_adds_a_peak_of_height_one(self):
        """§ 9 — the sum follows the picture (§ 6.2): a frequency distribution."""
        got = curve([mode(1, 1000.0), mode(2, 2000.0)], 20)
        peak = max(got["y"])
        assert peak == pytest.approx(1.0, rel=0.02)

    def test_a_mode_without_a_strength_adds_nothing_when_others_have_one(self):
        """§ 9 — missing is not weak: it is absent from the sum, not a zero peak."""
        with_gap = curve([mode(1, 1000.0, 10.0), mode(2, 1030.0)], 20)
        alone = curve([mode(1, 1000.0, 10.0)], 20)
        assert max(with_gap["y"]) == pytest.approx(max(alone["y"]), rel=1e-9)

    def test_an_imaginary_mode_is_never_in_the_sum(self):
        """§ 6.4 / § 9 — adding one leaves the envelope unchanged, in both pictures."""
        real_only = curve([mode(1, 1000.0, 10.0)], 20)
        plus_imaginary = curve(
            [mode(1, 1000.0, 10.0), mode(2, 300.0, 40.0, imaginary=True)], 20
        )
        assert max(plus_imaginary["y"]) == pytest.approx(max(real_only["y"]), rel=1e-9)
        assert min(plus_imaginary["x"]) == pytest.approx(min(real_only["x"]), rel=1e-9)

    def test_an_imaginary_mode_is_out_of_the_density_picture_too(self):
        """§ 9 — 'in either picture'."""
        got = curve([mode(1, 1000.0), mode(2, 300.0, imaginary=True)], 20)
        assert max(got["y"]) == pytest.approx(1.0, rel=0.02)

    @pytest.mark.parametrize("width", [2, 20, 200])
    def test_a_peak_carries_at_least_eight_samples_across_its_full_width(self, width):
        """§ 9 — fewer and a Lorentzian shows its corners."""
        got = curve([mode(1, 1000.0, 10.0)], width)
        inside = [x for x in got["x"] if abs(x - 1000.0) <= width / 2]
        assert len(inside) >= 8, f"only {len(inside)} samples across the peak"

    @pytest.mark.parametrize("width", [2, 20, 200])
    def test_the_curve_ends_below_one_percent_of_the_tallest_peak(self, width):
        """§ 9 — the grid runs out rather than cutting a peak off mid-flight."""
        got = curve([mode(1, 1000.0, 10.0), mode(2, 1200.0, 3.0)], width)
        tallest = max(got["y"])
        assert got["y"][0] <= 0.01 * tallest
        assert got["y"][-1] <= 0.01 * tallest

    def test_the_grid_follows_the_width(self):
        """§ 9 — where the curve is sampled follows the width, not a fixed setting."""
        narrow = curve([mode(1, 1000.0, 10.0)], 4)
        broad = curve([mode(1, 1000.0, 10.0)], 40)
        step_narrow = narrow["x"][1] - narrow["x"][0]
        step_broad = broad["x"][1] - broad["x"][0]
        assert step_broad == pytest.approx(10 * step_narrow, rel=1e-6)


# --- § 6.3  the band bends, the picture does not ----------------------------

def test_the_floor_widens_the_band_without_touching_the_curve():
    """§ 6.3 — where the two must differ it is the invisible one that gives way."""
    modes = [mode(1, 1000.0, 10.0), mode(2, 1400.0, 6.0)]
    assert band([1000.0, 1400.0], 3) == [8, 8]          # the band is widened to the floor
    drawn = curve(modes, 3)
    gamma = 1.5                                          # ... and the curve is still 3 cm-1 wide
    expected = 10.0 + 6.0 * (gamma**2 / (400.0**2 + gamma**2))
    i = min(range(len(drawn["x"])), key=lambda k: abs(drawn["x"][k] - 1000.0))
    assert drawn["y"][i] == pytest.approx(expected, rel=0.02)
