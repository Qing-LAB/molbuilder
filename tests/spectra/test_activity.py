"""The activity decision: which modes are IR- and Raman-active.

Why this file exists.  A symmetry-forbidden intensity is zero in theory
and ~1e-9 on disk, so somebody has to decide where the line is.  The
plan's rule for W21 is that the decision has ONE home and is stored --
never re-invented as an epsilon in the viewer, where it could not be
tested against a real run.  These tests are that run.
"""
from __future__ import annotations

import pytest

from molbuilder.spectra.activity import (
    CLASS_BOTH, CLASS_IR_ONLY, CLASS_PARTIAL, CLASS_RAMAN_ONLY,
    CLASS_SILENT, adaptive_cut, channel_activity, classify_modes,
)


def test_co2_comes_back_mutually_exclusive():
    """The real CO2 run, and the rule of mutual exclusion.

    CO2 is centrosymmetric, so no mode may be both IR- and Raman-active
    -- a textbook selection rule, and the sharpest available check that
    the cut lands in the right place.  The numbers are the ones the W21
    design walk measured (2026-09-11): two bends and an asymmetric
    stretch carrying the IR, a symmetric stretch carrying the Raman, and
    residues from 3.6e-09 to 7.5e-08 standing for the forbidden halves.

    A cut placed too high loses the 14.74 Raman band; too low, and every
    residue becomes a band and every mode reads "both" -- which would
    break the selection rule this molecule exists to demonstrate.
    """
    ir = [32.85, 32.85, 4.8e-09, 613.04]
    raman = [1.0e-08, 7.5e-08, 14.74, 3.6e-09]

    rows = classify_modes(ir, raman)

    assert [r["activity_class"] for r in rows] == [
        CLASS_IR_ONLY, CLASS_IR_ONLY, CLASS_RAMAN_ONLY, CLASS_IR_ONLY,
    ]
    assert not any(r["activity_class"] == CLASS_BOTH for r in rows), (
        "a centrosymmetric molecule cannot have a mode active in both "
        "channels -- the cut has let residue through as signal")


def test_a_channel_that_was_never_computed_is_not_silence():
    """``None`` and zero are different statements about the world.

    "We did not compute the IR" must never render as "this mode has no
    IR": the rug colours those differently, and conflating them would
    let a frequencies-only run look like a full silence measurement.
    """
    rows = classify_modes([None, None], [14.74, 3.6e-09])

    assert [r["ir_active"] for r in rows] == [None, None]
    assert [r["raman_active"] for r in rows] == [True, False]
    assert [r["activity_class"] for r in rows] == [CLASS_PARTIAL,
                                                   CLASS_PARTIAL]


def test_a_mode_active_in_both_channels_is_reported_as_both():
    """Water is not centrosymmetric; mutual exclusion does not apply."""
    rows = classify_modes([55.0, 27.0], [9.1, 4.4])
    assert [r["activity_class"] for r in rows] == [CLASS_BOTH, CLASS_BOTH]


def test_an_all_zero_channel_promotes_nothing():
    """With no peak there is no fraction to clear.

    Taking the largest residue as "the strongest band" would make noise
    active on exactly the runs where nothing happened -- a mode list of
    pure numerical dust would come back fully active.
    """
    assert channel_activity([1e-9, 4e-9, 2e-9],
                            present_floor=1e-3) == [False, False, False]
    assert channel_activity([0.0, 0.0]) == [False, False]
    assert channel_activity([None, None]) == [None, None]


def test_silence_requires_both_channels_measured():
    rows = classify_modes([1e-9], [1e-9])
    assert rows[0]["activity_class"] == CLASS_SILENT


def test_discrimination_is_relative_but_presence_is_not():
    """Two different jobs, done by two different rules -- on purpose.

    WITHIN a channel that has a band, deciding which modes are active is
    RELATIVE, so it cannot depend on units: scaling a whole run up must
    not move a single mode across the line.

    Deciding whether the channel has a band AT ALL cannot be relative --
    there is nothing to be a fraction of -- so that gate is absolute and
    unit-bearing.  Scaling a run down until its strongest band is below
    observability therefore DOES change the answer, and should: a
    spectrum whose brightest line is 1e-9 km/mol has no lines.
    """
    ir = [32.85, 4.8e-09, 613.04]
    raman = [1.0e-08, 14.74, 3.6e-09]
    base = [r["activity_class"] for r in classify_modes(ir, raman)]

    for scale in (1e3, 1e6, 1e9):
        scaled = [r["activity_class"] for r in
                  classify_modes([v * scale for v in ir],
                                 [v * scale for v in raman])]
        assert scaled == base, (
            f"scaling UP by {scale} moved a mode across the line -- "
            f"within-channel discrimination must be unit-free")

    # ...and below observability, the bands are gone, not rescaled.
    faint = [r["activity_class"] for r in
             classify_modes([v * 1e-9 for v in ir],
                            [v * 1e-9 for v in raman])]
    assert set(faint) == {CLASS_SILENT}, (
        "a run whose strongest band is below the presence floor still "
        "reported active modes")


def test_mismatched_channel_lengths_raise():
    """One list of modes means one length; zipping to the shorter one
    would silently drop the tail of a run."""
    with pytest.raises(ValueError, match="channel lengths disagree"):
        classify_modes([1.0, 2.0], [1.0])


# --------------------------------------------------------------------- #
#  The cut is found in the data, not asserted                            #
# --------------------------------------------------------------------- #

def test_symmetry_breaking_leakage_is_not_a_band():
    """The defect a fixed 1e-6 floor shipped, and the case that found it.

    Ethylene built through the UI is an RDKit geometry relaxed to a
    finite gradient tolerance, so it is not exactly D2h.  Its two
    IR-active C-H stretches leak into Raman at 3.3e-06 and 1.5e-06 of
    the Raman peak -- above a 1e-6 line, so both were labelled active in
    BOTH channels, which centrosymmetry forbids outright.

    The numbers below are that run's.  What separates them is not their
    size but the GAP: three clear decades between the weakest real band
    (6.5e-03) and the strongest leak (3.3e-06).
    """
    raman = [212.9, 143.1, 37.31, 10.57, 1.721, 1.381,     # allowed
             7.07e-4, 3.25e-4,                             # leakage
             2.4e-7, 1.2e-7, 8.4e-7, 3.2e-8]               # residue
    active = channel_activity(raman, present_floor=1e-3)
    assert sum(bool(a) for a in active) == 6, (
        "only the six genuinely Raman-active modes may come back active; "
        "symmetry-breaking leakage is not a band")


def test_the_same_molecule_classifies_the_same_on_a_worse_geometry():
    """The property a fixed threshold could not hold.

    Where the residue sits depends on the CALCULATION -- geometry,
    basis, grid, convergence.  Whether a mode is allowed depends on the
    SYMMETRY.  A rule that reads the separation rather than the level
    therefore gives one answer for one molecule, and that is the whole
    argument for finding the cut in the data.

    Same six bands, residue two orders apart.
    """
    bands = [212.9, 143.1, 37.31, 10.57, 1.721, 1.381]
    clean = channel_activity(bands + [2.4e-7, 1.2e-7, 8.4e-7,
                                      3.2e-8, 6.8e-7, 3.1e-7],
                             present_floor=1e-3)
    noisy = channel_activity(bands + [7.07e-4, 3.25e-4, 2.4e-7,
                                      1.2e-7, 8.4e-7, 3.2e-8],
                             present_floor=1e-3)
    assert sum(bool(a) for a in clean) == sum(bool(a) for a in noisy) == 6


def test_a_channel_with_nothing_to_separate_is_left_alone():
    """Water: three modes, all allowed, widest gap 0.7 decades.

    A cut would have to invent a boundary, and the weakest of three real
    bands would be the one it threw away.  "Everything is active" is the
    right answer and the rule has to be able to reach it.
    """
    assert adaptive_cut([1.0, 0.479, 0.0886]) is None
    assert all(channel_activity([76.91, 36.82, 6.817],
                                present_floor=1e-3))


def test_a_gap_too_high_up_is_not_the_residue_boundary():
    """The way a widest-gap rule can be fooled, and the guard for it.

    A molecule with no forbidden modes but a wide dynamic range has its
    widest gap BETWEEN TWO REAL BANDS.  Cutting there would call a
    genuine band forbidden, so a cut above a thousandth of the peak is
    refused and the fixed floor is used instead.
    """
    assert adaptive_cut([1.0, 1e-4, 8e-5, 7e-5]) is None   # gap at 1e-2
    assert sum(bool(a) for a in
               channel_activity([1.0, 1e-4, 8e-5, 7e-5],
                                present_floor=0.0)) == 4


def test_a_narrow_gap_low_down_is_still_not_a_separation():
    """Isolates the minimum-width rule, which nothing else reaches.

    The water case above is refused by the OTHER guard -- its gap sits
    high, so the cut would land above a thousandth of the peak.  This
    one is built so the widest gap is genuinely low (a cut at ~1.3e-04,
    comfortably inside the permitted region) but only 1.8 decades wide.

    That is a continuum of weak bands, not two clusters: the intensities
    step down steadily with nothing that reads as a boundary.  Splitting
    it anywhere would discard real bands, so the rule must decline on
    WIDTH alone.
    """
    # The last value sits just ABOVE the fallback floor (1e-6), so the
    # fallback keeps every mode -- the point here is the refusal to
    # split, not where the fallback happens to land.
    ratios = [1.0, 10 ** -1.5, 10 ** -3.0, 10 ** -4.8, 10 ** -5.9]
    assert adaptive_cut(ratios) is None, (
        "a 1.8-decade gap is a slope, not a separation")
    # ...so every mode survives, on the fixed floor.
    assert sum(bool(a) for a in
               channel_activity([r * 50.0 for r in ratios],
                                present_floor=0.0)) == 5
