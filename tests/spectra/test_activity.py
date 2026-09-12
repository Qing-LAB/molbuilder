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
    CLASS_SILENT, channel_activity, classify_modes,
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
