"""Is a mode active in a channel?  One home for that decision.

A symmetry-forbidden mode has an intensity of exactly zero.  What
reaches ``.spectra.json`` is not zero -- it is floating-point residue
from a finite-difference derivative or a CPHF solve.  Measured on CO2
(2026-09-11, the run W21 was designed against): the silent entries came
back as ``4.8e-09``, ``1.0e-08``, ``7.5e-08`` and ``3.6e-09`` beside
bands of ``32.85`` and ``613.04`` km/mol.

So "is this IR-active" is a DECISION, not a read, and this module is
where it is made.  The rule must not be re-invented as an epsilon in the
viewer: two epsilons drift, and the one in the viewer cannot be tested
against a real run.

**Two questions, and the first is answered by looking.**

WHICH MODES ARE ACTIVE within a channel is decided by finding the
boundary in the data, not by asserting one.  Allowed and forbidden modes
do not merely differ, they CLUSTER -- orders apart with nothing in
between -- so the cut is the widest gap between consecutive intensities
in log space (``adaptive_cut``).  That is the primary rule, because
where the residue sits is a property of the CALCULATION (geometry,
basis, grid, convergence) while what we want to read is a property of
the SYMMETRY: measured across four real runs the derived cut ranged
2.6e-06 to 1.5e-04, and the same rule classified ethylene identically on
two geometries whose residue differed by two orders.

``ACTIVITY_REL_FLOOR`` is the FALLBACK for when the channel will not
separate itself -- no gap wide enough to trust, or the widest gap sitting
too high to be the band/residue boundary.  It is relative for the reason
any cut here must be: the channels carry incommensurate units (Å⁴/amu
for Raman, km/mol for IR) and no single epsilon can mean the same thing
in both.

WHETHER THE CHANNEL CONTAINS A BAND AT ALL cannot be answered by either:
with only residue to look at, the largest residue becomes the reference
and every mode is promoted.  That question needs an absolute,
unit-bearing floor, and it is the only place one appears -- see
``CHANNEL_PRESENT_FLOOR_*``.  The consequence is deliberate and tested:
scaling a run UP never moves a mode across the line, while scaling it
below observability empties the channel.

**Not-computed is not silent.**  ``None`` means the channel was never
asked for; a mode that is genuinely inactive is a different statement
about the world, and the rug in the Results view colours them
differently.  Every function here keeps the two apart.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

#: THE FALLBACK cut, used only when the data will not separate itself.
#:
#: It was the primary rule until 2026-09-11, and a real run showed why a
#: constant cannot be: on ethylene built through the UI -- an RDKit
#: geometry relaxed to a finite gradient tolerance, so not exactly
#: D2h -- symmetry-breaking leakage reached 1.5e-6 and 3.3e-6 of the
#: Raman peak, crossing a 1e-6 line and labelling two honestly IR-only
#: C-H stretches as active in both channels.  The SAME molecule on an
#: exactly symmetric geometry leaked at 1e-7 and classified correctly.
#: One ruler cannot serve both, because where the residue sits depends
#: on the geometry, the basis, the grid and the convergence tolerance --
#: properties of the calculation, not of the chemistry.
ACTIVITY_REL_FLOOR = 1e-6

#: HOW WIDE A GAP HAS TO BE before it counts as a separation.
#:
#: Allowed and forbidden modes do not merely differ, they cluster: on
#: every real run measured, the two groups sit ORDERS apart with nothing
#: in between (CO2 8.6-9.2 decades, ethylene 3.3-7.0).  Below two
#: decades there is no bimodality to find and splitting would be
#: inventing a boundary -- which is what protects a molecule whose modes
#: are all allowed.  Water's widest gap is 0.7 decades, and the right
#: answer there is "everything is active", not "the weakest one isn't".
MIN_GAP_DECADES = 2.0

#: AND WHERE A SPLIT MAY LAND.
#:
#: The widest gap is usually the one between bands and residue, but not
#: always: a molecule with no forbidden modes and a wide dynamic range
#: could have its widest gap sit BETWEEN TWO REAL BANDS, and cutting
#: there would call a genuine band forbidden.  Nothing above a thousandth
#: of the strongest band is residue, so a cut above this is refused and
#: the fallback is used instead.  Every cut derived from a real run so
#: far lands in 2.6e-06 .. 1.5e-04, well inside it.
MAX_CUT_FRACTION = 1e-3

#: DOES THIS CHANNEL CONTAIN A BAND AT ALL?
#:
#: The relative rule discriminates within a channel, but it cannot tell
#: a channel full of bands from a channel full of dust: given only
#: residue it takes the largest residue as "the strongest band" and
#: promotes every mode to active.  That is not hypothetical -- a
#: homonuclear diatomic has one mode and no dipole derivative, so its
#: entire IR channel is residue.
#:
#: Deciding THAT question is the one place a unit-bearing threshold is
#: unavoidable, so each channel carries its own, set where a band stops
#: being observable rather than where the arithmetic gets small:
#:   * IR:    km/mol.  Weak-but-real bands run 0.1-1; nothing at 1e-3 is
#:            measurable.
#:   * Raman: Å⁴/amu.  Same reasoning on the Placzek scale.
#: Used ONLY as a presence gate.  Discrimination between modes stays
#: relative, so the floors never have to be re-derived per basis or
#: functional -- only re-checked if the units ever change.
CHANNEL_PRESENT_FLOOR_IR_KM_MOL = 1e-3
CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU = 1e-3

#: Activity classes, in the order the Results rug lists them.
CLASS_BOTH = "both"
CLASS_IR_ONLY = "ir-only"
CLASS_RAMAN_ONLY = "raman-only"
CLASS_SILENT = "silent"
#: At least one channel was not computed, so the mode cannot be placed.
CLASS_PARTIAL = "partial"

CLASSES = (CLASS_BOTH, CLASS_IR_ONLY, CLASS_RAMAN_ONLY,
           CLASS_SILENT, CLASS_PARTIAL)


def adaptive_cut(
    ratios: Sequence[float],
    *,
    min_gap_decades: float = MIN_GAP_DECADES,
    max_cut_fraction: float = MAX_CUT_FRACTION,
) -> Optional[float]:
    """Where this channel separates itself, or ``None`` if it does not.

    ``ratios`` are intensities divided by the channel's strongest band,
    so they run from 1.0 downward.  Allowed and forbidden modes cluster
    ORDERS apart, so the boundary is simply the widest gap between
    consecutive values in log space, cut at its midpoint.

    Asking the data beats asserting a constant because the residue's
    position is a property of the CALCULATION -- geometry, basis, grid,
    convergence -- while the separation is a property of the SYMMETRY,
    which is what we actually want to read.  Measured across four real
    runs the derived cut ranged 2.6e-06 .. 1.5e-04, a sixtyfold spread
    no single constant sits correctly inside; yet the same rule gave
    ethylene the SAME classification on an idealised geometry and on an
    RDKit one whose residue was two orders larger.

    ``None`` when no gap is wide enough (nothing to separate -- every
    mode is allowed), or when the widest gap sits too high to be the
    band/residue boundary.  Both refusals fall back to the fixed floor.
    """
    finite = [r for r in ratios if r > 0.0]
    if len(finite) < 2:
        return None
    logs = sorted((math.log10(r) for r in finite), reverse=True)
    widest, at = 0.0, None
    for i in range(len(logs) - 1):
        gap = logs[i] - logs[i + 1]
        if gap > widest:
            widest, at = gap, i
    if at is None or widest < min_gap_decades:
        return None
    cut = 10.0 ** ((logs[at] + logs[at + 1]) / 2.0)
    return cut if cut <= max_cut_fraction else None


def channel_activity(
    values: Sequence[Optional[float]],
    *,
    rel_floor: float = ACTIVITY_REL_FLOOR,
    present_floor: float = 0.0,
) -> List[Optional[bool]]:
    """Active/inactive per mode for ONE channel.

    ``values`` is that channel's intensity for every mode, in mode
    order, with ``None`` where it was not computed.  Returns a list of
    the same length: ``None`` where the input was ``None``, else whether
    the value clears this channel's cut.

    WHERE THE CUT COMES FROM, in order: the widest gap in the data
    (``adaptive_cut``), and only if the channel will not separate itself,
    ``rel_floor`` times the strongest band.  A caller reading
    ``rel_floor`` as "the threshold" has the fallback, not the rule.

    A channel whose every computed value is at or below ``present_floor``
    yields no actives -- there is no band to be a fraction of, and
    calling the largest residue "the strongest band" would promote noise
    to signal on exactly the runs where nothing happened.  Note the
    DEFAULT is 0.0, which only catches an all-zero channel; the
    unit-bearing floors live on :func:`classify_modes`, which is the
    entry point that knows which channel is which.
    """
    computed = [v for v in values if v is not None]
    peak = max((abs(v) for v in computed), default=0.0)
    if peak <= present_floor:
        # No band in this channel -- every computed entry is residue.
        # Returning "all active" here (which a purely relative cut does,
        # by making the largest residue the reference) would turn a
        # molecule with NO allowed transitions into one where every mode
        # is allowed.
        return [None if v is None else False for v in values]
    # ASK THE DATA FIRST.  The fixed floor is what we fall back to when
    # the channel will not separate itself -- see `adaptive_cut`.
    ratios = [abs(v) / peak for v in computed]
    fraction = adaptive_cut(ratios)
    cut = (fraction if fraction is not None else rel_floor) * peak
    return [None if v is None else bool(abs(v) > cut) for v in values]


def classify_mode(ir_active: Optional[bool],
                  raman_active: Optional[bool]) -> str:
    """The rug's colour for one mode.

    ``CLASS_PARTIAL`` whenever either channel is unknown: a mode that is
    Raman-inactive is only "silent" if somebody actually looked at its
    IR, and saying otherwise turns "we did not compute it" into "it is
    not there".
    """
    if ir_active is None or raman_active is None:
        return CLASS_PARTIAL
    if ir_active and raman_active:
        return CLASS_BOTH
    if ir_active:
        return CLASS_IR_ONLY
    if raman_active:
        return CLASS_RAMAN_ONLY
    return CLASS_SILENT


def classify_modes(
    ir_values: Sequence[Optional[float]],
    raman_values: Sequence[Optional[float]],
    *,
    rel_floor: float = ACTIVITY_REL_FLOOR,
) -> List[Dict[str, object]]:
    """Per-mode ``{ir_active, raman_active, activity_class}``.

    The two sequences must be in the same mode order and of the same
    length; the caller holds one list of modes, so a mismatch is a bug
    worth raising rather than zipping silently to the shorter one.
    """
    if len(ir_values) != len(raman_values):
        raise ValueError(
            f"channel lengths disagree: {len(ir_values)} IR values vs "
            f"{len(raman_values)} Raman values -- both must be one entry "
            f"per mode, in mode order")
    ir = channel_activity(ir_values, rel_floor=rel_floor,
                          present_floor=CHANNEL_PRESENT_FLOOR_IR_KM_MOL)
    raman = channel_activity(
        raman_values, rel_floor=rel_floor,
        present_floor=CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU)
    return [
        {
            "ir_active": i,
            "raman_active": r,
            "activity_class": classify_mode(i, r),
        }
        for i, r in zip(ir, raman)
    ]
