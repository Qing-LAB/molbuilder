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

**Two questions, two rules.**  Which modes are active WITHIN a channel
is decided relatively -- a fraction of the strongest band in that same
channel -- because the channels carry incommensurate units (Å⁴/amu for
Raman, km/mol for IR) and no single epsilon can mean the same thing in
both.  The CO2 numbers sit ten orders of magnitude apart, so that
fraction is not a close call.

Whether the channel contains a band AT ALL cannot be answered that way:
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

from typing import Dict, List, Optional, Sequence

#: Fraction of the strongest band in the same channel, below which a
#: mode counts as inactive.  1e-6 sits far above the residue measured on
#: CO2 (ratios of order 1e-10) and far below any band a spectroscopist
#: would call weak -- a 1-in-a-million band is not a band.
ACTIVITY_REL_FLOOR = 1e-6

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
    the value clears ``rel_floor`` times the channel's strongest band.

    A channel whose every computed value is zero (or whose only entries
    are ``None``) yields no actives -- there is no peak to be a fraction
    of, and calling the largest residue "the strongest band" would
    promote noise to signal on exactly the runs where nothing happened.
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
    cut = rel_floor * peak
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
