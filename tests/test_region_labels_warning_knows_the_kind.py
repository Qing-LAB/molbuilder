"""The unconsumed-label warning must know WHICH calculation it is warning about.

`engines/transport.md` § 4: the region partition is what the entire five-rung
ladder is built from.  § 3.6a recorded, on 2026-09-16, that this warning was
"wrong for transport" and left it standing -- so every transport deck's
`.validation.txt` told the person their `L-electrode`/`bridge`/`R-electrode`
labels "do NOT consume ... do not shape this calculation".

The warning exists to stop someone believing their labels mattered when they
did not.  Saying it where they DO is the same defect pointed the other way,
and it is worse: it is printed beside the deck, at the moment of checking.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import validate


def _junction():
    return Structure(
        elements=["Au", "Au", "S", "Au", "Au"],
        positions=np.array([[0., 0, z] for z in (0, 2.4, 5.0, 7.6, 10.0)]),
        cell=np.diag([12., 12., 14.4]),
        axis_kind=("periodic", "periodic", "transport"),
        regions={"L-electrode": [0, 1], "bridge": [2], "R-electrode": [3, 4]})


def _named(struct, calculation):
    """The labels the warning calls unconsumed, or None when it stays quiet."""
    for i in validate(struct, SiestaConfig(system_label="t"),
                      calculation=calculation):
        if i.where == "structure.regions":
            return i.message
    return None


def test_transport_does_not_call_its_own_partition_unconsumed():
    assert _named(_junction(), "transport") is None


def test_an_optimization_still_names_them_because_it_reads_none():
    """The warning's original and correct job -- an optimization deck reads
    no region label, so silence would let a person believe otherwise."""
    msg = _named(_junction(), "optimization")
    assert msg is not None
    for label in ("L-electrode", "bridge", "R-electrode"):
        assert label in msg


def test_transport_still_names_a_label_it_genuinely_cannot_read():
    """§ 4's own rule survives: "a label this engine does not consume is
    WARNED about, never dropped in silence".  Making the check kind-aware
    must not turn it into silence for transport."""
    s = _junction()
    s.regions["my-notes"] = [2]
    msg = _named(s, "transport")
    assert msg is not None and "my-notes" in msg
    # The NAMED list is what matters, not the whole sentence: the advice
    # that follows it legitimately spells out the partition it does read.
    named = msg.split("region label(s)")[1].split("which the")[0]
    for consumed in ("L-electrode", "bridge", "R-electrode"):
        assert consumed not in named


def test_a_named_electrode_is_consumed_by_the_suffix_convention():
    """`tip-electrode` is a lead without a code change (§ 4), so the check
    asks `is_electrode_label` rather than matching the canonical four."""
    s = _junction()
    s.regions["tip-electrode"] = [2]
    assert _named(s, "transport") is None
