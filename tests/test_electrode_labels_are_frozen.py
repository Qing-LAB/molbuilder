"""The FIRST validation of a junction — are the labeled electrodes frozen?

`engines/transport.md` § 4 and the one-structure design *(user, 2026-09-16)*:
a junction is ONE file carrying the frozen leads at both ends and the relaxed
bridge between them, and transport takes the lead atoms **out of that file**
by their label.  The leads must come through the relaxation untouched, or the
self-energies attach to a geometry that is not the bulk they claim to be.

**What this is not.**  `transport.wizard.extract_electrode_model` already
refuses a lead that is not frozen, and one that moved.  Those gates are correct
and they are also too late: they run when transport composes, after the
relaxation has been paid for.  This asks the same question one step earlier,
where it is still cheap to act on — before the run, not after it.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.validation.sidecar import check_electrode_labels_are_frozen


def _junction(*, frozen):
    """Au | C | Au with both ends labeled as leads."""
    s = Structure(elements=["Au", "Au", "C", "Au", "Au"],
                  positions=np.zeros((5, 3)))
    s.regions = {"L-electrode": [0, 1], "bridge": [2], "R-electrode": [3, 4]}
    s.frozen_atoms = list(frozen)
    return s


def test_an_unfrozen_lead_is_named_atom_by_atom():
    """The whole point is that a person can act on it: which atoms, and
    what happens if they do not."""
    issues = check_electrode_labels_are_frozen(_junction(frozen=[0, 1]))
    assert len(issues) == 1
    msg = issues[0].message
    assert "3 (Au)" in msg and "4 (Au)" in msg, (
        f"the unfrozen lead atoms must be named: {msg}")
    assert "frozen_atoms" in msg, "and the fix must be named"


def test_a_fully_frozen_junction_is_clean():
    """The half without which 'it warns' would be satisfied by warning
    always."""
    assert check_electrode_labels_are_frozen(_junction(frozen=[0, 1, 3, 4])) == []


def test_a_structure_with_no_leads_is_not_this_check_s_business():
    """An ordinary molecule carries no electrode label and must not be
    told about transport."""
    s = Structure(elements=["C", "H"], positions=np.zeros((2, 3)))
    s.regions = {"bridge": [0, 1]}
    assert check_electrode_labels_are_frozen(s) == []


def test_the_bridge_is_not_required_to_be_frozen():
    """THE DISCRIMINATING CASE.  A junction's whole purpose is that the
    bridge relaxes while the leads do not, so a check that demanded every
    labeled atom be frozen would refuse every correct junction."""
    assert check_electrode_labels_are_frozen(_junction(frozen=[0, 1, 3, 4])) == []
    # ...and it is the LEADS it asks about, not the count of frozen atoms:
    partly = _junction(frozen=[0, 1, 2, 3])        # bridge frozen, one lead not
    assert len(check_electrode_labels_are_frozen(partly)) == 1


def test_it_warns_rather_than_refusing_and_the_line_is_deliberate():
    """A structure carrying electrode labels is heading for transport but
    has not committed: a person may deliberately relax the whole junction
    once before freezing the leads for the run that counts.  Refusing here
    would block that.  The REFUSAL belongs at compose, where transport IS
    the intent, and it is there --
    `wizard.extract_electrode_model` (tests/test_transport_wizard.py)."""
    assert check_electrode_labels_are_frozen(_junction(frozen=[]))[0].severity \
        == "warn"


def test_it_reaches_a_real_siesta_preflight():
    """Wired, not merely written — a check nothing calls is a check that
    does not exist."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    issues = validate(_junction(frozen=[0, 1]),
                      SiestaConfig(system_label="j"))
    assert any(i.where == "structure.electrode_frozen" for i in issues), (
        "the gate must run it; wiring is the half that makes it exist")
