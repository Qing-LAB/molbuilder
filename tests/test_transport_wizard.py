"""Electrode EXTRACTION — pulling a lead out of a labelled junction.

What the composite rests on: `compose.py` calls
`extract_electrode_model` and prep writes the result's deck."""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.transport.wizard import extract_electrode_model


def _au_device():
    """A toy [L-electrode][bridge][R-electrode] junction along z.

    Each electrode = 4 Au layers spaced 2.35 Å (one atom/layer for the
    test); a 2-atom S-S bridge sits between them.  Contiguous ordering
    so the device emitter is happy.
    """
    d = 2.35
    elems, pos, left, bridge, right = [], [], [], [], []
    i = 0
    for layer in range(4):                       # left electrode
        elems.append("Au"); pos.append([1.0, 1.0, layer * d]); left.append(i); i += 1
    z0 = 4 * d
    for j in range(2):                           # bridge
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); bridge.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for layer in range(4):                       # right electrode
        elems.append("Au"); pos.append([1.0, 1.0, z1 + layer * d]); right.append(i); i += 1
    zs = np.asarray(pos, dtype=float)[:, 2]
    # ITS CELL, as every junction states one (`engines/transport.md`
    # § 2a.9): a one-atom chain is a wire, isolated across in an 8 A box, and the transport boundary
    # leaves one layer spacing of room.
    s = Structure(
        elements=elems, positions=np.asarray(pos, dtype=float),
        regions={"L-electrode": left, "bridge": bridge, "R-electrode": right},
        cell=np.diag([8.0, 8.0, float(zs.max() - zs.min()) + d]),
        axis_kind=("isolated", "isolated", "transport"))
    # The leads are frozen, as a junction's are (`engines/transport.md` § 4);
    # the bridge is deliberately left free, so a gate asking about every atom
    # rather than the leads would fail here.
    s.frozen_atoms = left + right
    return s


# --------------------------------------------------------------------- #
#  extraction: the clone guarantees                                     #
# --------------------------------------------------------------------- #


def test_extract_clones_atoms_and_uses_device_lateral_cell():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode")
    assert len(m.elements) == 4
    assert set(m.elements) == {"Au"}
    assert m.positions[:, 2].min() == pytest.approx(0.0)   # shifted to 0
    # lateral cell is the DEVICE cell, not the electrode's own (1.0) extent
    lead = m.as_structure()
    assert np.asarray(lead.cell)[:2] == pytest.approx(
        np.asarray(dev.cell)[:2])


# --------------------------------------------------------------------- #
#  the gate: what disqualifies a labelled block from being a lead       #
# --------------------------------------------------------------------- #
#
# One function decides this, so the tests sit on the extraction: it is
# where the answer lives, for every caller and both citation forms.


def test_an_unfrozen_lead_is_refused_and_told_to_freeze():
    """The refusal a person acts on: WHICH atoms, and what to do."""
    dev = _au_device()
    dev.frozen_atoms = []
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode")
    msg = str(e.value)
    assert "NOT FROZEN" in msg
    assert "atom indices 0 (Au)" in msg, f"the atoms must be named: {msg}"
    assert "freeze them" in msg, f"and the fix must be named: {msg}"


def test_one_unfrozen_atom_is_enough():
    """A lead is bulk or it is not; there is no mostly-frozen lead."""
    dev = _au_device()
    dev.frozen_atoms = [0, 1, 2]          # the 4th Au of L is loose
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode")
    assert "1 atom(s)" in str(e.value) and "atom index 3 (Au)" in str(e.value)


def test_the_bridge_is_not_required_to_be_frozen():
    """THE DISCRIMINATING CASE.  A junction's whole point is that the
    bridge relaxes while the leads do not, so a gate that demanded every
    atom be frozen would refuse every correct junction."""
    dev = _au_device()                     # bridge atoms 4, 5 are free
    assert len(extract_electrode_model(dev, "L-electrode").elements) == 4


def test_a_lead_that_moved_is_refused_when_a_starting_geometry_is_given():
    """`frozen` is a declaration; this is the outcome.  Both are asked,
    because a label can carry the declaration and the relaxation can
    still have moved the atom."""
    dev = _au_device()
    prior = np.asarray(dev.positions, dtype=float).copy()
    prior[2, 2] -= 0.05
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode", prior_positions=prior)
    msg = str(e.value)
    assert "MOVED" in msg and "atom index 2 (Au)" in msg and "0.0500" in msg


def test_an_unmoved_lead_passes_with_a_starting_geometry():
    """The half without which 'it refuses' would be satisfied by
    refusing always."""
    dev = _au_device()
    prior = np.asarray(dev.positions, dtype=float).copy()
    m = extract_electrode_model(dev, "L-electrode", prior_positions=prior)
    assert m.z_period == pytest.approx(4 * 2.35)


def test_an_unevenly_spaced_lead_is_refused():
    """Wired, not merely present in `cell`: the extraction is where a
    caller meets this, and the refusal names the spacings."""
    dev = _au_device()
    pos = np.asarray(dev.positions, dtype=float)
    pos[3, 2] += 0.6                       # the 4th layer of L, pushed out
    dev.positions = pos
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode")
    msg = str(e.value)
    assert "not evenly spaced" in msg
    assert "L-electrode" in msg, (
        f"the refusal must say WHICH block -- on a two-lead junction of "
        f"one element there is otherwise no way to tell which end to "
        f"re-label: {msg}")


def test_frozen_is_asked_before_moved():
    """The ORDER is a dependency: an unfrozen lead's answer to 'did it
    move' is noise, and freezing is the fix for both.  A block that is
    neither frozen nor unmoved must be told the first thing."""
    dev = _au_device()
    dev.frozen_atoms = []
    prior = np.asarray(dev.positions, dtype=float).copy()
    prior[2, 2] -= 0.05
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode", prior_positions=prior)
    msg = str(e.value)
    assert "NOT FROZEN" in msg and "MOVED" not in msg


# --------------------------------------------------------------------- #
#  the seam: MEASURED and REPORTED, never refused                       #
# --------------------------------------------------------------------- #
#
# `junction-cell.md` § 3.1: a lead tiles along z, so its top layer meets
# its own bottom layer one cell up, and only a whole number of stacking
# periods continues the crystal -- a multiple of 3 on fcc(111).  Four
# layers puts the image's first layer back on the same sites (ECLIPSED, a
# head-on metal contact), five gives a mirror TWIN.
#
# NOT A REFUSAL, on the same line the electrode ORIENTATION is drawn on
# (`blueprints/transport.py`: "THE CONVENTION IS CHECKED AND REPORTED,
# NEVER ENFORCED", user ruling 2026-08-29).  The person owns the science;
# what they are owed is the measurement.


def _au111_junction(n_layers) -> dict:
    """A junction whose leads are built the way the app builds one --
    `modify.add_slab`, one grown down from a sulfur and one up: the leads held,
    in the cell the slab API gave them, the transport axis closed to one layer spacing past the top
    lead.

    THROUGH THE REAL SLAB API, not `ase.build.fcc111` by hand (user,
    2026-09-21: *"Do you even know that we actually have a slab creation
    API for all of this?"*).  The verdicts are identical either way —
    checked — but a fixture that hand-rolls the builder is testing a
    geometry nobody's workflow produces, and the whole subject here is
    what a real lead does at its periodic boundary.
    """
    from molbuilder.modify import add_slab
    seed = Structure(elements=["S"], positions=np.array([[0.0, 0.0, 30.0]]))
    left = add_slab(seed, element="Au", plane="111", size=(2, 2, n_layers),
                    grow="-z")
    st = add_slab(left, element="Au", plane="111", size=(2, 2, n_layers),
                  start_z=5.0, grow="+z")
    pos = np.asarray(st.positions, dtype=float).copy()
    pos[0, 2] = 2.5                             # the sulfur, between the leads
    pos[:, 2] -= pos[:, 2].min()
    n = 4 * n_layers
    # Listed lower lead, sulfur, upper lead, as the engine reads a junction.
    order = list(range(1, n + 1)) + [0] + list(range(n + 1, 2 * n + 1))
    layers = np.unique(np.round(pos[order[:n], 2], 4))
    cell = np.asarray(st.cell, dtype=float).copy()
    cell[2] = [0.0, 0.0, float(pos[:, 2].max()) + layers[1] - layers[0]]
    return {"elements": [st.elements[i] for i in order],
            "positions": pos[order].tolist(),
            "cell": cell.tolist(),
            "axis_kind": ["periodic", "periodic", "transport"],
            "regions": {"L-electrode": list(range(n)), "bridge": [n],
                        "R-electrode": list(range(n + 1, 2 * n + 1)),
                        "frozen_atoms": [i for i in range(2 * n + 1) if i != n]}}


def _lead_model(n_layers):
    """What the extraction makes of that junction's L lead."""
    from support.road import road_structure
    return extract_electrode_model(road_structure(_au111_junction(n_layers)),
                                   "L-electrode")


@pytest.mark.parametrize("n_layers,verdict,severity", [
    (3, "CONTINUES", "info"), (4, "ECLIPSED", "warn"), (5, "TWIN", "warn"),
    (6, "CONTINUES", "info"),
])
def test_the_seam_verdict_is_measured(n_layers, verdict, severity):
    """The whole table, because the interesting half is the failures and
    a test on 3 and 6 alone would pass with the check deleted: what the
    Transport card says of the lead's boundary (the lead's notes).  That prep
    says the same is `test_transport_decks_e2e.py`'s, a transport deck needing
    a real relaxation."""
    m = _lead_model(n_layers)
    seam = [n for n in m.notes if n.where == "lead.seam"]
    assert len(seam) == 1, f"exactly one seam note: {m.notes}"
    assert verdict in seam[0].message, f"{n_layers} layers -> {seam[0]}"
    assert seam[0].severity == severity, seam[0]


def test_a_faulted_seam_is_REPORTED_not_refused():
    """The discriminating half.  A 4-layer lead is not bulk and still
    composes -- the person may be doing exactly what they meant."""
    m = _lead_model(4)
    assert m.z_period == pytest.approx(9.4182, abs=1e-3)
    assert len(m.elements) == 2 * 2 * 4


def test_the_faulted_seam_note_names_the_LAYER_COUNT_as_the_cause():
    """'Your seam is eclipsed' is not actionable; 'four layers is not a
    whole number of 3-layer periods' is."""
    note = next(n.message for n in _lead_model(4).notes
                if n.where == "lead.seam")
    assert "LAYER COUNT" in note and "4 layers" in note and "3-layer" in note, note


def test_the_lead_states_the_DEVICE_s_transverse_periodicity():
    """A lead changes ONE axis: transport.  Across the wire it is whatever
    the device is.

    A nanowire or chain lead is vacuum-surrounded across the wire, and
    declaring those directions periodic has the shared transverse k-mesh
    sample vacuum (user, 2026-09-23).
    """
    def _lead_of(cell, kinds):
        """Both devices stated outright rather than taken from the module
        fixture, whose cell-less default is one of the two answers -- so a
        test reading only the other would pass on an accident."""
        dev = _au_device()
        dev.cell = cell
        dev.axis_kind = kinds
        return extract_electrode_model(dev, "L-electrode").as_structure()

    slab = _lead_of(np.array([[17.30, 0.0, 0.0],      # hexagonal Au(111)
                              [8.65, 14.98, 0.0],
                              [0.0, 0.0, 40.0]]),
                    ("periodic", "periodic", "transport"))
    assert slab.axis_kind == ("periodic", "periodic", "periodic")
    assert slab.cell[1][1] == pytest.approx(14.98), (
        "and the 60 degree vector is still carried verbatim -- I6")

    wire = _lead_of(np.diag([30.0, 30.0, 40.0]),
                    ("isolated", "isolated", "transport"))
    assert wire.axis_kind == ("isolated", "isolated", "periodic"), (
        "the lead is periodic along transport -- that is what makes it a "
        "lead -- and isolated across, because the device is")
