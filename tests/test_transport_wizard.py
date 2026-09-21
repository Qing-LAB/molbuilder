"""Electrode EXTRACTION — pulling a lead out of a labelled junction.

What the composite rests on: `compose.py` calls
`extract_electrode_model` and prep renders the result through the
framework (`as_structure` -> `spec_for`).  The rendering tests that
used to sit below went with `render_electrode_fdf` on 2026-09-17."""
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
    s = Structure(
        elements=elems, positions=np.asarray(pos, dtype=float),
        regions={"L-electrode": left, "bridge": bridge, "R-electrode": right})
    # THE LEADS ARE FROZEN, because a junction's are: `engines/transport.md`
    # § 4, and the extraction refuses a block that is not (2026-09-20).  The
    # bridge is deliberately left free -- that is the shape of a real
    # junction, and a fixture that froze everything would pass a gate that
    # only asks about the leads for the wrong reason.
    s.frozen_atoms = left + right
    return s


# --------------------------------------------------------------------- #
#  layer detection + bulk period                                        #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  extraction: the clone guarantees                                     #
# --------------------------------------------------------------------- #


def test_extract_clones_atoms_and_uses_device_lateral_cell():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode")
    assert m.n_atoms == 4
    assert set(m.elements) == {"Au"}
    assert m.n_layers == 4
    assert m.positions[:, 2].min() == pytest.approx(0.0)   # shifted to 0
    # lateral cell is the DEVICE cell, not the electrode's own (1.0) extent
    from molbuilder.transport.transiesta import _compute_cell_from_extents
    a, b, _c = _compute_cell_from_extents(dev)
    assert (m.cell_a, m.cell_b) == pytest.approx((a, b))


def test_extract_explicit_z_period_overrides():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode", z_period=8.5)
    assert m.z_period == pytest.approx(8.5)


def test_extract_thin_electrode_notes_warning():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode", min_thickness_ang=12.0)
    # 4 layers * 2.35 span ~7 Å < 12 -> a note is emitted
    assert any("principal layer" in n for n in m.notes)


# --------------------------------------------------------------------- #
#  the invariants hold by construction (device <-> electrode preflight)  #
# --------------------------------------------------------------------- #



# The four tests below this line went with `render_electrode_fdf` and
# `electrode_wizard` on 2026-09-17.  They rendered a lead deck and checked
# it against a device deck rendered by `TransiestaEngine.render_script` --
# two writers, both deleted.  What EXTRACTION does is still checked above,
# and that is the part `compose.py` uses: prep renders the extracted lead
# through the framework (`prep.py` -> `as_structure` -> `spec_for`).


# --------------------------------------------------------------------- #
#  the gate: what disqualifies a labelled block from being a lead       #
# --------------------------------------------------------------------- #
#
# ONE FUNCTION DECIDES THIS (user ruling, 2026-09-20: "one unified check
# and gate/extraction process").  Until then a frozen-unmoved loop in
# `compose_junction` and a tiling check in `_extract_and_gate_electrodes`
# answered two of these, and a junction handed in as a finished pair
# (compose's form B) reached neither -- so a lead that was never frozen
# produced a deck.  These tests are on the extraction because that is
# where the answer now lives, for every caller and both forms.


def test_an_unfrozen_lead_is_refused_and_told_to_freeze():
    """The refusal a person acts on: WHICH atoms, and what to do."""
    dev = _au_device()
    dev.frozen_atoms = []
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode")
    msg = str(e.value)
    assert "NOT FROZEN" in msg
    assert "0 (Au)" in msg, f"the atoms must be named: {msg}"
    assert "freeze them" in msg, f"and the fix must be named: {msg}"


def test_one_unfrozen_atom_is_enough():
    """A lead is bulk or it is not; there is no mostly-frozen lead."""
    dev = _au_device()
    dev.frozen_atoms = [0, 1, 2]          # the 4th Au of L is loose
    with pytest.raises(ValueError) as e:
        extract_electrode_model(dev, "L-electrode")
    assert "1 atom(s)" in str(e.value) and "3 (Au)" in str(e.value)


def test_the_bridge_is_not_required_to_be_frozen():
    """THE DISCRIMINATING CASE.  A junction's whole point is that the
    bridge relaxes while the leads do not, so a gate that demanded every
    atom be frozen would refuse every correct junction."""
    dev = _au_device()                     # bridge atoms 4, 5 are free
    assert extract_electrode_model(dev, "L-electrode").n_layers == 4


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
    assert "MOVED" in msg and "atom 2 (Au)" in msg and "0.0500" in msg


def test_an_unmoved_lead_passes_with_a_starting_geometry():
    """The half without which 'it refuses' would be satisfied by
    refusing always."""
    dev = _au_device()
    prior = np.asarray(dev.positions, dtype=float).copy()
    m = extract_electrode_model(dev, "L-electrode", prior_positions=prior)
    assert m.z_period == pytest.approx(4 * 2.35)


def test_without_a_starting_geometry_the_frozen_DECLARATION_still_holds():
    """Compose's form B — a labelled pair handed in as the finished
    structure — has nothing to compare against, and that is precisely why
    the declaration is asked separately.  It was the route with no frozen
    check at all before 2026-09-20."""
    dev = _au_device()
    dev.frozen_atoms = []
    with pytest.raises(ValueError, match="NOT FROZEN"):
        extract_electrode_model(dev, "L-electrode", prior_positions=None)


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
# what they are owed is the measurement.  Until 2026-09-20 they were
# owed it and did not get it: the note said "VERIFY ... a multiple of 3
# for FCC(111) ABC" -- homework about a quantity `cell.classify_seam`
# already measures and this module already imports.


def _au111_lead(n_layers):
    """A lead built the way the app builds one — `modify.add_slab`.

    THROUGH THE REAL SLAB API, not `ase.build.fcc111` by hand (user,
    2026-09-21: *"Do you even know that we actually have a slab creation
    API for all of this?"*).  The verdicts are identical either way —
    checked — but a fixture that hand-rolls the builder is testing a
    geometry nobody's workflow produces, and the whole subject here is
    what a real lead does at its periodic boundary.
    """
    from molbuilder.modify import add_slab
    seed = Structure(elements=["S"], positions=np.array([[0.0, 0.0, 30.0]]))
    out = add_slab(seed, element="Au", plane="111",
                   size=(2, 2, n_layers), grow="-z")
    st = out.structure if hasattr(out, "structure") else out
    au = [i for i, e in enumerate(st.elements) if e == "Au"]
    st.regions = {"L-electrode": au,
                  "bridge": [i for i in range(len(st.elements))
                             if i not in au]}
    st.frozen_atoms = list(au)
    return st


@pytest.mark.parametrize("n_layers,verdict", [
    (3, "CONTINUES"), (4, "ECLIPSED"), (5, "TWIN"), (6, "CONTINUES"),
])
def test_the_seam_verdict_is_measured_and_said(n_layers, verdict):
    """The whole table, because the interesting half is the failures and
    a test on 3 and 6 alone would pass with the check deleted."""
    m = extract_electrode_model(_au111_lead(n_layers), "L-electrode")
    seam = [n for n in m.notes if "periodic seam" in n]
    assert len(seam) == 1, f"exactly one seam note: {m.notes}"
    assert verdict in seam[0], f"{n_layers} layers -> {seam[0]}"


def test_a_faulted_seam_is_REPORTED_not_refused():
    """The discriminating half.  A 4-layer lead is not bulk and still
    composes -- the person may be doing exactly what they meant."""
    m = extract_electrode_model(_au111_lead(4), "L-electrode")
    assert m.z_period == pytest.approx(9.4182, abs=1e-3)
    assert m.n_layers == 4


def test_the_faulted_seam_note_names_the_LAYER_COUNT_as_the_cause():
    """'Your seam is eclipsed' is not actionable; 'four layers is not a
    whole number of 3-layer periods' is."""
    m = extract_electrode_model(_au111_lead(4), "L-electrode")
    note = next(n for n in m.notes if "periodic seam" in n)
    assert "LAYER COUNT" in note and "4 layers" in note and "3-layer" in note, note
