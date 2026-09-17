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
    return Structure(
        elements=elems, positions=np.asarray(pos, dtype=float),
        regions={"L-electrode": left, "bridge": bridge, "R-electrode": right})


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
