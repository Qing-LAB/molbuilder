"""Unit tests for ``molbuilder.modify``.

Spec source of truth: ``docs/spec/modify-tab.md``.

Covers M1: the four pure-function ops (delete_atoms, add_atom,
orient_along_axis, add_electrode_slab) plus the symmetric-electrode
convenience wrapper.  No web / CLI / UI yet -- those land in M2-M5.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.modify import (
    SUPPORTED_FCC_ELEMENTS,
    SUPPORTED_FCC_PLANES,
    add_atom,
    add_electrode_slab,
    add_symmetric_electrodes,
    delete_atoms,
    orient_along_axis,
    rotate_around_axis,
)
from molbuilder.structure import Structure


@pytest.fixture
def linear_dimer():
    """6 atoms: two carbons along x with two H pairs each.  Useful for
    delete + orient because the molecular axis is unambiguous."""
    return Structure(
        elements=["C", "H", "H", "H", "H", "C"],
        positions=np.array([
            [0.0,  0.0, 0.0],
            [0.5,  0.5, 0.0],
            [0.5, -0.5, 0.0],
            [2.5,  0.5, 0.0],
            [2.5, -0.5, 0.0],
            [3.0,  0.0, 0.0],
        ]),
        title="dimer",
    )


@pytest.fixture
def single_anchor():
    """One S atom at the origin -- minimal anchor for electrode tests."""
    return Structure(
        elements=["S"],
        positions=np.array([[0.0, 0.0, 0.0]]),
        title="anchor",
    )


# --------------------------------------------------------------------- #
#  delete_atoms                                                         #
# --------------------------------------------------------------------- #


def test_delete_drops_listed_indices(linear_dimer):
    out = delete_atoms(linear_dimer, [1, 2, 3, 4])
    assert out.n_atoms == 2
    assert out.elements == ["C", "C"]
    assert np.allclose(out.positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(out.positions[1], [3.0, 0.0, 0.0])


def test_delete_preserves_metadata_in_lockstep(linear_dimer):
    out = delete_atoms(linear_dimer, [1, 3])
    assert len(out.atom_names)     == out.n_atoms
    assert len(out.residue_ids)    == out.n_atoms
    assert len(out.residue_names)  == out.n_atoms
    assert len(out.chain_ids)      == out.n_atoms


def test_delete_no_op_when_indices_empty(linear_dimer):
    out = delete_atoms(linear_dimer, [])
    assert out.n_atoms == linear_dimer.n_atoms
    assert linear_dimer.n_atoms == 6


def test_delete_does_not_mutate_input(linear_dimer):
    delete_atoms(linear_dimer, [0])
    assert linear_dimer.n_atoms == 6
    assert linear_dimer.elements[0] == "C"


def test_delete_silently_ignores_out_of_range_indices(linear_dimer):
    out = delete_atoms(linear_dimer, [99, -1])
    assert out.n_atoms == linear_dimer.n_atoms


def test_delete_dedups_repeated_indices(linear_dimer):
    out = delete_atoms(linear_dimer, [1, 1, 1])
    assert out.n_atoms == linear_dimer.n_atoms - 1
    assert out.elements == ["C", "H", "H", "H", "C"]


# --------------------------------------------------------------------- #
#  add_atom                                                             #
# --------------------------------------------------------------------- #


def test_add_atom_at_offset(linear_dimer):
    out = add_atom(linear_dimer, "S", anchor_index=5, offset=[0.0, 0.0, 1.5])
    assert out.n_atoms == linear_dimer.n_atoms + 1
    assert out.elements[-1] == "S"
    expected = linear_dimer.positions[5] + np.array([0.0, 0.0, 1.5])
    assert np.allclose(out.positions[-1], expected)


def test_add_atom_gets_fresh_residue_id(linear_dimer):
    out = add_atom(linear_dimer, "S", 5, [0.0, 0.0, 1.5])
    anchor_residue = linear_dimer.residue_ids[5]
    new_residue = out.residue_ids[-1]
    assert new_residue != anchor_residue
    assert new_residue == max(linear_dimer.residue_ids) + 1


def test_add_atom_residue_name_default_and_override(linear_dimer):
    out = add_atom(linear_dimer, "S", 0, [1, 0, 0])
    assert out.residue_names[-1] == "MOD"
    out2 = add_atom(linear_dimer, "S", 0, [1, 0, 0], residue_name="THI")
    assert out2.residue_names[-1] == "THI"


def test_add_atom_atom_name_defaults_to_element(linear_dimer):
    out = add_atom(linear_dimer, "Au", 0, [0, 0, 1])
    assert out.atom_names[-1] == "Au"


def test_add_atom_rejects_bad_anchor(linear_dimer):
    with pytest.raises(IndexError):
        add_atom(linear_dimer, "S", anchor_index=99, offset=[0, 0, 0])


def test_add_atom_explicit_residue_id_groups_atoms_in_one_residue(linear_dimer):
    """SP-E: passing ``residue_id=`` lets a caller land multiple
    appended atoms in the same residue -- needed for polyatomic
    side-chain caps (e.g. -COOH = 4 atoms all in one residue).
    The default (no kwarg) still allocates a fresh id per call."""
    s = add_atom(linear_dimer, "C", 0, [1.5, 0, 0])
    rid = s.residue_ids[-1]
    s = add_atom(s, "O", anchor_index=s.n_atoms - 1, offset=[0.6, 1.0, 0],
                 residue_id=rid)
    s = add_atom(s, "O", anchor_index=s.n_atoms - 2, offset=[0.6, -1.0, 0],
                 residue_id=rid)
    s = add_atom(s, "H", anchor_index=s.n_atoms - 1, offset=[0.0, -1.0, 0],
                 residue_id=rid)
    # Last four atoms (the cap) all share rid.
    assert s.residue_ids[-4:] == [rid, rid, rid, rid]


def test_add_atom_default_still_allocates_fresh_residue(linear_dimer):
    """SP-E sanity: the default (no residue_id kwarg) preserves the
    pre-SP-E behaviour of giving each appended atom its own residue."""
    s = add_atom(linear_dimer, "S", 0, [1, 0, 0])
    s = add_atom(s, "S", 0, [-1, 0, 0])  # second call -- still fresh id
    assert s.residue_ids[-2] != s.residue_ids[-1]


# --------------------------------------------------------------------- #
#  orient_along_axis                                                    #
# --------------------------------------------------------------------- #


def test_orient_default_midpoint_centers_anchors_symmetrically(linear_dimer):
    """Default ``center="midpoint"``: anchors land symmetrically on +z and -z."""
    out = orient_along_axis(linear_dimer, anchor_indices=(0, 5), axis="z")
    a0 = out.positions[0]
    a1 = out.positions[5]
    assert np.allclose(0.5 * (a0 + a1), [0.0, 0.0, 0.0], atol=1e-10)
    assert np.isclose(a0[2], -a1[2])
    assert all(abs(p) < 1e-10 for p in (a0[0], a0[1], a1[0], a1[1]))
    # Anchor pair length is preserved.
    original_dist = np.linalg.norm(linear_dimer.positions[5] - linear_dimer.positions[0])
    assert np.isclose(a1[2] - a0[2], original_dist)


def test_orient_first_center_places_a0_at_origin(linear_dimer):
    out = orient_along_axis(linear_dimer, anchor_indices=(0, 5),
                             axis="z", center="first")
    a0 = out.positions[0]
    a1 = out.positions[5]
    assert np.allclose(a0, [0.0, 0.0, 0.0], atol=1e-10)
    assert a1[2] > 0
    assert abs(a1[0]) < 1e-10 and abs(a1[1]) < 1e-10


def test_orient_none_center_no_translation(linear_dimer):
    """center='none' rotates only.  After identity rotation of an
    already-on-x dimer toward x, atom 0 stays at the origin."""
    out = orient_along_axis(linear_dimer, (0, 5), axis="x", center="none")
    assert np.allclose(out.positions[0], [0.0, 0.0, 0.0], atol=1e-10)


def test_orient_along_x_axis(linear_dimer):
    out = orient_along_axis(linear_dimer, (0, 5), axis="x")
    # Default center='midpoint': anchor pair lies along x, midpoint at origin
    a0 = out.positions[0]
    a1 = out.positions[5]
    assert abs(a0[1]) < 1e-10 and abs(a0[2]) < 1e-10
    assert abs(a1[1]) < 1e-10 and abs(a1[2]) < 1e-10
    assert np.isclose(a0[0], -a1[0])


def test_orient_with_angle_tilts_in_xz_plane(linear_dimer):
    """angle=30: anchor pair lies in xz-plane at 30° from z.
    Midpoint at origin (default center)."""
    out = orient_along_axis(linear_dimer, (0, 5), axis="z", angle=30.0)
    a0 = out.positions[0]
    a1 = out.positions[5]
    d = np.linalg.norm(linear_dimer.positions[5] - linear_dimer.positions[0])
    # Anchor pair vector should be (sin(30°)*d, 0, cos(30°)*d)
    diff = a1 - a0
    expected = np.array([np.sin(np.radians(30)) * d, 0.0,
                         np.cos(np.radians(30)) * d])
    assert np.allclose(diff, expected, atol=1e-9)
    # Midpoint at origin
    mid = 0.5 * (a0 + a1)
    assert np.allclose(mid, 0, atol=1e-9)


def test_orient_angle_zero_matches_default(linear_dimer):
    """angle=0 (default) gives the same result as omitting angle."""
    out_default = orient_along_axis(linear_dimer, (0, 5), axis="z")
    out_zero    = orient_along_axis(linear_dimer, (0, 5), axis="z", angle=0.0)
    assert np.allclose(out_default.positions, out_zero.positions, atol=1e-12)


def test_orient_handles_antiparallel_case():
    s = Structure(
        elements=["C", "C"],
        positions=np.array([[0, 0, 0], [0, 0, -2.5]]),
        title="flip",
    )
    out = orient_along_axis(s, (0, 1), axis="z", center="first")
    a1 = out.positions[1]
    assert np.isclose(a1[2], 2.5)
    assert abs(a1[0]) < 1e-10 and abs(a1[1]) < 1e-10


def test_orient_rejects_coincident_anchors():
    s = Structure(
        elements=["C", "C"],
        positions=np.array([[0, 0, 0], [0, 0, 0]]),
        title="coincident",
    )
    with pytest.raises(ValueError, match="coincident"):
        orient_along_axis(s, (0, 1), axis="z")


def test_orient_rejects_same_anchor_twice(linear_dimer):
    with pytest.raises(ValueError, match="distinct"):
        orient_along_axis(linear_dimer, (3, 3), axis="z")


def test_orient_rejects_bad_axis(linear_dimer):
    with pytest.raises(ValueError, match="axis"):
        orient_along_axis(linear_dimer, (0, 5), axis="w")


# --------------------------------------------------------------------- #
#  rotate_around_axis                                                   #
# --------------------------------------------------------------------- #


def test_rotate_around_z_default_no_op(linear_dimer):
    """angle=0 returns positions unchanged."""
    out = rotate_around_axis(linear_dimer, axis="z", angle=0.0)
    assert np.allclose(out.positions, linear_dimer.positions, atol=1e-12)


def test_rotate_around_z_90_deg():
    """90° around z: (1, 0, 0) -> (0, 1, 0)."""
    s = Structure(elements=["C"], positions=np.array([[1.0, 0.0, 0.0]]))
    out = rotate_around_axis(s, axis="z", angle=90.0)
    assert np.allclose(out.positions[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_rotate_around_x_90_deg():
    """90° around x: (0, 1, 0) -> (0, 0, 1)."""
    s = Structure(elements=["C"], positions=np.array([[0.0, 1.0, 0.0]]))
    out = rotate_around_axis(s, axis="x", angle=90.0)
    assert np.allclose(out.positions[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_rotate_around_y_90_deg():
    """90° around y: (1, 0, 0) -> (0, 0, -1)."""
    s = Structure(elements=["C"], positions=np.array([[1.0, 0.0, 0.0]]))
    out = rotate_around_axis(s, axis="y", angle=90.0)
    assert np.allclose(out.positions[0], [0.0, 0.0, -1.0], atol=1e-12)


def test_rotate_then_unrotate_recovers_original(linear_dimer):
    """Rotating by +θ then -θ around the same axis is identity."""
    out = rotate_around_axis(linear_dimer, axis="z", angle=37.5)
    out = rotate_around_axis(out,         axis="z", angle=-37.5)
    assert np.allclose(out.positions, linear_dimer.positions, atol=1e-10)


def test_rotate_combined_with_orient_redirects_tilt():
    """Common workflow: orient with angle to tilt in xz-plane, then
    rotate around z to point the tilt in another direction (e.g. yz)."""
    s = Structure(
        elements=["C", "C"],
        positions=np.array([[0, 0, 0], [3.0, 0.0, 0.0]]),
        title="dimer",
    )
    # Orient with 30° tilt in xz-plane (default tilt direction)
    out = orient_along_axis(s, (0, 1), axis="z", angle=30.0)
    a0_xz = out.positions[0]
    a1_xz = out.positions[1]
    # Anchor pair at (sin(30)*3, 0, cos(30)*3) - (-sin(30)*1.5, 0, -cos(30)*1.5)
    assert abs(a1_xz[1]) < 1e-9 and abs(a0_xz[1]) < 1e-9   # in xz-plane
    # Now rotate 90° around z; tilt now in yz-plane
    out2 = rotate_around_axis(out, axis="z", angle=90.0)
    a0_yz = out2.positions[0]
    a1_yz = out2.positions[1]
    assert abs(a1_yz[0]) < 1e-9 and abs(a0_yz[0]) < 1e-9   # now in yz-plane


def test_rotate_rejects_bad_axis(linear_dimer):
    with pytest.raises(ValueError, match="axis"):
        rotate_around_axis(linear_dimer, axis="w", angle=10.0)


# --------------------------------------------------------------------- #
#  add_electrode_slab -- uniform (m, n, n_layers) per call              #
#                                                                       #
#  ASE supports each plane with specific (orthogonal, m, n) constraints #
#  (spec § 8); the function passes the user's choice to ASE and lets    #
#  ASE's error bubble up as a ValueError on incompatible inputs.        #
# --------------------------------------------------------------------- #


def test_electrode_supported_lists_match_table():
    """Spec § 8: closed list of 6 metals + 3 planes."""
    assert SUPPORTED_FCC_ELEMENTS == ("Au", "Ag", "Cu", "Ni", "Pt", "Pd")
    assert SUPPORTED_FCC_PLANES   == ("100", "110", "111")


# Valid per-(plane, orthogonal) tuples and their atom counts.  The
# count is m * n * n_layers regardless of cell shape, since ASE's slab
# is uniform across layers.
_VALID_COMBOS = [
    # (plane, orthogonal, size, expected_atom_count)
    ("111", False, (3, 3, 2), 18),    # primitive hex
    ("111", True,  (3, 4, 2), 24),    # orthogonal: n must be even
    ("100", True,  (3, 3, 2), 18),    # square primitive = orthogonal
    ("110", True,  (3, 3, 2), 18),    # rectangular primitive = orthogonal
]


@pytest.mark.parametrize("element", SUPPORTED_FCC_ELEMENTS)
@pytest.mark.parametrize("plane,orthogonal,size,n_expected",
                          _VALID_COMBOS,
                          ids=[f"{p}_{'orth' if o else 'prim'}"
                               for p, o, _, _ in _VALID_COMBOS])
def test_electrode_atom_count(single_anchor, element, plane,
                                orthogonal, size, n_expected):
    """Atom count = m * n * n_layers.  Same for every supported
    element + (plane, orthogonal) combo."""
    out = add_electrode_slab(single_anchor, element, plane,
                              size, anchor_index=0,
                              contact_distance=2.0, orthogonal=orthogonal)
    n_metal = sum(1 for e in out.elements if e == element)
    assert n_metal == n_expected, (
        f"{element}({plane}) orthogonal={orthogonal} size={size}: "
        f"got {n_metal}, expected {n_expected}"
    )


@pytest.mark.parametrize("plane,orthogonal,size", [
    ("111", False, (2, 2, 2)),
    ("111", True,  (2, 2, 2)),
    ("100", True,  (2, 2, 2)),
    ("110", True,  (2, 2, 2)),
])
def test_electrode_plus_z_atoms_above_anchor(single_anchor, plane,
                                              orthogonal, size):
    out = add_electrode_slab(single_anchor, "Au", plane, size,
                              anchor_index=0, contact_distance=2.0, side="+z",
                              orthogonal=orthogonal)
    au_z = np.array([p[2] for e, p in zip(out.elements, out.positions) if e == "Au"])
    assert au_z.min() >= 2.0 - 1e-6
    assert au_z.max() > au_z.min()


@pytest.mark.parametrize("plane,orthogonal,size", [
    ("111", False, (2, 2, 2)),
    ("111", True,  (2, 2, 2)),
    ("100", True,  (2, 2, 2)),
    ("110", True,  (2, 2, 2)),
])
def test_electrode_minus_z_atoms_below_anchor(single_anchor, plane,
                                                orthogonal, size):
    out = add_electrode_slab(single_anchor, "Au", plane, size,
                              anchor_index=0, contact_distance=2.0, side="-z",
                              orthogonal=orthogonal)
    au_z = np.array([p[2] for e, p in zip(out.elements, out.positions) if e == "Au"])
    assert au_z.max() <= -2.0 + 1e-6
    assert au_z.min() < au_z.max()


@pytest.mark.parametrize("plane,orthogonal,size", [
    ("111", False, (3, 3, 2)),
    ("111", True,  (3, 4, 2)),
    ("100", True,  (3, 3, 2)),
    ("110", True,  (3, 3, 2)),
])
def test_electrode_default_offset_centers_slab_on_anchor(plane, orthogonal, size):
    """Default offset=(0, 0): slab's whole-slab centroid sits over
    the anchor's (x, y).  Use a non-origin anchor to catch any
    centring bug that special-cases the origin."""
    s = Structure(
        elements=["S"], positions=np.array([[3.7, -1.2, 0.0]]), title="off-origin",
    )
    out = add_electrode_slab(s, "Au", plane, size, anchor_index=0,
                              contact_distance=2.0, orthogonal=orthogonal)
    au = np.array([p for e, p in zip(out.elements, out.positions) if e == "Au"])
    centroid_xy = au[:, :2].mean(axis=0)
    assert np.allclose(centroid_xy, [3.7, -1.2], atol=1e-6), (
        f"slab centroid {centroid_xy} should equal anchor (3.7, -1.2) "
        f"with default offset=(0, 0)"
    )


@pytest.mark.parametrize("offset_xy", [
    (0.5, 0.0),
    (0.0, 0.7),
    (-1.2, 0.4),
])
def test_electrode_offset_shifts_slab_centroid(single_anchor, offset_xy):
    """Non-zero offset shifts the slab's centroid by exactly that
    much, leaving everything else unchanged."""
    base = add_electrode_slab(single_anchor, "Au", "111",
                              (3, 3, 2), anchor_index=0, contact_distance=2.0)
    shifted = add_electrode_slab(single_anchor, "Au", "111",
                                  (3, 3, 2), anchor_index=0,
                                  contact_distance=2.0, offset=offset_xy)
    base_centroid = np.array([p[:2] for e, p in zip(base.elements, base.positions)
                               if e == "Au"]).mean(axis=0)
    shifted_centroid = np.array([p[:2] for e, p in zip(shifted.elements, shifted.positions)
                                   if e == "Au"]).mean(axis=0)
    delta = shifted_centroid - base_centroid
    assert np.allclose(delta, offset_xy, atol=1e-6), (
        f"offset={offset_xy} should shift centroid by exactly that; "
        f"got delta={delta}"
    )
    # Z positions unchanged: offset is xy only.
    base_z = np.array([p[2] for e, p in zip(base.elements, base.positions) if e == "Au"])
    shifted_z = np.array([p[2] for e, p in zip(shifted.elements, shifted.positions) if e == "Au"])
    assert np.allclose(np.sort(base_z), np.sort(shifted_z), atol=1e-9)


def test_electrode_metadata_marks_atoms_as_ELC(single_anchor):
    """Spec § 5: electrode atoms get residue_name='ELC' and a fresh
    residue_id so the molecule and electrode are separable."""
    out = add_electrode_slab(single_anchor, "Au", "111",
                              (2, 2, 1), anchor_index=0, contact_distance=2.0)
    elc_indices = [i for i, n in enumerate(out.residue_names) if n == "ELC"]
    assert len(elc_indices) > 0
    elc_residue_ids = {out.residue_ids[i] for i in elc_indices}
    anchor_residue = single_anchor.residue_ids[0]
    assert anchor_residue not in elc_residue_ids
    for i in elc_indices:
        assert out.elements[i]  == "Au"
        assert out.atom_names[i] == "Au"


def test_electrode_zero_layers_is_noop(single_anchor):
    out = add_electrode_slab(single_anchor, "Au", "111", (3, 3, 0), 0)
    assert out.n_atoms == single_anchor.n_atoms
    assert out.elements == single_anchor.elements


# --------------------------------------------------------------------- #
#  Rejection paths -- per-(plane, orthogonal) constraints from ASE      #
# --------------------------------------------------------------------- #


def test_electrode_rejects_unsupported_element(single_anchor):
    with pytest.raises(ValueError, match="unsupported electrode element"):
        add_electrode_slab(single_anchor, "Fe", "111", (2, 2, 1), 0)
    with pytest.raises(ValueError, match="unsupported electrode element"):
        add_electrode_slab(single_anchor, "Al", "111", (2, 2, 1), 0)


def test_electrode_rejects_unsupported_plane(single_anchor):
    with pytest.raises(ValueError, match="unsupported crystal plane"):
        add_electrode_slab(single_anchor, "Au", "101", (2, 2, 1), 0)


def test_electrode_rejects_bad_side(single_anchor):
    with pytest.raises(ValueError, match="side"):
        add_electrode_slab(single_anchor, "Au", "111", (2, 2, 1), 0, side="up")


def test_electrode_rejects_bad_anchor(single_anchor):
    with pytest.raises(IndexError):
        add_electrode_slab(single_anchor, "Au", "111", (2, 2, 1), anchor_index=5)


def test_electrode_orthogonal_111_rejects_odd_n(single_anchor):
    """fcc(111) orthogonal supercell requires n even.  ASE's own error
    bubbles up as a ValueError with operation context."""
    with pytest.raises(ValueError, match="orthogonal=True"):
        add_electrode_slab(single_anchor, "Au", "111",
                            (3, 3, 1), 0, orthogonal=True)


@pytest.mark.parametrize("plane", ["100", "110"])
def test_electrode_primitive_100_110_rejects_non_orthogonal(single_anchor, plane):
    """fcc(100) and fcc(110) only support orthogonal=True.  ASE raises
    NotImplementedError; we re-wrap as ValueError with context."""
    with pytest.raises(ValueError, match="orthogonal=False"):
        add_electrode_slab(single_anchor, "Au", plane,
                            (3, 3, 1), 0, orthogonal=False)


def test_electrode_lattice_constant_override(single_anchor):
    """Explicit lattice_constant changes the slab's overall extent
    (proxy for 'the kwarg actually reached ASE')."""
    default = add_electrode_slab(single_anchor, "Au", "100",
                                  (3, 3, 1), 0, contact_distance=2.0,
                                  orthogonal=True)
    expanded = add_electrode_slab(single_anchor, "Au", "100",
                                   (3, 3, 1), 0, contact_distance=2.0,
                                   orthogonal=True, lattice_constant=5.0)
    def slab_extent(s):
        au_xy = np.array([p[:2] for e, p in zip(s.elements, s.positions)
                          if e == "Au"])
        return au_xy.max(axis=0) - au_xy.min(axis=0)
    d_extent = slab_extent(default).mean()
    e_extent = slab_extent(expanded).mean()
    assert e_extent > d_extent + 0.5


def test_electrode_inter_layer_offset_rescales_z():
    """inter_layer_offset overrides ASE's natural inter-layer spacing.
    Useful for strained-distance studies."""
    s = Structure(elements=["S"], positions=np.array([[0.0, 0, 0]]))
    natural = add_electrode_slab(s, "Au", "111", (2, 2, 3), 0,
                                   contact_distance=2.0)
    nat_z = sorted({float(p[2])
                    for e, p in zip(natural.elements, natural.positions)
                    if e == "Au"})
    nat_dz = nat_z[1] - nat_z[0]

    forced_dz = nat_dz * 1.5
    stretched = add_electrode_slab(s, "Au", "111", (2, 2, 3), 0,
                                    contact_distance=2.0,
                                    inter_layer_offset=forced_dz)
    stretch_z = sorted({float(p[2])
                        for e, p in zip(stretched.elements, stretched.positions)
                        if e == "Au"})
    stretch_dz = stretch_z[1] - stretch_z[0]
    assert np.isclose(stretch_dz, forced_dz, rtol=1e-6), (
        f"inter_layer_offset={forced_dz} should rescale spacing; "
        f"got natural dz={nat_dz}, stretched dz={stretch_dz}"
    )


# --------------------------------------------------------------------- #
#  add_symmetric_electrodes                                             #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("plane,orthogonal,size,per_side", [
    ("111", False, (2, 2, 2), 8),
    ("111", True,  (2, 2, 2), 8),
    ("100", True,  (2, 2, 2), 8),
    ("110", True,  (2, 2, 2), 8),
])
def test_symmetric_electrodes_doubles_metal_count(linear_dimer, plane,
                                                    orthogonal, size, per_side):
    """gap is now the total electrode-to-electrode distance.  After
    midpoint orient, linear_dimer's anchor pair is 3 Å apart along z;
    pick gap = 7.0 Å so each side's contact distance = (7-3)/2 = 2.0 Å,
    matching the old per-side semantics for a clean atom-count check."""
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z",
                                  center="midpoint")
    # NEW convention: anchor_indices = (a_top, a_bot).  After orient,
    # atom 5 ends up on +z (top), atom 0 on -z (bottom).
    out = add_symmetric_electrodes(oriented, "Au", plane, size,
                                    anchor_indices=(5, 0),
                                    gap=7.0, orthogonal=orthogonal)
    n_au = sum(1 for e in out.elements if e == "Au")
    assert n_au == 2 * per_side


def test_symmetric_electrodes_above_and_below(linear_dimer):
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z",
                                  center="midpoint")
    out = add_symmetric_electrodes(oriented, "Au", "111",
                                    (2, 2, 2),
                                    anchor_indices=(5, 0),
                                    gap=7.0)
    a0_z = oriented.positions[0, 2]
    a1_z = oriented.positions[5, 2]
    assert a0_z < a1_z, "expected midpoint centring to put a0 below a1"
    au_z = np.array([p[2] for e, p in zip(out.elements, out.positions) if e == "Au"])
    above = (au_z > a1_z).sum()
    below = (au_z < a0_z).sum()
    assert above == 8 and below == 8


def test_symmetric_electrodes_gap_centered_on_anchor_midpoint(linear_dimer):
    """Spec § "tilted molecule + gap": the closest layers of the two
    electrodes sit at z = mid.z ± gap/2 where mid = anchor-pair midpoint."""
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z")
    a_top = oriented.positions[5]
    a_bot = oriented.positions[0]
    mid_z = 0.5 * (a_top[2] + a_bot[2])
    gap = 7.0
    out = add_symmetric_electrodes(oriented, "Au", "111",
                                    (2, 2, 1),
                                    anchor_indices=(5, 0),
                                    gap=gap)
    au_z = sorted({float(p[2])
                   for e, p in zip(out.elements, out.positions)
                   if e == "Au"})
    # Two layers: one at mid + gap/2, one at mid - gap/2.
    assert np.isclose(au_z[-1], mid_z + gap / 2, atol=1e-9)
    assert np.isclose(au_z[0],  mid_z - gap / 2, atol=1e-9)


def test_symmetric_electrodes_rejects_too_small_gap(linear_dimer):
    """gap smaller than anchor-pair z-extent must raise ValueError --
    otherwise the electrodes would overlap the molecule."""
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z")
    # anchor pair is 3.0 Å apart in z after orient; gap=1.0 too small.
    with pytest.raises(ValueError, match="gap"):
        add_symmetric_electrodes(oriented, "Au", "111", (2, 2, 1),
                                  anchor_indices=(5, 0), gap=1.0)


def test_symmetric_electrodes_rejects_reversed_anchor_order(linear_dimer):
    """T1 (post-static-review): pass anchors in (a_top, a_bot) order
    where a_top has LOWER z than a_bot.  Without validation the math
    silently produces overlapping slabs -- now caught at the API
    boundary with a descriptive ValueError."""
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z")
    # After orient: atom 5 is on +z, atom 0 on -z.  Reversed:
    with pytest.raises(ValueError, match="lower than the labelled bottom"):
        add_symmetric_electrodes(oriented, "Au", "111", (2, 2, 1),
                                  anchor_indices=(0, 5),    # ← swapped
                                  gap=7.0)


def test_symmetric_electrodes_handles_tilted_molecule():
    """For a tilted molecule (anchor pair NOT along z), gap is still
    measured along z and the slabs are collinear along z, centred on
    the anchor-pair midpoint's xy."""
    s = Structure(
        elements=["S", "C", "C", "S"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]),
        title="bdt-stub",
    )
    # 30-degree tilt in xz-plane via the new angle parameter
    tilted = orient_along_axis(s, (0, 3), axis="z", angle=30.0)
    a_top = tilted.positions[3]
    a_bot = tilted.positions[0]
    mid = 0.5 * (a_top + a_bot)
    # gap = 9.0 Å > anchor z-extent
    junction = add_symmetric_electrodes(tilted, "Au", "111", (3, 3, 1),
                                          anchor_indices=(3, 0),
                                          gap=9.0)
    # The two electrode layers' centroids should sit at (mid.x, mid.y)
    # in xy and at mid.z ± gap/2 in z.
    au_pos = np.array([p for e, p in zip(junction.elements, junction.positions)
                       if e == "Au"])
    z_layers = sorted({round(float(z), 6) for z in au_pos[:, 2]})
    assert len(z_layers) == 2
    assert np.isclose(z_layers[1], mid[2] + 9.0 / 2, atol=1e-6)
    assert np.isclose(z_layers[0], mid[2] - 9.0 / 2, atol=1e-6)
    # Lateral centroid of all Au atoms should equal (mid.x, mid.y).
    centroid_xy = au_pos[:, :2].mean(axis=0)
    assert np.allclose(centroid_xy, mid[:2], atol=1e-6), (
        f"slab centroid {centroid_xy} should equal anchor midpoint "
        f"({mid[0]}, {mid[1]})"
    )


def test_symmetric_electrodes_anchorless_centres_on_origin(linear_dimer):
    """``anchor_indices=None`` (default) places the slab pair
    symmetrically around the world origin: closest layers at ±gap/2
    in absolute z, regardless of where the molecule sits."""
    out = add_symmetric_electrodes(linear_dimer, "Au", "111",
                                   size=(2, 2, 2), gap=10.0)
    elc_z = np.array([p[2] for n, p in zip(out.residue_names, out.positions)
                      if n == "ELC"])
    top = elc_z[elc_z > 0].min()
    bot = elc_z[elc_z < 0].max()
    assert abs(top - 5.0) < 1e-9, top
    assert abs(bot + 5.0) < 1e-9, bot


def test_symmetric_electrodes_anchorless_works_for_offset_origin():
    """With atom 0 NOT near the centroid, the closest layers still
    land at ±gap/2 absolute (regression: the old contact_top<=0
    guard rejected this case spuriously)."""
    s = Structure(elements=["C", "C", "C"],
                  positions=np.array([[10.0, 0, -1],
                                       [10.0, 0,  0],
                                       [10.0, 0,  1]]))
    out = add_symmetric_electrodes(s, "Au", "111",
                                   size=(2, 2, 2), gap=10.0)
    elc_z = np.array([p[2] for n, p in zip(out.residue_names, out.positions)
                      if n == "ELC"])
    assert abs(elc_z[elc_z > 0].min() - 5.0) < 1e-9
    assert abs(elc_z[elc_z < 0].max() + 5.0) < 1e-9


def test_symmetric_electrodes_anchorless_rejects_too_small_gap():
    """The gap must accommodate the molecule's z-extent + 2× M-X
    bond margin (1.5 Å each side).  A 6 Å molecule cannot fit in a
    4 Å gap; reject with an actionable message."""
    s = Structure(elements=["C"]*5, positions=np.array([
        [0, 0, -3], [0, 0, -1.5], [0, 0, 0], [0, 0, 1.5], [0, 0, 3],
    ]))
    with pytest.raises(ValueError, match="too small"):
        add_symmetric_electrodes(s, "Au", "111", size=(2, 2, 2), gap=4.0)


def test_symmetric_electrodes_anchorless_rejects_nonpositive_gap():
    """gap == 0 and gap < 0 must be rejected explicitly."""
    s = Structure(elements=["C"], positions=np.array([[0, 0, 0]]))
    with pytest.raises(ValueError, match="must be > 0"):
        add_symmetric_electrodes(s, "Au", "111", size=(2, 2, 2), gap=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        add_symmetric_electrodes(s, "Au", "111", size=(2, 2, 2), gap=-3.0)


def test_symmetric_electrodes_anchorless_rejects_empty_structure():
    """An empty struct can't carry the slab op; the error must point
    the user at how to load a structure."""
    s = Structure(elements=[], positions=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty structure"):
        add_symmetric_electrodes(s, "Au", "111", size=(2, 2, 2), gap=8.0)


def test_symmetric_electrodes_offset_propagates_to_both_sides(linear_dimer):
    """The single offset arg shifts BOTH the +z and -z slabs by the
    same (Δx, Δy)."""
    oriented = orient_along_axis(linear_dimer, (0, 5), axis="z")
    base = add_symmetric_electrodes(oriented, "Au", "111", (2, 2, 2),
                                      anchor_indices=(5, 0), gap=7.0)
    shifted = add_symmetric_electrodes(oriented, "Au", "111", (2, 2, 2),
                                         anchor_indices=(5, 0), gap=7.0,
                                         offset=(0.6, -0.4))
    def side_centroid(s, sign):
        au = np.array([p[:2] for e, p in zip(s.elements, s.positions)
                       if e == "Au" and (sign * p[2]) > 0])
        return au.mean(axis=0)
    base_top = side_centroid(base, +1)
    base_bot = side_centroid(base, -1)
    shifted_top = side_centroid(shifted, +1)
    shifted_bot = side_centroid(shifted, -1)
    assert np.allclose(shifted_top - base_top, (0.6, -0.4), atol=1e-6)
    assert np.allclose(shifted_bot - base_bot, (0.6, -0.4), atol=1e-6)


# --------------------------------------------------------------------- #
#  End-to-end: build a Au(111)-bdt-Au(111) junction                    #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("orthogonal,size,per_side", [
    # Spec § 2 walkthrough (now uniform per call): user calls the
    # symmetric helper with one (m, n, n_layers).  For stepped contacts
    # ("3×3 close, 4×4 further out") the user makes two add_electrode_slab
    # calls instead of one; covered separately by the stacked test below.
    (False, (3, 3, 2), 3 * 3 * 2),    # 18 atoms per side
    (True,  (3, 4, 2), 3 * 4 * 2),    # 24 atoms per side (n must be even)
])
def test_junction_end_to_end(orthogonal, size, per_side):
    """Mini "BDT-like" stub (S–S linker), oriented on z, with Au(111)
    on both sides via add_symmetric_electrodes."""
    bdt = Structure(
        elements=["S", "C", "C", "S"],
        positions=np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]),
        title="bdt-stub",
    )
    oriented = orient_along_axis(bdt, (0, 3), axis="z", center="midpoint")
    assert abs(oriented.positions[0, 2]) > 0
    assert np.isclose(oriented.positions[0, 2], -oriented.positions[3, 2])

    junction = add_symmetric_electrodes(
        oriented, "Au", "111", size,
        # NEW convention: (a_top, a_bot).  After orient with default
        # midpoint centring, atom 3 is on +z (top), atom 0 on -z (bottom).
        anchor_indices=(3, 0), gap=9.0, orthogonal=orthogonal,
    )
    n_au = sum(1 for e in junction.elements if e == "Au")
    # symmetric => 2 sides × per_side atoms
    assert n_au == 2 * per_side
    assert junction.n_atoms == 4 + 2 * per_side
    elc_count = sum(1 for n in junction.residue_names if n == "ELC")
    assert elc_count == n_au
    elc_residues = {r for r, n in zip(junction.residue_ids, junction.residue_names)
                    if n == "ELC"}
    assert len(elc_residues) == 2


def test_junction_stepped_contacts_via_two_calls():
    """Spec § 2: stepped "3×3 close, 4×4 further out" pattern is built
    by two add_electrode_slab calls per side -- inner stack with one
    gap, outer stack with a larger gap that puts it past the inner stack.
    No per-layer-list needed."""
    bdt = Structure(
        elements=["S", "C", "C", "S"],
        positions=np.array([
            [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
            [3.5, 0.0, 0.0], [5.0, 0.0, 0.0],
        ]),
        title="bdt-stub",
    )
    oriented = orient_along_axis(bdt, (0, 3), axis="z", center="midpoint")
    inner_gap = 2.0
    inner_layers = 1
    # ASE's natural fcc(111) Au inter-layer spacing is ~2.355 Å.
    # Place the outer 4×4 stack one such layer further out.
    outer_gap = inner_gap + 2.355

    # Inner stacks: 3×3 single layer, both sides.  Single-electrode
    # mode uses ``contact_distance`` (anchor-to-closest-layer), not
    # ``gap`` (which is reserved for the pair-mode total junction gap).
    s1 = add_electrode_slab(oriented, "Au", "111", (3, 3, inner_layers),
                              anchor_index=0, contact_distance=inner_gap,
                              side="-z")
    s2 = add_electrode_slab(s1, "Au", "111", (3, 3, inner_layers),
                              anchor_index=3, contact_distance=inner_gap,
                              side="+z")
    # Outer stacks: 4×4 single layer, both sides, larger contact distance.
    s3 = add_electrode_slab(s2, "Au", "111", (4, 4, 1),
                              anchor_index=0, contact_distance=outer_gap,
                              side="-z")
    junction = add_electrode_slab(s3, "Au", "111", (4, 4, 1),
                                    anchor_index=3, contact_distance=outer_gap,
                                    side="+z")

    n_au = sum(1 for e in junction.elements if e == "Au")
    # 9 (3×3 close) × 2 sides + 16 (4×4 far) × 2 sides
    assert n_au == 2 * 9 + 2 * 16
    # Four distinct electrode-residue ids: one per call.
    elc_residues = {r for r, n in zip(junction.residue_ids, junction.residue_names)
                    if n == "ELC"}
    assert len(elc_residues) == 4


# --------------------------------------------------------------------- #
#  SP-C: import does NOT load the FCC lattice file.  Lazy load only      #
#  triggers when a function actually needs the table.                    #
# --------------------------------------------------------------------- #


def test_modify_module_import_does_not_eagerly_load_fcc_table(monkeypatch):
    """SP-C: the FCC lattice-constant table loads on first call to
    ``_get_fcc_lattice``, NOT at module import.  This way a broken
    ``MOLBUILDER_DATA_DIR`` doesn't cascade into ``import molbuilder``
    failing -- only operations that actually consume a lattice
    constant surface the error, and only when the user runs them."""
    import importlib
    import molbuilder.modify as mod
    # Reset the cache and re-import the module.  Right after import
    # the cache must be None (lazy).
    mod._FCC_LATTICE_A_CACHE = None
    importlib.reload(mod)
    assert mod._FCC_LATTICE_A_CACHE is None, (
        "FCC table was loaded at import time -- SP-C regressed"
    )
    # Asking for the table populates the cache.
    table = mod._get_fcc_lattice()
    assert "Au" in table
    assert mod._FCC_LATTICE_A_CACHE is not None
    # Public closed list of metals is hardcoded so a missing JSON
    # doesn't take it down.
    assert mod.SUPPORTED_FCC_ELEMENTS == ("Au", "Ag", "Cu", "Ni", "Pt", "Pd")


def test_modify_module_falls_back_to_packaged_data_when_env_var_broken(
        tmp_path, monkeypatch):
    """SP-C: even when ``MOLBUILDER_DATA_DIR`` points at a directory
    without ``fcc_lattice.json``, the loader falls back to the
    packaged ``molbuilder/data/`` so the operation still succeeds."""
    monkeypatch.setenv("MOLBUILDER_DATA_DIR", str(tmp_path / "nonexistent"))
    import importlib
    import molbuilder.modify as mod
    mod._FCC_LATTICE_A_CACHE = None
    importlib.reload(mod)
    # Import didn't crash even with a broken env var.
    assert mod.SUPPORTED_FCC_ELEMENTS == ("Au", "Ag", "Cu", "Ni", "Pt", "Pd")
    # Lookup falls back to the packaged data dir.
    table = mod._get_fcc_lattice()
    assert "Au" in table
    # Reset for the rest of the suite.
    monkeypatch.delenv("MOLBUILDER_DATA_DIR", raising=False)
    mod._FCC_LATTICE_A_CACHE = None
    importlib.reload(mod)


# --------------------------------------------------------------------- #
#  Element-aware default contact distance                               #
# --------------------------------------------------------------------- #


def test_default_contact_distance_table_has_supported_metals():
    """The contact-distance table covers every metal the slab op
    accepts; the Au-S canonical 2.40 A is the headline value, and
    Pt-N is shorter (2.05) which is the whole reason for the table."""
    from molbuilder.modify import (
        default_contact_distance, SUPPORTED_FCC_ELEMENTS,
    )
    for sym in SUPPORTED_FCC_ELEMENTS:
        d = default_contact_distance(sym)
        assert isinstance(d, float)
        assert 1.5 < d < 3.5, (sym, d)
    assert default_contact_distance("Au") == 2.40
    assert default_contact_distance("Pt") == 2.05
    assert default_contact_distance("Ag") == 2.50
    # Unsupported element falls back to the legacy Au-S value (back-
    # compat with callers that relied on the old hardcoded default).
    assert default_contact_distance("Fe") == 2.4


def test_add_electrode_slab_uses_element_aware_contact_distance(linear_dimer):
    """``add_electrode_slab`` resolves contact_distance from
    ``default_contact_distance(element)`` when not overridden -- so
    a Pt slab on the same anchor lands at a different z than an Au
    slab.  Concretely: the Pt closest layer is 0.35 A closer to the
    anchor than Au's (2.40 - 2.05)."""
    au_out = add_electrode_slab(linear_dimer, "Au", "111",
                                size=(2, 2, 1), anchor_index=0, side="+z")
    pt_out = add_electrode_slab(linear_dimer, "Pt", "111",
                                size=(2, 2, 1), anchor_index=0, side="+z")
    # Anchor sits at positions[0]; the closest metal layer's z =
    # anchor.z + contact_distance.  Find the smallest metal z above
    # the anchor.
    anchor_z = linear_dimer.positions[0, 2]
    au_z = min(p[2] for n, p in zip(au_out.residue_names, au_out.positions)
               if n == "ELC" and p[2] > anchor_z)
    pt_z = min(p[2] for n, p in zip(pt_out.residue_names, pt_out.positions)
               if n == "ELC" and p[2] > anchor_z)
    assert abs((au_z - anchor_z) - 2.40) < 1e-6, au_z - anchor_z
    assert abs((pt_z - anchor_z) - 2.05) < 1e-6, pt_z - anchor_z


# --------------------------------------------------------------------- #
#  rotate_around_axis(center=...) pivot                                 #
# --------------------------------------------------------------------- #


def test_rotate_around_axis_centroid_pivot_leaves_centroid_invariant():
    """``center='centroid'`` rotates each atom about the molecule's
    atom-mean centroid; the centroid is fixed under the rotation."""
    s = Structure(elements=["C"] * 4,
                  positions=np.array([[0., 0., 0.],
                                       [1., 1., 0.],
                                       [2., 2., 0.],
                                       [3., 3., 0.]]))
    centroid_before = s.positions.mean(axis=0)
    rot = rotate_around_axis(s, axis="z", angle=90.0, center="centroid")
    centroid_after = rot.positions.mean(axis=0)
    assert np.allclose(centroid_before, centroid_after, atol=1e-9)
    # Atom 1 was at (1, 1, 0); centroid is (1.5, 1.5, 0).  In centroid
    # frame: (-0.5, -0.5, 0).  Rotate +90 about z: (0.5, -0.5, 0).
    # Back to world: (2, 1, 0).
    assert np.allclose(rot.positions[1], [2.0, 1.0, 0.0], atol=1e-9)


def test_rotate_around_axis_python_default_is_origin_pivot():
    """The Python API's default ``center='origin'`` preserves the
    legacy world-axis behaviour for any existing caller that didn't
    pass the kwarg.  The Modify-tab UI defaults to ``'centroid'``
    on its own (HTML <select>), which is independent from the
    function's default."""
    s = Structure(elements=["C"] * 2,
                  positions=np.array([[1., 1., 0.], [2., 2., 0.]]))
    rot_default  = rotate_around_axis(s, axis="z", angle=90.0)
    rot_centroid = rotate_around_axis(s, axis="z", angle=90.0,
                                       center="centroid")
    # The two results differ for an off-origin molecule.
    assert not np.allclose(rot_default.positions, rot_centroid.positions)
    # Default = origin: atom 0 at (1,1,0) -> (-1, 1, 0).
    assert np.allclose(rot_default.positions[0],
                       [-1.0, 1.0, 0.0], atol=1e-9)


def test_rotate_around_axis_rejects_unknown_center():
    s = Structure(elements=["C"], positions=np.array([[0., 0., 0.]]))
    with pytest.raises(ValueError, match="center"):
        rotate_around_axis(s, axis="z", angle=10.0, center="midpoint")


# --------------------------------------------------------------------- #
#  Structure.copy()                                                     #
# --------------------------------------------------------------------- #


def test_structure_copy_is_independent():
    """``Structure.copy()`` returns a fresh Structure whose
    positions array and metadata lists can be mutated without
    touching the original.  Used by ``delete_atoms`` /
    ``add_electrode_slab`` no-op branches."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0., 0., 0.], [1., 0., 0.], [-0.3, 0.9, 0.]]),
        atom_names=["O1", "H1", "H2"],
        residue_ids=[1, 1, 1],
        residue_names=["MOL", "MOL", "MOL"],
        chain_ids=["A", "A", "A"],
        title="water",
    )
    c = s.copy()
    # Mutating the copy doesn't reach the original.
    c.positions[0, 0] = 99.0
    c.atom_names[0] = "X"
    c.residue_ids[0] = 42
    assert s.positions[0, 0] == 0.0
    assert s.atom_names[0] == "O1"
    assert s.residue_ids[0] == 1
