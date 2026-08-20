"""Tests for molbuilder.validation.geometry.

Per docs/process/testing.md (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate, validate_geometry
from ._helpers import _vacuum_cell




# --------------------------------------------------------------------- #
#  Geometry: min atom-atom distance                                     #
# --------------------------------------------------------------------- #


def test_min_distance_too_small_is_error():
    """< 0.3 Å -- atoms effectively coincide; SCF will diverge."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    errs = [i for i in issues if i.severity == "error"
            and i.where == "geometry.min_distance"]
    assert len(errs) == 1
    assert "0.100" in errs[0].message



def test_min_distance_short_is_warn():
    """0.3 - 0.7 Å -- shorter than any real bond, likely broken."""
    s = Structure(
        elements=["C", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.min_distance"]
    assert len(warns) == 1



def test_min_distance_normal_bonds_no_issue(water_struct):
    """Real water (O-H ~0.957 Å) must not flag any geometry issue."""
    issues = validate(water_struct, SiestaConfig())
    geo = [i for i in issues if i.where == "geometry.min_distance"]
    assert geo == []



# --------------------------------------------------------------------- #
#  H/heavy ratio: catches heavy-atom skeletons headed for DFT           #
# --------------------------------------------------------------------- #


def test_h_ratio_skeleton_is_warn():
    """A heavy-atom-only structure (typical X3DNA fiber raw output, or a
    user-loaded heavy-atom PDB) is missing electrons -- DFT will compute
    the wrong total electron count.  Severity is warn (not error)
    because the user may legitimately want to inspect / hand-process
    the skeleton; the warning surfaces the issue prominently."""
    # 4 heavy atoms, 0 H -> ratio 0.0
    s = Structure(
        elements=["C", "N", "O", "P"],
        positions=np.array([[0,0,0],[1.5,0,0],[3.0,0,0],[4.5,0,0]],
                           dtype=float),
    )
    issues = validate(s, SiestaConfig())
    warns = [i for i in issues if i.severity == "warn"
             and i.where == "geometry.h_ratio"]
    assert len(warns) == 1, f"expected 1 h_ratio warn, got {warns}"
    assert "0.00" in warns[0].message    # ratio printed to 2 decimals



def test_h_ratio_low_but_not_zero_is_warn():
    """Borderline case: ratio < 0.3 still warns (3 H, 12 heavy = 0.25)."""
    elements = ["C"] * 12 + ["H"] * 3
    pos = np.zeros((15, 3))
    for i in range(15):
        pos[i] = (i * 1.5, 0, 0)
    s = Structure(elements=elements, positions=pos)
    issues = validate(s, SiestaConfig())
    assert any(i.severity == "warn" and i.where == "geometry.h_ratio"
               for i in issues)



def test_h_ratio_organic_no_warn():
    """A canonical organic molecule (water, H/heavy = 2.0) must not
    warn -- typical organic ratios are 0.6 to 1.5."""
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0,0,0],[0.957,0,0],[-0.24,0.927,0]]),
    )
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "geometry.h_ratio"] == []



def test_polymer_orientation_normal_chain_no_warn():
    """A clean 4-mer DNA chain with 5'->3' residue ordering must not
    trip the orientation validator."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []



def test_polymer_orientation_reversed_listing_warns():
    """If a backend (or a user-loaded PDB) lists residues 3'->5', the
    structural 5' end (no incoming O3'-P bridge) won't match
    residue_ids[0], so the validator must warn.

    Build a normal chain, then flip residue_ids so the highest-numbered
    residue ends up at index 0.  Atom positions stay the same -- only
    the ID listing is reversed."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")
    rid_max = max(s.residue_ids)
    rid_min = min(s.residue_ids)
    # Map r -> (rid_max + rid_min - r) so 1->4, 2->3, 3->2, 4->1.
    flipped = [rid_max + rid_min - r for r in s.residue_ids]
    s_rev = type(s)(
        elements=list(s.elements), positions=s.positions.copy(),
        atom_names=list(s.atom_names), residue_ids=flipped,
        residue_names=list(s.residue_names),
        chain_ids=list(s.chain_ids), title=s.title,
    )
    issues = validate(s_rev, SiestaConfig())
    orient = [i for i in issues if i.where == "polymer.orientation"]
    assert len(orient) == 1, (
        f"expected one orientation warn, got: {orient}"
    )
    assert orient[0].severity == "warn"



def test_polymer_orientation_no_phosphorus_silent():
    """A peptide (no P, no O3') must not trigger the polymer-orientation
    check -- it's not a nucleic acid."""
    pytest.importorskip("PeptideBuilder")
    from molbuilder import build_peptide
    s = build_peptide("ARNDC")
    issues = validate(s, SiestaConfig())
    assert [i for i in issues if i.where == "polymer.orientation"] == []



def test_h_ratio_runs_after_layer2_protonation():
    """The user contract: validation runs at FDF/PySCF emission time,
    AFTER any add_hydrogens step at build time.  An X3DNA-built ATGC
    chain with default kwargs (add_hydrogens=True) must NOT trip the
    h_ratio warn -- protonation already happened, the ratio is healthy."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    from molbuilder import build_dna
    s = build_dna("ATGC", backend="threedna")     # default: add_hydrogens=True
    issues = validate(s, SiestaConfig())
    h_ratio_warns = [i for i in issues if i.where == "geometry.h_ratio"]
    assert h_ratio_warns == [], (
        f"X3DNA + add_hydrogens=True should produce a healthy H/heavy "
        f"ratio; got warning: {h_ratio_warns}"
    )



# --------------------------------------------------------------------- #
#  Cell: determinant + volume                                           #
# --------------------------------------------------------------------- #


def test_cell_determinant_zero_is_error(water_struct):
    """A degenerate cell (det == 0) is unusable; flag as error and
    skip the volume check below."""
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    # `cell.determinant` said "degenerate OR left-handed" -- two faults with
    # two different repairs under one id, so a flat molecule was told to swap
    # its lattice vectors.  Split 2026-08-03; the checker reports one.
    errs = [i for i in issues if i.where == "cell.no_volume"]
    assert len(errs) == 1, [i.where for i in issues]
    assert errs[0].severity == "error"
    assert not [i for i in issues if i.where == "cell.left_handed"], (
        "a flat cell is not a handedness problem")


def test_cell_determinant_negative_is_error(water_struct):
    """A left-handed cell (negative det) breaks SIESTA's PBC math."""
    cell = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 10
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    errs = [i for i in issues if i.where == "cell.left_handed"]
    assert len(errs) == 1, [i.where for i in issues]
    assert errs[0].severity == "error"
    assert not [i for i in issues if i.where == "cell.no_volume"], (
        "a mirrored cell has volume; it is the wrong way round, not empty")



def test_cell_volume_tight_is_warn(water_struct):
    """Cell volume / atom-bounding-volume < 3 -- molecule fills the box."""
    # Water bounding box ~1 x 1 x 1 Å^3 -> atom_box ~1.  Cell 1 Å -> det = 1.
    cell = np.eye(3) * 1.0
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    vol = [i for i in issues if i.where == "cell.volume"]
    assert len(vol) == 1
    assert vol[0].severity == "warn"



def test_cell_volume_generous_no_warn(water_struct):
    """A 30 Å cubic box around water is generously vacuum-padded."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    vol = [i for i in issues if i.where == "cell.volume"]
    assert vol == []



# --------------------------------------------------------------------- #
#  Cell: atom-to-nearest-image distance                                 #
# --------------------------------------------------------------------- #


def test_image_distance_too_close_is_warn(water_struct):
    """A 5 Å cubic cell makes water see its own image ~5 Å away --
    well below the 6 Å "atoms still interacting" threshold."""
    cell = _vacuum_cell(5.0)
    issues = validate(water_struct, SiestaConfig(), cell=cell)
    msgs = [i for i in issues if i.where == "cell.image_distance"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"



def test_image_distance_generous_no_warn(water_struct):
    """A 30 Å cubic cell puts water's image >25 Å away; safely
    isolated, no warning."""
    issues = validate(water_struct, SiestaConfig(), cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "cell.image_distance"] == []



# --------------------------------------------------------------------- #
#  Geometry: net dipole in vacuum                                       #
# --------------------------------------------------------------------- #


def test_dipole_in_vacuum_polar_molecule_is_warn():
    """An HF-shaped molecule (H + F at ~0.92 Å) has a strong dipole
    (~1.8 D real, ~1.8 D heuristic).  In a Gamma-only vacuum cell
    the validator should warn about image-image dipole interactions."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    msgs = [i for i in issues if i.where == "geometry.dipole"]
    assert len(msgs) == 1
    assert msgs[0].severity == "warn"
    assert "dipole" in msgs[0].message.lower()



def test_dipole_in_vacuum_nonpolar_molecule_no_warn():
    """N2 has zero dipole by symmetry (homonuclear diatomic).  No warn."""
    s = Structure(
        elements=["N", "N"],
        positions=np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []



def test_dipole_with_kgrid_no_warn():
    """A polar molecule in a periodic cell (k > 1) is INTENDED to
    have image-image interactions; the dipole warning is for the
    Gamma-only vacuum case where the user probably didn't realise."""
    s = Structure(
        elements=["F", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
    )
    cfg = SiestaConfig(kgrid=(4, 4, 4))   # genuinely periodic
    issues = validate(s, cfg, cell=_vacuum_cell(30.0))
    assert [i for i in issues if i.where == "geometry.dipole"] == []



# --------------------------------------------------------------------- #
#  SIESTA: kgrid sanity                                                 #
# --------------------------------------------------------------------- #


# The k-grid sanity check now trusts the AUTHORITATIVE per-axis periodicity
# (``struct.axis_kind``) instead of a geometry span-ratio guess (which
# mis-flagged real crystals whose atoms don't reach the cell edge).  These
# tests carry the axis_kind their scenario implies, as a real structure would.

def _axis_struct(kind, extent, cell_len, n_atoms=8):
    """Atoms strung along z inside a cell, with an explicit per-axis
    ``axis_kind``.  ``extent``/``cell_len`` set the z span vs the z cell so
    a crystal whose atoms DON'T reach the edge can be represented."""
    zs = np.linspace(0.0, extent, n_atoms)
    pos = np.column_stack([np.zeros(n_atoms), np.zeros(n_atoms), zs])
    cell = np.diag([cell_len, cell_len, cell_len])
    return Structure(elements=["C"] * n_atoms, positions=pos,
                     cell=cell, axis_kind=kind), cell


def test_kgrid_on_isolated_axis_is_warn():
    """k > 1 on an ISOLATED (vacuum) axis is wasted -- flagged off the
    authoritative axis_kind, regardless of how far the atoms span."""
    s, cell = _axis_struct(("isolated", "isolated", "isolated"),
                           extent=2.0, cell_len=20.0)
    cfg = SiestaConfig(kgrid=(4, 4, 1))
    msgs = [i for i in validate(s, cfg, cell=cell)
            if i.where == "config.kgrid"]
    assert any("kgrid[0]" in i.message and "isolated" in i.message
               for i in msgs)


def test_kgrid_one_on_a_periodic_axis_is_silent():
    """k == 1 states NOTHING (user rule, 2026-08-20;
    `science/validation.md`): correct for an isolated axis and a
    legitimate Gamma-only choice for a periodic one, so it is validated
    not at all -- even beside a sampled sibling.  (This replaces the
    retired forgotten-axis warn, which validated an axis about which the
    user had stated nothing.)"""
    s, cell = _axis_struct(("periodic", "isolated", "isolated"),
                           extent=3.0, cell_len=6.0)
    cfg = SiestaConfig(kgrid=(1, 4, 1))   # k=1 on the periodic x
    msgs = [i for i in validate(s, cfg, cell=cell)
            if i.where == "config.kgrid"]
    assert not any("kgrid[0]" in i.message for i in msgs)


def test_kgrid_periodic_crystal_partial_span_no_false_positive():
    """SCIENTIFIC-AUDIT FIX: a real periodic crystal whose atoms span only
    ~50% of the cell (rocksalt, metals, oxides) with a full k-mesh must NOT
    be flagged 'k>1 wasted'.  The old span-ratio heuristic (periodic iff
    atoms span >85%) mis-read such axes as vacuum -- the false positive that
    told users to drop the k-points a crystal actually needs."""
    # A block spanning ~50% of the cell on EVERY axis (the old fixture
    # was a z-line: x/y extents 0, so those axes really did hold 6 A of
    # emptiness -- a geometry the 2026-08-20 statement rule legitimately
    # hints on, and not the crystal this docstring names).
    corners = np.array([[x, y, z] for x in (0.0, 3.0)
                        for y in (0.0, 3.0) for z in (0.0, 3.0)])
    cell = np.diag([6.0, 6.0, 6.0])
    s = Structure(elements=["C"] * 8, positions=corners, cell=cell,
                  axis_kind=("periodic", "periodic", "periodic"))
    cfg = SiestaConfig(kgrid=(4, 4, 4))
    msgs = [i for i in validate(s, cfg, cell=cell)
            if i.where == "config.kgrid"]
    assert msgs == [], (
        f"periodic crystal (50% span) with a full k-mesh must not warn; "
        f"got {[i.message for i in msgs]}")


def test_kgrid_long_vacuum_padded_axis_no_false_positive():
    """A 12-mer DNA in an 80 Å vacuum cell (isolated z) with k=1 on the
    long molecular axis is correct vacuum, NOT under-sampled -- no warn on
    the isolated long axis."""
    s, cell = _axis_struct(("isolated", "isolated", "isolated"),
                           extent=30.0, cell_len=80.0)
    cfg = SiestaConfig(kgrid=(1, 1, 1))    # Gamma-only, correct for vacuum
    msgs = [i for i in validate(s, cfg, cell=cell)
            if i.where == "config.kgrid"]
    assert not any("kgrid[2]" in i.message for i in msgs)



def test_all_gamma_in_vacuum_no_warn(water_struct):
    """Pure 1x1x1 Gamma in vacuum is the molecule case; no kgrid warning."""
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(kgrid=(1, 1, 1))
    issues = validate(water_struct, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.kgrid"] == []



# --------------------------------------------------------------------- #
#  SIESTA: wrap_into_cell                                               #
# --------------------------------------------------------------------- #


def test_atoms_outside_unit_cell_with_no_wrap_is_warn():
    """Atoms outside [0,1) fractional with wrap_into_cell=False -- the
    visualiser will draw them in a neighbour cell."""
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[15.0, 5.0, 5.0], [16.0, 5.0, 5.0]]),
    )
    cell = _vacuum_cell(10.0)    # atoms at x=15 are outside [0, 10)
    cfg = SiestaConfig(wrap_into_cell=False)
    issues = validate(s, cfg, cell=cell)
    wrap = [i for i in issues if i.where == "config.wrap_into_cell"]
    assert len(wrap) == 1
    assert wrap[0].severity == "warn"



def test_atoms_inside_with_no_wrap_no_warn(water_struct):
    cell = _vacuum_cell(30.0)
    cfg = SiestaConfig(wrap_into_cell=False)
    # Shift water into the box so all atoms sit in [0, 30).
    s = Structure(
        elements=water_struct.elements,
        positions=water_struct.positions + np.array([15.0, 15.0, 15.0]),
    )
    issues = validate(s, cfg, cell=cell)
    assert [i for i in issues if i.where == "config.wrap_into_cell"] == []


class TestImageDistanceOnlyCrossesVacuum:
    """The image-distance check measures ARTEFACTS, so it may only step along
    isolated (vacuum) axes.

    Along a periodic axis the neighbouring cell holds the crystal's real
    neighbours -- bulk gold sits 2.88 Å across the boundary by construction --
    and along a transport axis the device is meant to tile seamlessly.  Stepping
    there reported the intended physics as a defect: a warn on every crystal and
    every junction (owner's catch, 2026-07-29).  Same reasoning as containment
    being required on non-periodic axes only (structure-periodicity.md § 2).
    """

    @staticmethod
    def _au_bulk():
        """4 Au atoms in a 2.88 Å cube: nearest neighbours ARE across the
        boundary."""
        s = Structure(elements=["Au"] * 4,
                      positions=np.array([[0.00, 0.00, 0.00],
                                          [1.44, 1.44, 0.00],
                                          [1.44, 0.00, 1.44],
                                          [0.00, 1.44, 1.44]]))
        s.cell = np.diag([2.88, 2.88, 2.88])
        return s

    def _wheres(self, s):
        return {i.where for i in validate_geometry(s, s.resolve_cell())}

    def test_a_fully_periodic_crystal_is_not_reported(self):
        """A 3-D crystal has no vacuum direction, so it has no artificial
        images at all and the check is NOT APPLICABLE -- which is different from
        a check that could not run, so it stays quiet rather than reporting
        itself (contrast clause F4's "say so as info" in
        docs/science/validation.md § 4.1)."""
        s = self._au_bulk()
        s.axis_kind = ("periodic", "periodic", "periodic")
        s.__post_init__()
        assert "cell.image_distance" not in self._wheres(s)

    def test_a_transport_axis_is_not_reported_either(self):
        """The device tiles along the transport direction by design."""
        s = self._au_bulk()
        s.axis_kind = ("periodic", "periodic", "transport")
        s.__post_init__()
        assert "cell.image_distance" not in self._wheres(s)

    def test_a_slab_is_measured_across_its_vacuum_axis_only(self):
        """periodic in-plane, isolated out-of-plane: the out-of-plane gap is
        exactly what should be checked."""
        s = Structure(elements=["Au", "Au"],
                      positions=np.array([[0.0, 0.0, 0.0], [1.44, 1.44, 0.0]]),
                      vacuum=(0.0, 0.0, 1.0))
        s.cell = np.diag([2.88, 2.88, 2.0])
        s.axis_kind = ("periodic", "periodic", "isolated")
        s.__post_init__()
        found = [i for i in validate_geometry(s, s.resolve_cell())
                 if i.where == "cell.image_distance"]
        assert found, "the slab's vacuum gap must still be checked"
        # 2.0 Å cell along c with a flat slab -> the images are 2 Å apart.
        assert "2.00" in found[0].message
        # And the message says WHICH direction was measured.
        assert "direction(s) c" in found[0].message

    def test_an_all_isolated_molecule_is_measured_in_every_direction(self):
        """The hemeC shape: every axis is vacuum, so every axis is measured and
        the message names all three."""
        s = Structure(elements=["C", "C"],
                      positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
                      vacuum=(1.0, 1.0, 1.0))
        found = [i for i in validate_geometry(s, s.resolve_cell())
                 if i.where == "cell.image_distance"]
        assert found and "direction(s) a, b, c" in found[0].message

    def test_the_helper_refuses_to_step_where_it_was_not_asked(self):
        """Unit-level: a translation along a non-permitted axis must not be
        considered even when it is the shortest one."""
        from molbuilder.validation.geometry import _min_image_distance
        pos = np.array([[0.0, 0.0, 0.0]])
        cell = np.diag([2.0, 40.0, 40.0])     # x images are 2 Å away
        assert _min_image_distance(pos, cell, axes=[0]) == pytest.approx(2.0)
        # Not allowed to step along x -> the nearest permitted image is 40 Å.
        assert _min_image_distance(pos, cell, axes=[1, 2]) == pytest.approx(40.0)
        # Nothing steppable at all -> no artificial images exist.
        assert _min_image_distance(pos, cell, axes=[]) == float("inf")
