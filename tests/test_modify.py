"""Unit tests for ``molbuilder.modify``.

Spec source of truth: ``docs/web/tabs.md``.

Covers M1: the four pure-function ops (delete_atoms, add_atom,
orient_along_axis, add_slab) plus the junction
convenience wrapper.  No web / CLI / UI yet -- those land in M2-M5.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.modify import (
    append_structure,
    SUPPORTED_FCC_ELEMENTS,
    SUPPORTED_FCC_PLANES,
    add_atom,
    add_slab,
    delete_atoms,
    orient_along_axis,
    rotate_around_axis,
)
from molbuilder.structure import FROZEN_LABEL, Structure




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
def periodic_dimer():
    """The linear dimer with a full periodic + transport cell set (as the
    Cell op-tab would produce), so ops can be checked for lattice survival."""
    return Structure(
        elements=["C", "H", "H", "H", "H", "C"],
        positions=np.array([
            [0.0,  0.0, 0.0], [0.5,  0.5, 0.0], [0.5, -0.5, 0.0],
            [2.5,  0.5, 0.0], [2.5, -0.5, 0.0], [3.0,  0.0, 0.0],
        ]),
        title="dimer",
        cell=np.diag([10.0, 10.0, 20.0]),
        axis_kind=("periodic", "periodic", "transport"),
        vacuum=(5.0, 5.0, 0.0),
    )


def _assert_lattice_preserved(out, ref):
    """The op must carry cell / axis_kind / vacuum verbatim -- these are NOT
    per-atom, so an edit must never revert them to isolated defaults.  (k-grid
    is no longer geometry; it lives on SiestaConfig -- structure-periodicity.md.)"""
    assert out.cell is not None and np.allclose(out.cell, ref.cell)
    assert out.axis_kind == ref.axis_kind
    assert out.vacuum == ref.vacuum


class TestOpsPreservePeriodicity:
    """Regression (2026-07 fresh-eyes review): a modify op must not silently
    wipe the periodic cell / transport axis / vacuum / k-grid.  Before the fix
    delete/add/orient/rotate/translate dropped them and a subsequent SIESTA
    FDF omitted LatticeVectors / the k-grid -- a scientifically wrong cell,
    no warning.

    NB: ``axis_kind`` / ``vacuum`` are non-geometric and carry VERBATIM through
    every op.  The lattice VECTORS carry verbatim through atom-count edits +
    translation (translation-invariant), but a whole-structure ROTATION rotates
    them WITH the atoms so the box keeps wrapping the structure (§ 3c) -- that is
    checked separately in :class:`TestRigidTransformMovesTheBox`, not here."""

    def test_delete_preserves_lattice(self, periodic_dimer):
        """SCIENCE. Deleting an atom leaves the cell, axis kinds and vacuum untouched.

        Catches the 2026-07 regression this class exists for: a modify op returning
        a Structure built without `_carry_periodicity`, so the periodic box quietly
        became None. The emitted FDF then had no `LatticeVectors` block, SIESTA
        built a default cell, and the run converged on a different physical system
        with no warning anywhere. Atom-count edits are lattice-VECTOR-invariant --
        removing an atom does not change the crystal.

        Contract: `model/structure-periodicity.md` § 3c (a rigid transform moves the
        box; an atom-count edit does not touch it).
        """
        _assert_lattice_preserved(delete_atoms(periodic_dimer, [1]), periodic_dimer)

    def test_add_atom_preserves_lattice(self, periodic_dimer):
        """SCIENCE. Appending an atom leaves the cell, axis kinds and vacuum untouched.

        Same regression, other direction: `add_atom` constructs a new Structure and
        has its own chance to forget the periodicity fields. Adding an atom to a
        junction must not silently turn a transport cell into an isolated box --
        the transport axis length IS the device length, and losing it makes the
        electrodes non-commensurate with the bulk lead.

        Contract: `model/structure-periodicity.md` § 3c + § 2 (axis kinds).
        """
        _assert_lattice_preserved(
            add_atom(periodic_dimer, "S", 0, [1.5, 0, 0]), periodic_dimer)

    def test_orient_preserves_axis_kind_and_vacuum(self, periodic_dimer):
        # orient is a whole-structure rotation -> it ROTATES the lattice vectors
        # (checked in TestRigidTransformMovesTheBox); the non-geometric axis_kind /
        # vacuum tags still carry verbatim.
        """SCIENCE. `orient_along_axis` carries the NON-geometric periodicity tags --
        axis kinds and vacuum -- verbatim.

        Catches a whole-structure rotation resetting the axis kinds to isolated
        defaults. The lattice VECTORS legitimately rotate here (that is
        `TestRigidTransformMovesTheBox`), which is exactly why this test cannot use
        `_assert_lattice_preserved` -- and why the tags need their own check: they
        are the half that must NOT change, and a rotation that rebuilds the cell
        from scratch loses them without touching a single coordinate.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        out = orient_along_axis(periodic_dimer, (0, 5), axis="z")
        assert out.axis_kind == periodic_dimer.axis_kind
        assert out.vacuum == periodic_dimer.vacuum

    def test_rotate_preserves_axis_kind_and_vacuum(self, periodic_dimer):
        # A rotation ROTATES the lattice vectors (checked in
        # TestRigidTransformMovesTheBox), but the non-geometric axis_kind / vacuum
        # tags still carry verbatim.
        """SCIENCE. `rotate_around_axis` carries axis kinds and vacuum verbatim.

        The same split as the orient test above, for the other rotation op: the
        vectors turn with the atoms, the KINDS do not. A rotation that reset
        `("periodic", "periodic", "transport")` to `("isolated",)*3` would leave the
        geometry perfect and the calculation non-periodic.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        out = rotate_around_axis(periodic_dimer, axis="z", angle=30)
        assert out.axis_kind == periodic_dimer.axis_kind
        assert out.vacuum == periodic_dimer.vacuum

    def test_translate_preserves_lattice(self, periodic_dimer):
        # Translation is lattice-VECTOR-invariant (only the origin corner moves),
        # so cell / axis_kind / vacuum all carry verbatim.
        """SCIENCE. A translation leaves the lattice VECTORS unchanged (only the origin
        corner moves).

        Catches a translate that rebuilds the cell from the moved atoms' bounding
        box -- which would silently resize a crystal's lattice constant because the
        user shifted the molecule. Translation is the one rigid transform under
        which the vectors are invariant, and that has to stay stated separately from
        rotation, where they are not.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        _assert_lattice_preserved(periodic_dimer.translated((1, 0, 0)), periodic_dimer)

    def test_center_preserves_lattice(self, periodic_dimer):
        """SCIENCE. `Structure.centered()` is a translation, so it too leaves the
        lattice vectors alone.

        Catches the centring helper being written as "rebuild the box around the
        centred atoms". Centring is the op most likely to be implemented that way --
        its whole purpose is to move atoms relative to a frame -- and doing so
        would change a periodic cell's dimensions as a side effect of a cosmetic act.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        _assert_lattice_preserved(periodic_dimer.centered(), periodic_dimer)

    def test_copy_preserves_lattice(self, periodic_dimer):
        """SCIENCE. `Structure.copy()` carries the periodicity fields.

        Catches the quietest path of all: `delete_atoms` short-circuits to
        `struct.copy()` whenever nothing is actually removed, so a `copy()` that
        dropped the cell would make a no-op delete destroy the box. No coordinate
        changes, no atom count changes, and the lattice is gone -- and the other
        tests in this class all take the non-no-op branch.

        Contract: `model/structure.md` § 1 (the object and what a copy holds) +
        `model/structure-periodicity.md` § 3c.
        """
        _assert_lattice_preserved(periodic_dimer.copy(), periodic_dimer)


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
    """SCIENCE. Delete removes exactly the listed atoms and leaves the survivors'
    COORDINATES untouched.

    Catches the index bookkeeping being off. Deleting the four H atoms must
    leave the two C atoms at 0.0 and 3.0 -- assert the positions, not just the
    count, because a slice that kept the wrong four atoms gives the same
    `n_atoms == 2` and a completely different molecule. Which atoms survive a
    delete decides which atoms are computed.

    Contract: `model/structure.md` § 1 (per-atom arrays are index-parallel).
    """
    out = delete_atoms(linear_dimer, [1, 2, 3, 4])
    assert out.n_atoms == 2
    assert out.elements == ["C", "C"]
    assert np.allclose(out.positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(out.positions[1], [3.0, 0.0, 0.0])


def test_delete_preserves_metadata_in_lockstep(linear_dimer):
    """SCIENCE. Every per-atom metadata column comes back the same LENGTH as the
    surviving atom list.

    Catches a column left unsliced. `elements`, `positions`, `atom_names`,
    `residue_ids`, `residue_names` and `chain_ids` are parallel arrays indexed
    by atom, so one column that is not sliced makes every atom after the
    deletion point wear its neighbour's name and residue -- an atom-identity
    error that reads as a perfectly valid structure and reaches the emitters
    intact.

    Contract: `model/structure.md` § 1 (the index-parallel invariant).

    NOTE: this asserts LENGTH only. A column sliced with the wrong `keep` set
    would have the right length and the wrong contents.
    """
    out = delete_atoms(linear_dimer, [1, 3])
    assert len(out.atom_names)     == out.n_atoms
    assert len(out.residue_ids)    == out.n_atoms
    assert len(out.residue_names)  == out.n_atoms
    assert len(out.chain_ids)      == out.n_atoms


def test_delete_no_op_when_indices_empty(linear_dimer):
    """An empty index list changes nothing, and does not mutate the input.

    Contract: `modify.delete_atoms`'s own docstring (indices in any order,
    duplicates tolerated).

    CUT CANDIDATE: it reaches the same `len(keep) == n_atoms -> struct.copy()`
    branch as `test_delete_silently_ignores_out_of_range_indices` and
    `test_delete_atoms_no_op_branch_preserves_metadata`, both of which assert
    strictly more about it.
    """
    out = delete_atoms(linear_dimer, [])
    assert out.n_atoms == linear_dimer.n_atoms
    assert linear_dimer.n_atoms == 6


def test_delete_does_not_mutate_input(linear_dimer):
    """Delete is pure: the structure passed in is unchanged afterwards.

    Catches an in-place rewrite. The Modify tab keeps the pre-edit structure
    for undo, and every op in this module returns a new object -- an op that
    edited its argument would corrupt that history silently, because the caller
    holds the same object it just "copied from".

    Contract: `web/tabs.md` § 2 (the ops are pure functions the tab composes).

    THIN: `delete_atoms` builds its lists with comprehensions and slices
    `positions` with a fancy index (which copies), so purity is structural
    rather than defended. See the audit note.
    """
    delete_atoms(linear_dimer, [0])
    assert linear_dimer.n_atoms == 6
    assert linear_dimer.elements[0] == "C"


def test_delete_silently_ignores_out_of_range_indices(linear_dimer):
    """SCIENCE-ADJACENT. An index outside [0, n_atoms) is dropped, not acted on --
    and -1 does NOT mean "the last atom".

    Catches Python's negative-index semantics leaking into an atom selection.
    `delete_atoms` computes `set(range(n)) - set(indices)`, so -1 falls out;
    written as a list comprehension with `del`, -1 would delete the LAST atom
    instead. The browser sends indices from a selection that may be stale after
    an earlier edit, so out-of-range arrivals are ordinary -- and deleting the
    wrong atom because of one is silent.

    Contract: `modify.delete_atoms` (the tolerated-input rule in its docstring).
    """
    out = delete_atoms(linear_dimer, [99, -1])
    assert out.n_atoms == linear_dimer.n_atoms


def test_delete_dedups_repeated_indices(linear_dimer):
    """A repeated index removes one atom, not three.

    Catches the delete being written as repeated removal rather than a set
    difference: `[1, 1, 1]` would then take out atoms 1, 2 and 3 as the list
    shifted underneath it -- three atoms gone for one the user picked, and the
    indices that vanished are neighbours, so the result still looks like a
    plausible molecule.

    Contract: `modify.delete_atoms` ("duplicates are tolerated").
    """
    out = delete_atoms(linear_dimer, [1, 1, 1])
    assert out.n_atoms == linear_dimer.n_atoms - 1
    assert out.elements == ["C", "H", "H", "H", "C"]


# --------------------------------------------------------------------- #
#  append_structure                                                     #
# --------------------------------------------------------------------- #


def _frag(elements, xs, regions=None):
    return Structure(
        elements=list(elements),
        positions=np.array([[float(x), 0.0, 0.0] for x in xs]),
        regions=regions or {},
    )


def test_append_centres_the_incoming_fragment_on_the_origin():
    """"just add the generated/loaded content centered at (0,0,0)" (user,
    2026-09-07).

    The origin is where `add_slab` places from and where an anchorless
    `add_atom` measures its offset, so a fragment arrives somewhere STATED
    rather than wherever its file happened to put it.
    """
    base = _frag(["C"], [5.0])
    add = _frag(["N", "N"], [10.0, 12.0])
    out, _notes = append_structure(base, add)
    assert out.n_atoms == 3
    # The base did not move.
    assert np.allclose(out.positions[0], [5.0, 0, 0])
    # The addition's CENTROID is the origin, and its internal geometry is
    # untouched -- centring is a rigid translation, not a rescale.
    added = out.positions[1:]
    assert np.allclose(added.mean(axis=0), [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(added[1] - added[0]), 2.0)


def test_append_numbers_a_label_the_open_structure_already_carries():
    """Two fragments both called ``benzene#`` would be ONE region, and a label
    exists to pick its atoms apart.  The incoming one is numbered.
    """
    base = _frag(["C"], [0.0], {"benzene#": [0]})
    add = _frag(["C"], [3.0], {"benzene#": [0]})
    out, notes = append_structure(base, add)
    assert out.regions == {"benzene#": [0], "benzene#2": [1]}
    assert any("benzene#2" in n for n in notes), notes
    # And again: the next one is 3, not a second 2.
    out2, _ = append_structure(out, add)
    assert sorted(out2.regions) == ["benzene#", "benzene#2", "benzene#3"]


def test_append_leaves_the_reserved_frozen_label_spelled_as_it_is():
    """The exception is about the NAME, not the atoms: something downstream
    acts on that exact spelling (`Structure.frozen_atoms`, the SIESTA
    constraints emitter, the PySCF freeze list).  Numbered, the incoming atoms
    would carry a label nothing recognises -- silently unfrozen.
    """
    base = _frag(["C", "C"], [0.0, 1.0], {FROZEN_LABEL: [0]})
    add = _frag(["N", "N"], [5.0, 6.0], {FROZEN_LABEL: [1]})
    out, _notes = append_structure(base, add)
    assert FROZEN_LABEL + "2" not in out.regions
    assert out.frozen_atoms == [0, 3], (
        "the incoming frozen atom did not stay frozen after the merge")


def test_append_keeps_the_open_structures_cell_and_says_so():
    """The canvas being built in owns its box; an incoming fragment contributes
    atoms.  A dropped lattice is invisible in the result, so it is said.
    """
    base = _frag(["C"], [0.0])
    base.cell = np.diag([10.0, 10.0, 10.0])
    base.__post_init__()
    add = _frag(["N"], [3.0])
    add.cell = np.diag([20.0, 20.0, 20.0])
    add.__post_init__()
    out, notes = append_structure(base, add)
    assert np.allclose(out.cell, np.diag([10.0, 10.0, 10.0]))
    assert any("not adopted" in n for n in notes), notes


def test_append_of_nothing_says_nothing_and_changes_nothing():
    """Appending an empty structure is a true no-op -- same atoms, same regions,
    and NO notice.

    Catches the notice machinery firing on nothing. `append_structure` reports
    what it did (a renamed label, a cell it did not adopt); an empty addition
    did none of those, and a spurious sentence on the Cell page after an action
    that changed nothing is how a notice surface stops being read.

    Contract: `web/molview.md` § 6.8 (a notice describes something that is
    true); the append op is `archive/2026-09-01-modify-redesign-plan.md`.
    """
    base = _frag(["C"], [0.0], {"a": [0]})
    out, notes = append_structure(base, Structure(elements=[],
                                                  positions=np.zeros((0, 3))))
    assert out.n_atoms == 1 and out.regions == {"a": [0]}
    assert notes == []


# --------------------------------------------------------------------- #
#  add_atom                                                             #
# --------------------------------------------------------------------- #


def test_add_atom_at_offset(linear_dimer):
    """SCIENCE. The offset is measured FROM THE ANCHOR ATOM, so the new atom lands
    at `positions[anchor] + offset`.

    Catches the offset being read as an absolute position. The anchor here is
    atom 5 at (3, 0, 0), not the origin, so the two readings differ -- with an
    anchor at the origin they would not, and the test would prove nothing. This
    is how a user places a bond at a stated length: get the reference point
    wrong and the "1.5 Å" they typed becomes a distance from somewhere else.

    Contract: `web/tabs.md` § 2 (the add-atom op); the anchorless reading is
    `test_add_atom_without_anchor_measures_from_the_origin`.
    """
    out = add_atom(linear_dimer, "S", anchor_index=5, offset=[0.0, 0.0, 1.5])
    assert out.n_atoms == linear_dimer.n_atoms + 1
    assert out.elements[-1] == "S"
    expected = linear_dimer.positions[5] + np.array([0.0, 0.0, 1.5])
    assert np.allclose(out.positions[-1], expected)


def test_add_atom_gets_fresh_residue_id(linear_dimer):
    """A newly added atom gets a residue id nobody else has -- specifically
    `max(existing) + 1`.

    Catches the new atom inheriting the anchor's residue. Residue ids are what
    separate the molecule from what was added to it (and what `add_slab` uses to
    keep an electrode separable), so an atom that joins the anchor's residue
    cannot afterwards be selected apart from it.

    Contract: `model/structure.md` § 1 (per-atom residue columns).
    """
    out = add_atom(linear_dimer, "S", 5, [0.0, 0.0, 1.5])
    anchor_residue = linear_dimer.residue_ids[5]
    new_residue = out.residue_ids[-1]
    assert new_residue != anchor_residue
    assert new_residue == max(linear_dimer.residue_ids) + 1


def test_add_atom_residue_name_default_and_override(linear_dimer):
    """A new atom's residue name defaults to `MOD` and honours an explicit
    override.

    Catches the override being ignored -- the caller passes `residue_name=` for
    a chemically meaningful group (a thiol cap, a solvent molecule) and gets
    `MOD` anyway, so the residue label that would let it be picked out later
    never lands.

    Contract: `model/structure.md` § 1.
    """
    out = add_atom(linear_dimer, "S", 0, [1, 0, 0])
    assert out.residue_names[-1] == "MOD"
    out2 = add_atom(linear_dimer, "S", 0, [1, 0, 0], residue_name="THI")
    assert out2.residue_names[-1] == "THI"


def test_add_atom_atom_name_defaults_to_element(linear_dimer):
    """With no `atom_name`, the new atom's name is its element symbol.

    Catches an appended atom arriving with an empty name column -- the PDB
    writer and the viewer's atom list both read `atom_names`, so a blank there
    shows as an unnamed row and writes a malformed PDB ATOM record.

    Contract: `model/structure.md` § 1.

    THIN: this restates a one-line `atom_name or element` default.
    """
    out = add_atom(linear_dimer, "Au", 0, [0, 0, 1])
    assert out.atom_names[-1] == "Au"


def test_add_atom_rejects_bad_anchor(linear_dimer):
    """An anchor index that is not an atom raises `IndexError`.

    Catches a stale selection silently landing an atom somewhere. The browser
    sends the anchor as an index; if atoms were deleted since, that index may no
    longer exist -- and the failure mode without a bounds check is not a crash
    but numpy's negative-index wraparound placing the atom relative to the
    WRONG anchor, which looks like a successful edit.

    Contract: `web/tabs.md` § 2.
    """
    with pytest.raises(IndexError):
        add_atom(linear_dimer, "S", anchor_index=99, offset=[0, 0, 0])


def test_add_atom_without_anchor_measures_from_the_origin():
    """``anchor_index=None`` is "nothing is selected", and it means the WORLD
    ORIGIN -- not atom 0, and not a refusal (user, 2026-09-07).

    The offset is then the position outright, which is the only reading that
    lets the same panel place an atom whether or not something is picked.

    NO ATOM SITS AT THE ORIGIN HERE, deliberately.  Against a fixture whose
    first atom is at (0, 0, 0) -- which `linear_dimer` is -- "measured from
    the origin" and "quietly fell back to atom 0" produce the SAME position,
    so the check that matters cannot fail and proves nothing.
    """
    offset = np.array([1.25, -0.5, 2.0])
    s = Structure(elements=["C", "O"],
                  positions=np.array([[3.0, 1.0, 0.0], [4.0, 1.0, 0.0]]))
    out = add_atom(s, "S", None, offset)
    assert out.n_atoms == 3
    assert np.allclose(out.positions[-1], offset)
    # Not a fallback to ANY existing atom -- which is the bug shape here.
    for i, p in enumerate(s.positions):
        assert not np.allclose(out.positions[-1], p + offset), (
            f"anchorless add fell back to atom {i}")


def test_add_atom_without_anchor_places_the_first_atom_on_an_empty_structure():
    """The case that forced the change: an EMPTY canvas has no index to pass,
    so requiring an anchor made a one-atom structure unbuildable from here.
    """
    empty = Structure(elements=[], positions=np.zeros((0, 3)))
    out = add_atom(empty, "C", None, [0.0, 0.0, 0.0])
    assert out.n_atoms == 1
    assert out.elements == ["C"]
    assert np.allclose(out.positions[0], [0.0, 0.0, 0.0])
    # The Structure's own default fills the columns an anchor would have
    # supplied, so nothing downstream can tell this atom had no anchor.
    assert out.chain_ids == ["A"]


def test_add_atom_rejects_unknown_element(linear_dimer):
    """Scientific guard: a non-periodic-table symbol must be rejected at the
    op boundary, not ride silently into the Structure and detonate later in
    the SIESTA/PySCF emitters (KeyError on ase.data.atomic_numbers)."""
    with pytest.raises(ValueError, match="unknown element symbol"):
        add_atom(linear_dimer, "Xx", anchor_index=0, offset=[1.0, 0, 0])


def test_add_atom_canonicalises_element_case(linear_dimer):
    """A mis-cased but real symbol ("au", "AU") is accepted and stored in
    canonical Element-case so downstream symbol tables key on it."""
    out = add_atom(linear_dimer, "au", anchor_index=0, offset=[1.0, 0, 0])
    assert out.elements[-1] == "Au"
    out2 = add_atom(linear_dimer, "FE", anchor_index=0, offset=[1.0, 0, 0])
    assert out2.elements[-1] == "Fe"


def test_add_atom_zero_offset_is_advisory_not_blocked(linear_dimer):
    """Advisory-not-enforcing (validation contract): a zero offset places the new
    atom on top of the anchor, but the op does NOT block -- it builds the
    structure (coincident atoms) and ``validate_geometry`` surfaces the clash as a
    non-blocking ``geometry.min_distance`` finding.  The user fixes it later /
    the generation gate enforces at emit time."""
    from molbuilder.validation import validate_geometry
    out = add_atom(linear_dimer, "H", anchor_index=0, offset=[0.0, 0.0, 0.0])
    assert out.n_atoms == linear_dimer.n_atoms + 1                 # no raise
    assert any(i.where == "geometry.min_distance"                  # advisory surfaced
               for i in validate_geometry(out))


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
    """SCIENCE. `center="first"` puts anchor 0 exactly at the world origin with the
    pair along +z.

    Catches the two centring modes being confused. Under `midpoint` the anchors
    straddle the origin; under `first` one of them IS the origin. `add_slab`
    stacks from the origin along z, so building an electrode against a molecule
    oriented with the wrong convention offsets the whole junction by half the
    molecular length -- a geometry error that produces a valid-looking cell.

    Contract: `web/tabs.md` § 2 (the orient op's centring modes).
    """
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
    """SCIENCE. Orienting to `axis="x"` puts the anchor pair on x with its midpoint
    at the origin.

    Catches a wrong entry in the axis lookup table. The z case is the one every
    other orient test uses, so an x/y swap -- or an x entry that is actually a
    reflection -- would leave all of them green while every structure a user
    orients along x came out rotated into the wrong plane.

    Contract: `web/tabs.md` § 2.
    """
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
    """SCIENCE. The 180-degree case is a PROPER rotation -- it flips the pair to
    +z without inverting the molecule's handedness.

    THE FAILURE THIS CATCHES.  `cross(v, target)` is the ZERO VECTOR when the
    two are antiparallel, so the general axis-angle formula divides by zero or
    normalises a zero vector into NaNs.  The implementation needs a special
    case, and a special case is exactly where an IMPROPER transform slips in: a
    reflection or an inversion also maps -z to +z, also leaves every distance
    unchanged, and is not a rotation.  `-I` would pass a distances-only test and
    turn every molecule into its mirror image -- a different compound, silently.

    WHY THE SIGNED TRIPLE PRODUCT.  A rotation has det = +1 and an improper
    transform det = -1, and the observable difference is CHIRALITY: the signed
    volume of any tetrahedron the atoms form. Distances, angles and the pair's
    final direction are all invariant under both, so none of them can tell the
    two apart. This is the only quantity that can.

    Contract: `web/tabs.md` § 2. MEASURED 2026-09-09 (#76): the previous fixture
    had TWO atoms, in which a reflection and a rotation are indistinguishable --
    replacing the branch with `-np.eye(3)` left this test, all 101 in this file,
    and all 320 in every file touching `orient` green, `validation/test_geometry.py`
    included.
    """
    # FOUR atoms, deliberately NOT coplanar: three of them would still span a
    # plane, and a reflection through that plane is undetectable.  A tetrahedron
    # has a handedness; that is the whole point of the fixture.
    s = Structure(
        elements=["C", "C", "N", "O"],
        positions=np.array([[0.0, 0.0,  0.0],
                            [0.0, 0.0, -2.5],     # the anchor pair, along -z
                            [1.3, 0.0, -0.8],
                            [0.0, 1.1, -1.7]]),
        title="flip",
    )

    def _chirality(p):
        """The signed volume of the tetrahedron, from atom 0."""
        return float(np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0]))

    before = _chirality(s.positions)
    assert abs(before) > 1e-6, "the fixture is coplanar and cannot see a mirror"

    out = orient_along_axis(s, (0, 1), axis="z", center="first")

    # 1. the original claim: the pair ends along +z.
    a1 = out.positions[1]
    assert np.isclose(a1[2], 2.5)
    assert abs(a1[0]) < 1e-10 and abs(a1[1]) < 1e-10

    # 2. the claim that makes it a ROTATION: handedness survives, sign and all.
    after = _chirality(out.positions)
    assert np.isclose(after, before, rtol=1e-9, atol=1e-9), (
        f"the transform is not a proper rotation: the signed volume went "
        f"{before:+.6f} -> {after:+.6f}. A sign flip is a mirror image -- a "
        f"different compound.")

    # 3. and it is rigid: every pair distance is unchanged.
    for i in range(4):
        for j in range(i + 1, 4):
            d0 = np.linalg.norm(s.positions[i] - s.positions[j])
            d1 = np.linalg.norm(out.positions[i] - out.positions[j])
            assert np.isclose(d0, d1), f"distance {i}-{j} moved: {d0} -> {d1}"


def test_orient_rejects_coincident_anchors():
    """SCIENCE. Two anchors at the same position are refused by name.

    Catches the zero-length direction vector reaching the normaliser. There is
    no axis through two coincident points, so the rotation is undefined --
    without the check, `v / |v|` is 0/0 and the whole structure comes back as
    NaN coordinates, which then travel silently into a written file.

    Contract: `web/tabs.md` § 2.
    """
    s = Structure(
        elements=["C", "C"],
        positions=np.array([[0, 0, 0], [0, 0, 0]]),
        title="coincident",
    )
    with pytest.raises(ValueError, match="coincident"):
        orient_along_axis(s, (0, 1), axis="z")


def test_orient_rejects_same_anchor_twice(linear_dimer):
    """The same atom given as both anchors is refused, naming "distinct".

    Catches the coincident-anchor check being the only guard: a user picking one
    atom twice in the UI is the common way to reach the degenerate case, and
    "the two anchors are at the same place" is the wrong sentence to show them --
    it sends them looking for overlapping atoms rather than at their selection.

    Contract: `web/tabs.md` § 2.
    """
    with pytest.raises(ValueError, match="distinct"):
        orient_along_axis(linear_dimer, (3, 3), axis="z")


def test_orient_rejects_bad_axis(linear_dimer):
    """An axis name outside {x, y, z} is refused.

    Catches an unknown axis falling through to a default. If the lookup used
    `.get(axis, z_axis)`, a typo in the request body would orient the structure
    along z and report success -- the user's molecule silently points the wrong
    way rather than the request being rejected.

    Contract: `web/tabs.md` § 2.
    """
    with pytest.raises(ValueError, match="axis"):
        orient_along_axis(linear_dimer, (0, 5), axis="w")


# --------------------------------------------------------------------- #
#  A whole-structure rigid transform moves the unit-cell BOX with the    #
#  atoms so it keeps wrapping them (structure-periodicity.md § 3c).       #
# --------------------------------------------------------------------- #


class TestRigidTransformMovesTheBox:
    """A rigid whole-structure transform (rotate / translate) must move the
    explicit unit-cell box WITH the atoms -- the lattice vectors and the
    world-space origin corner -- else the box stops wrapping the structure
    (the bug: a rotation left an axis-aligned box behind the rotated atoms)."""

    # +90° about z: R = [[0,-1,0],[1,0,0],[0,0,1]]; atoms/vectors map via `@ Rᵀ`.
    _RT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rᵀ

    def _boxed(self):
        return Structure(
            elements=["C", "H"],
            positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            cell=np.diag([10.0, 10.0, 20.0]),
            cell_origin=np.array([1.0, 2.0, 3.0]),
            axis_kind=("periodic", "periodic", "transport"),
            vacuum=(5.0, 5.0, 0.0),
        )

    def test_rotate_origin_pivot_rotates_vectors_and_origin(self):
        """SCIENCE. Under `center="origin"`, the atoms, the lattice VECTORS and the
        world-space origin CORNER all rotate the same way, and the non-geometric
        tags do not.

        Catches the bug this class was written for: a rotation that turned the atoms
        and left an axis-aligned box behind them. The box then no longer wraps the
        structure, so atoms sit outside a cell the user never changed, and a
        periodic run folds them onto images of the wrong neighbours. All three
        quantities are asserted against the explicit R-transpose so a rotation
        applied in the wrong sense or the wrong frame is visible as a value, not
        just as "something moved".

        Contract: `model/structure-periodicity.md` § 3c.
        """
        s = self._boxed()
        out = rotate_around_axis(s, axis="z", angle=90.0, center="origin")
        # atoms rotate about the world origin
        assert np.allclose(out.positions, [[0, 1, 0], [0, 2, 0]], atol=1e-9)
        # lattice VECTORS rotate the same way (cell @ Rᵀ)
        assert np.allclose(out.cell, s.cell @ self._RT, atol=1e-9)
        # the world-space origin CORNER rotates about the pivot (origin): origin @ Rᵀ
        assert np.allclose(out.cell_origin, s.cell_origin @ self._RT, atol=1e-9)
        # non-geometric tags carry verbatim
        assert out.axis_kind == s.axis_kind and out.vacuum == s.vacuum

    def test_rotate_centroid_pivot_rotates_origin_about_centroid(self):
        """SCIENCE. Under `center="centroid"`, the box corner rotates about THE SAME
        pivot the atoms do.

        Catches the two halves using different pivots -- atoms about the centroid,
        the corner about the world origin. Nothing raises; the box simply slides off
        the structure by an amount that grows with how far the molecule is from the
        origin, so it looks correct for a centred molecule and wrong for every other.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        s = self._boxed()
        out = rotate_around_axis(s, axis="z", angle=90.0, center="centroid")
        c = s.positions.mean(axis=0)
        # origin corner rotates about the SAME centroid the atoms pivot on
        assert np.allclose(out.cell_origin, (s.cell_origin - c) @ self._RT + c, atol=1e-9)
        assert np.allclose(out.cell, s.cell @ self._RT, atol=1e-9)

    def test_rotation_keeps_the_box_wrapping_the_atoms(self):
        # The invariant the fix exists for: after a whole-structure rotation, every
        # atom's FRACTIONAL coordinate in the (rotated) box is unchanged -- the box
        # still wraps the atoms exactly as before.
        """SCIENCE, and the invariant the other two tests are special cases of: after a
        whole-structure rotation, every atom's FRACTIONAL coordinate in the box is
        unchanged.

        Catches any rotation error the explicit-matrix tests miss, because it states
        the physics instead of the arithmetic: a rigid rotation of the system is a
        change of viewpoint, so nothing about where an atom sits INSIDE its cell may
        change. An arbitrary 37 degrees, not 90, so a mistake that happens to be
        symmetric under a quarter turn cannot hide.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        s = self._boxed()
        out = rotate_around_axis(s, axis="z", angle=37.0, center="centroid")
        def frac(st):
            rel = st.positions - st.cell_origin      # world -> box-corner frame
            return np.linalg.solve(st.cell.T, rel.T).T
        assert np.allclose(frac(out), frac(s), atol=1e-9)

    def test_translate_moves_origin_not_vectors(self):
        """SCIENCE. A translation moves the box CORNER with the atoms and leaves the
        lattice vectors alone.

        Catches the corner being left behind: translate the structure 10 Å and the
        box stays where it was, so the atoms are now outside a cell nobody edited.
        This is the exact asymmetry with rotation -- there the vectors turn too --
        and getting it backwards (translating the vectors) would resize the cell.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        s = self._boxed()
        out = s.translated((10.0, 0.0, 0.0))
        assert np.allclose(out.cell_origin, [11.0, 2.0, 3.0], atol=1e-9)  # corner follows
        assert np.allclose(out.cell, s.cell, atol=1e-9)                   # vectors invariant

    def test_orient_moves_the_box(self):
        # orient is ALWAYS whole-structure (anchors only define the rotation), so it
        # moves the box too: the atoms' fractional coords in the (rotated) box are
        # unchanged -- the box still wraps the structure.
        """SCIENCE. `orient_along_axis` is a whole-structure rotation, so it moves the
        box too -- fractional coordinates unchanged, and the vectors visibly no
        longer axis-aligned.

        Catches orient being treated as "a rotation of the selection". The anchors
        only DEFINE the rotation; every atom and the box turn. The second assertion
        is what makes this test able to fail for the right reason: without it, an
        op that rotated nothing at all would satisfy the fractional-coordinate check
        trivially.

        Contract: `model/structure-periodicity.md` § 3c.
        """
        s = self._boxed()
        out = orient_along_axis(s, (0, 1), axis="z", center="none")
        def frac(st):
            rel = st.positions - st.cell_origin
            return np.linalg.solve(st.cell.T, rel.T).T
        assert np.allclose(frac(out), frac(s), atol=1e-9)
        # the lattice vectors actually rotated (not left axis-aligned)
        assert not np.allclose(out.cell, s.cell)


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
    """An unknown rotation axis is refused.

    Catches the same silent-default failure as the orient sibling: a mistyped
    axis rotating the structure about z and reporting success, so the user's
    molecule ends up in an orientation nothing asked for and nothing reported.

    Contract: `web/tabs.md` § 2.
    """
    with pytest.raises(ValueError, match="axis"):
        rotate_around_axis(linear_dimer, axis="w", angle=10.0)


# --------------------------------------------------------------------- #
#  add_slab -- uniform (m, n, n_layers) per call                       #
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
    out = add_slab(single_anchor, element, plane, size,
                   start_z=2.0, orthogonal=orthogonal)
    n_metal = sum(1 for e in out.elements if e == element)
    assert n_metal == n_expected, (
        f"{element}({plane}) orthogonal={orthogonal} size={size}: "
        f"got {n_metal}, expected {n_expected}"
    )


def test_electrode_metadata_marks_atoms_as_ELC(single_anchor):
    """Spec § 5: electrode atoms get residue_name='ELC' and a fresh
    residue_id so the molecule and electrode are separable."""
    out = add_slab(single_anchor, "Au", "111", (2, 2, 1), start_z=2.0)
    elc_indices = [i for i, n in enumerate(out.residue_names) if n == "ELC"]
    assert len(elc_indices) > 0
    elc_residue_ids = {out.residue_ids[i] for i in elc_indices}
    anchor_residue = single_anchor.residue_ids[0]
    assert anchor_residue not in elc_residue_ids
    for i in elc_indices:
        assert out.elements[i]  == "Au"
        assert out.atom_names[i] == "Au"


# --------------------------------------------------------------------- #
#  Rejection paths -- per-(plane, orthogonal) constraints from ASE      #
# --------------------------------------------------------------------- #


def test_electrode_rejects_unsupported_element(single_anchor):
    """SCIENCE. An element outside the supported fcc set is refused by name -- both
    a non-fcc metal (Fe) and an fcc one we do not support (Al).

    Catches the slab builder being handed a metal whose crystal structure the
    fcc surface constructor does not describe. Fe is bcc: ASE would still return
    a slab, built on an fcc lattice constant that is not iron's, and the
    resulting electrode is a fictitious material. Al is the sharper half -- it
    IS fcc, so the refusal is about what this project has lattice constants and
    pseudopotentials FOR, not about crystallography, and it must be refused for
    that reason rather than accidentally accepted because the geometry works.

    Contract: `SUPPORTED_FCC_ELEMENTS` (the closed list, pinned by
    `test_electrode_supported_lists_match_table`) + `science/junction-cell.md`.
    """
    with pytest.raises(ValueError, match="unsupported electrode element"):
        add_slab(single_anchor, "Fe", "111", (2, 2, 1))
    with pytest.raises(ValueError, match="unsupported electrode element"):
        add_slab(single_anchor, "Al", "111", (2, 2, 1))


def test_electrode_rejects_unsupported_plane(single_anchor):
    """SCIENCE. A Miller index outside {100, 110, 111} is refused by name.

    Catches a plane reaching ASE that we have no surface geometry for. "101" is
    a legitimate index and ASE has no `fcc101` builder, so the failure without
    this check is an AttributeError from inside a library -- and the user is
    told nothing about which planes a junction can actually be built on. The
    three supported planes are the ones whose interlayer spacings and surface
    unit cells the junction geometry is derived from.

    Contract: `SUPPORTED_FCC_PLANES` + `science/junction-cell.md`.
    """
    with pytest.raises(ValueError, match="unsupported crystal plane"):
        add_slab(single_anchor, "Au", "101", (2, 2, 1))


def test_electrode_orthogonal_111_rejects_odd_n(single_anchor):
    """fcc(111) orthogonal supercell requires n even.  ASE's own error
    bubbles up as a ValueError with operation context."""
    with pytest.raises(ValueError, match="orthogonal=True"):
        add_slab(single_anchor, "Au", "111", (3, 3, 1), orthogonal=True)


@pytest.mark.parametrize("plane", ["100", "110"])
def test_electrode_primitive_100_110_rejects_non_orthogonal(single_anchor, plane):
    """fcc(100) and fcc(110) only support orthogonal=True.  ASE raises
    NotImplementedError; we re-wrap as ValueError with context."""
    with pytest.raises(ValueError, match="orthogonal=False"):
        add_slab(single_anchor, "Au", plane, (3, 3, 1), orthogonal=False)


def test_electrode_lattice_constant_override(single_anchor):
    """Explicit lattice_constant changes the slab's overall extent
    (proxy for 'the kwarg actually reached ASE')."""
    default = add_slab(single_anchor, "Au", "100", (3, 3, 1),
                       start_z=2.0, orthogonal=True)
    expanded = add_slab(single_anchor, "Au", "100", (3, 3, 1),
                        start_z=2.0, orthogonal=True, lattice_constant=5.0)
    def slab_extent(s):
        au_xy = np.array([p[:2] for e, p in zip(s.elements, s.positions)
                          if e == "Au"])
        return au_xy.max(axis=0) - au_xy.min(axis=0)
    d_extent = slab_extent(default).mean()
    e_extent = slab_extent(expanded).mean()
    assert e_extent > d_extent + 0.5


# --------------------------------------------------------------------- #
#  a junction: two slabs, one per side                                  #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("orthogonal,size,per_side", [
    # Spec § 2 walkthrough (now uniform per call): user calls the
    # two slabs with one (m, n, n_layers).  For stepped contacts
    # ("3×3 close, 4×4 further out") the user makes two add_slab calls
    # instead of one; covered separately by the stacked test below.
    (False, (3, 3, 2), 3 * 3 * 2),    # 18 atoms per side
    (True,  (3, 4, 2), 3 * 4 * 2),    # 24 atoms per side (n must be even)
])
def test_junction_end_to_end(orthogonal, size, per_side):
    """Mini "BDT-like" stub (S–S linker), oriented on z, with Au(111) on both
    sides — **built as two slabs**, one per flag, which is what a junction is
    now (redesign plan § 3.4).

    `add_symmetric_electrodes` did exactly this in one call and was deleted
    with the Junction panel.  The junction is still worth an end-to-end test;
    what changed is that each side's position is stated rather than derived
    from a gap.
    """
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

    # Two slabs, each at half the old `gap` from the same centre -- the
    # arithmetic the deleted wrapper did internally, now written down.
    # After orient with default midpoint centring, atom 3 is on +z (top),
    # atom 0 on -z (bottom).
    junction = oriented
    # BOTH SIDES CONTINUE THE CRYSTAL outward from the junction, and in the
    # walk vocabulary that is a different value per side: read along the
    # growth direction, going up from a layer is the forward walk and going
    # down from one is the backward walk.  (Under the old `stacking` this was
    # unsayable growing up -- the argument had no effect there at all.)
    for start_z, grow, sequence in ((+4.5, "+z", "ABC"), (-4.5, "-z", "ACB")):
        junction = add_slab(
            junction, "Au", "111", size, start_z=start_z, grow=grow,
            sequence=sequence, orthogonal=orthogonal,
        )
    n_au = sum(1 for e in junction.elements if e == "Au")
    # two sides × per_side atoms
    assert n_au == 2 * per_side
    assert junction.n_atoms == 4 + 2 * per_side
    elc_count = sum(1 for n in junction.residue_names if n == "ELC")
    assert elc_count == n_au
    elc_residues = {r for r, n in zip(junction.residue_ids, junction.residue_names)
                    if n == "ELC"}
    assert len(elc_residues) == 2


def test_junction_stepped_contacts_via_two_calls():
    """Spec § 2: stepped "3×3 close, 4×4 further out" pattern is built
    by two add_slab calls per side -- inner stack with one
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
    z0 = float(oriented.positions[0, 2])
    z3 = float(oriented.positions[3, 2])
    s1 = add_slab(oriented, "Au", "111", (3, 3, inner_layers),
                  start_z=z0 - inner_gap, grow="-z", sequence="ACB")
    s2 = add_slab(s1, "Au", "111", (3, 3, inner_layers),
                  start_z=z3 + inner_gap, grow="+z")
    # Outer stacks: 4×4 single layer, both sides, further out.
    s3 = add_slab(s2, "Au", "111", (4, 4, 1),
                  start_z=z0 - outer_gap, grow="-z", sequence="ACB")
    junction = add_slab(s3, "Au", "111", (4, 4, 1),
                        start_z=z3 + outer_gap, grow="+z")

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


# The element-aware DEFAULT contact distance retired 2026-09-03 with its
# supplier.  Junctions are not built from bond distances any more -- metal is
# added by hand, a slab is placed by its z offset -- so nothing asks for a
# default gap.  The table itself lives on as a REFERENCE shown beside a
# two-atom measurement; `tests/test_contact_distance_reference.py` owns it,
# including the assertion that the supplier stays gone.


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
    """A pivot name outside the supported set is refused -- `"midpoint"` is
    `orient`'s vocabulary, not `rotate`'s.

    Catches the two ops' centring vocabularies being conflated. `orient` takes
    midpoint/first/none, `rotate` takes origin/centroid; passing one op's word
    to the other must fail rather than fall through to a default, because a
    silent fallback rotates the structure about the wrong pivot and the result
    is a valid structure in the wrong place.

    Contract: `web/tabs.md` § 2.
    """
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
    ``add_slab`` no-op branches."""
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


# --------------------------------------------------------------------- #
#  All modify ops MUST carry frozen_atoms + regions through.            #
#                                                                       #
#  Audit task #186 (2026-06-02) found every modify op silently         #
#  dropped these fields when it returned a new Structure.  Concretely:  #
#    * Pure-rotation ops (orient_along_axis, rotate_around_axis) and   #
#      add_atom / add_slab must carry the lists through               #
#      verbatim (existing atom indices unchanged).                      #
#    * delete_atoms must remap surviving indices to the post-delete    #
#      0-based numbering and drop deleted atoms from the lists.        #
#                                                                       #
#  These tests close the contract loop end-to-end.                     #
# --------------------------------------------------------------------- #


def _struct_with_meta():
    return Structure(
        elements=["C"] * 5,
        positions=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                            [3., 0., 0.], [4., 0., 0.]]),
        regions={"electrode": [0, 4], "bridge": [1, 2, 3]},
        frozen_atoms=[0, 4],
    )


def test_delete_atoms_remaps_frozen_atoms_and_regions():
    """Delete atom 2 (which is in 'bridge', not frozen).  Surviving
    frozen atoms [0, 4] should remap to [0, 3] in the new 4-atom
    structure; 'bridge' [1, 2, 3] should remap to [1, 2] (atom 2 in
    the new numbering corresponds to old atom 3)."""
    s  = _struct_with_meta()
    s2 = delete_atoms(s, [2])
    assert s2.n_atoms == 4
    assert s2.frozen_atoms == [0, 3]
    assert s2.regions == {"electrode": [0, 3], "bridge": [1, 2],
                          FROZEN_LABEL: [0, 3]}


def test_delete_atoms_drops_deleted_indices_from_frozen():
    """Delete a frozen atom and verify it's removed from the list
    (not just shifted to a nonsense index)."""
    s  = _struct_with_meta()
    s2 = delete_atoms(s, [0])         # remove a frozen atom
    assert s2.frozen_atoms == [3]     # old 4 -> new 3; old 0 dropped
    assert "electrode" in s2.regions
    assert s2.regions["electrode"] == [3]   # only the surviving one


def test_delete_atoms_drops_empty_region_after_delete():
    """A region that loses all its members should disappear, not
    persist as an empty list in the result."""
    s = Structure(
        elements=["C"] * 3, positions=np.zeros((3, 3)),
        regions={"keep": [0, 1], "lose": [2]},
    )
    s2 = delete_atoms(s, [2])
    assert "lose" not in s2.regions
    assert s2.regions["keep"] == [0, 1]


def test_delete_atoms_no_op_branch_preserves_metadata():
    """When ``indices`` is empty (or all out-of-range) delete_atoms
    short-circuits to ``struct.copy()`` -- copy() must also preserve
    frozen_atoms + regions for this branch to hold."""
    s  = _struct_with_meta()
    s2 = delete_atoms(s, [])
    assert s2.frozen_atoms == s.frozen_atoms
    assert s2.regions == s.regions


def test_add_atom_preserves_existing_frozen_and_regions():
    """The new atom is appended at index n; existing frozen + region
    indices carry through unchanged.  The new atom is NOT frozen and
    NOT in any region by default."""
    s  = _struct_with_meta()
    s2 = add_atom(s, "H", anchor_index=0, offset=(0.5, 0.0, 0.0))
    assert s2.n_atoms == 6
    assert s2.frozen_atoms == [0, 4]
    assert s2.regions == {"electrode": [0, 4], "bridge": [1, 2, 3],
                          FROZEN_LABEL: [0, 4]}


def test_orient_along_axis_preserves_metadata():
    """Pure rotation: atom indices are unchanged."""
    s  = _struct_with_meta()
    s2 = orient_along_axis(s, anchor_indices=[0, 4], axis="z",
                            center="first")
    assert s2.frozen_atoms == [0, 4]
    assert s2.regions == {"electrode": [0, 4], "bridge": [1, 2, 3],
                          FROZEN_LABEL: [0, 4]}


def test_rotate_around_axis_preserves_metadata():
    """SCIENCE. A rotation carries `frozen_atoms` and every region through
    unchanged.

    Catches the atom LABELS being lost or renumbered by a pure rotation. No atom
    index changes under a rotation, so the frozen set and the transport regions
    must come back identical -- and if they do not, the calculation holds the
    wrong atoms fixed, or the transport code reads the wrong atoms as the left
    electrode. Neither is visible in the geometry, which is what makes it worth
    a test. Commemorates audit task #186 (2026-06-02), when every modify op
    dropped these fields.

    Contract: `model/structure-annotations.md` (what each reserved label means)
    + `web/molview.md` § 6.6 (reserved labels are interpreted downstream).
    """
    s  = _struct_with_meta()
    s2 = rotate_around_axis(s, axis="z", angle=90.0,
                             center="centroid")
    assert s2.frozen_atoms == [0, 4]
    assert s2.regions == {"electrode": [0, 4], "bridge": [1, 2, 3],
                          FROZEN_LABEL: [0, 4]}


def test_add_slab_preserves_existing_metadata():
    """Adding electrode atoms must not perturb the existing structure's
    frozen + region bookkeeping.  New electrode atoms are appended at
    indices [old_n, new_n) and are NOT auto-frozen / auto-regioned."""
    s = Structure(
        elements=["S"],
        positions=np.array([[0., 0., 0.]]),
        frozen_atoms=[0],
        regions={"anchor": [0]},
    )
    s2 = add_slab(
        s, element="Au", plane="100",
        size=(2, 2, 1), start_z=2.0, orthogonal=True,
    )
    assert s2.n_atoms > 1
    # The original S at index 0 must still be frozen + in 'anchor'.
    assert 0 in s2.frozen_atoms
    assert s2.regions == {"anchor": [0], FROZEN_LABEL: [0]}
    # No electrode atom was added to the existing lists.
    assert all(i == 0 for i in s2.frozen_atoms)
    assert all(i == 0 for v in s2.regions.values() for i in v)
