"""The categorical sort — `plans/transport-design.md` § 4.1a (build P2).

Properties under guard, each named for its failure:

* the canonical layout comes out `[buffer][L][bridge][R][buffer]`
  whatever order the atoms arrived in;
* the bridge keeps its ORIGINAL relative order (stability) — the
  user's mental map of their molecule survives the sort;
* every index-carrying field crosses the sort with its atom — regions,
  frozen list, annotations, the per-atom parallel arrays;
* the permutation is a recorded bijection, both directions;
* the refusals NAME atoms: unlabeled, double-labeled, missing block,
  labels-vs-geometry disagreement.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import (REGION_BRIDGE, REGION_BUFFER,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from molbuilder.structure import AtomChannel, Structure
from molbuilder.transport.sort import (PERMUTATION_SCHEMA, SortError,
                                       categorical_sort)


def _junction(shuffle=True, extra=()):
    """Au(2 layers)-S-C-C-S-Au(2 layers) along z, one atom per layer,
    deliberately interleaved in the input order when ``shuffle``.
    ``extra`` rows are appended AFTER the shuffle (post_init fills the
    parallel arrays, so atoms must all exist at construction)."""
    # (element, z, partition)
    rows = [
        ("Au", 0.0, REGION_LEFT_ELECTRODE),
        ("Au", 2.0, REGION_LEFT_ELECTRODE),
        ("S",  4.0, REGION_BRIDGE),
        ("C",  5.0, REGION_BRIDGE),
        ("C",  6.0, REGION_BRIDGE),
        ("S",  7.0, REGION_BRIDGE),
        ("Au", 9.0, REGION_RIGHT_ELECTRODE),
        ("Au", 11.0, REGION_RIGHT_ELECTRODE),
    ]
    if shuffle:                      # a relaxation-order mess, on purpose
        order = [6, 2, 0, 4, 7, 1, 5, 3]
        rows = [rows[i] for i in order]
    rows = rows + list(extra)
    elements = [r[0] for r in rows]
    positions = np.array([[0.0, 0.0, r[1]] for r in rows])
    regions: dict = {}
    for i, r in enumerate(rows):
        if r[2] is not None:
            regions.setdefault(r[2], []).append(i)
    return Structure(elements=elements, positions=positions,
                     regions=regions), rows


class TestCanonicalOrder:

    def test_shuffled_input_comes_out_L_bridge_R(self):
        struct, rows = _junction(shuffle=True)
        out = categorical_sort(struct)
        got = out.structure
        # blocks in canonical positions; electrodes layer-major by z
        assert got.regions[REGION_LEFT_ELECTRODE] == [0, 1]
        assert got.regions[REGION_BRIDGE] == [2, 3, 4, 5]
        assert got.regions[REGION_RIGHT_ELECTRODE] == [6, 7]
        assert list(got.positions[:2, 2]) == [0.0, 2.0]
        assert list(got.positions[6:, 2]) == [9.0, 11.0]
        # the bridge is STABLE: it keeps the (shuffled) input's own
        # relative order -- S(4.0), C(6.0), S(7.0), C(5.0)
        assert got.elements[2:6] == ["S", "C", "S", "C"]
        assert list(got.positions[2:6, 2]) == [4.0, 6.0, 7.0, 5.0]

    def test_bridge_keeps_its_original_relative_order(self):
        """Stability: the molecule's atoms stay in the order the user
        knows, whatever z says."""
        struct, _ = _junction(shuffle=False)
        # scramble the bridge's z so a z-sort WOULD reorder it
        struct.positions[3, 2] = 6.5      # C above the next C
        struct.positions[4, 2] = 5.5
        out = categorical_sort(struct)
        assert out.structure.elements[2:6] == ["S", "C", "C", "S"], (
            "the bridge must keep input order -- it is the user's map")

    def test_buffers_land_outermost_by_nearest_end(self):
        struct, _ = _junction(shuffle=False,
                              extra=[("He", -2.0, REGION_BUFFER),
                                     ("He", 13.0, REGION_BUFFER)])
        out = categorical_sort(struct)
        assert out.structure.elements[0] == "He" and \
            out.structure.elements[-1] == "He"
        assert out.structure.positions[0, 2] == -2.0
        assert out.structure.positions[-1, 2] == 13.0

    def test_a_lopsided_buffer_still_lands_at_the_end_it_is_outside_of(
            self):
        """The side and the legality of a buffer atom are ONE question
        (found by reading, 2026-08-29).  They used to be two rules --
        the side from the midpoint of the WHOLE structure, the legality
        from the electrode blocks -- and on a lopsided junction they
        part company: padding above the upper lead that is taller than
        everything below it drags the midpoint up past that lead, files
        genuine top-buffer atoms at the BOTTOM, and then refuses a
        perfectly good structure for "buffer inside the electrode"."""
        struct, _ = _junction(
            shuffle=False,
            extra=[("He", 13.0, REGION_BUFFER),
                   ("He", 40.0, REGION_BUFFER),
                   ("He", 80.0, REGION_BUFFER)])   # midpoint = 40.0
        out = categorical_sort(struct)             # no refusal
        assert list(out.structure.positions[-3:, 2]) == [13.0, 40.0, 80.0], (
            "every one of these sits above the upper electrode, so all "
            "three belong at the high end")
        assert out.structure.elements[0] == "Au", (
            "nothing should have been pushed below the lower lead")

    def test_electrode_blocks_are_layer_major_along_transport(self):
        struct, _ = _junction(shuffle=False)
        # swap the two L-electrode layers in INPUT order; z must win
        struct.positions[0, 2], struct.positions[1, 2] = 2.0, 0.0
        out = categorical_sort(struct)
        assert list(out.structure.positions[:2, 2]) == [0.0, 2.0], (
            "electrode blocks sort by transport coordinate -- the "
            "extracted cell inherits this order, so it must be "
            "layer-major, not input-order")


class TestBookkeepingCrossesTheSort:

    def test_permutation_is_a_recorded_bijection(self):
        struct, rows = _junction(shuffle=True)
        out = categorical_sort(struct)
        n = len(rows)
        assert sorted(out.sorted_to_original) == list(range(n))
        for old in range(n):
            assert out.sorted_to_original[out.original_to_sorted[old]] == old
        side = out.sidecar()
        assert side["schema"] == PERMUTATION_SCHEMA
        assert side["original_to_sorted"] == list(out.original_to_sorted)

    def test_every_atom_keeps_its_identity(self):
        """The user's 'nothing missed': same multiset of
        (element, position), each exactly once."""
        struct, _ = _junction(shuffle=True)
        out = categorical_sort(struct)
        before = sorted((e, tuple(p)) for e, p in
                        zip(struct.elements, struct.positions))
        after = sorted((e, tuple(p)) for e, p in
                       zip(out.structure.elements, out.structure.positions))
        assert before == after

    def test_frozen_and_annotations_and_names_travel(self):
        struct, rows = _junction(shuffle=True)
        # freeze the electrode atoms (by INPUT index), tag one bridge C
        struct.frozen_atoms = [i for i, r in enumerate(rows)
                               if r[2] != REGION_BRIDGE]
        c_in = next(i for i, r in enumerate(rows)
                    if r[0] == "C" and r[1] == 5.0)
        struct.annotations = {"charge": AtomChannel("value",
                                                    {c_in: -0.3}, None, None)}
        struct.atom_names = [f"n{i}" for i in range(len(rows))]
        out = categorical_sort(struct)
        got = out.structure
        # frozen == the four electrode positions in the sorted frame
        assert got.frozen_atoms == [0, 1, 6, 7]
        # the tagged C is the z=5.0 carbon wherever it landed
        (tagged, val), = got.annotations["charge"].data.items()
        assert got.elements[tagged] == "C" and \
            got.positions[tagged, 2] == 5.0 and val == -0.3
        # parallel array rides with its atom
        for j, name in enumerate(got.atom_names):
            assert name == f"n{out.sorted_to_original[j]}"

    def test_cell_title_vacuum_are_untouched(self):
        struct, _ = _junction(shuffle=True)
        struct.title = "shuffled junction"
        struct.cell = np.eye(3) * 20.0
        struct.vacuum = (5.0, 5.0, 0.0)
        out = categorical_sort(struct)
        assert out.structure.title == "shuffled junction"
        assert np.allclose(out.structure.cell, struct.cell)
        assert out.structure.vacuum == (5.0, 5.0, 0.0)


class TestRefusalsNameAtoms:

    def test_an_unlabeled_atom_is_refused_by_index_and_element(self):
        struct, _ = _junction(shuffle=False, extra=[("H", 8.0, None)])
        with pytest.raises(SortError) as e:
            categorical_sort(struct)
        assert "atom 8 (H)" in str(e.value)
        assert "no partition label" in str(e.value)

    def test_a_double_labeled_atom_is_refused_naming_both(self):
        struct, _ = _junction(shuffle=False)
        struct.regions[REGION_BRIDGE] = struct.regions[REGION_BRIDGE] + [0]
        with pytest.raises(SortError) as e:
            categorical_sort(struct)
        msg = str(e.value)
        assert "atom 0 (Au)" in msg
        assert REGION_LEFT_ELECTRODE in msg and REGION_BRIDGE in msg

    def test_a_buffer_atom_inside_the_blocks_is_refused(self):
        """§ 3 buffer sanity: buffer means padding OUTSIDE the
        electrodes.  One at z = 5 sits mid-bridge -- the sort would
        park it at an outer end its geometry contradicts."""
        struct, _ = _junction(shuffle=False,
                              extra=[("He", 5.0, REGION_BUFFER)])
        with pytest.raises(SortError) as e:
            categorical_sort(struct)
        msg = str(e.value)
        assert "atom 8 (He)" in msg and "OUTSIDE" in msg

    def test_interface_rides_on_top_without_tripping_the_partition(self):
        struct, _ = _junction(shuffle=False)
        struct.regions["interface"] = [2, 5]        # the two S anchors
        out = categorical_sort(struct)              # must NOT refuse
        assert out.structure.regions["interface"] == [2, 5]

    def test_a_missing_block_is_refused_by_name(self):
        struct, _ = _junction(shuffle=False)
        del struct.regions[REGION_RIGHT_ELECTRODE]
        struct.regions[REGION_BRIDGE] += [6, 7]     # relabel to keep all
        with pytest.raises(SortError) as e:
            categorical_sort(struct)
        assert REGION_RIGHT_ELECTRODE in str(e.value)

    def test_swapped_names_sort_by_geometry_and_carry_a_note(self):
        """CHECK z, WARN, the author decides (user ruling, 2026-08-29).
        A junction labeled the other way round is not mislabeled -- it
        biases the other end -- so the sort must NOT refuse it.  What
        it must do is put the LOWER block first anyway (that block is
        the -A3 lead; the upper one first would aim a self-energy into
        the bridge) and say what it saw."""
        struct, _ = _junction(shuffle=False)
        low = list(struct.regions[REGION_LEFT_ELECTRODE])
        struct.regions[REGION_LEFT_ELECTRODE] = \
            struct.regions[REGION_RIGHT_ELECTRODE]
        struct.regions[REGION_RIGHT_ELECTRODE] = low

        res = categorical_sort(struct)          # no refusal
        # The block that was lowest in z still leads the atom list --
        # now wearing the R-electrode name.
        n_low = len(low)
        assert sorted(res.structure.regions[REGION_RIGHT_ELECTRODE]) == \
            list(range(n_low)), "the LOWER block must lead, whatever it is called"
        assert res.notes, "an inverted pair must be reported, not passed silently"
        note = res.notes[0]
        assert "HIGH-z" in note and "+V/2" in note
