"""``molbuilder/web/blueprints/_shared.py``: what a direct look at the two
serializers sees and a route's answer does not (`web/web-api.md` § 1) -- the
labels come from the kind table, the second serializer carries every canonical
key, each atom holds its own coordinates, and a channel stays on its atoms.
Which keys the real routes carry is `test_web.py`'s.
"""
from __future__ import annotations

import numpy as np

from molbuilder.structure import Structure
from molbuilder.web.blueprints._shared import (
    structure_to_dict,
    workspace_payload,
)


def _h2o() -> Structure:
    """Three atoms lying in no coordinate plane, each with its own x, y and z,
    so a coordinate read from the wrong place shows."""
    return Structure(
        elements  = ["O", "H", "H"],
        positions = np.array([
            [0.000,  0.000, 0.117],
            [0.957,  0.000, 0.233],
            [-0.239, 0.927, 0.351],
        ]),
        title="water",
    )


def test_labels_are_the_kind_tables():
    """The labels a calculation kind reads arrive with every structure,
    each with what it is for, read from the kind table -- a transport
    calculation's leads, bridge and buffer (`web/molview.md`) -- so no
    page spells them.  SILENT FAILURE: a page's copy drifting from the
    table after a rename, a lead offered under its old name."""
    served = workspace_payload(_h2o())["labels"]
    assert [e["name"] for e in served] == [
        "L-electrode", "R-electrode", "bridge", "buffer"]
    assert all(set(e) == {"name", "description"} and e["description"]
               for e in served), served
    # ...each sentence the kind table's own for that label.
    from molbuilder.kind_table import labels
    assert [(e["name"], e["description"]) for e in served] == \
        list(labels())


def test_atoms_carry_coordinates_on_the_atom():
    """Each atom row carries its own x/y/z as numbers (`web/web-api.md` § 1) --
    the atom is the geometric truth, not a re-parsed xyz string.  SILENT
    FAILURE: a row that disagrees with the structure, in what the Selection
    panel and every question about atoms read."""
    s = _h2o()
    atoms = workspace_payload(s)["atoms"]
    for i, row in enumerate(atoms):
        assert isinstance(row["x"], float)
        assert isinstance(row["y"], float)
        assert isinstance(row["z"], float)
        assert (row["x"], row["y"], row["z"]) == (
            float(s.positions[i][0]), float(s.positions[i][1]),
            float(s.positions[i][2]))


def test_emits_the_canonical_keys():
    """Every key the canonical door sends -- the second serializer adds
    to it, never re-lists it.  THE SILENT FAILURE (2.65.0, found
    headless): a hand-listed copy of the canonical keys dropped the
    kind table's `labels`, and the palette offered MolView's two."""
    d = structure_to_dict(_h2o())
    missing = set(workspace_payload(_h2o())) - set(d.keys())
    assert not missing, (
        f"structure_to_dict is missing canonical keys: {sorted(missing)}")


# --------------------------------------------------------------------- #
#  Annotations round-trip through the modify pipeline (§19.3.2)          #
# --------------------------------------------------------------------- #


def test_annotations_survive_a_delete_op_index_aligned():
    """§19.3.2 completeness: per-atom annotation channels ride the modify
    round-trip (envelope -> from_dict -> op reindexes -> structure_to_dict) so a
    delete keeps them ALIGNED to the surviving atoms -- the field the pipeline
    used to drop.

    The channels arrive inside the structure and `Structure.from_dict` applies
    them."""
    from molbuilder.web.blueprints._shared import struct_from_body
    from molbuilder.modify import delete_atoms
    body = {"structure": {
        "elements":  ["C", "C", "C", "C"],
        "positions": [[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]],
        "metadata":  {"annotations": {"spin": {"kind": "tag",
                                               "data": [2, 3]}}},
    }}
    struct = struct_from_body(body)
    assert struct.annotations["spin"].data == [2, 3]         # read off the body
    # Delete atom 1: old [0,2,3] -> new [0,1,2]; the tag on old {2,3} remaps to {1,2}.
    out = structure_to_dict(delete_atoms(struct, [1]))
    assert out["n_atoms"] == 3
    spin = out["annotations"]["spin"]
    assert spin["kind"] == "tag" and spin["data"] == [1, 2], out["annotations"]


def test_annotations_out_of_range_index_is_refused():
    """An annotation channel naming an atom that isn't there is REFUSED, by the
    same validator that refuses a malformed Structure anywhere -- and the
    message names the channel and the index.

    The channels ride inside the structure, so a bad index cannot produce a
    half-built structure to warn about -- the structure is simply not built."""
    import pytest
    from molbuilder.web.blueprints._shared import struct_from_body
    body = {"structure": {
        "elements":  ["H", "H"],
        "positions": [[0, 0, 0], [1, 0, 0]],
        # index 9 on a 2-atom struct
        "metadata":  {"annotations": {"bad": {"kind": "tag", "data": [9]}}},
    }}
    with pytest.raises(ValueError) as exc:
        struct_from_body(body)
    assert "annotations" in str(exc.value) and "9" in str(exc.value)
