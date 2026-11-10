"""``molbuilder/web/blueprints/_shared.py``: what a direct look at the one
structure answer (`structure_answer`) sees and a route's answer does not
(`web/web-api.md` § 1) -- the labels come from the kind table, and a channel
stays on its atoms.  Which keys the real routes carry is `test_web.py`'s; the
structure's dict reading back whole is `test_structure_authority_roundtrip.py`'s.
"""
from __future__ import annotations

import numpy as np

from molbuilder.structure import Structure
from molbuilder.web.blueprints._shared import structure_answer


def test_labels_are_the_kind_tables():
    """The labels a calculation kind reads arrive with every structure,
    each with what it is for, read from the kind table -- a transport
    calculation's leads, bridge and buffer (`web/molview.md`) -- so no
    page spells them.  SILENT FAILURE: a page's copy drifting from the
    table after a rename, a lead offered under its old name."""
    served = structure_answer(
        Structure(elements=["O"], positions=np.zeros((1, 3))))["labels"]
    assert [e["name"] for e in served] == [
        "L-electrode", "R-electrode", "bridge", "buffer"]
    assert all(set(e) == {"name", "description"} and e["description"]
               for e in served), served
    # ...each sentence the kind table's own for that label.
    from molbuilder.kind_table import labels
    assert [(e["name"], e["description"]) for e in served] == \
        list(labels())


# --------------------------------------------------------------------- #
#  Annotations round-trip through the modify pipeline (§19.3.2)          #
# --------------------------------------------------------------------- #


def test_annotations_survive_a_delete_op_index_aligned():
    """§19.3.2 completeness: per-atom annotation channels ride the modify
    round-trip (envelope -> from_dict -> op reindexes -> structure_answer) so a
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
    out = structure_answer(delete_atoms(struct, [1]))["structure"]
    assert len(out["elements"]) == 3
    channels = out["metadata"]["annotations"]
    spin = channels["spin"]
    assert spin["kind"] == "tag" and spin["data"] == [1, 2], channels


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
