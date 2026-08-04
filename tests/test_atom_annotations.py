"""Phase 1 of the viewer/selection merge (atom-annotations.md): the
Structure per-atom annotations layer + all-channel remap."""
import numpy as np
import pytest

from molbuilder.structure import (FROZEN_LABEL, Structure, AtomChannel,
                                   copy_annotations, remap_annotations)
from molbuilder import modify


def _struct(n=5):
    return Structure(elements=["C"] * n,
                     positions=np.arange(n * 3, dtype=float).reshape(n, 3))


# ---- AtomChannel ---------------------------------------------------- #

def test_channel_kind_validation_and_defaults():
    assert AtomChannel("tag").data == []
    assert AtomChannel("value").data == {}
    with pytest.raises(ValueError, match="kind"):
        AtomChannel("bogus")


def test_channel_remapped_tag_and_value():
    tag = AtomChannel("tag", [0, 2, 4])
    r = tag.remapped({0: 0, 2: 1})          # 4 fell off
    assert r.kind == "tag" and r.data == [0, 1]
    val = AtomChannel("value", {0: 1.5, 2: -1.0, 4: 9.0})
    rv = val.remapped({2: 0, 4: 1})
    assert rv.data == {0: -1.0, 1: 9.0}


# ---- Structure annotations ----------------------------------------- #

def test_annotations_default_empty_and_backcompat():
    s = _struct()
    assert s.annotations == {}
    assert s.channels() == {}               # no regions/frozen either


def test_channels_unify_every_label_and_the_extras():
    s = _struct()
    s.regions = {"L-electrode": [0, 1]}
    s.frozen_atoms = [4]
    s.set_channel("charge", AtomChannel("value", {0: -1.0, 1: 0.5}))
    ch = s.channels()
    assert ch["L-electrode"].kind == "tag" and ch["L-electrode"].data == [0, 1]
    # The reserved label is a `tag` like every other label -- same kind, same
    # shape.  It was a `flag` channel synthesised beside the labels while it had
    # a store of its own (atom-annotations.md § 2).
    assert ch[FROZEN_LABEL].kind == "tag" and ch[FROZEN_LABEL].data == [4]
    assert ch["charge"].kind == "value" and ch["charge"].data == {0: -1.0, 1: 0.5}


def test_atom_annotations_per_atom_view():
    s = _struct()
    s.regions = {"bridge": [0]}
    s.frozen_atoms = [0]
    s.set_channel("charge", AtomChannel("value", {0: -1.0}))
    assert s.atom_annotations(0) == {"bridge": True, FROZEN_LABEL: True,
                                     "charge": -1.0}
    assert s.atom_annotations(3) == {}


def test_set_channel_rejects_a_name_a_label_already_has():
    """An extra channel sits BESIDE the labels, so it may not take a name a
    label already holds -- and the reserved label is covered by that one rule
    rather than by a clause naming it."""
    s = _struct()
    s.regions = {"L": [0]}
    s.frozen_atoms = [2]
    with pytest.raises(ValueError, match="already a label"):
        s.set_channel(FROZEN_LABEL, AtomChannel("flag", [1]))
    with pytest.raises(ValueError, match="already a label"):
        s.set_channel("L", AtomChannel("tag", [1]))
    with pytest.raises(ValueError, match="out of range"):
        s.set_channel("charge", AtomChannel("value", {99: 1.0}))


def test_validation_rejects_out_of_range_and_collision_at_construction():
    with pytest.raises(ValueError, match="out of range"):
        Structure(elements=["C"], positions=[[0, 0, 0]],
                  annotations={"x": AtomChannel("tag", [5])})
    with pytest.raises(ValueError, match="already a label"):
        Structure(elements=["C", "C"], positions=[[0, 0, 0], [1, 0, 0]],
                  regions={"L": [0]}, annotations={"L": AtomChannel("tag", [1])})


def test_copy_and_translated_carry_annotations():
    s = _struct()
    s.set_channel("charge", AtomChannel("value", {0: -1.0}))
    for clone in (s.copy(), s.translated([1, 0, 0])):
        assert clone.get_channel("charge").data == {0: -1.0}
        clone.annotations["charge"].data[0] = 99.0   # deep copy: original intact
    assert s.get_channel("charge").data == {0: -1.0}


# ---- remap helpers + modify integration ---------------------------- #

def test_remap_annotations_drops_emptied_channels():
    ann = {"a": AtomChannel("tag", [0, 1]), "b": AtomChannel("flag", [4])}
    out = remap_annotations(ann, {0: 0, 1: 1})   # 4 gone -> channel "b" emptied
    assert set(out) == {"a"} and out["a"].data == [0, 1]


def test_delete_atoms_remaps_annotations():
    s = _struct(5)
    s.set_channel("charge", AtomChannel("value", {2: -1.0, 4: 9.0}))
    s.regions = {"tail": [4]}
    out = modify.delete_atoms(s, [0, 1])         # keep 2,3,4 -> new idx 0,1,2
    assert out.get_channel("charge").data == {0: -1.0, 2: 9.0}
    assert out.get_channel("tail").data == [2]


def test_passthrough_op_carries_annotations_verbatim():
    s = _struct(3)
    s.set_channel("charge", AtomChannel("value", {0: -1.0}))
    out = modify.translate(s, [1.0, 0, 0]) if hasattr(modify, "translate") \
        else s.translated([1.0, 0, 0])
    assert out.get_channel("charge").data == {0: -1.0}


# ---- ATOM-METADATA comment-block persistence (§3, script_emit) ------ #

def test_atom_metadata_block_roundtrips_annotations():
    from molbuilder import script_emit as sc
    s = _struct(5)
    s.regions = {"bridge": [1]}
    s.frozen_atoms = [4]
    s.set_channel("charge", AtomChannel("value", {0: -1.0, 2: 0.5}))
    block = sc.emit_atom_metadata(
        regions=s.regions,
        annotations=s.annotations, n_atoms_total=5)
    # THE CURRENT schema, read from the one place that defines it.  This
    # pinned the literal "v4" until 2026-08-03 and had been failing since the
    # schema moved on -- a version bump is not a regression, and a test that
    # hard-codes a version number reports one every time.  What this test is
    # actually about is the ROUND TRIP below.
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION
    assert f"molstruct-json/v{SCHEMA_VERSION}" in block
    assert '"annotations"' in block
    back = _struct(5)
    assert sc.apply_inbody_atom_metadata(back, block) is True
    assert back.regions == {"bridge": [1], FROZEN_LABEL: [4]}
    assert back.frozen_atoms == [4]
    assert back.get_channel("charge").data == {0: -1.0, 2: 0.5}   # int keys


def test_atom_metadata_block_annotations_only():
    """Block emits even when ONLY annotations are present (no regions/frozen)."""
    from molbuilder import script_emit as sc
    s = _struct(3)
    s.set_channel("tail", AtomChannel("tag", [2]))
    block = sc.emit_atom_metadata(regions={},
                                  annotations=s.annotations, n_atoms_total=3)
    assert block is not None
    back = _struct(3)
    assert sc.apply_inbody_atom_metadata(back, block) is True
    assert back.get_channel("tail").data == [2]


def test_atom_metadata_none_when_empty():
    from molbuilder import script_emit as sc
    assert sc.emit_atom_metadata(regions={},
                                 annotations={}, n_atoms_total=3) is None


def test_concat_carries_and_reindexes_annotations():
    """Structure.concat must carry + re-index annotation channels (§2.1),
    not just regions/frozen (regression for the concat data-loss bug)."""
    a = _struct(3)
    a.set_channel("charge", AtomChannel("value", {0: -1.0, 2: 0.5}))
    a.set_channel("bridge", AtomChannel("tag", [1]))
    b = _struct(2)
    b.set_channel("charge", AtomChannel("value", {1: 2.0}))
    b.set_channel("bridge", AtomChannel("tag", [0]))
    m = Structure.concat([a, b])
    assert m.n_atoms == 5
    # value channel: a's {0,2} unioned with b's {1} -> offset +3 -> {4}
    assert m.get_channel("charge").data == {0: -1.0, 2: 0.5, 4: 2.0}
    # tag channel: a's [1] unioned with b's [0] -> offset +3 -> [3]
    assert m.get_channel("bridge").data == [1, 3]
    # a channel only on one input still survives
    a.set_channel("spin", AtomChannel("value", {0: 1.0}))
    m2 = Structure.concat([a, b])
    assert m2.get_channel("spin").data == {0: 1.0}
