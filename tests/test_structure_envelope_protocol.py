"""The structure envelope — one shape, both directions, added not swapped.

`web-api.md` § 1 defines it: a structure crosses in one envelope — the atoms as
**numbers**, the facts beside them, and (outbound only) the coordinate document
the server would write — and the server is the only thing that turns a structure
into a file.

Before it, a structure went *out* of the browser in four different shapes, one
per door family, and two of them required the caller to write a coordinate
document first. That is the only reason a `.xyz` writer exists in the browser at
all, and it had already drifted from `Structure.to_xyz`.

These tests pin the **mechanics**, because a transition rule that is only
described is a second protocol in disguise:

  * which shape a body is, and what wins when it carries both;
  * that responses carry both views, derived from one Structure;
  * that the envelope can express what the current designs need.
"""
from __future__ import annotations

import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import Structure
from molbuilder.web.blueprints._shared import struct_from_body
from molbuilder.workingcopy_structure import StructureCodec


def _envelope(**metadata):
    """`frozen_atoms=` is accepted here as a convenience and written where it
    belongs -- into `regions`. The metadata dict itself has no such key; the
    gate refuses one rather than dropping it."""
    """An envelope is the structure's own canonical dict — `Structure.to_dict()`'s
    shape — so these tests build it the way the codec does, not a shape invented
    beside it."""
    columns = {k: metadata.pop(k) for k in
               ("atom_names", "residue_ids", "residue_names", "chain_ids", "title")
               if k in metadata}
    frozen = metadata.pop("frozen_atoms", None)
    if frozen:
        metadata.setdefault("regions", {})["frozen_atoms"] = list(frozen)
    return {"structure": dict({
        "elements":  ["C", "O", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "metadata":  metadata,
    }, **columns)}


@pytest.fixture
def client():
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# --------------------------------------------------------------------- #
#  Which shape a body is                                                #
# --------------------------------------------------------------------- #

def test_a_body_is_an_envelope_when_it_carries_one_and_not_otherwise():
    """"A body carrying a `structure` key is an envelope; a body without one is
    read the old way. That is the whole test — one key, present or absent."

    Both shapes must work for as long as the legacy keys exist, because a
    browser tab loaded before a deploy is still calling the old way.
    """
    from_envelope = struct_from_body(_envelope())
    from_legacy = struct_from_body({"xyz": "3\n\nC 0 0 0\nO 1.4 0 0\nH 0 1 0\n"})

    assert from_envelope.n_atoms == from_legacy.n_atoms == 3
    assert list(from_envelope.elements) == list(from_legacy.elements)
    assert from_envelope.positions.tolist() == from_legacy.positions.tolist()


def test_when_a_body_carries_both_the_envelope_wins_and_the_legacy_is_ignored():
    """"Not merged: a caller that sends both is a caller mid-migration, and
    merging would let a stale field silently override a fresh one."

    The two halves here describe different structures on purpose — if anything
    from the legacy half survives, the merge that must not happen happened.
    """
    body = dict(_envelope(regions={"kept": [0]}))
    body["xyz"] = "1\n\nAu 9 9 9\n"          # a different structure entirely
    body["regions"] = {"stale": [0]}
    body["title"] = "the stale one"

    struct = struct_from_body(body)

    assert struct.n_atoms == 3, "the legacy geometry was used"
    assert list(struct.elements) == ["C", "O", "H"]
    assert struct.regions == {"kept": [0]}, (
        f"a legacy field survived beside the envelope: {struct.regions}"
    )
    assert struct.title != "the stale one"


def test_an_envelope_that_cannot_be_read_says_so_rather_than_guessing():
    """A malformed envelope is the caller's error and is answered as one. The
    alternative — filling in what is missing — is how a half-built structure
    reaches a calculation."""
    for broken, why in [
        ({"structure": {}}, "no atoms at all"),
        ({"structure": {"elements": [], "positions": []}}, "empty"),
        ({"structure": {"elements": ["C"]}}, "no positions"),
        ({"structure": {"elements": ["C"], "positions": []}}, "count mismatch"),
        ({"structure": {"elements": ["C"], "positions": [[0, 0]]}}, "not a point"),
    ]:
        with pytest.raises(ValueError):
            struct_from_body(broken)


# --------------------------------------------------------------------- #
#  What the envelope carries                                            #
# --------------------------------------------------------------------- #

def test_the_metadata_a_coordinate_file_cannot_hold_arrives_with_the_atoms():
    """The point of one envelope: the facts and the geometry travel together, so
    they cannot be one edit apart (§ 9.3, "the facts that leave together were
    read together")."""
    struct = struct_from_body(_envelope(
        regions={"L-electrode": [0], "α-helix": [2]},
        frozen_atoms=[1],
        cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        residue_names=["ALA", "ALA", "ALA"],
        title="a junction",
    ))

    # The reserved label is IN the label store with the others -- one store --
    # and the accessor is the way to ask which atoms carry it (§ 6.6).
    assert struct.regions == {"L-electrode": [0], "α-helix": [2], "frozen_atoms": [1]}
    assert list(struct.frozen_atoms) == [1]
    assert struct.cell is not None
    assert list(struct.residue_names) == ["ALA", "ALA", "ALA"]
    assert struct.title == "a junction"


def test_an_identity_column_is_honoured_only_at_full_length():
    """"A metadata column is sent only when every atom has one, otherwise `[]`."

    A short column is REFUSED rather than partially applied. The legacy body
    ignores a malformed column and keeps the default — a defensive choice for
    callers that were already sending them — but an envelope is new, and a
    caller sending three atoms and one residue name has a bug worth being told
    about. The server also does `max(residue_ids)`, where a hole poisons the
    comparison; this project has shipped that once.
    """
    with pytest.raises(ValueError):
        struct_from_body(_envelope(residue_names=["ALA"]))              # 1 of 3


def test_a_subset_envelope_is_accepted_and_its_map_back_is_the_callers():
    """"An envelope may describe a subset, with `geometry.source_index` giving
    each atom's number in the structure it came from. The receiver answers about
    the subset; the caller maps the coordinates back."

    So the receiver must not choke on it — and must not try to use it.
    """
    body = _envelope()
    body["structure"]["source_index"] = [4, 7, 9]

    struct = struct_from_body(body)

    assert struct.n_atoms == 3, "the subset was expanded to something it is not"
    assert list(struct.elements) == ["C", "O", "H"]


# --------------------------------------------------------------------- #
#  Responses carry both views, from one Structure                       #
# --------------------------------------------------------------------- #

def test_a_response_carries_the_envelope_beside_todays_keys(client):
    """"Added, not swapped." Nothing in the tabs changes on the day this lands,
    which is the whole reason the transition is safe."""
    answer = client.post("/api/build/load",
                         json={"text": "2\n\nC 0 0 0\nO 1 0 0\n",
                               "filename": "x.xyz"}).get_json()

    assert answer["ok"] is True
    for legacy in ("text", "atoms", "periodicity", "annotations",
                   "xyz", "elements", "atom_names", "residue_names"):
        assert legacy in answer, f"the legacy key '{legacy}' stopped being sent"

    envelope = answer["structure"]
    assert "document" not in envelope, (
        "a coordinate document is the export door's answer, not a per-response cost"
    )
    assert envelope["elements"] == answer["elements"]
    assert envelope["positions"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_the_two_views_cannot_disagree_because_they_come_from_one_structure(client):
    """They are two views of one object, not two objects. Anything a reader can
    compare between them must match."""
    answer = client.post("/api/build/load",
                         json={"text": "2\n\nC 0 0 0\nO 1 0 0\n",
                               "filename": "x.xyz"}).get_json()
    envelope = answer["structure"]

    assert envelope["elements"] == answer["elements"]
    assert envelope["metadata"]["cell"] == answer["periodicity"]["cell"]
    assert envelope["metadata"]["cell_origin"] == answer["periodicity"]["cell_origin"]
    assert len(envelope["positions"]) == answer["n_atoms"]


def test_a_document_in_a_REQUEST_is_ignored(client):
    """"`document` travels one way on purpose … a request that contains one is a
    request from something that wrote a file it should not have."""
    body = _envelope()
    body["structure"]["document"] = "9\nnonsense\nXx 1 2 3\n"

    struct = struct_from_body(body)
    assert struct.n_atoms == 3, "the request's document was read"


# --------------------------------------------------------------------- #
#  The export door, born in the envelope                                #
# --------------------------------------------------------------------- #

def test_export_returns_what_a_save_would_write(client, tmp_path):
    """Save-to-project and download differ only in destination — a promise about
    BYTES, which holds because both come from `StructureCodec.pair`.

    Since 2026-07-31 it is a promise about NAMES too. The door answers with the
    files named, so the two paths agree on *which files exist* and *what each is
    called*, not merely on their contents — the gap that let a download go out
    under a name a save would never have written.
    """
    import json as _json
    body = _envelope(regions={"L-electrode": [0]}, frozen_atoms=[1],
                     cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]])
    body["name"] = "same"
    answer = client.post("/api/structure/export", json=body).get_json()
    assert answer["ok"] is True

    struct = struct_from_body(body)
    from molbuilder.periodicity_gate import validate_periodicity
    struct, _ = validate_periodicity(struct)
    target = tmp_path / "same.xyz"
    StructureCodec().write(struct, target)

    handed = {f["name"]: f["text"] for f in answer["files"]}
    on_disk = {p.name: p.read_text(encoding="utf-8")
               for p in tmp_path.iterdir() if p.is_file()}
    assert set(handed) == set(on_disk), (
        f"the download and the save disagree about WHICH files exist: "
        f"{sorted(handed)} vs {sorted(on_disk)}")
    for name, text in handed.items():
        if name.endswith(".molstruct.json"):
            # Content, not text: `created_at` is provenance stamped per call.
            got = {k: v for k, v in _json.loads(text).items() if k != "created_at"}
            want = {k: v for k, v in _json.loads(on_disk[name]).items()
                    if k != "created_at"}
            assert got == want, f"{name}: different sidecar content"
        else:
            assert text == on_disk[name], (
                f"{name}: the bytes handed over differ from the bytes written")

    # `regions` carries the reserved label, so there is no second key to compare
    # -- and the designated read agrees with it on both sides.
    handed_sidecar = _json.loads(handed["same.molstruct.json"])
    saved = molstruct.load(molstruct.sidecar_path_for(target))
    assert molstruct.frozen_atoms(handed_sidecar) == [1]
    assert molstruct.frozen_atoms(saved) == [1]
    assert "frozen_atoms" not in handed_sidecar, (
        "a second key for a fact the label store already holds"
    )


def test_export_offers_no_sidecar_when_a_save_would_write_none(client):
    """"No `.json` means no metadata" is how the load door reads a pair, so a
    plain structure must come back as ONE file — exactly what a save writes."""
    answer = client.post("/api/structure/export", json=_envelope()).get_json()
    assert answer["ok"] is True
    assert [f["name"] for f in answer["files"]] == ["structure.xyz"], (
        f"a plain structure got a sidecar it would not have been saved with, "
        f"or a name nobody asked for: {answer['files']}")


def test_export_names_the_files_so_no_caller_has_to(client):
    """The caller supplies a STEM and the server supplies the rest, because the
    extension follows the format and the format follows the frame count
    (molview.md § 11.7). A path-shaped or missing stem is flattened rather than
    passed on — it reaches the browser as a download name."""
    frames = [[[0.0, 0, 0], [1.4, 0, 0], [0, 1.0, 0]],
              [[0.1, 0, 0], [1.5, 0, 0], [0, 1.1, 0]]]

    one = client.post("/api/structure/export",
                      json=dict(_envelope(), name="wire")).get_json()
    assert [f["name"] for f in one["files"]] == ["wire.xyz"]

    many = client.post("/api/structure/export",
                       json=dict(_envelope(), name="wire",
                                 frames=frames)).get_json()
    assert [f["name"] for f in many["files"]] == ["wire.xyz"], (
        "a range must be named under the extension that reads it back -- "
        "extended XYZ is a superset of plain XYZ and shares its extension")
    assert many["frames"] == 2

    for hostile in ("../../etc/passwd", "", "  ", ".."):
        answer = client.post("/api/structure/export",
                             json=dict(_envelope(), name=hostile)).get_json()
        assert "/" not in answer["files"][0]["name"], answer["files"]
        assert answer["files"][0]["name"] in ("passwd.xyz", "structure.xyz"), (
            f"{hostile!r} produced {answer['files'][0]['name']!r}")


def test_export_is_born_in_the_envelope_and_takes_nothing_else(client):
    """A new door has no callers owed compatibility, so it does not accept the
    `{xyz, sidecar}` blob its neighbours still take. Compatibility is a debt to
    existing callers, not a style."""
    legacy = client.post("/api/structure/export",
                         json={"blob": {"xyz": "1\n\nC 0 0 0\n", "sidecar": {}}})
    assert legacy.status_code == 400
    assert legacy.get_json()["ok"] is False


def test_export_reports_what_the_server_did_to_the_structure(client):
    """`notices` sits beside `ok`, never inside the structure: it belongs to the
    CALL — what this door did on the way through — not to the atoms.

    The export door runs the same periodicity gate the save door runs, which is
    what makes `test_export_returns_what_a_save_would_write` true rather than
    coincidental; anything that gate says is passed on instead of swallowed.
    (Which structures it heals is `structure-periodicity.md`'s business, and is
    tested there — this pins only that the channel exists and is a list.)
    """
    answer = client.post("/api/structure/export", json=_envelope()).get_json()
    assert isinstance(answer["notices"], list)


# --------------------------------------------------------------------- #
#  A geometry edit keeps what it was given                              #
# --------------------------------------------------------------------- #

def test_an_edit_returns_the_structure_it_was_sent_labels_and_cell_included(client):
    """The defect this closes, at the door where it bit.

    `applyOp` shipped `regions` and `periodicity` at the TOP level of the
    structure object. `Structure.from_dict` reads them from `metadata`, so they
    were read by nothing: every geometry edit answered HTTP 200 with the labels
    and the cell silently gone, and the browser adopted that answer as the new
    master copy. A user who translated a labelled junction lost every label,
    with no error anywhere.
    """
    body = {
        "structure": {
            "elements":  ["C", "O", "H"],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "metadata":  {
                "regions": {"L-electrode": [0], "frozen_atoms": [1]},
                "cell":    [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            },
        },
        # The op's own arguments ride on the BODY ROOT, which is where the
        # route reads them (§ 11.1). They were nested under `params` and read by
        # nothing: translate answered 200 and moved the structure by (0, 0, 0).
        "dx": 1.0, "dy": 0.0, "dz": 0.0,
    }
    answer = client.post("/api/modify/translate", json=body).get_json()

    assert answer["ok"] is True, answer
    # Read through the ENVELOPE -- the canonical place these live (§ 1).
    assert answer["structure"]["metadata"]["regions"] == {
        "L-electrode": [0], "frozen_atoms": [1]
    }, f"the edit dropped the labels it was given: {answer['structure']['metadata']}"
    assert answer["periodicity"]["cell"] == [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]
    # and it really did the edit
    assert answer["structure"]["positions"][0][0] == 1.0


def test_a_structure_key_the_envelope_does_not_define_is_refused(client):
    """Membership at the gate. A key outside the envelope is a fact the sender
    believes it transmitted; accepting the request and ignoring it is how the
    bug above stayed quiet. The error names the keys and where they belong."""
    body = {
        "structure": {
            "elements":  ["C", "O"],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "regions":   {"L-electrode": [0]},      # belongs under `metadata`
        },
        "dx": 1.0, "dy": 0.0, "dz": 0.0,
    }
    answer = client.post("/api/modify/translate", json=body)

    assert answer.status_code == 400
    assert "regions" in answer.get_json()["error"]
    assert "metadata" in answer.get_json()["error"]


def test_the_envelope_defines_exactly_these_members(client):
    """Pinned by membership, both directions: what a response emits is what a
    request may carry (plus the caller's own `source_index`, and `document`
    which travels outbound only)."""
    answer = client.post("/api/build/load",
                         json={"text": "2\n\nC 0 0 0\nO 1 0 0\n",
                               "filename": "x.xyz"}).get_json()

    assert set(answer["structure"]) == {
        "title", "elements", "positions", "atom_names", "residue_ids",
        "residue_names", "chain_ids", "metadata",
    }, f"the envelope's members drifted: {sorted(answer['structure'])}"


# --------------------------------------------------------------------- #
#  What an edit is allowed to change                                    #
# --------------------------------------------------------------------- #

def _labelled_body(**args):
    """A labelled, celled structure plus one op's arguments, flat."""
    body = {
        "structure": {
            "elements":  ["C", "O", "H", "N"],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "metadata":  {
                "regions": {"L-electrode": [0, 1], "bridge": [2],
                            "frozen_atoms": [3]},
                "cell":    [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            },
        },
    }
    body.update(args)
    return body


COUNT_PRESERVING = [
    ("translate", {"dx": 1.0, "dy": 0.0, "dz": 0.0}),
    ("rotate",    {"axis": "z", "angle": 90.0, "center": "centroid"}),
    ("orient",    {"anchors": [0, 1], "axis": "z", "center": "first"}),
    ("calibrate", {}),
]


@pytest.mark.parametrize("op,args", COUNT_PRESERVING, ids=[o for o, _ in COUNT_PRESERVING])
def test_an_op_that_keeps_the_atom_count_returns_every_label_unchanged(client, op, args):
    """"applyOp should not lose labels, and the atom index order is invariable."

    An operation that does not add or remove atoms MOVES them. The atom at index
    i before is the atom at index i after, so every label — reserved ones
    included — comes back naming exactly the atoms it named going in. Nothing
    about a rotation can change which atom is frozen.
    """
    before = _labelled_body(**args)["structure"]["metadata"]["regions"]
    answer = client.post(f"/api/modify/{op}", json=_labelled_body(**args)).get_json()

    assert answer["ok"] is True, answer
    after = answer["structure"]["metadata"]["regions"]
    assert after == before, f"{op} changed the labels: {before} -> {after}"
    assert answer["structure"]["elements"] == ["C", "O", "H", "N"], (
        f"{op} reordered or replaced the atoms"
    )


@pytest.mark.parametrize("op,args", COUNT_PRESERVING, ids=[o for o, _ in COUNT_PRESERVING])
def test_a_rigid_move_carries_the_box_with_the_atoms(client, op, args):
    """The box goes WITH the atoms, it is not preserved verbatim. A rigid
    rotation that left the lattice behind would describe a different crystal —
    nothing moved relative to anything else, so the box must turn too. What is
    invariant is the cell's SHAPE: same volume, same edge lengths."""
    import numpy as np
    answer = client.post(f"/api/modify/{op}", json=_labelled_body(**args)).get_json()

    assert answer["ok"] is True, answer
    cell = np.array(answer["periodicity"]["cell"], dtype=float)
    assert cell is not None and cell.shape == (3, 3)
    assert abs(abs(np.linalg.det(cell)) - 9.0 ** 3) < 1e-6, (
        f"{op} changed the cell VOLUME: {cell.tolist()}"
    )
    np.testing.assert_allclose(sorted(np.linalg.norm(cell, axis=1)), [9.0] * 3,
                               atol=1e-6, err_msg=f"{op} changed the cell's shape")


def test_moving_part_of_a_structure_is_an_argument_not_a_smaller_request(client):
    """§ 11.7: one path out — the whole structure goes, the whole structure comes
    back. A partial move names its atoms; it does not ship a smaller structure.

    And the box STAYS: those atoms moved relative to the ones that did not, so a
    lattice that followed them would stop describing the atoms it was drawn
    around. That is the difference from a rigid move, and it is why these are two
    operations rather than one with a flag.
    """
    body = _labelled_body(dx=1.0, dy=0.0, dz=0.0, indices=[0, 1])
    answer = client.post("/api/modify/translate", json=body).get_json()

    assert answer["ok"] is True, answer
    xs = [p[0] for p in answer["structure"]["positions"]]
    assert xs == [1.0, 2.0, 0.0, 0.0], f"the wrong atoms moved: {xs}"
    assert answer["periodicity"]["cell"] == [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]], (
        "a partial move dragged the box with it"
    )
    assert answer["structure"]["metadata"]["regions"] == {
        "L-electrode": [0, 1], "bridge": [2], "frozen_atoms": [3],
    }, "a partial move disturbed the labels"
    assert len(answer["structure"]["elements"]) == 4, (
        "the answer is not the whole structure"
    )


def test_delete_remaps_every_label_to_the_atoms_that_survived(client):
    """A shrink is the one case where the numbers must move, and they must move
    TOGETHER. Deleting atom 1 shifts 2 -> 1 and 3 -> 2: `bridge` and
    `frozen_atoms` follow their atoms, and the deleted atom's membership goes
    with it. A label left un-remapped points at whichever atom took the index —
    for `frozen_atoms` that means the wrong atoms are held still in the next
    calculation."""
    answer = client.post("/api/modify/delete",
                         json=_labelled_body(indices=[1])).get_json()

    assert answer["ok"] is True, answer
    assert answer["structure"]["elements"] == ["C", "H", "N"]
    assert answer["structure"]["metadata"]["regions"] == {
        "L-electrode": [0],      # atom 1 deleted, atom 0 kept its index
        "bridge":      [1],      # was 2
        "frozen_atoms": [2],     # was 3
    }


def test_add_atom_keeps_every_existing_label_and_labels_nothing_new(client):
    """A grow appends: the existing indices are untouched, and the new atom
    carries no label — the server does not guess that an atom attached to a
    frozen anchor should itself be frozen."""
    answer = client.post("/api/modify/add_atom", json=_labelled_body(
        element="H", anchor_index=3, offset=[1.0, 0.0, 0.0])).get_json()

    assert answer["ok"] is True, answer
    assert len(answer["structure"]["elements"]) == 5
    assert answer["structure"]["metadata"]["regions"] == {
        "L-electrode": [0, 1], "bridge": [2], "frozen_atoms": [3],
    }, "the appended atom disturbed the existing labels"
