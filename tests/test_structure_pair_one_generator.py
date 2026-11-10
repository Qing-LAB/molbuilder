"""One generator: the bytes a save writes and the bytes a download gets.

`StructureCodec.pair` is the single place a Structure becomes the two things
that represent it outside memory — the coordinate document and the sidecar
payload. One generator, and one adapter per destination: `write` puts it on
disk (`/api/structure/save`), `files` hands it over as bytes WITH THEIR NAMES
(`/api/structure/export`), and `read` brings it back (`/api/build/load`).

Before they shared it there were three code paths computing the same three
calls, agreeing by coincidence: `write` recomputed the payload inline, and
`files` serialised the JSON without `ensure_ascii=False`, so a non-ASCII region
label came out escaped on one path and literal on the other. Two files for one
structure, depending which half of the application produced them.

These tests assert the property that makes that impossible: **the same structure
produces the same bytes, whichever door it leaves by.**
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import StructureCodec


@pytest.fixture(autouse=True)
def _a_clock_that_does_not_tick_mid_test(monkeypatch):
    """Freeze `created_at`, so a second boundary cannot fail these tests.

    The property here is *the same structure produces the same bytes,
    whichever door it leaves by* -- and each door stamps the sidecar with the
    wall clock as it goes.  `molstruct.stamp` has ONE-SECOND resolution, so a save
    and an export that straddle a tick differ by one character and the
    comparison fails for the one reason it does not care about.  Rare, timing
    dependent, and it never reproduces on a re-run: the shape of flake that
    gets a real assertion deleted for being "unreliable".

    Frozen rather than stripped from the comparison, deliberately.  The
    timestamp is a field the writer really emits, and a test that skipped it
    would stop noticing if one door ever stamped a field the other did not.
    """
    monkeypatch.setattr("molbuilder.sidecars.molstruct.stamp",
                        lambda: "2026-09-05T00:00:00Z")


def _with_metadata() -> Structure:
    """Metadata applied the way it actually arrives — through
    `apply_metadata_dict`, the one dict->struct authority, which coerces and
    validates. Assigning the fields directly bypasses that and builds a
    Structure the rest of the system could not have made."""
    s = Structure.from_xyz("3\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    s.apply_metadata_dict({
        "regions":       {"L-electrode": [0], "α-helix": [2],
                          "frozen_atoms": [1]},
        "cell":          [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
    })
    # THROUGH THE PERIODICITY GATE, as every structure that reaches a save has
    # been: a cell whose axes still say "isolated" is inconsistent, and the gate
    # judges it. The save route and the export route both run the gate, so both
    # answer the same way — but a fixture that skips it is a structure the
    # application could not have produced, and comparing it against a checked one
    # measures the gate rather than the generator.
    from molbuilder.periodicity_gate import validate_periodicity
    checked, _ = validate_periodicity(s)
    return checked


def _plain() -> Structure:
    return Structure.from_xyz("2\n\nC 0 0 0\nO 1 0 0\n")


# --------------------------------------------------------------------- #
#  The disk and the wire agree                                          #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("make", [_with_metadata, _plain], ids=["metadata", "plain"])
def test_what_write_puts_on_disk_is_what_files_hands_over(tmp_path, make):
    """`files()` is "what write() writes, without writing it". If the two can
    disagree, one of them is a second answer to what a saved structure looks
    like — which is the whole defect this pins.
    """
    struct = make()
    target = tmp_path / "wire.xyz"
    StructureCodec().write(struct, target)

    handed = dict(StructureCodec().files(struct, target))

    on_disk = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert set(handed) == set(on_disk), (
        f"files() and write() disagree about WHICH files exist: "
        f"{sorted(p.name for p in handed)} vs {sorted(p.name for p in on_disk)}"
    )
    for path, given in handed.items():
        assert given == on_disk[path], (
            f"{path.name}: the bytes handed over differ from the bytes written"
        )


@pytest.mark.parametrize("target, offered", [("wire.pdb", "wire.xyz"),
                                              ("wire", "wire.xyz")],
                         ids=["a pdb", "a name no reader opens"])
def test_a_structure_is_written_as_its_xyz_alone(tmp_path, target, offered):
    """`model/structure.md` § 2.4 (the user, 2026-11-09): molbuilder writes a
    structure as a `.xyz` with its sidecar, and refuses any other name,
    naming the `.xyz`, nothing written (`StructureCodec.refused_name`).  THE
    SILENT FAILURES: a `.pdb` with a sidecar of its own, beside the `.xyz`
    whose stem it shares -- one sidecar, two geometries, a save of one taking
    over the other's labels; a pair under a name no reader opens."""
    with pytest.raises(ValueError, match=f"Name it {offered}"):
        StructureCodec().write(_plain(), tmp_path / target)
    assert not any(tmp_path.iterdir())


def test_a_non_ascii_label_survives_both_paths_identically(tmp_path):
    """The concrete drift this closes: `files()` used `json.dumps` without
    `ensure_ascii=False` while the sidecar's other writer had it, so "α-helix" was `\\u03b1-helix`
    in one and literal in the other. Same structure, two different files.
    """
    struct = _with_metadata()
    target = tmp_path / "greek.xyz"
    StructureCodec().write(struct, target)

    written = molstruct.sidecar_path_for(target).read_text(encoding="utf-8")
    handed = dict(StructureCodec().files(struct, target))
    given = handed[molstruct.sidecar_path_for(target)].decode("utf-8")

    assert "α-helix" in written, "the label was escaped on the way to disk"
    assert given == written
    assert json.loads(given)["regions"]["α-helix"] == [2]


def test_a_plain_structure_gets_its_sidecar_from_either(tmp_path):
    """Every pair has its sidecar, a structure with nothing to say about its
    atoms included -- it carries the pin (`structure-molstruct.md` § 3) --
    from the writer AND from anything that hands the pair over.
    """
    target = tmp_path / "plain.xyz"
    StructureCodec().write(_plain(), target)

    side = molstruct.sidecar_path_for(target)
    assert side.is_file(), "a plain structure was written without its sidecar"
    handed = [p for p, _ in StructureCodec().files(_plain(), target)]
    assert handed == [target, side], f"the pair handed over: {handed}"


# --------------------------------------------------------------------- #
#  The names, which the caller must not be left to derive                #
# --------------------------------------------------------------------- #

def test_a_range_and_a_single_frame_are_both_named_xyz():
    """The defect this closes, and it was live.

    `pair()` named a range `.extxyz`, on the reasoning that a name should say
    which format it holds. It should not: extended XYZ is a strict SUPERSET of
    plain XYZ -- the cell rides in the comment line, which a plain reader skips
    -- so `.xyz` covers both, and that is the convention ASE established.

    And it could not: `load()` dispatches on the extension and takes `.xyz` /
    `.pdb` only, so a trajectory saved into a project COULD NOT BE REOPENED.
    """
    struct = _with_metadata()
    frames = [struct.positions, struct.positions + 0.1]

    one = [p.name for p, _ in StructureCodec().files(struct, "wire")]
    many = [p.name for p, _ in StructureCodec().files(
        struct.with_frames(frames), "wire")]

    assert one == ["wire.xyz", "wire.molstruct.json"], one
    assert many == ["wire.xyz", "wire.molstruct.json"], (
        f"a range must be named under the extension that reads it back: {many}")


def test_a_stem_that_already_carries_a_suffix_is_corrected_not_appended_to():
    """Both callers exist: the export door hands a bare stem, and a comparison
    against `write()` hands a full path. Neither may end up with two suffixes,
    and a stem carrying a suffix that is NOT a geometry one keeps it -- `run.v2`
    is a name, not a mistake, and `with_suffix` would eat the `.v2`.
    """
    struct = _with_metadata()
    frame_set = struct.with_frames([struct.positions, struct.positions + 0.1])
    name = lambda target, s=struct: StructureCodec().files(  # noqa: E731
        s, target)[0][0].name

    assert name("wire.xyz") == "wire.xyz"
    assert name("wire.xyz", frame_set) == "wire.xyz"
    assert name("wire.extxyz") == "wire.xyz", (
        "a non-standard extension is corrected to the one load() accepts")
    assert name("run.v2") == "run.v2.xyz", "the .v2 was eaten"
    assert name("prot.pdb") == "prot.xyz", (
        "a .pdb brought in is written as its .xyz (`StructureCodec.named`)")
    assert name("run.v2", frame_set) == "run.v2.xyz"


@pytest.mark.parametrize("stated", [None, [1.0, 1.0, 1.0]],
                         ids=["automatic", "assigned-origin"])
def test_channels_survive_the_pipeline_through_the_real_translators(
        isolated_projects_root, stated):
    """The 2026-08-20 widening, proven disk -> disk: a pair whose sidecar
    carries per-atom annotation CHANNELS -- and mixed axis kinds, a vacuum,
    and a stated offset or none -- goes ``/api/build/load`` -> the browser's
    REAL translations (``structureFromServer`` -> ``structureForServer``,
    executed under node — not a Python mirror) -> ``/api/structure/export``
    -> sidecar again, EQUAL apart from provenance.  Until the widening the
    fold never read the wire's ``annotations`` block, so any trip through the
    viewer erased them.  The offset is the raw one the wire echoes; the
    resolved ``box_corner`` beside it must never come back (plan § 5q.3), and
    an Automatic pair must come back Automatic, not frozen at the rule's
    corner (`structure-periodicity.md` § 6.0).

    KILL SITES, verified red at introduction: the fold skipping
    ``payload.annotations``; the unfold dropping a column.
    """
    import json

    from molbuilder.structure import Structure
    from molbuilder.web.app import create_app
    from molbuilder.workingcopy_structure import StructureCodec

    from tests._node_esm import run_node

    tree = isolated_projects_root
    struct = Structure.from_dict({
        "elements": ["O", "H", "H"],
        "positions": [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        "metadata": {
            "regions": {"frozen": [0]},
            "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 30]],
            "engine_offset": stated,
            "axis_kind": ["periodic", "periodic", "isolated"],
            "vacuum": [0, 0, 8],
            "annotations": {
                "spin_up": {"kind": "tag", "data": [0, 2],
                            "color": "#f00"},
                "charge": {"kind": "value",
                           "data": {"0": -0.8, "1": 0.4},
                           "fdf": "net-charge"},
            },
        },
    })
    StructureCodec().write(struct, tree / "wet.xyz")
    side_in = json.loads((tree / "wet.molstruct.json").read_text())
    assert set(side_in["annotations"]) == {"spin_up", "charge"}, (
        "precondition: the sidecar holds the channels")

    client = create_app(config={}).test_client()
    wire = client.post("/api/build/load",
                       json={"path": str(tree / "wet.xyz")}).get_json()
    assert wire.get("ok"), wire

    jobs = (Path(__file__).resolve().parents[1]
            / "molbuilder/web/static/lib/molview/model-jobs.js")
    env = run_node([], (
        "const m = await import(" + json.dumps(jobs.as_uri()) + ");\n"
        "const adopted = m.structureFromServer("
        + json.dumps(wire) + ");\n"
        "console.log(JSON.stringify(m.structureForServer("
        "adopted.structure, adopted.coordinates)));\n"
    ))

    out = client.post("/api/structure/export",
                      json={"structure": env, "name": "wet"}).get_json()
    assert out["ok"], out
    side_out = json.loads(next(f["text"] for f in out["files"]
                               if f["name"].endswith(".json")))

    volatile = {"created_at", "schema_version"}
    diffs = {k: (side_in.get(k), side_out.get(k))
             for k in set(side_in) | set(side_out)
             if k not in volatile and side_in.get(k) != side_out.get(k)}
    assert diffs == {}, f"the pipeline changed the sidecar: {diffs}"
    assert side_out["annotations"]["charge"]["data"] == {"0": -0.8,
                                                         "1": 0.4}
    assert side_out["engine_offset"] == stated
    assert "cell_origin" not in side_out


def test_identity_and_channels_survive_a_pair_round_trip_through_an_edit(
        tmp_path):
    """The user's 2026-08-20 contract, executed: per-atom information rides
    the atom list and SURVIVES atom edits; system information is stored
    separately.  A pair carrying real identity columns (schema 8), labelled
    species (v16) and a per-atom channel is read back, an atom is deleted
    through the one edit layer (modify.delete_atoms, whose header promises
    per-atom metadata preservation), and the re-written pair still says
    everything -- sliced to the survivors, placeholders nowhere.

    A SPECIES IS A LABEL AND AN ELEMENT (`model/chemistry.md` § 3): the
    geometry file names each atom by its label's name, which every reader of
    the format reads as an element, and the labels come back from the
    sidecar.  A structure holding `Au1` was written with `Au1` in its
    `.xyz`, which its own reader refused.

    KILL SITES, verified red at introduction: the sidecar writer dropping
    the identity spread; the reader skipping the identity apply."""

    from molbuilder.modify import delete_atoms
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    s = Structure(
        elements=["Au1", "S", "C1"],
        positions=[[0, 0, 0], [2.0, 0, 0], [3.5, 0, 0]],
        title="hemeC anchor",
        atom_names=["AU1", "SG", "CA"],
        residue_ids=[1, 14, 14],
        residue_names=["AUX", "CYS", "CYS"],
        chain_ids=["A", "B", "B"],
    )
    s.apply_metadata_dict({
        "regions": {"anchor": [1]},
        "annotations": {"charge": {"kind": "value",
                                   "data": {"1": -0.4, "2": 0.1}}},
    })
    StructureCodec().write(s, tmp_path / "a.xyz")
    side = molstruct.load(tmp_path / "a.molstruct.json")
    assert side["residue_names"] == ["AUX", "CYS", "CYS"]
    # The geometry file, read by its format's own reader, holds elements.
    alone = Structure.from_xyz((tmp_path / "a.xyz").read_text())
    assert alone.elements == ["Au", "S", "C"]

    back = StructureCodec().load(tmp_path / "a.xyz")
    assert back.elements == ["Au1", "S", "C1"]
    # A record about these coordinates names each atom by its element, so
    # it matches the pair and the file read alone (`model/parse.md` § 5b.1).
    assert back.geometry_fingerprint() == alone.geometry_fingerprint()
    assert back.title == "hemeC anchor"
    assert back.atom_names == ["AU1", "SG", "CA"]

    # THE LABELS BELONG TO THEIR ATOMS: the geometry file edited alone --
    # its first atom made silver by a person's tool -- and opened anyway
    # (`GEOMETRY_CHANGED` in ``opened``), `Au1` no longer names that atom:
    # refused, naming it, never put back over silver.
    from molbuilder.sidecars.molstruct import MolstructPairingError
    from molbuilder.workingcopy_structure import GEOMETRY_CHANGED
    edited_alone = Structure.from_xyz((tmp_path / "a.xyz").read_text())
    edited_alone.elements[0] = "Ag"
    (tmp_path / "c.xyz").write_text(edited_alone.to_xyz())
    (tmp_path / "c.molstruct.json").write_bytes(
        (tmp_path / "a.molstruct.json").read_bytes())
    with pytest.raises(MolstructPairingError,
                       match=r"'Au1' for atom 0 names Au.*holds Ag"):
        StructureCodec().load(tmp_path / "c.xyz",
                              opened=(GEOMETRY_CHANGED,), said_out=[])

    # The edit: drop the Au anchor.  Per-atom facts must follow their atoms.
    edited = delete_atoms(back, [0])
    StructureCodec().write(edited, tmp_path / "b.xyz")
    again = StructureCodec().load(tmp_path / "b.xyz")
    assert again.elements == ["S", "C1"]
    assert again.atom_names == ["SG", "CA"]
    assert again.residue_ids == [14, 14]
    assert again.residue_names == ["CYS", "CYS"]
    assert again.chain_ids == ["B", "B"]
    assert again.title == "hemeC anchor"
    ch = again.annotations["charge"]
    assert ch.kind == "value" and ch.data == {0: -0.4, 1: 0.1}, (
        "the channel did not remap with its atoms")


def test_an_xyz_born_pairs_sidecar_carries_no_identity_keys(tmp_path):
    """The other half of the write rule: synthesized placeholders (names =
    elements, resid 1, residue MOL, chain A, empty title) are the server's
    own defaults, not statements -- persisting them would make every
    xyz-born sidecar claim an identity nobody gave it."""
    import json

    from molbuilder.structure import IDENTITY_FIELDS, Structure
    from molbuilder.workingcopy_structure import StructureCodec

    s = Structure(elements=["O"], positions=[[0, 0, 0]])
    s.apply_metadata_dict({"regions": {"top": [0]}})
    StructureCodec().write(s, tmp_path / "p.xyz")
    side = json.loads((tmp_path / "p.molstruct.json").read_text())
    assert not [k for k in IDENTITY_FIELDS if k in side], side.keys()
