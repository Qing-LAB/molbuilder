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


def _with_metadata() -> Structure:
    """Metadata applied the way it actually arrives — through the codec's own
    door, which coerces and validates. Assigning the fields directly bypasses
    that and builds a Structure the rest of the system could not have made."""
    s = Structure.from_xyz("3\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    molstruct.apply_to_structure(s, {
        "n_atoms_total": 3,
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


def test_a_non_ascii_label_survives_both_paths_identically(tmp_path):
    """The concrete drift this closes: `files()` used `json.dumps` without
    `ensure_ascii=False` while `save()` had it, so "α-helix" was `\\u03b1-helix`
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


def test_a_plain_structure_gets_no_sidecar_from_either(tmp_path):
    """"No `.json` means no metadata" is how the load door reads a pair, so a
    structure with nothing worth keeping must produce no sidecar — from the
    writer AND from anything that hands the pair over.
    """
    target = tmp_path / "plain.xyz"
    StructureCodec().write(_plain(), target)

    assert not molstruct.sidecar_path_for(target).exists()
    handed = [p for p, _ in StructureCodec().files(_plain(), target)]
    assert handed == [target], f"a sidecar was handed over for a plain structure: {handed}"

    made = StructureCodec().pair(_plain())
    assert made.keep_sidecar is False
    assert made.sidecar["schema_version"], (
        "the payload is still built — a blob is a round trip, not a file, and "
        "its reader expects one"
    )


def test_a_stale_sidecar_is_removed_rather_than_left_disagreeing(tmp_path):
    """A structure that loses its metadata must lose its sidecar, or the pair on
    disk says two different things about the same atoms.
    """
    target = tmp_path / "was-labelled.xyz"
    StructureCodec().write(_with_metadata(), target)
    assert molstruct.sidecar_path_for(target).exists()

    StructureCodec().write(_plain(), target)
    assert not molstruct.sidecar_path_for(target).exists(), (
        "the old sidecar outlived the metadata it described"
    )


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
    many = [p.name for p, _ in StructureCodec().files(struct, "wire",
                                                      frames=frames)]

    assert one == ["wire.xyz", "wire.molstruct.json"], one
    assert many == ["wire.xyz", "wire.molstruct.json"], (
        f"a range must be named under the extension that reads it back: {many}")


def test_a_saved_range_can_be_opened_again(tmp_path):
    """The pin that was missing, which is why the naming defect survived: the
    save test wrote a trajectory and never read its file back.

    A project save is the scientific record (molview.md § 11.3). A record that
    cannot be reopened is not one.
    """
    struct = _with_metadata()
    frames = [struct.positions, struct.positions + 0.1, struct.positions + 0.2]
    target = tmp_path / "run.xyz"
    written = StructureCodec().write(struct, target, frames=frames)

    got = []
    back = StructureCodec().read(written, frames_out=got)
    assert len(got) == 3, f"the frames did not survive the round trip: {len(got)}"
    assert back.regions == struct.regions, "the labels did not come back"
    assert back.cell is not None, "the cell did not come back"
    assert got[2] == pytest.approx(frames[2])


def test_a_stem_that_already_carries_a_suffix_is_corrected_not_appended_to():
    """Both callers exist: the export door hands a bare stem, and a comparison
    against `write()` hands a full path. Neither may end up with two suffixes,
    and a stem carrying a suffix that is NOT a geometry one keeps it -- `run.v2`
    is a name, not a mistake, and `with_suffix` would eat the `.v2`.
    """
    struct = _with_metadata()
    frames = [struct.positions, struct.positions + 0.1]
    name = lambda target, **kw: StructureCodec().files(  # noqa: E731
        struct, target, **kw)[0][0].name

    assert name("wire.xyz") == "wire.xyz"
    assert name("wire.xyz", frames=frames) == "wire.xyz"
    assert name("wire.extxyz") == "wire.xyz", (
        "a non-standard extension is corrected to the one load() accepts")
    assert name("run.v2") == "run.v2.xyz", "the .v2 was eaten"
    assert name("run.v2", frames=frames) == "run.v2.xyz"


def test_the_suffix_travels_with_the_pair_rather_than_being_re_derived():
    """`pair()` decides the format; carrying the suffix beside the document is
    what stops anything downstream deciding it a second time and disagreeing."""
    struct = _with_metadata()
    assert StructureCodec().pair(struct).suffix == ".xyz"
    assert StructureCodec().pair(
        struct, frames=[struct.positions, struct.positions]).suffix == ".xyz", (
        "extended XYZ is a superset of plain XYZ and shares its extension")


def test_every_metadata_field_survives_the_whole_export_pipeline(tmp_path,
                                                                 monkeypatch):
    """The user's 2026-08-20 question ("is the pbc preserved?"), answered by
    execution for EVERY field: a sidecar with mixed axis kinds (periodic,
    periodic, isolated — so ``pbc`` is [T, T, F]), regions, an off-origin
    cell and a vacuum goes disk → ``/api/build/load`` wire → the exact
    envelope the browser's ``structureForServer`` builds from that wire →
    ``/api/structure/export`` → sidecar again — and the two sidecars are
    EQUAL apart from ``created_at`` (provenance) and the hash/version
    envelope.  ``pbc`` itself never rides the wire: it is derived from
    ``axis_kind`` by the one deserialiser, which is why preserving the
    axis kinds preserves it.

    KILL SITE, verified red: ``structure.py``'s ``apply_metadata_dict``
    dropping ``axis_kind`` fails this test.  (The commit that introduced
    the test named ``_stated_periodicity`` as the checked mutation — that
    reader serves the modify-wire path, not this envelope, and mutating it
    leaves this green; the record is corrected here.)
    """
    import json

    from molbuilder import diagnostics
    from molbuilder.structure import Structure
    from molbuilder.web.app import create_app

    # The picker-roots seam, the way every route test registers one.
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)

    struct = Structure.from_dict({
        "elements": ["O", "H", "H"],
        "positions": [[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
        "metadata": {
            "regions": {"frozen": [0], "top": [1]},
            "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 30]],
            "cell_origin": [0.5, 0.5, 0.5],
            "axis_kind": ["periodic", "periodic", "isolated"],
            "vacuum": [0, 0, 8],
        },
    })
    StructureCodec().write(struct, tmp_path / "slab.xyz")
    side_in = json.loads((tmp_path / "slab.molstruct.json").read_text())
    assert side_in["pbc"] == [True, True, False], "precondition: mixed axes"

    client = create_app(config={}).test_client()
    wire = client.post("/api/build/load",
                       json={"path": str(tmp_path / "slab.xyz")}).get_json()
    assert wire.get("ok"), wire
    per = wire["periodicity"]
    assert per["axis_kind"] == ["periodic", "periodic", "isolated"], (
        "the wire lost the axis kinds — everything downstream would too"
    )

    # The browser's envelope, exactly as structureForServer composes it
    # (model-jobs.js): regions grouped from the per-atom rows, the
    # periodicity fields copied by the server's own names.
    atoms = wire["atoms"]
    regions = {}
    for a in atoms:
        for name in a.get("regions", []):
            regions.setdefault(name, []).append(a["index"])
    out = client.post("/api/structure/export", json={
        "structure": {
            "elements": [a["element"] for a in atoms],
            "positions": [[a["x"], a["y"], a["z"]] for a in atoms],
            "metadata": {
                "regions": regions,
                "cell": per.get("cell"),
                "cell_origin": per.get("cell_origin"),
                "axis_kind": per.get("axis_kind"),
                "vacuum": per.get("vacuum"),
            },
        },
        "name": "slab",
    }).get_json()
    assert out["ok"], out
    side_out = json.loads(next(f["text"] for f in out["files"]
                               if f["name"].endswith(".json")))

    volatile = {"created_at", "structure_hash", "schema_version"}
    diffs = {k: (side_in.get(k), side_out.get(k))
             for k in set(side_in) | set(side_out)
             if k not in volatile and side_in.get(k) != side_out.get(k)}
    assert diffs == {}, (
        f"metadata changed crossing the pipeline: {diffs}"
    )
    assert side_out["pbc"] == [True, True, False]
