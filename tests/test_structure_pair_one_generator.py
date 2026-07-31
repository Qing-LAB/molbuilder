"""One generator: the bytes a save writes and the bytes a download gets.

`StructureCodec.pair` is the single place a Structure becomes the two things
that represent it outside memory — the coordinate document and the sidecar
payload. `write` puts that on disk, `files` hands it over as bytes, and `scratch_blob`
hands it over as a round trip. The door that RETURNS it for a download is not
here yet: a new door should be born speaking the one envelope
(`web-api.md` § 1), and that envelope does not exist server-side until the phase
after this one.

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
    # heals it. Both the save route and the export route rebuild through
    # `from_scratch`, so both heal the same way — but a fixture that skips it is
    # a structure the application could not have produced, and comparing it
    # against a healed one measures the gate rather than the generator.
    return StructureCodec().from_scratch(StructureCodec().scratch_blob(s))


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


def test_the_pair_round_trips_through_the_blob():
    """`scratch_blob` and `from_scratch` are inverses, and the pair is what sits
    between them — so a structure that goes out and comes back is the same
    structure, labels and cell included."""
    struct = _with_metadata()
    back = StructureCodec().from_scratch(StructureCodec().scratch_blob(struct))

    assert back.regions == struct.regions
    assert list(back.frozen_atoms) == list(struct.frozen_atoms)
    assert back.cell is not None
    assert StructureCodec().pair(back).document == StructureCodec().pair(struct).document
