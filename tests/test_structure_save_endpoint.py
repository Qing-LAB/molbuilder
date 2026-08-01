"""Round-trip contract for ``/api/structure/save`` -- the FILE-ONLY save door.

REGRESSION (task #75, the save->reload breaker): the browser used to write the
``.molstruct.json`` sidecar itself -- a blob with NO ``schema_version`` and an EMPTY
``structure_hash`` -- which the strict file-only load door (``StructureCodec.read`` ->
``molstruct.load_text``) then REJECTED, so every save produced a pair the app could no
longer open.  The fix routes the save through the SERVER, which reconstructs the Structure
from the wire envelope (``_shared.struct_from_body``) and writes the pair via
``StructureCodec.write`` -- Python owns the pairing AND stamps the sidecar schema.

These tests pin:
  1. a BROWSER-shaped payload (no schema_version, empty hash) saved through the
     endpoint lands on disk as a VALID pair the load door reads back without error, with
     the metadata (frozen / regions / off-origin cell) preserved;
  2. the overwrite gate (409 -> needsOverwrite);
  3. path-traversal + bad-input guards return 400, never 500.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("flask")


@pytest.fixture
def web(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    from molbuilder.web.app import create_app

    # Point the picker roots at the tmp dir so a save-as target resolves within a root.
    caps = diagnostics.get_capabilities()
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    client = create_app(config={}).test_client()
    client._root = tmp_path
    return client


def _browser_blob():
    """Exactly what MolView's ``exportFile`` hands over (molview.md § 11.7): the
    STRUCTURE, not bytes. The browser assembles no coordinate document -- the
    server writes both files from this, through the one paired-file writer."""
    return {
        "elements":  ["C", "S"],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "metadata": {
            "regions":     {"anchor": [0], "frozen_atoms": [1]},
            "cell":        [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            "cell_origin": [-3.0, -4.0, -5.0],
        },
    }



def test_overwrite_gate_drives_the_dialog(web):
    blob = _browser_blob()
    path = str(web._root / "proj" / "x.xyz")
    assert web.post("/api/structure/save", json={"path": path, "structure": blob}).get_json()["ok"]

    # A second save WITHOUT overwrite -> 409 + needsOverwrite (the tab's confirm dialog).
    r2 = web.post("/api/structure/save", json={"path": path, "structure": blob})
    assert r2.status_code == 409
    assert r2.get_json().get("needsOverwrite") is True

    # WITH overwrite -> ok.
    r3 = web.post("/api/structure/save",
                  json={"path": path, "structure": blob, "overwrite": True})
    assert r3.status_code == 200 and r3.get_json()["ok"] is True


def test_bad_inputs_are_400_not_500(web):
    # missing path / missing blob.
    assert web.post("/api/structure/save",
                    json={"structure": _browser_blob()}).status_code == 400
    assert web.post("/api/structure/save",
                    json={"path": "proj/y.xyz"}).status_code == 400
    # path traversal is refused by _resolve_within_roots (400, not a write).
    rt = web.post("/api/structure/save",
                  json={"path": "../secret.xyz", "structure": _browser_blob()})
    assert rt.status_code == 400


# --------------------------------------------------------------------- #
#  A range of frames -- one read, both destinations                      #
# --------------------------------------------------------------------- #

_ENV = {
    "elements": ["Au", "Au"],
    "positions": [[0, 0, 0], [2, 2, 0]],
    "metadata": {"regions": {"bridge": [0]},
                 "cell": [[8, 0, 0], [0, 8, 0], [0, 0, 8]],
                 "cell_origin": None, "axis_kind": None, "vacuum": None},
}
_FRAMES = [[[0, 0, 0], [2, 2, 0]],
           [[1, 0, 0], [3, 2, 0]],
           [[2, 0, 0], [4, 2, 0]]]


def test_saving_a_range_and_downloading_it_produce_identical_bytes(web):
    """molview.md § 11.3: "Save-to-project and Download produce identical bytes."

    That is a claim about CONSTRUCTION, not a promise two code paths keep: both
    go through ``StructureCodec.pair``, so the only way they could differ is if
    one of them stopped doing so. This checks the bytes rather than the wiring.

    § 11.3 also fixes what a range produces: one extended-XYZ document with a
    block per frame, and **one** sidecar -- the labels and the cell are the
    structure's shared identity, so writing them per frame would be N chances to
    disagree.
    """
    path = str(web._root / "run.extxyz")
    downloaded = web.post("/api/structure/export",
                          json={"structure": _ENV, "frames": _FRAMES,
                                "name": "run"}).get_json()
    saved = web.post("/api/structure/save",
                     json={"structure": _ENV, "frames": _FRAMES,
                           "path": path}).get_json()
    assert saved["ok"] is True, saved

    on_disk = (web._root / "run.extxyz").read_text(encoding="utf-8")
    handed = {f["name"]: f["text"] for f in downloaded["files"]}
    assert on_disk == handed["run.extxyz"], (
        "the project save and the download produced different bytes")
    assert on_disk.count("Lattice=") == 3, "one block per frame, each with the cell"
    assert downloaded["frames"] == 3

    written = sorted(p.name for p in web._root.iterdir())
    assert written == ["run.extxyz", "run.molstruct.json"], (
        f"a range must write ONE sidecar beside the one document: {written}")
    # The NAMES agree too, not just the bytes -- given the same stem, the two
    # destinations produce the same pair of filenames.
    assert sorted(handed) == written, (
        f"the download and the save named the pair differently: "
        f"{sorted(handed)} vs {written}")
    assert json.loads((web._root / "run.molstruct.json").read_text()
                      )["regions"] == {"bridge": [0]}


def test_a_frame_that_does_not_carry_these_atoms_is_refused_at_both_doors(web):
    """The same-atoms rule reaches the wire: a frame of the wrong length is a
    400 at each door rather than a half-written file or a torn document."""
    for route, extra in (("/api/structure/export", {}),
                         ("/api/structure/save",
                          {"path": str(web._root / "x.extxyz")})):
        r = web.post(route, json={"structure": _ENV,
                                  "frames": [[[0, 0, 0]]], **extra})
        assert r.status_code == 400, (route, r.get_json())
        assert "atoms" in r.get_json()["error"]
    assert not list(web._root.iterdir()), "a refused save left a file behind"


def test_one_frame_is_the_request_it_always_was(web):
    """`frames` is ADDITIVE. Omitted, the door writes the plain `.xyz` it always
    wrote -- so a caller that knows nothing about ranges keeps working."""
    path = str(web._root / "one.xyz")
    assert web.post("/api/structure/save",
                    json={"structure": _ENV, "path": path}).get_json()["ok"]
    text = (web._root / "one.xyz").read_text(encoding="utf-8")
    assert text.count("Lattice=") == 0, "a single frame stays a plain .xyz"
    assert text.splitlines()[0].strip() == "2"
