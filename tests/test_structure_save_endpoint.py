"""Round-trip contract for ``/api/structure/save`` -- the FILE-ONLY save door.

REGRESSION (task #75, the save->reload breaker): the browser used to write the
``.molstruct.json`` sidecar itself -- a blob with NO ``schema_version`` and an EMPTY
``structure_hash`` -- which the strict file-only load door (``StructureCodec.read`` ->
``molstruct.load_text``) then REJECTED, so every save produced a pair the app could no
longer open.  The fix routes the save through the SERVER, which reconstructs the Structure
(``StructureCodec.from_scratch``) and writes the pair via ``StructureCodec.write`` -- Python
owns the pairing AND stamps the sidecar schema.

These tests pin:
  1. a BROWSER-shaped scratch blob (no schema_version, empty hash) saved through the
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
