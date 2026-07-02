"""/api/workingcopy/* — open, update (draft), save (overwrite / save-as)."""
import json

import numpy as np
import pytest

pytest.importorskip("flask")

from molbuilder import diagnostics
from molbuilder.structure import Structure


@pytest.fixture
def client_project(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.Capabilities, "file_picker_roots",
                        lambda self: ((tmp_path, "projects"),))
    s = Structure(elements=["H", "C", "N", "O", "F"],
                  positions=np.array([[float(i), 0.0, 0.0] for i in range(5)]),
                  cell=np.diag([40.0, 40.0, 40.0]), pbc=[True, True, True])
    (tmp_path / "mol.xyz").write_text(s.to_xyz())
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client(), tmp_path


def _json(r):
    return json.loads(r.data)


def test_open_update_save(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    assert r["ok"] and r["data"]["xyz"]
    blob = r["data"]
    blob["sidecar"]["frozen_atoms"] = [1]
    assert _json(client.post("/api/workingcopy/update",
                             json={"source": xyz, "data": blob}))["ok"]
    assert (proj / ".molbuilder_workspace").exists()
    assert not (proj / "mol.molstruct.json").exists()   # not saved yet
    r = _json(client.post("/api/workingcopy/save",
                          json={"source": xyz, "data": blob}))
    assert r["ok"]
    assert json.loads((proj / "mol.molstruct.json").read_text())["frozen_atoms"] == [1]


def test_save_as_new_path(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    blob = r["data"]
    blob["sidecar"]["regions"] = {"L-electrode": [0]}
    r = _json(client.post("/api/workingcopy/save",
                          json={"source": xyz, "target": str(proj / "copy.xyz"),
                                "data": blob}))
    assert r["ok"] and r["saved"].endswith("copy.xyz")
    assert (proj / "copy.xyz").exists()
    assert json.loads((proj / "copy.molstruct.json").read_text())["regions"] == {"L-electrode": [0]}


def test_orphans_and_clean(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    client.post("/api/workingcopy/update", json={"source": xyz, "data": r["data"]})
    # Same live session -> its own draft is not an orphan.
    assert _json(client.post("/api/workingcopy/orphans", json={"path": xyz}))["orphans"] == []
    assert _json(client.post("/api/workingcopy/clean", json={"path": xyz}))["removed"] == 1
