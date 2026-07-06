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
                          json={"source": xyz, "data": blob, "overwrite": True}))
    assert r["ok"]
    assert json.loads((proj / "mol.molstruct.json").read_text())["frozen_atoms"] == [1]


def test_update_sourceless_drafts_under_projects_root(client_project, monkeypatch):
    """A brand-new molecule (NO source file) STILL gets a transient draft -- the
    contract persists ANY in-memory data, no project dir required.  Home = the
    top-level projects-root draft dir, keyed by ``workspace_id``."""
    client, proj = client_project
    # projects_root() is the sourceless draft home; isolate it to tmp.
    import molbuilder.web.blueprints.workingcopy as wcbp
    monkeypatch.setattr(wcbp, "projects_root", lambda: proj)
    blob = _json(client.post("/api/workingcopy/open",
                             json={"path": str(proj / "mol.xyz")}))["data"]
    r = client.post("/api/workingcopy/update",
                    json={"workspace_id": "ws-abc123", "data": blob})
    assert _json(r)["ok"]
    drafts = list((proj / ".molbuilder_workspace").glob("new-ws-abc123.*.wc.json"))
    assert len(drafts) == 1                      # the sourceless draft exists


def test_update_sourceless_without_id_is_400(client_project):
    """No source AND no workspace_id -> the server can't place the draft."""
    client, proj = client_project
    blob = _json(client.post("/api/workingcopy/open",
                             json={"path": str(proj / "mol.xyz")}))["data"]
    r = client.post("/api/workingcopy/update", json={"data": blob})
    assert r.status_code == 400
    assert "workspace_id" in _json(r)["error"]


def test_save_refuses_existing_target_without_overwrite(client_project):
    """Overwrite gate (§1): saving onto an existing file without overwrite=true is
    409 -- the caller must confirm first."""
    client, proj = client_project
    xyz = str(proj / "mol.xyz")           # exists (the fixture wrote it)
    blob = _json(client.post("/api/workingcopy/open", json={"path": xyz}))["data"]
    r = client.post("/api/workingcopy/save", json={"source": xyz, "data": blob})
    assert r.status_code == 409
    # with overwrite=true it goes through
    r2 = _json(client.post("/api/workingcopy/save",
                           json={"source": xyz, "data": blob, "overwrite": True}))
    assert r2["ok"]


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


def test_save_writes_full_periodicity_and_hash_tie(client_project):
    """A1 (molview-migration-plan): the working-copy save writes the WHOLE dataset
    atomically -- .xyz + .json TOGETHER, the .json carrying the full periodicity
    (cell/axis_kind/kgrid) and structure_hash = sha256 of the just-written .xyz."""
    import hashlib
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    blob = _json(client.post("/api/workingcopy/open", json={"path": xyz}))["data"]
    # The store would carry periodicity in the sidecar blob:
    blob["sidecar"]["cell"] = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 10.0]]
    blob["sidecar"]["axis_kind"] = ["periodic", "periodic", "transport"]
    blob["sidecar"]["kgrid"] = [4, 4, 1]
    r = _json(client.post("/api/workingcopy/save",
                          json={"source": xyz, "data": blob, "overwrite": True}))
    assert r["ok"]
    # BOTH files exist (the atomic pair).
    assert (proj / "mol.xyz").exists()
    assert (proj / "mol.molstruct.json").exists()
    sidecar = json.loads((proj / "mol.molstruct.json").read_text())
    assert sidecar["cell"] == [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 10.0]]
    assert sidecar["axis_kind"] == ["periodic", "periodic", "transport"]
    assert sidecar["kgrid"] == [4, 4, 1]
    # Hash-tie: the .json's structure_hash matches the just-written .xyz.
    xyz_bytes = (proj / "mol.xyz").read_bytes()
    assert sidecar["structure_hash"] == hashlib.sha256(xyz_bytes).hexdigest()


def test_orphans_and_clean(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    client.post("/api/workingcopy/update", json={"source": xyz, "data": r["data"]})
    # Same live session -> its own draft is not an orphan.
    assert _json(client.post("/api/workingcopy/orphans", json={"path": xyz}))["orphans"] == []
    assert _json(client.post("/api/workingcopy/clean", json={"path": xyz}))["removed"] == 1
