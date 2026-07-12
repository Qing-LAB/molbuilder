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


def test_open_returns_atoms_and_periodicity_for_the_store(client_project):
    """A6: /api/workingcopy/open returns the per-atom rows (WITH the sidecar's
    regions + frozen) + the full periodicity in ONE call, so the Modify load takes
    its DATA from the framework, not /api/selection/atoms."""
    from molbuilder.sidecars import molstruct as _msj
    client, proj = client_project
    xyz = proj / "mol.xyz"
    _msj.save(_msj.sidecar_path_for(xyz), _msj.to_dict(
        n_atoms_total=5, structure_hash=_msj.sha256_of_file(xyz),
        regions={"L-electrode": [0, 1]}, frozen_atoms=[0],
        cell=[[40, 0, 0], [0, 40, 0], [0, 0, 40]],
        axis_kind=["periodic", "periodic", "periodic"], kgrid=[2, 2, 2]))
    r = _json(client.post("/api/workingcopy/open", json={"path": str(xyz)}))
    assert r["ok"]
    assert len(r["atoms"]) == 5
    assert "L-electrode" in r["atoms"][0]["regions"]      # sidecar region on the atom
    assert r["atoms"][0]["is_frozen"] is True             # sidecar frozen on the atom
    assert r["periodicity"]["kgrid"] == [2, 2, 2]         # periodicity carried
    assert r["periodicity"]["axis_kind"] == ["periodic", "periodic", "periodic"]


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


def test_save_as_drops_the_source_keyed_draft(client_project):
    """Review b1: a save-as writes to a NEW name but MUST drop the draft keyed by the
    ORIGINAL source -- else the draft leaks as a permanent orphan and a later recover
    surfaces phantom 'unsaved edits' for a file the user saved."""
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    blob = _json(client.post("/api/workingcopy/open", json={"path": xyz}))["data"]
    client.post("/api/workingcopy/update", json={"source": xyz, "data": blob})
    assert len(list((proj / ".molbuilder_workspace").glob("mol.*.wc.json"))) == 1
    r = _json(client.post("/api/workingcopy/save",
                          json={"source": xyz, "target": str(proj / "final.xyz"),
                                "data": blob}))
    assert r["ok"] and (proj / "final.xyz").exists()
    # the source-keyed draft is dropped, not leaked
    assert list((proj / ".molbuilder_workspace").glob("mol.*.wc.json")) == []


def test_sourceless_save_drops_the_workspace_id_draft(client_project, monkeypatch):
    """Review b1: a new molecule's draft (new-<id>) must be dropped when it's saved
    to a real path -- else every new-molecule session leaves a permanent orphan."""
    import molbuilder.web.blueprints.workingcopy as wcbp
    client, proj = client_project
    monkeypatch.setattr(wcbp, "projects_root", lambda: proj)
    blob = _json(client.post("/api/workingcopy/open",
                             json={"path": str(proj / "mol.xyz")}))["data"]
    client.post("/api/workingcopy/update", json={"workspace_id": "ws-xyz", "data": blob})
    assert list((proj / ".molbuilder_workspace").glob("new-ws-xyz.*.wc.json"))
    r = _json(client.post("/api/workingcopy/save",
                          json={"workspace_id": "ws-xyz", "target": str(proj / "new.xyz"),
                                "data": blob}))
    assert r["ok"] and (proj / "new.xyz").exists()
    assert list((proj / ".molbuilder_workspace").glob("new-ws-xyz.*.wc.json")) == []


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


def test_state_timeline_roundtrip_and_prune(client_project, monkeypatch):
    """§4.7: write-state stores an OPAQUE session snapshot format-blind under
    <workspace_id>.<state_index>.wc.json; read-state round-trips the exact JSON
    (NOT through the {xyz,sidecar} codec); prune tail-deletes above an index and
    -1 clears the whole timeline."""
    import molbuilder.web.blueprints.workingcopy as wcbp
    client, proj = client_project
    monkeypatch.setattr(wcbp, "projects_root", lambda: proj)
    ws = "ws-abc123"
    # A snapshot the structure codec would REJECT (no xyz) -> proves format-blind.
    snap = {"state": {"frames": [1, 2], "selection": [3]}, "meta": {"tag": "opaque"}}
    for i in (0, 1, 2):
        s = dict(snap, idx=i)
        r = _json(client.post("/api/workingcopy/write-state",
                              json={"workspace_id": ws, "state_index": i, "data": s}))
        assert r["ok"]
    # round-trip the exact opaque bytes at a chosen index
    r = _json(client.post("/api/workingcopy/read-state",
                          json={"workspace_id": ws, "state_index": 1}))
    assert r["data"] == dict(snap, idx=1)
    # missing index -> 404 + null
    r = client.post("/api/workingcopy/read-state",
                    json={"workspace_id": ws, "state_index": 9})
    assert r.status_code == 404 and _json(r)["data"] is None
    # prune tail: drop everything above index 1
    r = _json(client.post("/api/workingcopy/prune-states",
                          json={"workspace_id": ws, "above_index": 1}))
    assert r["removed"] == 1
    d = proj / ".molbuilder_workspace"
    assert sorted(p.name for p in d.glob(f"{ws}.*.wc.json")) == [
        f"{ws}.0.wc.json", f"{ws}.1.wc.json"]
    # above_index = -1 clears the whole timeline
    r = _json(client.post("/api/workingcopy/prune-states",
                          json={"workspace_id": ws, "above_index": -1}))
    assert r["removed"] == 2
    assert list(d.glob(f"{ws}.*.wc.json")) == []


def test_state_write_keeps_rolling_window(client_project, monkeypatch):
    """§4.7: each write keeps only the most-recent 30 indices; older ones drop."""
    import molbuilder.web.blueprints.workingcopy as wcbp
    client, proj = client_project
    monkeypatch.setattr(wcbp, "projects_root", lambda: proj)
    ws = "ws-window"
    for i in range(35):
        client.post("/api/workingcopy/write-state",
                    json={"workspace_id": ws, "state_index": i, "data": {"i": i}})
    d = proj / ".molbuilder_workspace"
    kept = sorted(int(p.name[len(ws) + 1:-len(".wc.json")])
                  for p in d.glob(f"{ws}.*.wc.json"))
    assert kept == list(range(5, 35))            # oldest 5 pruned, 30 kept


def test_state_rejects_bad_identity(client_project, monkeypatch):
    """Path-traversal / bad index are refused with 400 (no file written)."""
    import molbuilder.web.blueprints.workingcopy as wcbp
    client, proj = client_project
    monkeypatch.setattr(wcbp, "projects_root", lambda: proj)
    assert client.post("/api/workingcopy/write-state",
                       json={"workspace_id": "../evil", "state_index": 0,
                             "data": {}}).status_code == 400
    assert client.post("/api/workingcopy/write-state",
                       json={"workspace_id": "ok", "state_index": -1,
                             "data": {}}).status_code == 400
    assert client.post("/api/workingcopy/write-state",
                       json={"workspace_id": "ok", "state_index": "x",
                             "data": {}}).status_code == 400


def test_orphans_and_clean(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    client.post("/api/workingcopy/update", json={"source": xyz, "data": r["data"]})
    # Same live session -> its own draft is not an orphan.
    assert _json(client.post("/api/workingcopy/orphans", json={"path": xyz}))["orphans"] == []
    assert _json(client.post("/api/workingcopy/clean", json={"path": xyz}))["removed"] == 1
