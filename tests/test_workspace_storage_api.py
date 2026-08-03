"""/api/workspace-storage/* — the workspace's byte storage (blueprints/workspace_storage.py).

Format-blind indexed session snapshots (workspace-contract §4.7): ``state/write``
stores an OPAQUE blob under ``<workspace_id>.<state_index>.wc.json``; ``state/read``
round-trips the exact JSON (NOT through the {xyz,sidecar} codec); ``state/prune``
tail-deletes above an index (-1 clears the whole timeline).  Extracted from the
retired ``/api/workingcopy/*`` door.
"""
import json

import pytest

pytest.importorskip("flask")

from molbuilder import diagnostics


@pytest.fixture
def client_project(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.Capabilities, "file_picker_roots",
                        lambda self: ((tmp_path, "projects"),))
    # The state routes file under projects_root(); isolate it to tmp.
    import molbuilder.web.blueprints.workspace_storage as wsbp
    monkeypatch.setattr(wsbp, "projects_root", lambda: tmp_path)
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client(), tmp_path


def _json(r):
    return json.loads(r.data)


def test_workspace_storage_roundtrip_and_prune(client_project):
    """§4.7: state/write stores an OPAQUE session snapshot format-blind under
    <workspace_id>.<state_index>.wc.json; state/read round-trips the exact JSON
    (NOT through the {xyz,sidecar} codec); state/prune tail-deletes above an index
    and -1 clears the whole timeline."""
    client, proj = client_project
    ws = "ws-abc123"
    # A snapshot the structure codec would REJECT (no xyz) -> proves format-blind.
    snap = {"state": {"frames": [1, 2], "selection": [3]}, "meta": {"tag": "opaque"}}
    for i in (0, 1, 2):
        s = dict(snap, idx=i)
        r = _json(client.post("/api/workspace-storage/write",
                              json={"workspace_id": ws, "state_index": i, "data": s}))
        assert r["ok"]
    # round-trip the exact opaque bytes at a chosen index
    r = _json(client.post("/api/workspace-storage/read",
                          json={"workspace_id": ws, "state_index": 1}))
    assert r["data"] == dict(snap, idx=1)
    # missing index -> 404 + null
    r = client.post("/api/workspace-storage/read",
                    json={"workspace_id": ws, "state_index": 9})
    assert r.status_code == 404 and _json(r)["data"] is None
    # prune tail: drop everything above index 1
    r = _json(client.post("/api/workspace-storage/prune",
                          json={"workspace_id": ws, "above_index": 1}))
    assert r["removed"] == 1
    # State-timeline files live in a ``states/`` SUBDIR (isolated from the sourceless
    # drafts so the draft ``*.wc.json`` cleanup can't delete live undo timelines).
    d = proj / ".molbuilder_workspace" / "states"
    assert sorted(p.name for p in d.glob(f"{ws}.*.wc.json")) == [
        f"{ws}.0.wc.json", f"{ws}.1.wc.json"]
    # above_index = -1 clears the whole timeline
    r = _json(client.post("/api/workspace-storage/prune",
                          json={"workspace_id": ws, "above_index": -1}))
    assert r["removed"] == 2
    assert list(d.glob(f"{ws}.*.wc.json")) == []


def test_state_write_keeps_rolling_window(client_project):
    """§4.7: each write keeps only the most-recent 30 indices; older ones drop."""
    client, proj = client_project
    ws = "ws-window"
    for i in range(35):
        client.post("/api/workspace-storage/write",
                    json={"workspace_id": ws, "state_index": i, "data": {"i": i}})
    d = proj / ".molbuilder_workspace" / "states"
    kept = sorted(int(p.name[len(ws) + 1:-len(".wc.json")])
                  for p in d.glob(f"{ws}.*.wc.json"))
    assert kept == list(range(5, 35))            # oldest 5 pruned, 30 kept


def test_state_rejects_bad_identity(client_project):
    """Path-traversal / bad index are refused with 400 (no file written)."""
    client, proj = client_project
    assert client.post("/api/workspace-storage/write",
                       json={"workspace_id": "../evil", "state_index": 0,
                             "data": {}}).status_code == 400
    assert client.post("/api/workspace-storage/write",
                       json={"workspace_id": "ok", "state_index": -1,
                             "data": {}}).status_code == 400
    assert client.post("/api/workspace-storage/write",
                       json={"workspace_id": "ok", "state_index": "x",
                             "data": {}}).status_code == 400
