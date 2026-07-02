"""/api/workingcopy/* surface (Phase 2b) — open -> update -> commit end to end
through the Flask app, plus the changed-underneath REFUSE."""
import hashlib
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


def _json(resp):
    return json.loads(resp.data)


def test_open_update_commit_roundtrip(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")

    # open
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    assert r["ok"] and r["data"]["xyz"]
    blob, source_hash = r["data"], r["source_hash"]

    # edit: freeze atom 1 + a region (via the sidecar half of the blob)
    blob["sidecar"]["frozen_atoms"] = [1]
    blob["sidecar"]["regions"] = {"bridge": [2, 3]}

    # update (writes server scratch)
    assert _json(client.post("/api/workingcopy/update",
                             json={"source": xyz, "source_hash": source_hash,
                                   "data": blob}))["ok"]
    assert (proj / ".molbuilder_workspace").exists()
    assert not (proj / "mol.molstruct.json").exists()      # not durable yet

    # commit
    r = _json(client.post("/api/workingcopy/commit",
                          json={"source": xyz, "source_hash": source_hash,
                                "data": blob}))
    assert r["ok"], r
    sidecar = json.loads((proj / "mol.molstruct.json").read_text())
    assert sidecar["frozen_atoms"] == [1]
    assert sidecar["regions"] == {"bridge": [2, 3]}
    # atom-identity invariant: sidecar hash == sha256(the committed .xyz)
    assert sidecar["structure_hash"] == hashlib.sha256(
        (proj / "mol.xyz").read_bytes()).hexdigest()


def test_commit_refuses_when_source_changed(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    blob, source_hash = r["data"], r["source_hash"]
    blob["sidecar"]["frozen_atoms"] = [0]

    # someone edits mol.xyz on disk after we loaded it
    (proj / "mol.xyz").write_text("1\nchanged\nH 9.0 9.0 9.0\n")

    r = _json(client.post("/api/workingcopy/commit",
                          json={"source": xyz, "source_hash": source_hash,
                                "data": blob}))
    assert not r["ok"] and r["reason"] == "mismatch"
    assert not (proj / "mol.molstruct.json").exists()      # no laundering


def test_orphans_and_clean(client_project):
    client, proj = client_project
    xyz = str(proj / "mol.xyz")
    r = _json(client.post("/api/workingcopy/open", json={"path": xyz}))
    client.post("/api/workingcopy/update",
                json={"source": xyz, "source_hash": r["source_hash"],
                      "data": r["data"]})
    # Same live session -> its own scratch is NOT an orphan.
    assert _json(client.post("/api/workingcopy/orphans",
                             json={"path": xyz}))["orphans"] == []
    # clean wipes it.
    assert _json(client.post("/api/workingcopy/clean",
                             json={"path": xyz}))["removed"] == 1
