"""HTTP-route tests for the checkpoint blueprint (PR-B Phase 1).

Real Flask test client + real filesystem (tmp_path) + real git
subprocess.  Mirrors the no-mocks discipline of
test_checkpoint_lifecycle.py and test_checkpoint_manifest_format.py.

Envelope contract: web-api.md § 1.6 four-bucket rule.
Behaviour contract: docs/protocols/run-checkpoints.md § 8.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from molbuilder.web.app import create_app


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"],
                       capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _have_git(),
    reason="git not on PATH; checkpoint-route tests need git ≥ 2.20",
)


@pytest.fixture
def client():
    return create_app(config={}).test_client()


def _seed(tmp_path: Path) -> Path:
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x00" * 2048)
    return tmp_path


# ----------------------------------------------------------------- #
#  GET /api/checkpoint/state                                         #
# ----------------------------------------------------------------- #


def test_state_uninitialised_dir_reports_initialized_false(client, tmp_path):
    _seed(tmp_path)
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["state"]["initialized"] is False
    assert body["state"]["head"] is None


def test_state_after_init_reports_head_and_clean(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["state"]["initialized"] is True
    assert body["state"]["head"]
    assert body["state"]["dirty"] is False
    assert body["state"]["archive_total_bytes"] == 2048


def test_state_missing_path_param_is_400(client):
    r = client.get("/api/checkpoint/state")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_state_nonexistent_path_is_400(client, tmp_path):
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


# ----------------------------------------------------------------- #
#  GET /api/checkpoint/list                                          #
# ----------------------------------------------------------------- #


def test_list_uninitialised_dir_returns_empty_with_flag(client, tmp_path):
    _seed(tmp_path)
    r = client.get("/api/checkpoint/list",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["initialized"] is False
    assert body["checkpoints"] == []


def test_list_after_init_and_tag(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first save point")
    r = client.get("/api/checkpoint/list",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["initialized"] is True
    assert len(body["checkpoints"]) == 1
    cp = body["checkpoints"][0]
    assert any("baseline" in r for r in cp["refs"])
    assert cp["has_archive"] is True
    assert cp["archive_bytes"] == 2048


def test_list_invalid_limit_is_400(client, tmp_path):
    _seed(tmp_path)
    r = client.get("/api/checkpoint/list",
                   query_string={"path": str(tmp_path),
                                 "limit": "garbage"})
    assert r.status_code == 400


# ----------------------------------------------------------------- #
#  POST /api/checkpoint/init                                         #
# ----------------------------------------------------------------- #


def test_init_creates_repo_and_archives_binaries(client, tmp_path):
    _seed(tmp_path)
    r = client.post("/api/checkpoint/init",
                    json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["state"]["initialized"] is True
    assert body["state"]["archive_total_bytes"] == 2048


def test_init_idempotent_returns_already_initialised(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/init",
                    json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["note"] == "already initialised; no-op"


def test_init_refuses_nested_working_dirs_as_advisory(client, tmp_path):
    """P5: nested working dirs → bucket B advisory (HTTP 200 + ok:false +
    errors.message names the nested paths)."""
    _seed(tmp_path)
    nested = tmp_path / "child"
    nested.mkdir()
    (nested / "inner.fdf").write_text("SystemLabel inner\n")
    r = client.post("/api/checkpoint/init",
                    json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert body["errors_only"] is True
    msg = body["errors"][0]["message"]
    assert "nested working dirs" in msg


# ----------------------------------------------------------------- #
#  POST /api/checkpoint/commit                                       #
# ----------------------------------------------------------------- #


def test_commit_clean_tree_returns_null_with_polite_note(client, tmp_path):
    """User clicks 'Checkpoint now' on a clean tree → 200 + ok:true +
    checkpoint:null + note.  NOT an error."""
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/commit",
                    json={"path": str(tmp_path),
                          "message": "should be no-op"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["checkpoint"] is None
    assert "nothing to checkpoint" in body["note"]


def test_commit_after_edit_returns_new_checkpoint(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel changed\n")
    r = client.post("/api/checkpoint/commit",
                    json={"path": str(tmp_path),
                          "message": "tweaked fdf"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["checkpoint"] is not None
    assert body["checkpoint"]["summary"] == "tweaked fdf"


def test_commit_uninitialised_dir_is_409(client, tmp_path):
    _seed(tmp_path)
    r = client.post("/api/checkpoint/commit",
                    json={"path": str(tmp_path),
                          "message": "x"})
    assert r.status_code == 409


# ----------------------------------------------------------------- #
#  POST /api/checkpoint/tag                                          #
# ----------------------------------------------------------------- #


def test_tag_creates_annotated_tag(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/tag",
                    json={"path": str(tmp_path),
                          "label": "baseline",
                          "message": "first save point"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["label"] == "baseline"


def test_tag_missing_message_is_400(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/tag",
                    json={"path": str(tmp_path),
                          "label": "x"})
    assert r.status_code == 400


# ----------------------------------------------------------------- #
#  POST /api/checkpoint/restore                                      #
# ----------------------------------------------------------------- #


def test_restore_text_and_binaries_round_trip(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    # Change both text and binary, checkpoint, then restore.
    (tmp_path / "siesta-test.fdf").write_text("v2\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x42" * 4096)
    repo.checkpoint(message="iter 2")
    r = client.post("/api/checkpoint/restore",
                    json={"path": str(tmp_path),
                          "ref": "baseline"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert "siesta-test.DM" in body["restored"]
    assert (tmp_path / "siesta-test.fdf").read_text() == "SystemLabel test\n"
    assert (tmp_path / "siesta-test.DM").read_bytes() == b"\x00" * 2048


def test_restore_on_dirty_tree_returns_advisory(client, tmp_path):
    """Bucket B: HTTP 200 + ok:false + advisory message; the UI
    surfaces this to the user who must commit or discard first."""
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    (tmp_path / "siesta-test.fdf").write_text("dirty\n")
    r = client.post("/api/checkpoint/restore",
                    json={"path": str(tmp_path),
                          "ref": "baseline"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert body["errors_only"] is True
    assert "uncommitted changes" in body["errors"][0]["message"]


def test_restore_unknown_ref_is_404(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/restore",
                    json={"path": str(tmp_path),
                          "ref": "does-not-exist"})
    assert r.status_code == 404


def test_restore_on_legacy_manifest_returns_advisory_with_hint(
        client, tmp_path):
    """Bucket B: legacy 2-col MANIFEST → advisory whose message names
    the migrate-manifest fix."""
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    # Replace the canonical MANIFEST with a 2-col legacy version.
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    dm_sha = hashlib.sha256((arch / "siesta-test.DM").read_bytes()).hexdigest()
    (arch / "MANIFEST").write_text(
        f"{dm_sha}  siesta-test.DM\n", encoding="ascii")
    # Make a dirty commit so restore has something to do.
    (tmp_path / "siesta-test.fdf").write_text("v2\n")
    repo.checkpoint(message="iter 2")
    r = client.post("/api/checkpoint/restore",
                    json={"path": str(tmp_path),
                          "ref": "baseline"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert "migrate-manifest" in body["errors"][0]["message"]


# ----------------------------------------------------------------- #
#  POST /api/checkpoint/migrate-manifest                             #
# ----------------------------------------------------------------- #


def test_migrate_manifest_converts_legacy_in_place(client, tmp_path):
    from molbuilder.checkpoint import (
        Repo, _parse_canonical_manifest,
    )
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    sha = repo._resolve_ref("baseline")
    arch = tmp_path / ".binsnapshots" / sha
    dm_sha = hashlib.sha256((arch / "siesta-test.DM").read_bytes()).hexdigest()
    (arch / "MANIFEST").write_text(
        f"{dm_sha}  siesta-test.DM\n", encoding="ascii")
    r = client.post("/api/checkpoint/migrate-manifest",
                    json={"path": str(tmp_path),
                          "ref": "baseline"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["ref"] == "baseline"
    names = [e["name"] for e in body["entries"]]
    assert "siesta-test.DM" in names
    # On-disk MANIFEST is now canonical.
    parsed = _parse_canonical_manifest(
        (arch / "MANIFEST").read_bytes(), where=str(arch))
    assert "siesta-test.DM" in parsed


def test_migrate_manifest_no_such_ref_is_404(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.post("/api/checkpoint/migrate-manifest",
                    json={"path": str(tmp_path),
                          "ref": "ghost"})
    assert r.status_code == 404


# ----------------------------------------------------------------- #
#  GET /api/checkpoint/diff                                          #
# ----------------------------------------------------------------- #


def test_diff_text_files_round_trips_through_endpoint(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    repo = Repo(str(tmp_path))
    repo.init()
    repo.tag("baseline", message="first")
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel changed\n")
    repo.checkpoint(message="tweaked fdf")
    r = client.get("/api/checkpoint/diff",
                   query_string={"path": str(tmp_path),
                                 "a": "baseline",
                                 "b": "HEAD",
                                 "pathspec": "*.fdf"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    diff = body["diff"]
    assert "SystemLabel test" in diff
    assert "SystemLabel changed" in diff


def test_diff_unknown_ref_is_404(client, tmp_path):
    from molbuilder.checkpoint import Repo
    _seed(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.get("/api/checkpoint/diff",
                   query_string={"path": str(tmp_path),
                                 "a": "HEAD",
                                 "b": "no-such-ref"})
    assert r.status_code == 404
