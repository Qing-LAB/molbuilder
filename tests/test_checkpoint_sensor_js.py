"""Sidebar sensor refresh-contract tests (run-checkpoints.md § 12, L3).

This file pins the *refresh model* the checkpoint sensor depends on,
after the 2026-06-26 decision to replace background polling with
explicit refresh (run-checkpoints.md § 6.2 + § 11 decision 7).  Three
guarantees, all server-side except the last (a cheap source guard):

  1. ``/api/checkpoint/state`` is CHEAP: it returns the documented
     fields and does NOT walk ``.binsnapshots/`` -- archive size stays
     at its ``0`` default even when an archive exists on disk.  This is
     the regression gate for the Option-A fix (the walk used to run on
     every 5 s poll; it now would run on every directory-enter, so it
     was moved off the refresh path entirely).
  2. A git failure inside ``state()`` surfaces as a structured
     ``{ok:false, error}`` envelope (HTTP 500, web-api.md bucket D)
     that the JS sensor renders as a 🔴 error pill -- NOT an unhandled
     crash or empty body.
  3. ``checkpoint.js`` contains no ``setInterval`` -- there is no
     background poll loop to reintroduce.  The full JS refresh
     behaviour (fires on directory-enter, fires on manual Refresh,
     gated to rel-depth-3 run dirs) is browser-scope and belongs to
     the Playwright graph/sensor E2E; this is the cheap structural
     guard that the polling timer does not come back.

Real Flask test client + real filesystem + real git, mirroring the
no-mocks discipline of test_checkpoint_routes.py.
"""
from __future__ import annotations

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
    reason="git not on PATH; checkpoint sensor tests need git ≥ 2.20",
)


@pytest.fixture
def client():
    return create_app(config={}).test_client()


def _seed_with_archive(tmp_path: Path) -> Path:
    """A run dir with a tracked .fdf and a binary .DM that init()
    archives into .binsnapshots/<sha>/ (2048 bytes)."""
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    (tmp_path / "siesta-test.DM").write_bytes(b"\x00" * 2048)
    return tmp_path


# ----------------------------------------------------------------- #
#  1. /state is cheap -- no .binsnapshots/ walk on the refresh path #
# ----------------------------------------------------------------- #


def test_state_does_not_walk_binsnapshots(client, tmp_path):
    """Even with a real 2048-byte archive on disk, the refresh-path
    state() reports archive_total_bytes == 0 (it never stats the
    archive).  Regression gate for run-checkpoints.md § 5.2 / § 6.2."""
    from molbuilder.checkpoint import Repo
    _seed_with_archive(tmp_path)
    Repo(str(tmp_path)).init()

    # The archive really exists on disk -- so a 0 below means "did not
    # walk", not "nothing to walk".
    archive_files = list((tmp_path / ".binsnapshots").rglob("*.DM"))
    assert archive_files, "precondition: init() should archive the .DM"

    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["state"]["initialized"] is True
    assert body["state"]["archive_total_bytes"] == 0


def test_state_returns_documented_cheap_shape(client, tmp_path):
    """The sensor reads exactly these fields; none requires an archive
    walk (run-checkpoints.md § 5.2 wire shape)."""
    from molbuilder.checkpoint import Repo
    _seed_with_archive(tmp_path)
    Repo(str(tmp_path)).init()
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 200
    state = r.get_json()["state"]
    for key in ("path", "initialized", "head", "current_branch",
                "dirty", "untracked", "archive_total_bytes"):
        assert key in state, f"sensor field {key!r} missing from /state"


# ----------------------------------------------------------------- #
#  2. git failure -> structured envelope, not a crash               #
# ----------------------------------------------------------------- #


def test_state_git_failure_is_structured_error_not_crash(
        client, tmp_path, monkeypatch):
    """When state() raises (transient git failure), the route returns a
    structured {ok:false, error} envelope (HTTP 500, web-api.md bucket
    D) the sensor renders as a 🔴 error pill -- never a bare crash or
    empty body."""
    import molbuilder.web.blueprints.checkpoint as bp
    _seed_with_archive(tmp_path)

    def _boom(self):
        raise bp.CheckpointError("git rev-parse exploded (simulated)")

    monkeypatch.setattr(bp.Repo, "state", _boom)
    r = client.get("/api/checkpoint/state",
                   query_string={"path": str(tmp_path)})
    assert r.status_code == 500
    body = r.get_json()
    assert body is not None, "must return a JSON envelope, not empty body"
    assert body["ok"] is False
    assert isinstance(body.get("error"), str) and body["error"]
    assert "simulated" in body["error"]


# ----------------------------------------------------------------- #
#  3. no background poll loop in the JS                              #
# ----------------------------------------------------------------- #


def test_checkpoint_js_has_no_polling_timer():
    """The explicit-refresh model (run-checkpoints.md § 11 decision 7)
    forbids a background poll loop.  Guard against setInterval creeping
    back into the sensor.  We strip line comments first so the doc-
    comment mention of 'no setInterval' doesn't trip the check."""
    js = (Path(__file__).resolve().parent.parent
          / "molbuilder" / "web" / "static" / "lib"
          / "projects" / "checkpoint.js").read_text()
    code_lines = []
    for ln in js.splitlines():
        stripped = ln.strip()
        if stripped.startswith("*") or stripped.startswith("//") \
                or stripped.startswith("/*"):
            continue
        code_lines.append(ln)
    code = "\n".join(code_lines)
    assert "setInterval" not in code, (
        "checkpoint.js must not poll -- refresh is explicit "
        "(directory-enter + manual Refresh) per run-checkpoints.md "
        "§ 6.2 / § 11 decision 7")
