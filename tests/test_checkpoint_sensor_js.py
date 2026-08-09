"""Sidebar sensor refresh-contract tests (docs/web/projects.md;
test design: docs/execution/checkpointing.md § 13).

This file pins the *refresh model* the checkpoint sensor depends on,
after the 2026-06-26 decision to replace background polling with
explicit refresh (docs/web/projects.md).  Three
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


def test_state_does_not_read_big_files_it_does_not_have_to(client, tmp_path):
    """The sidebar calls this on every directory-enter (docs/web/projects.md).

    A folder holding a 2 GB density matrix must not be read end-to-end to
    answer "is anything unsaved".  A file whose size matches its record and
    which has not been touched since the state was saved cannot have changed,
    so it is ruled out by two ``stat`` fields rather than by hashing.

    The gate is behavioural, not a timing assertion: make the file unreadable
    after saving it.  If the answer still comes back clean, nothing opened it.
    """
    import json
    import os
    from molbuilder.checkpoint import Repo
    from molbuilder.runtime_config import PROJECT_CONFIG_FILENAME

    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(json.dumps(
        {"checkpoint": {"size_limit_bytes": 1024, "engines": {"generic": []}}}))
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    big = tmp_path / "siesta-test.DM"
    big.write_bytes(b"\x00" * 4096)
    Repo(str(tmp_path)).init(note="set up")
    assert list((tmp_path / ".binsnapshots").rglob("siesta-test.DM")), (
        "precondition: the .DM is over the limit and must be archived")

    # Age it past the save.  A state's timestamp has one-second resolution, so
    # a file written in the same second is deliberately NOT ruled out -- this
    # makes the "untouched since the save" case the one under test rather than
    # a race.
    saved = os.stat(big).st_mtime - 60
    os.utime(big, (saved, saved))
    os.chmod(big, 0o000)
    try:
        r = client.get("/api/checkpoint/state",
                       query_string={"path": str(tmp_path)})
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] and body["clean"], (
            "the sensor read the file it did not need to read")
    finally:
        os.chmod(big, 0o644)


def test_state_returns_the_shape_the_sensor_reads(client, tmp_path):
    """§ 5's vocabulary, and nothing that would need an archive walk.

    No archive size: the number was never true (hard links counted in full)
    and fed no decision, since nothing prunes (checkpointing.md § 12).
    """
    from molbuilder.checkpoint import Repo
    (tmp_path / "siesta-test.fdf").write_text("SystemLabel test\n")
    Repo(str(tmp_path)).init(note="set up")
    body = client.get("/api/checkpoint/state",
                      query_string={"path": str(tmp_path)}).get_json()
    for key in ("path", "initialized", "standing_at", "clean",
                "changed", "added", "deleted", "unsaved", "ignore_edited"):
        assert key in body, f"sensor field {key!r} missing from /state"
    assert "archive_total_bytes" not in body
    assert body["standing_at"]["note"] == "set up"


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

    monkeypatch.setattr(bp.Repo, "status", _boom)
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
    """The explicit-refresh model (docs/web/projects.md)
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
        "(directory-enter + manual Refresh) per docs/web/projects.md "
        "docs/web/projects.md")
