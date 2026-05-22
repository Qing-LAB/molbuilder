"""End-to-end tests for /api/load: JSON path mode + multipart upload mode.

The multipart branch is the file-picker fallback for users who click
Load without typing a path.  These tests verify both flows produce a
parseable response and that uploaded files come back tagged
``uploaded=True``.
"""

from __future__ import annotations

import io

import pytest

from molbuilder.web.app import create_app
from molbuilder.web.blueprints import watch as app_module


_SIESTA_HEAD = (
    "Welcome to SIESTA -- v4.1\n"
    "redata: prelude\n"
    "outcoor: Atomic coordinates (Ang):\n"
    "   1.00000000    2.00000000    3.00000000   1       1  C\n"
    "\n"
    "siesta: E_KS(eV) =          -50.0000\n"
)


@pytest.fixture
def client():
    return create_app(config={}).test_client()


@pytest.fixture(autouse=True)
def _reset_app_state():
    """Clear the global state between tests so they don't leak."""
    with app_module._lock:
        app_module._state["path"]     = None
        app_module._state["mtime"]    = None
        app_module._state["data"]     = None
        app_module._state["parser"]   = None
        app_module._state["uploaded"] = False
        app_module._state["run_dir"]  = None
    yield


# --------------------------------------------------------------------- #
#  JSON path mode (live-watch)                                          #
# --------------------------------------------------------------------- #


def test_load_by_json_path(client, tmp_path):
    p = tmp_path / "run.out"
    p.write_text(_SIESTA_HEAD)
    r = client.post("/api/watch/load", json={"path": str(p)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["uploaded"] is False
    assert body["format"] == "siesta"


def test_load_by_json_path_missing_file(client, tmp_path):
    r = client.post("/api/watch/load", json={"path": str(tmp_path / "nope.out")})
    body = r.get_json()
    assert r.status_code == 404
    assert body["ok"] is False


def test_load_by_json_path_empty(client):
    r = client.post("/api/watch/load", json={"path": ""})
    body = r.get_json()
    assert r.status_code == 400
    assert body["ok"] is False


# --------------------------------------------------------------------- #
#  Multipart upload mode (file-picker fallback)                         #
# --------------------------------------------------------------------- #


def test_load_by_multipart(client):
    fd = {
        "file": (io.BytesIO(_SIESTA_HEAD.encode()), "run.out"),
    }
    r = client.post("/api/watch/load",
                    data=fd,
                    content_type="multipart/form-data")
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["uploaded"] is True
    assert body["uploaded_filename"] == "run.out"
    assert body["format"] == "siesta"


def test_load_by_multipart_unrecognised_format(client):
    """An upload that no parser claims should 400 cleanly and not
    leave a stale temp file referenced in _state."""
    fd = {
        "file": (io.BytesIO(b"junk content nothing recognises\n"),
                 "garbage.txt"),
    }
    r = client.post("/api/watch/load",
                    data=fd,
                    content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False


def test_load_by_multipart_replaces_previous_upload(client, tmp_path):
    """A second upload must clean up the previous temp file (best-effort
    -- we just check that _last_temp_upload moves to the new path)."""
    a = io.BytesIO(_SIESTA_HEAD.encode())
    client.post("/api/watch/load",
                data={"file": (a, "first.out")},
                content_type="multipart/form-data")
    first_temp = app_module._last_temp_upload
    assert first_temp is not None

    b = io.BytesIO(_SIESTA_HEAD.encode())
    client.post("/api/watch/load",
                data={"file": (b, "second.out")},
                content_type="multipart/form-data")
    second_temp = app_module._last_temp_upload
    assert second_temp is not None
    assert second_temp != first_temp


def test_load_by_multipart_persists_path_for_data_polls(client):
    """After an upload, /api/data must still return the parsed payload.
    The temp file lingers (we don't delete it on the same request) so
    the existing _refresh_if_changed machinery handles it normally."""
    fd = {"file": (io.BytesIO(_SIESTA_HEAD.encode()), "polled.out")}
    client.post("/api/watch/load",
                data=fd,
                content_type="multipart/form-data")

    r = client.get("/api/watch/data")
    body = r.get_json()
    assert body["ok"] is True
    assert body["uploaded"] is True
    assert body["data"]["source_format"] == "siesta"


# --------------------------------------------------------------------- #
#  Directory mode (job-layout v1)                                       #
#                                                                       #
#  Per docs/protocols/job-layout.md the loader resolves a directory path     #
#  to a single file via a documented discovery chain.  These tests pin  #
#  each rung of the chain so a regression at the protocol boundary      #
#  fails loudly here rather than as user-visible "load failed" errors. #
# --------------------------------------------------------------------- #


_MOLWATCH_HEAD = (
    "# molwatch trajectory log v1\n"
    "# engine: siesta\n"
    "# step: 0\n"
)


def test_load_directory_picks_molwatch_log_first(client, tmp_path):
    """A directory with a .molwatch.log resolves to that file even if a
    .out file is also present (the protocol prefers .molwatch.log)."""
    (tmp_path / "my-job.molwatch.log").write_text(_MOLWATCH_HEAD)
    (tmp_path / "my-job.out").write_text(_SIESTA_HEAD)
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["resolved_from"] == str(tmp_path)
    assert body["path"].endswith("my-job.molwatch.log")


def test_load_directory_falls_back_to_fdf_system_label(client, tmp_path):
    """No .molwatch.log; an .fdf is present.  Loader parses
    SystemLabel and looks for <label>.molwatch.log, then <label>.out."""
    (tmp_path / "input.fdf").write_text("SystemLabel my-job\n")
    (tmp_path / "my-job.out").write_text(_SIESTA_HEAD)
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["path"].endswith("my-job.out")


def test_load_directory_falls_back_to_py_job_name(client, tmp_path):
    """No .molwatch.log or .fdf; a .py is present with a molbuilder-
    style ``job_name = "..."`` declaration.  Loader picks up
    <job>.molwatch.log."""
    (tmp_path / "script.py").write_text(
        '"""molbuilder PySCF script"""\n'
        'job_name = "my-pyscf-run"\n'
        'print("hi")\n'
    )
    (tmp_path / "my-pyscf-run.molwatch.log").write_text(_MOLWATCH_HEAD)
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["path"].endswith("my-pyscf-run.molwatch.log")


def test_load_directory_empty_returns_chain_error(client, tmp_path):
    """An empty directory returns a 404 whose error message names the
    discovery chain so the user can see what was tried."""
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert r.status_code == 404
    assert body["ok"] is False
    assert "job-layout.md" in body["error"]
    assert "*.molwatch.log" in body["error"]
    assert "*.fdf" in body["error"]


def test_load_directory_picks_newest_molwatch_log(client, tmp_path):
    """When multiple *.molwatch.log files exist (staged run), the
    newest mtime wins."""
    import os, time
    older = tmp_path / "stage1.molwatch.log"
    newer = tmp_path / "stage2.molwatch.log"
    older.write_text(_MOLWATCH_HEAD)
    newer.write_text(_MOLWATCH_HEAD)
    # Force the older to actually be older.
    past = time.time() - 60
    os.utime(older, (past, past))
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["path"].endswith("stage2.molwatch.log")


def test_load_file_path_unchanged(client, tmp_path):
    """File-mode (back-compat): passing a regular file path skips the
    discovery chain and ``resolved_from`` is null."""
    p = tmp_path / "run.out"
    p.write_text(_SIESTA_HEAD)
    r = client.post("/api/watch/load", json={"path": str(p)})
    body = r.get_json()
    assert body["ok"] is True
    assert body["resolved_from"] is None


# --------------------------------------------------------------------- #
#  Multi-stage merge (job-layout v1, Cut 3)                             #
#                                                                       #
#  When a directory contains > 1 *.molwatch.log files (the staged       #
#  relaxation case), the loader concatenates their trajectories into    #
#  one merged dict and tags each source as a stage.                     #
# --------------------------------------------------------------------- #


_MOLWATCH_TWO_STEPS = (
    "# molwatch trajectory log v1\n"
    "# engine: pyscf\n"
    "==== molwatch step 0 begin ====\n"
    "step_index: 0\n"
    "n_atoms: 3\n"
    "coordinates (Ang):\n"
    "   O   0.00000000   0.00000000   0.00000000\n"
    "   H   0.95700000   0.00000000   0.00000000\n"
    "   H  -0.23900000   0.92700000   0.00000000\n"
    "energy (eV): -76.40000000\n"
    "==== molwatch step 0 end ====\n"
    "==== molwatch step 1 begin ====\n"
    "step_index: 1\n"
    "n_atoms: 3\n"
    "coordinates (Ang):\n"
    "   O   0.00000000   0.00000000   0.00000000\n"
    "   H   0.95700000   0.00000000   0.00000000\n"
    "   H  -0.23900000   0.92700000   0.00000000\n"
    "energy (eV): -76.50000000\n"
    "==== molwatch step 1 end ====\n"
)


def test_load_directory_multi_stage_merges_trajectories(client, tmp_path):
    """Two *.molwatch.log files in a directory -> one merged
    trajectory; ``stages`` metadata attributes each frame range to a
    source file."""
    import os, time
    s1 = tmp_path / "my-job-stage1.molwatch.log"
    s2 = tmp_path / "my-job-stage2.molwatch.log"
    s1.write_text(_MOLWATCH_TWO_STEPS)
    s2.write_text(_MOLWATCH_TWO_STEPS)
    past = time.time() - 60
    os.utime(s1, (past, past))
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True, body
    # 2 stages * 2 frames each = 4 merged frames.
    assert len(body["data"]["frames"]) == 4
    # Iterations are renumbered globally for the plot x-axis.
    assert body["data"]["iterations"] == [0, 1, 2, 3]
    # Stages metadata names each source file in mtime order (oldest
    # first) and tags frame ranges.
    stages = body["stages"]
    assert [s["name"] for s in stages] == [
        "my-job-stage1.molwatch.log",
        "my-job-stage2.molwatch.log",
    ]
    assert stages[0]["start_frame"] == 0 and stages[0]["n_frames"] == 2
    assert stages[1]["start_frame"] == 2 and stages[1]["n_frames"] == 2
    # Active polling target = the newest log (stage 2).
    assert body["path"].endswith("my-job-stage2.molwatch.log")


def test_load_directory_single_log_no_stages_field(client, tmp_path):
    """A directory with exactly one *.molwatch.log goes the single-
    log path; the response should NOT carry a ``stages`` field
    (so the frontend's stage-marker logic stays inert)."""
    p = tmp_path / "my-job.molwatch.log"
    p.write_text(_MOLWATCH_HEAD)
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert body["ok"] is True
    assert "stages" not in body or not body["stages"]


def test_multi_stage_merge_survives_subsequent_poll(client, tmp_path):
    """Regression for H1: after a multi-stage load, the next
    /api/watch/data poll must NOT collapse the merged view back to
    the newest stage's frames alone.  ``_refresh_if_changed`` re-runs
    the merge over the full set of *.molwatch.log files."""
    import os, time
    s1 = tmp_path / "my-job-stage1.molwatch.log"
    s2 = tmp_path / "my-job-stage2.molwatch.log"
    s1.write_text(_MOLWATCH_TWO_STEPS)
    s2.write_text(_MOLWATCH_TWO_STEPS)
    past = time.time() - 60
    os.utime(s1, (past, past))
    # First load merges 4 frames.
    r1 = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body1 = r1.get_json()
    assert body1["ok"] is True
    assert len(body1["data"]["frames"]) == 4
    # Now bump the newest stage's mtime so the poll sees a change,
    # without modifying the file contents -- the parser will return
    # the same 2 frames for stage 2.  After the poll, the merged
    # trajectory must STILL contain all 4 frames (not collapse to 2).
    future = time.time() + 60
    os.utime(s2, (future, future))
    r2 = client.get("/api/watch/data")
    body2 = r2.get_json()
    assert body2["ok"] is True
    assert body2["changed"] is not False
    assert len(body2["data"]["frames"]) == 4, (
        "multi-stage merge collapsed on poll; expected 4 merged "
        f"frames, got {len(body2['data']['frames'])}"
    )
    assert body2["data"]["stages"][0]["name"] == "my-job-stage1.molwatch.log"


def test_multi_stage_merge_survives_empty_stage(client, tmp_path):
    """A *.molwatch.log that the parser accepts but extracts zero
    frames from (e.g. header-only, no step blocks yet -- common
    for a stage that hasn't started writing) must NOT take down
    the merge.  Stages with n_frames == 0 are recorded but
    contribute no frames."""
    import os, time
    s1 = tmp_path / "my-job-stage1.molwatch.log"
    s2 = tmp_path / "my-job-stage2.molwatch.log"     # header only
    s3 = tmp_path / "my-job-stage3.molwatch.log"
    s1.write_text(_MOLWATCH_TWO_STEPS)
    s2.write_text("# molwatch trajectory log v1\n# engine: pyscf\n")
    s3.write_text(_MOLWATCH_TWO_STEPS)
    base = time.time() - 60
    os.utime(s1, (base,      base))
    os.utime(s2, (base + 10, base + 10))
    os.utime(s3, (base + 20, base + 20))
    body = client.post("/api/watch/load",
                       json={"path": str(tmp_path)}).get_json()
    assert body["ok"] is True, body
    assert len(body["data"]["frames"]) == 4         # 2 from s1 + 2 from s3
    by_name = {s["name"]: s for s in body["stages"]}
    assert by_name["my-job-stage2.molwatch.log"]["n_frames"] == 0
    assert by_name["my-job-stage1.molwatch.log"]["n_frames"] == 2
    assert by_name["my-job-stage3.molwatch.log"]["n_frames"] == 2


def test_multi_stage_merge_survives_parse_exception(
        client, tmp_path, monkeypatch):
    """If MolwatchLogParser.parse raises mid-merge (mid-write tear,
    binary garbage, etc.), the merge must continue across the
    surviving stages and tag the failed stage with an ``error``
    field in its stages-metadata entry."""
    import os, time
    from molbuilder.parsers.molwatch_log import MolwatchLogParser
    s1 = tmp_path / "my-job-stage1.molwatch.log"
    s2 = tmp_path / "my-job-stage2.molwatch.log"
    s3 = tmp_path / "my-job-stage3.molwatch.log"
    s1.write_text(_MOLWATCH_TWO_STEPS)
    s2.write_text(_MOLWATCH_TWO_STEPS)
    s3.write_text(_MOLWATCH_TWO_STEPS)
    base = time.time() - 60
    os.utime(s1, (base,      base))
    os.utime(s2, (base + 10, base + 10))
    os.utime(s3, (base + 20, base + 20))
    real_parse = MolwatchLogParser.parse

    def fake_parse(cls, path):
        if str(path).endswith("stage2.molwatch.log"):
            raise RuntimeError("simulated mid-write tear")
        return real_parse(path)

    monkeypatch.setattr(MolwatchLogParser, "parse",
                        classmethod(fake_parse))
    body = client.post("/api/watch/load",
                       json={"path": str(tmp_path)}).get_json()
    assert body["ok"] is True, body
    assert len(body["data"]["frames"]) == 4         # 2 from s1 + 2 from s3
    by_name = {s["name"]: s for s in body["stages"]}
    assert "error" in by_name["my-job-stage2.molwatch.log"]
    assert "simulated mid-write tear" \
        in by_name["my-job-stage2.molwatch.log"]["error"]
    assert by_name["my-job-stage2.molwatch.log"]["n_frames"] == 0


def test_multi_stage_merge_preserves_per_stage_step_indices(client, tmp_path):
    """The merged dict carries both ``iterations`` (renumbered
    globally for the plot x-axis) and ``step_indices`` (the per-
    stage step numbers from each source log).  Save-frame-as-XYZ
    and tooltip use cases need the latter."""
    import os, time
    s1 = tmp_path / "my-job-stage1.molwatch.log"
    s2 = tmp_path / "my-job-stage2.molwatch.log"
    s1.write_text(_MOLWATCH_TWO_STEPS)
    s2.write_text(_MOLWATCH_TWO_STEPS)
    past = time.time() - 60
    os.utime(s1, (past, past))
    body = client.post("/api/watch/load",
                       json={"path": str(tmp_path)}).get_json()
    # 4 global frames; iterations renumbered globally.
    assert body["data"]["iterations"] == [0, 1, 2, 3]
    # step_indices per-stage: each stage starts at 0 and increments.
    assert body["data"]["step_indices"] == [0, 1, 0, 1]
