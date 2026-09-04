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
def client(tmp_path):
    """Flask test client with ``tmp_path`` injected as the picker root.

    2026-06-18 security hotfix (audit B1) routes /api/watch/load
    through ``_resolve_within_roots``, which constrains the JSON-path
    mode to the configured picker roots (default: ``<cwd>/projects``).
    These tests want to point /api/watch/load at fixture files under
    ``tmp_path``, so we inject the tmp directory as the sole picker
    root for the duration of the test.  The conftest-level
    ``_reset_caps_each_test`` autouse fixture resets after each test.
    """
    from molbuilder import diagnostics
    from molbuilder.diagnostics import Capabilities

    class _TmpRootCaps(Capabilities):
        def file_picker_roots(self):  # type: ignore[override]
            return ((tmp_path.resolve(), "test-tmp"),)

    # ``create_app`` calls ``_initialize_diagnostics()`` which
    # OVERWRITES any previously-set capabilities, so we must inject
    # AFTER it runs.  Order-dependent — pin it here.
    app = create_app(config={})
    diagnostics.set_capabilities(_TmpRootCaps())
    return app.test_client()


@pytest.fixture
def client_with_default_roots():
    """Test client that does NOT inject tmp_path — used by the
    security regression to verify the default deployment (projects/
    only) rejects an out-of-root path.
    """
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
#  Security regression — audit B1 (2026-06-18)                          #
# --------------------------------------------------------------------- #


def test_load_by_json_path_rejects_path_outside_picker_roots(
        client_with_default_roots):
    """Pre-fix /api/watch/load resolved arbitrary host paths via
    ``os.path.realpath`` with an OPTIONAL ``MOLBUILDER_WATCH_ROOT``
    gate that the default deployment left unset.  A logged-in user
    could POST ``{"path": "/etc/shadow"}`` and the parser read it.

    Hotfix routes through ``_resolve_within_roots``, which constrains
    the path to picker roots (default: ``<cwd>/projects``).  This
    test posts ``/etc/passwd`` and asserts the picker error fires
    BEFORE any disk read attempt — proving the read-arbitrary-file
    primitive is gone.

    Pinned to web-api.md § 2.1 (every path-taking endpoint goes
    through ``_resolve_within_roots``).
    """
    r = client_with_default_roots.post(
        "/api/watch/load",
        json={"path": "/etc/passwd"},
    )
    body = r.get_json()
    assert r.status_code == 400, (
        f"expected 400 (outside picker roots); got {r.status_code} "
        f"body={body!r}.  If you see 200/404, the security fix has "
        f"regressed and the endpoint is reading arbitrary host files."
    )
    assert body["ok"] is False
    # The picker error names the resolved path + roots so the user
    # knows WHY it was rejected.
    assert "outside" in body["error"].lower(), body["error"]


def test_load_by_json_path_rejects_dot_dot_traversal(
        client_with_default_roots):
    """``..`` is rejected early per the defense-in-depth check in
    ``_resolve_within_roots``."""
    r = client_with_default_roots.post(
        "/api/watch/load",
        json={"path": "projects/../etc/passwd"},
    )
    body = r.get_json()
    assert r.status_code == 400
    assert body["ok"] is False
    # The picker's defense-in-depth check rejects raw ``..`` before
    # resolution, so the error names ``..`` rather than "outside".
    assert ".." in body["error"], body["error"]


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
#  runtime_info.convergence_targets contract                            #
#                                                                       #
#  Pin the END-TO-END API contract: parser → Trajectory.runtime_info →  #
#  HTTP /api/watch/data response → frontend.  The first two links now   #
#  sit together here: the parser test moved in from                     #
#  test_live_poll_invariants_audit.py when that file was retired        #
#  2026-09-03 (testing.md § 3a — 18 of its 21 tests asserted on the     #
#  spelling of lines in core.js rather than on anything the code did).  #
#  It was the only genuinely behavioural test in the file with no home  #
#  elsewhere, and it belongs beside the HTTP link it feeds.             #
# --------------------------------------------------------------------- #


def test_the_siesta_parser_reads_convergence_targets_from_the_input_echo():
    """The first link of the chain: SIESTA's own input echo → runtime_info.

    The hemeC stage-2 fixture tightened `MD.MaxForceTol` to 0.02 eV/Å from
    the 0.04 default — a stage-2 run's whole point — so the parsed value
    proves the echo was read rather than a default reproduced.
    """
    from pathlib import Path

    from molbuilder.parse.engines.siesta import SiestaParser

    path = (Path(__file__).resolve().parent
            / "fixtures" / "siesta_frozen"
            / "hemeC-stage2-run3-finished-42fr.out")
    traj = SiestaParser.parse(str(path))
    ct = traj.runtime_info.get("convergence_targets")
    assert ct is not None, (
        "SIESTA parser dropped its convergence_targets extraction.  The "
        "Results-tab threshold line + summary block now have nothing to "
        "render.  Check that _SIESTA_FORCE_TOL_RE et al. are still "
        "matching 'redata: Force tolerance = ...'.")
    assert ct.get("source") == "siesta_input_echo"
    assert ct.get("max_force_tol_eV_per_A") == 0.02, (
        "0.04 is the SIESTA default -- reading it back here means the echo "
        "was not parsed and a default leaked through instead")
    assert ct.get("dm_tolerance") == 1e-4
    assert ct.get("max_scf_iter") == 500


_SIESTA_WITH_REDATA = (
    "Welcome to SIESTA -- v4.1\n"
    "redata: Force tolerance              =        0.0400 eV/Ang\n"
    "redata: DM tolerance for SCF          =     0.000100\n"
    "redata: Max. number of SCF Iter        =          500\n"
    "redata: Max atomic displ per move      =        0.1000 Ang\n"
    "redata: Maximum number of optimization moves        =       80\n"
    "outcoor: Atomic coordinates (Ang):\n"
    "   1.00000000    2.00000000    3.00000000   1       1  C\n"
    "\n"
    "siesta: E_KS(eV) =          -50.0000\n"
)


def test_watch_data_surfaces_runtime_info_convergence_targets(client):
    """``/api/watch/data`` MUST carry ``data.runtime_info.convergence_targets``
    when the SIESTA parser extracted it from the input echo.

    Documented contract: docs/web/web-api.md
    ("runtime_info: per-stage CPU/MPI/GPU report — see types/parsers.md")
    + docs/web/results.md (the convergence_targets
    sub-shape with per-key units and parser sources).

    Frontend consumer: lib/trajectory/core.js::_renderConvergenceSummary
    reads ``data.runtime_info.convergence_targets`` to render the
    threshold lines on the force plot + the "Convergence targets"
    summary band in the trajectory inspector.

    Pre-2026-06-13 the parser → traj link had a test, and the
    traj → partial link had a test, but the HTTP-layer link
    (parser → traj → HTTP /api/watch/data → frontend) was unpinned.
    A silent removal of the field at the serializer layer would
    have silently disabled the threshold lines.
    """
    fd = {"file": (io.BytesIO(_SIESTA_WITH_REDATA.encode()),
                   "with_redata.out")}
    client.post("/api/watch/load",
                data=fd,
                content_type="multipart/form-data")
    r = client.get("/api/watch/data")
    body = r.get_json()
    assert body["ok"] is True
    data = body["data"]
    assert "runtime_info" in data, (
        "data.runtime_info missing from /api/watch/data response — "
        "web-api.md § 4 documents it as part of the contract")
    runtime_info = data["runtime_info"]
    assert "convergence_targets" in runtime_info, (
        "runtime_info.convergence_targets missing — the SIESTA parser "
        "captured the redata: lines but the serializer dropped the "
        "field on the way to the HTTP response.  Threshold lines on "
        "the trajectory inspector force plot will be missing.")
    ct = runtime_info["convergence_targets"]
    # Every key the threshold lines need, + the source tag
    # (`web/trajectory.md` § 3: the targets come from the run's own
    # output, and the label says which reader found them).
    for key in ("max_force_tol_eV_per_A", "dm_tolerance",
                "max_scf_iter", "max_geom_iter", "max_displ_ang", "source"):
        assert key in ct, (
            f"convergence_targets missing documented key {key!r}: "
            f"{sorted(ct)}")
    assert ct["source"] == "siesta_input_echo"
    # max_geom_iter is the optimization-step cap (MD.NumCGsteps);
    # added 2026-06-13 after the user reported the gap in the
    # trajectory inspector's convergence summary.
    assert ct["max_geom_iter"] == 80


# --------------------------------------------------------------------- #
#  Directory mode (job-layout v1)                                       #
#                                                                       #
#  Per docs/execution/job-contracts.md the loader resolves a directory path     #
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
    # The discovery chain cites its protocol doc; that doc was renamed
    # protocols/job-layout.md -> execution/job-contracts.md in the
    # 2026-07 docs migration, and this assertion was left behind.
    assert "docs/execution/job-contracts.md" in body["error"]
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
    from molbuilder.parse.engines.molwatch import MolwatchLogFileParser
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
    real_parse = MolwatchLogFileParser.parse

    def fake_parse(cls, path):
        if str(path).endswith("stage2.molwatch.log"):
            raise RuntimeError("simulated mid-write tear")
        return real_parse(path)

    monkeypatch.setattr(MolwatchLogFileParser, "parse",
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


# --------------------------------------------------------------------- #
#  `format` names the ENGINE, not the parser that read the file         #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_format_names_the_engine_not_the_parser_that_read_it(
        client, tmp_path, engine):
    """A `.molwatch.log` is read by the parser named `molwatch` whatever
    wrote it — and the wire must still say which ENGINE ran.

    Two different facts share one field's name if you are not careful:

        `label`  — who read it   ("molwatch unified log (.molwatch.log)")
        `format` — what ran it   ("siesta" / "pyscf")

    The route sent `parser_cls.name` for both.  For an engine-native file
    they coincide — a SIESTA `.out` is read by the parser called `siesta`,
    which is why `test_load_by_json_path` above passed throughout — so nothing looked wrong.  They diverge for exactly the
    file `job-contracts.md` § calls *"THE canonical trajectory, preferred
    by every reader"*: every molbuilder-generated run arrived as
    `"molwatch"`.

    The cost was visible: `lib/trajectory/core.js` branches on
    `state.format === "siesta"` / `"pyscf"` to title the SCF banner, and
    its own comment calls the third branch *"the rare fallback"* for a log
    with no engine header.  It was the only branch that ever ran, so every
    run showed the neutral "SCF progress / Opt step" instead of "SIESTA
    DFT SCF progress / CG/MD step".

    The engine is not inferred here from a filename: the log DECLARES it
    (`# engine: <name>`), `parse/engines/molwatch.py` reads that line into
    `source_format`, and the route now reports what the parser found.
    """
    import numpy as np

    from molbuilder.structure import Structure
    from molbuilder.trajectory_log.format import write_initial_preview

    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]))
    log = tmp_path / "probe.molwatch.log"
    # THE PRODUCTION WRITER, with the engine as its own parameter -- the
    # same door SIESTA's parser-on-stdout path and the PySCF emitter use.
    write_initial_preview(struct, log, job="probe", engine=engine)

    r = client.post("/api/watch/load", json={"path": str(log)})
    body = r.get_json()
    assert body["ok"], body
    assert body["format"] == engine, (
        f"the wire says format={body['format']!r} for a log whose own "
        f"header declares `# engine: {engine}`.  `format` names the engine "
        f"that ran; `label` names the parser that read it -- and for a "
        f".molwatch.log the parser is always `molwatch`, which is why "
        f"reporting the parser's name here erased the distinction for "
        f"every molbuilder-generated run.")
    assert "molwatch" in body["label"].lower(), (
        f"label={body['label']!r} -- it should still name the READER, so "
        f"the two facts stay separable")


def test_an_upload_never_asks_the_temp_directory_which_engine_ran(
        client, tmp_path, monkeypatch):
    """LOAD and POLL must agree about an uploaded file's engine.

    An upload has no run directory. `web-api.md`'s `/api/watch/*` row:
    *"`source_format` is the fallback and only an upload reaches it"* --
    so the load path passes `None` to `_engine_of` deliberately.  The
    POLL path passed `os.path.dirname(state["path"])`, which for an
    upload is the SYSTEM TEMP DIRECTORY: shared, and full of files
    belonging to other work.  One file then got two answers, the second
    decided by litter -- and every `*.py` / `*.fdf` / `*.run.sh` in
    `/tmp` was read on every poll to reach it.

    The `.fdf` planted below is the whole point: it makes the temp
    directory sniff as SIESTA, so a poll that asks the directory must
    disagree with the load.  Without it this test passes against the bug
    (found by adversarial review, 2026-09-04 -- the full 8360-test suite
    was green while this defect sat in it, because no test built this
    directory shape).
    """
    import tempfile as _tempfile

    monkeypatch.setattr(_tempfile, "tempdir", str(tmp_path))
    (tmp_path / "someone_elses_run.fdf").write_text("SystemLabel other\n")

    # Built through the production writer, not hand-typed: a log the
    # parser refuses proves nothing about which directory was asked.
    import numpy as np

    from molbuilder.structure import Structure
    from molbuilder.trajectory_log.format import write_initial_preview

    src = tmp_path / "src" / "co2.molwatch.log"
    src.parent.mkdir()
    write_initial_preview(
        Structure(elements=["O", "C", "O"],
                  positions=np.array([[0.0, 0.0, -1.16],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.16]]),
                  vacuum=(10.0, 10.0, 10.0)),
        src, job="co2", engine="pyscf")
    r = client.post("/api/watch/load",
                    data={"file": (io.BytesIO(src.read_bytes()),
                                   "co2.molwatch.log")},
                    content_type="multipart/form-data")
    load = r.get_json()
    assert load["ok"] is True, load
    assert load["uploaded"] is True

    poll = client.get("/api/watch/data").get_json()
    assert poll["ok"] is True, poll
    assert poll["format"] == load["format"], (
        f"the load says the engine is {load['format']!r} and the poll says "
        f"{poll['format']!r}. The poll is asking the shared temp directory, "
        f"where an unrelated '.fdf' is sitting.")


def test_a_single_geometry_is_refused_by_name_not_by_crashing(
        client, isolated_projects_root):
    """`<job>_optimized.xyz` must answer 400 with a sentence, never 500.

    PySCF writes this file for every optimization, and it was TWO bugs
    stacked, both measured through this route:

    1. Two parsers claimed it -- `pyscf` accepts any structurally valid
       XYZ, `pyscf-geom` accepts any `*_optimized.xyz` -- so `detect()`,
       which is exactly-one-or-raise, raised `AmbiguousFormatError`.
       That is the SIBLING of `UnknownFormatError`, not its subclass,
       and all five detection call sites caught only the latter: an
       unhandled exception, HTTP 500, HTML body.  `pyscf-geom`'s own
       `can_parse` already documented the division of labour; only one
       side of it had been written.
    2. Underneath that, `/api/watch/*` is the TRAJECTORY route and never
       checked what the detected parser produces, so a `StructureResult`
       reached code that reads `.frames`.

    The file is normally ABSORBED into the run's `.molwatch.log` entry
    (`results.md` § 2.3) so the picker does not offer it -- but
    absorption narrows the MENU, not what can be opened.

    The `_geom_optim.xyz` control is load-bearing: a fix that refused
    every `.xyz` would pass the first assertion and break the viewer.
    """
    d = isolated_projects_root / "optim_refusal"
    d.mkdir(parents=True)
    xyz = "3\nCO2\nO 0 0 -1.16\nC 0 0 0\nO 0 0 1.16\n"

    final = d / "bdt_optimized.xyz"
    final.write_text(xyz)
    r = client.post("/api/watch/load", json={"path": str(final)})
    assert r.status_code == 400, (
        f"a single-geometry file must be refused with a message; got "
        f"{r.status_code} ({r.headers.get('Content-Type')})")
    err = r.get_json()["error"]
    assert "trajectory" in err and "molwatch" in err, (
        f"the refusal must say what the file is and where the trajectory "
        f"lives; got: {err}")

    traj = d / "bdt_geom_optim.xyz"
    traj.write_text(xyz)
    assert client.post("/api/watch/load",
                       json={"path": str(traj)}).status_code == 200, (
        "the trajectory file must still load -- refusing every .xyz would "
        "satisfy the assertion above and break the viewer")
