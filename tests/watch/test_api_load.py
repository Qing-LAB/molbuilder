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


# --------------------------------------------------------------------- #
#  JSON path mode (live-watch)                                          #
# --------------------------------------------------------------------- #


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 8 tests here loaded a SIESTA output, a progress log or a PySCF
# product typed by hand, or a copy of the measured relaxation outside its folder (`process/testing.md` § 6).


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
    assert ct.get("max_scf_iter") == 500
    # the SCF's own criteria beside them, per phase (`web/trajectory.md` § 3)
    crit = traj.runtime_info["scf_criteria"]["periodic"]
    assert crit["dDmax"]["tolerance"] == 1e-4


# --------------------------------------------------------------------- #
#  Directory mode (job-layout v1)                                       #
#                                                                       #
#  The loader resolves a directory path to a single file through the   #
#  run door (`model/parse.md` § 5.1-§ 5.2, job-contracts.md § 2.4).     #
#  These tests pin the route's half of it, so a regression at the      #
#  protocol boundary fails here rather than as "load failed".          #
# --------------------------------------------------------------------- #


# `test_load_directory_picks_molwatch_log_first` RETIRED 2026-09-25.  It
# pinned "the protocol prefers .molwatch.log", which `model/parse.md` § 5.5
# withdrew on 2026-09-24 (`0f914577`): SIESTA never writes into the log prep
# seeds -- a 613-byte `initial_preview` beside a 34 KB `.out` -- so offered
# first, the seed outranked the run's own result.  An optimization's engine
# output is what opens now (`runfiles.result_roles` names it first), pinned on
# a measured run of ours:
# `tests/test_results_blueprint.py::TestTheContractEndpoint::test_a_measured_run_of_ours_is_answered_by_the_run_door`
# -- whose folder holds no seeded log, so the order against one is not pinned
# there (plan W56 unit 4 re-measures it).


def test_load_directory_empty_returns_chain_error(client, tmp_path):
    """An empty directory returns a 404 whose error message names the
    discovery chain so the user can see what was tried."""
    r = client.post("/api/watch/load", json={"path": str(tmp_path)})
    body = r.get_json()
    assert r.status_code == 404
    assert body["ok"] is False
    # The trail cites the rule it followed (`model/parse.md` § 5.2) and says
    # what it found: a folder no calculation claims, holding no file a run
    # of ours writes.  *(It cited the retired deck-label search's roles,
    # `*.fdf` among them, until 2026-10-04, plan B11.)*
    assert "docs/model/parse.md" in body["error"]
    assert "no file here is one a run of ours writes" in body["error"]


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
