"""The Results viewers follow THE RUN, and the server says how it is doing --
`web/results.md` § 4.1, `web/web-api.md` (the two watch routes and
`/api/spectra/load`).

The server half of the rule: every answer whose file has nothing new
carries ``run: {state, detail, live}`` -- the one door's answer
(`parse.dirs.run_answer`, built on `runrecord.ending`) for the run the file
belongs to -- and when the run is no longer live the file is read again if
it changed, so the data a viewer stops on is the file's last.  The browser
half -- what the viewer does with it -- is
`test_trajectory_settle_post_load_js.py`, `test_trajectory_transition_js.py`
and the spectra follow in `test_inspector_registry_e2e.py`.

API-level, through the Flask client: a viewer's poll is not something the
`jobset` road makes, and the run's files are the measured relaxation's
(`support.road.a_finished_run`).

PREVENTS (plan W38 M2f): a run killed mid-step followed until the page
closed, because each viewer decided by the file's own ending; and a run's
end read after the file's last read, which left the viewer on a short
file.
"""
from __future__ import annotations

import pytest

from support.road import a_finished_run


@pytest.fixture
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """The picker roots at ``tmp_path``, which the routes read within."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    return tmp_path


def _run(where, *, monitor_ended=False, **ran):
    """A coarse stage's run in ``where``, laid as it ended (`a_finished_run`)
    -- its deck beside it, which names the run its files belong to -- and,
    when ``monitor_ended``, the monitor's closing record: the process seen
    to go.  Returns the output."""
    from molbuilder.parse.dirs.job import MONITOR_ENDED
    where.mkdir(parents=True, exist_ok=True)
    (where / "H2_01_coarse.fdf").write_text("SystemLabel H2\n")
    a_finished_run(where, **ran)
    if monitor_ended:
        (where / "H2_01_coarse-run0.monitor.log").write_text(
            f"[12:00:00] [INFO ] {MONITOR_ENDED}\n")
    return where / "H2_01_coarse-run0.out"


#: (case, how the run is laid, its state, live)
CASES = [
    ("a run whose output is still being written is live",
     dict(concluded=False, output="cut"), "running", True),
    ("an output that ended with no conclusion is live: the job has not concluded",
     dict(concluded=False), "running", True),
    ("one that concluded with exit code 0 is finished",
     dict(), "finished", False),
    ("one killed mid-step -- its monitor saw it go -- is failed",
     dict(concluded=False, output="cut", monitor_ended=True), "failed", False),
]


@pytest.mark.parametrize("case, ran, state, live", CASES,
                         ids=[c[0] for c in CASES])
def test_the_load_says_how_the_run_is_doing(client, roots, case, ran, state,
                                            live):
    out = _run(roots / "run-0", **ran)
    d = client.post("/api/watch/load", json={"path": str(out)}).get_json()
    assert d["ok"] is True, d
    assert (d["run"]["state"], d["run"]["live"]) == (state, live), d["run"]


def test_a_quiet_poll_brings_the_end_of_a_run_killed_mid_step(client, roots):
    """THE CASE M2f EXISTS FOR.  The output never changes again and states
    nothing; the poll that finds it unchanged brings how the run is doing --
    failed, its monitor saw it go.  A poll with nothing new and no answer
    was all a viewer had until 2026-10-03, so it polled until the page
    closed."""
    out = _run(roots / "run-0", concluded=False, output="cut")
    loaded = client.post("/api/watch/load",
                         json={"path": str(out)}).get_json()
    assert loaded["run"]["live"] is True, loaded["run"]
    from molbuilder.parse.dirs.job import MONITOR_ENDED
    (out.parent / "H2_01_coarse-run0.monitor.log").write_text(
        f"[12:00:00] [INFO ] {MONITOR_ENDED}\n")
    d = client.get(f"/api/watch/data?mtime={loaded['mtime']}").get_json()
    assert d["changed"] is False, d
    assert (d["run"]["state"], d["run"]["live"]) == ("failed", False), d


def test_a_poll_with_new_content_leaves_the_run_out(client, roots):
    """The run was writing: how it is doing is asked when the file is quiet,
    and the field left out keeps what the viewer holds."""
    out = _run(roots / "run-0", concluded=False, output="cut")
    loaded = client.post("/api/watch/load",
                         json={"path": str(out)}).get_json()
    import os
    import time
    out.write_text(out.read_text() + "\nscf:  2  -100.0\n")
    later = time.time() + 5
    os.utime(out, (later, later))
    d = client.get(f"/api/watch/data?mtime={loaded['mtime']}").get_json()
    assert d["changed"] is True, d
    assert "run" not in d, d


def test_the_runs_end_is_read_before_the_files_last_read(client, roots,
                                                         monkeypatch):
    """A run's final write lands between the poll's look at the file and its
    question about the run, and the run concludes: the answer carries the
    file's LAST content with the run's end -- never the end beside a short
    file, which the viewer would then stop on.

    MUTATION THIS MUST FAIL AGAINST: the file not read again once the run is
    no longer live."""
    out = _run(roots / "run-0", concluded=False, output="cut")
    loaded = client.post("/api/watch/load",
                         json={"path": str(out)}).get_json()
    assert loaded["data"]["run_state"] == "running", loaded["data"]["run_state"]

    from molbuilder.web.blueprints import watch
    real = watch.run_answer

    def the_run_ends_now(path):
        # the engine's last lines, then the wrapper's conclusion
        import os
        import time
        a_finished_run(out.parent)
        later = time.time() + 5
        os.utime(out, (later, later))
        return real(path)

    monkeypatch.setattr(watch, "run_answer", the_run_ends_now)
    d = client.get(f"/api/watch/data?mtime={loaded['mtime']}").get_json()
    assert d["run"]["state"] == "finished", d.get("run")
    assert d["changed"] is True and d["data"]["run_state"] == "ended", (
        "the run's end was sent beside the file's earlier content")


def test_an_upload_belongs_to_no_run(client, roots):
    """Nothing to follow, and said in the field the path load answers."""
    import io
    out = _run(roots / "run-0")
    d = client.post("/api/watch/load", data={
        "file": (io.BytesIO(out.read_bytes()), "H2_01_coarse-run0.out")},
        content_type="multipart/form-data").get_json()
    assert d["ok"] is True, d
    assert d["run"] is None


@pytest.mark.parametrize("ended, state, live", [
    (False, "queued", True),
    (True, "finished", False),
], ids=["launched and silent: live", "concluded: finished"])
def test_the_spectra_load_says_how_the_run_is_doing(client, roots, ended,
                                                    state, live):
    """The spectra viewer follows by the same answer: a result whose run is
    launched and not over is live; one whose run concluded is not."""
    from molbuilder.runrecord import write_launch
    from molbuilder.sidecars.spectra import dump_spectra_json
    from tests.spectra._helpers import _make_results
    where = roots / "freq"
    where.mkdir()
    dest = where / "job.spectra.json"
    dump_spectra_json(_make_results(complete=ended), dest)
    write_launch(where, mode="direct", command=["bash", "job.run.sh"])
    if ended:
        (where / "job-run0.concluded").write_text("rc=0 at then\n")
    d = client.post("/api/spectra/load", json={"path": str(dest)}).get_json()
    assert d["ok"] is True, d
    assert (d["run"]["state"], d["run"]["live"]) == (state, live), d["run"]
