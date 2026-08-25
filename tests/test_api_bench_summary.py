"""``GET /api/bench/summary`` — the sweep, composed for the Results tab.

Contract: ``docs/web/bench-summary.md``.  The composition itself is
``summarize.sweep_view`` and is tested against a real prepped sweep in
``tests/test_prep_bench_fold.py``; what is tested HERE is the three things
the route owns and the verb does not:

  * which paths may be read (the picker's fence — the same one
    ``results.py`` imports, so a hole here is a hole everywhere),
  * the HTTP failure shapes a person can actually hit by clicking a file
    in the picker,
  * that the composed answer survives JSON.
"""
from __future__ import annotations

import json

import pytest

from molbuilder import diagnostics

pytest.importorskip("flask")


def _set_picker_root(monkeypatch, tmp_path):
    """Make ``tmp_path`` the only allowed picker root, so the endpoint's
    ``_resolve_within_roots`` accepts paths inside it and nothing else."""
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from molbuilder.web.app import create_app
    _set_picker_root(monkeypatch, tmp_path)
    app = create_app(config={})
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------- #
#  The fence                                                            #
# --------------------------------------------------------------------- #

def test_a_readable_sweep_outside_the_roots_is_still_refused(
        client, tmp_path_factory):
    """The file is a PERFECTLY GOOD job-set — it is simply not somewhere
    this server may read from.

    Pointing this at ``/etc/passwd`` instead would prove nothing: that
    path answers 400 whether the fence holds or not, because it fails to
    parse a moment later.  A test that passes for the wrong reason is how
    a removed fence goes unnoticed, so the bait here has to be something
    the endpoint would happily serve if it ever got that far.
    """
    from molbuilder.jobset.model import Job, JobSet, Resources
    # NB: a neutral name.  Calling it "outside-the-roots" put the
    # word "root" into the resolved path, so the assertion below
    # matched the PATH rather than the refusal and passed even with
    # the fence removed.
    outside = tmp_path_factory.mktemp("elsewhere")
    p = outside / "job-set.json"
    JobSet(name="s", engine="siesta", kind="sweep",
           jobs=[Job(name="A", script="JOB-A.fdf",
                     resources=Resources(mpi_np=1))]).write(p)

    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    # and refused BY THE FENCE, in its own words -- not by something
    # downstream failing to make sense of a file it should never have read
    assert "outside every configured root" in body["error"], body["error"]


def test_a_traversal_is_refused_by_its_own_name(client, tmp_path):
    r = client.get(f"/api/bench/summary?path={tmp_path}/../../etc/passwd")
    assert r.status_code == 400
    assert ".." in r.get_json()["error"]


def test_a_missing_path_argument_is_refused(client):
    r = client.get("/api/bench/summary")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


# --------------------------------------------------------------------- #
#  What a person can hit by clicking the wrong file                     #
# --------------------------------------------------------------------- #

def test_a_file_that_is_not_there_is_a_404(client, tmp_path):
    r = client.get(f"/api/bench/summary?path={tmp_path}/nope/job-set.json")
    assert r.status_code == 404


def test_a_file_that_is_not_a_job_set_is_a_400_not_a_500(client, tmp_path):
    """The picker lists whatever is on disk, so this is reachable by
    clicking -- it must read as a refusal, never as a crash."""
    p = tmp_path / "job-set.json"
    p.write_text('{"not": "a job set"}')
    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 400, r.get_json()
    assert r.get_json()["ok"] is False


def test_a_plain_run_job_set_is_refused_by_kind(client, tmp_path):
    """A run's plan is not a sweep: there is no comparison to draw, and
    saying so is better than drawing a chart of one column."""
    from molbuilder.jobset.model import JobSet
    p = tmp_path / "job-set.json"
    JobSet(name="calc", engine="siesta", kind="run", jobs=[]).write(p)
    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 400
    assert "not a sweep" in r.get_json()["error"]


def test_a_sweep_whose_calculation_cannot_be_found_is_a_400(client, tmp_path):
    """A job-set.json on its own, with no trial directories anywhere above
    it: the bundle resolver must refuse rather than compose an empty sweep
    that looks like one which has not started."""
    from molbuilder.jobset.model import Job, JobSet, Resources
    p = tmp_path / "job-set.json"
    JobSet(name="s", engine="siesta", kind="sweep",
           jobs=[Job(name="A", script="JOB-A.fdf",
                     resources=Resources(mpi_np=1))]).write(p)
    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


# --------------------------------------------------------------------- #
#  The composed answer, over the wire                                   #
# --------------------------------------------------------------------- #

def test_the_composed_sweep_survives_json(client, tmp_path, monkeypatch):
    """The verb's own behaviour is covered in test_prep_bench_fold; this
    pins that the route returns it whole and JSON-clean."""
    import molbuilder.jobset.summarize as S

    fake = {
        "name": "calc", "engine": "siesta", "kind": "sweep",
        "complete": False, "generated_at": "2026-08-25T00:00:00Z",
        "environment": {}, "system": {}, "choice": {"label": "A"},
        "varied": ["ranks"], "n_trials": 1, "n_done": 0,
        "trials": [{"label": "A", "state": "running", "artifacts": "unknown",
                    "s_per_iter": None, "point": {"ranks": 4}, "knobs": {},
                    "effective": {}, "mismatch": {}, "metrics": {},
                    "bound": None, "detail": "", "dir": "bench-A",
                    "attempts": []}],
    }
    from molbuilder.jobset.model import Job, JobSet, Resources
    p = tmp_path / "job-set.json"
    JobSet(name="s", engine="siesta", kind="sweep",
           jobs=[Job(name="A", script="JOB-A.fdf",
                     resources=Resources(mpi_np=1))]).write(p)
    monkeypatch.setattr(S, "bundle_for_sweep_file", lambda js, path: tmp_path)
    monkeypatch.setattr(S, "sweep_view", lambda js, bundle: dict(fake))

    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert body["choice"] == {"label": "A"}
    assert body["varied"] == ["ranks"]
    assert body["trials"][0]["state"] == "running"
    # and it really is JSON -- no dataclasses, no Paths
    json.dumps(body)
