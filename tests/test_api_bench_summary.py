"""``GET /api/bench/summary`` — the sweep, composed for the Results tab.

Contract: ``docs/web/bench-summary.md``.  The composition itself is
``summarize.sweep_view`` and is tested against a real prepared sweep in
``tests/test_prep_bench_fold.py``; what is tested HERE is what the route
owns and the verb does not:

  * which paths may be read (the picker's fence — the same one
    ``results.py`` imports, so a hole here is a hole everywhere),
  * the HTTP failure shapes a person can actually hit by clicking a file
    in the picker.
"""
from __future__ import annotations


import pytest

pytest.importorskip("flask")


@pytest.fixture
def client(isolated_projects_root):
    """The app on the test's own projects tree, the one door pointed at it
    (`isolated_projects_root`): the endpoint's ``_resolve_within_roots``
    accepts paths inside it and nothing else."""
    from molbuilder.web.app import create_app
    app = create_app(config={})
    app.config.update(TESTING=True)
    return app.test_client()


# --------------------------------------------------------------------- #
#  The fence                                                            #
# --------------------------------------------------------------------- #


def test_a_missing_path_argument_is_refused(client):
    r = client.get("/api/bench/summary")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


# --------------------------------------------------------------------- #
#  What a person can hit by clicking the wrong file                     #
# --------------------------------------------------------------------- #

def test_a_file_that_is_not_there_is_a_404(client, isolated_projects_root):
    r = client.get(
        f"/api/bench/summary?path={isolated_projects_root}/nope/job-set.json")
    assert r.status_code == 404


def test_a_file_that_is_not_a_job_set_is_a_400_not_a_500(
        client, isolated_projects_root):
    """The picker lists whatever is on disk, so this is reachable by
    clicking -- it must read as a refusal, never as a crash."""
    p = isolated_projects_root / "job-set.json"
    p.write_text('{"not": "a job set"}')
    r = client.get(f"/api/bench/summary?path={p}")
    assert r.status_code == 400, r.get_json()
    assert r.get_json()["ok"] is False
