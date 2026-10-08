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
