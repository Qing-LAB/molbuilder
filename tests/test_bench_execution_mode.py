"""``get_execution`` — the run-vs-submit policy key, read and validated.

The launch CONTRACT itself — flag, else ``execution.mode``, else a refusal,
never a derivation from the detected scheduler (`running-a-job.md` § 5.4) —
is pinned at its live door, tests/test_jobset.py's submit tests.
"""

import json

import pytest

from molbuilder.runtime_config import (PROJECT_CONFIG_FILENAME,
                                       RuntimeConfigError, get_execution)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Isolated cwd + $HOME + XDG for every test in this file.

    ``get_execution`` deep-merges the server-wide scope, which is CWD-first
    with an XDG fallback — so without this the verdicts here depended on the
    developer's repo-root ``molbuilder.json`` (found 2026-08-12; the same
    class as the inert-fixture bug in test_prep_calculation.py)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()


# ---- get_execution: read, default, validate, survive _normalise ------ #

def _write_project(tmp_path, cfg):
    (tmp_path / PROJECT_CONFIG_FILENAME).write_text(json.dumps(cfg))


def test_get_execution_reads_mode_and_submit_via(tmp_path):
    _write_project(tmp_path, {"execution": {"mode": "submit",
                                            "submit_via": "slurm"}})
    out = get_execution(project_dir=tmp_path)
    assert out == {"mode": "submit", "submit_via": "slurm", "domain": None}


def test_get_execution_absent_is_none_mode(tmp_path):
    _write_project(tmp_path, {})
    out = get_execution(project_dir=tmp_path)
    assert out["mode"] is None
    assert out["submit_via"] == "slurm"          # default backend


def test_get_execution_survives_normalise_allowlist(tmp_path):
    # The key must NOT be dropped by _normalise's top-level allowlist
    # (the bug found while wiring this up): a sibling known key present.
    _write_project(tmp_path, {"scheduler": {"kind": "slurm"},
                              "execution": {"mode": "direct"}})
    assert get_execution(project_dir=tmp_path)["mode"] == "direct"


def test_get_execution_bad_mode_raises(tmp_path):
    _write_project(tmp_path, {"execution": {"mode": "fling"}})
    with pytest.raises(RuntimeConfigError, match="execution.mode"):
        get_execution(project_dir=tmp_path)


def test_get_execution_non_object_raises(tmp_path):
    _write_project(tmp_path, {"execution": "submit"})
    with pytest.raises(RuntimeConfigError, match="'execution' must be an"):
        get_execution(project_dir=tmp_path)


# ---- resolve_mode / resolve_launch_adapter --------------------------- #


# --------------------------------------------------------------------- #
#  Everything below this file's get_execution block was RETIRED          #
#  2026-08-12 (u5): `resolve_mode`/`resolve_launch_adapter` died with    #
#  the shipped-bundle launchers (their callers), `bake_run_bench` /      #
#  `render_bench_plan` / `run_prep_bench` with the bundle itself.  The   #
#  mode CONTRACT -- flag, else execution.mode, else a refusal, never a   #
#  derivation from the detected scheduler (running-a-job.md § 5.4) --    #
#  is pinned where it lives now: tests/test_jobset.py's submit tests.    #
# --------------------------------------------------------------------- #
