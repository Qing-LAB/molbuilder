"""Every ``jobset`` verb opens by naming its ``molbuilder.json`` (user,
2026-08-23): *"this gives the user some root information of the starting
point of config information."*  A person who hits the
``script_generation.activation`` refusal (a real terminal transcript, the
same day) should not have to already know which of three candidate files to
edit -- the first line of output says so before anything else runs.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """The cwd branch of `machine_config_path` checks ``./molbuilder.json``
    literally -- proven necessary by this file's own first draft, which
    (before this fixture existed) reported the repo's real, gitignored dev
    config as this test's answer."""
    monkeypatch.chdir(tmp_path)


def _first_line(args):
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, args)
    assert r.exit_code == 0, r.output
    return r.output.splitlines()[0]


def test_a_real_verb_opens_with_the_config_root():
    line = _first_line(["machines"])
    assert line.startswith("molbuilder.json: ")
    assert line.endswith("(not found -- defaults in effect, via xdg)")


def test_help_stays_clean_no_config_line():
    """The line is root information for a REAL run, not help-text noise."""
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, ["--help"])
    assert "molbuilder.json:" not in r.output


def test_a_cwd_file_is_named_and_marked_found(tmp_path):
    (tmp_path / "molbuilder.json").write_text("{}")
    line = _first_line(["machines"])
    assert str(tmp_path / "molbuilder.json") in line
    assert "(found, via cwd)" in line


def test_the_line_names_the_same_file_config_provenance_would():
    """Two lines describing one file must not be able to name two different
    ones -- both go through `machine_config_path`, so this pins that they
    stay wired to the SAME function rather than each growing its own
    resolution."""
    from molbuilder.runtime_config import machine_config_path
    path, via = machine_config_path()
    line = _first_line(["machines"])
    assert str(path) in line
    assert f"via {via}" in line
