"""Every ``jobset`` verb opens by naming its ``molbuilder.json`` (user,
2026-08-23): *"this gives the user some root information of the starting
point of config information."*  A person who hits the
``script_generation.activation`` refusal (a real terminal transcript, the
same day) should not have to work out which file to edit -- the first line
of output says so before anything else runs.

**That question had three candidate answers when this was written and now has
one** (`configuration.md` § 2.1a): the machine scope lives in the config
directory, and a `./molbuilder.json` is not read.  The line matters more
rather than less for it -- the location is no longer the directory you are
standing in, so it is no longer something a person can infer.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _a_config_root_of_its_own(tmp_path, monkeypatch):
    """A root nothing has written to, so "not found" is this test's answer.

    The first draft of this file had no isolation and reported the repo's
    real, gitignored dev config -- and `conftest`'s blanket guard only clears
    the override, leaving the XDG fallback, so the directory is named here.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "config-root"))


def _first_line(args):
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, args)
    assert r.exit_code == 0, r.output
    return r.output.splitlines()[0]


def test_a_real_verb_opens_with_the_config_root():
    line = _first_line(["machines"])
    assert line.startswith("molbuilder.json: ")
    assert line.endswith("(not found -- defaults in effect, via config-dir)")


def test_help_stays_clean_no_config_line():
    """The line is root information for a REAL run, not help-text noise."""
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, ["--help"])
    assert "molbuilder.json:" not in r.output


def test_the_file_is_named_and_marked_found(tmp_path):
    root = tmp_path / "config-root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "molbuilder.json").write_text("{}")
    line = _first_line(["machines"])
    assert str(root / "molbuilder.json") in line
    assert "(found, via config-dir)" in line


def test_a_file_in_the_working_directory_is_not_what_it_names(tmp_path,
                                                              monkeypatch):
    """The banner's job inverted on 2026-08-31 and this is the half that
    matters now.

    It used to save a person from editing the wrong one of three files.  It
    now saves them from editing a file that is not read at all -- so the line
    must name the config directory even when there is a `molbuilder.json`
    sitting right where the command was run.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text('{"execution": {"mode": "local"}}')
    line = _first_line(["machines"])
    assert str(tmp_path / "molbuilder.json") not in line
    assert str(tmp_path / "config-root") in line


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
