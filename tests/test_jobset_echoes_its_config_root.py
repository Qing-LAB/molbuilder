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
    assert line.endswith("(not found -- defaults in effect)")


def test_the_groups_own_help_has_no_config_line():
    """The GROUP's own ``--help`` carries no config line.  A verb's
    ``--help`` does: Click runs the group's callback before it parses the
    verb, whose eager ``--help`` fires only then (`_echo_config_root`)."""
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, ["--help"])
    assert "molbuilder.json:" not in r.output


def test_the_file_is_named_and_marked_found(tmp_path):
    root = tmp_path / "config-root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "molbuilder.json").write_text("{}")
    line = _first_line(["machines"])
    assert str(root / "molbuilder.json") in line
    assert line.endswith("(found)")


# `test_a_file_in_the_working_directory_is_not_what_it_names` retired 2026-10-02 (W54): a cwd file not being read is `test_config_warnings.py`'s to say, once.


# `test_the_line_names_the_same_file_config_provenance_would` retired 2026-10-02 (W54): it repeated `test_the_file_is_named_and_marked_found` -- both read `machine_config_path`.


