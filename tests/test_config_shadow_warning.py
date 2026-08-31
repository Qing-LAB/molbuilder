"""A working-directory `molbuilder.json` is honoured and SAID OUT LOUD.

`configuration.md` § 2.1a.  User, 2026-08-31:

    *"I had instances where information are saved in two places and I did not
    realize which one was the effective one … I prefer consistency rather than
    all based on implicit rules."*

The machine scope's home is the per-user config directory.  A cwd file still
wins when it exists, because § 2.1's search stops at the first hit — nothing is
merged, and until now nothing was said, so two files could hold the same setting
while one took effect.

The state worth the noise is **both files present**: that is the only one in
which a setting can be written twice and read once.  It is a warning and never a
refusal — refusing would break a machine that has such a file today, and obeying
quietly is the bug.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Fresh cwd AND fresh config home — never the real ones."""
    home = tmp_path / "xdg"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.chdir(work)
    return work, home / "molbuilder" / "molbuilder.json"


def _shadow():
    from molbuilder.runtime_config import machine_config_shadow
    return machine_config_shadow()


class TestWhenItSaysNothing:

    def test_no_cwd_file_is_the_quiet_case(self, isolated):
        assert _shadow() is None, (
            "the correct arrangement must not warn on every invocation")

    def test_a_home_file_alone_is_the_quiet_case(self, isolated):
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        home.write_text("{}")
        assert _shadow() is None, "this is where a machine config belongs"


class TestWhenItSpeaks:

    def test_a_cwd_file_is_named_and_so_is_its_home(self, isolated):
        work, home = isolated
        (work / "molbuilder.json").write_text("{}")
        msg = _shadow()
        assert msg is not None
        assert str(work / "molbuilder.json") in msg, "name the file in effect"
        assert str(home) in msg, (
            "and where it belongs -- 'move it' is not actionable without the "
            "destination")

    def test_both_present_says_the_home_one_is_ignored(self, isolated):
        """The case the user hit: written twice, read once."""
        work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        home.write_text('{"execution": {"mode": "submit"}}')
        (work / "molbuilder.json").write_text('{"execution": {"mode": "local"}}')
        msg = _shadow()
        assert "IGNORED" in msg, (
            "a reader must not have to infer that the other file lost")
        assert "not layers" in msg or "not merged" in msg or "nothing" in msg, (
            "and that nothing from it is merged in -- the search STOPS")

    def test_it_points_at_the_project_scope_for_per_directory_settings(
            self, isolated):
        """Telling someone to move a file without saying what replaces it
        leaves the need that put it there unmet.  The project scope is the
        answer, and it merges."""
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        assert ".molbuilder.json" in _shadow()


class TestItIsAWarningAndNotARefusal:

    def test_the_cwd_file_still_takes_effect(self, isolated):
        from molbuilder.runtime_config import machine_config_path
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        path, via = machine_config_path()
        assert path == (work / "molbuilder.json").resolve()
        assert via == "cwd", (
            "§ 2.1a keeps the step working; it only stops being silent")


class TestEverySurfaceSaysTheSameThing:

    def test_provenance_carries_it(self, isolated):
        from molbuilder.runtime_config import (config_provenance,
                                               machine_config_shadow)
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        assert config_provenance()["shadow"] == machine_config_shadow(), (
            "the phrasing lives in one place; a display must not re-word it")

    def test_the_jobset_banner_prints_it(self, isolated, capsys):
        from molbuilder.jobset import _cli
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        _cli._echo_config_root()
        out = capsys.readouterr()
        assert "working directory" in out.err, (
            "the banner names the resolved path on stdout; the warning says "
            "what that path is standing in front of, and goes to stderr so it "
            "reaches a person without entering piped output")
        assert str(work / "molbuilder.json") in out.out, (
            "and the banner itself still names the file, as it always did")
