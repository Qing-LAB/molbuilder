"""What the machine config says about itself: which file, and is it safe.

Two warnings, both on the way IN, both phrased in exactly one place so every
surface says the same thing -- `configuration.md` §§ 2.1a and 2.1b.

A working-directory `molbuilder.json` is honoured and SAID OUT LOUD.

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


# ---------------------------------------------------------------------------
# § 2.1b — it holds secrets, so its mode is checked and not merely written
# ---------------------------------------------------------------------------

def _mode_warning():
    from molbuilder.runtime_config import machine_config_mode_warning
    return machine_config_mode_warning()


class TestTheModeIsCheckedOnTheWayIn:
    """Writing it tightly was already handled; a file that ARRIVES loose --
    copied, restored, unpacked, or made by an editor -- never passes through
    the careful writer, so a `0644` config was `0644` in silence."""

    def test_a_tight_file_says_nothing(self, isolated):
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o600)
        assert _mode_warning() is None, "the correct case must stay quiet"

    def test_no_file_says_nothing(self, isolated):
        assert _mode_warning() is None, (
            "there is nothing to say about a file nobody wrote")

    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o666, 0o604])
    def test_any_reach_past_the_owner_is_named(self, isolated, mode):
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(mode)
        msg = _mode_warning()
        assert msg is not None, f"mode {mode:04o} reaches past the owner"
        assert f"{mode:04o}" in msg, "say what the mode IS, not just that it is wrong"
        assert str(f) in msg

    def test_it_names_the_exact_command_to_fix_it(self, isolated):
        """A warning whose remedy the reader has to work out is a nag."""
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o644)
        assert f"chmod 0600 {f}" in _mode_warning()

    def test_group_only_and_world_readable_read_differently(self, isolated):
        """`0640` and `0644` are not the same exposure, and a message that
        called both "everyone" would overstate one and understate nothing."""
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o640)
        group_only = _mode_warning()
        f.chmod(0o644)
        world = _mode_warning()
        assert "everyone" not in group_only
        assert "everyone" in world

    def test_it_says_why_the_mode_matters(self, isolated):
        """Not decoration: without the reason this reads as pedantry about a
        config file, and the reason is that it carries key paths and provider
        credentials."""
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o644)
        msg = _mode_warning()
        assert "credential" in msg or "private-key" in msg
        assert "2.1b" in msg, "and where the rule lives"


class TestItIsAWarningAndNotARefusalEither:

    def test_a_loose_file_is_still_read(self, isolated):
        from molbuilder.runtime_config import machine_config_path
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o644)
        assert machine_config_path()[0] == f.resolve(), (
            "refusing would lock a person out over something they can fix in "
            "one command -- which the message names")


class TestBothWarningsTravelTogether:

    def test_provenance_carries_the_mode_warning_too(self, isolated):
        from molbuilder.runtime_config import config_provenance
        work, _home = isolated
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o644)
        assert config_provenance()["mode_warning"] == _mode_warning()

    def test_the_banner_prints_both_when_both_apply(self, isolated, capsys):
        """A cwd file that is also loose has two things wrong with it, and
        hearing one of them is how the other survives."""
        from molbuilder.jobset import _cli
        work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        home.write_text("{}")
        f = work / "molbuilder.json"
        f.write_text("{}")
        f.chmod(0o644)
        _cli._echo_config_root()
        err = capsys.readouterr().err
        assert "IGNORED" in err, "the shadow warning"
        assert "chmod 0600" in err, "and the mode warning"
