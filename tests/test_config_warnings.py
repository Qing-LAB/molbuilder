"""What the machine config says about itself: which file, and is it safe.

Two warnings, both on the way IN, both phrased in exactly one place so every
surface says the same thing -- `configuration.md` §§ 2.1a and 2.1b.

A working-directory `molbuilder.json` is honoured and SAID OUT LOUD.

`configuration.md` § 2.1a.  User, 2026-08-31:

    *"I had instances where information are saved in two places and I did not
    realize which one was the effective one … I prefer consistency rather than
    all based on implicit rules."*

The machine scope has ONE location, the per-user config directory. A
working-directory file is **not read at all** — so the danger inverts: it used
to win silently, and now it loses silently. A person editing one would watch
their changes do nothing, which is why its presence is still said out loud.

It is a warning and never a refusal, in both directions: an unread file does not
stop the program, and a loose mode does not either.
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

    def test_an_unread_cwd_file_is_named_and_so_is_the_one_location(
            self, isolated):
        work, home = isolated
        (work / "molbuilder.json").write_text("{}")
        msg = _shadow()
        assert msg is not None
        assert "NOT READ" in msg, (
            "the danger inverted when the cwd step was deleted: this file "
            "used to win silently and now loses silently, and someone editing "
            "it would watch their changes do nothing")
        assert str(work / "molbuilder.json") in msg
        assert str(home) in msg, (
            "'move it' is not actionable without the destination")

    def test_it_says_when_the_one_location_is_still_empty(self, isolated):
        """Otherwise "move it there" reads as though a file already exists to
        merge with, and the reader hesitates."""
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        assert "no file there yet" in _shadow()

    def test_it_points_at_the_project_scope_for_per_directory_settings(
            self, isolated):
        """Telling someone to move a file without saying what replaces it
        leaves the need that put it there unmet.  The project scope is the
        answer, and it merges."""
        work, _home = isolated
        (work / "molbuilder.json").write_text("{}")
        assert ".molbuilder.json" in _shadow()


class TestItIsAWarningAndNotARefusal:

    def test_the_cwd_file_is_not_read(self, isolated):
        from molbuilder.runtime_config import machine_config_path
        work, home = isolated
        (work / "molbuilder.json").write_text('{"execution": {"mode": "local"}}')
        path, via = machine_config_path()
        assert path == home.resolve(), (
            "one location -- and it is not the working directory")
        assert via == "config-dir"

    def test_and_its_contents_do_not_leak_in(self, isolated):
        """The sharp end: not merely 'a different path' but 'that file's
        settings are not applied'."""
        from molbuilder.runtime_config import read_config
        work, _home = isolated
        (work / "molbuilder.json").write_text('{"execution": {"mode": "local"}}')
        assert read_config() == {}


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
        work, home = isolated
        (work / "molbuilder.json").write_text("{}")
        _cli._echo_config_root()
        out = capsys.readouterr()
        assert "NOT READ" in out.err, (
            "the warning goes to stderr so it reaches a person without "
            "entering piped output")
        assert str(work / "molbuilder.json") in out.err, (
            "and names the stray file, since 'somewhere in the working "
            "directory' is not something you can go and delete")
        assert str(home) in out.out, (
            "while the banner on stdout names the file actually read -- which "
            "is now the one location, not the one in the working directory")


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
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        home.write_text("{}")
        home.chmod(0o600)
        assert _mode_warning() is None, "the correct case must stay quiet"

    def test_no_file_says_nothing(self, isolated):
        assert _mode_warning() is None, (
            "there is nothing to say about a file nobody wrote")

    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o666, 0o604])
    def test_any_reach_past_the_owner_is_named(self, isolated, mode):
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
        f.write_text("{}")
        f.chmod(mode)
        msg = _mode_warning()
        assert msg is not None, f"mode {mode:04o} reaches past the owner"
        assert f"{mode:04o}" in msg, "say what the mode IS, not just that it is wrong"
        assert str(f) in msg

    def test_it_names_the_exact_command_to_fix_it(self, isolated):
        """A warning whose remedy the reader has to work out is a nag."""
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
        f.write_text("{}")
        f.chmod(0o644)
        assert f"chmod 0600 {f}" in _mode_warning()

    def test_group_only_and_world_readable_read_differently(self, isolated):
        """`0640` and `0644` are not the same exposure, and a message that
        called both "everyone" would overstate one and understate nothing."""
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
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
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
        f.write_text("{}")
        f.chmod(0o644)
        msg = _mode_warning()
        assert "credential" in msg or "private-key" in msg
        assert "2.1b" in msg, "and where the rule lives"


class TestItIsAWarningAndNotARefusalEither:

    def test_a_loose_file_is_still_read(self, isolated):
        from molbuilder.runtime_config import machine_config_path
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
        f.write_text("{}")
        f.chmod(0o644)
        assert machine_config_path()[0] == f.resolve(), (
            "refusing would lock a person out over something they can fix in "
            "one command -- which the message names")


class TestBothWarningsTravelTogether:

    def test_provenance_carries_the_mode_warning_too(self, isolated):
        from molbuilder.runtime_config import config_provenance
        _work, home = isolated
        home.parent.mkdir(parents=True, exist_ok=True)
        f = home
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
        home.chmod(0o644)
        (work / "molbuilder.json").write_text("{}")
        _cli._echo_config_root()
        err = capsys.readouterr().err
        assert "NOT READ" in err, "the stray-file warning"
        assert "chmod 0600" in err, "and the mode warning, about the real one"
