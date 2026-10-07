"""One directory, one function — the callers cannot disagree about it.

The directory is ``molbuilder/config_dir.py``'s.  The tests below move the variables and ask
every door where its file went; a module that spells the rule a second time
is review's to find (`process/code-audit.md` § 1c), because an agreeing copy
gives the same answer as the door until the day the rule moves.
"""
from __future__ import annotations

import pytest


def test_the_directory_is_read_at_call_time_not_captured_at_import(
        monkeypatch, tmp_path):
    """Captured at import, a test (or an operator) could move the variable
    and have half the callers keep the old answer -- which is precisely how
    the suite came to read the developer's real `~/.config/molbuilder`."""
    from molbuilder.config_dir import config_dir
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "a"))
    first = config_dir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "b"))
    assert config_dir() != first


# ---------------------------------------------------------------------------
# The override — `configuration.md` § 2.1c.
# ---------------------------------------------------------------------------

class TestTheRootCanBeNamedOutright:

    def test_it_is_used_exactly_as_given(self, monkeypatch, tmp_path):
        """No ``molbuilder`` component is appended, and that asymmetry with
        ``XDG_CONFIG_HOME`` is the design.

        ``XDG_CONFIG_HOME`` names a root shared by every application, so ours
        must add its own name under it.  ``MOLBUILDER_CONFIG_DIR`` names OUR
        directory; appending to it would put the files somewhere the person
        did not ask for.
        """
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "here"))
        assert config_dir() == tmp_path / "here"

    def test_it_beats_xdg_and_does_not_fall_back_past_it(
            self, monkeypatch, tmp_path):
        """An override, not a search step.

        A fallback here would recreate exactly the shadowing
        `configuration.md` § 2.1a exists to warn about: one setting, two
        files, one of them silently winning.
        """
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "named"))
        assert config_dir() == tmp_path / "named"

    def test_empty_is_not_set(self, monkeypatch, tmp_path):
        """``MOLBUILDER_CONFIG_DIR=`` is how a shell unsets a variable it
        cannot unset; treating it as a root would put the config at the
        filesystem root."""
        from molbuilder.config_dir import config_dir, CONFIG_DIR_ENV
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv(CONFIG_DIR_ENV, "")
        assert config_dir() == tmp_path / "xdg" / "molbuilder"


# ---------------------------------------------------------------------------
# Operational state
# ---------------------------------------------------------------------------

class TestOperationalStateFollowsXdg:
    """`$XDG_STATE_HOME` entered the Base Directory spec in 0.8 for state that
    persists across restarts but is not portable enough for `$XDG_DATA_HOME`,
    and the spec names LOGS first.  `$XDG_RUNTIME_DIR` is the one for pidfiles.
    """

    def test_state_follows_its_variable(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import state_dir
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert state_dir() == tmp_path / "s" / "molbuilder"

    def test_state_defaults_to_the_spec_location(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import state_dir
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert state_dir() == tmp_path / ".local" / "state" / "molbuilder"

    def test_runtime_prefers_its_own_variable(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import runtime_dir
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "r"))
        assert runtime_dir() == tmp_path / "r" / "molbuilder"

    def test_runtime_falls_back_to_state_and_not_to_a_temp_dir(
            self, monkeypatch, tmp_path):
        """XDG_RUNTIME_DIR is cleared when the session ends, and is not always
        set (cron, a detached ssh, some containers).  A supervisor's pidfile
        that vanished under it would leave a running server nothing can find,
        so the fallback persists."""
        from molbuilder.config_dir import runtime_dir, state_dir
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert runtime_dir() == state_dir() / "run"

    def test_state_is_not_under_the_config_root(self, monkeypatch, tmp_path):
        """Configuration is edited and backed up; logs grow and are deleted.
        A person who wants them elsewhere moves them with `XDG_STATE_HOME`
        (`configuration.md` § 2.1d)."""
        from molbuilder.config_dir import config_dir, state_dir
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        assert config_dir() not in state_dir().parents
        assert state_dir() != config_dir()


class TestOperationalStateFollowsTheVariablesOnly:
    """`$XDG_STATE_HOME` and `$XDG_RUNTIME_DIR` alone move these
    directories: `serve_daemon` is L1, the config reader is L2, and a
    supervisor must be able to write its log before any config is read.
    """

    @pytest.fixture()
    def cfg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        (tmp_path / "cfg").mkdir(parents=True, exist_ok=True)
        return tmp_path

    def _write(self, cfg, obj):
        import json
        (cfg / "cfg" / "molbuilder.json").write_text(json.dumps(obj))

    def test_the_variables_move_them(self, cfg):
        from molbuilder.config_dir import logs_dir, reports_dir, runtime_dir
        assert logs_dir() == cfg / "state" / "molbuilder" / "logs"
        assert reports_dir() == cfg / "state" / "molbuilder" / "reports"
        assert runtime_dir() == cfg / "state" / "molbuilder" / "run"


    def test_projects_stays(self, cfg):
        """Data rather than operational state, no XDG equivalent, and
        `$MOLBUILDER_PROJECTS` is its documented override."""
        from molbuilder.runtime_config import read_config
        self._write(cfg, {"paths": {"projects": "/data/projects"}})
        assert read_config()["paths"]["projects"] == "/data/projects"


# ---------------------------------------------------------------------------
# One API
# ---------------------------------------------------------------------------

#: THE DIVISION A11 DRAWS: the module that owns a FORMAT owns its NAME, and
#: `config_dir` owns the DIRECTORY.  So a file with a format owner keeps its
#: name there and that owner exposes the path function; what lives in
#: `config_dir` is the files with no format to own.
class TestEveryFileHasADoor:

    def test_the_ported_ones_take_the_port(self, monkeypatch, tmp_path):
        from molbuilder.config_dir import serve_log, serve_pidfile
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "r"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
        assert serve_pidfile(8888).name == "serve-8888.pid"
        assert serve_log(8888).name == "serve-8888.log"

    def test_every_door_moves_with_the_one_variable(self, monkeypatch,
                                                    tmp_path):
        """The property the whole change exists for: name one directory and
        every configuration file is under it — the format owners' doors
        included, since they ask this module for the directory."""
        from molbuilder.config_dir import (client_secret, config_dir,
                                           session_key)
        from molbuilder.monitor import default_notify_path, notify_keys_path
        from molbuilder.runtime_config import machine_config_path
        from molbuilder.scheduler.record import (environments_dir,
                                                 machine_scope_path)
        root = tmp_path / "named"
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
        doors = {
            "session_key": session_key(),
            "google_client_secret": client_secret("google"),
            "machine config": machine_config_path(),
            "environment record": machine_scope_path(),
            "environments/": environments_dir(),
            "notify": default_notify_path(),
            "notify_keys": notify_keys_path(),
        }
        assert config_dir() == root
        for name, got in doors.items():
            assert root in got.parents or got == root, f"{name} -> {got}"

        # AND EVERY CREDENTIAL IS IN `secrets/`, asked of the audit rather
        # than listed here.  `placement.misplaced()` reads the same table
        # `envs doctor` prints from, so this pins the rule where it is
        # DEFINED -- a credential added to that table is covered the moment
        # it is added, and a test listing names would have to be remembered.
        #
        # Measured 2026-09-20, before that function existed: pointing
        # `session_key()` back at the config root left 222 tests green,
        # because every test and every audit row asks the same resolver and
        # moves with it.  Nothing compared the answer against the rule.
        from molbuilder.placement import misplaced
        assert misplaced() == [], (
            "a credential store resolves outside secrets/:\n"
            + "\n".join(misplaced()))


def test_a_printed_remedy_names_the_resolved_directory(monkeypatch, tmp_path):
    """I3.  `prep` refused a remote target whose record states no activation and
    told the person to *"copy the record into `~/.config/molbuilder/environments/`
    here"* -- a literal, on the machine where `MOLBUILDER_CONFIG_DIR` is the
    whole point of the variable.  Following it put the record where `prep` does
    not look, and `prep` then refused again with the same message.

    API-LEVEL, and why: the road reaches this refusal (`launch_values.toml`'s
    "...a named target too -- ... the refusal says the copy back"), but a row
    asserts fixed
    text, and what this pins is the RESOLVED directory -- a per-test root a
    row cannot spell.
    """
    import pytest as _pytest

    root = tmp_path / "named"
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
    from molbuilder.jobset.errors import PrepError
    from molbuilder.jobset.machine import require_activation
    from molbuilder.scheduler.record import Environment

    with _pytest.raises(PrepError) as excinfo:
        require_activation("sol", Environment(scheduler="slurm"))

    message = str(excinfo.value)
    assert str(root / "environments") in message, message
    assert "~/.config/molbuilder" not in message, (
        "the remedy still hard-codes the default root:\n" + message)
