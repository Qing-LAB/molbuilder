"""One directory, one function — the callers cannot disagree about it.

`configuration.md` M-4 gave ``environment.json`` one home for its FILENAME,
which "was a string literal in three modules".  The DIRECTORY that filename
sits in stayed spelled three times:

    runtime_config._machine_config_file      -> molbuilder.json
    scheduler/record.machine_scope_path      -> environment.json, environments/
    auth_setup.default_secret_dir            -> secret_key

They agreed, and two of them said so in prose -- *"Mirrors
auth_setup.default_secret_dir's convention"* and *"mirrored rather than
imported"*.  **A comment is not a mechanism.**  Fixed 2026-08-23 by
``molbuilder/config_dir.py``; this pins it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "molbuilder"


@pytest.fixture
def moved(monkeypatch, tmp_path):
    """One operator action -- set ``XDG_CONFIG_HOME`` -- moves everything."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "elsewhere"))
    return tmp_path / "elsewhere" / "molbuilder"


def _all_four():
    from molbuilder.auth_setup import default_secret_dir, secret_key_path
    from molbuilder.runtime_config import _machine_config_file
    from molbuilder.scheduler import environments_dir, machine_scope_path
    return {
        "secret_key":     secret_key_path(),
        "secret_dir":     default_secret_dir(),
        "environment":    machine_scope_path(),
        "environments":   environments_dir(),
        "molbuilder.json": _machine_config_file(),
    }


def test_every_per_user_file_sits_under_the_one_directory(moved):
    """The property that matters to an operator: move the variable, and the
    secrets, the machine record and the config all move TOGETHER."""
    for name, p in _all_four().items():
        assert moved in p.parents or p == moved, (
            f"{name} -> {p} is not under {moved}")


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


def test_no_module_spells_the_rule_a_second_time():
    """**The guard that would have caught the original drift.**

    Any module joining ``XDG_CONFIG_HOME`` to a path itself is a fourth copy
    growing.  ``config_dir.py`` is the one place allowed to read it.
    """
    offenders = []
    for py in _SRC.rglob("*.py"):
        if py.name == "config_dir.py":
            continue
        src = py.read_text(encoding="utf-8")
        if "XDG_CONFIG_HOME" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            # os.environ.get("XDG_CONFIG_HOME") / os.environ["XDG_CONFIG_HOME"]
            if isinstance(node, ast.Constant) and node.value == "XDG_CONFIG_HOME":
                offenders.append(str(py.relative_to(_SRC)))
                break
    assert not offenders, (
        "these modules read XDG_CONFIG_HOME directly instead of calling "
        f"config_dir(): {sorted(set(offenders))}")


# ---------------------------------------------------------------------------
# The override — archive/2026-09-01-config-access-plan.md § 3.1
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

    def test_it_moves_every_per_user_file_together(self, monkeypatch, tmp_path):
        """The property the variable exists for -- the same one
        ``XDG_CONFIG_HOME`` has, asserted for the new door too."""
        from molbuilder.config_dir import CONFIG_DIR_ENV
        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "one"))
        for name, path in _all_four().items():
            assert (tmp_path / "one") in path.parents \
                or path == tmp_path / "one", f"{name} -> {path}"

    def test_no_module_spells_the_override_a_second_time(self):
        """Same guard as `XDG_CONFIG_HOME`'s, for the same reason.

        Matched against the SYNTAX TREE, not the text.  A first version
        matched the name anywhere and flagged `envs/_cli.py`, whose only
        mention is a docstring explaining why its log directory moved -- the
        same trap the second-root pin fell into, and the same answer: a test
        that cannot tell a path from a sentence about one punishes the
        documentation we want.
        """
        offenders = []
        for py in _SRC.rglob("*.py"):
            if py.name == "config_dir.py":
                continue
            src = py.read_text(encoding="utf-8")
            if "MOLBUILDER_CONFIG_DIR" not in src:
                continue
            for node in ast.walk(ast.parse(src)):
                # os.environ.get("MOLBUILDER_CONFIG_DIR") / os.environ[...]
                if isinstance(node, (ast.Call, ast.Subscript)) \
                        and "MOLBUILDER_CONFIG_DIR" in {
                            n.value for n in ast.walk(node)
                            if isinstance(n, ast.Constant)
                            and isinstance(n.value, str)}:
                    offenders.append(str(py.relative_to(_SRC)))
                    break
        assert not offenders, (
            "these modules read MOLBUILDER_CONFIG_DIR directly instead of "
            f"calling config_dir(): {sorted(set(offenders))}")


# ---------------------------------------------------------------------------
# The second root — RETIRED 2026-08-31
# ---------------------------------------------------------------------------
#
# `_SECOND_ROOT_HOLDOUTS` listed the three modules that still computed
# `~/.molbuilder/...` themselves, and its own failure message said: "when the
# list empties, delete the list and this test with it: the root is gone and
# there is nothing to shrink."  Step 2 emptied it, so it is deleted rather than
# left standing as an empty allowance.
#
# What replaces it is not a smaller allow-list but a stricter question, below:
# no module may compute a per-user root AT ALL.
# ---------------------------------------------------------------------------
# Operational state — archive/2026-09-01-config-access-plan.md § 3.2
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
        A person who wants them together says so with `paths`."""
        from molbuilder.config_dir import config_dir, state_dir
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        assert config_dir() not in state_dir().parents
        assert state_dir() != config_dir()


class TestOperationalStateFollowsTheVariablesOnly:
    """`paths.logs` / `paths.run` / `paths.reports` are RETIRED.

    They existed for a day.  `$XDG_STATE_HOME` and `$XDG_RUNTIME_DIR` already
    move these directories, so the keys were a second way to say one thing --
    and being a second way is what put the answer out of reach of the layer
    that needs it: `serve_daemon` is L1, the config reader is L2, and a
    supervisor must be able to write its log before any config is read.

    Deleting them removed the inversion instead of working around it with an
    injection point (`archive/2026-09-01-config-access-plan.md` § 5.3).
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

    @pytest.mark.parametrize("key", ["logs", "run", "reports"])
    def test_the_retired_key_is_refused(self, cfg, key):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {key: "/somewhere"}})
        with pytest.raises(RuntimeConfigError, match="no longer configured"):
            read_config()

    def test_the_refusal_names_the_variable_that_replaces_it(self, cfg):
        """A refusal that does not name the replacement is a dead end."""
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"logs": "/somewhere"}})
        with pytest.raises(RuntimeConfigError) as e:
            read_config()
        assert "XDG_STATE_HOME" in str(e.value)

    def test_it_does_not_read_as_a_typo(self, cfg):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"run": "/somewhere"}})
        with pytest.raises(RuntimeConfigError) as e:
            read_config()
        assert "unknown key" not in str(e.value)

    def test_projects_stays(self, cfg):
        """Data rather than operational state, no XDG equivalent, and
        `$MOLBUILDER_PROJECTS` is its documented override."""
        from molbuilder.runtime_config import read_config
        self._write(cfg, {"paths": {"projects": "/data/projects"}})
        assert read_config()["paths"]["projects"] == "/data/projects"

    def test_a_key_nothing_reads_is_still_refused(self, cfg):
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"cache": "/tmp/x"}})
        with pytest.raises(RuntimeConfigError, match="cache"):
            read_config()
# ---------------------------------------------------------------------------
# One API — archive/2026-09-01-config-access-plan.md § 5
# ---------------------------------------------------------------------------

#: THE DIVISION A11 DRAWS: the module that owns a FORMAT owns its NAME, and
#: `config_dir` owns the DIRECTORY.  So a file with a format owner keeps its
#: name there and that owner exposes the path function; what lives in
#: `config_dir` is the files with no format to own.
#:
#: Pulling `environment.json` and `notify` into `config_dir` was tried and
#: reverted the same day — it took a name from its format owner, and
#: `test_architecture_rules`' A11 said so before any of this shipped.
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
        from molbuilder.config_dir import (config_dir, google_client_secret,
                                           session_key)
        from molbuilder.monitor import default_notify_path, notify_keys_path
        from molbuilder.runtime_config import machine_config_path
        from molbuilder.scheduler.record import (environments_dir,
                                                 machine_scope_path)
        root = tmp_path / "named"
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(root))
        doors = {
            "session_key": session_key(),
            "google_client_secret": google_client_secret(),
            "machine config": machine_config_path()[0],
            "environment record": machine_scope_path(),
            "environments/": environments_dir(),
            "notify": default_notify_path(),
            "notify_keys": notify_keys_path(),
        }
        assert config_dir() == root
        for name, got in doors.items():
            assert root in got.parents or got == root, f"{name} -> {got}"


class TestNoModuleNamesOneOfThoseFilesItself:
    """**The rule, asked the strict way round.**

    Not *"is everything using the API"* — a module can pass that by doing
    nothing — but *"does anything still spell one of these filenames"*.  Seven
    modules each held one and joined it onto a directory; each join was correct
    and together they were seven places that had to agree with nothing making
    them.  `configuration.md` M-4 recorded exactly this for `environment.json`
    ("a string literal in three modules"), fixed that one file, and did not
    generalise — so the other six grew back.
    """

    #: The format owner of each, per A11.  Nobody else may join it.
    _OWNERS = {
        "config_dir.py": "the files with no format owner",
        "runtime_config.py": "molbuilder.json's schema",
        "scheduler/record.py": "environment.json's schema",
        "monitor.py": "the notify exchange",
    }

    @pytest.mark.parametrize("literal", [
        '"molbuilder.json"', '"secret_key"', '"google_client_secret"',
        '"environment.json"', '"notify_keys"',
    ])
    def test_the_literal_appears_only_where_it_is_owned(self, literal):
        offenders = {}
        for py in _SRC.rglob("*.py"):
            rel = str(py.relative_to(_SRC))
            if rel in self._OWNERS:
                continue
            src = py.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                stripped = line.strip()
                # a comment or docstring may NAME a file; only code may not
                # BUILD a path from it
                if literal in line and not stripped.startswith(("#", "*", ">")):
                    if "/" in line or "Path(" in line or "join" in line:
                        offenders.setdefault(rel, []).append(i)
        assert not offenders, (
            f"{literal} is joined into a path outside the module that owns it: "
            f"{offenders}.  Ask `config_dir` for the file instead "
            f"(archive/2026-09-01-config-access-plan.md § 5)")

