"""One directory, one function — the callers cannot disagree about it.

`configuration.md` M-4 gave ``environment.json`` one home for its FILENAME,
which "was a string literal in three modules".  The DIRECTORY that filename
sits in stayed spelled three times:

    runtime_config._per_user_fallback_path   -> molbuilder.json
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
    from molbuilder.runtime_config import _per_user_fallback_path
    from molbuilder.scheduler import environments_dir, machine_scope_path
    return {
        "secret_key":     secret_key_path(),
        "secret_dir":     default_secret_dir(),
        "environment":    machine_scope_path(),
        "environments":   environments_dir(),
        "molbuilder.json": _per_user_fallback_path(),
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
# The override — plans/config-access-plan.md § 3.1
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


def test_no_module_computes_a_per_user_root_itself():
    """The pin the plan promises, asked the strict way round.

    Not *"is everything using the door"* -- which a new module can pass by
    doing nothing -- but *"does anything build a per-user path without it"*.
    `config_dir.py` is the one place allowed to join a home directory to a
    name; every other module asks it.
    """
    offenders = {}
    for py in _SRC.rglob("*.py"):
        if py.name == "config_dir.py":
            continue
        src = py.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:                   # pragma: no cover -- defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) \
                else getattr(fn, "id", "")
            if name not in ("expanduser", "Path"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) \
                        and isinstance(arg.value, str) \
                        and arg.value.startswith("~/.molbuilder"):
                    offenders.setdefault(
                        str(py.relative_to(_SRC)), []).append(arg.value)
    assert not offenders, (
        f"these modules build a per-user path themselves: {offenders}.  "
        f"`~/.molbuilder/` was retired 2026-08-31 "
        f"(plans/config-access-plan.md § 3.2) -- ask runtime_config for "
        f"logs_dir(), run_dir() or reports_dir(), which honour both the XDG "
        f"directories and molbuilder.json's `paths` block")


# ---------------------------------------------------------------------------
# Operational state — plans/config-access-plan.md § 3.2
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


class TestThePathsOverride:

    @pytest.fixture()
    def cfg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _write(self, cfg, obj):
        import json
        (cfg / "molbuilder.json").write_text(json.dumps(obj))

    def test_defaults_when_nothing_is_named(self, cfg):
        from molbuilder.runtime_config import logs_dir, reports_dir, run_dir
        assert logs_dir() == cfg / "state" / "molbuilder" / "logs"
        assert reports_dir() == cfg / "state" / "molbuilder" / "reports"
        assert run_dir() == cfg / "state" / "molbuilder" / "run"

    def test_a_named_directory_wins(self, cfg):
        from molbuilder.runtime_config import logs_dir, run_dir
        self._write(cfg, {"paths": {"logs": str(cfg / "scratch" / "l"),
                                    "run": str(cfg / "scratch" / "r")}})
        assert logs_dir() == cfg / "scratch" / "l"
        assert run_dir() == cfg / "scratch" / "r"

    def test_naming_one_leaves_the_others_alone(self, cfg):
        from molbuilder.runtime_config import logs_dir, reports_dir
        self._write(cfg, {"paths": {"logs": str(cfg / "only")}})
        assert logs_dir() == cfg / "only"
        assert reports_dir() == cfg / "state" / "molbuilder" / "reports"

    def test_a_key_nothing_reads_is_refused(self, cfg):
        """A `paths` block naming a directory nothing consults would look
        effective and do nothing -- the argument behind every refusal in
        configuration.md."""
        from molbuilder.runtime_config import RuntimeConfigError, read_config
        self._write(cfg, {"paths": {"cache": "/tmp/x"}})
        with pytest.raises(RuntimeConfigError, match="cache"):
            read_config()

    def test_the_key_that_was_already_there_still_works(self, cfg):
        """**The regression the first attempt shipped.**

        `paths` was not a new section -- it already held `projects`, the tree
        every surface resolves through.  Adding a second `_read_paths` and a
        second registry entry gave the dict a duplicate key: the later one won
        silently, and `paths.projects` began being refused as unknown.  Python
        raises nothing for a repeated key in a literal, and no test named
        `projects`, so only a mutation run that could not find its own pattern
        twice exposed it.
        """
        from molbuilder.runtime_config import read_config
        self._write(cfg, {"paths": {"projects": "/data/projects"}})
        assert read_config()["paths"]["projects"] == "/data/projects"

    def test_the_two_kinds_of_path_coexist(self, cfg):
        from molbuilder.runtime_config import logs_dir, read_config
        self._write(cfg, {"paths": {"projects": "/data/p",
                                    "logs": str(cfg / "l")}})
        assert read_config()["paths"]["projects"] == "/data/p"
        assert logs_dir() == cfg / "l"

    def test_there_is_exactly_one_paths_reader(self):
        """A duplicate is invisible at runtime, so it is asserted at source."""
        import inspect
        from molbuilder import runtime_config as rc
        src = inspect.getsource(rc)
        assert src.count("def _read_paths(") == 1
        assert src.count('"paths":             {"read"') == 1

    def test_a_broken_config_still_has_somewhere_to_be_logged(self, cfg):
        """THE BOOTSTRAP RULE.  A log that could only be written after parsing
        a file that failed to parse is the one log nobody gets."""
        from molbuilder.runtime_config import logs_dir
        (cfg / "molbuilder.json").write_text("{ not json")
        assert logs_dir() == cfg / "state" / "molbuilder" / "logs"

    def test_the_override_is_machine_scope_only(self):
        """Where an installation writes its logs is a property of the
        installation.  A project able to redirect them could point one run's
        output somewhere the operator does not look."""
        from molbuilder.runtime_config import _SECTIONS
        assert _SECTIONS["paths"]["scopes"] == ("machine",)
