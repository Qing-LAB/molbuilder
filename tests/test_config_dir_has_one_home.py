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
        """Same guard as `XDG_CONFIG_HOME`'s, for the same reason."""
        offenders = []
        for py in _SRC.rglob("*.py"):
            if py.name == "config_dir.py":
                continue
            src = py.read_text(encoding="utf-8")
            if "MOLBUILDER_CONFIG_DIR" in src:
                offenders.append(str(py.relative_to(_SRC)))
        assert not offenders, (
            "these modules read MOLBUILDER_CONFIG_DIR directly instead of "
            f"calling config_dir(): {sorted(set(offenders))}")


# ---------------------------------------------------------------------------
# The second root — plans/config-access-plan.md § 3.2, step 2
# ---------------------------------------------------------------------------

#: Modules that still compute a per-user path themselves, with the path.
#:
#: `~/.molbuilder/` is the SECOND ROOT the plan retires: it moves with nothing,
#: so a person who sets either variable moves some of their files and not the
#: rest.  This list is the work, written down -- step 2 empties it, and the
#: test below fails the moment it is empty so the allowance is deleted rather
#: than left standing.
_SECOND_ROOT_HOLDOUTS = {
    "serve_daemon.py": "~/.molbuilder/run, ~/.molbuilder/logs",
    "envs/_cli.py": "~/.molbuilder/logs",
    "web/blueprints/notify.py": "~/.molbuilder/reports",
}


def _computes_a_second_root_path(src: str) -> bool:
    """Does this module BUILD a ``~/.molbuilder/...`` path, or merely mention one?

    The difference matters and the first version of this test missed it: it
    matched the string anywhere, and flagged `notify_setup.py` -- whose only
    mention is a docstring recording this exact class of bug (*"the Task-setup
    card said ``~/.molbuilder/notify`` while the monitor read
    ``config_dir()/notify``, and following the card put the file where nothing
    looks"*).  That module does the right thing and says why.  **A test that
    cannot tell a path from a sentence about a path punishes the documentation
    we want.**
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:                       # pragma: no cover -- defensive
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("expanduser", "Path", "home"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith("~/.molbuilder"):
                return True
    return False


def test_the_second_root_shrinks_and_this_list_is_the_work():
    found = {}
    for py in _SRC.rglob("*.py"):
        if _computes_a_second_root_path(py.read_text(encoding="utf-8")):
            found[str(py.relative_to(_SRC))] = True
    unexpected = sorted(set(found) - set(_SECOND_ROOT_HOLDOUTS))
    assert not unexpected, (
        f"new modules reached for ~/.molbuilder/: {unexpected}. That root is "
        f"being retired (plans/config-access-plan.md § 3.2); use the state or "
        f"runtime directory instead")
    gone = sorted(set(_SECOND_ROOT_HOLDOUTS) - set(found))
    assert not gone, (
        f"these no longer name ~/.molbuilder/: {gone} -- delete them from "
        f"_SECOND_ROOT_HOLDOUTS. When the list empties, delete the list and "
        f"this test with it: the root is gone and there is nothing to shrink")


def test_record_stays_importable_with_stdlib_only_at_module_level():
    """`record.py` claims to be stdlib-only, and `config_dir` is L1 so the
    claim survives importing it -- the same way `persist` does.  If someone
    later gives `config_dir` a molbuilder dependency, this fails."""
    import molbuilder.config_dir as cd
    tree = ast.parse(Path(cd.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] != "molbuilder", (
                f"config_dir imported {node.module} -- it must stay L1 stdlib")
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("molbuilder"), a.name


def test_an_explicit_root_still_wins_for_the_caller_that_passes_one():
    """`default_secret_dir(home=...)` names the root outright; a caller that
    has answered the question does not get XDG's answer instead."""
    from molbuilder.auth_setup import default_secret_dir
    got = default_secret_dir(home=Path("/opt/somewhere"))
    assert got == Path("/opt/somewhere/.config/molbuilder")
