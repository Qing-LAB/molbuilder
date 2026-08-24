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
