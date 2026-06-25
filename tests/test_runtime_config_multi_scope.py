"""Tests for the unified-data-model API (docs/config.md § 4):
``read_effective_config``, ``write_config_scope``, ``get_script_generation``.

Pinned contracts:
  * server-wide lookup chain: cwd ``molbuilder.json`` first, XDG
    fallback (``$XDG_CONFIG_HOME/molbuilder/molbuilder.json`` or
    ``~/.config/molbuilder/molbuilder.json``) second.
  * project-scope lookup: ``<project_dir>/.molbuilder.json``.
  * deep-merge rules: scalars + arrays = project replaces, objects =
    recurse; project wins on conflict.
  * ``script_generation.preactivate`` CONCATENATES server-wide ++
    project (per § 3.6, not the generic replace rule) -- pinned in
    ``test_preactivate_concatenates_across_scopes``.
  * ``script_generation.autodetect_conda`` and ``preactivate_format``
    follow the standard replace rule.
  * ``write_config_scope`` produces files mode 0600 and preserves
    keys outside the patch.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from molbuilder.runtime_config import (
    CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    RuntimeConfigError,
    get_script_generation,
    read_effective_config,
    write_config_scope,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A clean cwd + isolated $HOME + cleared XDG_CONFIG_HOME so
    read_effective_config + write_config_scope land in tmp_path.

    Yields the tmp_path (which becomes the cwd) for tests to write
    server-wide / project files into.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


# --------------------------------------------------------------------- #
#  read_effective_config -- server-wide lookup chain                    #
# --------------------------------------------------------------------- #


def test_no_config_files_returns_empty(sandbox):
    assert read_effective_config() == {}


def test_cwd_server_wide_is_read(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "alt-siesta"},
    }))
    cfg = read_effective_config()
    assert cfg.get("envs") == {"siesta": "alt-siesta"}


def test_xdg_fallback_is_read_when_cwd_absent(sandbox, monkeypatch):
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "alt-pyscf"},
    }))
    cfg = read_effective_config()
    assert cfg.get("envs") == {"pyscf": "alt-pyscf"}


def test_cwd_wins_over_xdg_fallback(sandbox, monkeypatch):
    """When both exist, cwd is the server-wide layer; XDG is ignored.
    Per docs/config.md § 2's lookup order."""
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "xdg-pyscf"},
    }))
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "cwd-pyscf"},
    }))
    cfg = read_effective_config()
    assert cfg["envs"]["pyscf"] == "cwd-pyscf"


def test_explicit_xdg_config_home_is_honored(sandbox, monkeypatch):
    xdg_dir = sandbox / "elsewhere"
    (xdg_dir / "molbuilder").mkdir(parents=True)
    (xdg_dir / "molbuilder" / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "elsewhere-pyscf"},
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    cfg = read_effective_config()
    assert cfg["envs"]["pyscf"] == "elsewhere-pyscf"


# --------------------------------------------------------------------- #
#  read_effective_config -- project scope + deep merge                  #
# --------------------------------------------------------------------- #


def test_project_overlay_replaces_scalar(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "server-siesta"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "project-siesta"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["envs"]["siesta"] == "project-siesta"


def test_project_overlay_deep_merges_objects(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "server-siesta", "pyscf": "server-pyscf"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "project-siesta"},
    }))
    cfg = read_effective_config(project_dir=proj)
    # pyscf preserved from server, siesta overridden.
    assert cfg["envs"] == {
        "siesta": "project-siesta",
        "pyscf":  "server-pyscf",
    }


def test_project_only_returns_project_layer(sandbox):
    """Server-wide absent + project present -> project values come
    through alone."""
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "envs": {"siesta": "project-siesta"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["envs"]["siesta"] == "project-siesta"


def test_project_dir_none_returns_server_layer_unchanged(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "server-siesta"},
    }))
    cfg = read_effective_config(project_dir=None)
    assert cfg.get("envs") == {"siesta": "server-siesta"}


# --------------------------------------------------------------------- #
#  get_script_generation -- preactivate concatenation                   #
# --------------------------------------------------------------------- #


def test_get_script_generation_defaults_when_all_empty(sandbox):
    sg = get_script_generation()
    assert sg["preactivate"] == ""
    assert sg["preactivate_format"] == "shell"
    assert sg["autodetect_conda"] is False
    assert sg["_preactivate_scopes"] == []


def test_preactivate_concatenates_across_scopes(sandbox):
    """Per docs/config.md § 3.6, preactivate is the ONE field that
    uses concatenation (server-wide first, then project) rather than
    the generic replace rule."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preactivate": "module load mamba"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "script_generation": {"preactivate": "export FOO=bar"},
    }))
    sg = get_script_generation(project_dir=proj)
    assert "module load mamba" in sg["preactivate"]
    assert "export FOO=bar" in sg["preactivate"]
    # Server first, project second -- order matters for the rendered
    # wrapper (cluster setup before project-specific tweaks).
    server_ix = sg["preactivate"].find("module load mamba")
    project_ix = sg["preactivate"].find("export FOO=bar")
    assert server_ix < project_ix
    assert sg["_preactivate_scopes"] == ["server", "project"]


def test_preactivate_format_uses_replace_rule(sandbox):
    """Other script_generation fields use the standard "project
    wins" replace rule, not concatenation."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preactivate_format": "shell"},
    }))
    sg = get_script_generation()
    assert sg["preactivate_format"] == "shell"


def test_autodetect_conda_uses_replace_rule(sandbox):
    """Project autodetect_conda overrides server-wide."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"autodetect_conda": False},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "script_generation": {"autodetect_conda": True},
    }))
    sg = get_script_generation(project_dir=proj)
    assert sg["autodetect_conda"] is True


def test_autodetect_conda_default_is_false(sandbox):
    """Per docs/config.md § 0 decision 6 + § 3.6: autodetect_conda
    is opt-in -- the default for fresh installs is False (fail loud
    when preactivate is misconfigured rather than masking it)."""
    sg = get_script_generation()
    assert sg["autodetect_conda"] is False


# --------------------------------------------------------------------- #
#  Validation                                                            #
# --------------------------------------------------------------------- #


def test_invalid_preactivate_type_rejected(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preactivate": 123},
    }))
    with pytest.raises(RuntimeConfigError, match="preactivate.*string"):
        read_effective_config()


def test_invalid_autodetect_conda_type_rejected(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"autodetect_conda": "yes"},
    }))
    with pytest.raises(RuntimeConfigError, match="autodetect_conda"):
        read_effective_config()


def test_invalid_preactivate_format_rejected(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "script_generation": {"preactivate_format": "yaml"},
    }))
    with pytest.raises(RuntimeConfigError, match="preactivate_format"):
        read_effective_config()


# --------------------------------------------------------------------- #
#  write_config_scope                                                    #
# --------------------------------------------------------------------- #


def test_write_server_wide_when_cwd_file_exists_writes_to_cwd(sandbox):
    """When the cwd molbuilder.json exists, a server-wide write lands
    there (per docs/config.md § 4.2: writes to the highest-precedence
    EXISTING location)."""
    (sandbox / "molbuilder.json").write_text("{}\n")
    target = write_config_scope(project_dir=None, patch={
        "script_generation": {"preactivate": "module load mamba"},
    })
    assert target == sandbox / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["preactivate"] == "module load mamba"


def test_write_server_wide_creates_xdg_when_cwd_absent(sandbox):
    """When NO server-wide file exists, the write lands at the XDG
    path (per docs/config.md § 4.2 last sentence)."""
    target = write_config_scope(project_dir=None, patch={
        "script_generation": {"preactivate": "module load mamba"},
    })
    assert target == sandbox / "home" / ".config" / "molbuilder" / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["preactivate"] == "module load mamba"


def test_write_project_scope_creates_hidden_file(sandbox):
    proj = sandbox / "myproject"
    proj.mkdir()
    target = write_config_scope(project_dir=proj, patch={
        "script_generation": {"autodetect_conda": True},
    })
    assert target == proj / ".molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["autodetect_conda"] is True


def test_write_preserves_existing_unrelated_keys(sandbox):
    """A patch only touches the keys it carries.  Sister sections
    survive."""
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta"},
        "secret_key_file": "~/.config/molbuilder/secret_key",
    }))
    write_config_scope(project_dir=None, patch={
        "script_generation": {"preactivate": "module load mamba"},
    })
    cfg = json.loads((sandbox / "molbuilder.json").read_text())
    assert cfg["envs"]["siesta"] == "molbuilder-siesta"
    assert cfg["secret_key_file"] == "~/.config/molbuilder/secret_key"
    assert cfg["script_generation"]["preactivate"] == "module load mamba"


def test_write_validates_before_writing(sandbox):
    """A patch with an invalid value is rejected -- the file is NOT
    written.  Otherwise the next read_effective_config call would fail
    even though the user thought their write succeeded."""
    with pytest.raises(RuntimeConfigError, match="autodetect_conda"):
        write_config_scope(project_dir=None, patch={
            "script_generation": {"autodetect_conda": "yes"},
        })
    # File never created.
    assert not (sandbox / "molbuilder.json").exists()


def test_write_overwriting_existing_corrupt_file(sandbox):
    """If the existing file is corrupt JSON, write_config_scope
    overwrites it with the patch (the user's chosen content) rather
    than refusing to write.  Documented behaviour in the docstring."""
    (sandbox / "molbuilder.json").write_text("not valid json {{{")
    target = write_config_scope(project_dir=None, patch={
        "script_generation": {"preactivate": "module load mamba"},
    })
    cfg = json.loads(target.read_text())
    assert cfg["script_generation"]["preactivate"] == "module load mamba"
