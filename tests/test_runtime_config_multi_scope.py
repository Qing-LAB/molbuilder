"""Tests for the unified-data-model API (docs/execution/running-a-job.md § 5):
``read_effective_config``, ``write_config_scope``.

Pinned contracts:
  * server-wide lookup chain: cwd ``molbuilder.json`` first, XDG
    fallback (``$XDG_CONFIG_HOME/molbuilder/molbuilder.json`` or
    ``~/.config/molbuilder/molbuilder.json``) second.
  * project-scope lookup: ``<project_dir>/.molbuilder.json``.
  * deep-merge rules: scalars + arrays = project replaces, objects =
    recurse; project wins on conflict.
  * ``write_config_scope`` produces files mode 0600 and preserves
    keys outside the patch.

(``script_generation``'s own merge rule -- preambles concatenating across
the two files -- went with the section on 2026-10-02: the activation and
preamble are the machine record's, `configuration.md` § 5 M-1.)
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
    # THE SANDBOX IS THE CONFIG ROOT (§ 2.1c).  The cwd step these
    # tests were written against is gone, so without this every
    # config they write is a file nothing reads.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


# --------------------------------------------------------------------- #
#  read_effective_config -- server-wide lookup chain                    #
# --------------------------------------------------------------------- #


def test_no_config_files_returns_empty(sandbox):
    assert read_effective_config() == {}


@pytest.fixture
def xdg_branch(sandbox, monkeypatch):
    """Exercise the XDG step, which the override deliberately outranks.

    `sandbox` names ``MOLBUILDER_CONFIG_DIR`` so the file a test writes is the
    file the reader opens.  A test whose SUBJECT is the XDG fallback has to
    clear it — the override is not a search step, and a test that left it set
    would be asserting the override while claiming to assert XDG
    (`configuration.md` § 2.1c).
    """
    monkeypatch.delenv("MOLBUILDER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(sandbox / "home" / ".config"))
    return sandbox


def test_xdg_fallback_is_read_when_cwd_absent(xdg_branch, monkeypatch):
    sandbox = xdg_branch
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "alt-pyscf"},
    }))
    cfg = read_effective_config()
    assert cfg.get("envs") == {"pyscf": "alt-pyscf"}


def test_explicit_xdg_config_home_is_honored(xdg_branch, monkeypatch):
    sandbox = xdg_branch
    xdg_dir = sandbox / "elsewhere"
    (xdg_dir / "molbuilder").mkdir(parents=True)
    (xdg_dir / "molbuilder" / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "elsewhere-pyscf"},
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    cfg = read_effective_config()
    assert cfg["envs"]["pyscf"] == "elsewhere-pyscf"


def test_the_bare_default_read_honours_the_same_fallback(xdg_branch):
    """A-7 (final review, 2026-08-13): the bare ``read_config()`` was
    cwd-only while the section getters honoured the XDG file too, so an
    operator with an XDG-only config got a no-auth, no-TLS server while
    `jobset` honoured the very same file.

    The cwd step is gone (2026-08-31) and the split-brain it enabled with it,
    but the property this pins is the one that outlived it: the default read
    and the section getters share ONE lookup."""
    sandbox = xdg_branch
    from molbuilder.runtime_config import get_tls, read_config
    xdg_dir = sandbox / "home" / ".config" / "molbuilder"
    xdg_dir.mkdir(parents=True)
    (xdg_dir / "molbuilder.json").write_text(json.dumps({
        "tls": {"cert": "c.pem", "key": "k.pem"}}))
    assert get_tls(read_config()) == {"cert": "c.pem", "key": "k.pem"}
    # A file in the working directory changes NOTHING -- it is not read
    # (§ 2.1a), which is the half of this that inverted.
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "tls": {"cert": "cwd.pem", "key": "cwd-k.pem"}}))
    assert read_config()["tls"]["cert"] == "c.pem", (
        "a working-directory file reached the reader")


# --------------------------------------------------------------------- #
#  read_effective_config -- project scope + deep merge                  #
# --------------------------------------------------------------------- #


def test_project_overlay_replaces_scalar(sandbox):
    # `launch` here, not `envs`: the merge mechanics need the section a
    # BUNDLE may carry -- the only one since 2026-10-02 -- and the registry
    # refuses machine-only sections in project scope.
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "launch": {"mode": "submit"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "launch": {"mode": "direct"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["launch"]["mode"] == "direct"


def test_project_overlay_deep_merges_objects(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "launch": {"_comment": "this box runs its own jobs",
                   "mode": "direct"},
    }))
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "launch": {"mode": "submit"},
    }))
    cfg = read_effective_config(project_dir=proj)
    # the comment preserved from the server, the mode overridden -- objects
    # recurse.
    assert cfg["launch"] == {"_comment": "this box runs its own jobs",
                             "mode": "submit"}


def test_project_only_returns_project_layer(sandbox):
    """Server-wide absent + project present -> project values come
    through alone."""
    proj = sandbox / "myproject"
    proj.mkdir()
    (proj / PROJECT_CONFIG_FILENAME).write_text(json.dumps({
        "launch": {"mode": "direct"},
    }))
    cfg = read_effective_config(project_dir=proj)
    assert cfg["launch"]["mode"] == "direct"


def test_project_dir_none_returns_server_layer_unchanged(sandbox):
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "server-siesta"},
    }))
    cfg = read_effective_config(project_dir=None)
    assert cfg.get("envs") == {"siesta": "server-siesta"}


# --------------------------------------------------------------------- #
#  write_config_scope                                                    #
# --------------------------------------------------------------------- #


def test_write_server_wide_when_cwd_file_exists_writes_to_cwd(sandbox):
    """When the cwd molbuilder.json exists, a server-wide write lands
    there (per docs/execution/running-a-job.md § 5: writes to the highest-precedence
    EXISTING location)."""
    (sandbox / "molbuilder.json").write_text("{}\n")
    target = write_config_scope(project_dir=None, patch={
        "paths": {"projects": "/srv/projects"},
    })
    assert target == sandbox / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["paths"]["projects"] == "/srv/projects"


def test_write_server_wide_creates_xdg_when_cwd_absent(xdg_branch):
    sandbox = xdg_branch
    """When NO server-wide file exists, the write lands at the XDG
    path (per docs/execution/running-a-job.md § 5 last sentence)."""
    target = write_config_scope(project_dir=None, patch={
        "paths": {"projects": "/srv/projects"},
    })
    assert target == sandbox / "home" / ".config" / "molbuilder" / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["paths"]["projects"] == "/srv/projects"


def test_write_project_scope_creates_hidden_file(sandbox):
    proj = sandbox / "myproject"
    proj.mkdir()
    # A LIVE key: this wrote `autodetect_conda` until 2026-09-14 and passed
    # because the writer tolerated a retired key.  It is refused now, so the
    # test would have been asserting the tolerance rather than the write.
    target = write_config_scope(project_dir=proj, patch={
        "launch": {"mode": "submit"},
    })
    assert target == proj / ".molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["launch"]["mode"] == "submit"


def test_write_preserves_existing_unrelated_keys(sandbox):
    """A patch only touches the keys it carries.  Sister sections
    survive."""
    # `secret_key_file` was the second sister section here until 2026-08-31.
    # A config carrying it is now REFUSED, so seeding one would test the
    # refusal rather than the merge (`configuration.md` § 2.1e).
    (sandbox / "molbuilder.json").write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta"},
        "launch": {"mode": "direct"},
    }))
    write_config_scope(project_dir=None, patch={
        "paths": {"projects": "/srv/projects"},
    })
    cfg = json.loads((sandbox / "molbuilder.json").read_text())
    assert cfg["envs"]["siesta"] == "molbuilder-siesta"
    assert cfg["launch"]["mode"] == "direct"
    assert cfg["paths"]["projects"] == "/srv/projects"


def test_a_file_that_was_already_refused_is_named_not_the_patch(sandbox):
    """The merge fails validation because the patch is bad -- or because the
    file already was.  The message says which: a person sent to fix "the
    patch" for a section it never touched fixes the wrong thing.  (The auth
    wizard blamed itself for a bad `envs` entry, measured 2026-09-10; the
    diagnosis lived in its private writer until 2026-09-13.)"""
    (sandbox / "molbuilder.json").write_text(json.dumps({"envs": {"siesta": 5}}))
    with pytest.raises(RuntimeConfigError, match="ALREADY") as e:
        write_config_scope(project_dir=None, patch={
            "paths": {"projects": "/srv/projects"},
        })
    assert "envs" in str(e.value)
    # ...and nothing was written.
    assert json.loads((sandbox / "molbuilder.json").read_text()) == {
        "envs": {"siesta": 5}}


def test_a_project_write_refuses_a_machine_section_already_in_the_file(sandbox):
    """The writer applied the scope rule to the PATCH only, so a project file
    already carrying `tls` was written -- and then refused by every read
    (review C-L2, measured 2026-09-14).  Reader and writer apply one rule to
    the whole file now."""
    proj = sandbox / "proj"; proj.mkdir()
    before = {"tls": {"cert": "/c", "key": "/k"}}
    (proj / ".molbuilder.json").write_text(json.dumps(before))
    with pytest.raises(RuntimeConfigError, match="'tls' may not live in a PROJECT"):
        write_config_scope(project_dir=proj, patch={"launch": {"mode": "submit"}})
    assert json.loads((proj / ".molbuilder.json").read_text()) == before


def test_a_refusal_names_the_file_once(sandbox):
    """Two shapes of one defect (review C-L6): the writer re-raised the
    validator's generic `molbuilder.json:` for a project file, and a
    retired-key refusal on a project file read `/p/.molbuilder.json:
    molbuilder.json: 'paths.logs' ...` -- two names, the second wrong."""
    proj = sandbox / "proj"; proj.mkdir()
    target = str(proj / ".molbuilder.json")
    with pytest.raises(RuntimeConfigError) as e:
        write_config_scope(project_dir=proj, patch={"launch": "nope"})
    assert str(e.value).startswith(target + ": "), str(e.value)
    assert str(e.value).count("molbuilder.json:") == 1, str(e.value)
    (proj / ".molbuilder.json").write_text(json.dumps({"paths": {"logs": "/x"}}))
    with pytest.raises(RuntimeConfigError) as e:
        read_effective_config(proj)
    assert str(e.value).startswith(target + ": 'paths.logs'"), str(e.value)


def test_write_validates_before_writing(sandbox):
    """A patch with an invalid value is rejected -- the file is NOT
    written.  Otherwise the next read_effective_config call would fail
    even though the user thought their write succeeded."""
    with pytest.raises(RuntimeConfigError, match="launch.mode"):
        write_config_scope(project_dir=None, patch={
            "launch": {"mode": "wrong-form"},
        })
    # File never created.
    assert not (sandbox / "molbuilder.json").exists()


def test_write_refuses_to_overwrite_a_corrupt_file(sandbox):
    """R10 (review-4 G7): this test pinned the OPPOSITE -- 'overwrites
    it with the patch... documented behaviour' -- and that tolerance
    silently destroyed whatever a hand-edit broke: a config carrying
    auth providers and TLS paths is exactly the file a user cannot
    afford to lose to a typo plus any later --write.  Corrupt now
    REFUSES, naming the path and the way out."""
    import pytest
    from molbuilder.runtime_config import RuntimeConfigError
    (sandbox / "molbuilder.json").write_text("not valid json {{{")
    with pytest.raises(RuntimeConfigError, match="refusing to overwrite"):
        write_config_scope(project_dir=None, patch={
            "paths": {"projects": "/srv/projects"},
        })
    # the broken content is untouched -- nothing was destroyed
    assert (sandbox / "molbuilder.json").read_text() == "not valid json {{{"


def test_write_config_scope_refuses_machine_sections_for_a_bundle(sandbox):
    """Refusing at WRITE time beats producing a file every later read
    refuses (U7 -- the same registry row drives both)."""
    import pytest
    from molbuilder.runtime_config import (RuntimeConfigError,
                                           write_config_scope)
    proj = sandbox / "proj2"
    proj.mkdir(exist_ok=True)
    with pytest.raises(RuntimeConfigError, match="admin"):
        write_config_scope(proj, {"admin": {"emails": ["x@y.edu"]}})
    assert not (proj / ".molbuilder.json").exists()
    # a bundle section still writes fine
    out = write_config_scope(proj, {"launch": {"mode": "direct"}})
    assert out.is_file()
