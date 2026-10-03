"""This machine's ``molbuilder.json`` -- where it is read from
(``read_config``) and its one writer (``write_config_scope``),
`configuration.md` § 2.1, § 2.3.

Pinned contracts:
  * one file, in the config directory: ``$MOLBUILDER_CONFIG_DIR``, else
    ``$XDG_CONFIG_HOME/molbuilder/``, else ``~/.config/molbuilder/``; a
    working-directory copy is not read.
  * ``write_config_scope`` produces the file mode 0600, preserves keys
    outside the patch, validates before writing, and never overwrites a
    corrupt file.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from molbuilder.runtime_config import (
    CONFIG_FILENAME,
    RuntimeConfigError,
    read_config,
    write_config_scope,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A clean cwd + isolated $HOME + cleared XDG_CONFIG_HOME so
    read_config + write_config_scope land in tmp_path.

    Yields the tmp_path (which becomes the cwd and the config root)."""
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
#  read_config -- where the one file is found                           #
# --------------------------------------------------------------------- #


def test_no_config_files_returns_empty(sandbox):
    assert read_config() == {}


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
    cfg = read_config()
    assert cfg.get("envs") == {"pyscf": "alt-pyscf"}


def test_explicit_xdg_config_home_is_honored(xdg_branch, monkeypatch):
    sandbox = xdg_branch
    xdg_dir = sandbox / "elsewhere"
    (xdg_dir / "molbuilder").mkdir(parents=True)
    (xdg_dir / "molbuilder" / "molbuilder.json").write_text(json.dumps({
        "envs": {"pyscf": "elsewhere-pyscf"},
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    cfg = read_config()
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
#  write_config_scope                                                    #
# --------------------------------------------------------------------- #


def test_write_server_wide_when_cwd_file_exists_writes_to_cwd(sandbox):
    """When the cwd molbuilder.json exists, a server-wide write lands
    there (per docs/execution/running-a-job.md § 5: writes to the highest-precedence
    EXISTING location)."""
    (sandbox / "molbuilder.json").write_text("{}\n")
    target = write_config_scope({
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
    target = write_config_scope({
        "paths": {"projects": "/srv/projects"},
    })
    assert target == sandbox / "home" / ".config" / "molbuilder" / "molbuilder.json"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    cfg = json.loads(target.read_text())
    assert cfg["paths"]["projects"] == "/srv/projects"


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
    write_config_scope({
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
        write_config_scope({
            "paths": {"projects": "/srv/projects"},
        })
    assert "envs" in str(e.value)
    # ...and nothing was written.
    assert json.loads((sandbox / "molbuilder.json").read_text()) == {
        "envs": {"siesta": 5}}


def test_write_validates_before_writing(sandbox):
    """A patch with an invalid value is rejected -- the file is NOT
    written.  Otherwise the next read_config call would fail
    even though the user thought their write succeeded."""
    with pytest.raises(RuntimeConfigError, match="launch.mode"):
        write_config_scope({
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
    with pytest.raises(RuntimeConfigError, match="refused, never overwritten"):
        write_config_scope({
            "paths": {"projects": "/srv/projects"},
        })
    # the broken content is untouched -- nothing was destroyed
    assert (sandbox / "molbuilder.json").read_text() == "not valid json {{{"
