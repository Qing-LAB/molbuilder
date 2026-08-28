"""Tests for the auth-setup wizard (``molbuilder auth-setup``).

Two layers:
  * Pure-function emitters in ``molbuilder.auth_setup`` -- tested
    without prompting (no subprocess, no terminal).
  * The Click CLI thin wrapper in ``molbuilder.cli.cmd_auth_setup`` --
    tested with CliRunner + ``input=`` to drive prompts.

Privacy contract under test:
  * Flask session key + Google client_secret never appear in
    ``molbuilder.json`` (only file paths do).
  * Every secret file is mode 0600.
  * ``molbuilder.json`` itself is mode 0600.
  * The system user (``getpass.getuser()``) is the only identifier
    the wizard assumes; no other email / username is hardcoded.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from molbuilder import auth_setup as _as
from molbuilder.cli import cli
from molbuilder.runtime_config import _validate_provider


# --------------------------------------------------------------------- #
#  Pure helpers                                                          #
# --------------------------------------------------------------------- #


def test_default_secret_dir_under_home(tmp_path):
    d = _as.default_secret_dir(home=tmp_path)
    assert d == tmp_path / ".config" / "molbuilder"


def test_default_secret_dir_honors_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    d = _as.default_secret_dir()
    assert d == tmp_path / "xdg" / "molbuilder"


def test_generate_session_secret_is_unique_and_long():
    a, b = _as.generate_session_secret(), _as.generate_session_secret()
    assert a != b
    # token_urlsafe(32) yields ~43 chars; permit a small floor in case
    # the implementation changes to a slightly different length.
    assert len(a) >= 32
    # URL-safe base64 alphabet only.
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", a)


# --------------------------------------------------------------------- #
#  write_secret_file -- the 0600 contract                                #
# --------------------------------------------------------------------- #


def test_write_secret_file_creates_with_0600(tmp_path):
    p = tmp_path / "sub" / "secret"
    _as.write_secret_file(p, "hello")
    st = p.stat()
    # Owner read/write only -- no group or world bits.
    assert stat.S_IMODE(st.st_mode) == 0o600
    assert p.read_text() == "hello"


def test_write_secret_file_overwrites_world_readable_file(tmp_path):
    """A pre-existing world-readable file at the same path must be
    tightened in-place -- defends against a stale 0644 file from an
    earlier hand-written setup attempt."""
    p = tmp_path / "secret"
    p.write_text("old")
    os.chmod(p, 0o644)
    _as.write_secret_file(p, "new")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert p.read_text() == "new"


def test_write_secret_file_rejects_empty():
    with pytest.raises(ValueError, match="empty secret"):
        _as.write_secret_file(Path("/tmp/whatever"), "")


# --------------------------------------------------------------------- #
#  Provider entry builders                                               #
# --------------------------------------------------------------------- #


def test_asu_cas_entry_shape_round_trips_validator():
    entry = _as.build_asu_cas_entry("jdoe")
    # Must round-trip through the canonical runtime_config validator.
    _validate_provider(entry, idx=0)
    assert entry["kind"] == "cas"
    assert entry["allowed_users"] == ["jdoe@asu.edu"]
    assert entry["login_url"].startswith("https://")
    assert entry["service_validate_url"].startswith("https://")
    assert entry["email_domain"] == "asu.edu"


def test_asu_cas_rejects_email_passed_as_asurite():
    """Common user mistake: pass 'foo@asu.edu' when the field wants
    just 'foo'.  Helpful error guides them, no silent doubling."""
    with pytest.raises(ValueError, match="username, not an email"):
        _as.build_asu_cas_entry("foo@asu.edu")


def test_asu_cas_rejects_empty_asurite():
    with pytest.raises(ValueError, match="required"):
        _as.build_asu_cas_entry("")


def test_google_entry_shape_round_trips_validator(tmp_path):
    secret_file = tmp_path / "google_secret"
    secret_file.write_text("dummy")
    os.chmod(secret_file, 0o600)
    entry = _as.build_google_entry(
        client_id="client-id-123",
        client_secret_file=secret_file,
        allowed_users=["alice@gmail.com", "bob@asu.edu"],
    )
    _validate_provider(entry, idx=0)
    assert entry["kind"] == "google"
    assert "client_secret" not in entry, (
        "secret literal must NOT appear in the entry; only the path"
    )
    assert entry["client_secret_file"] == str(secret_file)
    assert entry["allowed_users"] == ["alice@gmail.com", "bob@asu.edu"]


def test_google_entry_rejects_empty_allowlist(tmp_path):
    with pytest.raises(ValueError, match="at least one email"):
        _as.build_google_entry(
            client_id="c", client_secret_file=tmp_path / "x",
            allowed_users=[],
        )


# --------------------------------------------------------------------- #
#  build_auth_block + emit_molbuilder_json                              #
# --------------------------------------------------------------------- #


def test_emit_molbuilder_json_writes_0600(tmp_path):
    entry = _as.build_asu_cas_entry("jdoe")
    block = _as.build_auth_block(
        providers=[entry], secret_key_file=tmp_path / "sk",
    )
    out = tmp_path / "molbuilder.json"
    _as.emit_molbuilder_json(out, block)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    data = json.loads(out.read_text())
    assert data["auth"]["providers"][0]["kind"] == "cas"


def test_emit_preserves_other_top_level_keys(tmp_path):
    entry = _as.build_asu_cas_entry("jdoe")
    block = _as.build_auth_block([entry], tmp_path / "sk")
    out = tmp_path / "molbuilder.json"
    existing = {"envs": {"siesta": "molbuilder-siesta"}, "tls": {"cert": "/x"}}
    _as.emit_molbuilder_json(out, block, existing=existing)
    data = json.loads(out.read_text())
    # auth replaced; envs + tls survive.
    assert data["envs"] == {"siesta": "molbuilder-siesta"}
    assert data["tls"] == {"cert": "/x"}
    assert data["auth"]["providers"][0]["kind"] == "cas"


def test_emit_refuses_to_clobber_without_force(tmp_path):
    entry = _as.build_asu_cas_entry("jdoe")
    block = _as.build_auth_block([entry], tmp_path / "sk")
    out = tmp_path / "molbuilder.json"
    out.write_text('{"keep": "me"}')
    with pytest.raises(FileExistsError, match="--force"):
        _as.emit_molbuilder_json(out, block, force=False)
    # File untouched.
    assert json.loads(out.read_text()) == {"keep": "me"}


def test_emit_force_overrides_clobber_guard(tmp_path):
    entry = _as.build_asu_cas_entry("jdoe")
    block = _as.build_auth_block([entry], tmp_path / "sk")
    out = tmp_path / "molbuilder.json"
    out.write_text('{"orphan": "x"}')
    _as.emit_molbuilder_json(out, block, force=True)
    data = json.loads(out.read_text())
    assert "auth" in data
    # ``force`` without ``existing`` means we DON'T preserve old keys;
    # the orphan got cleared.  Documented in the docstring.
    assert "orphan" not in data


# --------------------------------------------------------------------- #
#  CLI: end-to-end shape for the ASU-only path                          #
# --------------------------------------------------------------------- #


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """$HOME pointed at tmp_path so the wizard writes secrets in a
    sandbox.  Clears XDG_CONFIG_HOME so default_secret_dir falls back
    to $HOME/.config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path


def test_cli_asu_only_writes_config_and_secret(isolated_home):
    out = isolated_home / "molbuilder.json"
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
        "--output", str(out),
    ], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    # molbuilder.json is mode 0600 and carries the CAS entry.
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    data = json.loads(out.read_text())
    assert data["auth"]["providers"][0]["kind"] == "cas"
    assert data["auth"]["providers"][0]["allowed_users"] == \
        ["jdoe@asu.edu"]
    # Secret file exists, 0600, non-empty.
    sk = isolated_home / ".config" / "molbuilder" / "secret_key"
    assert sk.exists()
    assert stat.S_IMODE(sk.stat().st_mode) == 0o600
    assert sk.read_text().strip()


def test_cli_asurite_defaults_to_system_user(isolated_home, monkeypatch):
    """When --asurite is not passed, the wizard prompts with the
    system user as the default.  Pressing Enter accepts that default.

    Pins the privacy contract: identity is derived from the OS-level
    account name; no other source.
    """
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    out = isolated_home / "molbuilder.json"
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--output", str(out),
    ], input="\n", catch_exceptions=False)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["auth"]["providers"][0]["allowed_users"] == \
        ["alice@asu.edu"]


def test_cli_refuses_to_clobber_without_force(isolated_home):
    out = isolated_home / "molbuilder.json"
    out.write_text('{"keep": true}')
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
        "--output", str(out),
    ], catch_exceptions=False)
    assert r.exit_code != 0
    assert "already exists" in r.output
    # File untouched.
    assert json.loads(out.read_text()) == {"keep": True}


def test_cli_force_overwrites_existing(isolated_home):
    out = isolated_home / "molbuilder.json"
    out.write_text('{"envs": {"siesta": "molbuilder-siesta"}}')
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
        "--output", str(out),
        "--force",
    ], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    # envs survived.
    assert data["envs"]["siesta"] == "molbuilder-siesta"
    # auth was added.
    assert data["auth"]["providers"][0]["kind"] == "cas"


# --------------------------------------------------------------------- #
#  CLI: secrets MUST NOT leak into the rendered config                  #
# --------------------------------------------------------------------- #


def test_cli_secrets_never_appear_in_emitted_json(isolated_home,
                                                    monkeypatch):
    """End-to-end check that no Google client_secret OR Flask session
    key ever lands inside molbuilder.json.  Mocks getpass.getpass +
    secrets.token_urlsafe so we know exactly what literals to look
    for, then asserts they're nowhere in the file."""
    sentinel_secret = "SUPER_SECRET_CLIENT_VALUE_123"
    sentinel_session = "FAKE_SESSION_KEY_456"
    monkeypatch.setattr("getpass.getpass", lambda prompt="": sentinel_secret)
    monkeypatch.setattr(
        "molbuilder.auth_setup.generate_session_secret",
        lambda: sentinel_session,
    )
    out = isolated_home / "molbuilder.json"
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "google",
        "--google-email", "alice@gmail.com",
        "--output", str(out),
    ], input="client-id-789\n", catch_exceptions=False)
    assert r.exit_code == 0, r.output
    rendered = out.read_text()
    assert sentinel_secret not in rendered, (
        "client_secret leaked into molbuilder.json"
    )
    assert sentinel_session not in rendered, (
        "Flask session key leaked into molbuilder.json"
    )
    # Sanity: the secret IS in the secret file, intact.
    google_sk = (isolated_home / ".config" / "molbuilder"
                 / "google_client_secret")
    assert google_sk.read_text() == sentinel_secret
    assert stat.S_IMODE(google_sk.stat().st_mode) == 0o600


def test_cli_google_requires_at_least_one_allowed_email(isolated_home,
                                                          monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "any")
    out = isolated_home / "molbuilder.json"
    runner = CliRunner()
    # --provider google with NO --google-email AND no interactive
    # email lines should fail.  Send empty Enter on every email prompt
    # to trigger the "need at least one" branch; immediately abort by
    # sending EOF.
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "google",
        "--output", str(out),
    ], input="client-id\n", catch_exceptions=True)
    # The wizard either re-prompts forever (in interactive shells) or
    # aborts on EOF (in CliRunner).  Either way: molbuilder.json must
    # not exist after this attempt.
    assert not out.exists()
