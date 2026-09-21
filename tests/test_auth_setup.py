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


# RETIRED 2026-09-13 (I8): `test_default_secret_dir_follows_the_one_override`
# and `test_default_secret_dir_honors_xdg` stood here.  Both asserted
# `config_dir()`'s behaviour through `auth_setup.default_secret_dir`, a
# one-line pass-through with no production caller -- so the alias existed
# because these tests asserted it, and they asserted `config_dir` through it.
# `test_config_dir_has_one_home.py` owns both facts against the real door:
# `TestTheRootCanBeNamedOutright::test_it_is_used_exactly_as_given` for the
# override, `test_empty_is_not_set` for the XDG fallback.  The alias is gone.


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


def test_write_secret_file_failure_leaves_the_previous_secret_intact(tmp_path):
    """A write that fails must not destroy what was there (§ 2.3).

    Until 2026-09-12 this function opened the target `O_TRUNC` and only THEN
    produced the bytes, so anything that went wrong after the open left an
    EMPTY secret file -- for `notify_keys`, every key the operator had ever
    issued.  Through the atomic writer the old content is either fully
    replaced or fully untouched.

    The trigger is a lone surrogate, which `str.encode("utf-8")` refuses.  It
    needs no fake and no monkeypatching: it stands in for every way the bytes
    can fail to land once the decision to write has been made, and it lands on
    the same side of the truncation as a full disk or a kill does.
    """
    p = tmp_path / "notify_keys"
    p.write_text('{"route": "abc", "keys": {"me": "real-key"}}')
    os.chmod(p, 0o600)
    with pytest.raises(UnicodeEncodeError):
        _as.write_secret_file(p, "\ud800")
    assert p.read_text() == '{"route": "abc", "keys": {"me": "real-key"}}'
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    # No temp litter beside it either -- a half-written file next to a secret
    # is a second copy of that secret at whatever mode it got.
    assert [x.name for x in tmp_path.iterdir()] == ["notify_keys"]


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
    # The wizard must NOT write `service_validate_url`: nothing reads it, and a
    # key in a config the client cannot consult is a lie about what it does.
    assert "service_validate_url" not in entry
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
#  CLI: end-to-end shape for the ASU-only path                          #
# --------------------------------------------------------------------- #


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """$HOME pointed at tmp_path so the wizard writes secrets in a
    sandbox.  Clears XDG_CONFIG_HOME so `config_dir()` falls back
    to $HOME/.config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path


def _machine_file(home):
    """The one file the server reads, under the isolated home."""
    p = home / ".config" / "molbuilder" / "molbuilder.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_cli_asu_only_writes_the_config_and_keeps_the_session_key(
        isolated_home):
    """The session key is the server's.  The wizard regenerated it on every
    run until 2026-09-13 -- every signed-in person logged out by a command
    that called itself idempotent -- so the key a server already made must
    come out of the wizard byte for byte as it went in."""
    out = _machine_file(isolated_home)
    # ASK THE DOOR.  This was `out.parent / "secret_key"`, and when the key
    # moved into `secrets/` the test went VACUOUS rather than red: it wrote a
    # file the wizard no longer touches and asserted nobody had changed it,
    # which nobody would.  Through `session_key()` it again pins the thing it
    # was written for -- that `auth-setup` preserves a key the server made.
    from molbuilder.config_dir import session_key
    sk = session_key()
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_bytes(b"the-server-made-this-key-on-first-start")
    r = CliRunner().invoke(cli, [
        "auth-setup", "--provider", "asu", "--asurite", "jdoe",
    ], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    # molbuilder.json is mode 0600 and carries the CAS entry.
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    data = json.loads(out.read_text())
    assert data["auth"]["providers"][0]["kind"] == "cas"
    assert data["auth"]["providers"][0]["allowed_users"] == \
        ["jdoe@asu.edu"]
    assert sk.read_bytes() == b"the-server-made-this-key-on-first-start"


class TestTheWizardWritesWhereTheReaderReads:
    """Where ``auth-setup`` puts ``molbuilder.json`` (2026-08-30).

    It defaulted to ``./molbuilder.json`` -- wherever the wizard happened to
    be launched from, which for anyone running it inside a checkout is the
    git root.  Wrong twice over: the same command already writes both SECRETS
    into the per-user config directory, so one command split its output across
    two conventions; and on a machine that already has a ``./molbuilder.json``,
    writing a fresh per-user file would have produced a config **the reader
    never looks at**, with the wizard reporting success while sign-in stayed
    off.

    So the default is the reader's own answer -- ``machine_config_path()``.

    **That answer had two branches until 2026-08-31 and now has one**: the
    machine scope lives in the config directory, and a `./molbuilder.json` is
    not read at all (`configuration.md` § 2.1a).  The pair of tests below used
    to drive both branches; the second now asserts the opposite of what it once
    did, because "always write the per-user file" -- the rule this class was
    written to disprove -- became correct when the other place stopped being
    read.
    """

    def _run(self, extra=()):
        return CliRunner().invoke(cli, [
            "auth-setup", "--provider", "asu", "--asurite", "jdoe", *extra,
        ], catch_exceptions=False)

    def test_with_no_cwd_config_it_writes_the_per_user_one(
            self, isolated_home, monkeypatch, tmp_path):
        run_dir = tmp_path / "somewhere-else"
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)

        r = self._run()
        assert r.exit_code == 0, r.output

        xdg = isolated_home / ".config" / "molbuilder" / "molbuilder.json"
        assert xdg.is_file(), r.output
        assert not (run_dir / "molbuilder.json").exists(), (
            "the wizard wrote into the directory it was launched from")
        assert json.loads(xdg.read_text())["auth"]["providers"][0]["kind"] == "cas"
        assert stat.S_IMODE(xdg.stat().st_mode) == 0o600
        # ...and it does not tell the user to cd anywhere: a per-user config is
        # read from any directory, and saying otherwise teaches the wrong model.
        assert "cd " not in r.output or "read from anywhere" in r.output

    def test_a_cwd_config_does_not_attract_it(
            self, isolated_home, monkeypatch, tmp_path):
        """The inverted half: a file in the launch directory is not the target.

        It once was -- the reader took it, so writing anywhere else would have
        left the auth block where nothing looks.  Now the reader never opens
        it, and writing there would be the mistake instead.  The stray file is
        left untouched, because the wizard has no business editing a file the
        program does not read.
        """
        run_dir = tmp_path / "deployment"
        run_dir.mkdir()
        stray = run_dir / "molbuilder.json"
        stray.write_text(
            json.dumps({"script_generation": {"activation": "conda activate"}}))
        monkeypatch.chdir(run_dir)

        r = self._run(("--force",))
        assert r.exit_code == 0, r.output

        xdg = isolated_home / ".config" / "molbuilder" / "molbuilder.json"
        assert json.loads(xdg.read_text())["auth"]["providers"][0]["kind"] == "cas", (
            "the auth block did not land in the one file the reader opens")
        assert json.loads(stray.read_text()) == {
            "script_generation": {"activation": "conda activate"}}, (
            "the wizard edited a file the program does not read")

    def test_it_merges_into_what_is_already_at_the_one_location(
            self, isolated_home, monkeypatch, tmp_path):
        """The wizard adds a section; it does not replace the file.

        This was asserted against the cwd branch before that branch existed
        no more, and the property is the one worth keeping: a machine with
        `execution` or `scheduler` already set must not lose them to a
        sign-in setup.
        """
        monkeypatch.chdir(tmp_path)
        target = isolated_home / ".config" / "molbuilder" / "molbuilder.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"script_generation": {"activation": "conda activate"}}))

        r = self._run(("--force",))
        assert r.exit_code == 0, r.output

        after = json.loads(target.read_text())
        assert after["auth"]["providers"][0]["kind"] == "cas"
        assert after["script_generation"]["activation"] == "conda activate", (
            "the wizard replaced the file instead of merging into it")

def test_cli_asurite_defaults_to_system_user(isolated_home, monkeypatch):
    """When --asurite is not passed, the wizard prompts with the
    system user as the default.  Pressing Enter accepts that default.

    Pins the privacy contract: identity is derived from the OS-level
    account name; no other source.
    """
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    out = _machine_file(isolated_home)
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
    ], input="\n", catch_exceptions=False)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["auth"]["providers"][0]["allowed_users"] == \
        ["alice@asu.edu"]


def test_cli_refuses_to_clobber_without_force(isolated_home):
    """WHAT --force GUARDS IS AN AUTH BLOCK, which is what it says it guards:
    *"overwrite an existing molbuilder.json's auth block.  Other top-level
    sections (envs, tls, ...) survive."*

    This asserted refusal on ANY existing file until 2026-09-08 -- stricter
    than the flag's own help, and stricter than the writer it guarded, whose
    docstring promises that "an install that already has e.g. ``envs`` or
    ``tls`` sections stays intact".  That promise described a path only
    --force could reach.  It went unnoticed while a fresh machine had no
    molbuilder.json; `envs bootstrap` now seeds one, so the old rule would
    have refused this wizard on every fresh install.
    """
    out = _machine_file(isolated_home)
    # A REAL block: the wizard reads the file through the server's reader
    # now, and a provider entry the server would refuse stops it for that
    # reason, not this one.
    out.write_text(json.dumps(
        {"auth": _as.build_auth_block([_as.build_asu_cas_entry("old")])}))
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
    ], catch_exceptions=False)
    assert r.exit_code != 0
    assert "auth providers" in r.output
    # File untouched.
    assert json.loads(out.read_text())["auth"]["providers"][0][
        "allowed_users"] == ["old@asu.edu"]


def test_cli_merges_into_a_config_that_has_no_auth_block(isolated_home):
    """The other half of the same rule, and the one a fresh install takes.

    A seeded molbuilder.json carries `script_generation` and no `auth`; there
    is nothing to clobber, so the wizard writes its block and leaves the rest.
    """
    out = _machine_file(isolated_home)
    out.write_text('{"script_generation": {"activation": "conda activate"}}')
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
    ], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["auth"]["providers"], "the wizard wrote its block"
    assert data["script_generation"]["activation"] == "conda activate"


def test_cli_force_replaces_the_providers_and_nothing_else(isolated_home):
    """--force replaces the providers LIST.  The wizard's own writer replaced
    `auth` wholesale until 2026-09-13, so re-running it to add a provider
    dropped `auth.trust_proxy`; through the door the write is a merge."""
    out = _machine_file(isolated_home)
    out.write_text(json.dumps({
        "envs": {"siesta": "molbuilder-siesta"},
        "auth": {**_as.build_auth_block([_as.build_asu_cas_entry("old")]),
                 "trust_proxy": True},
    }))
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "asu",
        "--asurite", "jdoe",
        "--force",
    ], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["envs"]["siesta"] == "molbuilder-siesta"
    assert data["auth"]["trust_proxy"] is True, (
        "a sibling key in `auth` was dropped by the providers replacement")
    assert [p["allowed_users"] for p in data["auth"]["providers"]] == [
        ["jdoe@asu.edu"]]


# --------------------------------------------------------------------- #
#  CLI: secrets MUST NOT leak into the rendered config                  #
# --------------------------------------------------------------------- #


def test_cli_secrets_never_appear_in_emitted_json(isolated_home,
                                                    monkeypatch):
    """End-to-end check that the Google client_secret never lands inside
    molbuilder.json.  Mocks getpass.getpass so we know exactly what literal
    to look for, then asserts it is nowhere in the file."""
    sentinel_secret = "SUPER_SECRET_CLIENT_VALUE_123"
    monkeypatch.setattr("getpass.getpass", lambda prompt="": sentinel_secret)
    out = _machine_file(isolated_home)
    runner = CliRunner()
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "google",
        "--google-email", "alice@gmail.com",
    ], input="client-id-789\n", catch_exceptions=False)
    assert r.exit_code == 0, r.output
    rendered = out.read_text()
    assert sentinel_secret not in rendered, (
        "client_secret leaked into molbuilder.json"
    )
    # Sanity: the secret IS in the secret file, intact.  ASK THE RESOLVER --
    # this spelled ".config/molbuilder/google_client_secret" by hand until
    # 2026-09-20 and broke the day every credential moved into `secrets/`.
    from molbuilder.config_dir import google_client_secret as _google_sk
    google_sk = _google_sk()
    assert google_sk.read_text() == sentinel_secret
    assert stat.S_IMODE(google_sk.stat().st_mode) == 0o600


def test_cli_google_requires_at_least_one_allowed_email(isolated_home,
                                                          monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "any")
    out = _machine_file(isolated_home)
    runner = CliRunner()
    # --provider google with NO --google-email AND no interactive
    # email lines should fail.  Send empty Enter on every email prompt
    # to trigger the "need at least one" branch; immediately abort by
    # sending EOF.
    r = runner.invoke(cli, [
        "auth-setup",
        "--provider", "google",
    ], input="client-id\n", catch_exceptions=True)
    # The wizard either re-prompts forever (in interactive shells) or
    # aborts on EOF (in CliRunner).  Either way: molbuilder.json must
    # not exist after this attempt.
    assert not out.exists()


class _App:
    """Just the surface `_install_secret_key` touches."""
    def __init__(self):
        self.config = {}


def test_first_server_run_creates_the_session_key_owner_only(isolated_home):
    """The key is made at its one home, 0600, and signs with what it wrote."""
    from molbuilder.config_dir import session_key
    from molbuilder.web.auth import _install_secret_key

    app = _App()
    _install_secret_key(app)

    sk = session_key()
    assert sk.exists(), "first run must create the key"
    assert stat.S_IMODE(sk.stat().st_mode) == 0o600
    assert app.config["SECRET_KEY"] == sk.read_bytes()
    assert len(sk.read_bytes()) >= 16


def test_losing_the_first_run_race_adopts_the_other_start_s_key(
        isolated_home, monkeypatch):
    """Two servers starting together: the loser signs with the WINNER's key.

    The interleaving is the defect, so it is what the test arranges: this
    process sees no key, another start creates one, and only then does this
    process write.  A plain replace would be quieter and worse -- the file
    would hold the loser's key and the winner's sessions would die at its next
    restart.  Neither: the write is refused and the existing key is adopted.
    """
    from molbuilder import config_dir as cd
    from molbuilder.web import auth as web_auth
    from molbuilder.config_dir import ensure_private_dir, session_key

    winner = b"w" * 32
    sk = session_key()
    ensure_private_dir(sk.parent)
    sk.write_bytes(winner)          # the other start got there first

    # ...but this process looked BEFORE that happened.  One `None`, then the
    # truth, which is exactly what the two calls in the function see.
    real = cd.read_session_key
    calls = {"n": 0}

    def racy():
        calls["n"] += 1
        return None if calls["n"] == 1 else real()
    # ON `config_dir`, NOT on `web.auth`: the function imports the door inside
    # its own body, so it gets a fresh reference every call and a patch on the
    # importing module would simply not be seen.
    monkeypatch.setattr(cd, "read_session_key", racy)

    app = _App()
    _install_secret_key = web_auth._install_secret_key
    _install_secret_key(app)        # must not raise FileExistsError

    assert app.config["SECRET_KEY"] == winner, "the loser must adopt"
    assert sk.read_bytes() == winner, "the winner's key must survive"
