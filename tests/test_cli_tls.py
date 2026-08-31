"""TLS bring-up for ``molbuilder serve`` and ``molbuilder watch serve``.

The CLI grew a pair of ``--cert`` / ``--key`` flags plus a
``molbuilder.json`` config-file lookup so a deployment can flip the
Flask dev server into HTTPS without code changes.  Precedence:

    CLI flag   >   the machine config   >   plain HTTP

The middle term was ``./molbuilder.json`` until 2026-08-31; the machine scope
now has one location and no working-directory step (`configuration.md`
§ 2.1a).  The precedence is unchanged -- only where the file is found.

The resolver lives at ``molbuilder.cli._resolve_tls``; both serve
commands pass the resolved pair to ``app.run(ssl_context=...)``.

Tests cover (a) the resolver under every combination, (b) the two
serve commands wire the resolved pair into Flask, and (c) the
``--help`` surface advertises the new flags.  Every test ``chdir``s
into ``tmp_path`` so the repo-root template ``molbuilder.json`` never
leaks into the resolver.
"""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from molbuilder import cli
from molbuilder.cli import _check_tls_readable, _resolve_tls


@pytest.fixture(autouse=True)
def _tmp_path_is_the_config_root(monkeypatch, tmp_path):
    """These tests write their ``molbuilder.json`` into ``tmp_path``.

    They arranged for it to be read by ``chdir``-ing there, which was the
    reader's first candidate.  That step is gone, so the directory is named
    outright instead -- and this file had NO other isolation, so without it
    the resolver would answer from the developer's own machine config.

    `conftest.config_root` is the general form; this file writes to
    ``tmp_path`` by name throughout.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))


# --------------------------------------------------------------------- #
#  _resolve_tls -- the precedence + parsing kernel                      #
# --------------------------------------------------------------------- #


def test_resolve_no_json_no_flags_returns_none_pair(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _resolve_tls(None, None) == (None, None)


def test_resolve_json_supplies_both_when_no_flags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": "/etc/ssl/c.pem", "key": "/etc/ssl/k.pem"}))
    assert _resolve_tls(None, None) == ("/etc/ssl/c.pem", "/etc/ssl/k.pem")


def test_resolve_cli_flag_overrides_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": "/from/json.pem", "key": "/from/json.key"}))
    # CLI cert wins, JSON key fills the gap.
    assert _resolve_tls("/from/cli.pem", None) == ("/from/cli.pem", "/from/json.key")
    # Both flags win outright.
    assert _resolve_tls("/cli.pem", "/cli.key") == ("/cli.pem", "/cli.key")


def test_resolve_json_with_only_cert_falls_back_to_http(
        monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps({"cert": "/c.pem"}))
    assert _resolve_tls(None, None) == (None, None)
    err = capsys.readouterr().err
    assert "incomplete" in err
    assert "HTTP" in err


def test_resolve_json_with_only_key_falls_back_to_http(
        monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps({"key": "/k.pem"}))
    assert _resolve_tls(None, None) == (None, None)
    assert "incomplete" in capsys.readouterr().err


def test_resolve_cli_flag_alone_without_pair_falls_back(
        monkeypatch, tmp_path, capsys):
    """``--cert`` without ``--key`` (and no JSON) is incomplete -- warn
    and fall back to HTTP rather than silently shipping half a pair."""
    monkeypatch.chdir(tmp_path)
    assert _resolve_tls("/c.pem", None) == (None, None)
    assert "incomplete" in capsys.readouterr().err


def test_resolve_malformed_json_raises_usage_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text("{ this is not json")
    with pytest.raises(click.UsageError, match="invalid JSON"):
        _resolve_tls(None, None)


def test_main_exits_cleanly_when_molbuilder_json_is_malformed(
        monkeypatch, tmp_path, capsys):
    """A malformed ``molbuilder.json`` should produce a clean
    ``Error: ...`` line on stderr + ``SystemExit(2)`` from ``cli.main``
    -- the same surface every other UsageError gets -- instead of a
    Python traceback from the diagnostics ``initialize()`` call."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text("{ this is not json")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "invalid JSON" in err


def test_resolve_refuses_unknown_keys(monkeypatch, tmp_path):
    """U7 (2026-08-12): an unknown top-level key is REFUSED with the known
    sections named -- 'tolerated so the file can grow' is the exact hole
    that silently ate admin/rate_limit.  This test fabricated a 'host'
    key (no consumer anywhere) to pin the retired tolerance, and lived in
    a file the fix session's hand-rolled batteries never ran -- the fresh
    full testrun.py batch is what caught it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": "/c.pem", "key": "/k.pem", "host": "example.com"}))
    with pytest.raises(Exception, match="unknown top-level key.*host"):
        _resolve_tls(None, None)
    # the flat tls aliases themselves remain read, as ever
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": "/c.pem", "key": "/k.pem"}))
    assert _resolve_tls(None, None) == ("/c.pem", "/k.pem")


# --------------------------------------------------------------------- #
#  _check_tls_readable -- pre-flight readability gate                   #
# --------------------------------------------------------------------- #
#
# The 2026-05-18 bug report: `serve` with a molbuilder.json pointing at
# /etc/letsencrypt/live/<domain>/{fullchain,privkey}.pem (root-owned,
# privkey 0600) crashed deep in Werkzeug with a bare
# ``PermissionError: [Errno 13] Permission denied`` -- no indication
# of which file failed.  ``_check_tls_readable`` runs before
# ``app.run`` and surfaces a click.UsageError naming the bad path +
# pointing at the reverse-proxy approach in docs/ops/deployment.md.


def test_check_tls_readable_noop_on_falsy_inputs():
    """No TLS configured -- the call is a no-op (no exception).  Pins
    the "skip when there's nothing to check" branch so callers don't
    have to gate on ``if cert and key`` themselves."""
    assert _check_tls_readable(None, None) is None
    assert _check_tls_readable("", "") is None
    assert _check_tls_readable("/path", None) is None
    assert _check_tls_readable(None, "/path") is None


def test_check_tls_readable_accepts_readable_pair(tmp_path):
    """Happy path: both files exist + are readable to this process."""
    cert = tmp_path / "cert.pem"
    key  = tmp_path / "key.pem"
    cert.write_bytes(b"---PEM---\n")
    key.write_bytes(b"---PEM---\n")
    # Returns None on success; absence of an exception is the contract.
    assert _check_tls_readable(str(cert), str(key)) is None


def test_check_tls_readable_rejects_missing_cert(tmp_path):
    """Missing cert path -- the error names the cert path + the OS
    reason (No such file or directory).  Catches the "operator typoed
    the path in molbuilder.json" case BEFORE the server tries to
    bind."""
    key = tmp_path / "key.pem"
    key.write_bytes(b"---PEM---\n")
    missing_cert = tmp_path / "does-not-exist.pem"
    with pytest.raises(click.UsageError) as excinfo:
        _check_tls_readable(str(missing_cert), str(key))
    msg = str(excinfo.value)
    assert "cert:" in msg
    assert str(missing_cert) in msg
    # Either "No such file" or "FileNotFoundError" -- both are
    # acceptable surfaces of the same condition.
    assert ("No such file" in msg or "FileNotFoundError" in msg), msg


def test_check_tls_readable_rejects_missing_key(tmp_path):
    """Missing key path -- error names ``key:`` (not ``cert:``) so the
    operator knows which file to fix."""
    cert = tmp_path / "cert.pem"
    cert.write_bytes(b"---PEM---\n")
    missing_key = tmp_path / "does-not-exist.pem"
    with pytest.raises(click.UsageError) as excinfo:
        _check_tls_readable(str(cert), str(missing_key))
    msg = str(excinfo.value)
    assert "key:" in msg
    assert str(missing_key) in msg


def test_check_tls_readable_reports_both_failures_at_once(tmp_path):
    """Both paths bad -- both surface in one error so the operator
    fixes them together rather than playing whack-a-mole (fix cert,
    rerun, see key error, fix key, rerun).  Saves a server restart
    cycle."""
    missing_cert = tmp_path / "no-cert.pem"
    missing_key  = tmp_path / "no-key.pem"
    with pytest.raises(click.UsageError) as excinfo:
        _check_tls_readable(str(missing_cert), str(missing_key))
    msg = str(excinfo.value)
    assert str(missing_cert) in msg
    assert str(missing_key)  in msg


def test_check_tls_readable_rejects_unreadable_key_mode_0000(tmp_path):
    """The actual incident from 2026-05-18: cert exists, key exists,
    but key has mode 0 (analogue of root-owned 0600 cert that the
    molbuilder user can't read).  Real Let's Encrypt setups put 0600
    on privkey + the cert install is owned by root; running molbuilder
    as a non-root user trips this.

    Skipped when running as root since root can read mode-0 files."""
    import os
    if os.geteuid() == 0:
        pytest.skip("root bypasses mode bits; test only meaningful "
                    "as non-root")
    cert = tmp_path / "cert.pem"
    key  = tmp_path / "key.pem"
    cert.write_bytes(b"---PEM---\n")
    key.write_bytes(b"---PEM---\n")
    key.chmod(0o000)
    try:
        with pytest.raises(click.UsageError) as excinfo:
            _check_tls_readable(str(cert), str(key))
        msg = str(excinfo.value)
        assert "key:" in msg
        assert str(key) in msg
        assert "Permission denied" in msg, msg
    finally:
        # Restore so tmp_path teardown can rm-rf cleanly.
        key.chmod(0o600)


def test_check_tls_readable_error_mentions_deployment_doc(tmp_path):
    """The error body must point at the recommended deploy shape so
    the operator doesn't reach for ``chmod 0644 privkey.pem`` (which
    weakens security to make the error go away).  Pins that the
    docs/ops/deployment.md mention + the reverse-proxy suggestion both
    show up in the failure message."""
    with pytest.raises(click.UsageError) as excinfo:
        _check_tls_readable(str(tmp_path / "no.pem"),
                            str(tmp_path / "no.key"))
    msg = str(excinfo.value)
    assert "reverse proxy" in msg
    assert "docs/ops/deployment.md" in msg


# --------------------------------------------------------------------- #
#  serve / watch serve -- end-to-end wiring                             #
# --------------------------------------------------------------------- #


@pytest.fixture
def capture_flask_run(monkeypatch):
    """Replace ``Flask.run`` with a no-op that records its kwargs.

    Returns a list that gets one dict appended per call.  Using the
    class attribute means every ``create_app()`` instance picks it up
    without us needing to intercept the factory.
    """
    calls = []

    def fake_run(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    from flask import Flask
    monkeypatch.setattr(Flask, "run", fake_run)
    return calls


def _touch(path):
    path.write_text("-----PEM-----")
    return str(path)


def test_serve_no_tls_passes_none_ssl_context(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(
        cli.cli, ["serve", "foreground", "--port", "0", "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert capture_flask_run, "Flask.run was not called"
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] is None
    assert "http://" in res.stderr  # log line scheme
    assert "https://" not in res.stderr


def test_serve_json_only_engages_https(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    key  = _touch(tmp_path / "k.pem")
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": cert, "key": key}))
    res = CliRunner().invoke(
        cli.cli, ["serve", "foreground", "--port", "0", "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] == (cert, key)
    assert "https://" in res.stderr


def test_serve_cli_flags_engage_https(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    key  = _touch(tmp_path / "k.pem")
    res = CliRunner().invoke(cli.cli, [
        "serve", "foreground", "--port", "0", "--cert", cert, "--key", key,
        "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] == (cert, key)
    assert "https://" in res.stderr


def test_serve_cli_cert_only_warns_and_falls_back(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    res = CliRunner().invoke(cli.cli, [
        "serve", "foreground", "--port", "0", "--cert", cert, "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] is None
    assert "incomplete" in res.stderr
    assert "http://" in res.stderr


def test_serve_help_advertises_cert_and_key():
    res = CliRunner().invoke(cli.cli, ["serve", "foreground", "--help"])
    assert res.exit_code == 0
    assert "--cert" in res.output
    assert "--key" in res.output


# ``molbuilder watch serve`` (legacy alias of ``molbuilder serve``)
# removed 2026-05-19 along with the /watch page; TLS-wiring tests
# for the canonical ``molbuilder serve`` command above already
# cover the same precedence + readability paths.


# --------------------------------------------------------------------- #
#  .gitignore must list the runtime config so it never gets committed   #
# --------------------------------------------------------------------- #


def test_gitignore_excludes_molbuilder_json():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    gi = (repo_root / ".gitignore").read_text().splitlines()
    assert "molbuilder.json" in gi, \
        "molbuilder.json must be gitignored so per-machine TLS paths " \
        "don't sync to the remote."
