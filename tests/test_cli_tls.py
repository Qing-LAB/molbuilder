"""TLS bring-up for ``molbuilder serve`` and ``molbuilder watch serve``.

The CLI grew a pair of ``--cert`` / ``--key`` flags plus a
``molbuilder.json`` config-file lookup so a deployment can flip the
Flask dev server into HTTPS without code changes.  Precedence:

    CLI flag   >   ./molbuilder.json   >   plain HTTP

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
from molbuilder.cli import _resolve_tls


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


def test_resolve_ignores_unknown_keys(monkeypatch, tmp_path):
    """Extra keys are tolerated -- only ``cert`` and ``key`` are read.
    Lets the file grow non-TLS sections later without breaking now."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": "/c.pem", "key": "/k.pem", "host": "example.com"}))
    assert _resolve_tls(None, None) == ("/c.pem", "/k.pem")


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
    res = CliRunner().invoke(cli.cli, ["serve", "--port", "0"])
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
    res = CliRunner().invoke(cli.cli, ["serve", "--port", "0"])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] == (cert, key)
    assert "https://" in res.stderr


def test_serve_cli_flags_engage_https(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    key  = _touch(tmp_path / "k.pem")
    res = CliRunner().invoke(cli.cli, [
        "serve", "--port", "0", "--cert", cert, "--key", key])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] == (cert, key)
    assert "https://" in res.stderr


def test_serve_cli_cert_only_warns_and_falls_back(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    res = CliRunner().invoke(cli.cli, [
        "serve", "--port", "0", "--cert", cert])
    assert res.exit_code == 0, res.output
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] is None
    assert "incomplete" in res.stderr
    assert "http://" in res.stderr


def test_serve_help_advertises_cert_and_key():
    res = CliRunner().invoke(cli.cli, ["serve", "--help"])
    assert res.exit_code == 0
    assert "--cert" in res.output
    assert "--key" in res.output


def test_watch_serve_no_tls_passes_none_ssl_context(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(cli.cli, [
        "watch", "serve", "--host", "127.0.0.1", "--port", "0"])
    assert res.exit_code == 0, res.output
    kwargs = capture_flask_run[-1]["kwargs"]
    assert kwargs["ssl_context"] is None
    assert kwargs["threaded"] is True   # preserved
    assert "http://" in res.stderr


def test_watch_serve_json_engages_https(
        monkeypatch, tmp_path, capture_flask_run):
    monkeypatch.chdir(tmp_path)
    cert = _touch(tmp_path / "c.pem")
    key  = _touch(tmp_path / "k.pem")
    (tmp_path / "molbuilder.json").write_text(json.dumps(
        {"cert": cert, "key": key}))
    res = CliRunner().invoke(cli.cli, [
        "watch", "serve", "--host", "127.0.0.1", "--port", "0"])
    assert res.exit_code == 0, res.output
    kwargs = capture_flask_run[-1]["kwargs"]
    assert kwargs["ssl_context"] == (cert, key)
    assert kwargs["threaded"] is True
    # Both helper log lines mention the resolved scheme.
    assert "https://127.0.0.1:0/" in res.stderr
    assert "https://127.0.0.1:0/watch" in res.stderr


def test_watch_serve_help_advertises_cert_and_key():
    res = CliRunner().invoke(cli.cli, ["watch", "serve", "--help"])
    assert res.exit_code == 0
    assert "--cert" in res.output
    assert "--key" in res.output


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
