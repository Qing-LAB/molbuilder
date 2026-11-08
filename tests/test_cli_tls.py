"""TLS bring-up for ``molbuilder serve``.

A pair of ``--cert`` / ``--key`` flags plus a ``molbuilder.json``
config-file lookup let a deployment flip the Flask dev server into HTTPS
without code changes.  Precedence:

    CLI flag   >   the machine config   >   plain HTTP

The resolver lives at ``molbuilder.cli._resolve_tls``; ``serve`` passes
the resolved pair to ``app.run(ssl_context=...)``.

Tests cover the pair the serve command hands Flask, under the flags and
the config as a person gives them, and the readability gate before it.
The resolver reads the config directory's ``molbuilder.json`` alone, which
the fixture below points at ``tmp_path``.
"""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from molbuilder import cli
from molbuilder.cli import _check_tls_readable


@pytest.fixture(autouse=True)
def _tmp_path_is_the_config_root(monkeypatch, tmp_path):
    """These tests write their ``molbuilder.json`` into ``tmp_path``.

    The directory is named outright: without it the resolver would answer
    from the developer's own machine config.

    `conftest.config_root` is the general form; this file writes to
    ``tmp_path`` by name throughout.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))


# --------------------------------------------------------------------- #
#  the machine config, read where it is read                            #
# --------------------------------------------------------------------- #


def test_a_malformed_molbuilder_json_is_refused_where_it_is_read(
        monkeypatch, tmp_path, capsys):
    """A malformed ``molbuilder.json`` is said by the command that reads it
    -- a clean ``Error: ...`` line on stderr and ``SystemExit(2)`` from
    ``cli.main``, the surface every other UsageError gets, never a
    traceback -- and a command that reads no config, ``--help``, runs
    (plan W36 ⑥).  Until 2026-09-29 `cli.main` took the machine snapshot
    before it parsed anything, so a broken file failed even ``--help``."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "molbuilder.json").write_text("{ this is not json")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    assert "Usage:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["envs", "list"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "invalid JSON" in err


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


def test_check_tls_readable_error_names_the_recommended_shape(tmp_path):
    """The error body must point at the recommended deploy shape so
    the operator doesn't reach for ``chmod 0644 privkey.pem`` (which
    weakens security to make the error go away): the reverse proxy."""
    with pytest.raises(click.UsageError) as excinfo:
        _check_tls_readable(str(tmp_path / "no.pem"),
                            str(tmp_path / "no.key"))
    msg = str(excinfo.value)
    assert "reverse proxy" in msg


# --------------------------------------------------------------------- #
#  serve -- end-to-end wiring                                           #
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


#: Each row: what the machine config's ``tls`` names -- ``None``, no
#: ``molbuilder.json`` at all -- the flags typed, the pair `serve` hands
#: Flask -- ``None`` for plain HTTP -- and what it says.
_SERVE_ROWS = [
    dict(name="no config and no flag: plain HTTP",
         config=None, flags={}, served=None, said=[]),
    dict(name="the config's pair alone: HTTPS on it",
         config={"cert": "json.pem", "key": "json.key"}, flags={},
         served=("json.pem", "json.key"), said=[]),
    dict(name="both flags, no config: HTTPS on the flags' pair",
         config=None, flags={"--cert": "cli.pem", "--key": "cli.key"},
         served=("cli.pem", "cli.key"), said=[]),
    dict(name="a cert flag alone, no config: plain HTTP, said incomplete",
         config=None, flags={"--cert": "cli.pem"}, served=None,
         said=["incomplete"]),
    dict(name="a flag over the config: the flag's cert, the config's key",
         config={"cert": "json.pem", "key": "json.key"},
         flags={"--cert": "cli.pem"}, served=("cli.pem", "json.key"),
         said=[]),
    dict(name="a key with no cert in the config: plain HTTP, said incomplete",
         config={"key": "json.key"}, flags={}, served=None,
         said=["incomplete"]),
]


@pytest.mark.parametrize("row", _SERVE_ROWS,
                         ids=[r["name"] for r in _SERVE_ROWS])
def test_serve_takes_the_pair_the_precedence_gives(
        row, monkeypatch, tmp_path, capture_flask_run):
    """`serve` hands Flask the pair the precedence above gives: a flag over
    the machine config, the half it does not name filled from the config;
    a pair still incomplete, plain HTTP, and said; neither naming one,
    plain HTTP -- and its log line names the scheme it serves, and only
    that one.  THE SILENT FAILURE: the server up on the wrong cert and key,
    or on plain HTTP with no word."""
    monkeypatch.chdir(tmp_path)
    at = {n: _touch(tmp_path / n)
          for n in ("json.pem", "json.key", "cli.pem", "cli.key")}
    if row["config"] is not None:
        (tmp_path / "molbuilder.json").write_text(json.dumps(
            {"tls": {k: at[v] for k, v in row["config"].items()}}))
    flags = [w for flag, name in row["flags"].items() for w in (flag, at[name])]
    res = CliRunner().invoke(cli.cli, ["serve", "foreground", "--port", "0",
                                       "--no-supervise", *flags])
    assert res.exit_code == 0, res.output
    want = tuple(at[n] for n in row["served"]) if row["served"] else None
    assert capture_flask_run[-1]["kwargs"]["ssl_context"] == want
    for words in row["said"]:
        assert words in res.stderr, res.stderr
    assert ("https://" if want else "http://") in res.stderr
    assert ("http://" if want else "https://") not in res.stderr
