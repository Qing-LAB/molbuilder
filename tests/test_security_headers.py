"""The server's security posture, as `ops/access-control.md` states it.

Two gaps that document named as known (plan W49): the header VALUES were
untested -- only the no-inline-script rule the CSP depends on had a test --
and the bind guard's refusal of a public interface without TLS was pinned
only through the ``--no-auth`` case.  Both are asked of the product's own
doors: the app's responses, and ``molbuilder serve``.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


@pytest.mark.parametrize("path", ["/api/health", "/results"])
def test_every_response_carries_the_security_headers(client, path):
    """`app.py`'s `_add_security_headers`, on an API answer and on a page:
    scripts only from molbuilder's own origin and never inline, no plugins,
    no framing, no MIME sniffing, the referrer kept to the origin."""
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code)
    csp = r.headers.get("Content-Security-Policy", "")
    directives = {d.split()[0]: d.split()[1:]
                  for d in (p.strip() for p in csp.split(";")) if d}
    assert directives.get("default-src") == ["'self'"], csp
    assert directives.get("script-src") == ["'self'"], csp
    assert directives.get("object-src") == ["'none'"], csp
    assert directives.get("frame-ancestors") == ["'none'"], csp
    assert directives.get("base-uri") == ["'self'"], csp
    assert directives.get("form-action") == ["'self'"], csp
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "same-origin"


def test_hsts_is_sent_over_https_and_only_over_https(client):
    """A browser honours HSTS only over HTTPS, and a plain-HTTP deploy that
    sent it would lock browsers out -- so it rides HTTPS answers alone:
    direct TLS, or a proxy that says it terminated TLS."""
    hsts = "Strict-Transport-Security"
    assert hsts not in client.get("/api/health").headers
    over_tls = client.get("/api/health", base_url="https://localhost")
    assert over_tls.headers.get(hsts) == "max-age=31536000; includeSubDomains"
    via_proxy = client.get("/api/health",
                           headers={"X-Forwarded-Proto": "https"})
    assert via_proxy.headers.get(hsts) == over_tls.headers.get(hsts)


def test_a_public_bind_without_tls_is_refused(tmp_path, monkeypatch):
    """`serve --host 0.0.0.0` with no certificate is refused before anything
    binds: every request would cross the network in clear text, the session
    cookie included (`cli._enforce_tls_for_remote_bind`).  The machine config
    is the test's own, so no certificate can be found by accident, and the
    app's `run` is a stand-in that records -- a server that started would be
    the failure, not a hang."""
    from click.testing import CliRunner

    import molbuilder.web.app as appmod
    from molbuilder import cli

    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    started = []

    class _App:
        config: dict = {}

        def run(self, **kw):
            started.append(kw)

    monkeypatch.setattr(appmod, "create_app", lambda **_: _App())
    res = CliRunner().invoke(cli.cli, ["serve", "foreground", "--host",
                                       "0.0.0.0", "--port", "8099",
                                       "--no-supervise"])
    assert res.exit_code != 0, res.output
    assert "is not a loopback address and there is no TLS" in res.output
    assert not started, "a public bind without TLS started a server"
