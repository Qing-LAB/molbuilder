"""The server's security posture, as `ops/access-control.md` states it:
the header values, no page carrying the inline JavaScript the CSP blocks,
and the bind guard's refusal of a public interface without TLS -- each
asked of the product's own doors: the app's responses, and
``molbuilder serve``.
"""
from __future__ import annotations

from html.parser import HTMLParser

import pytest


@pytest.fixture
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


#: `web-api.md` § 2's Content-Security-Policy, directive by directive.
_POLICY = {
    "default-src": ["'self'"],
    "script-src": ["'self'"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:"],
    "connect-src": ["'self'"],
    "font-src": ["'self'"],
    "object-src": ["'none'"],
    "frame-ancestors": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
}


@pytest.mark.parametrize("path,more", [
    ("/api/health", {}),
    ("/results", {}),
    # The JupyterNB tab frames the notebook, another origin: the same
    # policy with `frame-src` added (`web/jupyter.md`), `'none'` when the
    # server was started on no port this request knows.
    ("/jupyternb", {"frame-src": ["'none'"]}),
])
def test_every_response_carries_the_security_headers(client, path, more):
    """`app.py`'s `_add_security_headers`, on an API answer and on pages:
    scripts only from molbuilder's own origin and never inline, no plugins,
    no framing, no cross-origin fetch, no MIME sniffing, the referrer kept
    to the origin -- every directive `web-api.md` § 2 states, on every page,
    the one that frames the notebook included.

    MUTATIONS THIS MUST FAIL AGAINST: a directive loosened in the policy
    every page carries; the notebook page's policy written apart from it,
    and loosened there."""
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code)
    csp = r.headers.get("Content-Security-Policy", "")
    directives = {d.split()[0]: d.split()[1:]
                  for d in (p.strip() for p in csp.split(";")) if d}
    assert directives == {**_POLICY, **more}, csp
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "same-origin"


class _InlineJavaScript(HTMLParser):
    """What `script-src 'self'` blocks, found as the browser parses the
    page: a ``<script>`` with no ``src``, an ``on…`` attribute, a
    ``javascript:`` URL -- each with the line it stands on."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found = []

    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        if tag == "script" and not dict(attrs).get("src"):
            self.found.append(f"line {line}: <script> with no src")
        for name, value in attrs:
            if name.startswith("on"):
                self.found.append(f"line {line}: <{tag} {name}=...>")
            elif (value or "").strip().lower().startswith("javascript:"):
                self.found.append(f"line {line}: <{tag} {name}=javascript:...>")

    handle_startendtag = handle_starttag


def test_no_page_served_carries_inline_javascript(isolated_projects_root):
    """`script-src 'self'` blocks inline JavaScript, and blocks it in
    silence: the browser logs one violation and the wiring never runs
    (`web-api.md` § 2: *"no inline JavaScript anywhere"*; `ui-contract.md`:
    *"no `<script>` blocks, no `onclick=` attributes"*).  On 2026-05-18
    three tabs' *Load from current selection* buttons stayed dead that way.

    Every page the app serves -- each one its route map names, the sign-in
    page from an app that has sign-in of every kind, and each page again
    signed in, with
    the account strip only a signed-in page carries -- parsed as the
    browser parses it, partials and all.

    MUTATIONS THIS MUST FAIL AGAINST: a script block in a page's template,
    an ``onclick`` in a partial a page includes, a script in the sign-in
    page, an ``onload`` on a sign-in kind's button, an ``onclick`` on the
    account strip's *Sign out*."""
    from support.pages import every_served_page
    pages = every_served_page()
    assert {"/results", "/login", "/results (signed in)"} <= set(pages), \
        sorted(pages)
    found = {}
    for path, html in sorted(pages.items()):
        parsed = _InlineJavaScript()
        parsed.feed(html)
        parsed.close()
        if parsed.found:
            found[path] = parsed.found
    assert not found, (
        "inline JavaScript the CSP blocks -- move it into a file under "
        f"web/static/ and load it with <script src>: {found}")


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
