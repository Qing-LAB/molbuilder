"""Optional authentication layer for the molbuilder web UI.

molbuilder DOES NOT manage user accounts or store passwords.  When
the ``auth`` section is configured in ``molbuilder.json``, this module
outsources identity verification to one or more external identity
providers you trust (today: Google, GitHub, Microsoft / Azure AD,
ORCID via OAuth 2.0 / OIDC, and Apereo CAS).  molbuilder only sees
the YES/NO answer + a verified identity (email) from each provider,
then checks that identity against the provider's per-entry
``allowed_users`` list.

Default: no auth (the localhost-only single-user shape that the
``molbuilder serve`` command bootstraps).  See ``docs/ops/deployment.md``
for when to turn auth on + the recommended deployment shapes.

User-visible flow:

  1. Unauthenticated visitor hits any molbuilder URL.
  2. They're redirected to ``/login`` -- a page with one button per
     configured provider (Google / ASURITE / GitHub / ...).
  3. Click → provider's consent / sign-in flow → (institutional
     federation handled by the provider, e.g. @asu.edu Google
     accounts route through ASU's IdP transparently).
  4. Provider redirects back to ``/oauth-callback/<id>`` (OAuth
     providers) or ``/cas-callback/<id>`` (CAS).
  5. molbuilder validates the response, extracts the email, checks
     it against THAT provider's ``allowed_users`` list.
  6. On success, a signed Flask session cookie is set + the user
     lands on the originally-requested URL.

Public entry point:
    :func:`init_auth(app, auth_cfg, secret_key_file)` -- called from
    :func:`molbuilder.web.app.create_app` when an ``auth`` section is
    present in ``molbuilder.json``.

Endpoints installed:
    GET  /login                            login picker
    GET  /login/<provider_id>              start provider's flow
    GET  /oauth-callback/<provider_id>     OAuth providers come back here
    GET  /cas-callback/<provider_id>       CAS providers come back here
    GET  /logout                           clear local session

Every other endpoint goes through a ``before_request`` hook that
sends unauthenticated browsers to ``/login`` and unauthenticated
API callers to a clean JSON 401 (so JS clients render a "please
log in" prompt instead of choking on HTML).
"""
from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import List, Mapping, Optional


# --------------------------------------------------------------------- #
#  Public endpoint allowlist                                            #
# --------------------------------------------------------------------- #
#
# Endpoint NAMES (Flask url_rule endpoints, not URL paths) that the
# require-login hook MUST skip.  Static + the OAuth/CAS dance + the
# health probe go here.  Adding to this set means "this endpoint is
# public" -- weigh carefully.

_PUBLIC_ENDPOINTS = frozenset({
    "auth_login",             # /login -- the picker page
    "auth_login_provider",    # /login/<provider_id>
    "auth_oauth_callback",    # /oauth-callback/<provider_id>
    "auth_cas_callback",      # /cas-callback/<provider_id>
    "auth_logout",
    "api_health",             # liveness probes
    # THE SSO SESSION CHECK DOES NOT APPLY -- which is not the same as
    # unauthenticated.  A monitor on a compute node cannot do a browser
    # sign-in, so `notify.api_notify` authenticates itself: an HMAC-SHA256
    # signature over the exact body, compared by `hmac.compare_digest`.
    # It is registered at all only when the operator configured BOTH a key
    # file and a route segment (`app.py`), so on a server that has not,
    # this name matches no route.
    #
    # **This exemption must not lapse into a redirect.**  If the name ever
    # falls out of this set the gate answers 302 to the sign-in page,
    # `urllib.request.urlopen` FOLLOWS it, the POST body is dropped on the
    # way, and the monitor -- which swallows every failure by design -- sees
    # no error at all.  Reports would stop with nothing anywhere saying why.
    # `tests/test_notify_listener.py` holds that as a guard.
    "notify.api_notify",      # run reports from a job -- see run-reports.md
    "static",                 # CSS/JS; browser fetches before any session
    "vendor_plotly_js",       # third-party JS served from plotly package
})


# --------------------------------------------------------------------- #
#  Public entry point                                                   #
# --------------------------------------------------------------------- #


def init_auth(app, auth_cfg: Mapping, secret_key_file: Optional[str]) -> None:
    """Install authentication on a Flask app.

    ``auth_cfg`` shape: ``{"providers": [entry, entry, ...]}``, each
    entry already validated by ``runtime_config._validate_provider``.

    Raises ``ImportError`` (with an actionable message) when an
    enabled provider's third-party dep isn't installed (authlib for
    OAuth backends; python-cas for CAS).  The error fires at first
    sign-in attempt, not at startup -- so a misconfigured *secondary*
    provider doesn't take down the whole UI; only its button fails
    to work.
    """
    providers: List[dict] = list(auth_cfg.get("providers") or [])
    if not providers:
        raise ValueError(
            "init_auth: auth_cfg has no providers; pass an empty cfg "
            "or omit the call entirely to leave the app unauthenticated."
        )

    _install_secret_key(app, secret_key_file)
    _setup_session_security(app, auth_cfg)
    _store_providers(app, providers)

    _register_login_routes(app)
    _register_callback_routes(app)
    _register_logout_route(app)
    _register_auth_gate(app)
    _register_redirect_uri_logger(app, providers)
    _register_template_context(app)


# --------------------------------------------------------------------- #
#  Provider registry (per-app, kept on app.extensions)                  #
# --------------------------------------------------------------------- #


_PROVIDERS_EXT_KEY = "molbuilder_auth_providers"


def _store_providers(app, providers: List[dict]) -> None:
    """Stash the validated providers list on the app.

    Keyed on the app rather than module-level so multiple apps in
    the same process (the test harness creates fresh apps per test)
    don't bleed config across each other.
    """
    app.extensions[_PROVIDERS_EXT_KEY] = {
        "by_id": {p["id"]: p for p in providers},
        "order": [p["id"] for p in providers],
    }


def _get_providers_state(app):
    return app.extensions.get(_PROVIDERS_EXT_KEY)


def _provider_or_404(app, provider_id: str) -> dict:
    """Look up a provider entry by id; abort 404 when unknown."""
    state = _get_providers_state(app)
    if state is None or provider_id not in state["by_id"]:
        from flask import abort
        abort(404)
    return state["by_id"][provider_id]


def _providers_in_order(app) -> List[dict]:
    state = _get_providers_state(app)
    if state is None:
        return []
    return [state["by_id"][pid] for pid in state["order"]]


# --------------------------------------------------------------------- #
#  Route registration                                                   #
# --------------------------------------------------------------------- #


def _safe_next_target(raw_target) -> str:
    """Sanitise a ``_post_login_next`` value into a SAFE local path.

    Returns ``"/"`` if the input is empty, not a string, or doesn't
    look like a server-relative path.  Specifically rejects:

      * absolute URLs (``http://...``, ``//evil.example.com``,
        ``javascript:...``, etc.) -- guards against open-redirect
        phishing where the login bounce ends up at attacker.com;
      * Windows-style backslash paths (parsed as host on some clients);
      * empty / whitespace input.

    Today the only writer of ``_post_login_next`` is ``_require_login``,
    which assigns ``request.path`` -- always a relative path -- so this
    guard is defence-in-depth.  If a future caller writes the value
    from a user-controlled source (``?next=`` query, an API client,
    etc.) the guard prevents an open-redirect from sneaking in.
    """
    if not isinstance(raw_target, str) or not raw_target:
        return "/"
    t = raw_target.strip()
    # Must be a server-relative path: starts with a single "/" and is
    # NOT a protocol-relative URL ("//...") which browsers treat as
    # an absolute URL on the current scheme.  Backslashes are
    # rejected because Windows-flavoured clients can parse them as
    # path separators which a redirect implementation downstream
    # might mis-canonicalise.
    if (not t.startswith("/")
            or t.startswith("//")
            or "\\" in t):
        return "/"
    return t


def _register_login_routes(app) -> None:
    from flask import (current_app, redirect, render_template, request,
                       session, url_for)

    @app.route("/login", endpoint="auth_login")
    def auth_login():
        # Already signed in?  Bounce to wherever they were headed.
        if "user" in session:
            target = _safe_next_target(session.pop("_post_login_next", "/"))
            return redirect(target)
        # Pass a minimal list to the template -- only what the button
        # rendering needs.  Avoids leaking client_secret / URLs / etc.
        # into HTML inadvertently.
        provs = [
            {"id":    p["id"],
             "label": p["label"],
             "kind":  p["kind"]}
            for p in _providers_in_order(current_app)
        ]
        return render_template("login.html", providers=provs)

    @app.route("/login/<provider_id>", endpoint="auth_login_provider")
    def auth_login_provider(provider_id):
        from . import auth_providers
        entry = _provider_or_404(current_app, provider_id)
        try:
            return auth_providers.start_flow(current_app, entry)
        except ImportError as exc:
            return _missing_dep_response(entry, exc), 500


def _register_callback_routes(app) -> None:
    from flask import current_app

    @app.route("/oauth-callback/<provider_id>",
                endpoint="auth_oauth_callback")
    def auth_oauth_callback(provider_id):
        from . import auth_providers
        entry = _provider_or_404(current_app, provider_id)
        try:
            return auth_providers.finish_oauth(current_app, entry)
        except ImportError as exc:
            return _missing_dep_response(entry, exc), 500

    @app.route("/cas-callback/<provider_id>",
                endpoint="auth_cas_callback")
    def auth_cas_callback(provider_id):
        from . import auth_providers
        entry = _provider_or_404(current_app, provider_id)
        try:
            return auth_providers.finish_cas(current_app, entry)
        except ImportError as exc:
            return _missing_dep_response(entry, exc), 500


def _register_logout_route(app) -> None:
    from flask import redirect, session, url_for

    @app.route("/logout", endpoint="auth_logout")
    def auth_logout():
        session.clear()
        # We don't redirect to the provider's logout endpoint -- that
        # would also sign the user out of Gmail / GitHub / etc., which
        # is invasive for what should be a per-app logout.  Operators
        # who want global single-logout can wire that up themselves.
        return redirect(url_for("auth_login"))


def _register_auth_gate(app) -> None:
    """Install the before-request hook that gates non-public endpoints."""
    from flask import g, jsonify, redirect, request, session, url_for

    @app.before_request
    def _require_login():
        if request.endpoint is None:
            return None     # bogus URL; let Flask's 404 handler run
        if request.endpoint in _PUBLIC_ENDPOINTS:
            return None
        user = session.get("user")
        if user:
            g.user = user   # available to templates as g.user
            return None

        # THIS ANSWER IS NOT EVIDENCE OF ANYTHING, and the rate limiter is
        # told so.  "I do not know who you are yet" is what this gate says to
        # every ordinary visitor, including one whose session simply expired --
        # it is not a probe.
        #
        # Left uncounted it was: the limiter counts 4xx, a session expiring with
        # a tab open turns the page's own 1 Hz poll into one 4xx per second, and
        # twenty in thirty seconds blocked the user's address FOR AN HOUR on
        # every path -- the login page included.  From their side the site was
        # simply down, with no message.  The app locking its own user out.
        #
        # Marked HERE, on the one gate that produces it, rather than by
        # excluding 401 wherever it appears: a 401 from somewhere else has a
        # different author and may well mean something.  Everything else the
        # limiter watches is untouched -- the attack-string signature fires as
        # before, and a genuine 404 storm still counts.
        g.molbuilder_auth_challenge = True

        # API requests get a clean 401 so JS clients can render a
        # "please log in" prompt instead of trying to parse HTML.
        if request.path.startswith("/api/"):
            return jsonify({
                "ok":        False,
                "error":     "not authenticated",
                "login_url": url_for("auth_login", next=request.path),
            }), 401

        # Stash the path the user was trying to reach so we can
        # return there after login.  ``request.path`` is always a
        # server-relative path in Flask, so this is safe on input --
        # the read side runs it through ``_safe_next_target`` anyway
        # as defence-in-depth.
        session["_post_login_next"] = _safe_next_target(request.path)
        return redirect(url_for("auth_login"))


def _register_template_context(app) -> None:
    """Expose ``g.user`` (when set) into every template's context.

    The header partial reads ``g.user`` to render the
    "Signed in as X · Sign out" strip.  When auth is disabled,
    ``g.user`` is simply unset and the strip is omitted.
    """
    from flask import g

    @app.context_processor
    def _inject_user():
        return {"current_user": getattr(g, "user", None)}


def _register_redirect_uri_logger(app, providers: List[dict]) -> None:
    """Log the OAuth redirect URI of each OAuth provider on first
    request.  Operators copy these into each provider's console as
    "Authorized redirect URIs".  Printing them from the running server
    takes the guesswork out of host/port/scheme/proxy interactions.
    """
    from flask import url_for

    oauth_provider_ids = [
        p["id"] for p in providers
        if p["kind"] in ("google", "github", "microsoft", "orcid")
    ]
    if not oauth_provider_ids:
        return

    logged = [False]

    @app.before_request
    def _log_redirect_uris_once():
        if logged[0]:
            return
        logged[0] = True
        for pid in oauth_provider_ids:
            try:
                uri = url_for(
                    "auth_oauth_callback",
                    provider_id=pid,
                    _external=True,
                )
            except Exception:
                continue  # url_for may fail before app fully bound
            logging.info(
                "molbuilder auth: register THIS exact URL as an "
                "'Authorized redirect URI' in the provider's console "
                "for provider %r:\n    %s",
                pid, uri,
            )


# --------------------------------------------------------------------- #
#  Backend -> auth.py call: authenticate()                              #
# --------------------------------------------------------------------- #


def authenticate(provider_id: str, email: str, profile: Mapping):
    """Allowlist-check the email + start a session.

    Called by ``auth_providers/{oauth,cas}.py`` on a successful
    sign-in from the IdP.  Returns a Flask response: a redirect to
    the originally-requested URL on success, a 403 error string when
    the identity is not in the provider's allowed_users list.

    Allowlist match uses ``str.casefold`` on BOTH sides, which is
    stricter than ``str.lower`` (folds ``ß`` → ``ss`` etc.) so
    non-ASCII local-parts still match consistently.  Each provider
    has its OWN allowed_users list; an email allowed for provider X
    is *not* implicitly allowed for provider Y.

    Session ordering is load-bearing: ``_post_login_next`` is read
    BEFORE ``session.clear()`` (we need the value), then ``clear()``
    wipes any pre-login state (CSRF nonces, OAuth state cookies,
    arbitrary keys an unauthenticated visitor may have caused to be
    set), then the verified-user dict is written into the now-empty
    session.  Reordering breaks the clean-slate guarantee.
    """
    from flask import current_app, redirect, session

    entry = _provider_or_404(current_app, provider_id)
    allowed = {u.casefold() for u in entry.get("allowed_users") or []}
    email_cf = (email or "").casefold()
    if email_cf not in allowed:
        return (
            f"Sign-in declined: identity {email!r} is not in the "
            f"allowed_users list for provider {provider_id!r}.  "
            f"Contact the molbuilder administrator if you should "
            f"have access.",
            403,
        )

    target = _safe_next_target(session.pop("_post_login_next", "/"))
    session.clear()
    session["user"] = {
        "email":       email_cf,
        "provider_id": provider_id,
        "name":        profile.get("name") or email_cf,
        "picture":     profile.get("picture"),
        "sub":         profile.get("sub"),
    }
    return redirect(target)


# --------------------------------------------------------------------- #
#  Shared setup helpers                                                 #
# --------------------------------------------------------------------- #


def _setup_session_security(app, auth_cfg: Mapping) -> None:
    """Wire session-cookie security defaults and (optionally) ProxyFix.

    Session cookies (always on): SECURE (HTTPS-only), HTTPONLY (no JS
    access), SAMESITE=Lax (allows top-level navigation back from the
    OAuth provider but blocks cross-site state-changing requests).

    ProxyFix: installed ONLY when ``auth.trust_proxy`` is set to
    ``true`` in molbuilder.json.  When on, Flask honours one upstream
    proxy hop's ``X-Forwarded-*`` headers so ``url_for(_external=True)``
    builds the public URL the browser saw (e.g.,
    ``https://qlabsrv.physics.asu.edu/...``) rather than the internal
    address the proxy forwarded to (e.g., ``http://127.0.0.1:8000/...``).
    Without ProxyFix a strict OAuth provider rejects the callback with
    ``redirect_uri_mismatch``.

    SECURITY: do NOT enable ``trust_proxy`` for a direct-TLS deploy
    (``molbuilder serve --cert --key`` with no reverse proxy in front).
    A malicious request can then carry an attacker-chosen
    ``X-Forwarded-Host`` header, which url_for would honour -- crafting
    a CAS ``service_url`` that lands the user's ticket at attacker.com.
    OAuth providers reject the spoofed redirect_uri (it's not in their
    allowlist), but CAS has no such allowlist.  Default off is correct
    for the direct-TLS deployment shape documented in
    ``docs/ops/deployment.md`` § 1.
    """
    # Direct assignment, NOT setdefault: Flask pre-populates
    # ``SESSION_COOKIE_SECURE = False`` (and SAMESITE = None) in its
    # default config dict, so ``setdefault`` is a no-op against those
    # values and the cookie ships INSECURE in production -- exactly
    # the regression a security-relevant test should pin.  Operators
    # who genuinely need to disable any of these flags (rare; e.g.
    # local HTTP-only smoke testing) must override them AFTER
    # ``init_auth`` runs.
    app.config["SESSION_COOKIE_SECURE"]   = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    if auth_cfg.get("trust_proxy"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1,
        )
        logging.info(
            "molbuilder auth: ProxyFix enabled (auth.trust_proxy=true).  "
            "ONE upstream proxy hop's X-Forwarded-* headers will be "
            "trusted; ensure exactly one reverse proxy sits in front "
            "of molbuilder and that it scrubs client-supplied forwarded "
            "headers before adding its own."
        )


def _install_secret_key(app, secret_key_file: Optional[str]) -> None:
    """Load Flask's session-signing key from disk (or generate one
    on first run + persist with mode 0600).  Falls back to a
    per-process random key + warning when no path is configured."""
    if not secret_key_file:
        app.config["SECRET_KEY"] = secrets.token_bytes(32)
        logging.warning(
            "molbuilder auth: no 'secret_key_file' in molbuilder.json; "
            "using an ephemeral per-process key.  Sessions will "
            "invalidate on every restart.  Set "
            "'secret_key_file': '~/.config/molbuilder/secret_key' "
            "(or similar) for stable sessions."
        )
        return

    path = Path(os.path.expanduser(secret_key_file))
    if path.exists():
        key_bytes = path.read_bytes()
        if len(key_bytes) < 16:
            raise RuntimeError(
                f"molbuilder auth: secret_key_file {str(path)!r} is "
                f"only {len(key_bytes)} bytes; refusing to use it "
                f"(min 16).  Delete the file to regenerate."
            )
        app.config["SECRET_KEY"] = key_bytes
        return

    # First-run generate: make parent dir if needed, write atomically
    # with restrictive permissions.
    path.parent.mkdir(parents=True, exist_ok=True)
    key_bytes = secrets.token_bytes(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key_bytes)
    finally:
        os.close(fd)
    app.config["SECRET_KEY"] = key_bytes


def _missing_dep_response(entry: Mapping, exc: ImportError) -> str:
    """Format a 500 response body explaining which dep is missing
    for which provider, so the operator can install + retry."""
    return (
        f"molbuilder auth: provider {entry['id']!r} (kind="
        f"{entry['kind']!r}) is configured but its dependency "
        f"is not importable:\n\n    {exc}\n\n"
        f"Other configured providers (if any) still work."
    )
