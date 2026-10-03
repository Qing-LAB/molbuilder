"""Tests for the ``auth`` section of ``molbuilder.json``.

molbuilder's authentication layer is opt-in: omitting the ``auth``
section means localhost-only no-auth mode (the right default for
the single-user laptop deployment shape).  When present, the schema
must catch misconfiguration loudly + early so the operator sees a
clear error message instead of a broken login flow at runtime.

Schema overview::

    "auth": {
        "providers": [
            { "id": ..., "label": ..., "kind": ..., "allowed_users": [...],
              ...kind-specific fields... },
            ...
        ]
    }

Each provider entry is self-contained.  ``allowed_users`` is
per-provider (no global list); identity from provider X is matched
only against X's own list.

Coverage:
  * absence of ``auth`` is valid (no-auth mode)
  * providers list must be a non-empty list
  * common fields (id / label / kind / allowed_users) are required
  * id must be a URL-safe slug + unique across the list
  * unsupported ``kind`` is rejected
  * OAuth kinds (google/github/microsoft/orcid):
      - client_id required
      - the entry names no secret: `client_secret` and `client_secret_file`
        are refused by name, saying the kind's home in `secrets/`
      - kind-specific extras validated (hosted_domain, allowed_organizations,
        tenant_id)
  * CAS kind:
      - login_url required; service_validate_url refused by name
      - version must be 1, 2, or 3
      - at least one of email_attribute / email_domain (for allowlist match)
      - optional string fields validated when set
  * allowed_users must be list of strings (empty list = no one, valid)
  * a key a provider's kind does not hold, or `auth` does not hold, is
    refused (`configuration.md` § 4)
  * a config still naming `secret_key_file` is REFUSED, with its one home named

Does NOT test the runtime auth flow itself (Authlib OAuth + python-cas
ticket validation are integration-tested separately).
"""
from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (
    RuntimeConfigError, _normalise, read_config,
    get_auth, get_providers,
)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _google_entry(**overrides):
    """Minimal valid Google provider entry; override any field."""
    base = {
        "id":                 "google",
        "label":              "Sign in with Google",
        "kind":               "google",
        "allowed_users":      ["user@example.com"],
        "client_id":          "1234.apps.googleusercontent.com",
    }
    base.update(overrides)
    return base


def _cas_entry(**overrides):
    """Minimal valid CAS provider entry; override any field."""
    base = {
        "id":                   "asu_cas",
        "label":                "Sign in with ASURITE ID",
        "kind":                 "cas",
        "allowed_users":        ["user@example.com"],
        "login_url":            "https://cas.example.com/cas/login",
        "email_domain":         "example.com",
    }
    base.update(overrides)
    return base


def _wrap(*entries):
    """Wrap provider entries into a full config payload."""
    return {"auth": {"providers": list(entries)}}


# --------------------------------------------------------------------- #
#  Default: no auth section means no auth                               #
# --------------------------------------------------------------------- #


def _without(entry, key):
    """A provider entry with one key removed -- the half-finished hand edit."""
    e = dict(entry)
    del e[key]
    return e


def _gh(**kw):
    return _google_entry(kind="github", id="github", **kw)


def _ms(**kw):
    return _google_entry(kind="microsoft", id="microsoft", **kw)


#: Every way a `molbuilder.json` auth section can be wrong, and the words the
#: refusal must contain.  ONE LIST, so "is this case covered?" is answered by
#: READING IT -- which is the whole reason it exists.
#:
#: It replaced twenty-two separate tests on 2026-09-09, each feeding one
#: malformed value to `_normalise(dict)`.  That shape cost a full day: an
#: auditor read two of them, saw they reached the same LINE, and called one a
#: duplicate -- but the line is `not isinstance(x, T) or not x` and they reach
#: ONE CLAUSE EACH.  Three real guards were an approval away from deletion,
#: including the two marked below, and no amount of mutation testing found it
#: because the harness mutated per line too.  In a list you can see both rows.
MALFORMED_AUTH = [
    # ---- the providers list itself -- `deployment.md` § 3 ----------------
    ("providers-absent",      {"auth": {}},                                  "auth.providers"),
    # KEEP: the only row reaching `or not providers`.  Cutting it admits a
    # config that starts the server with a login page and no way through.
    ("providers-empty",       {"auth": {"providers": []}},                   "non-empty"),
    ("providers-an-object",   {"auth": {"providers": {"id": "x"}}},          "non-empty list"),
    ("entry-not-an-object",   {"auth": {"providers": ["not-an-object"]}},    "must be an object"),
    ("duplicate-ids",         _wrap(_google_entry(id="x"),
                                    _google_entry(id="x",
                                                  client_id="other.apps.googleusercontent.com")),
                                                                            "duplicate provider id"),

    # ---- fields every provider must have -- `deployment.md` § 3 ----------
    *[(f"missing-{k}", _wrap(_without(_google_entry(), k)), k)
      for k in ("id", "label", "kind")],
    # KEEP: the only rows reaching `or not val` in `_require_str`.  An empty
    # `id` makes the callback route `/oauth-callback/`.
    *[(f"empty-{k}", _wrap(_google_entry(**{k: ""})), k)
      for k in ("id", "label", "kind")],
    # A TRUTHY NON-STRING -- a number pasted where a string belongs.  `not val`
    # is False for it, so only the `isinstance` half of `_require_str` refuses
    # it; without this row that half can be deleted with every other row green
    # (measured 2026-09-09).
    *[(f"nonstring-{k}", _wrap(_google_entry(**{k: 42})), k)
      for k in ("id", "label")],
    *[(f"id-not-a-slug-{b!r}", _wrap(_google_entry(id=b)), "URL-safe slug")
      for b in ("Google", "google!", "asu cas", "_google", "-google")],
    # `mb_` is authlib's namespace; an id taking it collides with the wizard's.
    *[(f"id-reserved-{b}", _wrap(_google_entry(id=b)), "URL-safe slug")
      for b in ("mb_google", "mb_register")],
    *[(f"kind-{b}", _wrap(_google_entry(kind=b)), "not supported")
      for b in ("ldap", "saml", "kerberos", "facebook")],

    # ---- allowed_users: the per-provider gate -- `access-control.md` § 3.1
    ("allowed_users-absent",  _wrap(_without(_google_entry(), "allowed_users")),
                                                                            "allowed_users"),
    ("allowed_users-a-string", _wrap(_google_entry(allowed_users="u@e.com")), "list of strings"),
    ("allowed_users-mixed",   _wrap(_google_entry(allowed_users=["u@e.com", 42])),
                                                                            "list of strings"),

    # ---- the OAuth client -- `deployment.md` § 3 -------------------------
    ("oauth-no-client_id",    _wrap(_without(_google_entry(), "client_id")),  "client_id"),
    # No secret and no path to one in molbuilder.json (`configuration.md`
    # § 3.1, user 2026-10-02) -- each refused by name, saying the KIND's home,
    # which is why the second row is another kind.
    ("oauth-literal-secret",  _wrap(_google_entry(client_secret="literal")),
                                              ["'client_secret' is refused",
                                               "secrets/google_client_secret"]),
    ("oauth-secret-file",     _wrap(_gh(client_secret_file="/etc/gh.secret")),
                                              ["'client_secret_file' is refused",
                                               "secrets/github_client_secret"]),
    # A key the kind does not hold: a typo in `hosted_domain` dropped that
    # restriction in silence until 2026-10-02.
    ("provider-unknown-key",  _wrap(_google_entry(hosted_domains=["asu.edu"])),
                                              "got 'hosted_domains'"),

    # ---- per-kind fields -- `access-control.md` § 3.1, `deployment.md` § 3.4
    # KEEP: the routing of `hosted_domain` through the str-list helper is not
    # held by the acceptance test beside it (measured 2026-09-09).
    ("google-hosted_domain-a-string", _wrap(_google_entry(hosted_domain="asu.edu")),
                                                                            "list of strings"),
    ("github-orgs-a-string",  _wrap(_gh(allowed_organizations="my-org")),    "list of strings"),
    ("microsoft-tenant-empty", _wrap(_ms(tenant_id="")),                     "tenant_id"),
    ("microsoft-tenant-a-number", _wrap(_ms(tenant_id=42)),                  "tenant_id"),

    # ---- CAS -- `deployment.md` § 3 --------------------------------------
    # `service_validate_url` left this list 2026-09-12: it is no longer
    # required, because nothing ever read it -- python-cas derives the validate
    # endpoint from the login URL's root and takes no parameter for an explicit
    # one.  `login_url` IS required and still checked here.
    *[(f"cas-missing-{k}", _wrap(_without(_cas_entry(), k)), k)
      for k in ("login_url",)],
    ("cas-service_validate_url-retired",
     _wrap(_cas_entry(service_validate_url="https://cas.example.com/cas/serviceValidate")),
                                              "'service_validate_url' is no longer configured"),
    *[(f"cas-version-{b!r}", _wrap(_cas_entry(version=b)), "version")
      for b in (0, 4, "3", 3.0, None)],
    # An OPTIONAL key present but empty is a hand edit half-undone: the key is
    # there, so nothing defaults, and the empty value reaches the CAS URL.
    *[(f"cas-empty-{k}", _wrap(_cas_entry(**{k: ""})), k)
      for k in ("service_url", "ca_certs", "email_attribute", "email_domain")],
    ("cas-no-email-route",    _wrap(_without(_cas_entry(), "email_domain")),
                                                                            "email_attribute"),

    # ---- retired spellings: named, not reported as a typo ----------------
    # The session key's path inside `auth`, where the wizard wrote it before
    # 2026-08-31 (`configuration.md` § 2.1e; the top-level spelling is a
    # row of `tests/data/molbuilder_json.toml`).
    ("auth-secret_key_file-retired", {"auth": {"providers": [_google_entry()],
                                               "secret_key_file": "~/.mb/key"}},
                                              "no longer configured"),
    ("auth-unknown-key",      {"auth": {"providers": [_google_entry()],
                                       "trusted_proxy": True}},
                                              "got 'trusted_proxy'"),

    # ---- trust_proxy -- it decides whether forwarded client IPs are believed
    *[(f"trust_proxy-{b!r}", {"auth": {"providers": [_google_entry()],
                                       "trust_proxy": b}}, "trust_proxy")
      for b in ("yes", 1, None)],
]


@pytest.mark.parametrize("raw,must_say",
                         [(r, m) for _, r, m in MALFORMED_AUTH],
                         ids=[i for i, _, _ in MALFORMED_AUTH])
def test_a_malformed_auth_config_is_refused_at_the_door(tmp_path, raw, must_say):
    """Every malformed `molbuilder.json` auth section is refused, through the
    door the server uses, from a real file.

    THE FAILURE THIS CATCHES.  `web/app.py:254` reads the config at startup and
    refuses to start on anything `read_config` rejects.  A validator that stops
    refusing is SILENT -- the server comes up configured wrongly, and depending
    on which check lapsed that is either a lockout or a gate admitting accounts
    it should not.

    WHY THROUGH A FILE, not `_normalise(dict)`.  `read_config` is the whole
    validator: it parses, refuses a non-object top level, runs every section's
    reader, and re-raises NAMING WHICH FILE refused -- that last step exists
    because of a measured defect (R10, 2026-08-12: an operator with three config
    files got an error naming none of them).  A dict-level test sits below it.

    Contract: `deployment.md` § 3 and § 7 (this file owns `molbuilder.json` auth
    validation), `access-control.md` § 3.1 (`allowed_users` is per provider),
    `configuration.md` § 2.1e (the retired `secret_key_file`).
    """
    cfg = tmp_path / "molbuilder.json"
    cfg.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeConfigError) as exc:
        read_config(cfg)
    msg = str(exc.value)
    for frag in ([must_say] if isinstance(must_say, str) else must_say):
        assert frag in msg, f"refusal does not say {frag!r}: {msg}"
    assert str(cfg) in msg, f"refusal does not name the file (R10): {msg}"


class TestTheConfigFileIsRefusedAtTheDoor:
    """`read_config()` is the validator, and it is what the app calls.

    WHY THROUGH THE FILE, not `_normalise(dict)`.  `web/app.py:254` reads the
    config at startup and refuses to start on any failure -- *"Bad config is
    loud, not silent: refuse to start."*  `read_config` is the whole of that
    validator: it finds the file, refuses invalid JSON, refuses a non-object
    top level, runs the schema checks, and **re-raises naming WHICH FILE
    refused**.

    The tests here used to call `_normalise` with a hand-built dict, which is
    the middle step only -- so three things the door does were untested, and
    one of them is a RECORDED DEFECT.  `runtime_config.py:133` exists because
    *"a malformed file used to refuse naming 'molbuilder.json' with no path
    (R10, 2026-08-12)"*: an operator got an error that did not say which file
    was broken.  A dict-level
    test sits below that layer and cannot see it come back.

    Contract: `deployment.md` § 3 (what `auth` must contain) and § 7 (this file
    owns `molbuilder.json` auth validation); `access-control.md` § 3.1.
    """

    # (bad file contents, fragments the refusal must contain)
    # (The four `auth` rows that stood first here were MALFORMED_AUTH's own,
    # through the same door, until 2026-10-02.)
    BAD = [
        ('{"auth": {"providers": [', ["invalid JSON", "line"]),
        ('[]', ["top-level", "object"]),
        ('"a string"', ["top-level", "object"]),
    ]

    @pytest.mark.parametrize("text,must_say", BAD,
                             ids=[t[:34] for t, _ in BAD])
    def test_a_broken_config_file_is_refused_and_the_message_names_it(
            self, tmp_path, text, must_say):
        """A hand-edited `molbuilder.json` that is wrong is refused, and the
        refusal says BOTH what is wrong and which file it was.

        Every row is a slip somebody makes editing JSON by hand: an `auth`
        section started and abandoned, a providers list left empty, one object
        where a list belongs, a bare string in the list, a truncated file, and a
        top level that is not an object at all. Each must stop startup rather
        than configure the server wrongly -- an empty providers list is a login
        page with no buttons, and a half-written `auth` section is a gate with
        nothing behind it.
        """
        cfg = tmp_path / "molbuilder.json"
        cfg.write_text(text, encoding="utf-8")
        with pytest.raises(RuntimeConfigError) as exc:
            read_config(cfg)
        msg = str(exc.value)
        for frag in must_say:
            assert frag in msg, f"refusal does not say {frag!r}: {msg}"
        assert str(cfg) in msg, (
            f"the refusal does not name WHICH file was broken (R10, "
            f"2026-08-12) -- an operator with several config files is told "
            f"nothing: {msg}")

    def test_a_config_with_no_auth_section_is_accepted_with_auth_off(
            self, tmp_path):
        """No `auth` section means no login -- the right shape for a personal
        machine -- and another section must not switch one on as a side effect.

        `deployment.md` § 3: decided by a PRESENCE check, not truthiness. The
        second file is the live case: a machine that configures TLS and envs and
        never asked for sign-in must not acquire a gate.
        """
        for text in ('{}',
                     '{"tls": {"cert": "/tmp/c", "key": "/tmp/k"},'
                     ' "envs": {"siesta": "molbuilder-siesta"}}'):
            cfg = tmp_path / "molbuilder.json"
            cfg.write_text(text, encoding="utf-8")
            got = read_config(cfg)
            assert get_auth(got) == {}, text
            assert get_providers(got) == [], text


# --------------------------------------------------------------------- #
#  Providers list shape -- the per-entry checks, still at the schema    #
#  level because the door above already proves the file path reaches   #
#  them.                                                               #
# --------------------------------------------------------------------- #


class TestProvidersListShape:



    def test_two_providers_with_distinct_ids_ok(self):
        """A uniqueness check that rejects two DIFFERENT providers: the real
        deployment shape -- institutional CAS plus Google for outside
        collaborators -- would not load at all.

        `access-control.md` § 3.1. Order is asserted because the login page
        renders its buttons in config order.
        """
        cfg = _normalise(_wrap(_google_entry(), _cas_entry()))
        ids = [p["id"] for p in get_providers(cfg)]
        assert ids == ["google", "asu_cas"]


# --------------------------------------------------------------------- #
#  Common per-entry fields                                              #
# --------------------------------------------------------------------- #


class TestCommonFields:





    def test_id_good_slugs_accepted(self):
        """The slug regex tightening under maintenance until a working config
        stops loading: digits and hyphens are legal, and an operator's
        `asu_cas` or `my-org-github` must keep working across upgrades.

        `deployment.md` § 3.
        """
        for good_id in ("google", "asu_cas", "g-1", "my-org-github"):
            cfg = _normalise(_wrap(_google_entry(id=good_id)))
            assert get_providers(cfg)[0]["id"] == good_id



# --------------------------------------------------------------------- #
#  allowed_users (per-provider, required)                               #
# --------------------------------------------------------------------- #


class TestAllowedUsers:




    def test_empty_list_accepted_fail_closed(self):
        """An empty list means 'no one can sign in via this backend' --
        a valid degenerate state (useful for temporarily disabling a
        provider without deleting its entry)."""
        cfg = _normalise(_wrap(_google_entry(allowed_users=[])))
        assert get_providers(cfg)[0]["allowed_users"] == []

    def test_preserved_verbatim(self):
        """Case normalisation happens at the enforcement site (auth.py),
        not the parse site -- so the operator sees their list exactly
        as written when echoed back."""
        cfg = _normalise(_wrap(_google_entry(
            allowed_users=["User@Example.COM", "another@asu.edu"]
        )))
        assert get_providers(cfg)[0]["allowed_users"] == [
            "User@Example.COM", "another@asu.edu",
        ]

    def test_each_provider_has_its_own_list(self):
        """No global allowlist; each provider is matched only against
        its own entry."""
        cfg = _normalise(_wrap(
            _google_entry(id="g",        allowed_users=["a@x.com"]),
            _cas_entry  (id="cas",       allowed_users=["b@x.com"]),
        ))
        provs = get_providers(cfg)
        assert provs[0]["allowed_users"] == ["a@x.com"]
        assert provs[1]["allowed_users"] == ["b@x.com"]


# --------------------------------------------------------------------- #
#  OAuth-kind shared validation (google/github/microsoft/orcid)         #
# --------------------------------------------------------------------- #


class TestOAuthSharedFields:




    # `test_literal_secret_accepted` retired 2026-10-02: a literal
    # `client_secret` is refused by name (MALFORMED_AUTH's
    # `oauth-literal-secret`).

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_an_entry_naming_no_secret_is_accepted(self, kind):
        """Every OAuth kind is accepted with no secret key at all -- and no
        `client_secret` is SYNTHESISED into the entry from its home's
        contents, which would put secret bytes into anything that echoes the
        parsed config.

        `configuration.md` § 3.1: no secret in `molbuilder.json` except the
        cert files; the secret is at `secrets/<kind>_client_secret`.
        """
        cfg = _normalise(_wrap(_google_entry(kind=kind, id=kind)))
        p = get_providers(cfg)[0]
        assert "client_secret" not in p and "client_secret_file" not in p


class TestARotatedSecretIsTakenUp:
    """``_ensure_client`` in molbuilder/web/auth_providers/oauth.py reads the
    provider's client secret again, through the one door, on every call after
    the first, and applies a changed one in place.  An operator can fix a
    wrong GOCSPX value WITHOUT restarting the server (task #100).

    These tests pin:
      * A rotated secret is picked up, on the SAME authlib client object
        (cache identity preserved -- otherwise concurrent in-flight callbacks
        would see different clients).
      * A failed re-read (file deleted / emptied) keeps the previously-loaded
        secret rather than crashing or zeroing-out the client.

    (It watched the file's mtime until 2026-10-02, through a path computed
    beside the door; two tests of that mechanism -- the mtime recorded, a
    preserved mtime ignored -- retired with it.)
    """

    def _entry(self, kind="google"):
        return {
            "id":                 kind,
            "label":              f"Sign in with {kind}",
            "kind":               kind,
            "client_id":          "test.apps.googleusercontent.com",
            "allowed_users":      ["user@example.com"],
        }

    def _home(self, text, kind="google"):
        """The kind's secret at its one home, written by the wizard's own
        writer."""
        from molbuilder.auth_setup import write_secret_file
        from molbuilder.config_dir import client_secret
        home = client_secret(kind)
        write_secret_file(home, text)
        return home

    def _app(self):
        from flask import Flask
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = b"x" * 32
        return app

    @pytest.mark.parametrize("kind", ["google", "github"])
    def test_secret_change_is_picked_up_without_restart(self, kind):
        """The whole point of task #100: replace the secret, hit a callback,
        new secret is in effect.  Verifies (a) the value updates AND
        (b) the same client object is reused (cache identity).  Two kinds,
        because each reads ITS OWN home (`configuration.md` § 3.1)."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        self._home("GOCSPX-original", kind)
        app   = self._app()
        entry = self._entry(kind)

        with app.app_context():
            client_first  = _ensure_client(app, entry)
            assert client_first.client_secret == "GOCSPX-original"

            self._home("GOCSPX-rotated", kind)

            client_second = _ensure_client(app, entry)

        # Same authlib client object, with the secret mutated in place.
        # If the registry built a NEW client instead, in-flight token
        # exchanges using the old reference would silently keep the
        # old secret -- this test catches a "destroy + rebuild" regression.
        assert client_second is client_first
        assert client_second.client_secret == "GOCSPX-rotated"

    def test_file_deleted_keeps_previously_loaded_secret(self):
        """If the operator deletes / moves the secret file between
        calls, the running app must NOT crash; it keeps the secret
        already in memory (the active OAuth flow is non-fatal)."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        home  = self._home("GOCSPX-original")
        app   = self._app()
        entry = self._entry()

        with app.app_context():
            _ensure_client(app, entry)
            home.unlink()
            client = _ensure_client(app, entry)
        assert client.client_secret == "GOCSPX-original"

    def test_empty_file_after_rotation_keeps_previous_secret(self):
        """If the operator's rotation script writes an empty file
        (clobbered + not-yet-rewritten state, or a tool that truncates
        before writing), molbuilder must NOT zero out the active
        client_secret -- the next callback would then send no secret
        at all and Google would return invalid_client.  Keep the
        previously-loaded secret and log a warning."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        home  = self._home("GOCSPX-original")
        app   = self._app()
        entry = self._entry()

        with app.app_context():
            _ensure_client(app, entry)
            home.write_text("")    # empty -- mid-rotation
            client = _ensure_client(app, entry)
        # The door refuses an empty secret; the helper catches + logs;
        # the previously-loaded secret stays.
        assert client.client_secret == "GOCSPX-original"


class TestSetupSessionSecurity:
    """``_setup_session_security`` wires two security-relevant pieces:
      1. Session-cookie security flags (SECURE, HTTPONLY, SAMESITE).
         Always on, regardless of trust_proxy.  These keep the
         session cookie HTTPS-only, JS-unreadable, and CSRF-safer.
      2. ProxyFix middleware -- ONLY when ``auth.trust_proxy=True``.
         Gating matters: enabling ProxyFix unconditionally lets a
         direct-TLS deploy spoof X-Forwarded-Host (see auth review
         P1 #7 in docs/protocols).  Default off is correct for the
         most common deploy shape and must NOT silently change.
    """

    def _flask_with(self, providers, *, trust_proxy=False):
        from flask import Flask
        from molbuilder.web.auth import init_auth
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = b"x" * 32
        init_auth(app,
                  auth_cfg={"providers": providers,
                            "trust_proxy": trust_proxy})
        app.config["SECRET_KEY"] = b"x" * 32
        return app

    def test_session_cookie_flags_set_when_auth_on(self):
        """SECURE + HTTPONLY + SAMESITE are non-negotiable; auth must
        set them on the app config every time it installs."""
        app = self._flask_with([_google_entry()])
        assert app.config["SESSION_COOKIE_SECURE"]   is True
        assert app.config["SESSION_COOKIE_HTTPONLY"] is True
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"

    def test_session_cookie_flags_set_regardless_of_trust_proxy(self):
        """trust_proxy controls ProxyFix only; the cookie flags must
        be wired in both code paths."""
        app = self._flask_with([_google_entry()], trust_proxy=True)
        assert app.config["SESSION_COOKIE_SECURE"]   is True
        assert app.config["SESSION_COOKIE_HTTPONLY"] is True
        assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"

    def test_proxyfix_is_NOT_installed_by_default(self):
        """SECURITY-LOAD-BEARING.  trust_proxy=False is the default
        (the right shape for direct-TLS deploys).  ProxyFix MUST
        NOT be installed in that case -- otherwise a malicious
        request can spoof X-Forwarded-Host and influence the
        redirect URIs molbuilder hands to OAuth / CAS providers."""
        from werkzeug.middleware.proxy_fix import ProxyFix
        app = self._flask_with([_google_entry()], trust_proxy=False)
        assert not isinstance(app.wsgi_app, ProxyFix), (
            "ProxyFix is installed despite trust_proxy=False; this "
            "exposes direct-TLS deploys to X-Forwarded-* header "
            "spoofing.  See auth review P1 #7."
        )

    def test_proxyfix_IS_installed_when_trust_proxy_true(self):
        """Opt-in path: operators behind a reverse proxy set
        trust_proxy=True so the FIRST upstream hop's forwarded
        headers are honoured (otherwise OAuth redirect URIs end
        up with the proxy's internal address)."""
        from werkzeug.middleware.proxy_fix import ProxyFix
        app = self._flask_with([_google_entry()], trust_proxy=True)
        assert isinstance(app.wsgi_app, ProxyFix), (
            "trust_proxy=True did NOT install ProxyFix; OAuth "
            "redirect URIs built behind a reverse proxy will be "
            "wrong and providers will reject the callback"
        )


class TestAuthlibNamespaceCollisionProtection:
    """The OAuth providers module mangles every operator-chosen id
    into ``mb_<id>`` before passing it to Authlib's ``OAuth.register``.

    Why: Authlib exposes registered clients as attributes on the
    ``OAuth`` instance via ``__getattr__`` (so ``oauth.google``
    returns the registered "google" client).  If an operator chose
    ``id="register"``, calling ``getattr(oauth, "register")`` would
    return either the ``register`` METHOD (shadowing) or the
    registered client (depending on lookup order) -- a footgun.

    The protection has two halves that BOTH must hold:

      1. ``_authlib_name(operator_id)`` returns ``"mb_" + operator_id``
         -- every registration uses the mangled name, no collisions
         possible with attributes of the ``OAuth`` class.
      2. The schema validator REJECTS any ``id`` matching ``^mb_``
         (tested in ``TestCommonFields::test_id_reserved_mb_prefix_rejected``)
         -- an operator can't pick ``id="mb_register"`` and unwind
         the prefix.

    This test pins half (1); ``test_id_reserved_mb_prefix_rejected``
    pins half (2).  Together they prove no operator-chosen ``id``
    can ever resolve to an authlib instance attribute.
    """

    def test_authlib_name_prefixes_every_id(self):
        """An operator id reaching authlib unmangled. Authlib exposes
        registered clients as attributes on the `OAuth` instance, so an id of
        `register` or `cache` collides with the instance's own methods --
        registration then either shadows a method or silently returns the wrong
        object at login.

        `deployment.md` § 3. This is half one of the protection; half two is
        `TestCommonFields::test_id_reserved_mb_prefix_rejected`. Honest note
        (2026-09-09): the sibling
        `test_authlib_name_prefix_constant_is_mb_underscore` asserts the same
        `mb_` literal this test already hard-codes, and is raised as a cut
        candidate.
        """
        from molbuilder.web.auth_providers.oauth import _authlib_name
        # Operator ids -- valid slugs per the schema regex.  The
        # mangler must produce ``mb_<id>`` for every one.
        for operator_id in ("google", "github", "register", "cache",
                            "init_app", "x", "my-org-github", "g1"):
            assert _authlib_name(operator_id) == f"mb_{operator_id}", (
                f"_authlib_name({operator_id!r}) did not produce the "
                f"mb_-prefixed name; the authlib-namespace collision "
                f"protection is broken"
            )



# --------------------------------------------------------------------- #
#  Google-specific                                                      #
# --------------------------------------------------------------------- #


class TestGoogleSpecific:

    def test_hosted_domain_defaults_empty(self):
        """A Google provider without `hosted_domain` failing to load, or
        defaulting to a restriction: the documented default is no domain
        restriction, with `allowed_users` doing the gating.

        `access-control.md` § 3.1 (the allowlist is the gate); the field is
        normalised in `runtime_config._validate_google`.
        """
        cfg = _normalise(_wrap(_google_entry()))
        assert get_providers(cfg)[0]["hosted_domain"] == []

    def test_hosted_domain_list_of_strings(self):
        """The key dropped or misspelled during normalisation. The enforcement
        site then sees no domain restriction and admits accounts from outside
        the domain, while the config still shows the restriction the operator
        wrote -- a fail-OPEN with no error anywhere.

        `access-control.md` § 3.1.
        """
        cfg = _normalise(_wrap(_google_entry(
            hosted_domain=["asu.edu", "anothersite.org"]
        )))
        assert get_providers(cfg)[0]["hosted_domain"] == [
            "asu.edu", "anothersite.org",
        ]



# --------------------------------------------------------------------- #
#  GitHub-specific                                                      #
# --------------------------------------------------------------------- #


class TestGitHubSpecific:

    def _entry(self, **kw):
        return _google_entry(kind="github", id="github", **kw)

    def test_allowed_organizations_defaults_empty(self):
        """A GitHub provider without `allowed_organizations` failing to load,
        or defaulting to a restriction: as with Google's domain, the documented
        default is no org restriction and `allowed_users` gates.

        `access-control.md` § 3.1.
        """
        cfg = _normalise(_wrap(self._entry()))
        assert get_providers(cfg)[0]["allowed_organizations"] == []

    def test_allowed_organizations_accepted(self):
        """The key dropped or renamed in normalisation, so an org restriction
        the operator wrote is never enforced -- the same fail-open as
        `hosted_domain`, on the GitHub path.

        `access-control.md` § 3.1.
        """
        cfg = _normalise(_wrap(self._entry(
            allowed_organizations=["my-org", "another-org"]
        )))
        assert get_providers(cfg)[0]["allowed_organizations"] == [
            "my-org", "another-org",
        ]



# --------------------------------------------------------------------- #
#  Microsoft-specific                                                   #
# --------------------------------------------------------------------- #


class TestMicrosoftSpecific:

    def _entry(self, **kw):
        return _google_entry(kind="microsoft", id="microsoft", **kw)

    def test_tenant_id_defaults_to_common(self):
        """The Microsoft tenant defaulting to something other than `common`: a
        narrower default silently refuses every personal Microsoft account, and
        a wrong tenant sends users to a sign-in page for an organisation they
        are not in.

        `deployment.md` § 3.4 (the other OAuth providers take the same shape,
        written by hand); the default lives in
        `runtime_config._validate_microsoft`.
        """
        cfg = _normalise(_wrap(self._entry()))
        assert get_providers(cfg)[0]["tenant_id"] == "common"

    def test_tenant_id_string_accepted(self):
        """A configured tenant ignored in favour of the default: an operator
        restricting sign-in to their tenant gets `common` and every Microsoft
        account instead -- fail-open, and invisible in the config file.

        `deployment.md` § 3.4.
        """
        cfg = _normalise(_wrap(self._entry(tenant_id="asu.onmicrosoft.com")))
        assert get_providers(cfg)[0]["tenant_id"] == "asu.onmicrosoft.com"




# --------------------------------------------------------------------- #
#  ORCID-specific                                                       #
# --------------------------------------------------------------------- #


class TestORCIDSpecific:

    def test_minimal_entry_valid(self):
        """An ORCID entry routed through another kind's validator, which would
        add `hosted_domain` or `tenant_id` defaults that have no meaning for
        it.

        `deployment.md` § 3.4 (ORCID takes the shared OAuth shape and adds
        nothing). Honest note (2026-09-09): the audit records this as UNSURE
        rather than a cut -- no shipped code reads those keys for an ORCID
        provider, so the mixup it catches may have no consequence.
        """
        cfg = _normalise(_wrap(_google_entry(kind="orcid", id="orcid")))
        p = get_providers(cfg)[0]
        assert p["kind"] == "orcid"
        # No ORCID-only extras today; the entry should round-trip
        # without surprise additions.
        assert "hosted_domain" not in p
        assert "allowed_organizations" not in p
        assert "tenant_id" not in p


# --------------------------------------------------------------------- #
#  CAS-specific                                                         #
# --------------------------------------------------------------------- #


class TestCASSpecific:

    def test_minimal_entry_valid(self):
        """The CAS defaults drifting. `version` decides which ticket-validation
        protocol is spoken, so a default of 1 or 2 makes every ticket fail
        against a v3 server; and `service_url` / `ca_certs` / `email_attribute`
        must default to None rather than to a string, because the client
        branches on their absence.

        `deployment.md` § 3.2 (ASURITE sign-in);
        `runtime_config._validate_cas`.
        """
        cfg = _normalise(_wrap(_cas_entry()))
        p = get_providers(cfg)[0]
        assert p["kind"] == "cas"
        assert p["login_url"] == "https://cas.example.com/cas/login"
        assert p["version"] == 3                # default
        assert p["service_url"] is None
        assert p["ca_certs"] is None
        assert p["email_attribute"] is None
        assert p["email_domain"] == "example.com"


    def test_version_accepts_1_2_3(self):
        """The version check tightening to a single value, which would refuse a
        legitimate CAS 2 deployment at startup.

        `deployment.md` § 3.2.
        """
        for v in (1, 2, 3):
            cfg = _normalise(_wrap(_cas_entry(version=v)))
            assert get_providers(cfg)[0]["version"] == v


    def test_optional_strings_accepted(self):
        """An optional CAS field dropped in normalisation: `ca_certs`
        disappearing turns certificate verification into whatever the default
        trust store does, and `service_url` disappearing sends ticket
        validation to the wrong callback.

        `deployment.md` § 3.2.
        """
        cfg = _normalise(_wrap(_cas_entry(
            service_url="https://app.example.com/cas-callback/asu_cas",
            ca_certs="/etc/ssl/certs/ca-certificates.crt",
            email_attribute="mail",
        )))
        p = get_providers(cfg)[0]
        assert p["service_url"].endswith("/cas-callback/asu_cas")
        assert p["ca_certs"] == "/etc/ssl/certs/ca-certificates.crt"
        assert p["email_attribute"] == "mail"



    def test_attribute_only_accepted(self):
        """The two-way requirement read as 'both required': a CAS server that
        DOES release an email attribute would be forced to declare a synthesis
        domain too, and the synthesised address would then shadow the real one.

        `deployment.md` § 3.2.
        """
        entry = _cas_entry(email_attribute="mail")
        del entry["email_domain"]
        cfg = _normalise(_wrap(entry))
        p = get_providers(cfg)[0]
        assert p["email_attribute"] == "mail"
        assert p["email_domain"] is None

    def test_both_attribute_and_domain_accepted(self):
        """The fallback chain (try attribute first; synthesise from
        domain when missing) requires both to be settable."""
        cfg = _normalise(_wrap(_cas_entry(
            email_attribute="mail", email_domain="example.com"
        )))
        p = get_providers(cfg)[0]
        assert p["email_attribute"] == "mail"
        assert p["email_domain"] == "example.com"

    def test_cas_does_not_need_client_id(self):
        """CAS is not OAuth -- no client credentials, no secret."""
        entry = _cas_entry()
        cfg = _normalise(_wrap(entry))
        p = get_providers(cfg)[0]
        assert "client_id" not in p
        assert "client_secret" not in p


# --------------------------------------------------------------------- #
#  Mixed-provider config (typical real-world shape)                     #
# --------------------------------------------------------------------- #


class TestMixedConfig:

    def test_google_plus_cas_round_trip(self):
        """The expected real deployment shape: institutional CAS for
        on-network users + Google for external collaborators."""
        raw = _wrap(_google_entry(), _cas_entry())
        cfg = _normalise(raw)
        provs = get_providers(cfg)
        assert len(provs) == 2
        assert provs[0]["kind"] == "google"
        assert provs[1]["kind"] == "cas"

    def test_all_five_kinds_accepted(self):
        """A kind listed as supported but not wired into the validator
        registry: the operator's config is refused with 'not supported' for a
        provider the documentation offers.

        `deployment.md` § 3.4 (Google and CAS through the wizard; GitHub,
        Microsoft and ORCID by hand).
        """
        cfg = _normalise(_wrap(
            _google_entry(),
            _google_entry(kind="github",    id="github"),
            _google_entry(kind="microsoft", id="microsoft"),
            _google_entry(kind="orcid",     id="orcid"),
            _cas_entry(),
        ))
        kinds = [p["kind"] for p in get_providers(cfg)]
        assert kinds == ["google", "github", "microsoft", "orcid", "cas"]


# `TestSecretKeyFileIsRetired` retired 2026-10-02 (W54 T6, T8): its first
# check was met by the refused key's own quoted name, and its second
# repeated a row; the top-level key is a `molbuilder_json.toml` row, the
# key inside `auth` a MALFORMED_AUTH row.


# --------------------------------------------------------------------- #
#  auth.trust_proxy (ProxyFix opt-in flag)                              #
# --------------------------------------------------------------------- #


class TestTrustProxy:
    """``auth.trust_proxy`` defaults to False -- the safe choice for
    direct-TLS deployments where any incoming X-Forwarded-* header
    would be attacker-controlled.  Operators behind an actual reverse
    proxy that scrubs+sets those headers must set the flag explicitly."""

    def test_default_is_false(self):
        """`trust_proxy` defaulting to True: `ProxyFix` would be installed on a
        direct-TLS deployment, where `X-Forwarded-Host` is attacker-controlled
        -- and the OAuth redirect URI molbuilder builds could then be pointed
        at another host.

        `deployment.md` § 5 (`auth.trust_proxy` installs ProxyFix); auth review
        P1 #7.
        """
        cfg = _normalise(_wrap(_google_entry()))
        assert cfg["auth"]["trust_proxy"] is False

    def test_explicit_true_accepted(self):
        """The opt-in silently ignored: an operator behind a reverse proxy sets
        the flag, ProxyFix is not installed, and every OAuth redirect URI
        carries the proxy's internal address -- sign-in then fails at the
        provider with a URI mismatch.

        `deployment.md` § 5.
        """
        raw = _wrap(_google_entry())
        raw["auth"]["trust_proxy"] = True
        cfg = _normalise(raw)
        assert cfg["auth"]["trust_proxy"] is True




# --------------------------------------------------------------------- #
#  Runtime: authenticate() allowlist enforcement                        #
# --------------------------------------------------------------------- #
#
# Schema validation pins the SHAPE of allowed_users; these tests pin
# the RUNTIME match (case-insensitivity via casefold, per-provider
# isolation, fail-closed semantics on empty list).


class TestAuthenticate:

    def _app_with(self, providers):
        """Build a real Flask app with auth wired so authenticate()
        can run inside an app context."""
        from flask import Flask
        from molbuilder.web.auth import init_auth
        app = Flask(__name__)
        app.config["TESTING"] = True
        # Use an in-process secret key (no file needed for these tests).
        app.config["SECRET_KEY"] = b"x" * 32
        # init_auth expects an auth_cfg shape matching _normalise output.
        init_auth(app,
                  auth_cfg={"providers": providers, "trust_proxy": False})
        # init_auth installs the key from its ONE home, so restore the
        # in-process one these tests sign with.
        app.config["SECRET_KEY"] = b"x" * 32
        return app

    def test_exact_match_accepted(self):
        """An allowlist match that never succeeds: the gate refuses the people
        it exists to admit, and the only symptom is a 403 after a successful
        sign-in at the provider.

        `access-control.md` § 3 (step 5: the email is checked against that
        provider's list; step 6 returns the user to the page they asked for,
        which is the redirect asserted here).
        """
        from molbuilder.web.auth import authenticate
        app = self._app_with([_google_entry(
            allowed_users=["alice@example.com"]
        )])
        with app.test_request_context("/some-page"):
            resp = authenticate("google", "alice@example.com", {})
        # Successful sign-in returns a redirect.  Flask routes return
        # a Response; redirect() returns one with status 302.
        assert hasattr(resp, "status_code") and resp.status_code == 302

    def test_match_is_case_insensitive_both_sides(self):
        """Case handled on one side only: an operator who writes
        `Alice@Example.COM` -- as a mail client displays it -- locks Alice out,
        and the config looks correct.

        `access-control.md` § 3.1. The parse site preserves case deliberately
        (`TestAllowedUsers::test_preserved_verbatim`), so the enforcement site
        is the only place folding may happen.
        """
        from molbuilder.web.auth import authenticate
        app = self._app_with([_google_entry(
            allowed_users=["Alice@Example.COM"]
        )])
        with app.test_request_context("/"):
            resp = authenticate("google", "alice@example.com", {})
        assert hasattr(resp, "status_code") and resp.status_code == 302

    def test_casefold_handles_non_ascii(self):
        """casefold() folds German ß to 'ss', so the two strings
        match.  Plain .lower() would not."""
        from molbuilder.web.auth import authenticate
        app = self._app_with([_google_entry(
            allowed_users=["straße@example.com"]
        )])
        with app.test_request_context("/"):
            resp = authenticate("google", "STRASSE@EXAMPLE.COM", {})
        assert hasattr(resp, "status_code") and resp.status_code == 302

    def test_unknown_email_denied(self):
        """The gate admitting an identity that is not on the list -- the whole
        point of the allowlist. The message must name the rejected identity AND
        the provider, or an operator diagnosing a locked-out colleague has to
        read server logs to learn which of two providers refused them.

        `access-control.md` § 3 and § 3.1.
        """
        from molbuilder.web.auth import authenticate
        app = self._app_with([_google_entry(
            allowed_users=["alice@example.com"]
        )])
        with app.test_request_context("/"):
            body, status = authenticate("google", "bob@example.com", {})
        assert status == 403
        # The denial message should name the rejected identity + the
        # provider so the operator can diagnose without server logs.
        assert "bob@example.com" in body
        assert "google" in body

    def test_empty_allowlist_denies_everyone(self):
        """An empty list read as 'no restriction' instead of 'no one' -- the
        fail-OPEN reading of the same data, and the reason the schema accepts
        an empty list as valid rather than refusing it.

        `access-control.md` § 3.1; the schema half is
        `TestAllowedUsers::test_empty_list_accepted_fail_closed`.
        """
        from molbuilder.web.auth import authenticate
        app = self._app_with([_google_entry(allowed_users=[])])
        with app.test_request_context("/"):
            body, status = authenticate("google", "anyone@example.com", {})
        assert status == 403

    def test_per_provider_isolation(self):
        """An email allowed via google is NOT implicitly allowed via
        github -- each provider has its own list."""
        from molbuilder.web.auth import authenticate
        app = self._app_with([
            _google_entry(id="google", kind="google",
                          allowed_users=["alice@example.com"]),
            _google_entry(id="github", kind="github",
                          allowed_users=["bob@example.com"]),
        ])
        with app.test_request_context("/"):
            # alice CAN sign in via google
            resp = authenticate("google", "alice@example.com", {})
            assert resp.status_code == 302
            # alice CANNOT sign in via github
            body, status = authenticate("github", "alice@example.com", {})
            assert status == 403

    def test_unknown_provider_id_404(self):
        """A callback for a provider id that is not configured falling through
        to a real provider, or 500ing. The id arrives in the URL, so this is
        attacker-controlled input at the authentication boundary.

        `access-control.md` § 3 (the callback routes dispatch by id).
        """
        from molbuilder.web.auth import authenticate
        from werkzeug.exceptions import NotFound
        app = self._app_with([_google_entry()])
        with app.test_request_context("/"):
            with pytest.raises(NotFound):
                authenticate("not-a-real-provider", "x@y.z", {})


# --------------------------------------------------------------------- #
#  Runtime: CAS _extract_email fallback chain                           #
# --------------------------------------------------------------------- #


class TestCASExtractEmail:
    """Pin the (attribute -> domain) fallback documented in the
    cas.py module docstring."""

    def _entry(self, **kw):
        base = {"id": "asu_cas", "email_attribute": None,
                "email_domain": None}
        base.update(kw)
        return base

    def test_attribute_string_wins_when_present(self):
        """The configured attribute ignored in favour of the synthesised
        `principal@domain`: a CAS server that releases a real mailbox is
        overridden by a guess, and the guess is what gets matched against
        `allowed_users`.

        `access-control.md` § 3 (identity from CAS); the chain itself is
        documented in `web/auth_providers/cas.py`'s module docstring.
        """
        from molbuilder.web.auth_providers.cas import _extract_email
        email, denied = _extract_email(
            "jdoe",
            {"mail": "jdoe@asu.edu"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "jdoe@asu.edu"
        assert denied is None

    def test_attribute_list_first_element_wins(self):
        """CAS attributes can come back as ``"x"`` OR ``["x"]``; both
        are valid python-cas responses depending on the server."""
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "jdoe",
            {"mail": ["jdoe@asu.edu", "alt@asu.edu"]},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "jdoe@asu.edu"

    def test_attribute_empty_list_falls_through_to_domain(self):
        """An attribute that is present but EMPTY being returned as the email:
        an empty address reaches the allowlist match and denies a user the
        domain fallback would have admitted.

        `cas.py` module docstring (the attribute-then-domain chain); the guard
        is `isinstance(raw, list) and raw`.
        """
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "jdoe",
            {"mail": []},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        # Empty attribute -> synthesised
        assert email == "jdoe@asu.edu"

    def test_attribute_missing_falls_through_to_domain(self):
        """A configured attribute the server does not release ending the chain
        instead of falling through. ASU CAS releases only the principal, so
        this is the normal case rather than an edge one.

        `deployment.md` § 3.2; `cas.py` module docstring.
        """
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "jdoe",
            {"otherattr": "x"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "jdoe@asu.edu"

    def test_no_attribute_configured_just_synthesises(self):
        """The synthesis path requiring an attribute to be configured first:
        the ASURITE deployment sets only `email_domain`, so this is the shape
        that actually ships.

        `deployment.md` § 3.2.
        """
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "jdoe",
            {},   # no attributes at all (ASU CAS behaviour)
            self._entry(email_domain="asu.edu"),
        )
        assert email == "jdoe@asu.edu"

    def test_lowercases_the_result(self):
        """A mixed-case email leaving CAS unnormalised. The enforcement site
        folds both sides, so this is defence in depth -- what it really holds
        is that the attribute path and the synthesis path agree on case, so one
        person is one identity however their IdP spells it.

        `access-control.md` § 3.1.
        """
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "JDoe",
            {"mail": "JDoe@ASU.EDU"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "jdoe@asu.edu"

    def test_no_attribute_no_domain_yields_no_email(self):
        """Schema validation should make this unreachable, but pin the
        defensive behaviour anyway."""
        from molbuilder.web.auth_providers.cas import _extract_email
        email, denied = _extract_email(
            "jdoe", {}, self._entry()
        )
        assert email is None
        assert denied is None


# --------------------------------------------------------------------- #
#  Runtime: _safe_next_target open-redirect guard                       #
# --------------------------------------------------------------------- #


class TestSafeNextTarget:
    """Pin the open-redirect defence-in-depth helper."""

    @pytest.mark.parametrize("safe", [
        "/", "/spectra", "/api/health", "/projects/foo/bar",
        "/with-dash", "/with_under", "/with%20space",
    ])
    def test_safe_paths_pass_through(self, safe):
        """An over-tight guard clamping every legitimate `next` to `/`: signing
        in would always land on the home page instead of the page the user
        asked for, which is the only reason the parameter exists.

        `access-control.md` § 3 (step 6: the user lands on the page they
        originally asked for).
        """
        from molbuilder.web.auth import _safe_next_target
        assert _safe_next_target(safe) == safe

    @pytest.mark.parametrize("dangerous", [
        # Protocol-relative URL -> browsers go cross-host
        "//evil.example.com/phish",
        # Absolute URL with explicit scheme
        "http://evil.example.com/",
        "https://evil.example.com/",
        # JavaScript URL (would execute on redirect)
        "javascript:alert(1)",
        # Empty / whitespace / non-string
        "", "   ", None, 42, [], {},
        # Backslash trickery (Windows path)
        r"\\evil.example.com\path",
        r"/\\evil.example.com",
    ])
    def test_dangerous_inputs_clamped_to_root(self, dangerous):
        """An open redirect at the login boundary. `//evil.example.com` and
        `http://evil.example.com` are the classic phishing carriers -- a link
        to the real, trusted molbuilder host that deposits the user on someone
        else's page the moment they sign in; `javascript:` would execute on the
        redirect; and the non-string cases are what a crafted query string
        sends.

        `access-control.md` § 3 (the stashed next target), as defence in depth
        behind Flask's own redirect handling.
        """
        from molbuilder.web.auth import _safe_next_target
        assert _safe_next_target(dangerous) == "/"
