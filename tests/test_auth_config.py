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
      - EXACTLY one of client_secret / client_secret_file
      - kind-specific extras validated (hosted_domain, allowed_organizations,
        tenant_id)
  * CAS kind:
      - login_url + service_validate_url required
      - version must be 1, 2, or 3
      - at least one of email_attribute / email_domain (for allowlist match)
      - optional string fields validated when set
  * allowed_users must be list of strings (empty list = no one, valid)
  * secret_key_file is optional + string when present

Does NOT test the runtime auth flow itself (Authlib OAuth + python-cas
ticket validation are integration-tested separately).
"""
from __future__ import annotations

import pytest

from molbuilder.runtime_config import (
    RuntimeConfigError, _normalise,
    get_auth, get_providers, get_secret_key_file,
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
        "client_secret_file": "/etc/molbuilder/google.secret",
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
        "service_validate_url": "https://cas.example.com/cas/serviceValidate",
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


class TestAuthDefaultOff:

    def test_empty_config_has_no_auth(self):
        cfg = _normalise({})
        assert get_auth(cfg) == {}
        assert get_providers(cfg) == []
        assert get_secret_key_file(cfg) is None

    def test_config_without_auth_section_unaffected(self):
        cfg = _normalise({
            "tls": {"cert": "/tmp/c", "key": "/tmp/k"},
            "envs": {"siesta": "molbuilder-siesta"},
        })
        assert get_auth(cfg) == {}
        assert get_providers(cfg) == []


# --------------------------------------------------------------------- #
#  Providers list shape                                                 #
# --------------------------------------------------------------------- #


class TestProvidersListShape:

    def test_missing_providers_rejected(self):
        with pytest.raises(RuntimeConfigError, match="auth.providers"):
            _normalise({"auth": {}})

    def test_empty_providers_rejected(self):
        with pytest.raises(RuntimeConfigError, match="non-empty"):
            _normalise({"auth": {"providers": []}})

    def test_non_list_providers_rejected(self):
        with pytest.raises(RuntimeConfigError, match="non-empty list"):
            _normalise({"auth": {"providers": {"id": "x"}}})

    def test_non_object_entry_rejected(self):
        with pytest.raises(RuntimeConfigError, match="must be an object"):
            _normalise({"auth": {"providers": ["not-an-object"]}})

    def test_duplicate_ids_rejected(self):
        raw = _wrap(
            _google_entry(id="x"),
            _google_entry(id="x", client_id="other.apps.googleusercontent.com"),
        )
        with pytest.raises(RuntimeConfigError, match="duplicate provider id"):
            _normalise(raw)

    def test_two_providers_with_distinct_ids_ok(self):
        cfg = _normalise(_wrap(_google_entry(), _cas_entry()))
        ids = [p["id"] for p in get_providers(cfg)]
        assert ids == ["google", "asu_cas"]


# --------------------------------------------------------------------- #
#  Common per-entry fields                                              #
# --------------------------------------------------------------------- #


class TestCommonFields:

    @pytest.mark.parametrize("missing", ["id", "label", "kind"])
    def test_missing_required_field_rejected(self, missing):
        entry = _google_entry()
        del entry[missing]
        with pytest.raises(RuntimeConfigError, match=missing):
            _normalise(_wrap(entry))

    @pytest.mark.parametrize("field", ["id", "label", "kind"])
    def test_empty_required_field_rejected(self, field):
        with pytest.raises(RuntimeConfigError):
            _normalise(_wrap(_google_entry(**{field: ""})))

    def test_id_must_be_slug(self):
        # Digits are allowed anywhere (including leading), since they
        # are URL-safe; the regex bars only uppercase, whitespace,
        # punctuation, and leading underscore/hyphen.
        for bad_id in ("Google", "google!", "asu cas", "_google", "-google"):
            with pytest.raises(RuntimeConfigError, match="URL-safe slug"):
                _normalise(_wrap(_google_entry(id=bad_id)))

    def test_id_reserved_mb_prefix_rejected(self):
        """The mb_ prefix is reserved -- ``auth_providers/oauth.py``
        mangles operator ids to ``mb_<id>`` before passing them to
        Authlib (to avoid colliding with OAuth instance attributes
        like ``register`` / ``cache``).  Operators picking ``mb_X``
        would unwind that protection."""
        for bad_id in ("mb_google", "mb_register", "mb_anything"):
            with pytest.raises(RuntimeConfigError, match="URL-safe slug"):
                _normalise(_wrap(_google_entry(id=bad_id)))

    def test_id_good_slugs_accepted(self):
        for good_id in ("google", "asu_cas", "g-1", "my-org-github"):
            cfg = _normalise(_wrap(_google_entry(id=good_id)))
            assert get_providers(cfg)[0]["id"] == good_id

    def test_unsupported_kind_rejected(self):
        for bad in ("ldap", "saml", "openldap", "kerberos", "facebook"):
            with pytest.raises(RuntimeConfigError, match="not supported"):
                _normalise(_wrap(_google_entry(kind=bad)))


# --------------------------------------------------------------------- #
#  allowed_users (per-provider, required)                               #
# --------------------------------------------------------------------- #


class TestAllowedUsers:

    def test_missing_rejected(self):
        entry = _google_entry()
        del entry["allowed_users"]
        with pytest.raises(RuntimeConfigError, match="allowed_users"):
            _normalise(_wrap(entry))

    def test_string_rejected(self):
        with pytest.raises(RuntimeConfigError, match="list of strings"):
            _normalise(_wrap(_google_entry(allowed_users="user@example.com")))

    def test_list_of_non_strings_rejected(self):
        with pytest.raises(RuntimeConfigError, match="list of strings"):
            _normalise(_wrap(_google_entry(
                allowed_users=["user@example.com", 42]
            )))

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

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_missing_client_id_rejected(self, kind):
        entry = _google_entry(kind=kind, id=kind)
        del entry["client_id"]
        with pytest.raises(RuntimeConfigError, match="client_id"):
            _normalise(_wrap(entry))

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_both_secret_forms_rejected(self, kind):
        entry = _google_entry(
            kind=kind, id=kind,
            client_secret="literal",
            client_secret_file="/etc/x",
        )
        with pytest.raises(RuntimeConfigError, match="EITHER"):
            _normalise(_wrap(entry))

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_neither_secret_form_rejected(self, kind):
        entry = _google_entry(kind=kind, id=kind)
        del entry["client_secret_file"]
        with pytest.raises(RuntimeConfigError) as exc:
            _normalise(_wrap(entry))
        msg = str(exc.value)
        import re as _re
        # \b keeps this from being satisfied by "client_secret_file"
        # alone -- the error must offer BOTH forms.
        assert _re.search(r"client_secret\b", msg), (
            "the error must offer the inline client_secret form too")
        assert "client_secret_file" in msg

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_literal_secret_accepted(self, kind):
        entry = _google_entry(kind=kind, id=kind)
        del entry["client_secret_file"]
        entry["client_secret"] = "GOCSPX-literal"
        cfg = _normalise(_wrap(entry))
        p = get_providers(cfg)[0]
        assert p["client_secret"] == "GOCSPX-literal"
        assert "client_secret_file" not in p

    @pytest.mark.parametrize("kind", ["google", "github", "microsoft", "orcid"])
    def test_secret_file_accepted(self, kind):
        cfg = _normalise(_wrap(_google_entry(kind=kind, id=kind)))
        p = get_providers(cfg)[0]
        assert p["client_secret_file"] == "/etc/molbuilder/google.secret"
        assert "client_secret" not in p


class TestSecretFileMtimeReload:
    """``_ensure_client`` in molbuilder/web/auth_providers/oauth.py
    watches the OAuth provider's ``client_secret_file`` for mtime
    changes and re-reads + applies the new secret in-place when the
    file is rotated.  Operator can fix a wrong GOCSPX value WITHOUT
    restarting the server (task #100).

    These tests pin:
      * Second call after rotation picks up the new secret value.
      * The SAME authlib client object is mutated (cache identity
        preserved -- otherwise concurrent in-flight callbacks would
        see different clients).
      * Failed re-read (file deleted / emptied) falls back to keeping
        the previously-loaded secret rather than crashing or
        zeroing-out the client.
      * Literal-secret entries (no client_secret_file) skip the
        mtime check entirely (no file to stat).
    """

    def _entry(self, secret_file_path):
        return {
            "id":                 "google",
            "label":              "Sign in with Google",
            "kind":               "google",
            "client_id":          "test.apps.googleusercontent.com",
            "client_secret_file": secret_file_path,
            "hosted_domain":      [],
            "allowed_users":      ["user@example.com"],
        }

    def _app(self):
        from flask import Flask
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = b"x" * 32
        return app

    def _bump_mtime(self, path, seconds_ahead=2):
        """Move the file's mtime forward.  ``Path.write_text`` already
        bumps mtime on most filesystems, but second-granular FSes
        (ext4 sometimes, FAT always) can collapse two writes within
        the same second to the same mtime -- which would defeat the
        change-detection.  Explicit utime forces a detectable jump."""
        import os, time
        future = time.time() + seconds_ahead
        os.utime(str(path), (future, future))

    def test_first_call_reads_secret_and_records_mtime(self, tmp_path):
        from molbuilder.web.auth_providers.oauth import (
            _ensure_client, _OAUTH_CLIENTS_EXT_KEY,
        )
        secret_file = tmp_path / "client_secret"
        secret_file.write_text("GOCSPX-original")
        app   = self._app()
        entry = self._entry(str(secret_file))

        with app.app_context():
            client = _ensure_client(app, entry)
        # The client carries the secret literally; authlib clients
        # expose it as a plain attribute.
        assert client.client_secret == "GOCSPX-original"
        # mtime recorded for change-detection on the next call.
        ext = app.extensions[_OAUTH_CLIENTS_EXT_KEY]
        assert ext["secret_mtime"]["mb_google"] is not None

    def test_secret_change_is_picked_up_without_restart(self, tmp_path):
        """The whole point of task #100: edit the file, hit a callback,
        new secret is in effect.  Verifies (a) the value updates AND
        (b) the same client object is reused (cache identity)."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        secret_file = tmp_path / "client_secret"
        secret_file.write_text("GOCSPX-original")
        app   = self._app()
        entry = self._entry(str(secret_file))

        with app.app_context():
            client_first  = _ensure_client(app, entry)
            assert client_first.client_secret == "GOCSPX-original"

            secret_file.write_text("GOCSPX-rotated")
            self._bump_mtime(secret_file)

            client_second = _ensure_client(app, entry)

        # Same authlib client object, with the secret mutated in place.
        # If the registry built a NEW client instead, in-flight token
        # exchanges using the old reference would silently keep the
        # old secret -- this test catches a "destroy + rebuild" regression.
        assert client_second is client_first
        assert client_second.client_secret == "GOCSPX-rotated"

    def test_unchanged_mtime_keeps_cached_secret(self, tmp_path):
        """Second call with the SAME mtime MUST NOT re-read.  The
        detection signal is mtime, not content hash -- if a tool
        rewrites the file while preserving the original mtime (rsync
        ``--archive``, a same-second overwrite on a low-granularity
        FS, etc.), we don't pick up the change.  Documented contract:
        the mtime check is the fast-path; operators MUST cause an
        mtime advance for the hot-reload to fire."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        secret_file = tmp_path / "client_secret"
        secret_file.write_text("GOCSPX-original")
        app   = self._app()
        entry = self._entry(str(secret_file))

        with app.app_context():
            _ensure_client(app, entry)
            # Capture the post-registration mtime, mutate content,
            # then RESTORE the original mtime.  Models a tool that
            # writes-with-preserved-timestamps (cp -p, rsync --times).
            orig_stat = secret_file.stat()
            secret_file.write_text("GOCSPX-this-shouldnt-be-picked-up")
            import os
            os.utime(str(secret_file),
                      (orig_stat.st_atime, orig_stat.st_mtime))

            client = _ensure_client(app, entry)
        # mtime unchanged -> change-detection skipped -> cached
        # secret in memory stands.
        assert client.client_secret == "GOCSPX-original"

    def test_file_deleted_keeps_previously_loaded_secret(self, tmp_path):
        """If the operator deletes / moves the secret file between
        calls, the running app must NOT crash; it keeps the secret
        already in memory (the active OAuth flow is non-fatal)."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        secret_file = tmp_path / "client_secret"
        secret_file.write_text("GOCSPX-original")
        app   = self._app()
        entry = self._entry(str(secret_file))

        with app.app_context():
            _ensure_client(app, entry)
            secret_file.unlink()
            # _secret_file_mtime returns None now; that's treated as
            # "no detectable change", so the cached secret stands.
            client = _ensure_client(app, entry)
        assert client.client_secret == "GOCSPX-original"

    def test_empty_file_after_rotation_keeps_previous_secret(self, tmp_path):
        """If the operator's rotation script writes an empty file
        (clobbered + not-yet-rewritten state, or a tool that truncates
        before writing), molbuilder must NOT zero out the active
        client_secret -- the next callback would then send no secret
        at all and Google would return invalid_client.  Keep the
        previously-loaded secret and log a warning."""
        from molbuilder.web.auth_providers.oauth import _ensure_client
        secret_file = tmp_path / "client_secret"
        secret_file.write_text("GOCSPX-original")
        app   = self._app()
        entry = self._entry(str(secret_file))

        with app.app_context():
            _ensure_client(app, entry)
            secret_file.write_text("")    # empty -- mid-rotation
            self._bump_mtime(secret_file)
            client = _ensure_client(app, entry)
        # _read_secret raises RuntimeError on empty file; the helper
        # catches + logs; the previously-loaded secret stays.
        assert client.client_secret == "GOCSPX-original"

    def test_literal_secret_entries_skip_mtime_tracking(self):
        """When the entry uses ``client_secret`` (literal) instead of
        ``client_secret_file`` (path), there is no file to watch.
        ``_secret_file_mtime`` returns None; the mtime-change branch
        in ``_ensure_client`` is never taken; the literal secret is
        used verbatim across all calls."""
        from molbuilder.web.auth_providers.oauth import (
            _ensure_client, _OAUTH_CLIENTS_EXT_KEY,
        )
        app = self._app()
        entry = {
            "id":            "google",
            "label":         "Sign in with Google",
            "kind":          "google",
            "client_id":     "test.apps.googleusercontent.com",
            "client_secret": "GOCSPX-literal-not-a-file",
            "hosted_domain": [],
            "allowed_users": ["user@example.com"],
        }
        with app.app_context():
            c1 = _ensure_client(app, entry)
            c2 = _ensure_client(app, entry)
        assert c1 is c2
        assert c1.client_secret == "GOCSPX-literal-not-a-file"
        # Nothing recorded for the mtime watcher.
        ext = app.extensions[_OAUTH_CLIENTS_EXT_KEY]
        assert ext["secret_mtime"]["mb_google"] is None


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
        init_auth(
            app,
            auth_cfg={"providers": providers, "trust_proxy": trust_proxy},
            secret_key_file=None,
        )
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

    def test_authlib_name_prefix_constant_is_mb_underscore(self):
        """The slug regex hard-codes ``^mb_`` as the reserved prefix.
        If someone changes the prefix in oauth.py without updating
        the validator, this test fires."""
        from molbuilder.web.auth_providers.oauth import (
            _AUTHLIB_INTERNAL_PREFIX,
        )
        assert _AUTHLIB_INTERNAL_PREFIX == "mb_", (
            f"the authlib-internal prefix is {_AUTHLIB_INTERNAL_PREFIX!r}, "
            f"not the ``mb_`` that the schema validator's _ID_RE "
            f"reserves; either align them or the prefix loses its "
            f"protection"
        )


# --------------------------------------------------------------------- #
#  Google-specific                                                      #
# --------------------------------------------------------------------- #


class TestGoogleSpecific:

    def test_hosted_domain_defaults_empty(self):
        cfg = _normalise(_wrap(_google_entry()))
        assert get_providers(cfg)[0]["hosted_domain"] == []

    def test_hosted_domain_list_of_strings(self):
        cfg = _normalise(_wrap(_google_entry(
            hosted_domain=["asu.edu", "anothersite.org"]
        )))
        assert get_providers(cfg)[0]["hosted_domain"] == [
            "asu.edu", "anothersite.org",
        ]

    def test_hosted_domain_non_list_rejected(self):
        with pytest.raises(RuntimeConfigError, match="list of strings"):
            _normalise(_wrap(_google_entry(hosted_domain="asu.edu")))


# --------------------------------------------------------------------- #
#  GitHub-specific                                                      #
# --------------------------------------------------------------------- #


class TestGitHubSpecific:

    def _entry(self, **kw):
        return _google_entry(kind="github", id="github", **kw)

    def test_allowed_organizations_defaults_empty(self):
        cfg = _normalise(_wrap(self._entry()))
        assert get_providers(cfg)[0]["allowed_organizations"] == []

    def test_allowed_organizations_accepted(self):
        cfg = _normalise(_wrap(self._entry(
            allowed_organizations=["my-org", "another-org"]
        )))
        assert get_providers(cfg)[0]["allowed_organizations"] == [
            "my-org", "another-org",
        ]

    def test_allowed_organizations_non_list_rejected(self):
        with pytest.raises(RuntimeConfigError, match="list of strings"):
            _normalise(_wrap(self._entry(allowed_organizations="my-org")))


# --------------------------------------------------------------------- #
#  Microsoft-specific                                                   #
# --------------------------------------------------------------------- #


class TestMicrosoftSpecific:

    def _entry(self, **kw):
        return _google_entry(kind="microsoft", id="microsoft", **kw)

    def test_tenant_id_defaults_to_common(self):
        cfg = _normalise(_wrap(self._entry()))
        assert get_providers(cfg)[0]["tenant_id"] == "common"

    def test_tenant_id_string_accepted(self):
        cfg = _normalise(_wrap(self._entry(tenant_id="asu.onmicrosoft.com")))
        assert get_providers(cfg)[0]["tenant_id"] == "asu.onmicrosoft.com"

    def test_tenant_id_empty_rejected(self):
        with pytest.raises(RuntimeConfigError, match="tenant_id"):
            _normalise(_wrap(self._entry(tenant_id="")))

    def test_tenant_id_non_string_rejected(self):
        with pytest.raises(RuntimeConfigError, match="tenant_id"):
            _normalise(_wrap(self._entry(tenant_id=42)))


# --------------------------------------------------------------------- #
#  ORCID-specific                                                       #
# --------------------------------------------------------------------- #


class TestORCIDSpecific:

    def test_minimal_entry_valid(self):
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
        cfg = _normalise(_wrap(_cas_entry()))
        p = get_providers(cfg)[0]
        assert p["kind"] == "cas"
        assert p["login_url"] == "https://cas.example.com/cas/login"
        assert p["version"] == 3                # default
        assert p["service_url"] is None
        assert p["ca_certs"] is None
        assert p["email_attribute"] is None
        assert p["email_domain"] == "example.com"

    @pytest.mark.parametrize("missing", ["login_url", "service_validate_url"])
    def test_missing_required_url_rejected(self, missing):
        entry = _cas_entry()
        del entry[missing]
        with pytest.raises(RuntimeConfigError, match=missing):
            _normalise(_wrap(entry))

    def test_version_accepts_1_2_3(self):
        for v in (1, 2, 3):
            cfg = _normalise(_wrap(_cas_entry(version=v)))
            assert get_providers(cfg)[0]["version"] == v

    def test_version_other_values_rejected(self):
        for bad in (0, 4, "3", 3.0, None):
            with pytest.raises(RuntimeConfigError, match="version"):
                _normalise(_wrap(_cas_entry(version=bad)))

    def test_optional_strings_accepted(self):
        cfg = _normalise(_wrap(_cas_entry(
            service_url="https://app.example.com/cas-callback/asu_cas",
            ca_certs="/etc/ssl/certs/ca-certificates.crt",
            email_attribute="mail",
        )))
        p = get_providers(cfg)[0]
        assert p["service_url"].endswith("/cas-callback/asu_cas")
        assert p["ca_certs"] == "/etc/ssl/certs/ca-certificates.crt"
        assert p["email_attribute"] == "mail"

    @pytest.mark.parametrize("field", [
        "service_url", "ca_certs", "email_attribute", "email_domain",
    ])
    def test_optional_string_empty_rejected(self, field):
        with pytest.raises(RuntimeConfigError, match=field):
            _normalise(_wrap(_cas_entry(**{field: ""})))

    def test_neither_email_attribute_nor_domain_rejected(self):
        entry = _cas_entry()
        del entry["email_domain"]
        with pytest.raises(RuntimeConfigError, match="email_attribute"):
            _normalise(_wrap(entry))

    def test_attribute_only_accepted(self):
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
        cfg = _normalise(_wrap(
            _google_entry(),
            _google_entry(kind="github",    id="github"),
            _google_entry(kind="microsoft", id="microsoft"),
            _google_entry(kind="orcid",     id="orcid"),
            _cas_entry(),
        ))
        kinds = [p["kind"] for p in get_providers(cfg)]
        assert kinds == ["google", "github", "microsoft", "orcid", "cas"]


# --------------------------------------------------------------------- #
#  secret_key_file (unchanged from previous schema)                     #
# --------------------------------------------------------------------- #


class TestSecretKeyFile:

    def test_optional(self):
        cfg = _normalise({})
        assert get_secret_key_file(cfg) is None

    def test_string_path_preserved(self):
        cfg = _normalise({"secret_key_file": "~/.mb/key"})
        assert get_secret_key_file(cfg) == "~/.mb/key"

    def test_non_string_rejected(self):
        with pytest.raises(RuntimeConfigError, match="string path"):
            _normalise({"secret_key_file": 42})


# --------------------------------------------------------------------- #
#  auth.trust_proxy (ProxyFix opt-in flag)                              #
# --------------------------------------------------------------------- #


class TestTrustProxy:
    """``auth.trust_proxy`` defaults to False -- the safe choice for
    direct-TLS deployments where any incoming X-Forwarded-* header
    would be attacker-controlled.  Operators behind an actual reverse
    proxy that scrubs+sets those headers must set the flag explicitly."""

    def test_default_is_false(self):
        cfg = _normalise(_wrap(_google_entry()))
        assert cfg["auth"]["trust_proxy"] is False

    def test_explicit_true_accepted(self):
        raw = _wrap(_google_entry())
        raw["auth"]["trust_proxy"] = True
        cfg = _normalise(raw)
        assert cfg["auth"]["trust_proxy"] is True

    def test_explicit_false_accepted(self):
        raw = _wrap(_google_entry())
        raw["auth"]["trust_proxy"] = False
        cfg = _normalise(raw)
        assert cfg["auth"]["trust_proxy"] is False

    @pytest.mark.parametrize("bad", [1, 0, "true", "false", None, []])
    def test_non_bool_rejected(self, bad):
        raw = _wrap(_google_entry())
        raw["auth"]["trust_proxy"] = bad
        with pytest.raises(RuntimeConfigError, match="trust_proxy"):
            _normalise(raw)


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
        import os, tempfile
        from flask import Flask
        from molbuilder.web.auth import init_auth
        app = Flask(__name__)
        app.config["TESTING"] = True
        # Use an in-process secret key (no file needed for these tests).
        app.config["SECRET_KEY"] = b"x" * 32
        # init_auth expects an auth_cfg shape matching _normalise output.
        init_auth(
            app,
            auth_cfg={"providers": providers, "trust_proxy": False},
            secret_key_file=None,   # already populated above
        )
        # ... but init_auth ALSO overwrites SECRET_KEY when called with
        # None -- restore.
        app.config["SECRET_KEY"] = b"x" * 32
        return app

    def test_exact_match_accepted(self):
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
        from molbuilder.web.auth_providers.cas import _extract_email
        email, denied = _extract_email(
            "qqing",
            {"mail": "qqing@asu.edu"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "qqing@asu.edu"
        assert denied is None

    def test_attribute_list_first_element_wins(self):
        """CAS attributes can come back as ``"x"`` OR ``["x"]``; both
        are valid python-cas responses depending on the server."""
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "qqing",
            {"mail": ["qqing@asu.edu", "alt@asu.edu"]},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "qqing@asu.edu"

    def test_attribute_empty_list_falls_through_to_domain(self):
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "qqing",
            {"mail": []},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        # Empty attribute -> synthesised
        assert email == "qqing@asu.edu"

    def test_attribute_missing_falls_through_to_domain(self):
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "qqing",
            {"otherattr": "x"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "qqing@asu.edu"

    def test_no_attribute_configured_just_synthesises(self):
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "qqing",
            {},   # no attributes at all (ASU CAS behaviour)
            self._entry(email_domain="asu.edu"),
        )
        assert email == "qqing@asu.edu"

    def test_lowercases_the_result(self):
        from molbuilder.web.auth_providers.cas import _extract_email
        email, _ = _extract_email(
            "QQing",
            {"mail": "QQing@ASU.EDU"},
            self._entry(email_attribute="mail", email_domain="asu.edu"),
        )
        assert email == "qqing@asu.edu"

    def test_no_attribute_no_domain_yields_no_email(self):
        """Schema validation should make this unreachable, but pin the
        defensive behaviour anyway."""
        from molbuilder.web.auth_providers.cas import _extract_email
        email, denied = _extract_email(
            "qqing", {}, self._entry()
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
        from molbuilder.web.auth import _safe_next_target
        assert _safe_next_target(dangerous) == "/"
