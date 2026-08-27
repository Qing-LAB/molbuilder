"""OAuth 2.0 / OIDC backends -- Google, GitHub, Microsoft, ORCID.

All four providers share the same Authlib-based plumbing: register a
client with discovery metadata, kick off ``authorize_redirect``, then
exchange the auth code in the callback for an access token + userinfo.
The differences are per-provider config (scope strings, where the
email lives in the userinfo response, additional IdP-side filters
like ``hosted_domain`` for Google or ``allowed_organizations`` for
GitHub) and the OIDC discovery URL (when applicable).

We keep all four in one module rather than splitting per-kind because
the per-kind code is small (~20 lines each) and centralizing the
Authlib registration ergonomics (one ``OAuth()`` instance per app,
deferred import, secret-file reading) avoids duplicating that boilerplate.

The CAS protocol is a different beast (it's not OAuth) and lives in
``cas.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Tuple

from flask import url_for, request


# --------------------------------------------------------------------- #
#  Per-process OAuth() instance + client registry                       #
# --------------------------------------------------------------------- #
#
# Authlib's flask_client.OAuth expects to be wired to a specific Flask
# app once.  We attach it to the app via app.extensions so multiple
# apps in the same process (the test harness creates fresh apps per
# test) get fresh OAuth instances and don't bleed registrations across
# each other.


_OAUTH_CLIENTS_EXT_KEY = "molbuilder_oauth_clients"


# Authlib's ``OAuth`` instance exposes attributes named after each
# registered client via ``__getattr__`` — so if we passed entry["id"]
# verbatim as the registration ``name``, an id of "register" / "cache" /
# "init_app" / etc. would collide with the OAuth object's own methods
# and either shadow the method or return the wrong object on the
# second login attempt.  Prefix internally with ``mb_`` to keep the
# namespaces disjoint regardless of which ids operators pick.
_AUTHLIB_INTERNAL_PREFIX = "mb_"


def _authlib_name(provider_id: str) -> str:
    return f"{_AUTHLIB_INTERNAL_PREFIX}{provider_id}"


def _get_oauth(app):
    """Return (or create) the Authlib OAuth() instance bound to ``app``."""
    try:
        from authlib.integrations.flask_client import OAuth
    except ImportError:
        raise ImportError(
            "molbuilder auth: 'Authlib' is required for OAuth backends "
            "(google / github / microsoft / orcid).  Install with "
            "`pip install molbuilder[auth]` or `pip install authlib`."
        ) from None
    if _OAUTH_CLIENTS_EXT_KEY not in app.extensions:
        app.extensions[_OAUTH_CLIENTS_EXT_KEY] = {
            "oauth":      OAuth(app),
            "registered": set(),    # authlib-internal names already registered
            # authlib-internal name -> mtime of client_secret_file at the
            # last successful read.  ``None`` for literal-secret providers
            # (no file to watch).  Used by _ensure_client to detect
            # operator-initiated secret rotations and hot-reload them
            # without bouncing the server.
            "secret_mtime": {},
        }
    return app.extensions[_OAUTH_CLIENTS_EXT_KEY]


def _secret_file_mtime(entry: Mapping):
    """Return ``st_mtime`` of the provider's secret file, or ``None`` if
    the entry doesn't have one (literal ``client_secret``) or the file
    can't be stat'd (deleted between reads, permission denied, etc.).
    Defensively returns ``None`` rather than raising -- the caller
    treats ``None`` as "no change to detect", which fails closed
    (keeps the previously-loaded secret in memory)."""
    path_spec = entry.get("client_secret_file")
    if not path_spec:
        return None
    try:
        return os.stat(os.path.expanduser(path_spec)).st_mtime
    except OSError:
        return None


def _read_secret(entry: Mapping) -> str:
    """Read the OAuth client secret -- file path or literal string.

    Both shapes get an explicit empty-content check: an operator who
    accidentally ``echo '' > ~/.config/molbuilder/<provider>_client_secret``
    or leaves the literal string blank would otherwise see a confusing
    OAuth-error-at-first-login instead of a clear config diagnostic.
    """
    if "client_secret_file" in entry and entry["client_secret_file"]:
        path = Path(os.path.expanduser(entry["client_secret_file"]))
        secret = path.read_text().strip()
        if not secret:
            raise RuntimeError(
                f"auth.providers[id={entry['id']!r}]: "
                f"client_secret_file {str(path)!r} exists but is "
                f"empty (or contains only whitespace).  Write the "
                f"provider's client secret into that file and re-"
                f"start molbuilder."
            )
        return secret
    if "client_secret" in entry and entry["client_secret"]:
        return entry["client_secret"]
    raise RuntimeError(
        f"auth.providers[id={entry['id']!r}]: neither client_secret "
        f"nor client_secret_file is set (schema validation should "
        f"have caught this -- check runtime_config._validate_secret_pair)."
    )


def _ensure_client(app, entry: Mapping):
    """Register the Authlib client for this provider if not already.

    Returns the client object for use by ``authorize_redirect`` /
    ``authorize_access_token``.  Names are mangled with the
    ``mb_`` prefix so that operator-chosen ids (``"register"``,
    ``"cache"``, ...) can never shadow Authlib's own attributes.

    Hot-reload of ``client_secret_file`` (added 2026-05-18, task #100):
    on every call AFTER the first registration, this function stats
    the secret file's mtime and compares against the value cached at
    last read.  When the mtime differs, it re-reads the file and
    assigns ``client.client_secret`` in place so the next token
    exchange uses the new value -- the operator can rotate the
    ``GOCSPX-`` secret WITHOUT bouncing molbuilder.  If the re-read
    fails (file deleted, empty, permission denied), the helper logs
    and keeps the previously-loaded secret so an unrelated edit
    doesn't break sign-in mid-session.
    """
    import logging

    ext = _get_oauth(app)
    oauth      = ext["oauth"]
    registered = ext["registered"]
    mtimes     = ext["secret_mtime"]

    authlib_name = _authlib_name(entry["id"])

    if authlib_name in registered:
        client = getattr(oauth, authlib_name)
        current_mtime = _secret_file_mtime(entry)
        cached_mtime  = mtimes.get(authlib_name)
        # current_mtime is None for literal-client_secret entries
        # (no file to watch) or when the file is gone -- both cases
        # mean "no change we can detect"; the cached secret stands.
        if current_mtime is not None and current_mtime != cached_mtime:
            try:
                new_secret = _read_secret(entry)
            except Exception as exc:
                logging.warning(
                    "molbuilder auth: client_secret_file for provider "
                    "%r changed but re-read failed (%s: %s); keeping "
                    "the previously-loaded secret in memory.",
                    entry["id"], type(exc).__name__, exc,
                )
            else:
                # In-place update on the cached authlib client.
                # Mutating ``client_secret`` is safe: authlib reads
                # it at token-exchange time, not at register time.
                client.client_secret = new_secret
                mtimes[authlib_name] = current_mtime
                logging.info(
                    "molbuilder auth: client_secret_file for provider "
                    "%r changed on disk; re-read + updated the "
                    "authlib client without server restart.",
                    entry["id"],
                )
        return client

    kind = entry["kind"]
    register_kwargs = _client_kwargs_for_kind(kind, entry)
    oauth.register(
        name=authlib_name,
        client_id=entry["client_id"],
        client_secret=_read_secret(entry),
        **register_kwargs,
    )
    registered.add(authlib_name)
    mtimes[authlib_name] = _secret_file_mtime(entry)
    return getattr(oauth, authlib_name)


def _client_kwargs_for_kind(kind: str, entry: Mapping) -> dict:
    """Per-kind Authlib registration kwargs.

    Discovery URL (``server_metadata_url``) is preferred over hand-
    coded endpoint URLs whenever the provider publishes OIDC discovery
    -- Authlib then fetches the canonical token/authorize/userinfo
    endpoints from the well-known document and cached after first hit.
    """
    if kind == "google":
        return {
            "server_metadata_url":
                "https://accounts.google.com/.well-known/openid-configuration",
            "client_kwargs": {"scope": "openid email profile"},
        }
    if kind == "microsoft":
        # Tenant-aware metadata URL.  "common" = any Microsoft account
        # (work, school, personal); a GUID or verified domain restricts
        # to one tenant.
        tenant = entry.get("tenant_id", "common")
        return {
            "server_metadata_url":
                f"https://login.microsoftonline.com/{tenant}/v2.0/"
                f".well-known/openid-configuration",
            "client_kwargs": {"scope": "openid email profile"},
        }
    if kind == "orcid":
        return {
            "server_metadata_url":
                "https://orcid.org/.well-known/openid-configuration",
            "client_kwargs": {"scope": "openid"},
        }
    if kind == "github":
        # GitHub does NOT implement OIDC discovery; endpoints are
        # hardcoded.  ``user:email`` scope is required to read the
        # primary email when the user hides it from their public
        # profile; ``read:org`` is required for the org-membership
        # check (allowed_organizations).
        return {
            "api_base_url":     "https://api.github.com/",
            "access_token_url": "https://github.com/login/oauth/access_token",
            "authorize_url":    "https://github.com/login/oauth/authorize",
            "client_kwargs":    {"scope": "user:email read:org"},
        }
    raise ValueError(f"_client_kwargs_for_kind: unsupported kind {kind!r}")


# --------------------------------------------------------------------- #
#  Public entry points (called by auth_providers/__init__.py)           #
# --------------------------------------------------------------------- #


def start_flow(app, entry: Mapping):
    """Kick off the OAuth redirect to the provider's authorize URL."""
    client = _ensure_client(app, entry)
    redirect_uri = url_for(
        "auth_oauth_callback",
        provider_id=entry["id"],
        _external=True,
    )
    return client.authorize_redirect(redirect_uri)


def finish_flow(app, entry: Mapping):
    """Handle the provider's callback: exchange code, extract email,
    apply IdP-side filters, hand off to ``auth.authenticate``."""
    import logging

    from ..auth import authenticate  # local import: avoid circular at import time
    client = _ensure_client(app, entry)
    try:
        token = client.authorize_access_token()
    except Exception as exc:
        # Log the full exception server-side -- some authlib /
        # provider error paths surface a ``params`` dict that echoes
        # back request parameters, which on certain misbehaving
        # providers can include the client_secret.  Don't risk
        # leaking that into the user-visible response; show a
        # generic message + tell the operator to check the logs.
        logging.exception(
            "molbuilder auth: OAuth callback failed for provider %r",
            entry["id"],
        )
        return (
            f"OAuth callback failed for provider {entry['id']!r}.  "
            f"Check the molbuilder server log for the full traceback "
            f"({type(exc).__name__}).",
            401,
        )

    email, profile, denied_reason = _extract_identity(client, token, entry)
    if denied_reason is not None:
        return denied_reason, 403
    if not email:
        return (
            f"OAuth callback for provider {entry['id']!r}: provider "
            f"did not return a usable email address.",
            401,
        )
    return authenticate(entry["id"], email, profile)


# --------------------------------------------------------------------- #
#  Per-kind identity extraction                                         #
# --------------------------------------------------------------------- #


def _extract_identity(client, token, entry: Mapping
                       ) -> Tuple[Optional[str], dict, Optional[str]]:
    """Return ``(email, profile, denied_reason)``.

    ``denied_reason`` is non-None when an IdP-side filter rejects the
    user (Google ``hosted_domain``, GitHub ``allowed_organizations``);
    in that case the caller emits a 403 with that message.  ``email``
    can be None when the provider didn't return one (rare for OIDC;
    handled by the caller as a 401).
    """
    kind = entry["kind"]
    handler = _IDENTITY_HANDLERS[kind]
    return handler(client, token, entry)


def _identity_google(client, token, entry: Mapping):
    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").lower() or None
    if not userinfo.get("email_verified"):
        return None, {}, (
            "Google did not return a verified email address."
        )
    hd_required = [d.lower() for d in (entry.get("hosted_domain") or [])]
    if hd_required:
        hd_actual = (userinfo.get("hd") or "").lower()
        if hd_actual not in hd_required:
            # Personal @gmail.com accounts (and other non-Workspace
            # Google accounts) have NO ``hd`` claim at all; surface
            # that distinctly so users don't read "hosted domain None
            # is not allowed" and wonder what to fix.
            if not hd_actual:
                reason = (
                    "This molbuilder deployment only allows Google "
                    "Workspace accounts (the allowed hosted_domain "
                    f"list is {hd_required}).  Your Google account is "
                    f"a personal account, not a Workspace one -- "
                    f"sign in with your @-organisation account instead."
                )
            else:
                reason = (
                    f"Your Google account's hosted domain "
                    f"{hd_actual!r} is not in this deployment's "
                    f"allowed list ({hd_required})."
                )
            return None, {}, reason
    return email, _profile_from_oidc(userinfo), None


def _identity_microsoft(client, token, entry: Mapping):
    userinfo = token.get("userinfo") or {}
    # Microsoft sometimes puts the email in ``email`` and sometimes
    # only in ``preferred_username`` (depending on tenant config).
    email = (userinfo.get("email")
              or userinfo.get("preferred_username")
              or "").lower() or None
    return email, _profile_from_oidc(userinfo), None


def _identity_orcid(client, token, entry: Mapping):
    userinfo = token.get("userinfo") or {}
    # ORCID's OIDC userinfo returns the ORCID iD as ``sub`` (e.g.,
    # "0000-0001-2345-6789").  Email is OPTIONAL and only present if
    # the user marked it public in their ORCID profile.  When absent,
    # we use the ORCID iD itself as the identity -- allowed_users
    # should contain ORCID iDs in that case, not email addresses.
    email = (userinfo.get("email") or "").lower() or None
    if not email:
        sub = userinfo.get("sub")
        if sub:
            email = sub.lower()  # ORCID iDs are case-insensitive anyway
    profile = _profile_from_oidc(userinfo)
    if userinfo.get("sub"):
        profile["orcid_id"] = userinfo["sub"]
    return email, profile, None


def _identity_github(client, token, entry: Mapping):
    # GitHub does not implement OIDC; we call REST endpoints manually.
    user_resp = client.get("user", token=token)
    user = user_resp.json() if user_resp.ok else {}

    email = (user.get("email") or "").lower() or None
    if not email:
        # User has email private on their profile -- fetch from the
        # authenticated /user/emails endpoint (requires user:email scope,
        # which we request unconditionally).
        emails_resp = client.get("user/emails", token=token)
        if emails_resp.ok:
            for e in emails_resp.json():
                if e.get("primary") and e.get("verified"):
                    email = e.get("email", "").lower() or None
                    break
        else:
            # /user/emails refused -- most likely the OAuth app was
            # registered before molbuilder requested ``user:email``,
            # so the existing token's grant doesn't include it.  Tell
            # the user how to fix it instead of leaving them at a
            # generic "no email" 401.
            return None, {}, (
                f"GitHub refused to release your email (HTTP "
                f"{emails_resp.status_code}).  This usually means the "
                f"OAuth app's existing grant is missing the "
                f"``user:email`` scope.  Revoke the molbuilder grant "
                f"at https://github.com/settings/applications and try "
                f"signing in again -- you'll be prompted to approve "
                f"the new scope."
            )

    allowed_orgs = [o.lower() for o in (entry.get("allowed_organizations") or [])]
    if allowed_orgs:
        orgs_resp = client.get("user/orgs", token=token)
        if not orgs_resp.ok:
            return None, {}, (
                f"Could not verify GitHub organization membership "
                f"(HTTP {orgs_resp.status_code}); allowed_organizations "
                f"is set so access denied."
            )
        user_orgs = {o.get("login", "").lower() for o in orgs_resp.json()}
        if not user_orgs & set(allowed_orgs):
            return None, {}, (
                f"Your GitHub account is not a member of any allowed "
                f"organization ({sorted(allowed_orgs)})."
            )

    profile = {
        "name":    user.get("name") or user.get("login") or email,
        "picture": user.get("avatar_url"),
        "sub":     str(user.get("id")) if user.get("id") else None,
    }
    return email, profile, None


def _profile_from_oidc(userinfo: dict) -> dict:
    """Standard OIDC userinfo -> molbuilder profile shape."""
    return {
        "name":    userinfo.get("name") or userinfo.get("email"),
        "picture": userinfo.get("picture"),
        "sub":     userinfo.get("sub"),
    }


_IDENTITY_HANDLERS = {
    "google":    _identity_google,
    "microsoft": _identity_microsoft,
    "orcid":     _identity_orcid,
    "github":    _identity_github,
}
