"""Apereo CAS protocol backend (versions 1, 2, 3).

CAS is older and simpler than OAuth: the server returns a ticket
string in a URL query parameter; the app POSTs that ticket back to
the CAS server to validate it; the server replies with the
authenticated principal (a username, e.g. an ASURITE) and an
optional set of attributes.  No tokens, no JWTs, no scopes.

Because CAS rarely releases an email attribute (institutional CASes
typically return only the principal), molbuilder builds the
allowlist-matching email from one of two paths:

  1. ``email_attribute`` -- the name of a CAS attribute carrying the
     email (e.g., ``"mail"`` on many AD-backed CAS deployments).
     Tried first when set.
  2. ``email_domain`` -- synthesises ``{principal}@{email_domain}``
     when no attribute is set or the attribute didn't come through.
     This is what ASU CAS users need: it returns only the ASURITE
     (e.g., ``"qqing"``), paired with ``email_domain="asu.edu"`` to
     yield ``qqing@asu.edu`` for the allowlist match.

At least one of the two must be set (enforced at config-validation
time in ``runtime_config._validate_cas``).
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

from flask import url_for, request, redirect


def _cas_client(entry: Mapping):
    """Build a python-cas client for this provider entry.

    Imports python-cas lazily so the dependency is only required
    when a CAS provider is actually configured.

    CAS service_url MUST match between the start_flow and finish_flow
    requests -- the CAS server validates the ticket against the same
    URL it was originally handed.  When ``service_url`` is omitted
    from the config we derive it from ``url_for(_external=True)``,
    which depends on (a) Flask's view of the request's scheme/host
    and (b) molbuilder's ProxyFix settings.  In a stable single-host
    deployment (direct TLS, or one proxy with consistent
    ``X-Forwarded-Host``), the derived value is identical across both
    requests and ticket validation works.  In any deployment where
    those two requests can take different network paths and the
    derived URL might differ by even one byte (port, trailing slash,
    host capitalization), set ``service_url`` EXPLICITLY in the
    provider entry to lock it -- mismatches show up as a 401 with
    "INVALID_SERVICE" from the CAS server.
    """
    try:
        from cas import CASClient
    except ImportError:
        raise ImportError(
            "molbuilder auth: 'python-cas' is required for the CAS "
            "backend.  Install with `pip install python-cas` or "
            "`conda install -c conda-forge python-cas`."
        ) from None

    service_url = (
        entry.get("service_url")
        or url_for("auth_cas_callback",
                    provider_id=entry["id"],
                    _external=True)
    )

    kwargs = {
        "version":     entry.get("version", 3),
        "server_url":  _server_root_from_login_url(entry["login_url"]),
        "service_url": service_url,
    }
    if entry.get("ca_certs"):
        kwargs["verify_ssl_certificate"] = entry["ca_certs"]
    return CASClient(**kwargs)


def _server_root_from_login_url(login_url: str) -> str:
    """Derive the CAS server root URL from its login endpoint.

    python-cas wants ``server_url`` to be the CAS *root* (e.g.,
    ``https://weblogin.asu.edu/cas/``); it appends ``login`` /
    ``serviceValidate`` / etc. on its own.  Operators copy these
    URLs from CAS admin docs as full paths, so we strip the trailing
    ``login`` (with or without trailing slash) to get the root.
    """
    url = login_url.rstrip("/")
    if url.endswith("/login"):
        url = url[:-len("/login")]
    # Always end with a trailing slash -- python-cas concatenates.
    return url + "/"


# --------------------------------------------------------------------- #
#  Public entry points (called by auth_providers/__init__.py)           #
# --------------------------------------------------------------------- #


def start_flow(app, entry: Mapping):
    """Redirect to the CAS server's login URL."""
    client = _cas_client(entry)
    return redirect(client.get_login_url())


def finish_flow(app, entry: Mapping):
    """Validate the ticket sent by CAS, extract the identity, hand
    off to ``auth.authenticate``."""
    from ..auth import authenticate  # local import: avoid circular at import

    ticket = request.args.get("ticket")
    if not ticket:
        # User landed on the callback without a ticket (shouldn't
        # happen in a clean flow; covers refresh-the-callback-URL
        # cases gracefully).
        return redirect(
            url_for("auth_login_provider", provider_id=entry["id"])
        )

    client = _cas_client(entry)
    try:
        user, attributes, _pgtiou = client.verify_ticket(ticket)
    except Exception as exc:
        import logging
        logging.exception(
            "molbuilder auth: CAS ticket validation failed for provider %r",
            entry["id"],
        )
        return (
            f"CAS callback failed for provider {entry['id']!r}.  "
            f"Check the molbuilder server log for the full traceback "
            f"({type(exc).__name__}).",
            401,
        )

    if not user:
        return (
            f"CAS ticket validation rejected by "
            f"{entry['login_url']!r}.",
            401,
        )

    email, denied_reason = _extract_email(user, attributes or {}, entry)
    if denied_reason is not None:
        return denied_reason, 401
    if not email:
        return (
            f"CAS provider {entry['id']!r} could not produce an "
            f"email for principal {user!r} (neither email_attribute "
            f"nor email_domain yielded one).",
            401,
        )

    profile = {
        "name":    user,
        "picture": None,
        "sub":     user,                # CAS principal as stable id
        "cas_attributes": attributes or {},
    }
    return authenticate(entry["id"], email, profile)


def _extract_email(principal: str,
                    attributes: Mapping,
                    entry: Mapping
                    ) -> Tuple[Optional[str], Optional[str]]:
    """Apply the (attribute -> domain) fallback chain documented at
    module top.  Returns ``(email, denied_reason)``."""
    # Try the explicit attribute first when configured.  CAS
    # attributes can carry either a single string or a list of
    # strings -- handle both.
    attr_name = entry.get("email_attribute")
    if attr_name:
        raw = attributes.get(attr_name)
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, str) and raw:
            return raw.lower(), None

    # Fall back to synthesising from email_domain.
    domain = entry.get("email_domain")
    if domain:
        return f"{principal}@{domain}".lower(), None

    # Neither path produced one -- but the schema layer should have
    # made this unreachable.
    return None, None
