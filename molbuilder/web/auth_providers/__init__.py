"""Authentication backend implementations.

Each backend "kind" (google, github, microsoft, orcid, cas) has its
own module under this package.  ``auth.py`` calls :func:`start_flow`
and :func:`finish_flow` here; this module dispatches to the
appropriate backend.

The runtime contract between ``auth.py`` and a backend module:

* ``start_flow(app, entry)`` -- called from ``/login/<provider_id>``.
  Returns a Flask response that redirects the browser to the
  identity provider.  May raise ``ImportError`` if the backend's
  third-party dependency (e.g., authlib for OAuth, python-cas for
  CAS) isn't installed -- the caller surfaces the install hint.

* ``finish_flow(app, entry)`` -- called from the provider's callback
  route (``/oauth-callback/<provider_id>`` or
  ``/cas-callback/<provider_id>``).  Returns a Flask response that
  either redirects to the originally-requested URL (on successful
  sign-in) or renders an error page (on failure / not in allowlist).
  Concretely: on success the backend extracts an email + minimal
  profile dict, then calls ``auth.authenticate(...)`` which sets the
  session cookie and returns the post-login redirect.

Adding a new kind = drop a sibling module here with the same two
entry points, and add the kind to ``runtime_config._SUPPORTED_KINDS``
+ ``_KIND_VALIDATORS`` so the JSON schema accepts it.
"""
from __future__ import annotations

from typing import Mapping


_OAUTH_KINDS = frozenset({"google", "github", "microsoft", "orcid"})


def start_flow(app, entry: Mapping):
    """Dispatch /login/<provider_id> to the right backend."""
    kind = entry["kind"]
    if kind in _OAUTH_KINDS:
        from . import oauth
        return oauth.start_flow(app, entry)
    if kind == "cas":
        from . import cas
        return cas.start_flow(app, entry)
    # runtime_config schema validation should make this unreachable.
    raise ValueError(f"auth_providers: unknown kind {kind!r}")


def finish_oauth(app, entry: Mapping):
    """Dispatch /oauth-callback/<provider_id> to the OAuth backend."""
    if entry["kind"] not in _OAUTH_KINDS:
        # Route mismatch: a CAS provider should never land here.
        from flask import abort
        abort(404)
    from . import oauth
    return oauth.finish_flow(app, entry)


def finish_cas(app, entry: Mapping):
    """Dispatch /cas-callback/<provider_id> to the CAS backend."""
    if entry["kind"] != "cas":
        from flask import abort
        abort(404)
    from . import cas
    return cas.finish_flow(app, entry)
