"""Who may do the things only an operator should do.

MODULE: admin identity  (web/admin.py).  Contract: docs/ops/access-control.md § 5.
  Callers: web/app.py (installs it; gates the reload route),
           web/rate_limit.py (the block-list routes).

ONE LIST, ONE MEANING.  Two subsystems ask this question -- who may read and
clear the rate limiter's block list, and who may restart the server everyone is
using -- and they get the same answer:

    "admin": { "emails": ["operator@asu.edu"] }      in molbuilder.json

**Absent or empty means NOBODY.**  That is the shape, not an accident: the state
you get by writing no config is the safe one, and a misconfiguration takes a
capability away rather than handing it to everybody.

WHY IT IS NOT ``rate_limit.admin_emails`` ANY MORE (2026-08-03).  It lived
inside the limiter's settings, where an empty list meant "any signed-in user".
That is defensible for reading a block list and wrong for stopping a shared
process, so the restart route had to INVERT it for itself -- one value, two
opposite readings, depending on which subsystem asked.  And it was reached
through the limiter's own object, so turning the limiter off silently changed
who counted as an admin: a connection nothing in the names would suggest.

WHAT AN EMPTY LIST COSTS ON A LAPTOP: nothing.  Loopback is never rate-limited,
so there is no block list to clear, and the restart button needs a supervisor
before it exists at all.
"""
from __future__ import annotations

from typing import Iterable

from flask import current_app, session


#: Where the resolved set hangs on the app.  Per-app, not module-global: tests
#: build several apps in one process, and a module-global would leak one app's
#: admins into the next one's requests.
_EXT_KEY = "molbuilder_admin_emails"


def install_admins(app, emails: Iterable[str]) -> frozenset:
    """Resolve the admin set once, at app setup, and hang it on the app."""
    resolved = frozenset(
        e.strip().lower() for e in (emails or ())
        if isinstance(e, str) and e.strip()
    )
    app.extensions[_EXT_KEY] = resolved
    return resolved


def named_admins(app=None) -> frozenset:
    """The configured set, or empty when nobody was named.

    Empty is the honest answer for "no `admin` section", "an empty list", and
    "this app never installed one" alike -- all three mean nobody is an admin.
    """
    target = app if app is not None else current_app
    try:
        return target.extensions.get(_EXT_KEY) or frozenset()
    except Exception:  # noqa: BLE001 -- outside an app context
        return frozenset()


def is_admin_request() -> bool:
    """True iff the session making this request belongs to a named admin.

    No session, no email, or an email nobody named -- all not an admin.  There
    is no "empty means everybody" branch, deliberately: that was the reading
    that made one value mean two things.

    The auth layer stores ``session["user"] = {"email": …}`` lowercased
    (auth.py::authenticate), so membership is case-stable.
    """
    admins = named_admins()
    if not admins:
        return False
    try:
        user = session.get("user") or {}
    except Exception:  # noqa: BLE001
        # Outside a request context, or no session backend (no secret key).
        # Both mean "we cannot tell who this is", which is not an admin.
        return False
    if not isinstance(user, dict):
        return False
    email = (user.get("email") or "").strip().lower()
    return bool(email) and email in admins
