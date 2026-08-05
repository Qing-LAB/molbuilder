"""Who may do the things only an operator should do.

MODULE: admin identity  (web/admin.py).  Contract: docs/ops/access-control.md § 5.
  Callers: web/app.py (installs it; gates the reload route),
           web/rate_limit.py (the block-list routes).

ONE LIST, ONE MEANING.  Two subsystems ask this question -- who may read and
clear the rate limiter's block list, and who may restart the server everyone is
using -- and they get the same answer.

**BY DEFAULT, ANYONE WHO CAN SIGN IN AT ALL.**  Not because the door is open, but
because it was already locked upstream: ``auth.providers[].allowed_users`` is a
REQUIRED field (runtime_config.py), so nobody reaches a session without being
named there.  A second list repeating those same names would be two lists to keep
in step for one question -- and on a single-operator server it is the SAME name,
written twice, with the button silently missing until you notice you owe the file
a second copy of yourself.

    "admin": { "emails": ["operator@asu.edu"] }      in molbuilder.json

Naming anyone here NARROWS it: with the section present, only those addresses are
admins, and everyone else who can sign in is not.  That is the setting to reach
for on a shared deployment where signing in and operating the process are
different privileges.

WHAT MAKES THIS SAFE is the required allow-list, not a second list here.  There is
no configuration in which "anyone who can sign in" means the public: an operator
who writes no auth config has no login at all, and one who writes it has had to
name every person by hand.

WHY IT IS NOT ``rate_limit.admin_emails`` ANY MORE (2026-08-03).  It lived
inside the limiter's settings, and the restart route had to INVERT its meaning
for itself -- one value, two opposite readings, depending on which subsystem
asked.  And it was reached through the limiter's own object, so turning the
limiter off silently changed who counted as an admin.  BOTH of those are what
was wrong; the default itself was not, and it is restored here with one meaning
in one place.

WHAT THIS COSTS ON A LAPTOP: nothing.  Loopback is never rate-limited, so there
is no block list to clear, and the restart button needs a supervisor before it
exists at all.
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
    """True iff the session making this request may operate the server.

    ANONYMOUS IS NEVER AN ADMIN.  No session, no email: not an admin, whatever
    the config says.  That is the one rule with no exception.

    NAMED NOBODY MEANS ANYONE WHO SIGNED IN.  Reaching a session at all required
    being listed in a provider's ``allowed_users``, which is a required field --
    so this is not an open door, it is the door that was already locked upstream.
    Asking an operator to write their own address a second time, in a different
    section, to get a button on their own server is bookkeeping rather than
    safety.

    NAMED SOMEBODY NARROWS IT to those addresses.  On a deployment where signing
    in and operating the process are different privileges, that is the setting.

    The auth layer stores ``session["user"] = {"email": …}`` lowercased
    (auth.py::authenticate), so membership is case-stable.
    """
    try:
        user = session.get("user") or {}
    except Exception:  # noqa: BLE001
        # Outside a request context, or no session backend (no secret key).
        # Both mean "we cannot tell who this is", which is not an admin.
        return False
    if not isinstance(user, dict):
        return False
    email = (user.get("email") or "").strip().lower()
    if not email:
        return False
    admins = named_admins()
    return (email in admins) if admins else True
