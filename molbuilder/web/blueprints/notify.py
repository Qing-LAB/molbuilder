"""``POST /api/notify`` — the receiving end for a run's own reports.

A job on a cluster tells you how it is going by POSTing here
(`execution/run-reports.md`).  This module is the whole receiving end, and
it is deliberately the smallest thing that can be: it **appends one line to
a log and answers `{"ok": true}`**.

**Append-only is the security model, not a detail.**  `job-contracts.md`
gives the monitor its boundary — *"it observes and notifies. It never
decides, and never mutates the calculation."*  The same holds one hop out.
A message that arrives becomes a line in a file: it is not parsed into
application state, does not touch a project, and cannot start, stop or
alter a job.  So the worst a stolen token buys is noise in a log that one
tab reads.

**How it gets past the login gate, and what that does NOT mean.**
``api_notify`` is in `auth.py`'s ``_PUBLIC_ENDPOINTS``, which carries its
own warning: *"Adding to this set makes the endpoint public. It is a
decision, not a convenience."*  It means the SSO **session** check does not
apply — a monitor on a compute node cannot do a browser sign-in.  It does
not mean unauthenticated: the first thing :func:`api_notify` does is
compare a bearer token with :func:`hmac.compare_digest`.

**A bad token counts against the rate limiter, and a rejected session does
not.**  `auth.py`'s gate marks its own 401 as *not evidence*
(``g.molbuilder_auth_challenge``) because an expired session is an ordinary
visitor — counting it once locked a user out of their own site for an hour.
**That reasoning does not carry here.**  Nobody reaches this route by
accident, so a wrong token is somebody trying one.  This module never sets
that flag, which is what leaves its 401 counted.

**Absent beats refused.**  With no ``notify_tokens_file`` configured the
blueprint is never registered, so the path 404s like any other nonexistent
one — `access-control.md` § 8 rule 2, *"a capability that cannot be
exercised safely should not appear"*, and rule 1, *"the safe state is the
one you get by doing nothing."*
"""
from __future__ import annotations

import hmac
import json
import logging
import logging.handlers
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

bp = Blueprint("notify", __name__)

#: The body cap.  A run report is a handful of scalars; anything larger is
#: not one.  Enforced before parsing, so an oversized body is refused
#: without being read into memory as JSON first.
MAX_BODY_BYTES = 8 * 1024

#: What a report may carry.  A fixed set, because the log is rendered in a
#: browser and an open-ended blob is an open-ended rendering problem.
_FIELDS = ("event", "text", "state", "elapsed_s", "n_iters", "energy",
           "geom_step", "per_iter_s")

#: One rotating file per user, 1 MB × 5.  A cap is not optional on a file
#: fed from the internet: without one a leaked token fills the disk the
#: app itself runs on.  The limiter bounds the rate; this bounds the total.
LOG_BYTES = 1 * 1024 * 1024
LOG_KEEP = 5

#: A user id becomes a FILENAME, so it is constrained to what is safe as
#: one.  The ids come from the token file, which is the operator's own —
#: but a path separator in one would write outside the log directory, and
#: "the operator would not do that" is not a mechanism.
_SAFE_USER = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")

_loggers: Dict[str, logging.Logger] = {}


def log_root() -> Path:
    """Where reports are written.

    Beside the env installer's own logs (`envs/_cli.py`), because these are
    the same kind of thing: a record of what happened, not configuration.
    *(Both belong under `$XDG_STATE_HOME` by the letter of the XDG spec;
    moving them is its own change, not this one's to smuggle in.)*
    """
    return Path(os.path.expanduser("~/.molbuilder/logs/notify"))


def read_tokens(path: str) -> Dict[str, str]:
    """``{user: token}`` from the operator's 0600 file, or ``{}``.

    Absent, unreadable or malformed all give ``{}`` — and ``{}`` means the
    route accepts nothing, which is the safe reading.  A misconfiguration
    here removes a capability; it never grants one.
    """
    try:
        raw = Path(os.path.expanduser(path)).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        obj = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {str(k): str(v) for k, v in obj.items()
            if isinstance(v, str) and v}


def _resolve_user(presented: str, tokens: Dict[str, str]) -> Optional[str]:
    """Which user owns ``presented``, or ``None``.

    Compared against every entry with :func:`hmac.compare_digest` and
    **without an early exit** — returning on the first match would make the
    time taken depend on the token's position in the file.  The loop is
    short and the cost is nothing; the property is worth keeping by
    construction rather than by argument.

    The sender never states who it is: **the secret is the claim.**  That is
    what stops a valid token being used to write into somebody else's
    record.
    """
    found = None
    for user, token in tokens.items():
        if hmac.compare_digest(presented, token):
            found = user
    return found


def _bearer(header: Optional[str]) -> str:
    if not header:
        return ""
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


def _logger_for(user: str) -> logging.Logger:
    """A rotating logger per user, made once and reused."""
    lg = _loggers.get(user)
    if lg is not None:
        return lg
    root = log_root()
    root.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"molbuilder.notify.{user}")
    lg.propagate = False          # these are records, not app logs
    lg.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        root / f"{user}.jsonl", maxBytes=LOG_BYTES, backupCount=LOG_KEEP,
        encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(handler)
    _loggers[user] = lg
    return lg


def _clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The declared fields, as scalars, and nothing else.

    Everything here is attacker-controlled text once a token is known, and
    it ends up on a page.  Strings are length-capped and nested structures
    are dropped rather than flattened — a report has no use for them, and
    accepting one means deciding later how to render it.
    """
    out: Dict[str, Any] = {}
    for k in _FIELDS:
        v = payload.get(k)
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:500]
    return out


@bp.route("/api/notify", methods=["POST"])
def api_notify():
    """Accept one run report.  Append it.  Say ``ok``."""
    from flask import current_app

    tokens = read_tokens(current_app.config["MB_NOTIFY_TOKENS_FILE"])
    user = _resolve_user(_bearer(request.headers.get("Authorization")), tokens)
    if user is None or not _SAFE_USER.match(user):
        # NOT marked as an auth challenge: see the module docstring.  A
        # wrong token is a probe and the limiter should hear about it.
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    body = request.get_data(cache=False)
    if len(body) > MAX_BODY_BYTES:
        return jsonify({"ok": False, "error": "too large"}), 413
    try:
        payload = json.loads(body.decode("utf-8", "replace") or "{}")
    except ValueError:
        return jsonify({"ok": False, "error": "not JSON"}), 400
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "not an object"}), 400

    record = _clean(payload)
    record["received_at"] = time.time()
    try:
        _logger_for(user).info(json.dumps(record))
    except OSError:
        # The log is the point, so a failure to write is worth an honest
        # 500 rather than an "ok" for a message nobody kept.
        return jsonify({"ok": False, "error": "could not record"}), 500

    # Nothing about the payload comes back, and nothing stored is readable
    # through this route.  Reading is a logged-in browser's job.
    return jsonify({"ok": True})
