"""IP-based rate limiting + scanner detection for the molbuilder UI.

Threat model
============

molbuilder runs as a Flask app on a host that may be reachable from
the public internet (typical deployment: a lab workstation at
``molbuilder.example.edu`` with TLS).  The auth layer protects
authenticated surfaces (CAS / OAuth) but the unauthenticated probe
surface remains:

* ``/login`` returns 200 to anyone (it's the login picker).
* ``/api/*`` returns 401 ``{ok:false}`` for unauthenticated requests
  but the response is cheap.  A scanner can probe at 5-10 req/s.
* Any URL that doesn't match a registered route returns 404.

Production logs from the active scanner attack (2026-06-18, IP
``100.27.42.242``, ~40 requests in 9 s) showed the canonical
shape: rapid 404 enumeration of ``.jsp`` / ``.php`` / ``.cfc`` /
``.nsf`` / ``.dll`` extensions with ``<script>document.cookie=...``
and ``<meta http-equiv=Set-Cookie content=...>`` reflection probes
in the query string.  This module catches that pattern.

Three signals
=============

Any one signal flips the IP onto the blocklist with TTL = cooldown.
Subsequent requests from the IP are dropped with empty 429 +
``Connection: close`` before any handler runs.

1. **Signature match** — the request path or query string contains
   a string like ``<script``, ``<meta http-equiv``, ``union select``,
   ``;drop table``, etc.  These are unambiguous attacker
   fingerprints; no legitimate molbuilder URL contains them.
   Immediate block.

2. **404-storm** — N+ HTTP-4xx responses from this IP within
   ``window_404_s`` seconds.  Catches path enumeration where the
   scanner probes extensions / admin paths / CMS routes.

3. **Total-burst** — M+ requests from this IP within
   ``window_total_s`` seconds, regardless of response status.
   Catches slower scanners that don't trip the 404 storm signal
   (e.g., a scanner that finds a real 200 and starts hammering).

Defaults
========

* signature match              → immediate block
* >= 20 4xx responses in 30 s  → block
* >= 60 total requests in 60 s → block
* cooldown                     → 3600 s (1 hour)
* allowlist                    → empty (no IP is exempt by default)
* trust_proxy                  → False (use ``request.remote_addr``)

Multi-worker caveat
===================

State lives in-process.  In a single-worker deployment (the default
``molbuilder serve``), the thresholds work as advertised.  Under
``gunicorn -w N`` the same IP could hit each worker separately, so
the effective threshold becomes ``N * configured``.  The honest
mitigation is a shared store (Redis); the honest tradeoff is
"acceptable for typical molbuilder deployments, document the gap."
For full-strength protection across workers, add an nginx
``limit_req`` rule upstream.

Admin
=====

* ``GET /api/admin/rate_limit/status`` (admin auth) — list blocked
  IPs + TTL remaining + reason.
* ``POST /api/admin/rate_limit/clear`` body ``{"ip": "..."}`` —
  unblock a specific IP.  Body ``{"all": true}`` — wipe everything
  (use sparingly).

Persistence
===========

NONE.  Blocklist clears on restart.  Restart is itself an effective
mitigation (drops the attacker's connections); persistence would
mostly serve incident-response needs that are out of scope for this
PR.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from flask import Flask, Response, g, jsonify, request, session

logger = logging.getLogger(__name__)


# Default tunables — overridable per-deployment via cfg["rate_limit"].
#
# DESIGN NOTE (2026-06-18 hotfix):
#
#   ``threshold_total`` is DISABLED by default (set to 0).  The
#   original 60/60s ceiling tripped on every legitimate 1 Hz
#   poll from the system-load monitor (#472) — within a minute
#   the very user-driven traffic the limiter is meant to protect
#   gets killed.  Successful 200s should not count toward an
#   abuse signal; the 404-storm signal already catches the
#   canonical scanner pattern (rapid path enumeration generates
#   4xx, not 2xx).
#
#   Operators who want total-burst back on for a paranoid
#   deployment set ``threshold_total`` to a non-zero value in
#   ``molbuilder.json``.  Recommended floor: 600 (= 10/s
#   sustained for a minute), well above any legitimate poll
#   cadence.
#
#   ``127.0.0.1`` / ``::1`` are baked into the default
#   allowlist.  When the bind guard refuses non-loopback HTTP
#   without TLS (cli.py::_refuse_remote_bind_without_tls),
#   localhost requests really are local — never the scanner.
#
DEFAULTS: Dict[str, Any] = {
    "enabled":         True,
    "window_404_s":    30,
    "threshold_404":   20,
    "window_total_s":  60,
    "threshold_total": 0,          # 0 = disabled; see DESIGN NOTE
    "cooldown_s":      3600,
    "trust_proxy":     False,
    "allowlist":       ["127.0.0.1", "::1"],
    "max_tracked_ips": 10_000,
    # NO `admin_emails` HERE.  Who may read and clear the block list is not a
    # rate-limiting setting -- it is the same question "who may restart the
    # server" asks, and one list answers both: the top-level `admin` section
    # (web/admin.py).  It lived here until 2026-08-03, where its empty default
    # meant "any signed-in user" to this subsystem and had to be inverted by
    # the other one.
}


# Scanner-signature patterns.  Match → immediate block.  Each
# pattern has been verified against the attacker log to fire on the
# real scanner; legitimate molbuilder URLs do NOT contain these
# strings (we don't ever embed angle-bracket HTML in URLs).
SIGNATURE_PATTERNS: List[re.Pattern] = [
    re.compile(r"<\s*script",              re.IGNORECASE),
    re.compile(r"<\s*meta\s+http-equiv",   re.IGNORECASE),
    re.compile(r"document\.cookie",        re.IGNORECASE),
    re.compile(r"\bunion\s+select",        re.IGNORECASE),
    re.compile(r";\s*drop\s+(table|database)", re.IGNORECASE),
    re.compile(r"/etc/passwd",             re.IGNORECASE),
    re.compile(r"\.\.[\\/]\.\.[\\/]\.\.",  re.IGNORECASE),  # ../../../
]


# Block reasons surfaced in logs + admin endpoint.
REASON_SIGNATURE  = "signature_match"
REASON_STORM_404  = "404_storm"
REASON_STORM_TOTAL = "total_burst"


@dataclass
class _IPState:
    """Per-IP rolling state.

    Each deque is bounded by its threshold; the oldest timestamp
    stays at index 0.  ``len(buf) == buf.maxlen`` AND
    ``now - buf[0] <= window`` is the trigger condition.
    """
    requests_buf:   Deque[float] = field(default_factory=deque)
    errors_4xx_buf: Deque[float] = field(default_factory=deque)
    blocked_until:  Optional[float] = None
    block_reason:   Optional[str] = None
    last_touched:   float = 0.0


class RateLimiter:
    """In-process IP rate-limit store.

    Thread-safe (single RLock around all mutations).  Bounded by
    ``max_tracked_ips`` to prevent IP-rotation from DoS'ing the
    limiter itself; LRU eviction on full.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        # Resolve config with defaults.  Unknown keys are ignored
        # (forward-compat: a future cfg may carry extra knobs).
        self.enabled         = bool(cfg.get("enabled",         DEFAULTS["enabled"]))
        self.window_404_s    = int (cfg.get("window_404_s",    DEFAULTS["window_404_s"]))
        self.threshold_404   = int (cfg.get("threshold_404",   DEFAULTS["threshold_404"]))
        self.window_total_s  = int (cfg.get("window_total_s",  DEFAULTS["window_total_s"]))
        self.threshold_total = int (cfg.get("threshold_total", DEFAULTS["threshold_total"]))
        self.cooldown_s      = int (cfg.get("cooldown_s",      DEFAULTS["cooldown_s"]))
        self.trust_proxy     = bool(cfg.get("trust_proxy",     DEFAULTS["trust_proxy"]))
        self.max_tracked_ips = int (cfg.get("max_tracked_ips", DEFAULTS["max_tracked_ips"]))
        self.allowlist_nets  = _parse_allowlist(
            cfg.get("allowlist", DEFAULTS["allowlist"]),
        )
        # Who may reach /api/admin/rate_limit/* is NOT held here: it is the
        # top-level `admin` section, read through web/admin.py, because the
        # same list answers "who may restart the server" (§ 5).
        # OrderedDict gives O(1) move-to-end + popitem(last=False)
        # for LRU eviction.
        self._states: "OrderedDict[str, _IPState]" = OrderedDict()
        self._lock = threading.RLock()

    # -- public API ------------------------------------------------------

    def client_ip(self, req=None) -> str:
        """Resolve the client IP for the current request.

        Defaults to ``request.remote_addr``.  When ``trust_proxy``
        is true, honours the leftmost entry of ``X-Forwarded-For``
        (the original client per RFC 7239); ignores the header
        when ``trust_proxy`` is false because an attacker can spoof
        it.
        """
        req = req or request
        if self.trust_proxy:
            xff = req.headers.get("X-Forwarded-For", "")
            if xff:
                return xff.split(",")[0].strip() or "unknown"
        return req.remote_addr or "unknown"

    def is_allowlisted(self, ip: str) -> bool:
        """True iff ``ip`` is in the configured allowlist."""
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in self.allowlist_nets)

    def check_request(
        self, ip: str, path_with_query: str,
        *, authenticated: bool = False,
    ) -> Optional[Tuple[str, int]]:
        """Run pre-handler signals against this request.

        Returns ``(reason, ttl_s)`` if the IP should be blocked
        (either already blocked, or just tripped a signal), else
        ``None``.  Caller (the before_request hook) returns an
        empty 429 response on a tuple result.

        ``authenticated`` short-circuits the limiter entirely.
        A logged-in principal has already passed the SSO gate
        (CAS / OAuth); the limiter's purpose is to deflect
        anonymous scanners, not to throttle real users.  When
        auth is disabled at the deployment level the caller
        always passes ``authenticated=False`` and the normal
        signal chain runs.
        """
        if not self.enabled or authenticated or self.is_allowlisted(ip):
            return None
        now = time.monotonic()
        with self._lock:
            st = self._touch(ip, now)

            # Already blocked?
            if st.blocked_until is not None:
                if now < st.blocked_until:
                    return (st.block_reason or "blocked",
                            int(st.blocked_until - now))
                # TTL expired — clear and continue.
                self._clear_block_locked(st)

            # Signature match — immediate block, before counting.
            if _matches_signature(path_with_query):
                self._block_locked(ip, st, REASON_SIGNATURE, now)
                return (REASON_SIGNATURE, self.cooldown_s)

            # Record the request.  When ``threshold_total`` is 0
            # the total-burst signal is disabled — skip the buffer
            # bookkeeping entirely.  This is the documented default
            # (see DEFAULTS docstring): successful 200s from a
            # legitimate poller should not accumulate toward an
            # abuse counter.
            if self.threshold_total > 0:
                _append_bounded(st.requests_buf, now, self.threshold_total)
                if _exceeds_window(st.requests_buf,
                                   self.threshold_total,
                                   self.window_total_s, now):
                    self._block_locked(ip, st, REASON_STORM_TOTAL, now)
                    return (REASON_STORM_TOTAL, self.cooldown_s)

        return None

    def record_response(
        self, ip: str, status_code: int,
        *, authenticated: bool = False,
    ) -> None:
        """Post-handler hook.  Tracks 4xx responses toward the
        404-storm signal.  Idempotent w.r.t. already-blocked IPs
        (they'll already get the connection-close path next time).

        Authenticated requests don't accumulate the 4xx counter
        either — same rationale as ``check_request``.  A real
        user mis-clicking through their session shouldn't grow
        toward a blocklist trip.
        """
        if not self.enabled or authenticated or self.is_allowlisted(ip):
            return
        if not (400 <= status_code < 500):
            return
        # threshold_404 <= 0 disables the 404-storm signal — same
        # opt-out shape as threshold_total.  The signature-match
        # signal stays on regardless because it is the canonical
        # "this is unambiguous abuse" detector.
        if self.threshold_404 <= 0:
            return
        now = time.monotonic()
        with self._lock:
            st = self._touch(ip, now)
            if st.blocked_until is not None:
                # Already blocked — no point growing the buffer.
                return
            _append_bounded(st.errors_4xx_buf, now, self.threshold_404)
            if _exceeds_window(st.errors_4xx_buf,
                               self.threshold_404,
                               self.window_404_s, now):
                self._block_locked(ip, st, REASON_STORM_404, now)

    def status(self) -> Dict[str, Any]:
        """Snapshot for the admin endpoint."""
        now = time.monotonic()
        with self._lock:
            blocked: List[Dict[str, Any]] = []
            for ip, st in self._states.items():
                if st.blocked_until is None:
                    continue
                ttl = int(st.blocked_until - now)
                if ttl <= 0:
                    continue
                blocked.append({
                    "ip":     ip,
                    "reason": st.block_reason,
                    "ttl_s":  ttl,
                })
            return {
                "enabled":        self.enabled,
                "blocked_count":  len(blocked),
                "blocked":        blocked,
                "tracked_count":  len(self._states),
            }

    def clear(self, ip: str) -> bool:
        """Unblock ``ip``.  Returns True if an entry existed."""
        with self._lock:
            st = self._states.get(ip)
            if st is None:
                return False
            self._clear_block_locked(st)
            return True

    def clear_all(self) -> int:
        """Drop every tracked IP.  Returns the count cleared."""
        with self._lock:
            n = len(self._states)
            self._states.clear()
            return n

    # -- internals -------------------------------------------------------

    def _touch(self, ip: str, now: float) -> _IPState:
        """Get-or-create the per-IP state; refresh LRU position."""
        st = self._states.get(ip)
        if st is None:
            # LRU eviction when full.  We bound the dict to prevent
            # an IP-rotating attacker from OOM'ing the limiter.
            if len(self._states) >= self.max_tracked_ips:
                self._states.popitem(last=False)
            st = _IPState()
            self._states[ip] = st
        else:
            self._states.move_to_end(ip)
        st.last_touched = now
        return st

    def _block_locked(
        self, ip: str, st: _IPState, reason: str, now: float,
    ) -> None:
        """Set the block + log once.  Caller holds the lock."""
        st.blocked_until = now + self.cooldown_s
        st.block_reason  = reason
        st.requests_buf.clear()
        st.errors_4xx_buf.clear()
        logger.warning(
            "rate_limit: blocking %s reason=%s ttl=%ds",
            ip, reason, self.cooldown_s,
        )

    @staticmethod
    def _clear_block_locked(st: _IPState) -> None:
        st.blocked_until = None
        st.block_reason  = None


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _parse_allowlist(entries: Any) -> List[ipaddress._BaseNetwork]:
    """Convert config entries to ``ip_network`` objects.

    Accepts a list of strings, each ``"ip"`` or ``"ip/prefix"``.
    Bare IPs become /32 (v4) or /128 (v6) networks.  Malformed
    entries log a warning and are skipped — startup never fails
    because of an allowlist typo.
    """
    nets: List[ipaddress._BaseNetwork] = []
    if not isinstance(entries, (list, tuple)):
        return nets
    for raw in entries:
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            nets.append(ipaddress.ip_network(raw.strip(), strict=False))
        except ValueError:
            logger.warning(
                "rate_limit: ignoring malformed allowlist entry %r", raw,
            )
    return nets


def _matches_signature(path_with_query: str) -> bool:
    """True if any SIGNATURE_PATTERNS matches the URL."""
    for pat in SIGNATURE_PATTERNS:
        if pat.search(path_with_query):
            return True
    return False


def _is_authenticated_session() -> bool:
    """True iff the current Flask session carries a logged-in user.

    The auth layer (``molbuilder.web.auth``) stashes the principal
    under ``session["user"]`` after a successful CAS / OAuth flow.
    When auth is disabled at the deployment level the key is never
    set and this returns False — the limiter then runs normally
    against everyone.

    Safe to call outside a request context (returns False); safe
    to call when auth is not installed (no exception, just False).
    """
    try:
        return bool(session.get("user"))
    except Exception:  # noqa: BLE001
        # Outside-request-context (RuntimeError), unsigned cookie,
        # missing secret key, malformed session — anything that
        # makes the session unreadable counts as anonymous.
        return False


def _append_bounded(buf: Deque[float], now: float, maxlen: int) -> None:
    """Append, keeping the deque at most ``maxlen`` entries.

    deque(maxlen=...) auto-evicts on append, but we construct the
    deques without maxlen so we can adjust it without rebuilding.
    Equivalent behaviour, slightly more verbose.
    """
    buf.append(now)
    while len(buf) > maxlen:
        buf.popleft()


def _exceeds_window(
    buf: Deque[float], threshold: int, window_s: int, now: float,
) -> bool:
    """True iff the buffer holds ``threshold`` entries within the
    last ``window_s`` seconds.

    The oldest entry sits at ``buf[0]``.  If
    ``len(buf) >= threshold`` AND ``now - buf[0] <= window_s``, the
    threshold is crossed.
    """
    if len(buf) < threshold:
        return False
    return (now - buf[0]) <= window_s


# --------------------------------------------------------------------- #
#  Flask wiring                                                         #
# --------------------------------------------------------------------- #


def init_rate_limit(app: Flask, cfg: Mapping[str, Any]) -> RateLimiter:
    """Install the before/after_request hooks on ``app``.

    Returns the ``RateLimiter`` so tests can inspect state.  The
    instance is stashed on ``app.extensions['rate_limiter']``.
    """
    rl = RateLimiter(cfg)
    app.extensions = getattr(app, "extensions", {})
    app.extensions["rate_limiter"] = rl

    @app.before_request
    def _rate_limit_before():
        # Reconstruct the full path + query so the signature match
        # sees what the attacker actually typed.  ``request.path``
        # is just the route; ``request.query_string`` is the raw
        # bytes — decode forgivingly AND ``unquote_plus`` so an
        # attacker can't bypass the signature regex by URL-encoding
        # `<script>` as `%3Cscript%3E` or padding spaces with `+`.
        from urllib.parse import unquote_plus
        ip = rl.client_ip()
        qs = request.query_string.decode("utf-8", errors="replace")
        path_with_query = request.path
        if qs:
            path_with_query = f"{path_with_query}?{qs}"
        path_with_query = unquote_plus(path_with_query)
        decision = rl.check_request(
            ip, path_with_query,
            authenticated=_is_authenticated_session(),
        )
        if decision is None:
            return None
        # Drop the connection.  Empty body + Connection: close
        # signals "go away" without spending a render cycle.
        reason, ttl = decision
        resp = Response("", status=429)
        resp.headers["Connection"] = "close"
        resp.headers["Retry-After"] = str(max(1, ttl))
        return resp

    @app.after_request
    def _rate_limit_after(response):
        try:
            # THE AUTH GATE'S OWN ANSWER IS NOT EVIDENCE.  "I do not know who
            # you are yet" is what that gate says to every ordinary visitor,
            # including one whose session just expired -- so it is skipped here,
            # marked by the gate that produced it (auth.py::_require_login).
            #
            # Everything else is counted as before.  Not "ignore 401": a 401
            # from anywhere else has a different author and may mean something.
            if getattr(g, "molbuilder_auth_challenge", False):
                return response
            rl.record_response(
                rl.client_ip(), response.status_code,
                authenticated=_is_authenticated_session(),
            )
        except Exception:  # noqa: BLE001
            # Never let a rate-limit bookkeeping error tank the
            # response.  Log + move on.
            logger.exception("rate_limit: record_response failed")
        return response

    _register_admin_routes(app, rl)
    logger.info(
        "rate_limit: enabled=%s threshold_404=%d/%ds "
        "threshold_total=%d/%ds cooldown=%ds trust_proxy=%s allowlist=%d",
        rl.enabled, rl.threshold_404, rl.window_404_s,
        rl.threshold_total, rl.window_total_s, rl.cooldown_s,
        rl.trust_proxy, len(rl.allowlist_nets),
    )
    return rl


def _register_admin_routes(app: Flask, rl: RateLimiter) -> None:
    """Admin HTTP surface for live inspection / clearing.

    Routed under ``/api/admin/rate_limit/*``.  Two layers of gate:

    1. The auth gate's ``_require_login`` before_request hook (when
       auth is configured) refuses unauthenticated requests with
       HTTP 401.  Without auth installed, this layer is a no-op —
       the deployment is expected to be loopback-only, gated by the
       bind guard in ``cli.py::_refuse_remote_bind_without_tls``.
    2. ``admin.is_admin_request()`` then refuses a session nobody named,
       with HTTP 403.  The set is the top-level ``admin`` section
       (web/admin.py), which also answers "who may restart the server":
       **absent or empty means nobody**, so the state you get by writing
       no config is the safe one.
    """
    from .admin import is_admin_request

    def _refuse_non_admin():
        return jsonify({
            "ok": False,
            "error": (
                "admin auth required to access "
                "/api/admin/rate_limit/*: this session is not one of the "
                "emails named in the `admin` section of molbuilder.json "
                "(an absent or empty list means nobody is an admin)"
            ),
        }), 403

    @app.route("/api/admin/rate_limit/status", methods=["GET"])
    def _rl_status():
        if not is_admin_request():
            return _refuse_non_admin()
        return jsonify({"ok": True, **rl.status()})

    @app.route("/api/admin/rate_limit/clear", methods=["POST"])
    def _rl_clear():
        if not is_admin_request():
            return _refuse_non_admin()
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({
                "ok": False, "error": "body must be a JSON object",
            }), 400
        if body.get("all") is True:
            n = rl.clear_all()
            return jsonify({"ok": True, "cleared": n})
        ip = body.get("ip")
        if not isinstance(ip, str) or not ip.strip():
            return jsonify({
                "ok": False, "error": "missing 'ip' (or 'all': true)",
            }), 400
        ok = rl.clear(ip.strip())
        return jsonify({"ok": True, "cleared": 1 if ok else 0})
