"""Tests for the IP rate-limit + scanner-detection module.

Surfaces under test (docs/protocols/rate-limit.md):

* Signature match → immediate block + connection-close 429.
* 404-storm → block on the Nth 4xx within window.
* Total-burst → block on the Mth total request within window.
* TTL expiry → blocked → past TTL → cleared automatically.
* Allowlist → IP in allowlist never blocked, even on signature.
* X-Forwarded-For trust → ``trust_proxy=True`` honors XFF leftmost
  entry; ``trust_proxy=False`` (default) ignores it.
* Admin endpoints — list status; clear single IP; clear all.

The tests construct their own Flask app with rate_limit enabled.
The default ``web_client`` fixture in conftest.py disables the
limiter so unrelated tests don't trip the total-burst threshold.
"""
from __future__ import annotations

import time

import pytest

# The module under test relies on Flask.  Skip the whole file
# cleanly if Flask isn't available in the test env.
flask = pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


def _build_client(rate_limit_cfg=None, environ_base=None):
    """Build a test client with the rate-limit module enabled.

    ``rate_limit_cfg`` overrides apply on top of the module
    defaults.  ``environ_base`` lets a caller pin
    ``REMOTE_ADDR`` and headers for every request without
    repeating the kwargs at every call site.
    """
    from molbuilder.web.app import create_app
    cfg = {"rate_limit": {"enabled": True, **(rate_limit_cfg or {})}}
    app = create_app(config=cfg)
    client = app.test_client()
    if environ_base:
        client.environ_base = {**client.environ_base, **environ_base}
    return app, client


@pytest.fixture
def fast_client():
    """Tight thresholds for fast tests (signature still immediate)."""
    _app, client = _build_client({
        "window_404_s":    60,
        "threshold_404":   5,
        "window_total_s":  60,
        "threshold_total": 10,
        "cooldown_s":      60,
    })
    client.environ_base = {**client.environ_base,
                           "REMOTE_ADDR": "203.0.113.7"}
    return client


# --------------------------------------------------------------------- #
#  Signal: signature match                                              #
# --------------------------------------------------------------------- #


class TestSignatureMatch:
    """Scanner-fingerprint patterns immediately blacklist the IP."""

    @pytest.mark.parametrize("path", [
        "/?<script>alert(1)</script>",
        "/<meta%20http-equiv=Set-Cookie%20content=foo>",
        "/api/health?x=union+select+1",
        "/api/health?cmd=;drop+table+users",
        "/api/health?x=document.cookie",
        "/etc/passwd",
        # ../../../ traversal patterns
        "/files?p=../../../etc/passwd",
    ])
    def test_known_signatures_trigger_immediate_block(
            self, fast_client, path):
        r = fast_client.get(path)
        assert r.status_code == 429
        assert r.headers.get("Connection") == "close"
        # Retry-After present + sensible.
        retry = int(r.headers.get("Retry-After", "0"))
        assert 1 <= retry <= 7200

    def test_legitimate_url_does_not_match_signatures(self, fast_client):
        # Real molbuilder URLs MUST NOT trip the signature.
        r = fast_client.get("/api/health")
        assert r.status_code == 200, r.data
        r = fast_client.get("/api/backends")
        assert r.status_code == 200

    def test_block_persists_to_next_request(self, fast_client):
        # First: signature hit blocks the IP.
        fast_client.get("/?<script>x</script>")
        # Second: even a legitimate URL is dropped.
        r = fast_client.get("/api/health")
        assert r.status_code == 429
        assert r.headers.get("Connection") == "close"


# --------------------------------------------------------------------- #
#  Signal: 404-storm                                                    #
# --------------------------------------------------------------------- #


class TestStorm404:
    """N+ 4xx responses within the window flip the IP to blocked."""

    def test_blocks_on_threshold_404s(self, fast_client):
        # threshold_404 = 5 (set by fast_client fixture).  Six
        # 404s should land the 6th one as a 429 (or the same
        # 404, depending on after-vs-before timing).  Verify the
        # *7th* request — which would otherwise return 404 — is
        # blocked.
        for i in range(6):
            fast_client.get(f"/never-{i}")
        r = fast_client.get("/api/health")
        assert r.status_code == 429, (
            f"7th request after 6×404 should be blocked; got "
            f"{r.status_code}"
        )

    def test_200s_do_not_count_toward_storm(self):
        # Many 200s do NOT trip 404 storm (it counts only 4xx).
        # Build a client whose total-burst is effectively
        # disabled so this test isolates the 4xx-storm signal
        # exclusively.
        _app, client = _build_client({
            "threshold_404":   5,
            "threshold_total": 10_000,
        })
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "203.0.113.42"}
        for _ in range(20):
            r = client.get("/api/health")
            assert r.status_code == 200
        # Should still be able to make a fresh request.
        r = client.get("/api/backends")
        assert r.status_code == 200


# --------------------------------------------------------------------- #
#  Signal: total-burst                                                  #
# --------------------------------------------------------------------- #


class TestStormTotal:
    """M+ requests within the total-burst window flip the IP."""

    def test_blocks_on_threshold_total(self):
        # Build a client where 4xx threshold is sky-high so it
        # CAN'T be the trigger.  Only the total-burst signal can
        # fire here.
        _app, client = _build_client({
            "threshold_404":   10_000,
            "threshold_total": 8,
            "window_total_s":  60,
            "cooldown_s":      60,
        })
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "203.0.113.8"}
        for _ in range(9):
            client.get("/api/health")
        r = client.get("/api/health")
        # 10th request is past the threshold + window — blocked.
        assert r.status_code == 429


# --------------------------------------------------------------------- #
#  TTL expiry                                                           #
# --------------------------------------------------------------------- #


class TestTTLExpiry:
    """Once cooldown elapses, the IP is auto-cleared on next probe."""

    def test_block_clears_after_cooldown(self, fast_client, monkeypatch):
        # Block via signature.
        r = fast_client.get("/?<script>x</script>")
        assert r.status_code == 429
        # Fast-forward time on the module's monotonic clock.
        import molbuilder.web.rate_limit as rl_mod
        real_monotonic = rl_mod.time.monotonic
        offset = [0.0]
        monkeypatch.setattr(
            rl_mod.time, "monotonic",
            lambda: real_monotonic() + offset[0],
        )
        # Advance past the cooldown (fast_client uses 60s).
        offset[0] = 61.0
        # A clean request should now pass.
        r = fast_client.get("/api/health")
        assert r.status_code == 200, (
            f"expected clean pass after cooldown; got "
            f"{r.status_code}"
        )


# --------------------------------------------------------------------- #
#  Allowlist                                                            #
# --------------------------------------------------------------------- #


class TestAllowlist:
    """Allowlisted IPs are never blocked, even on signature match."""

    def test_allowlist_exempts_signature_match(self):
        _app, client = _build_client({"allowlist": ["10.0.0.5"]})
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "10.0.0.5"}
        # Signature would normally block immediately.
        r = client.get("/?<script>x</script>")
        # Allowlisted → no rate-limit interference.  The route's
        # actual handler responds (in this case 404 since the URL
        # doesn't match a route).  Whatever the handler returns,
        # it MUST NOT be the 429+Connection-close from the
        # limiter.
        assert r.headers.get("Connection") != "close"

    def test_cidr_allowlist_matches(self):
        _app, client = _build_client({"allowlist": ["10.0.0.0/24"]})
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "10.0.0.42"}
        r = client.get("/?<script>x</script>")
        assert r.headers.get("Connection") != "close"

    def test_non_allowlisted_neighbour_still_blocked(self):
        _app, client = _build_client({"allowlist": ["10.0.0.0/24"]})
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "10.0.1.7"}
        r = client.get("/?<script>x</script>")
        assert r.status_code == 429

    def test_malformed_allowlist_entries_logged_and_skipped(self, caplog):
        with caplog.at_level("WARNING"):
            _app, _client = _build_client(
                {"allowlist": ["10.0.0.0/24", "not-an-ip", ""]}
            )
        assert any("malformed allowlist entry" in r.message
                   for r in caplog.records)


# --------------------------------------------------------------------- #
#  X-Forwarded-For trust                                                #
# --------------------------------------------------------------------- #


class TestXFFTrust:
    """``trust_proxy`` controls whether XFF is honoured."""

    def test_xff_ignored_by_default(self):
        # trust_proxy defaults to False.  An attacker setting XFF
        # to spoof an allowlisted IP must NOT bypass the limiter.
        _app, client = _build_client({"allowlist": ["10.0.0.5"]})
        # request comes from 203.0.113.99; spoofs XFF=10.0.0.5.
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "203.0.113.99"}
        r = client.get(
            "/?<script>x</script>",
            headers={"X-Forwarded-For": "10.0.0.5"},
        )
        assert r.status_code == 429, (
            "XFF must be ignored when trust_proxy=False"
        )

    def test_xff_honoured_when_trust_proxy_true(self):
        _app, client = _build_client({
            "trust_proxy": True,
            "allowlist":   ["10.0.0.5"],
        })
        # Direct connection from proxy at 127.0.0.1; XFF carries
        # the actual client (10.0.0.5, allowlisted).
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "127.0.0.1"}
        r = client.get(
            "/?<script>x</script>",
            headers={"X-Forwarded-For": "10.0.0.5, 192.168.1.1"},
        )
        # Allowlisted via XFF leftmost → not blocked.
        assert r.headers.get("Connection") != "close"


# --------------------------------------------------------------------- #
#  Admin endpoints                                                      #
# --------------------------------------------------------------------- #


class TestAdminEndpoints:
    """``/api/admin/rate_limit/*`` surfaces for live inspection."""

    def test_status_reports_blocked_ips(self, fast_client):
        # Trigger a block.
        fast_client.get("/?<script>x</script>")
        # Switch to a NON-attacker IP so the status query itself
        # isn't dropped.
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "127.0.0.1"}
        r = fast_client.get("/api/admin/rate_limit/status")
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert body["blocked_count"] >= 1
        ips = [entry["ip"] for entry in body["blocked"]]
        assert "203.0.113.7" in ips, ips
        # Reason + TTL surface.
        entry = next(e for e in body["blocked"] if e["ip"] == "203.0.113.7")
        assert entry["reason"] == "signature_match"
        assert entry["ttl_s"] > 0

    def test_clear_single_ip(self, fast_client):
        fast_client.get("/?<script>x</script>")
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "127.0.0.1"}
        r = fast_client.post(
            "/api/admin/rate_limit/clear",
            json={"ip": "203.0.113.7"},
        )
        assert r.status_code == 200
        assert r.get_json() == {"ok": True, "cleared": 1}
        # Original IP can now make requests again.
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "203.0.113.7"}
        r = fast_client.get("/api/health")
        assert r.status_code == 200

    def test_clear_all(self, fast_client):
        fast_client.get("/?<script>x</script>")
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "127.0.0.1"}
        r = fast_client.post(
            "/api/admin/rate_limit/clear",
            json={"all": True},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body["ok"] is True
        assert body["cleared"] >= 1

    def test_clear_rejects_non_dict_body(self, fast_client):
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "127.0.0.1"}
        r = fast_client.post(
            "/api/admin/rate_limit/clear",
            json=["not", "a", "dict"],
        )
        assert r.status_code == 400
        assert r.get_json()["ok"] is False

    def test_clear_rejects_missing_ip(self, fast_client):
        fast_client.environ_base = {**fast_client.environ_base,
                                    "REMOTE_ADDR": "127.0.0.1"}
        r = fast_client.post("/api/admin/rate_limit/clear", json={})
        assert r.status_code == 400


# --------------------------------------------------------------------- #
#  Authenticated bypass                                                 #
# --------------------------------------------------------------------- #


class TestAuthenticatedBypass:
    """A logged-in session short-circuits the limiter.

    Rationale: a principal that passed CAS / OAuth is by
    definition not an anonymous scanner.  The 2026-06-18 hotfix
    added this bypass after the original total-burst threshold
    killed legitimate 1 Hz pollers; even with total-burst now
    off by default, an authenticated user mis-clicking through
    their session shouldn't accumulate toward a blocklist trip.
    """

    def test_signature_match_allowed_for_authenticated(self):
        # Build a client with an absurdly tight threshold so
        # even ONE bad request would trip.  Then put a "user" in
        # the session and verify the limiter stays out of the
        # way.
        app, client = _build_client({
            "threshold_404": 1,
            "cooldown_s":    60,
        })
        # Sessions need a signed secret key; the no-auth default
        # config doesn't install one, so set a test-only value.
        app.secret_key = "test-only-not-for-prod"
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "203.0.113.99"}
        # Without the session, the signature match blocks.
        r = client.get("/?<script>x</script>")
        assert r.status_code == 429
        # Clear the blocklist so the next test isolates the
        # session signal.
        app.extensions["rate_limiter"].clear_all()
        # With a logged-in session, the same URL must pass
        # through (the route's 404 / 302 is whatever Flask
        # returns; the key is "NOT 429+Connection-close").
        with client.session_transaction() as s:
            s["user"] = {"email": "alice@example.com"}
        r = client.get("/?<script>x</script>")
        assert r.headers.get("Connection") != "close", (
            "authenticated session must bypass the limiter; "
            f"got Connection={r.headers.get('Connection')!r}, "
            f"status={r.status_code}"
        )

    def test_404_storm_not_counted_for_authenticated(self):
        # 4xx responses from an authenticated user must NOT
        # accumulate toward the 404 storm signal.
        app, client = _build_client({
            "threshold_404":  3,
            "window_404_s":   60,
            "cooldown_s":     60,
        })
        app.secret_key = "test-only-not-for-prod"
        client.environ_base = {**client.environ_base,
                               "REMOTE_ADDR": "203.0.113.100"}
        with client.session_transaction() as s:
            s["user"] = {"email": "alice@example.com"}
        # 5 404s with an active session — none should count.
        for i in range(5):
            r = client.get(f"/no-such-{i}")
            assert r.status_code in (302, 404), r.status_code
        # The next normal request must still pass (no block).
        r = client.get("/api/health")
        assert r.status_code == 200


# --------------------------------------------------------------------- #
#  Disabled mode                                                        #
# --------------------------------------------------------------------- #


class TestDisabledMode:
    """``rate_limit.enabled=false`` makes the module a no-op."""

    def test_signature_passes_through_when_disabled(self):
        from molbuilder.web.app import create_app
        app = create_app(config={"rate_limit": {"enabled": False}})
        client = app.test_client()
        # The classic attack pattern should sail through (the URL
        # is malformed so the handler returns 404, but it's NOT
        # the 429+Connection-close that means "rate-limited").
        r = client.get("/?<script>x</script>")
        assert r.headers.get("Connection") != "close"


# --------------------------------------------------------------------- #
#  LRU eviction                                                         #
# --------------------------------------------------------------------- #


class TestLRUEviction:
    """The tracked-IP dict is bounded.  When full, the oldest entry
    is evicted on the next-touched IP.
    """

    def test_eviction_drops_oldest_ip(self):
        # max_tracked_ips=3 so we can trigger eviction with 4 IPs.
        _app, client = _build_client({"max_tracked_ips": 3})
        # Hit from 4 different IPs.
        for i in range(4):
            client.environ_base = {**client.environ_base,
                                   "REMOTE_ADDR": f"203.0.113.{i + 1}"}
            client.get("/api/health")
        # State should hold at most 3 IPs (the most recent).
        rl = _app.extensions["rate_limiter"]
        assert len(rl._states) == 3
        # The oldest IP (.1) should have been evicted.
        assert "203.0.113.1" not in rl._states
        assert "203.0.113.4" in rl._states
