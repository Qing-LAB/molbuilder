# Rate-limit + scanner-detection contract

The IP rate-limit module (`molbuilder.web.rate_limit`) installs a
pair of Flask request hooks that detect scanner-style abuse and
blacklist the offending IP for a configurable cooldown.  This doc
describes the threat model, the three detection signals, the
on-wire shape of a blocked response, and the admin surface.

The module is **on by default** when the app is created.
Operators who don't want it set `rate_limit.enabled = false` in
`molbuilder.json`.  Tests get it disabled via the default
`web_client` fixture in `tests/conftest.py`.

## 1. Threat model

molbuilder typically runs as a Flask app on a single host
reachable from the public internet (e.g.
`qlabsrv.physics.asu.edu`).  The auth layer protects the
authenticated surface (CAS / OAuth) but unauthenticated surfaces
remain probeable:

* `/login` returns 200 to anyone — it's the login picker.
* `/api/*` returns `{ok:false}` with 401 to unauthenticated
  callers, but the response is cheap.
* Any URL not matching a route returns 404.

A scanner can therefore enumerate paths and probe for reflected
XSS at 5-10 req/s.  The on-the-record incident that motivated
this module: 2026-06-18, IP `100.27.42.242`, ~40 requests in 9 s,
probing every common backend extension (`.jsp`, `.php`, `.cfc`,
`.nsf`, `.dll`, …) with `<script>document.cookie=...` and
`<meta http-equiv=Set-Cookie content=...>` payloads in the query
string.

## 2. Detection signals

Any one of three signals flips the IP onto the blocklist for
`cooldown_s` seconds.

### 2.1 Signature match (immediate)

The request URL (path + URL-decoded query) is matched against
`SIGNATURE_PATTERNS` in `rate_limit.py`.  A match means "this is
an attacker fingerprint, not a real client" — no legitimate
molbuilder URL contains any of these strings.

Current patterns:

| Pattern (case-insensitive) | What it catches |
|---|---|
| `<\s*script` | XSS reflection probe |
| `<\s*meta\s+http-equiv` | XSS cookie-set probe |
| `document\.cookie` | XSS cookie-exfil probe |
| `\bunion\s+select` | SQLi probe (post-decode) |
| `;\s*drop\s+(table\|database)` | SQLi DROP probe |
| `/etc/passwd` | LFI probe |
| `\.\.[\\/]\.\.[\\/]\.\.` | Directory traversal `../../../` |

The match runs against the **URL-decoded** path+query.  An
attacker can't bypass by writing `%3Cscript%3E` instead of
`<script>` — the decode happens before the regex.

### 2.2 404-storm

`threshold_404` 4xx responses within `window_404_s` seconds from
the same IP → block.  Catches path enumeration where the
scanner probes many non-existent paths.

The 4xx count is tracked in a bounded deque per IP (size =
threshold).  When the deque is full AND the oldest entry is
within the window, the IP trips.

### 2.x Authenticated bypass (2026-06-18)

If the current request carries a logged-in Flask session
(`session["user"]` is set after a successful CAS / OAuth flow),
the limiter short-circuits the whole signal chain.  Rationale: a
principal that passed the SSO gate is by definition not an
anonymous scanner, and the limiter exists to deflect anonymous
scanners.  An authenticated user mis-clicking through their
session shouldn't accumulate toward a blocklist trip.

This bypass is in addition to the IP-based allowlist; either one
short-circuits.  When auth isn't installed at the deployment
level, `session["user"]` is never set and every request runs
through the signal chain normally.

### 2.3 Total-burst (off by default)

`threshold_total` total requests (any status) within
`window_total_s` seconds → block.  Catches slower scanners that
don't trip the 404 signal because they found a real 200 and
started hammering.

**Disabled by default** (`threshold_total = 0`) since the
2026-06-18 hotfix: the 60/60s ceiling killed the legitimate 1 Hz
poll of `/api/system/load` from the system-load monitor (#472),
which is exactly the user-driven traffic the limiter exists to
protect.  Successful 200s should not count toward an abuse
signal; the 404-storm + signature signals already catch the
canonical scanner pattern.

To turn it back on (paranoid deployment, no legitimate poller),
set `threshold_total` to a non-zero value — recommended floor is
**600** (= 10/s sustained for a minute), well above any
legitimate poll cadence.  Setting either signal's threshold to
`0` disables it; the signature-match signal is always on.

## 3. Defaults

| Knob | Default | Tuned against |
|---|---|---|
| `enabled` | `true` | always on; disable entirely via cfg |
| `window_404_s` | `30` | scanner cadence (~5 req/s sustained) |
| `threshold_404` | `20` | 30-second false-positive bound on a careful real user |
| `window_total_s` | `60` | (unused while `threshold_total = 0`) |
| `threshold_total` | `0` (off) | see § 2.3 — disabled to protect legitimate pollers |
| `cooldown_s` | `3600` | 1 hour — enough to deter, short enough to recover from a false positive |
| `trust_proxy` | `false` | use `request.remote_addr` only |
| `allowlist` | `["127.0.0.1", "::1"]` | localhost always trusted; bind guard ensures localhost really is local |
| `max_tracked_ips` | `10_000` | LRU eviction bound |

Override any subset in `molbuilder.json`:

```json
{
  "rate_limit": {
    "threshold_404": 30,
    "cooldown_s":    7200,
    "allowlist":     ["10.0.0.0/24", "203.0.113.5"],
    "trust_proxy":   true
  }
}
```

`allowlist` accepts both bare IPs (treated as `/32` or `/128`)
and CIDR strings.  Malformed entries are logged at WARNING and
skipped — startup never fails on a typo.

## 4. Wire shape — blocked response

A blocked request gets:

```
HTTP/1.1 429 Too Many Requests
Connection: close
Retry-After: <ttl-remaining-seconds>
Content-Length: 0
```

Empty body.  The `Connection: close` tells the OS to drop the
socket; the scanner pays the TCP teardown cost on every
subsequent attempt, which is the entire point.

Per `web-api.md` § 1.6 this is a **server-fault-class** response
in the sense that the *request was rejected at the protocol /
infrastructure layer* before any application code ran.  4xx is
the right HTTP class here (the client is the cause).  429 is
the spec-correct sub-code for rate-limit triggers.

## 5. Admin surface

Two endpoints, both under `/api/admin/rate_limit/*`.  They go
through the auth gate when auth is enabled; otherwise they're
reachable from localhost only (the bind guard at `cli.py`
prevents non-loopback HTTP without TLS).

### 5.1 `GET /api/admin/rate_limit/status`

```json
{
  "ok": true,
  "enabled":       true,
  "blocked_count": 2,
  "blocked": [
    {"ip": "100.27.42.242", "reason": "signature_match", "ttl_s": 3593},
    {"ip": "203.0.113.7",   "reason": "404_storm",       "ttl_s": 3551}
  ],
  "tracked_count": 7
}
```

### 5.2 `POST /api/admin/rate_limit/clear`

Body shape (one of):

| Body | Effect |
|---|---|
| `{"ip": "100.27.42.242"}` | Unblock a specific IP |
| `{"all": true}` | Drop every tracked IP |

Returns `{"ok": true, "cleared": <n>}`.  Malformed body → 400
(`{ok:false, error: "..."}`).

## 6. Multi-worker caveat

State lives in-process.  Single-worker `molbuilder serve` (the
default): thresholds work as advertised.  Multi-worker (e.g.
`gunicorn -w 4`): each worker has its own state, so the same IP
could hit each worker independently — effective threshold
becomes `N * configured`.

For full-strength protection across workers, add an nginx
`limit_req` rule upstream.  See `docs/deployment.md` for the
production deployment notes.

## 7. Persistence

NONE.  Restart clears the blocklist.  Two reasons this is the
right tradeoff:

1. **Restart is itself a mitigation.**  An attacker mid-scan loses
   their connection on restart.
2. **A persisted blocklist tends to grow stale.**  A `cooldown_s`
   that was correct at write-time may not match what the
   operator wants after restart.

Incident-response durability is out of scope; the warning log
each block emits (`rate_limit: blocking <ip> reason=<r>
ttl=<n>s`) is the audit trail.

## 8. Trust-proxy security note

When `trust_proxy = true`, the limiter honours the leftmost
entry of `X-Forwarded-For` as the client IP.  This is correct
behind a reverse proxy that scrubs client-supplied XFF before
adding its own (nginx with the default `proxy_set_header
X-Forwarded-For $proxy_add_x_forwarded_for` setup does this).

DO NOT enable `trust_proxy` on a direct-bind deployment.  An
attacker can craft `X-Forwarded-For: <allowlisted-IP>` and
bypass every signal.  The default (`false`) is correct for the
TLS-direct shape `cli.py::_refuse_remote_bind_without_tls`
documents.

## 9. Tests that pin the contract

| Surface | Test |
|---|---|
| Signature match (all canonical patterns) | `tests/test_rate_limit.py::TestSignatureMatch` |
| 404-storm threshold + window | `tests/test_rate_limit.py::TestStorm404` |
| Total-burst threshold + window | `tests/test_rate_limit.py::TestStormTotal` |
| TTL auto-expiry on cooldown elapse | `tests/test_rate_limit.py::TestTTLExpiry` |
| Allowlist (IP, CIDR, malformed entry) | `tests/test_rate_limit.py::TestAllowlist` |
| XFF off (default) ignores spoofed header | `tests/test_rate_limit.py::TestXFFTrust::test_xff_ignored_by_default` |
| XFF on honours leftmost only | `tests/test_rate_limit.py::TestXFFTrust::test_xff_honoured_when_trust_proxy_true` |
| Admin endpoints (status, clear single, clear all, malformed body) | `tests/test_rate_limit.py::TestAdminEndpoints` |
| Disabled mode passes everything through | `tests/test_rate_limit.py::TestDisabledMode` |
| LRU eviction of oldest tracked IP | `tests/test_rate_limit.py::TestLRUEviction` |

## 10. Operational runbook

* **An attack is active.**  Check
  `GET /api/admin/rate_limit/status` — you should see the
  attacker IP in `blocked`.  If it's NOT there, the signals
  aren't tripping; consider lowering thresholds or adding a
  pattern.
* **An IP is wrongly blocked.**  `POST /api/admin/rate_limit/clear
  {"ip": "..."}` immediately unblocks.  Then either (a) add it
  to `allowlist`, or (b) raise the threshold that fired.
* **The limiter is itself eating memory.**  Check
  `tracked_count`.  If it's near `max_tracked_ips`, you're being
  IP-rotated.  Either raise `max_tracked_ips` (eats more memory
  but tracks more) or add an upstream nginx rate-limit (deflects
  before the request reaches Flask).
* **You want to silence the limiter completely.**  Set
  `rate_limit.enabled = false` in `molbuilder.json` and restart.
