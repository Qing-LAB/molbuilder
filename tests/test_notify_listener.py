"""The receiving end for a run's own reports, and what it refuses.

`execution/run-reports.md` § 4. This is the smallest endpoint in the app on
purpose — **it appends one line to a log and answers ok** — and every test
here is about a way it must not do more than that.

**Four gates, and only one is the control.**

1. *Has anyone enabled this?* — registered only when **both** the key file and
   the route segment are configured.
2. *Where is it?* — a per-deployment random segment, never a word in this
   repository.
3. *May this sender write?* — an HMAC-SHA256 signature over the exact body.
   **This is the control**; the other three exist so a stranger cannot learn
   whether it is even there.
4. *What does a failure reveal?* — nothing. Every one answers a plain `404`.

The rules from `ops/access-control.md` § 8 it is built on:

* **rule 1** — *"the safe state is the one you get by doing nothing"*: with
  either config key missing, there is no route.
* **rule 2** — *"absent beats refused, when existence is itself the answer"*:
  a wrong signature and an unconfigured server answer identically, so probing
  cannot tell them apart.
* **rule 4** — *"judge behaviour, not people"*: nobody reaches this by
  accident, so a failure is a probe and must reach the limiter. `auth.py`'s
  gate exempts its OWN 401 (an expired session is an ordinary visitor, and
  counting it once locked a user out of their site for an hour) — that
  exemption must not spread here.
* **rule 7** — *"prefer the secret that never travels"*: the key signs, and
  stays on the cluster.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from molbuilder.web.app import create_app
from molbuilder.web.blueprints.notify import sign


KEY = "test-key-not-a-real-one"
USER = "someone@example.org"
ROUTE = "x7KqTestSegment"
PATH = f"/api/{ROUTE}"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A configured server, with the report store pointed inside tmp.

    Reports live under the STATE directory since 2026-08-31 -- XDG's own home
    for data that persists but is not configuration (`configuration.md`
    § 2.1d).  Setting ``HOME`` alone stopped being enough then: it moved
    ``~/.molbuilder`` because that path was built from the home directory, and
    the state directory is named by its own variable.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    keys = tmp_path / "notify_keys"
    keys.write_text(json.dumps({USER: KEY}))
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_keys_file": str(keys),
                             "notify_route": ROUTE})
    from molbuilder.web.blueprints import notify as N
    N._loggers.clear()          # rotating handlers are cached per user
    N._recent.clear()           # and so is the per-key rate window
    return app.test_client(), tmp_path / "state" / "molbuilder" / "reports"


def _post(client, key=KEY, body=None, raw=None, path=PATH, ts=None, sig=None):
    """One correctly signed report, unless the caller asks for otherwise."""
    data = raw if raw is not None else json.dumps(
        body if body is not None else {"event": "finish", "text": "done"})
    blob = data.encode() if isinstance(data, str) else data
    stamp = ts if ts is not None else "%d" % int(time.time())
    headers = {"Content-Type": "application/json",
               "X-Molbuilder-Timestamp": stamp}
    if sig is not None:
        headers["X-Molbuilder-Signature"] = sig
    elif key is not None:
        headers["X-Molbuilder-Signature"] = sign(key, stamp, blob)
    return client.post(path, data=data, headers=headers)


def _lines(log_root: Path):
    f = log_root / f"{USER}.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------- #
#  gate 1: it exists only when it was enabled                            #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("cfg,why", [
    ({}, "neither key"),
    ({"notify_route": ROUTE}, "a route but no keys"),
    ({"notify_keys_file": "/nonexistent/keys"}, "keys but no route"),
])
def test_both_config_keys_are_required_or_there_is_no_route(
        tmp_path, monkeypatch, cfg, why):
    """`access-control.md` § 8 rule 1: *the safe state is the one you get by
    doing nothing.* Half a configuration must not open a door."""
    monkeypatch.setenv("HOME", str(tmp_path))
    app = create_app(config={"rate_limit": {"enabled": False}, **cfg})
    # The LISTENER's blueprint specifically.  `notify_setup` also lives
    # under /api/notify/ and is always registered -- it is the signed-in
    # SETUP api, about sending reports FROM here, where this is about
    # receiving them (`run-reports.md` § 3.1).  Matching on the substring
    # caught both and made this fail for the wrong reason.
    assert not [r for r in app.url_map.iter_rules()
                if r.endpoint.startswith("notify.")], why


def test_no_fixed_notify_path_exists_anywhere_in_the_source():
    """**The segment is generated, never named.**

    A cleverer word would be committed to a public repository and so be
    exactly as public as `notify`, only less honest about what it does
    (`access-control.md` § 8 rule 7). This is the same shape of guard as
    `test_config_dir_has_one_home` — a comment saying *do not hard-code it*
    is not a mechanism.
    """
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify.py").read_text()
    body = src.split('"""', 2)[2]          # past the module docstring
    assert '"/api/notify"' not in body
    assert "'/api/notify'" not in body
    assert '"/api/<route>"' in body, "the route must be a parameter"


# --------------------------------------------------------------------- #
#  gate 2 + 4: a wrong guess is indistinguishable from nothing           #
# --------------------------------------------------------------------- #

def test_results_are_stored_apart_from_molbuilders_own_LOGS(store):
    """User, 2026-08-27: *this is a different kind of log, not of the status
    of molbuilder but collection of computation results.*

    the log directory holds diagnostics — you read it when something is
    wrong and delete it when it is fixed. These are measurements from
    calculations, the kind you keep and grep a year later. Filing them
    together invited exactly one mistake: treating results as disposable.
    """
    client, reports = store
    _post(client)
    assert reports.name == "reports"
    assert "logs" not in reports.parts, \
        "results were filed under molbuilder's own operational logs"
    assert (reports / f"{USER}.jsonl").exists()


def test_a_stored_line_stands_on_its_own(store):
    """*Each line should be self-contained with all key information,
    something that can be parsed easily.*

    A line used to read `{"event": "scf_converged", "energy": "-1740.2"}`
    and nothing said WHICH calculation, on WHAT machine, or WHEN it was
    sent. With two jobs running the lines were indistinguishable. Somebody
    parses this file later with no session to ask.
    """
    client, reports = store
    from molbuilder import monitor as M
    ident = M.run_identity("BDT_Au_relax-run0.out")
    body = {**ident, "sent_at": 1756000000.5, "event": "scf_converged",
            "state": "running", "n_iters": 7, "energy": "-1740.21",
            "geom_step": 3, "elapsed_s": 1234.5}
    assert _post(client, body=body).status_code == 200
    line = _lines(reports)[0]
    assert line["run"] == "BDT_Au_relax", "the label names the calculation"
    assert line["host"], "which machine"
    assert line["user"] == USER, "whose, in the LINE and not only the filename"
    assert line["v"] == 1, "which shape, for a reader a year from now"
    assert line["sent_at"] == 1756000000.5, "the sender's clock"
    assert line["received_at"] >= line["sent_at"], "and ours, beside it"
    assert line["energy"] == "-1740.21" and line["n_iters"] == 7


def test_the_user_is_STAMPED_never_accepted(store):
    """The key is the claim. A payload that names a user must not become
    one, or a valid key could write into somebody else's record."""
    client, reports = store
    _post(client, body={"event": "tick", "user": "somebody-else"})
    assert _lines(reports)[0]["user"] == USER


def test_a_report_is_readable_line_by_line_with_no_parser_of_ours(store):
    """JSON Lines, so `jq` and `pandas` both read it directly. A string
    containing newlines must not become two records."""
    client, reports = store
    _post(client, body={"event": "tick", "text": "a\nb\nc"})
    raw = (reports / f"{USER}.jsonl").read_text()
    assert len(raw.strip().splitlines()) == 1
    assert json.loads(raw.strip())["text"] == "a\nb\nc"


def test_a_valid_key_cannot_flood_the_record(store):
    """User, 2026-08-27: *we should have a rate-limit on the notify port
    too.*

    `rate_limit.py` bounds FAILURES — its 404-storm signal counts 4xx — and
    its total-request threshold ships disabled. So a valid key could POST
    without bound. **The harm is not the disk**: the record rotates at
    1 MB × 5, so a flood would silently push a run's real reports out of
    the window. A cap is what keeps the results the results.
    """
    from molbuilder.web.blueprints.notify import MAX_REPORTS_PER_MIN
    client, reports = store
    ok = sum(1 for _ in range(MAX_REPORTS_PER_MIN)
             if _post(client).status_code == 200)
    assert ok == MAX_REPORTS_PER_MIN, "a legitimate burst was cut short"
    assert _post(client).status_code == 404, "the cap did not engage"
    assert len(_lines(reports)) == MAX_REPORTS_PER_MIN


def test_the_cap_is_per_KEY_not_per_address(store, tmp_path):
    """A cluster NATs every compute node behind one address, so a per-IP cap
    would punish somebody running several jobs for using the machine they
    were given."""
    from molbuilder.web.blueprints import notify as N
    client, _ = store
    N._recent.clear()
    for _ in range(N.MAX_REPORTS_PER_MIN):
        _post(client)
    assert _post(client).status_code == 404
    # a different key, same client address, is unaffected
    N._recent.pop(USER, None)
    assert _post(client).status_code == 200


def test_a_capped_report_answers_404_like_every_other_refusal(store):
    """Answering differently would say *this key is valid, you are merely
    early* — exactly what the other gates exist to withhold."""
    from molbuilder.web.blueprints import notify as N
    client, _ = store
    N._recent.clear()
    for _ in range(N.MAX_REPORTS_PER_MIN):
        _post(client)
    r = _post(client)
    assert r.status_code == 404
    assert b"limit" not in r.get_data().lower()
    assert b"rate" not in r.get_data().lower()


def test_the_cap_is_a_WINDOW_not_a_lifetime_quota(store, monkeypatch):
    """**Found by mutation, 2026-08-27.** Deleting the trim passed all 52
    tests: the deque stops growing at the cap either way, so the
    memory-bound test could not tell the difference.

    What the trim actually buys is RECOVERY — without it a monitor that
    once burst past the cap is refused for the life of the process, and
    silently, because a notifier swallows failures. A run would simply stop
    reporting and never start again.
    """
    from molbuilder.web.blueprints import notify as N
    client, reports = store
    N._recent.clear()

    now = [1_000_000.0]
    monkeypatch.setattr(N.time, "time", lambda: now[0])
    for _ in range(N.MAX_REPORTS_PER_MIN):
        assert _post(client).status_code == 200
    assert _post(client).status_code == 404, "the cap did not engage"

    now[0] += N._RATE_WINDOW_S + 1        # the minute passes
    assert _post(client).status_code == 200, \
        "the window never reopened -- one burst silenced the run for good"


def test_the_window_memory_is_bounded_by_the_CAP_not_the_traffic(store):
    """A flood must not become a way to spend the server's RAM instead of
    its disk."""
    from molbuilder.web.blueprints import notify as N
    client, _ = store
    N._recent.clear()
    for _ in range(N.MAX_REPORTS_PER_MIN + 40):
        _post(client)
    assert len(N._recent[USER]) <= N.MAX_REPORTS_PER_MIN


def test_results_are_not_world_readable(store):
    """The KEY file was always 0600; the DATA it protects was 0664 in an
    0775 directory, inheriting the umask — the wrong way round on a shared
    server."""
    import stat
    client, reports = store
    _post(client)
    f = reports / f"{USER}.jsonl"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert stat.S_IMODE(reports.stat().st_mode) == 0o700


def test_the_wrong_segment_is_a_plain_404(store):
    client, log_root = store
    assert _post(client, path="/api/notify").status_code == 404
    assert _post(client, path="/api/webhook").status_code == 404
    assert _lines(log_root) == []


def test_a_wrong_signature_answers_EXACTLY_like_an_unconfigured_server(
        tmp_path, monkeypatch, store):
    """**The point of the whole design.** A 401 would say *there is something
    here and you got it wrong*, which is the one fact the other gates exist
    to keep. Both answers must be byte-identical."""
    client, _ = store
    bad = _post(client, sig="0" * 64)

    monkeypatch.setenv("HOME", str(tmp_path))
    off = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    absent = off.post(PATH, data="{}",
                      headers={"Content-Type": "application/json"})

    assert bad.status_code == absent.status_code == 404
    assert bad.get_data() == absent.get_data(), \
        "a refusal that looks different from absence gives away the capability"


@pytest.mark.parametrize("kw,why", [
    ({"sig": ""},                       "empty signature"),
    ({"sig": "not-hex-at-all"},         "garbage"),
    ({"sig": "a" * 64},                 "well-formed but wrong"),
    ({"key": None},                     "no signature header at all"),
    ({"key": KEY + "x"},                "a key nobody holds"),
])
def test_anything_but_a_valid_signature_is_refused(store, kw, why):
    client, log_root = store
    assert _post(client, **kw).status_code == 404, why
    assert _lines(log_root) == [], f"{why} still wrote a record"


def test_the_signature_covers_the_BODY_so_it_cannot_be_edited(store):
    """What a bearer token could not do. Sign one body, send another, and
    the report is refused — an intercepted report cannot be altered in
    flight, only replayed verbatim."""
    client, log_root = store
    stamp = "%d" % int(time.time())
    honest = json.dumps({"event": "tick", "n_iters": 5}).encode()
    tampered = json.dumps({"event": "tick", "n_iters": 9999}).encode()
    r = client.post(PATH, data=tampered, headers={
        "Content-Type": "application/json",
        "X-Molbuilder-Timestamp": stamp,
        "X-Molbuilder-Signature": sign(KEY, stamp, honest)})
    assert r.status_code == 404
    assert _lines(log_root) == []


def test_the_signature_covers_the_TIMESTAMP_too(store):
    """Signed WITH the body, not sent beside it — otherwise the timestamp
    could be rewritten freely and the freshness window would mean nothing."""
    client, log_root = store
    body = json.dumps({"event": "tick"}).encode()
    real = "%d" % int(time.time())
    r = client.post(PATH, data=body, headers={
        "Content-Type": "application/json",
        "X-Molbuilder-Timestamp": "%d" % (int(time.time()) - 60),
        "X-Molbuilder-Signature": sign(KEY, real, body)})
    assert r.status_code == 404
    assert _lines(log_root) == []


@pytest.mark.parametrize("ts,why", [
    ("%d" % (int(time.time()) - 3600), "an hour old"),
    ("%d" % (int(time.time()) + 3600), "an hour ahead"),
    ("not-a-number", "unparseable"),
    ("", "absent"),
])
def test_a_stale_or_unreadable_timestamp_is_refused(store, ts, why):
    """Bounds how long a captured report stays replayable. Generous, because
    a compute node's clock is not ours to trust closely and a run that
    reports late is not a run that is lying."""
    client, log_root = store
    assert _post(client, ts=ts).status_code == 404, why
    assert _lines(log_root) == []


# --------------------------------------------------------------------- #
#  gate 3: the key is the claim                                          #
# --------------------------------------------------------------------- #

def test_a_valid_signature_is_accepted_and_the_report_is_kept(store):
    client, log_root = store
    r = _post(client, body={"event": "finish", "text": "done", "n_iters": 12})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    kept = _lines(log_root)
    assert len(kept) == 1
    assert kept[0]["event"] == "finish" and kept[0]["n_iters"] == 12
    assert "received_at" in kept[0]


def test_the_sender_never_states_who_it_is(store):
    """**The key is the claim.** One person's key cannot write into
    another's record — the point of issuing one each, and what lets one be
    revoked alone.

    The line *does* carry a `user` field since 2026-08-27, so that a record
    pasted somewhere else still says whose it is. It is **stamped from the
    key that verified**, never read from the payload, so the property is
    unchanged and the mechanism is now visible in the line.
    """
    client, reports = store
    _post(client, body={"event": "tick", "user": "someone-else", "text": "hi"})
    kept = _lines(reports)
    assert len(kept) == 1
    assert kept[0]["user"] == USER, "a payload field became an identity"
    assert (reports / f"{USER}.jsonl").exists()
    assert not (reports / "someone-else.jsonl").exists()


def test_every_key_is_tried_with_no_early_exit(store, tmp_path):
    """Returning on the first match would make the time taken depend on a
    key's position in the file. The loop is short and the cost is nothing;
    the property is worth holding by construction."""
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify.py").read_text()
    fn = src[src.index("def _resolve_user"):src.index("def _logger_for")]
    assert "return" not in fn.split("found = None")[1].split("return found")[0], \
        "_resolve_user gained an early return"


# --------------------------------------------------------------------- #
#  rule 4: the limiter must hear about a probe                           #
# --------------------------------------------------------------------- #

def test_a_failure_is_counted_by_the_limiter(store):
    """`auth.py`'s gate marks its own 401 as *not evidence* — an expired
    session is an ordinary visitor. **That reasoning does not carry here**:
    nobody reaches this route by accident. 404 is still 4xx, so switching
    from 401 costs nothing (`rate_limit.record_response` takes any
    `400 <= status < 500`)."""
    client, _ = store
    with client.application.test_request_context():
        from flask import g
        _post(client, sig="0" * 64)
        assert not getattr(g, "molbuilder_auth_challenge", False)
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify.py").read_text()
    assert "molbuilder_auth_challenge" not in src.split('"""', 2)[2], \
        "the notify route exempts its own failure from the limiter"


@pytest.mark.parametrize("ts,why", [
    ("nan",  "NaN: every comparison against it is False"),
    ("inf",  "infinity"),
    ("-inf", "negative infinity"),
])
def test_a_timestamp_that_is_not_a_REAL_number_is_refused(store, ts, why):
    """**Found reviewing, 2026-08-27.** The gate read
    `if skew > MAX_SKEW_S: deny`, and `float("nan")` parses fine — so
    `nan > 900` was False and a timestamp of `nan` walked straight through
    the freshness window.

    It needed a valid signature over `"nan"` to exploit, so it was never an
    auth bypass; but a gate with a hole in it is not a gate, and a captured
    report signed that way would have been replayable forever. Asking for
    the good case (`not skew <= MAX`) refuses anything that is not a real,
    small number.
    """
    client, log_root = store
    assert _post(client, ts=ts).status_code == 404, why
    assert _lines(log_root) == []


def test_deny_actually_RAISES(store):
    """Every gate is a bare `_deny()` statement, not `return _deny()`. If it
    ever stopped raising, each one would fall through to the next step — a
    wrong route would go on to be signature-checked, a bad signature would
    go on to be logged. A `NoReturn` annotation says so; this makes it a
    mechanism."""
    from molbuilder.web.blueprints.notify import _deny
    with store[0].application.test_request_context():
        with pytest.raises(Exception) as exc:
            _deny()
        assert "404" in str(exc.value) or "Not Found" in str(exc.value)


def test_the_route_never_REDIRECTS(store):
    """**The trap the Sol egress test surfaced.** A `curl` to the app's root
    answered `302` — the sign-in redirect. If this endpoint ever fell out of
    `auth.py`'s `_PUBLIC_ENDPOINTS` it would answer the same way,
    `urllib.request.urlopen` would FOLLOW it, the POST body would be dropped
    on the way to a login page, and the monitor — which swallows every
    failure by design — would see no error at all. Reports would stop with
    nothing anywhere saying why."""
    client, _ = store
    for r in (_post(client), _post(client, sig="0" * 64),
              _post(client, path="/api/notify")):
        assert not (300 <= r.status_code < 400), \
            f"{r.status_code} redirect: a POST body does not survive one"


# --------------------------------------------------------------------- #
#  it appends, and does nothing else                                     #
# --------------------------------------------------------------------- #

def test_there_is_no_way_to_read_anything_back(store):
    """A route that both takes a credential and serves data is a route where
    a stolen credential reads instead of only writing."""
    client, _ = store
    assert client.get(PATH).status_code in (404, 405)


def test_the_answer_carries_nothing_from_the_payload(store):
    client, _ = store
    r = _post(client, body={"event": "finish", "text": "MARKER-abc123"})
    assert "MARKER-abc123" not in r.get_data(as_text=True)


@pytest.mark.parametrize("raw,why", [
    ("not json at all", "unparseable"),
    ('["a", "list"]',   "not an object"),
    ('"a bare string"', "not an object"),
])
def test_a_body_that_is_not_a_report_is_refused(store, raw, why):
    """Correctly signed, but not a report. Still 404 — the shape of a body
    is not something a stranger should learn either."""
    client, log_root = store
    assert _post(client, raw=raw).status_code == 404, why
    assert _lines(log_root) == []


def test_an_oversized_body_is_refused_before_it_is_parsed(store):
    """A cap on a route fed from the internet is not optional: without one a
    leaked key fills the disk the app runs on."""
    client, log_root = store
    from molbuilder.web.blueprints.notify import MAX_BODY_BYTES
    huge = json.dumps({"event": "x", "text": "A" * (MAX_BODY_BYTES + 100)})
    assert _post(client, raw=huge).status_code == 404
    assert _lines(log_root) == []


def test_only_declared_fields_survive(store):
    """The log is rendered in a browser. An open-ended blob is an open-ended
    rendering problem, so nested structures are dropped, not flattened."""
    client, log_root = store
    _post(client, body={"event": "tick", "text": "ok", "n_iters": 4,
                        "surprise": "not a field",
                        "nested": {"a": [1, 2, 3]}})
    kept = _lines(log_root)[0]
    assert kept["event"] == "tick" and kept["n_iters"] == 4
    assert "surprise" not in kept and "nested" not in kept


def test_a_long_string_is_capped(store):
    client, log_root = store
    _post(client, body={"event": "tick", "text": "B" * 5000})
    assert len(_lines(log_root)[0]["text"]) <= 500


# --------------------------------------------------------------------- #
#  the loop, end to end                                                  #
# --------------------------------------------------------------------- #

def test_the_MONITOR_signs_what_the_LISTENER_verifies(store):
    """The two halves live in different files and ship separately — the
    monitor's copy travels to a compute node as a standalone stdlib-only
    script, so it cannot import the server's `sign` and the server cannot
    import its `sign_report`. **The same rule, written twice, kept in step
    by this test.** A change on one side would show up as reports that
    vanish in silence, because the notifier swallows every failure by
    design."""
    client, log_root = store
    from molbuilder import monitor as M

    sent = {}

    class _Resp:
        def close(self):
            pass

    def _fake_urlopen(req, timeout=None):
        sent["body"] = req.data
        sent["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp()

    hook = M.make_webhook_notifier(f"https://example{PATH}", key=KEY)
    import urllib.request as _u
    real = _u.urlopen
    _u.urlopen = _fake_urlopen
    try:
        hook(M.JobStatus(elapsed_s=90.0, n_iters=7, energy="-1740.2",
                         geom_step=3), "scf_converged")
    finally:
        _u.urlopen = real

    r = client.post(PATH, data=sent["body"], headers={
        "Content-Type": "application/json",
        "X-Molbuilder-Timestamp": sent["headers"]["x-molbuilder-timestamp"],
        "X-Molbuilder-Signature": sent["headers"]["x-molbuilder-signature"]})
    assert r.status_code == 200, r.get_data(as_text=True)
    kept = _lines(log_root)[0]
    assert kept["event"] == "scf_converged"
    assert kept["geom_step"] == 3 and kept["n_iters"] == 7
    assert kept["energy"] == "-1740.2"


def test_the_monitor_does_not_send_the_key_itself(store):
    """**Rule 7, checked on the wire.** What travels is a signature, never
    the secret — that is the whole difference from the bearer token this
    replaced."""
    from molbuilder import monitor as M
    sent = {}

    class _Resp:
        def close(self):
            pass

    def _fake_urlopen(req, timeout=None):
        sent["body"] = req.data
        sent["headers"] = dict(req.header_items())
        return _Resp()

    import urllib.request as _u
    real = _u.urlopen
    _u.urlopen = _fake_urlopen
    try:
        M.make_webhook_notifier("https://example/api/x", key=KEY)(
            M.JobStatus(elapsed_s=1.0), "tick")
    finally:
        _u.urlopen = real

    blob = repr(sent["headers"]) + sent["body"].decode()
    assert KEY not in blob, "the signing key travelled"


# --------------------------------------------------------------------- #
#  the key file itself                                                   #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("body", ["", "not json", "[]", '{"u": 5}', "{}"])
def test_a_broken_key_file_accepts_nothing(tmp_path, monkeypatch, body):
    """A misconfiguration must remove a capability, never grant one
    (rule 1). An unreadable key file means an empty key set, and an empty
    set verifies nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    keys = tmp_path / "notify_keys"
    keys.write_text(body)
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_keys_file": str(keys),
                             "notify_route": ROUTE})
    assert _post(app.test_client()).status_code == 404


def test_a_user_id_that_would_escape_the_log_directory_is_refused(
        tmp_path, monkeypatch):
    """The id becomes a FILENAME. A path separator would write outside the
    log root, and *"the operator would not do that"* is not a mechanism."""
    monkeypatch.setenv("HOME", str(tmp_path))
    keys = tmp_path / "notify_keys"
    keys.write_text(json.dumps({"../../escaped": KEY}))
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_keys_file": str(keys),
                             "notify_route": ROUTE})
    assert _post(app.test_client()).status_code == 404
    assert not (tmp_path / "escaped.jsonl").exists()


def test_the_route_segment_is_validated_as_one_url_segment(tmp_path,
                                                           monkeypatch):
    """A value with a slash in it would silently mean a different path than
    the one written down."""
    from molbuilder.runtime_config import RuntimeConfigError, _normalise
    for bad in ("a/b", "", "has space", "x" * 200, "a.b", 42):
        with pytest.raises(RuntimeConfigError):
            _normalise({"notify_keys_file": "/k", "notify_route": bad})
    # and one that is fine, so the test cannot pass by refusing everything
    ok = _normalise({"notify_keys_file": "/k", "notify_route": "  /x7Kq/  "})
    assert ok["notify_route"] == "x7Kq", "surrounding slashes are trimmed"
