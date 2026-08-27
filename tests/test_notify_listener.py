"""``POST /api/notify`` — the receiving end, and what it refuses.

A job on a cluster POSTs how it is going (`execution/run-reports.md`).  This
is the smallest endpoint in the app on purpose: **it appends one line to a
log and answers ok**, and every test here is about a way it must not do
more than that.

The three rules from `ops/access-control.md` § 8 it is built on:

* **rule 1** — *"the safe state is the one you get by doing nothing"*: with
  no token file configured, there is no route.
* **rule 2** — *"absent beats refused, when existence is itself the
  answer"*: so the path 404s rather than sitting there refusing, and a
  scanner learns nothing about a capability nobody enabled.
* **rule 4** — *"judge behaviour, not people"*: a wrong token is a probe,
  so its 401 must reach the limiter. `auth.py`'s gate exempts its OWN 401
  (an expired session is an ordinary visitor, and counting it once locked a
  user out of their site for an hour) — that exemption must not spread here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from molbuilder.web.app import create_app


TOKEN = "test-token-not-a-real-one"
USER = "someone@example.org"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A configured server, with the log root pointed inside tmp."""
    monkeypatch.setenv("HOME", str(tmp_path))
    tokens = tmp_path / "notify_tokens"
    tokens.write_text(json.dumps({USER: TOKEN}))
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_tokens_file": str(tokens)})
    from molbuilder.web.blueprints import notify as N
    N._loggers.clear()          # rotating handlers are cached per user
    return app.test_client(), tmp_path / ".molbuilder/logs/notify"


def _post(client, token=TOKEN, body=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else json.dumps(
        body if body is not None else {"event": "finish", "text": "done"})
    return client.post("/api/notify", data=data, headers=headers)


def _lines(log_root: Path):
    f = log_root / f"{USER}.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------- #
#  absent beats refused                                                  #
# --------------------------------------------------------------------- #

def test_with_no_token_file_configured_the_route_does_not_exist(tmp_path,
                                                                monkeypatch):
    """Not 401, not 403 — **404**.

    A server whose operator never set this up should not advertise that the
    capability exists.  `access-control.md` § 8 rule 2: *"a capability that
    cannot be exercised safely should not appear. 404 is not rudeness; it is
    the honest statement that there is nothing there."*
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    client = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    r = client.post("/api/notify", data="{}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 404


def test_the_endpoint_is_not_in_the_url_map_at_all(tmp_path, monkeypatch):
    """The stronger form of the above: nothing to reach, rather than
    something that says no."""
    monkeypatch.setenv("HOME", str(tmp_path))
    app = create_app(config={"rate_limit": {"enabled": False}})
    assert not [r for r in app.url_map.iter_rules()
                if str(r) == "/api/notify"]


# --------------------------------------------------------------------- #
#  the token is the claim                                                #
# --------------------------------------------------------------------- #

def test_a_valid_token_is_accepted_and_the_report_is_kept(store):
    client, log_root = store
    r = _post(client, body={"event": "finish", "text": "done", "n_iters": 12})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    kept = _lines(log_root)
    assert len(kept) == 1
    assert kept[0]["event"] == "finish"
    assert kept[0]["n_iters"] == 12
    assert "received_at" in kept[0]


@pytest.mark.parametrize("token,why", [
    (None,            "no header at all"),
    ("",              "empty bearer"),
    ("wrong-token",   "a token nobody holds"),
    (TOKEN[:-1],      "one character short"),
])
def test_anything_but_the_token_is_refused(store, token, why):
    client, log_root = store
    assert _post(client, token=token).status_code == 401, why
    assert _lines(log_root) == [], f"{why} still wrote a record"


def test_the_sender_never_states_who_it_is(store):
    """**The secret is the claim.**  There is no user field to send, so a
    valid token cannot be used to write into somebody else's record —
    which is the point of issuing one per person."""
    client, log_root = store
    _post(client, body={"event": "tick", "user": "someone-else",
                        "text": "hi"})
    kept = _lines(log_root)
    assert len(kept) == 1
    assert "user" not in kept[0], "a payload field became an identity"
    assert (log_root / f"{USER}.jsonl").exists()
    assert not (log_root / "someone-else.jsonl").exists()


def test_a_bad_token_is_counted_by_the_limiter(store):
    """`auth.py`'s gate marks its own 401 as *not evidence* — an expired
    session is an ordinary visitor.  **That reasoning does not carry
    here**, so this route must not set the same flag: nobody reaches it by
    accident, and a wrong token is somebody trying one.
    """
    client, _ = store
    with client.application.test_request_context():
        from flask import g
        _post(client, token="wrong-token")
        assert not getattr(g, "molbuilder_auth_challenge", False)
    # And the source says so, at the one place that could set it.
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify.py").read_text()
    assert "molbuilder_auth_challenge" not in src.split('"""', 2)[2], \
        "the notify route exempts its own 401 from the limiter"


# --------------------------------------------------------------------- #
#  it appends, and does nothing else                                     #
# --------------------------------------------------------------------- #

def test_there_is_no_way_to_read_anything_back(store):
    """No GET.  Reading is a logged-in browser's job on the ordinary tabs;
    a route that both accepts a token and serves data is a route where a
    stolen token reads instead of only writing."""
    client, _ = store
    assert client.get("/api/notify").status_code == 405


def test_the_answer_carries_nothing_from_the_payload(store):
    """An endpoint that echoes is an endpoint that can be used to render
    somebody else's text somewhere it was not expected."""
    client, _ = store
    r = _post(client, body={"event": "finish", "text": "MARKER-abc123"})
    assert "MARKER-abc123" not in r.get_data(as_text=True)


@pytest.mark.parametrize("raw,code,why", [
    ("not json at all",           400, "unparseable"),
    ('["a", "list"]',             400, "not an object"),
    ('"a bare string"',           400, "not an object"),
])
def test_a_body_that_is_not_a_report_is_refused(store, raw, code, why):
    client, log_root = store
    assert _post(client, raw=raw).status_code == code, why
    assert _lines(log_root) == []


def test_an_oversized_body_is_refused_before_it_is_parsed(store):
    """A cap on a route fed from the internet is not optional: without one
    a leaked token fills the disk the app runs on."""
    client, log_root = store
    from molbuilder.web.blueprints.notify import MAX_BODY_BYTES
    huge = json.dumps({"event": "x", "text": "A" * (MAX_BODY_BYTES + 100)})
    assert _post(client, raw=huge).status_code == 413
    assert _lines(log_root) == []


def test_only_declared_fields_survive(store):
    """The log is rendered in a browser.  An open-ended blob is an
    open-ended rendering problem, so the field set is fixed and a nested
    structure is dropped rather than flattened."""
    client, log_root = store
    _post(client, body={"event": "tick", "text": "ok",
                        "surprise": "not a field",
                        "nested": {"a": [1, 2, 3]},
                        "n_iters": 4})
    kept = _lines(log_root)[0]
    assert kept["event"] == "tick" and kept["n_iters"] == 4
    assert "surprise" not in kept
    assert "nested" not in kept


def test_a_long_string_is_capped(store):
    client, log_root = store
    _post(client, body={"event": "tick", "text": "B" * 5000})
    assert len(_lines(log_root)[0]["text"]) <= 500


# --------------------------------------------------------------------- #
#  the loop, end to end                                                  #
# --------------------------------------------------------------------- #

def test_the_body_the_MONITOR_sends_is_one_the_LISTENER_accepts(store):
    """The two halves are written in different files, ship separately, and
    must agree about a wire format.

    The monitor's copy travels to a compute node as a standalone script; a
    field renamed on one side shows up as reports that vanish silently.  So
    this builds the body with the monitor's own notifier and feeds it to
    the real route.
    """
    client, log_root = store
    from molbuilder import monitor as M

    sent = {}

    class _Resp:
        def close(self):
            pass

    def _fake_urlopen(req, timeout=None):
        sent["body"] = req.data
        sent["headers"] = dict(req.header_items())
        return _Resp()

    hook = M.make_webhook_notifier("https://example/api/notify",
                                   {"Authorization": f"Bearer {TOKEN}"})
    import urllib.request as _u
    real = _u.urlopen
    _u.urlopen = _fake_urlopen
    try:
        hook(M.JobStatus(elapsed_s=90.0, n_iters=7, energy="-1740.2",
                         geom_step=3), "scf_converged")
    finally:
        _u.urlopen = real

    auth = next(v for k, v in sent["headers"].items()
                if k.lower() == "authorization")
    r = client.post("/api/notify", data=sent["body"],
                    headers={"Content-Type": "application/json",
                             "Authorization": auth})
    assert r.status_code == 200, r.get_data(as_text=True)
    kept = _lines(log_root)[0]
    assert kept["event"] == "scf_converged"
    assert kept["geom_step"] == 3
    assert kept["n_iters"] == 7
    assert kept["energy"] == "-1740.2"


# --------------------------------------------------------------------- #
#  the token file itself                                                 #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("body", ["", "not json", "[]", '{"u": 5}', '{}'])
def test_a_broken_token_file_accepts_nothing(tmp_path, monkeypatch, body):
    """A misconfiguration must remove a capability, never grant one
    (`access-control.md` § 8 rule 1).  An unreadable token file means an
    empty token set, and an empty set matches nothing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    tokens = tmp_path / "notify_tokens"
    tokens.write_text(body)
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_tokens_file": str(tokens)})
    r = app.test_client().post(
        "/api/notify", data="{}",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 401


def test_a_user_id_that_would_escape_the_log_directory_is_refused(
        tmp_path, monkeypatch):
    """The id becomes a FILENAME.  A path separator in one would write
    outside the log root, and *"the operator would not do that"* is not a
    mechanism."""
    monkeypatch.setenv("HOME", str(tmp_path))
    tokens = tmp_path / "notify_tokens"
    tokens.write_text(json.dumps({"../../escaped": TOKEN}))
    app = create_app(config={"rate_limit": {"enabled": False},
                             "notify_tokens_file": str(tokens)})
    r = app.test_client().post(
        "/api/notify", data="{}",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 401
    assert not (tmp_path / "escaped.jsonl").exists()
