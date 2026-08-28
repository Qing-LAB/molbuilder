"""Setting the destination up from the browser — `run-reports.md` § 3.1.

The format is the hard part. Today you run `notify-token`, copy the JSON,
reach the machine that runs the jobs, `mkdir -p -m 700`, paste, `chmod 600`,
and remember the directory: **four chances to be wrong and every one fails
silently**, because absent or malformed means no notifier, which looks exactly
like never having set it up. A wrong-path defect on 2026-08-27 came from
precisely that.

**This is not the listener.** `notify.py` is the public receiving end — one
route, no session, append-only. This is login-gated, writes a file, and reads
local config. Keeping them apart keeps that boundary readable.

**And it does not touch § 1's split.** That rule is about what *travels*.
Writing the non-travelling half, on the machine it belongs to, is putting a
secret where the contract says it lives.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from molbuilder.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    app = create_app(config={"rate_limit": {"enabled": False}})
    return app.test_client(), tmp_path / ".config/molbuilder/notify"


SECRET = "s3cr3t-value-nobody-should-see"


# --------------------------------------------------------------------- #
#  the secret goes in and never comes back                               #
# --------------------------------------------------------------------- #

def test_the_key_is_never_readable_back(client):
    """**A settings page that can show you a secret is one that can leak
    one**, and there is no reason to: you are setting it, not consulting it.
    The page reports only *whether* a key is present."""
    c, _ = client
    c.post("/api/notify/destination",
           json={"url": "https://x/api/seg", "key": SECRET})
    for resp in (c.get("/api/notify/destination"),
                 c.post("/api/notify/destination",
                        json={"url": "https://x/api/seg", "key": SECRET})):
        assert SECRET not in resp.get_data(as_text=True)
    assert c.get("/api/notify/destination").get_json()["has_key"] is True


def test_the_file_is_written_0600_in_a_0700_directory(client):
    """Through `write_secret_file`, which sets the mode on the descriptor
    **before the first byte** — so the secret is never briefly on disk at a
    looser one. That was a real window until 2026-08-27."""
    c, path = client
    c.post("/api/notify/destination",
           json={"url": "https://x/api/seg", "key": SECRET})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text())["key"] == SECRET


def test_it_writes_where_the_MONITOR_reads(client):
    """The one property that makes this worth having. The card once said
    `~/.molbuilder/notify` while the monitor read `config_dir()/notify`, and
    following it put the file where nothing looks — silently.

    The path comes from the monitor's own function, so the page and the
    process that reads the file on a compute node cannot disagree.
    """
    from molbuilder.monitor import default_notify_path
    c, path = client
    c.post("/api/notify/destination", json={"url": "https://x/api/seg"})
    assert Path(c.get("/api/notify/destination").get_json()["path"]) \
        == default_notify_path() == path
    assert path.exists()


# --------------------------------------------------------------------- #
#  what it refuses, and what it reports                                  #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("body,why", [
    ({}, "no url"),
    ({"url": ""}, "empty url"),
    ({"url": "devbox:8888/api/x"}, "no scheme -- would not parse as a url"),
    ({"url": "ftp://x/api"}, "a scheme nothing here speaks"),
])
def test_a_url_that_could_not_work_is_refused(client, body, why):
    c, path = client
    assert c.post("/api/notify/destination", json=body).status_code == 400, why
    assert not path.exists(), f"{why} still wrote a file"


def test_a_malformed_file_is_REPORTED_not_hidden(client):
    """Absent and malformed both mean no notifier, and they look identical
    from the outside — which is the whole reason this page exists. It says
    which one it found."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    got = c.get("/api/notify/destination").get_json()
    assert got["configured"] is False
    assert "not valid JSON" in got["problem"]


def test_a_key_is_optional_because_slack_needs_none(client):
    """Slack and Discord put the credential in the URL itself. A page that
    demanded a key would refuse a destination the contract explicitly
    supports (§ 3)."""
    c, path = client
    r = c.post("/api/notify/destination",
               json={"url": "https://hooks.slack.com/services/XXX"})
    assert r.status_code == 200
    assert r.get_json()["has_key"] is False
    assert "key" not in json.loads(path.read_text())


def test_absent_is_reachable_without_a_shell(client):
    """**Absent is off**, and off is a state a person is entitled to reach
    from the page that set it up."""
    c, path = client
    c.post("/api/notify/destination", json={"url": "https://x/api/seg"})
    assert path.exists()
    assert c.delete("/api/notify/destination").get_json()["configured"] is False
    assert not path.exists()
    # and deleting nothing is not an error
    assert c.delete("/api/notify/destination").status_code == 200


# --------------------------------------------------------------------- #
#  which machine runs the jobs decides what the page can do               #
# --------------------------------------------------------------------- #

def test_submit_mode_says_it_cannot_write_the_far_side(tmp_path, monkeypatch):
    """`submit` means the job runs somewhere this server cannot reach, so
    the most the page can do is hand you the exact content. Saying so is
    the difference between a page that helps and one that lies."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    cfgdir = tmp_path / ".config/molbuilder"
    cfgdir.mkdir(parents=True)
    (cfgdir / "molbuilder.json").write_text(
        json.dumps({"execution": {"mode": "submit"}}))
    monkeypatch.chdir(tmp_path)
    c = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    got = c.get("/api/notify/destination").get_json()
    assert got["execution_mode"] == "submit"
    assert got["can_write_here"] is False


# --------------------------------------------------------------------- #
#  the test report — the only check that exercises the whole path         #
# --------------------------------------------------------------------- #

def test_testing_with_no_destination_says_so(client):
    c, _ = client
    r = c.post("/api/notify/destination/test")
    assert r.status_code == 400
    assert "no destination" in r.get_json()["error"]


def test_the_test_report_is_signed_by_the_MONITORS_own_function(client):
    """A second signing implementation here could pass while the real one
    failed. It signs with `monitor.sign_report`, so a signature this
    produces is one the listener accepts."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify_setup.py").read_text()
    assert "from ...monitor import load_destination, sign_report" in src
    assert "sign_report(dest[" in src and "], ts, body)" in src


def test_a_404_is_explained_rather_than_reported_bare(client, monkeypatch):
    """**The listener refuses everything the same way** so a stranger cannot
    tell the gates apart — which means it cannot tell YOU apart either. A
    bare "404" would send a person hunting the wrong thing, so all the
    possibilities are named."""
    c, _ = client
    c.post("/api/notify/destination",
           json={"url": "https://x/api/seg", "key": SECRET})
    import urllib.error
    import urllib.request

    def _404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _404)
    got = c.post("/api/notify/destination/test").get_json()
    assert got["ok"] is False and got["status"] == 404
    for phrase in ("route segment", "key"):
        assert phrase in got["hint"], f"the hint does not mention {phrase}"


def test_unreachable_is_distinguished_from_refused(client, monkeypatch):
    """*Could not connect* and *was refused* send a person to two completely
    different places, so they are two different answers."""
    c, _ = client
    c.post("/api/notify/destination", json={"url": "https://x/api/seg"})
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no route")))
    got = c.post("/api/notify/destination/test").get_json()
    assert got["reached"] is False
    assert "could not reach it" in got["error"]


def test_a_url_only_save_KEEPS_the_key(client):
    """**Two correct decisions making a trap between them, found by a round
    trip on 2026-08-27.**

    The card clears the key field after every save — correctly, because a
    secret left in the DOM is one that ends up in a screenshot. So the
    ordinary next action, fixing a typo in the address and saving again,
    arrived with no key and wiped the one on disk. Reports then stop, and
    **silently**: an unsigned report gets the listener's 404 and the
    notifier swallows it.

    An empty key means *leave it alone*. Dropping one deliberately —
    switching to a Slack url that needs none — is Remove and then save.
    """
    c, path = client
    c.post("/api/notify/destination",
           json={"url": "https://a/api/seg", "key": SECRET})
    c.post("/api/notify/destination", json={"url": "https://b/api/seg"})
    doc = json.loads(path.read_text())
    assert doc["url"] == "https://b/api/seg", "the url did not change"
    assert doc["key"] == SECRET, "the key was destroyed by a url-only save"
    assert c.get("/api/notify/destination").get_json()["has_key"] is True


def test_a_new_key_still_replaces_the_old_one(client):
    """*Leave it alone* must not become *cannot be changed*."""
    c, path = client
    c.post("/api/notify/destination",
           json={"url": "https://a/api/seg", "key": SECRET})
    c.post("/api/notify/destination",
           json={"url": "https://a/api/seg", "key": "a-different-key"})
    assert json.loads(path.read_text())["key"] == "a-different-key"


def test_remove_then_save_is_how_a_key_is_DROPPED(client):
    """The deliberate path, since a blank field no longer means delete."""
    c, path = client
    c.post("/api/notify/destination",
           json={"url": "https://a/api/seg", "key": SECRET})
    c.delete("/api/notify/destination")
    c.post("/api/notify/destination",
           json={"url": "https://hooks.slack.com/services/XXX"})
    doc = json.loads(path.read_text())
    assert "key" not in doc
    assert c.get("/api/notify/destination").get_json()["has_key"] is False


@pytest.mark.parametrize("verb", ["get", "post", "delete"])
def test_no_verb_returns_a_private_field(client, verb):
    """**One door out** (`_public`) strips anything underscored, so a route
    cannot forget to."""
    c, _ = client
    c.post("/api/notify/destination",
           json={"url": "https://a/api/seg", "key": SECRET})
    r = getattr(c, verb)("/api/notify/destination",
                         json={"url": "https://a/api/seg"} if verb == "post"
                         else None)
    body = r.get_json()
    assert not [k for k in body if k.startswith("_")], body
    assert SECRET not in r.get_data(as_text=True)


def test_a_save_UPDATES_it_does_not_replace(client):
    """**The general form of the key bug, found by widening the review.**

    Writing a fresh `{url, key}` destroyed everything else the file held.
    A `headers` block — which `monitor.load_destination` reads and this
    page has no input for — vanished the first time anybody edited the url
    here, exactly as the key did.

    So a save writes the fields it manages over whatever is there. That
    also means a field added to this file LATER cannot be dropped by a page
    that predates it, which is the property worth having rather than a
    second special case.
    """
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "url": "https://a/x", "key": SECRET,
        "headers": {"Authorization": "Bearer T"},
        "a_field_this_page_never_heard_of": 42}))
    c.post("/api/notify/destination", json={"url": "https://a/CHANGED"})
    doc = json.loads(path.read_text())
    assert doc["url"] == "https://a/CHANGED", "the url did not update"
    assert doc["key"] == SECRET
    assert doc["headers"] == {"Authorization": "Bearer T"}
    assert doc["a_field_this_page_never_heard_of"] == 42


def test_a_malformed_file_can_still_be_FIXED_from_the_page(client):
    """Preserving what is there must not mean being stuck with a file that
    cannot be parsed. Unreadable reads as empty, so a save writes a fresh
    valid one rather than refusing."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")
    r = c.post("/api/notify/destination",
               json={"url": "https://a/api/seg", "key": SECRET})
    assert r.status_code == 200
    assert json.loads(path.read_text())["key"] == SECRET
    assert c.get("/api/notify/destination").get_json()["problem"] == ""
