"""Setting this machine up from the browser — `this-machine.md`.

The format is the hard part. The alternative is running `notify-token`,
copying the JSON, reaching the machine that runs the jobs, `mkdir -p -m 700`,
pasting, `chmod 600`, and remembering the directory: **four chances to be
wrong and every one fails silently**, because absent or malformed means no
notifier, which looks exactly like never having set it up. A wrong-path defect
on 2026-08-27 came from precisely that.

**This is not the listener.** `notify.py` is the public receiving end — one
route, no session, append-only. This is login-gated, writes files, and reads
local config. Keeping them apart keeps that boundary readable.

**And it does not touch § 1's split.** That rule is about what *travels*.
Writing the non-travelling half, on the machine it belongs to, is putting a
secret where the contract says it lives.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from molbuilder.web.app import create_app

CH = "/api/notify/channels"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    app = create_app(config={"rate_limit": {"enabled": False}})
    return app.test_client(), tmp_path / ".config/molbuilder/notify"


SECRET = "s3cr3t-value-nobody-should-see"
#: A Slack-shaped address.  The tail is what a masked view keeps and the rest
#: is what it must drop -- for this kind the URL IS the credential.
WEBHOOK = "https://hooks.slack.com/services/T0/B0/AbCdEfGhIjKlMnOp"


def _row(c, name):
    for r in c.get(CH).get_json()["channels"]:
        if r["name"] == name:
            return r
    return None


# --------------------------------------------------------------------- #
#  the secret goes in and never comes back                               #
# --------------------------------------------------------------------- #

def test_the_key_is_never_readable_back(client):
    """**A settings page that can show you a secret is one that can leak
    one**, and there is no reason to: you are setting it, not consulting it.
    The page reports only *whether* a key is present."""
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg", "key": SECRET})
    for resp in (c.get(CH),
                 c.put(CH + "/lab",
                       json={"url": "https://x/api/seg", "key": SECRET})):
        assert SECRET not in resp.get_data(as_text=True)
    assert _row(c, "lab")["has_key"] is True


def test_a_webhook_address_is_masked_because_it_IS_the_credential(client):
    """**The rule that was missing until 2026-08-31.** The route this
    replaced returned every stored URL in full on the strength of *"an
    address, not a secret"* — true of the listener URL it was written for,
    and false of the Slack webhook actually in the file, which anyone signed
    in could then read back out.

    Enough survives to tell two webhooks apart; nowhere near enough to use.
    """
    c, _ = client
    c.put(CH + "/slack", json={"url": WEBHOOK})
    r = c.get(CH)
    # THE STATUS FIRST.  A 404 would make every `not in body` below pass
    # against Flask's error page, pinning nothing -- which is what
    # `test_negative_body_assert_lint` exists to catch, and did (2026-09-01).
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "T0/B0" not in body, "the secret path is still being handed out"
    assert WEBHOOK not in body
    row = _row(c, "slack")
    assert "hooks.slack.com" in row["where"], "which service is not a secret"
    assert row["where"].endswith("MnOp"), "a tail, so two can be told apart"


def test_every_address_is_masked_even_a_listeners(client):
    """**Masking only the kind that needs it is a rule that can be
    defeated.** The kind is derived from whether a key is stored, so a Slack
    url saved with a key in the box would be classed a listener and printed
    whole — the exact secret this is protecting.

    A listener address loses nothing by it: the tail still names the segment,
    the listener section shows the route in full, and an address is proved by
    testing rather than by reading.
    """
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://qlab:8888/api/GfmVt99",
                             "key": SECRET})
    c.put(CH + "/mislabelled", json={"url": WEBHOOK, "key": SECRET})
    assert _row(c, "lab")["where"] == "https://qlab:8888/…Vt99"
    r = c.get(CH)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "T0/B0" not in body, "a mislabelled webhook was printed whole"


def test_the_kind_this_page_shows_is_the_one_the_SENDER_will_use(client):
    """**Replaced the test that asserted the opposite** (2026-09-02).

    It read `test_the_kind_is_derived_from_the_key_not_stored_beside_it`, on
    the rule that *having a key* is the one thing that differs.  That held
    while there were two kinds.  There are three wire formats now
    (`run-reports.md` § 4.1b) and having a key cannot tell Slack from
    Discord -- so the derivation collapsed the one distinction the sender
    cannot avoid making, and a Discord channel was saved, listed and tested
    as though it were a Slack one.

    The rule now: this page reports `monitor.channel_kind`, the SAME
    function that chooses the envelope.  A test that pinned the old
    derivation would fail the next time the right thing is done, which is
    what it just did."""
    from molbuilder.monitor import channel_kind
    c, path = client
    c.put(CH + "/team", json={"url": WEBHOOK})
    c.put(CH + "/lab", json={"url": "https://qlab/api/x", "key": SECRET})
    c.put(CH + "/chat", json={"url": "https://discord.com/api/webhooks/1/tok",
                              "kind": "discord"})
    assert _row(c, "team")["kind"] == "slack"       # off the host
    assert _row(c, "lab")["kind"] == "molbuilder"
    assert _row(c, "chat")["kind"] == "discord"     # declared
    # And what the page shows is what the SENDER reads -- one function, not
    # a page-side lookalike.
    stored = json.loads(path.read_text())["channels"]
    for name in ("team", "lab", "chat"):
        assert _row(c, name)["kind"] == channel_kind(stored[name])


def test_a_declared_kind_is_stored_and_a_cleared_one_is_removed(client):
    """A channel edited from Discord to Slack must not keep the old
    envelope with the new address -- so an absent `kind` CLEARS a stored
    one rather than leaving it standing."""
    c, path = client
    c.put(CH + "/x", json={"url": "https://relay.example/hook",
                           "kind": "discord"})
    assert json.loads(path.read_text())["channels"]["x"]["kind"] == "discord"
    c.put(CH + "/x", json={"url": "https://relay.example/hook"})
    assert "kind" not in json.loads(path.read_text())["channels"]["x"]
    assert _row(c, "x")["kind"] == "molbuilder"     # back to the host's word


def test_a_misspelled_kind_is_refused_by_name(client):
    """Named and wrong is not the same as absent: absent means *read it off
    the host*, and a typo taking that default would send a Slack-shaped body
    to Discord and earn a 400 nobody could trace back to a spelling."""
    c, _ = client
    r = c.put(CH + "/x", json={"url": WEBHOOK, "kind": "discrod"})
    assert r.status_code == 400
    assert "discord" in r.get_json()["error"]


def test_the_file_is_written_0600_in_a_0700_directory(client):
    """Through `write_secret_file`, which sets the mode on the descriptor
    **before the first byte** — so the secret is never briefly on disk at a
    looser one. That was a real window until 2026-08-27."""
    c, path = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg", "key": SECRET})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text())["channels"]["lab"]["key"] == SECRET


def test_it_writes_where_the_MONITOR_reads(client):
    """The one property that makes this worth having. The card once said
    `~/.molbuilder/notify` while the monitor read `config_dir()/notify`, and
    following it put the file where nothing looks — silently.

    The path comes from the monitor's own function, so the page and the
    process that reads the file on a compute node cannot disagree.
    """
    from molbuilder.monitor import default_notify_path, load_channels
    c, path = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg", "key": SECRET})
    assert Path(c.get(CH).get_json()["path"]) == default_notify_path() == path
    assert path.exists()
    # AND IT PARSES AS WHAT THE MONITOR EXPECTS.  Same directory but a shape
    # the reader rejects would fail in exactly the same silence.
    assert list(load_channels(str(path))) == ["lab"]


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
    assert c.put(CH + "/lab", json=body).status_code == 400, why
    assert not path.exists(), f"{why} still wrote a file"


@pytest.mark.parametrize("name", ["has space", "a/b", "", "x" * 65, "a.b"])
def test_a_name_that_could_not_travel_is_refused(client, name):
    """A name is written into a description and read back out of one, and it
    is rendered into the monitor's command line. Anything outside the rule
    would mean one thing in the file and another in the shell."""
    c, path = client
    r = c.put(CH + "/" + name, json={"url": "https://x/api/seg"})
    assert r.status_code in (400, 404, 405), name
    assert not path.exists()


def test_a_malformed_file_is_REPORTED_not_hidden(client):
    """Absent and malformed both mean no notifier, and they look identical
    from the outside — which is the whole reason this page exists. It says
    which one it found."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    got = c.get(CH).get_json()
    assert got["channels"] == []
    assert "not valid JSON" in got["problem"]


def test_the_old_single_destination_file_is_named_as_such(client):
    """`{"url": ...}` parses, so "no channels" would be indistinguishable
    from an empty machine. The page says what it is looking at."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": WEBHOOK, "key": SECRET}))
    got = c.get(CH).get_json()
    assert "old single-destination file" in got["problem"]
    assert SECRET not in c.get(CH).get_data(as_text=True)


def test_a_key_is_optional_because_slack_needs_none(client):
    """Slack and Discord put the credential in the URL itself. A page that
    demanded a key would refuse a channel the contract explicitly
    supports (`run-reports.md` § 3)."""
    c, path = client
    r = c.put(CH + "/slack", json={"url": WEBHOOK})
    assert r.status_code == 200
    assert _row(c, "slack")["has_key"] is False
    assert "key" not in json.loads(path.read_text())["channels"]["slack"]


def test_absent_is_reachable_without_a_shell(client):
    """**Absent is off**, and off is a state a person is entitled to reach
    from the page that set it up."""
    c, path = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg"})
    assert path.exists()
    assert c.delete(CH + "/lab").get_json()["channels"] == []
    # and deleting nothing is not an error
    assert c.delete(CH + "/lab").status_code == 200


# --------------------------------------------------------------------- #
#  the merges — across channels, and within one                          #
# --------------------------------------------------------------------- #

def test_saving_one_channel_leaves_the_others_alone(client):
    """**The reason the file is a map at all.** One destination per machine
    meant configuring Slack silently replaced the listener you were using
    (user, 2026-08-31). A whole-file write here would reintroduce that as a
    data loss rather than a limitation."""
    c, path = client
    c.put(CH + "/lab", json={"url": "https://qlab/api/x", "key": SECRET})
    c.put(CH + "/slack", json={"url": WEBHOOK})
    stored = json.loads(path.read_text())["channels"]
    assert sorted(stored) == ["lab", "slack"]
    assert stored["lab"]["key"] == SECRET


def test_a_url_only_save_KEEPS_the_key(client):
    """**Two correct decisions making a trap between them, found by a round
    trip on 2026-08-27.**

    The page clears the key field after every save — correctly, because a
    secret left in the DOM is one that ends up in a screenshot. So the
    ordinary next action, fixing a typo in the address and saving again,
    arrived with no key and wiped the one on disk. Reports then stop, and
    **silently**: an unsigned report gets the listener's 404 and the
    notifier swallows it.

    An empty key means *leave it alone*. Dropping one deliberately —
    switching to a Slack url that needs none — is Remove and then save.
    """
    c, path = client
    c.put(CH + "/lab", json={"url": "https://a/api/seg", "key": SECRET})
    c.put(CH + "/lab", json={"url": "https://b/api/seg"})
    doc = json.loads(path.read_text())["channels"]["lab"]
    assert doc["url"] == "https://b/api/seg", "the url did not change"
    assert doc["key"] == SECRET, "the key was destroyed by a url-only save"
    assert _row(c, "lab")["has_key"] is True


def test_a_new_key_still_replaces_the_old_one(client):
    """*Leave it alone* must not become *cannot be changed*."""
    c, path = client
    c.put(CH + "/lab", json={"url": "https://a/api/seg", "key": SECRET})
    c.put(CH + "/lab", json={"url": "https://a/api/seg", "key": "another"})
    assert json.loads(path.read_text())["channels"]["lab"]["key"] == "another"


def test_remove_then_save_is_how_a_key_is_DROPPED(client):
    """The deliberate path, since a blank field no longer means delete."""
    c, path = client
    c.put(CH + "/lab", json={"url": "https://a/api/seg", "key": SECRET})
    c.delete(CH + "/lab")
    c.put(CH + "/lab", json={"url": WEBHOOK})
    assert "key" not in json.loads(path.read_text())["channels"]["lab"]
    assert _row(c, "lab")["has_key"] is False


def test_a_save_UPDATES_it_does_not_replace(client):
    """**The general form of the key bug, found by widening the review.**

    Writing a fresh `{url, key}` destroyed everything else the channel held.
    A `headers` block — which `monitor.load_channels` reads and this page
    has no input for — vanished the first time anybody edited the url here,
    exactly as the key did.

    So a save writes the fields it manages over whatever is there. That
    also means a field added to this file LATER cannot be dropped by a page
    that predates it, which is the property worth having rather than a
    second special case.
    """
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"channels": {"lab": {
        "url": "https://a/x", "key": SECRET,
        "headers": {"Authorization": "Bearer T"},
        "a_field_this_page_never_heard_of": 42}}}))
    c.put(CH + "/lab", json={"url": "https://a/CHANGED"})
    doc = json.loads(path.read_text())["channels"]["lab"]
    assert doc["url"] == "https://a/CHANGED", "the url did not update"
    assert doc["key"] == SECRET
    assert doc["headers"] == {"Authorization": "Bearer T"}
    assert doc["a_field_this_page_never_heard_of"] == 42


def test_saving_over_the_old_file_CLEARS_the_retired_top_level_keys(client):
    """The single-destination file kept `url` / `key` / `headers` at the top
    level. Once the file has a `channels` map nothing reads them, so leaving
    them behind leaves a live credential in a file whose whole purpose is
    holding one deliberately — and makes the page's own message ("save a
    channel below and it becomes a named one") false.

    Narrow on purpose: the merge rule says a field added LATER survives a
    page that predates it, and it still does. These three are not unknown
    fields, they are the previous format's, by name.
    """
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "url": WEBHOOK, "key": SECRET,
        "headers": {"Authorization": "Bearer T"}}))
    c.put(CH + "/slack", json={"url": WEBHOOK})
    doc = json.loads(path.read_text())
    assert list(doc) == ["channels"], f"the retired shape survived: {doc}"
    assert SECRET not in path.read_text()
    assert c.get(CH).get_json()["problem"] == "", \
        "the page still calls it the old file after converting it"


def test_a_field_this_page_never_heard_of_still_survives(client):
    """The other side of the same coin, and the reason the clear above is
    keyed by name rather than by "anything not `channels`"."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"channels": {},
                                "something_added_later": 42}))
    c.put(CH + "/slack", json={"url": WEBHOOK})
    assert json.loads(path.read_text())["something_added_later"] == 42


def test_a_changed_address_drops_the_old_verdict(client):
    """A green tick beside a channel nobody has reached at its new address
    is the page lying about the one thing it exists to prove."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"channels": {"lab": {
        "url": "https://a/x", "tested_ok": True, "tested_at": 1.0}}}))
    c.put(CH + "/lab", json={"url": "https://a/CHANGED"})
    assert _row(c, "lab")["tested_ok"] is None


def test_a_malformed_file_can_still_be_FIXED_from_the_page(client):
    """Preserving what is there must not mean being stuck with a file that
    cannot be parsed. Unreadable reads as empty, so a save writes a fresh
    valid one rather than refusing."""
    c, path = client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")
    r = c.put(CH + "/lab", json={"url": "https://a/api/seg", "key": SECRET})
    assert r.status_code == 200
    assert json.loads(path.read_text())["channels"]["lab"]["key"] == SECRET
    assert c.get(CH).get_json()["problem"] == ""


# --------------------------------------------------------------------- #
#  it writes here, always                                                 #
# --------------------------------------------------------------------- #

def test_execution_mode_does_not_gate_the_write(tmp_path, monkeypatch):
    """A `submit` machine is a machine with a scheduler
    (`running-a-job.md` § 5.4) -- a login node, which is exactly where the
    file belongs.  The route read it as *"the jobs run somewhere I cannot
    reach"* and refused to save from 2026-08-27 until 2026-09-01.

    **Every config file molbuilder manages is saved on the machine molbuilder
    runs on** (user, 2026-09-01).  Nothing about the launch mode changes that.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    cfgdir = tmp_path / ".config/molbuilder"
    cfgdir.mkdir(parents=True)
    (cfgdir / "molbuilder.json").write_text(
        json.dumps({"execution": {"mode": "submit"}}))
    monkeypatch.chdir(tmp_path)
    c = create_app(config={"rate_limit": {"enabled": False}}).test_client()
    r = c.put(CH + "/lab", json={"url": "https://qlab/api/x", "key": SECRET})
    assert r.status_code == 200, "a scheduler on this box refused the write"
    assert (cfgdir / "notify").exists()
    body = c.get(CH).get_json()
    assert body["channels"][0]["name"] == "lab"
    for gone in ("execution_mode", "can_write_here"):
        assert gone not in body, f"the route still reports {gone}"



# --------------------------------------------------------------------- #
#  the test report — the only check that exercises the whole path         #
# --------------------------------------------------------------------- #

def test_testing_a_channel_that_is_not_set_up_says_so(client):
    c, _ = client
    r = c.post(CH + "/nope/test")
    assert r.status_code == 404
    assert "nope" in r.get_json()["error"]


def test_the_test_button_sends_what_the_MONITOR_would_send(client,
                                                           monkeypatch):
    """**A second implementation here could pass while the real one failed**
    -- and for Discord it did, twice over: no `User-Agent` (403 at
    Cloudflare) and no `content`/`embeds` (400 at Discord).

    This asserted a SOURCE STRING until 2026-09-02, so it broke on a
    refactor that kept the rule and would have passed on a copy-paste that
    broke it.  It now sends for real, to a recording stand-in, and compares
    the bytes against `monitor.webhook_request` -- the producer both senders
    call (`run-reports.md` § 4.1b)."""
    import urllib.request
    from molbuilder import monitor as M

    c, _ = client
    c.put(CH + "/chat", json={"url": "https://discord.com/api/webhooks/1/t",
                              "kind": "discord"})
    c.put(CH + "/lab", json={"url": "https://qlab/api/seg", "key": SECRET})

    seen = {}

    class _Resp:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def close(self): pass

    def _fake(req, timeout=None, **kw):
        seen["url"] = req.full_url
        seen["body"] = req.data
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake)

    # -- the Discord channel: an embed, and a real User-Agent --------------
    assert c.post(CH + "/chat/test").status_code == 200
    body = json.loads(seen["body"])
    assert "embeds" in body, (
        "Discord ignores a bare `text` and refuses a body with neither "
        "`content` nor `embeds`: " + repr(sorted(body)))
    ua = seen["headers"].get("User-agent".lower()) or ""
    assert ua and "python-urllib" not in ua.lower(), (
        "Discord's edge answers a default urllib User-Agent with 403 "
        "(Cloudflare 1010): " + repr(ua))
    assert "x-molbuilder-signature" not in seen["headers"], (
        "a signature means nothing to Discord and is not sent there")

    # -- the listener: the record, whole, signed by the monitor's own rule -
    assert c.post(CH + "/lab/test").status_code == 200
    rec = json.loads(seen["body"])
    assert rec["state"] == "test" and rec["run"] == "notify-setup-test"
    ts = seen["headers"]["x-molbuilder-timestamp"]
    assert (seen["headers"]["x-molbuilder-signature"]
            == M.sign_report(SECRET, ts, seen["body"])), (
        "the signature is not the one the listener verifies")


def test_a_404_is_explained_rather_than_reported_bare(client, monkeypatch):
    """**The listener refuses everything the same way** so a stranger cannot
    tell the gates apart — which means it cannot tell YOU apart either. A
    bare "404" would send a person hunting the wrong thing, so all the
    possibilities are named."""
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg", "key": SECRET})
    import urllib.error
    import urllib.request

    def _404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _404)
    got = c.post(CH + "/lab/test").get_json()
    assert got["ok"] is False and got["status"] == 404
    for phrase in ("route segment", "key"):
        assert phrase in got["hint"], f"the hint does not mention {phrase}"


def test_unreachable_is_distinguished_from_refused(client, monkeypatch):
    """*Could not connect* and *was refused* send a person to two completely
    different places, so they are two different answers."""
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg"})
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no route")))
    got = c.post(CH + "/lab/test").get_json()
    assert got["reached"] is False
    assert "could not reach it" in got["error"]


def test_the_verdict_is_remembered_so_task_setup_can_show_it(client,
                                                             monkeypatch):
    """A name with no evidence behind it is exactly the silent failure this
    area keeps producing. The tick list shows what the last test said."""
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://x/api/seg"})
    import urllib.request

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert c.post(CH + "/lab/test").get_json()["ok"] is True
    row = _row(c, "lab")
    assert row["tested_ok"] is True and row["tested_at"]


# --------------------------------------------------------------------- #
#  every answer, from every verb, is safe to look at                     #
# --------------------------------------------------------------------- #

class _Ok:
    """A urlopen answer that succeeded, as a context manager."""
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("verb,path,body,reachable", [
    ("get",    CH,               None, None),
    ("put",    CH + "/lab",      {"url": "https://a/api/seg"}, None),
    ("delete", CH + "/other",    None, None),
    # BOTH BRANCHES of the test route, and it has to be both: a fake that
    # only ever throws exercises the failure return and leaves the success
    # return -- the one that was actually wrong -- untouched.  Mutation
    # testing caught exactly that on 2026-08-31.
    ("post",   CH + "/lab/test", None, True),
    ("post",   CH + "/lab/test", None, False),
])
def test_every_answer_carries_the_WHOLE_state(client, verb, path, body,
                                              reachable, monkeypatch):
    """**Found in the browser, 2026-08-31.**

    The page repaints from whatever the response carries, so a mutation that
    replied with a narrower object left the painter reading fields that were
    not there: after a test the card read *"2 channels in undefined"*,
    because `path` was in the GET's answer and in no other.

    A response is the state or it is a trap for the next painter, so every
    route answers through one function.
    """
    import urllib.request
    if reachable is True:
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Ok())
    elif reachable is False:
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no route")))
    c, _ = client
    c.put(CH + "/lab", json={"url": "https://a/api/seg"})
    c.put(CH + "/other", json={"url": WEBHOOK})
    got = getattr(c, verb)(path, json=body).get_json()
    for field in ("path", "channels", "problem", "mode"):
        assert field in got, f"{verb} {path} omitted {field!r}"


@pytest.mark.parametrize("verb,path,body", [
    ("get",    CH,             None),
    ("put",    CH + "/lab",    {"url": "https://a/api/seg"}),
    ("delete", CH + "/other",  None),
    ("post",   CH + "/lab/test", None),
])
def test_no_verb_hands_out_a_secret(client, verb, path, body):
    """**One door out** (`_row`), so a route cannot forget. Every response
    goes through it, including the ones that mutate."""
    c, _ = client
    c.put(CH + "/lab", json={"url": WEBHOOK, "key": SECRET})
    c.put(CH + "/other", json={"url": WEBHOOK})
    text = getattr(c, verb)(path, json=body).get_data(as_text=True)
    assert SECRET not in text
    assert "T0/B0" not in text, "an unmasked webhook path escaped"


# --------------------------------------------------------------------- #
#  the listener half — the round trip that needed a terminal              #
# --------------------------------------------------------------------- #

def test_the_listener_reports_absent_before_any_key_exists(client):
    c, _ = client
    got = c.get("/api/notify/listener").get_json()
    assert got["configured"] is False and got["users"] == []


def test_issuing_a_key_returns_it_once_and_never_again(client, tmp_path):
    """A key that is never shown at the moment it is made cannot reach the
    machine it is for. A key readable afterwards is a leak with no purpose.
    Issuing and displaying are different acts (`this-machine.md` § 2)."""
    c, _ = client
    got = c.post("/api/notify/listener/keys/alice").get_json()
    assert got["ok"] is True and got["key"]
    key = got["key"]
    later = c.get("/api/notify/listener")
    assert key not in later.get_data(as_text=True)
    assert later.get_json()["users"] == ["alice"]
    assert later.get_json()["route"] == got["route"]


def test_a_second_key_joins_the_route_already_in_the_file(client):
    """A new segment would move the route out from under everybody already
    set up — silently, since a notifier swallows failures. That hazard was
    the duplication, not a step people kept getting wrong."""
    c, _ = client
    first = c.post("/api/notify/listener/keys/alice").get_json()
    second = c.post("/api/notify/listener/keys/bob").get_json()
    assert second["route"] == first["route"]
    assert second["joined"] is True
    assert c.get("/api/notify/listener").get_json()["users"] == ["alice", "bob"]


def test_reissuing_without_replace_is_refused(client):
    """Rotation stops the old key immediately and any job still running with
    it goes silent, so it is never something you do by accident."""
    c, _ = client
    c.post("/api/notify/listener/keys/alice")
    r = c.post("/api/notify/listener/keys/alice")
    assert r.status_code == 400 and "replace" in r.get_json()["error"]
    ok = c.post("/api/notify/listener/keys/alice", json={"replace": True})
    assert ok.status_code == 200


def test_a_user_id_that_could_not_be_a_log_filename_is_refused(client):
    """It becomes a filename in the report log, so it is bounded at the
    moment it is issued rather than at the moment it is written."""
    c, _ = client
    r = c.post("/api/notify/listener/keys/" + "a/b")
    assert r.status_code in (400, 404, 405)


def test_the_web_and_the_cli_issue_through_one_door(client):
    """Two implementations would be free to generate two route segments from
    one file — the failure `run-reports.md` § 4.3 records from when the route
    lived in two places."""
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/web/blueprints/notify_setup.py").read_text()
    cli = (Path(__file__).resolve().parents[1]
           / "molbuilder/cli.py").read_text()
    assert "issue_notify_key" in src and "issue_notify_key" in cli


def test_configured_but_not_live_is_a_state_the_page_can_report(client):
    """The route is registered at startup from the key file, so the FIRST
    key ever issued does not open it until then. A person watching a 404
    deserves to be told, rather than left to guess between the four things a
    404 can mean."""
    c, _ = client
    c.post("/api/notify/listener/keys/alice")
    got = c.get("/api/notify/listener").get_json()
    assert got["configured"] is True
    assert got["live"] is False, "this app started before the file existed"
