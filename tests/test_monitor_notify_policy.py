"""How often the monitor LOOKS and how often it TELLS you are not the same
-- `execution/run-reports.md` § 2.1.

**The defect this closes.** The monitor wakes on its interval, samples
CPU/GPU/memory, and — in the same pass — called every registered notifier,
guarded only by *did anything change*.  On a running job the SCF iteration
count advances constantly, so that was true almost every wake.  A Slack
webhook configured against it would have received a message **every ten
seconds for the length of the run**.

Sampling stays dense: `util.csv` is the diagnostic record and its whole
point is showing whether the CPU/GPU/memory allocation is being used.
Notifying becomes policy (`archive/2026-09-01-bench-and-junction-plan.md` § 2.9), and the
policy is the calculation's own — carried from `task.json`'s `notify` block.

**The destination is not policy and does not live here.**  It is the user's
own file on the machine that runs the job, because `task.json` travels and a
token must not travel with it.
"""
from __future__ import annotations

import json
import os

import pytest

from molbuilder import monitor as M


# --------------------------------------------------------------------- #
#  the destination -- the user's file, never the description              #
# --------------------------------------------------------------------- #

def _file(tmp_path, monkeypatch, channels):
    """The channel file AT ITS ONE HOME.

    `load_channels` takes no `path=`, because a file holding webhook URLs (for Slack and Discord the URL IS the
    credential) is reached through one resolver and nowhere else -- the same
    rule the notify key file already followed.  A test points the config
    directory, like everything else in the suite.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    # ASK THE RESOLVER.  A test that hand-builds the path is the same defect
    # the production rule forbids, one layer out.
    f = M.default_notify_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"channels": channels}))
    return f


def test_a_configured_channel_is_read(tmp_path, monkeypatch):
    """A third party's shape: the credential is in the URL or a header,
    because Slack and Discord have nowhere else to keep one."""
    _file(tmp_path, monkeypatch, {"slack": {"url": "https://example/hook",
                                   "headers": {"Authorization": "Bearer t"}}})
    assert M.load_channels() == {
        "slack": {"url": "https://example/hook", "key": None, "kind": None,
                  "headers": {"Authorization": "Bearer t"}}}


def test_the_report_selection_decides_which_fields_a_card_shows():
    """`notify.report` is a CEILING, never a floor (`stages.md` § 6.9).

    Three states, and they are three answers:
      * absent -> every field the monitor could determine
      * ``()`` -> the summary line, with no field grid
      * a list -> those of them the monitor could determine

    **And the name is in all three.**  It is not on the list because it is
    not optional: a report you cannot attribute to a job is a notification
    you have to go and look up.  The record names no event, so the layout
    has no row for it and the declaration's order lays it out, whole
    (`run-reports.md` § 4.1c)."""
    rec = {"run": "BDT_Au_relax", "job": "62238108", "state": "running",
           "text": "state=running elapsed=1234s", "elapsed_s": 1234.5,
           "n_iters": 7, "energy": "-1740.21", "geom_step": 3,
           "per_iter_s": 12.8}
    dest = {"url": "https://discord.com/api/webhooks/1/t", "kind": "discord"}

    def fields(items):
        body, _ = M.webhook_request(dest, rec, items)
        embed = json.loads(body)["embeds"][0]
        assert "BDT_Au_relax" in embed["title"], (
            "the calculation's name is not settable and must always be sent")
        assert "62238108" in embed["title"]
        return [f["name"] for f in embed.get("fields", [])]

    assert len(fields(None)) == 5, "absent must mean every field"
    assert fields(()) == [], "an empty selection must leave no field grid"
    assert fields(("elapsed_s", "energy")) == ["elapsed", "energy"]
    # ORDER IS THE CARD'S, not the person's: two people who ticked the same
    # boxes in a different order must get the same card.
    assert fields(("energy", "elapsed_s")) == ["elapsed", "energy"]


def test_a_field_the_monitor_never_determined_stays_absent_when_asked_for():
    """A ceiling, not a floor.  Asking for `energy` on a run that has not
    printed one yields no `energy` field, not an empty one
    (`run-reports.md` § 4.1a)."""
    rec = {"run": "J", "state": "running", "text": "x", "elapsed_s": 12.0}
    body, _ = M.webhook_request(
        {"url": "https://discord.com/api/webhooks/1/t", "kind": "discord"},
        rec, ("elapsed_s", "energy", "geom_step"))
    names = [f["name"] for f in json.loads(body)["embeds"][0]["fields"]]
    assert names == ["elapsed"], names


def test_the_two_chat_destinations_show_the_SAME_fields():
    """One card, two vocabularies -- the user asked for them to look alike,
    so a field shown in one and not the other is the drift to catch."""
    rec = {"run": "J", "job": "7", "state": "done", "text": "x",
           "elapsed_s": 12.0, "n_iters": 3, "energy": "-1.0"}
    d_body, _ = M.webhook_request(
        {"url": "https://discord.com/api/webhooks/1/t", "kind": "discord"}, rec)
    s_body, _ = M.webhook_request(
        {"url": "https://hooks.slack.com/services/x", "kind": "slack"}, rec)
    d = json.loads(d_body)["embeds"][0]
    a = json.loads(s_body)["attachments"][0]
    assert d["title"] == a["title"]
    assert [f["name"] for f in d["fields"]] == [f["title"] for f in a["fields"]]
    assert "#%06X" % d["color"] == a["color"]


_EVERY = {"run": "J", "job": "7", "state": "running", "text": "x",
          "elapsed_s": 12.0, "n_iters": 3, "energy": "-1.0", "per_iter_s": 2.0,
          "max_force": 0.01, "geom_step": 4}


def _wire_fields(kind, rec):
    """The field labels a chat wire is sent, read from the generated body."""
    body, _ = M.webhook_request(
        {"url": "https://example/hook", "kind": kind}, rec)
    doc = json.loads(body)
    if kind == "slack":
        att = doc["attachments"][0]
        return [f["title"] for f in att.get("fields", [])], \
            [f["short"] for f in att.get("fields", [])], att["text"]
    emb = doc["embeds"][0]
    return [f["name"] for f in emb.get("fields", [])], \
        [f["inline"] for f in emb.get("fields", [])], emb["description"]


@pytest.mark.parametrize("event, kind, labels", [
    # `run-reports.md` § 4.1c, *What the shipped layout chooses*
    ("finish", "discord",
     ["energy", "max force", "step", "elapsed", "SCF iters", "per iter"]),
    ("finish", "slack",
     ["energy", "max force", "step", "elapsed", "SCF iters", "per iter"]),
    ("scf_converged", "discord",
     ["elapsed", "SCF iters", "per iter", "energy", "max force", "step"]),
    ("scf_converged", "slack",
     ["elapsed", "per iter", "SCF iters", "step", "energy", "max force"]),
    ("wall_near", "discord",
     ["elapsed", "SCF iters", "energy", "max force", "step"]),
    ("start", "discord", []),
    ("start", "slack", []),
    ("test", "discord", []),
])
def test_the_layout_lays_a_card_out_by_event_and_receiver(event, kind, labels):
    """`run-reports.md` § 4.1c: the card's fields are the receiver's order
    for the report's event, read from the layout file -- a finish card the
    result first, an SCF card the time and rate first, a start or test card
    the title and summary alone; Slack's order pairs its two columns.  The
    order flows from JSON through the reader into one producer, so a card
    drifting from the file is silent: the wires still render it."""
    rec = {**_EVERY, "event": event}
    got, inline, text = _wire_fields(kind, rec)
    assert got == labels, got
    assert all(inline), "the shipped layout sets every receiver inline"
    assert text == "x", "the shipped layout shows the summary"
    web = M.card(rec, receiver="web")
    assert [f[0] for f in web["fields"]] == _wire_fields("discord", rec)[0]
    assert web["inline"] is True


def test_a_part_that_has_ended_leaves_the_grid_for_the_line():
    """`run-reports.md` § 2.8: a report sent while another part of the run
    is underway -- SIESTA's optical calculation, after its SCF -- leaves
    the SCF's own fields (those declared its part, `report_fields`) off
    every receiver's grid; the summary line says them as ended.  THE
    SILENT FAILURE: the user's Raman card, two hours into the optical
    calculation, read *per iter 5.43 s · SCF iters 6* as if the SCF were
    in flight."""
    rec = {**_EVERY, "event": "periodic", "part": "optical"}
    for kind in ("discord", "slack"):
        got, _inline, _text = _wire_fields(kind, rec)
        assert "SCF iters" not in got and "per iter" not in got, (kind, got)
        assert {"elapsed", "energy", "max force", "step"} <= set(got), (kind,
                                                                        got)
    web = M.card(rec, receiver="web")
    assert [f[0] for f in web["fields"]] == _wire_fields("discord", rec)[0]


def test_the_layouts_limit_cuts_the_grid_and_the_web_has_none(tmp_path, monkeypatch):
    """§ 4.1c `limit`: a copy that lets Slack show two fields cuts the grid
    there; the web receiver, `0`, shows every field the run stated."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    from molbuilder import report_cards as C
    doc = json.loads(C.shipped_text())
    doc["receivers"]["slack"]["limit"] = 2
    C.layout_path().write_text(json.dumps(doc))
    rec = {**_EVERY, "event": "finish"}
    assert _wire_fields("slack", rec)[0] == ["energy", "max force"]
    assert len(M.card(rec, receiver="web")["fields"]) == 6


@pytest.mark.parametrize("change, words", [
    # (what the copy says, the words of its refusal)
    (lambda d: d["receivers"]["slack"]["order"].update(
        finish=["energy", "enrgy"]), "'enrgy', which is not a report field"),
    (lambda d: d["receivers"]["slack"].update(limit=11),
     "slack takes 10 fields at most"),
    (lambda d: d["receivers"]["discord"].update(limit=0),
     "discord takes 25 fields at most"),
    (lambda d: d["receivers"].pop("web"), "no layout for the receiver 'web'"),
    (lambda d: d["receivers"]["web"]["order"].update(
        {event: [n for n in names if n not in ("n_iters", "energy")]
         for event, names in d["receivers"]["web"]["order"].items()}),
     "shows n_iters, energy"),
    (lambda d: d["receivers"]["web"]["order"].update(stop=["energy"]),
     "stop: not an event"),
    (lambda d: d.update(schema="molbuilder/report-cards@2"),
     "schema 'molbuilder/report-cards@2'"),
    (lambda d: d["colors"].pop("other"), "naming 'other'"),
])
def test_a_copy_that_does_not_read_is_refused_by_its_reason(
        tmp_path, monkeypatch, change, words):
    """§ 4.1c: the copy at `<config dir>/report-cards.json` replaces the
    shipped file whole, and one that does not read is refused with its
    path and the reason -- an unknown field, a limit past the wire's own,
    a receiver missing, a field no event shows, an event that is not one,
    another schema, no colour for `other`.  Nothing is guessed at: the
    monitor stops at its start and the Test button answers the refusal."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    from molbuilder import report_cards as C
    doc = json.loads(C.shipped_text())
    change(doc)
    C.layout_path().write_text(json.dumps(doc))
    with pytest.raises(C.LayoutError) as err:
        M.webhook_request({"url": "https://example/hook", "kind": "slack"},
                          {**_EVERY, "event": "finish"})
    assert words in str(err.value), str(err.value)
    assert str(C.layout_path()) in str(err.value)


def test_a_copy_in_the_config_dir_lays_the_card_out(tmp_path, monkeypatch):
    """§ 4.1c: this machine's copy replaces the shipped layout whole -- its
    order, its colours, its summary switch -- and is read through the one
    resolver (`report_cards.layout_path`)."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    from molbuilder import report_cards as C
    doc = json.loads(C.shipped_text())
    disc = doc["receivers"]["discord"]
    disc["order"]["finish"] = list(reversed(disc["order"]["finish"]))
    disc["inline"] = False
    disc["summary"] = False
    doc["colors"]["running"] = "#123456"
    C.layout_path().write_text(json.dumps(doc))
    rec = {**_EVERY, "event": "finish"}
    got, inline, text = _wire_fields("discord", rec)
    assert got == ["per iter", "SCF iters", "elapsed", "step", "max force",
                   "energy"]
    assert not any(inline) and text == ""
    body, _ = M.webhook_request({"url": "https://example/hook",
                                 "kind": "discord"}, rec)
    assert json.loads(body)["embeds"][0]["color"] == 0x123456


def test_a_declared_kind_survives_the_loader(tmp_path, monkeypatch):
    """`kind` says WHICH WIRE FORMAT the destination reads, and the loader
    dropped it on the floor until 2026-09-02 -- so a channel that declared
    `"discord"` was still shaped from its host, and one behind a proxy could
    not be told apart at all (`run-reports.md` § 4.1b)."""
    _file(tmp_path, monkeypatch, {"relay": {"url": "https://relay.example/hook",
                                   "kind": "discord"}})
    got = M.load_channels()["relay"]
    assert got["kind"] == "discord"
    assert M.channel_kind(got) == "discord"


def test_a_misspelled_kind_falls_back_to_the_host_and_says_so(tmp_path, capsys, monkeypatch):
    """Named and wrong is not absent.  A typo silently taking the host's
    default would send a Slack-shaped body to Discord and earn a 400 nobody
    could trace back to a spelling -- and one bad field must not cost the
    channel (§ "one bad channel does not cost the others")."""
    _file(tmp_path, monkeypatch, {"chat": {"url": "https://discord.com/api/webhooks/1/t",
                                  "kind": "discrod"}})
    got = M.load_channels()["chat"]
    assert got["kind"] is None
    assert M.channel_kind(got) == "discord"          # the host still answers
    said = capsys.readouterr().out
    assert "channel 'chat'" in said, (
        "a misspelled kind must be reported against the channel it was found "
        f"on; got {said!r}")
    assert "reading it off the URL instead" in said, (
        "the report must say what it did INSTEAD, or a reader cannot tell a "
        "warning from a refusal")


def test_a_molbuilder_channel_carries_a_signing_key(tmp_path, monkeypatch):
    """Our own listener's shape: a plain url and a `key` that signs the
    body and never travels (`run-reports.md` § 4.1)."""
    _file(tmp_path, monkeypatch, {"lab": {"url": "https://qlab/api/x7Kq",
                                 "key": "s3cr3t"}})
    assert M.load_channels() == {
        "lab": {"url": "https://qlab/api/x7Kq", "key": "s3cr3t",
                "kind": None, "headers": {}}}


def test_several_channels_are_all_read(tmp_path, monkeypatch):
    """The point of naming them: one run can reach a Slack AND a listener
    (`run-reports.md` § 1).
    """
    _file(tmp_path, monkeypatch, {"slack": {"url": "https://example/hook"},
                         "lab": {"url": "https://qlab/api/x", "key": "k"}})
    assert sorted(M.load_channels()) == ["lab", "slack"]


def test_one_bad_channel_does_not_cost_the_others(tmp_path, monkeypatch):
    """A file with three channels and a typo in one reports on two.

    Refusing the file whole would turn one mistake into total silence, which
    is the failure this whole area keeps producing.
    """
    _file(tmp_path, monkeypatch, {"good": {"url": "https://qlab/api/x", "key": "k"},
                         "badkey": {"url": "https://qlab/api/y", "key": 12345},
                         "nourl": {"key": "k"},
                         "bad name": {"url": "https://qlab/api/z"}})
    assert list(M.load_channels()) == ["good"]


def test_a_key_that_is_not_a_string_skips_that_channel(tmp_path, monkeypatch):
    """Not "ignore the key and send unsigned" -- an unsigned report is one
    the listener will drop, and it would drop it in SILENCE."""
    _file(tmp_path, monkeypatch, {"lab": {"url": "https://qlab/api/x7Kq",
                                 "key": 12345}})
    assert M.load_channels() == {}


def test_no_file_means_no_notifier_and_no_complaint(tmp_path, monkeypatch):
    """Absent is not an error -- it is the feature being off, which is the
    default state for everybody who has not set it up."""
    assert M.load_channels() == {}


@pytest.mark.parametrize("body,why", [
    ("{not json",                  "unparseable"),
    ('{"channels": []}',           "channels is not an object"),
    ('["not", "an object"]',       "not an object"),
    ('{}',                         "no channels key"),
])
def test_a_broken_file_degrades_rather_than_raises(tmp_path, body, why, monkeypatch):
    """This is a MONITOR.  Refusing to watch a job because a notification
    could not be configured would be the tail wagging the dog: the run is
    the thing, and it is already going."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    f = M.default_notify_path(); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body)
    assert M.load_channels() == {}, why


def test_the_old_single_destination_file_is_named_not_just_skipped(tmp_path, monkeypatch):
    """`{"url": ...}` is a valid JSON object, so a silent skip would be
    indistinguishable from never having set anything up -- which is the
    exact failure the setup surface exists to stop.  It says which."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    f = M.default_notify_path(); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"url": "https://hooks.slack.com/services/T/B/X"}))
    log = tmp_path / "m.log"
    log.write_text("")
    assert M.load_channels(log=log) == {}
    text = log.read_text()
    assert "old single-destination file" in text
    assert "channels" in text, "it must say what the shape is now"


# --------------------------------------------------------------------- #
#  which channels one run uses -- `run-reports.md` § 3.0                  #
# --------------------------------------------------------------------- #

def test_naming_none_means_every_channel_the_machine_has():
    """The reading of a description that predates channels, and of one
    written by hand.  Nothing that already worked stops working."""
    have = {"a": {"url": "u"}, "b": {"url": "v"}}
    chosen, missing = M.channels_for(None, have)
    assert chosen == have and missing == []


def test_an_empty_list_means_nowhere_and_is_not_the_same_as_absent():
    """Two spellings because they are two intentions: reports off for THIS
    calculation, on a machine where they are otherwise set up.  Collapsing
    them sends a report to a channel the person just unticked."""
    have = {"a": {"url": "u"}}
    assert M.channels_for((), have) == ({}, [])
    assert M.channels_for(None, have) == (have, [])


def test_a_named_channel_the_machine_lacks_comes_back_as_missing():
    """The travelling case -- written at a desk, opened on a cluster.  It is
    not an error (the run is fine) and it must not be silent (a channel that
    resolves to nothing sends nothing, which looks exactly like working)."""
    chosen, missing = M.channels_for(("a", "gone"), {"a": {"url": "u"}})
    assert list(chosen) == ["a"]
    assert missing == ["gone"]


def test_the_notify_line_has_ONE_prefix_on_both_roads(tmp_path, capsys):
    """It was written twice -- a closure in `load_channels` and a copy in
    `_install_env_notifiers` that stamped the timestamp and then printed it
    under a second prefix. The log got the bare form and stdout got
    `[monitor] [2026-...] [NOTIFY] ...`, which is the shape of a message
    nobody greps successfully."""
    log = tmp_path / "m.log"
    log.write_text("")
    M._notify_say("something to say", log)
    text = log.read_text().strip()
    assert text.count("[NOTIFY]") == 1
    assert not text.startswith("[monitor]"), "the log form is not prefixed"

    M._notify_say("something to say", None)
    out = capsys.readouterr().out.strip()
    assert out == "[monitor] something to say", out


def test_a_missing_channel_is_said_in_the_monitor_log(tmp_path, monkeypatch):
    """Returned by `channels_for` is not enough: it has to reach the file a
    person actually opens."""
    monkeypatch.delenv("MB_NOTIFY_URL", raising=False)
    f = _file(tmp_path, monkeypatch, {"here": {"url": "https://example/hook"}})
    monkeypatch.setattr(M, "default_notify_path", lambda: f)
    M.clear_notifiers()
    log = tmp_path / "m.log"
    log.write_text("")
    M._install_env_notifiers(log, None, ("here", "elsewhere"))
    M.clear_notifiers()
    text = log.read_text()
    assert "elsewhere" in text
    assert "not set up on this machine" in text


def test_the_webhook_body_is_json_a_channel_can_render(tmp_path):
    """Slack and Discord both render a bare ``text`` field, and a private
    endpoint appending to a record log wants the structure beside it.  One
    body serves both, so there is no per-service code."""
    sent = {}

    class _Resp:
        def close(self):
            pass

    def _fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode())
        sent["headers"] = dict(req.header_items())
        sent["timeout"] = timeout
        return _Resp()

    hook = M.make_webhook_notifier("https://example/hook", headers={"Authorization": "Bearer t"})
    st = M.JobStatus(elapsed_s=12.0, n_iters=3, energy=-1.5, geom_step=2)
    import urllib.request as _u
    real = _u.urlopen
    _u.urlopen = _fake_urlopen
    try:
        hook(st, "scf_converged")
    finally:
        _u.urlopen = real

    assert sent["body"]["event"] == "scf_converged"
    assert sent["body"]["geom_step"] == 2
    assert sent["body"]["n_iters"] == 3
    assert "text" in sent["body"], "a channel needs something to render"
    # The credential rides in a header, so it is never in the URL for a
    # destination that does not put it there itself.
    assert any(k.lower() == "authorization" for k in sent["headers"])
    assert sent["timeout"] == M.NOTIFY_TIMEOUT_S


def test_a_count_nothing_stated_is_sent_absent_not_zero():
    """`run-reports.md` § 4.1a: a field the monitor could not determine is
    left out, so a reader can tell *unknown* from *wrong*.  The iteration
    count defaulted to 0, so a run that had stated nothing -- a PySCF Hessian
    whose progress log holds only its preview -- was reported as zero SCF
    iterations, on the card as ``SCF iters 0`` (2026-09-26).

    MUTATION THIS MUST FAIL AGAINST: default the count to 0."""
    rec = {"run": "J", "state": "running", "text": "x", "elapsed_s": 12.0,
           **{f: getattr(M.JobStatus(elapsed_s=12.0), f)
              for f in ("n_iters", "energy", "geom_step", "max_force",
                        "per_iter_s")}}
    assert rec["n_iters"] is None, rec
    body, _ = M.webhook_request(
        {"url": "https://discord.com/api/webhooks/1/t", "kind": "discord"},
        rec)
    names = [f["name"] for f in json.loads(body)["embeds"][0]["fields"]]
    assert names == ["elapsed"], names


def test_an_unreachable_destination_costs_the_run_nothing(tmp_path):
    """A dead server must not raise, must not retry, and must not block --
    this runs beside compute ranks."""
    hook = M.make_webhook_notifier("http://127.0.0.1:9/nowhere")
    hook(M.JobStatus(), "finish")        # must simply return


# --------------------------------------------------------------------- #
#  found by reading, not by testing                                      #
# --------------------------------------------------------------------- #

def test_a_misconfigured_destination_says_so_where_it_can_be_READ(tmp_path, monkeypatch):
    """The wrapper backgrounds this process with its stdout at ``/dev/null``.

    So anything printed goes nowhere.  A user whose notify file has a typo
    would get no notifications AND no explanation -- the diagnostic existed
    and was written to the bit bucket.  It goes to the monitor log, which is
    the file they actually open.

    Found by reading the diff, not by any test: every assertion about this
    path checked the RETURN value, which was correct all along.
    """
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    dest = M.default_notify_path(); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("{not json")
    log = tmp_path / "m.log"
    log.write_text("")

    assert M.load_channels(log=log) == {}
    text = log.read_text()
    assert "not valid JSON" in text
    assert str(dest) in text, "the message must name the file to go and fix"


def test_the_users_secret_is_never_echoed_into_the_log(tmp_path, monkeypatch):
    """The log is written into the run directory, which travels.  A
    complaint about a bad destination must not quote the destination."""
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
    dest = M.default_notify_path(); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('{"channels": {"s": {"url": '
                    '"https://hooks.slack.com/services/T/B/SECRET"}}')
    log = tmp_path / "m.log"
    log.write_text("")
    M.load_channels(log=log)
    assert "SECRET" not in log.read_text()
    assert "hooks.slack.com" not in log.read_text()


#: `run-reports.md` § 3, *an address is one the monitor can send to safely*:
#: each row a channel as a hand-edited file may hold it, and the fault the
#: monitor's log must name -- ``None`` for one it sends to.
_ADDRESSES = [
    ({"url": "http://hooks.slack.com/services/T/B/SECRETTAIL"},
     "a slack address is its credential"),
    ({"url": "http://discord.com/api/webhooks/1/SECRETTAIL"},
     "a discord address is its credential"),
    ({"url": "http://relay.example/SECRETTAIL", "kind": "slack"},
     "a slack address is its credential"),
    ({"url": "https://hooks.slack.com/services/T/B/SECRET TAIL"},
     "a space or a control character"),
    ({"url": "https:///SECRETTAIL"}, "must start with https://"),
    ({"url": "http://127.0.0.1:8888/api/SECRETTAIL", "key": "k"}, None),
]


@pytest.mark.parametrize("spec,fault", _ADDRESSES,
                         ids=[f or "a listener over http" for _s, f in _ADDRESSES])
def test_an_address_the_monitor_will_not_send_to_is_skipped_by_its_fault(
        spec, fault, tmp_path, monkeypatch):
    """THE SILENT FAILURE: a Slack or Discord webhook written ``http://``
    sends its credential unencrypted on every report; a malformed address
    fails every send, and a complaint quoting it would copy the credential
    into the run's log.  The monitor skips the channel, naming it and its
    fault, never the address -- and a molbuilder listener over ``http://``,
    signed, is sent to.

    MUTATIONS THIS MUST FAIL AGAINST: the reader not asking the address rule
    (every row sent to); the rule refusing every ``http://`` (the listener
    row skipped); the fault quoting the address."""
    _file(tmp_path, monkeypatch, {"lab": spec})
    log = tmp_path / "m.log"
    log.write_text("")
    got = M.load_channels(log=log)
    said = log.read_text()
    assert "SECRET" not in said, said
    if fault is None:
        assert list(got) == ["lab"], said
    else:
        assert got == {}, got
        assert "'lab'" in said and fault in said, said


def _closed_port() -> int:
    """A port on this host nothing listens on: bound by the system, then
    let go."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_a_failed_send_is_said_in_the_log_by_what_failed_never_the_address(
        tmp_path):
    """`run-reports.md` § 3: a failure costs the run nothing and is said in
    the monitor log -- the channel's name, the event, what failed.

    THE SILENT FAILURE: a channel that never delivers looks exactly like
    one that does, until a run that mattered sends nothing; and the
    exception's own text quotes the address, for Slack and Discord the
    credential, into a log that sits in the run's folder.

    MUTATIONS THIS MUST FAIL AGAINST: the failure swallowed (no line); the
    line built from the exception's text (the address in it)."""
    port = _closed_port()
    log = tmp_path / "m.log"
    log.write_text("")
    hook = M.make_webhook_notifier(f"http://127.0.0.1:{port}/api/SECRETTAIL",
                                   key="k", channel="lab", log=log)
    hook(M.JobStatus(), "finish")          # must simply return
    said = log.read_text()
    assert ("[NOTIFY] channel 'lab': finish not delivered -- the connection "
            "was refused") in said, said
    assert "SECRETTAIL" not in said and str(port) not in said, said


# --------------------------------------------------------------------- #
#  the destination path is a COPY of a rule, so it is pinned to it       #
# --------------------------------------------------------------------- #

def test_the_monitors_path_and_molbuilders_own_are_one_function(
        monkeypatch, tmp_path):
    """Not "they agree" -- they are the SAME function.

    `config_dir.py` is the one module allowed to spell this rule, because
    three once spelled it independently and two of them said so in prose:
    *"a comment is not a mechanism"*.  The monitor imports it rather than
    restating it.
    """
    from molbuilder.config_dir import secrets_dir
    assert M.default_notify_path() == secrets_dir() / M.NOTIFY_FILENAME

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "scratch"))
    assert M.default_notify_path() == secrets_dir() / M.NOTIFY_FILENAME
    assert str(M.default_notify_path()).startswith(str(tmp_path / "scratch")), \
        "XDG must move the token off $HOME -- that is the point of honouring it"


# That no module spells the config-dir rule twice is review's to hold
# (`process/code-audit.md` § 1c).


def test_the_shipped_monitor_resolves_it_with_no_molbuilder_installed(
        tmp_path):
    """THE CASE THAT MOTIVATES ALL OF THIS.

    The monitor runs on a compute node with the job's own python and no
    molbuilder installed.  It still has to find the destination file -- and
    it must not answer that by keeping its own copy of the rule.  So
    `config_dir.py` travels with it, in its one file (`mb_monitor.pyz`), and
    the monitor imports that.

    This writes that file into an otherwise empty directory, runs the monitor
    from it with no molbuilder reachable, and asks it where it would look.
    """
    import subprocess
    import sys
    from molbuilder import runwrap

    ship = tmp_path / "jobdir"
    ship.mkdir()
    (ship / runwrap.MONITOR_BUNDLE).write_bytes(runwrap.monitor_bundle())

    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "scratch")
    env.pop("PYTHONPATH", None)          # nothing of molbuilder reachable
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {runwrap.MONITOR_BUNDLE!r}); "
         "import mb_monitor; print(mb_monitor.default_notify_path())"],
        cwd=str(ship), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"the shipped monitor could not resolve its own path:\n{proc.stderr}")
    assert proc.stdout.strip() == str(
        tmp_path / "scratch/molbuilder/secrets/notify")


def test_the_shipped_monitor_loads_on_a_python_with_nothing_installed(tmp_path):
    """`run-reports.md` § 2.3: the monitor runs beside the job on the job's
    own python, which on a cluster is a bare interpreter -- no numpy, no
    molbuilder.  The bundle must import under one.  Measured 2026-10-19 on
    Sol: a module-level numpy import in a companion (`emitter.py`, shipped
    for the grammar's keys) killed every monitor at import, so a day's
    runs reported nothing and kept no utilisation record.  Blocking numpy
    in the interpreter (`sys.modules['numpy'] = None` makes its import
    raise) is how a numpy-less python is had on a machine that has it."""
    import subprocess
    import sys
    from molbuilder import runwrap

    ship = tmp_path / "jobdir"
    ship.mkdir()
    (ship / runwrap.MONITOR_BUNDLE).write_bytes(runwrap.monitor_bundle())
    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "scratch")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['numpy'] = None; "
         f"sys.path.insert(0, {runwrap.MONITOR_BUNDLE!r}); "
         "import mb_monitor, _run_ending, siesta_reader, report_cards; "
         "lay = report_cards.load(); "
         "print('loaded', lay.source, mb_monitor.card("
         "{'event': 'finish', 'energy': 1.5}, receiver='discord', "
         "layout=lay)['fields'])"],
        cwd=str(ship), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "loaded" in proc.stdout, (
        f"the shipped monitor needs something a bare python lacks:\n"
        f"{proc.stderr[-2000:]}")
    # THE LAYOUT IS THE BUNDLE'S OWN MEMBER (`run-reports.md` § 4.1c), read
    # through the zip's loader -- not a file beside the job, not the
    # package's.
    assert f"{runwrap.MONITOR_BUNDLE}/report-cards.json" in proc.stdout
    assert "[('energy', '1.5 eV')]" in proc.stdout, proc.stdout


def test_the_shipped_vibration_finish_loads_its_readers_from_the_bundle_alone(
        tmp_path):
    """`engines/vibration.md` I23: the finish travels as one declared set
    (`runwrap.VIBRATION_COMPANIONS`) that imports only what the job env
    carries -- and what its own modules import of each other must be in
    the set.  The SIESTA reader it ships reads the output's ending and,
    since 2.21.0, composes where a run is (`parse/engines/transiesta.py`):
    a module the reader imports and the set did not ship made every
    finish die at import on the real-junction e2e (2026-10-19).  The
    readers are imported from the bundle on a python with no molbuilder
    on its path; numpy is left as the job env has it (the finish needs
    it, I23)."""
    import subprocess
    import sys
    from molbuilder import runwrap
    from molbuilder.runfiles import VIBRATION_BUNDLE

    ship = tmp_path / "jobdir"
    ship.mkdir()
    (ship / VIBRATION_BUNDLE).write_bytes(runwrap.vibration_bundle())
    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "scratch")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {VIBRATION_BUNDLE!r}); "
         "import siesta_reader, siesta_fc, fdf; print('loaded')"],
        cwd=str(ship), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "loaded" in proc.stdout, (
        f"the shipped finish's readers need a module the bundle does not "
        f"carry:\n{proc.stderr[-2000:]}")
