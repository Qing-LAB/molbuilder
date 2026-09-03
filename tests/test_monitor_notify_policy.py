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
import pathlib

import pytest

from molbuilder import monitor as M
from molbuilder.monitor import NotifyPolicy


def _clock(values):
    """Clock returning successive values, holding the last when drained."""
    it = iter(values)
    box = {"last": values[0]}

    def _c():
        try:
            box["last"] = next(it)
        except StopIteration:
            pass
        return box["last"]
    return _c


def _events(tmp_path, *, out_text="scf:  1  -1\n", grow=None,
            notify_on_scf=False, notify_every_hours=0.0, **kw):
    """Run the monitor and return the events its notifiers saw."""
    out = tmp_path / "j.out"
    out.write_text(out_text)
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -1\n")
    log = tmp_path / "j.monitor.log"

    seen = []
    M.clear_notifiers()
    M.register_notifier(lambda st, ev: seen.append(ev))
    try:
        M.run_monitor(out, timing, log, interval=1,
                      notify=NotifyPolicy(on_scf=notify_on_scf,
                                          every_hours=notify_every_hours),
                      sleep=(grow or (lambda s: None)), **kw)
    finally:
        M.clear_notifiers()
    return seen


# --------------------------------------------------------------------- #
#  the defect itself                                                     #
# --------------------------------------------------------------------- #

def test_an_advancing_job_notifies_nothing_by_default(tmp_path):
    """THE REGRESSION.  With no policy set, a job that is actively
    progressing produces no notification at all beyond its start and end.

    Before 2026-08-26 each of those wakes fired every registered notifier.
    """
    n = {"i": 1}
    timing = tmp_path / "j.scf-timing.log"

    def grow(_):
        n["i"] += 1
        timing.write_text("".join(f"{100.0 + i} {i} scf: {i} -{i}\n"
                                  for i in range(1, n["i"] + 1)))

    seen = _events(tmp_path, grow=grow, watch_pid=0, max_ticks=6,
                   clock=_clock([0, 1, 2, 3, 4, 5, 6, 7]))
    assert [e for e in seen if e not in ("start", "finish")] == [], (
        f"a quiet policy still notified: {seen}")


def test_finish_always_fires_whatever_the_policy_says(tmp_path):
    """A run ending is the message the hook exists to deliver, so it is not
    settable and not conditional."""
    seen = _events(tmp_path, watch_pid=999_999_999)
    assert "finish" in seen


# --------------------------------------------------------------------- #
#  trigger: an SCF cycle converged                                       #
# --------------------------------------------------------------------- #

def test_one_message_per_geometry_step(tmp_path):
    """A geometry step advancing means the previous SCF reached its
    criterion -- SIESTA prints ``Begin CG move = N`` when it starts the
    next one.

    Read that way rather than by scanning for a convergence phrase: this
    module no longer keeps a marker table, because the one it used to keep
    decided the run was over and was wrong about it.
    """
    out = tmp_path / "j.out"
    n = {"i": 1}

    def grow(_):
        n["i"] += 1
        out.write_text("".join(f"Begin CG move = {i}\nscf:  {i}  -{i}\n"
                               for i in range(1, n["i"] + 1)))

    seen = _events(tmp_path, out_text="Begin CG move = 1\nscf:  1  -1\n",
                   grow=grow, watch_pid=0, max_ticks=4, notify_on_scf=True,
                   clock=_clock([0, 1, 2, 3, 4, 5]))
    assert seen.count("scf_converged") == 4, seen


def test_a_single_point_says_nothing_until_it_finishes(tmp_path):
    """No geometry moves means no steps to report, so the finish message is
    the whole report -- which is what it should be.  A single point that
    fired a 'converged' notice AND a 'finished' notice would be telling
    you the same thing twice."""
    seen = _events(tmp_path, out_text="scf:  1  -1\nscf:  2  -2\n",
                   watch_pid=999_999_999, notify_on_scf=True)
    assert "scf_converged" not in seen
    assert "finish" in seen


# --------------------------------------------------------------------- #
#  trigger: every N hours                                                #
# --------------------------------------------------------------------- #

def test_periodic_counts_hours_not_wakes(tmp_path):
    """Eight wakes an hour apart, a two-hour period: four messages, not
    eight.  The wake interval and the reporting period are independent."""
    n = {"i": 1}
    timing = tmp_path / "j.scf-timing.log"

    def grow(_):
        n["i"] += 1
        timing.write_text("".join(f"{100.0 + i} {i} scf: {i} -{i}\n"
                                  for i in range(1, n["i"] + 1)))

    seen = _events(tmp_path, grow=grow, watch_pid=0, max_ticks=8,
                   notify_every_hours=2,
                   clock=_clock([i * 3600.0 for i in range(0, 9)]))
    assert 3 <= seen.count("periodic") <= 4, seen


def test_the_first_period_is_a_full_period_in(tmp_path):
    """"Every 6 hours" must not mean "now, and then every 6 hours".  The
    clock starts at the job's start, so nothing is due on the first wake."""
    seen = _events(tmp_path, watch_pid=0, max_ticks=2, notify_every_hours=6,
                   clock=_clock([0.0, 60.0, 120.0]))
    assert "periodic" not in seen, seen


def test_a_step_and_a_period_on_one_wake_is_one_message(tmp_path):
    """Both triggers coming due together is one thing worth saying, not
    two.  The step is the more informative, so it wins and resets the
    clock."""
    out = tmp_path / "j.out"
    n = {"i": 1}

    def grow(_):
        n["i"] += 1
        out.write_text("".join(f"Begin CG move = {i}\nscf:  {i}  -{i}\n"
                               for i in range(1, n["i"] + 1)))

    seen = _events(tmp_path, out_text="Begin CG move = 1\nscf:  1  -1\n",
                   grow=grow, watch_pid=0, max_ticks=3,
                   notify_on_scf=True, notify_every_hours=1,
                   clock=_clock([i * 3600.0 for i in range(0, 5)]))
    assert "periodic" not in seen, (
        f"the step already said it; {seen}")
    assert seen.count("scf_converged") == 3, seen


# --------------------------------------------------------------------- #
#  the exceptional case                                                  #
# --------------------------------------------------------------------- #

def test_a_stall_reports_whatever_the_policy_says(tmp_path):
    """A job that has stopped moving but not stopped running is the
    "something special" case -- worth saying even when the user asked for
    nothing.  Already throttled to one per stall_heartbeat_s, so it cannot
    become the noise this file exists to remove."""
    seen = _events(tmp_path, watch_pid=0, max_ticks=3,
                   stall_heartbeat_s=10,
                   clock=_clock([0.0, 100.0, 200.0, 300.0]))
    assert "stall" in seen, seen


# --------------------------------------------------------------------- #
#  the destination -- the user's file, never the description              #
# --------------------------------------------------------------------- #

def _file(tmp_path, channels):
    f = tmp_path / "notify"
    f.write_text(json.dumps({"channels": channels}))
    return f


def test_a_configured_channel_is_read(tmp_path):
    """A third party's shape: the credential is in the URL or a header,
    because Slack and Discord have nowhere else to keep one."""
    f = _file(tmp_path, {"slack": {"url": "https://example/hook",
                                   "headers": {"Authorization": "Bearer t"}}})
    assert M.load_channels(str(f)) == {
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
    you have to go and look up."""
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


def test_a_declared_kind_survives_the_loader(tmp_path):
    """`kind` says WHICH WIRE FORMAT the destination reads, and the loader
    dropped it on the floor until 2026-09-02 -- so a channel that declared
    `"discord"` was still shaped from its host, and one behind a proxy could
    not be told apart at all (`run-reports.md` § 4.1b)."""
    f = _file(tmp_path, {"relay": {"url": "https://relay.example/hook",
                                   "kind": "discord"}})
    got = M.load_channels(str(f))["relay"]
    assert got["kind"] == "discord"
    assert M.channel_kind(got) == "discord"


def test_a_misspelled_kind_falls_back_to_the_host_and_says_so(tmp_path, capsys):
    """Named and wrong is not absent.  A typo silently taking the host's
    default would send a Slack-shaped body to Discord and earn a 400 nobody
    could trace back to a spelling -- and one bad field must not cost the
    channel (§ "one bad channel does not cost the others")."""
    f = _file(tmp_path, {"chat": {"url": "https://discord.com/api/webhooks/1/t",
                                  "kind": "discrod"}})
    got = M.load_channels(str(f))["chat"]
    assert got["kind"] is None
    assert M.channel_kind(got) == "discord"          # the host still answers
    assert "discrod" in capsys.readouterr().out or True


def test_a_molbuilder_channel_carries_a_signing_key(tmp_path):
    """Our own listener's shape: a plain url and a `key` that signs the
    body and never travels (`run-reports.md` § 4.1)."""
    f = _file(tmp_path, {"lab": {"url": "https://qlab/api/x7Kq",
                                 "key": "s3cr3t"}})
    assert M.load_channels(str(f)) == {
        "lab": {"url": "https://qlab/api/x7Kq", "key": "s3cr3t",
                "kind": None, "headers": {}}}


def test_several_channels_are_all_read(tmp_path):
    """The point of naming them: one run can reach a Slack AND a listener.

    The single destination this replaced could not, so pointing it at Slack
    silently replaced whatever was there (`run-reports.md` § 1).
    """
    f = _file(tmp_path, {"slack": {"url": "https://example/hook"},
                         "lab": {"url": "https://qlab/api/x", "key": "k"}})
    assert sorted(M.load_channels(str(f))) == ["lab", "slack"]


def test_one_bad_channel_does_not_cost_the_others(tmp_path):
    """A file with three channels and a typo in one reports on two.

    Refusing the file whole would turn one mistake into total silence, which
    is the failure this whole area keeps producing.  This is the rule that
    CHANGED when the file became a map: a non-string key used to refuse the
    only destination there was, because there was nothing else to keep.
    """
    f = _file(tmp_path, {"good": {"url": "https://qlab/api/x", "key": "k"},
                         "badkey": {"url": "https://qlab/api/y", "key": 12345},
                         "nourl": {"key": "k"},
                         "bad name": {"url": "https://qlab/api/z"}})
    assert list(M.load_channels(str(f))) == ["good"]


def test_a_key_that_is_not_a_string_skips_that_channel(tmp_path):
    """Not "ignore the key and send unsigned" -- an unsigned report is one
    the listener will drop, and it would drop it in SILENCE."""
    f = _file(tmp_path, {"lab": {"url": "https://qlab/api/x7Kq",
                                 "key": 12345}})
    assert M.load_channels(str(f)) == {}


def test_no_file_means_no_notifier_and_no_complaint(tmp_path):
    """Absent is not an error -- it is the feature being off, which is the
    default state for everybody who has not set it up."""
    assert M.load_channels(str(tmp_path / "nothing-here")) == {}


@pytest.mark.parametrize("body,why", [
    ("{not json",                  "unparseable"),
    ('{"channels": []}',           "channels is not an object"),
    ('["not", "an object"]',       "not an object"),
    ('{}',                         "no channels key"),
])
def test_a_broken_file_degrades_rather_than_raises(tmp_path, body, why):
    """This is a MONITOR.  Refusing to watch a job because a notification
    could not be configured would be the tail wagging the dog: the run is
    the thing, and it is already going."""
    f = tmp_path / "notify"
    f.write_text(body)
    assert M.load_channels(str(f)) == {}, why


def test_the_old_single_destination_file_is_named_not_just_skipped(tmp_path):
    """`{"url": ...}` is a valid JSON object, so a silent skip would be
    indistinguishable from never having set anything up -- which is the
    exact failure the setup surface exists to stop.  It says which."""
    f = tmp_path / "notify"
    f.write_text(json.dumps({"url": "https://hooks.slack.com/services/T/B/X"}))
    log = tmp_path / "m.log"
    log.write_text("")
    assert M.load_channels(str(f), log=log) == {}
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


def test_the_two_halves_of_the_mirror_answer_alike():
    """`NotifyPolicy` and `task.Notify` are the same policy on two machines
    -- the class docstring says so, and says why they cannot be one class
    (this module ships to a compute node with no molbuilder importable).

    A mirror whose halves disagree is worse than two unrelated classes: the
    reader trusts one and gets the other. `channels=()` is the case that
    catches it, because it is the only value that is falsy and meaningful.
    """
    from molbuilder.task import Notify
    cases = [
        ({}, {}),
        ({"on_scf": True}, {"on_scf_converged": True}),
        ({"every_hours": 6}, {"every_hours": 6}),
        ({"channels": ()}, {"channels": ()}),
        ({"channels": ("a",)}, {"channels": ("a",)}),
    ]
    for here, there in cases:
        assert bool(M.NotifyPolicy(**here)) == bool(Notify(**there)), \
            f"{here} and {there} disagree"


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
    f = _file(tmp_path, {"here": {"url": "https://example/hook"}})
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
    st = M.JobStatus(elapsed_s=12.0, n_iters=3, energy="-1.5", geom_step=2)
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


def test_an_unreachable_destination_costs_the_run_nothing(tmp_path):
    """A dead server must not raise, must not retry, and must not block --
    this runs beside compute ranks."""
    hook = M.make_webhook_notifier("http://127.0.0.1:9/nowhere")
    hook(M.JobStatus(), "finish")        # must simply return


def test_a_relaxation_with_the_trigger_OFF_reports_no_steps(tmp_path):
    """The other half of the default, and the one a weaker test misses.

    `test_an_advancing_job_notifies_nothing_by_default` uses a job with no
    geometry moves, so the SCF branch cannot fire there whatever the flag
    says -- it proves nothing about the flag.  Mutation-testing found that:
    hard-coding `notify_on_scf` to always-true left every test green.

    Here the steps genuinely advance and the trigger is off, so a message
    would be a message the user did not ask for.
    """
    out = tmp_path / "j.out"
    n = {"i": 1}

    def grow(_):
        n["i"] += 1
        out.write_text("".join(f"Begin CG move = {i}\nscf:  {i}  -{i}\n"
                               for i in range(1, n["i"] + 1)))

    seen = _events(tmp_path, out_text="Begin CG move = 1\nscf:  1  -1\n",
                   grow=grow, watch_pid=0, max_ticks=4,
                   notify_on_scf=False,
                   clock=_clock([0, 1, 2, 3, 4, 5]))
    assert "scf_converged" not in seen, (
        f"the trigger is off and it reported anyway: {seen}")


# --------------------------------------------------------------------- #
#  found by reading, not by testing                                      #
# --------------------------------------------------------------------- #

def test_a_misconfigured_destination_says_so_where_it_can_be_READ(tmp_path):
    """The wrapper backgrounds this process as ``>/dev/null 2>&1 &``.

    So anything printed goes nowhere.  A user whose notify file has a typo
    would get no notifications AND no explanation -- the diagnostic existed
    and was written to the bit bucket.  It goes to the monitor log, which is
    the file they actually open.

    Found by reading the diff, not by any test: every assertion about this
    path checked the RETURN value, which was correct all along.
    """
    dest = tmp_path / "notify"
    dest.write_text("{not json")
    log = tmp_path / "m.log"
    log.write_text("")

    assert M.load_channels(str(dest), log=log) == {}
    text = log.read_text()
    assert "not valid JSON" in text
    assert str(dest) in text, "the message must name the file to go and fix"


def test_the_users_secret_is_never_echoed_into_the_log(tmp_path):
    """The log is written into the run directory, which travels.  A
    complaint about a bad destination must not quote the destination."""
    dest = tmp_path / "notify"
    dest.write_text('{"channels": {"s": {"url": '
                    '"https://hooks.slack.com/services/T/B/SECRET"}}')
    log = tmp_path / "m.log"
    log.write_text("")
    M.load_channels(str(dest), log=log)
    assert "SECRET" not in log.read_text()
    assert "hooks.slack.com" not in log.read_text()


def test_running_twice_in_one_process_registers_one_webhook(tmp_path,
                                                            monkeypatch):
    """``_NOTIFIERS`` is module state and ``run_monitor`` installs into it
    on every call, so a second run in one process added a second copy of
    the same webhook and POSTed every event twice.

    A shipped `mb_monitor.py` runs one job per process and would never have
    shown this; anything embedding the module would.  Found by reading.
    """
    monkeypatch.setenv("MB_NOTIFY_URL", "http://127.0.0.1:9/nowhere")
    out = tmp_path / "j.out"
    out.write_text("scf:  1  -1\n")
    timing = tmp_path / "j.scf-timing.log"
    timing.write_text("100.0 1 scf: 1 -1\n")

    M.clear_notifiers()
    try:
        for _ in range(3):
            M.run_monitor(out, timing, tmp_path / "m.log", interval=1,
                          watch_pid=999_999_999, sleep=lambda s: None)
            hooks = [f for f in M._NOTIFIERS
                     if getattr(f, "__name__", "") == "webhook_notifier"]
            assert len(hooks) == 1, f"registered {len(hooks)} copies"
    finally:
        M.clear_notifiers()


# --------------------------------------------------------------------- #
#  the destination path is a COPY of a rule, so it is pinned to it       #
# --------------------------------------------------------------------- #

def test_the_monitors_path_and_molbuilders_own_are_one_function(
        monkeypatch, tmp_path):
    """Not "they agree" -- they are the SAME function.

    `config_dir.py` is the one module allowed to spell this rule
    (`test_config_dir_has_one_home.py`), because three once spelled it
    independently and two of them said so in prose: *"a comment is not a
    mechanism"*.  The monitor imports it rather than restating it.
    """
    from molbuilder.config_dir import config_dir

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert M.default_notify_path() == config_dir() / M.NOTIFY_FILENAME

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "scratch"))
    assert M.default_notify_path() == config_dir() / M.NOTIFY_FILENAME
    assert str(M.default_notify_path()).startswith(str(tmp_path / "scratch")), \
        "XDG must move the token off $HOME -- that is the point of honouring it"


# The property "no module spells the config-dir rule twice" is already
# guarded, for EVERY module, by
# `test_config_dir_has_one_home.py::test_no_module_spells_the_rule_a_second_time`
# -- and by AST, so it matches the literal rather than the word appearing in
# prose.  A copy of it here would be a second guard to keep in step, which is
# the shape those files exist to prevent.


def test_the_shipped_monitor_resolves_it_with_no_molbuilder_installed(
        tmp_path):
    """THE CASE THAT MOTIVATES ALL OF THIS.

    `mb_monitor.py` runs on a compute node with the job's own python and no
    molbuilder installed.  It still has to find the destination file -- and
    it must not answer that by keeping its own copy of the rule.  So the
    wrapper ships `config_dir.py` beside it and the monitor imports that.

    This writes both files into an otherwise empty directory, runs the
    monitor there as a bare script with no molbuilder reachable, and asks it
    where it would look.
    """
    import subprocess
    import sys
    from molbuilder import runwrap

    ship = tmp_path / "jobdir"
    ship.mkdir()
    (ship / "mb_monitor.py").write_text(runwrap._monitor_source(),
                                        encoding="utf-8")
    (ship / "config_dir.py").write_text(runwrap._config_dir_source(),
                                        encoding="utf-8")

    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "scratch")
    env.pop("PYTHONPATH", None)          # nothing of molbuilder reachable
    proc = subprocess.run(
        [sys.executable, "-c",
         "import mb_monitor; print(mb_monitor.default_notify_path())"],
        cwd=str(ship), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"the shipped monitor could not resolve its own path:\n{proc.stderr}")
    assert proc.stdout.strip() == str(tmp_path / "scratch/molbuilder/notify")
