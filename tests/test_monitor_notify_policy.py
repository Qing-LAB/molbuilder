"""How often the monitor LOOKS and how often it TELLS you are not the same.

**The defect this closes.** The monitor wakes on its interval, samples
CPU/GPU/memory, and — in the same pass — called every registered notifier,
guarded only by *did anything change*.  On a running job the SCF iteration
count advances constantly, so that was true almost every wake.  A Slack
webhook configured against it would have received a message **every ten
seconds for the length of the run**.

Sampling stays dense: `util.csv` is the diagnostic record and its whole
point is showing whether the CPU/GPU/memory allocation is being used.
Notifying becomes policy (`plans/bench-and-junction-plan.md` § 2.9), and the
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

def test_a_configured_destination_is_read(tmp_path):
    f = tmp_path / "notify"
    f.write_text(json.dumps({"url": "https://example/hook",
                             "headers": {"Authorization": "Bearer t"}}))
    assert M.load_destination(str(f)) == {
        "url": "https://example/hook",
        "headers": {"Authorization": "Bearer t"}}


def test_no_file_means_no_notifier_and_no_complaint(tmp_path):
    """Absent is not an error -- it is the feature being off, which is the
    default state for everybody who has not set it up."""
    assert M.load_destination(str(tmp_path / "nothing-here")) is None


@pytest.mark.parametrize("body,why", [
    ("{not json",            "unparseable"),
    ('{"headers": {}}',      "no url"),
    ('{"url": ""}',          "empty url"),
    ('["not", "an object"]', "not an object"),
])
def test_a_broken_destination_degrades_rather_than_raises(tmp_path, body, why):
    """This is a MONITOR.  Refusing to watch a job because a notification
    could not be configured would be the tail wagging the dog: the run is
    the thing, and it is already going."""
    f = tmp_path / "notify"
    f.write_text(body)
    assert M.load_destination(str(f)) is None, why


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

    hook = M.make_webhook_notifier("https://example/hook",
                                   {"Authorization": "Bearer t"})
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

    assert M.load_destination(str(dest), log=log) is None
    text = log.read_text()
    assert "not valid JSON" in text
    assert str(dest) in text, "the message must name the file to go and fix"


def test_the_users_secret_is_never_echoed_into_the_log(tmp_path):
    """The log is written into the run directory, which travels.  A
    complaint about a bad destination must not quote the destination."""
    dest = tmp_path / "notify"
    dest.write_text('{"url": "https://hooks.slack.com/services/T/B/SECRET"')
    log = tmp_path / "m.log"
    log.write_text("")
    M.load_destination(str(dest), log=log)
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
