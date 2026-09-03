"""The admin Reload route, and the two locks on it.

Contract: `ops/access-control.md` -- *"`POST /api/admin/reload` is not
registered at all unless both hold"*.  *(How the mechanism was designed:
`archive/2026-08-19-server-reload-plan.md` § 3.3 and § 4.)*

WHAT THIS ROUTE DOES: exits the process with a code the supervisor is waiting
for, so a fresh server starts with every module imported again.  There is no
module-swapping and no partial reload -- a new child is the whole mechanism,
which is why it cannot leave half the app on old code.

WHY THE GATE IS THE INTERESTING PART.  Restarting the process everyone is using
is not something to inherit by omission.  This server binds 0.0.0.0 behind
OAuth, so a Reload button beside the user's email must not be reachable by
everyone who can authenticate -- pressing it disconnects them all mid-calculation
and loses workspace writes still in flight.

The admin list therefore means NOBODY when it is absent or empty, and it says
that to every subsystem that asks (web/admin.py).  It lived inside `rate_limit`
until 2026-08-03, where empty meant "anyone signed in" and this route inverted
it for itself -- one value, two opposite readings.

The tests below pin both halves of the gate: with no supervisor, or with no
named admins, **the route does not exist** -- 404, not 403.  A misconfiguration
then reads as "the button is missing", never as "anyone can restart the server",
and the safe state is the one you get by doing nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

from molbuilder.reload_protocol import SUPERVISED_ENV  # noqa: E402


def _app(monkeypatch, *, supervised: bool, admins: list[str] | None):
    from molbuilder.web.app import create_app
    if supervised:
        monkeypatch.setenv(SUPERVISED_ENV, "1")
    else:
        monkeypatch.delenv(SUPERVISED_ENV, raising=False)
    # The admin list is its own top-level section: the same list answers "who may
    # clear the block list" and "who may restart the server".  Absent or empty
    # means ANYONE WHO SIGNED IN -- which is not an open door, because reaching a
    # session at all required being named in a provider's REQUIRED allowed_users.
    # Naming addresses here NARROWS that.
    cfg = {"rate_limit": {"enabled": False}, "admin": {"emails": admins or []}}
    app = create_app(config=cfg)
    # A session needs a signing key.  These tests do not exercise auth -- they
    # exercise the ADMIN GATE, which reads an already-established session -- so
    # the key is supplied directly rather than by standing up an OAuth provider.
    app.secret_key = "test-only-not-a-real-key"
    return app


def _as_logged_in(client, email="someone@example.org"):
    with client.session_transaction() as sess:
        sess["user"] = {"email": email}
    return client


# --------------------------------------------------------------------- #
#  The gate                                                             #
# --------------------------------------------------------------------- #

def test_no_supervisor_means_no_route(monkeypatch):
    """Without a supervisor there is nobody to bring the server back.

    An endpoint that stops an unsupervised server leaves a dead site and no way
    back from the browser -- so it must not be reachable at all.
    """
    app = _app(monkeypatch, supervised=False, admins=["someone@example.org"])
    r = _as_logged_in(app.test_client()).post("/api/admin/reload")
    assert r.status_code == 404, (
        "the reload route answered on a server with no supervisor; pressing it "
        "would stop the server for good"
    )


def test_naming_nobody_means_anyone_who_signed_in(monkeypatch):
    """THE ONE THAT MATTERS: with no `admin` section, a signed-in caller is one.

    Not because the door is open -- because it was already locked upstream.
    ``auth.providers[].allowed_users`` is a REQUIRED field, so nobody reaches a
    session without an operator having written their address by hand.  A second
    list repeating those same names is two lists to keep in step for one
    question, and on a single-operator server it is the SAME name written twice,
    with the restart button silently missing until you work out that you owe the
    file another copy of yourself.

    Asserted through the AVAILABILITY read rather than by pressing the button:
    a successful reload calls ``os._exit`` half a second later, which would take
    the test runner with it.  (That is not a hypothetical -- this test used to
    expect a 404 here, and when the rule changed it started succeeding and
    killing pytest mid-run.)
    """
    app = _app(monkeypatch, supervised=True, admins=[])
    r = _as_logged_in(app.test_client()).get("/api/admin/reload/available")
    assert r.status_code == 200
    assert r.get_json()["available"] is True, (
        "an operator who has already been named in allowed_users is being asked "
        "to name themselves a second time before their own server will offer a "
        "restart button"
    )


def test_naming_nobody_still_refuses_a_stranger(monkeypatch):
    """The default widens to everyone who SIGNED IN, and to nobody else.

    Anonymous is never an admin, whatever the config says -- that is the one
    rule with no exception, and it is what makes the default safe rather than
    merely convenient.
    """
    app = _app(monkeypatch, supervised=True, admins=[])
    r = app.test_client().post("/api/admin/reload")
    assert r.status_code == 403
    avail = app.test_client().get("/api/admin/reload/available")
    assert avail.get_json()["available"] is False


def test_a_logged_in_non_admin_is_refused(monkeypatch):
    """Named admins exist, and this session is not one of them."""
    app = _app(monkeypatch, supervised=True, admins=["boss@example.org"])
    r = _as_logged_in(app.test_client(), "someone@example.org").post(
        "/api/admin/reload")
    assert r.status_code == 403
    assert not (r.get_json() or {}).get("ok")


def test_an_anonymous_request_is_refused(monkeypatch):
    """No session at all is not an admin, whatever the config says."""
    app = _app(monkeypatch, supervised=True, admins=["boss@example.org"])
    r = app.test_client().post("/api/admin/reload")
    assert r.status_code == 403


# --------------------------------------------------------------------- #
#  What the page uses to decide whether to draw the button              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "supervised, admins, email, expected",
    [
        (True,  ["boss@example.org"], "boss@example.org",    True),
        (True,  ["boss@example.org"], "someone@example.org", False),
        # No section named: the default is anyone who could sign in, which the
        # required per-provider allowed_users has already narrowed to people an
        # operator listed by hand.
        (True,  [],                   "boss@example.org",    True),
        (False, ["boss@example.org"], "boss@example.org",    False),
    ],
    ids=["admin-on-supervised", "narrowed-out", "no-section-named", "no-supervisor"],
)
def test_availability_answers_honestly(monkeypatch, supervised, admins,
                                       email, expected):
    """`/api/admin/reload/available` says whether the button should be drawn.

    It is always present and always answers 200 -- "no" is not a refusal, it is
    the honest state of a server started without a supervisor, or of a caller a
    narrowing `admin` section leaves out.  A page that got a 403 here could not
    tell "you may not" from "the server is broken".
    """
    app = _app(monkeypatch, supervised=supervised, admins=admins)
    r = _as_logged_in(app.test_client(), email).get("/api/admin/reload/available")
    assert r.status_code == 200
    assert r.get_json()["available"] is expected


# --------------------------------------------------------------------- #
#  The protocol between the two processes                               #
# --------------------------------------------------------------------- #

def test_both_sides_read_the_exit_code_from_one_place():
    """One copy of the sentinel, or a reload quietly stops respawning.

    If the child asked with 3 and the supervisor waited for 4, the server would
    exit and stay down -- looking, from the browser, exactly like a crash.
    """
    import inspect
    from molbuilder import cli, reload_protocol
    from molbuilder.web import app as web_app

    assert reload_protocol.RELOAD_EXIT_CODE == 3
    for mod in (cli, web_app):
        src = inspect.getsource(mod)
        assert "RELOAD_EXIT_CODE" in src, f"{mod.__name__} lost the import"
        assert "reload_protocol import" in src, (
            f"{mod.__name__} does not take the sentinel from the one place "
            f"that defines it"
        )


def test_the_supervisor_respawns_only_on_the_sentinel(monkeypatch):
    """Respawn when the child asked to be restarted; stop for anything else.

    Ctrl-C, a crash, a port already in use -- each must END the supervisor and
    hand its code back.  A parent that respawns on every exit turns a bad port
    into an infinite restart loop that nothing but ``kill`` can stop.
    """
    from molbuilder import cli

    calls = []
    codes = iter([cli.RELOAD_EXIT_CODE, cli.RELOAD_EXIT_CODE, 7])

    def _fake_call(args, env=None, **kw):
        calls.append((args, env))
        return next(codes)

    monkeypatch.setattr("subprocess.call", _fake_call)
    monkeypatch.setattr("sys.argv", ["molbuilder", "serve", "--supervise",
                                     "--port", "8123"])

    assert cli._supervise_forever() == 7, "the child's real exit code is lost"
    assert len(calls) == 3, f"respawned {len(calls)} times, expected 3"


def test_the_child_is_told_it_is_the_child(monkeypatch):
    """SUPERVISED_ENV in the child's environment, or `--supervise` forks forever.

    The child runs the SAME command line, `--supervise` included.  That flag is
    what makes a process become a supervisor -- so without this variable the
    child would spawn its own child, endlessly.  The same variable is what lets
    `create_app` know a restart is possible and register the reload route.
    """
    from molbuilder import cli

    seen = {}

    def _fake_call(args, env=None, **kw):
        seen["args"] = args
        seen["env"] = env
        return 0

    monkeypatch.setattr("subprocess.call", _fake_call)
    monkeypatch.setattr("sys.argv", ["molbuilder", "serve", "--supervise"])
    monkeypatch.delenv(SUPERVISED_ENV, raising=False)

    cli._supervise_forever()
    assert seen["env"][SUPERVISED_ENV] == "1", (
        "the child was not told it is the child; it would become a second "
        "supervisor and fork without end"
    )
    assert seen["args"][1:3] == ["-m", "molbuilder"], (
        f"the child is not started as this package: {seen['args']!r}"
    )
    assert seen["args"][3:] == ["serve", "--supervise"], (
        "the child must run the same command line, so --port/--host/--cert "
        "survive a reload"
    )


def test_the_supervisor_does_not_import_the_app_it_restarts():
    """The property the whole design rests on.

    The supervisor's value is that a child which fails to import leaves it
    alive, so the next reload can fix the mistake.  That only holds while the
    parent never imports application code -- and it briefly did not: the
    protocol constants first lived at ``web/reload_protocol.py``, where reading
    them ran ``web/__init__.py`` and pulled in the whole app plus Flask.
    """
    import subprocess
    import sys

    probe = (
        "import sys, molbuilder.cli;"
        "print('app' if 'molbuilder.web.app' in sys.modules else 'clean');"
        "print('flask' if 'flask' in sys.modules else 'clean')"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["clean", "clean"], (
        f"importing the CLI pulled in the application: {out!r}. The supervisor "
        f"lives there, so a syntax error in the app would now take the "
        f"supervisor down with it and the site would stay dead."
    )


def _serve_probe(argv):
    """Run ``molbuilder serve <argv>`` with the fork stubbed out.

    Returns ``(forked, app_imported_at_fork_time)``.

    In a SUBPROCESS on purpose: whether ``molbuilder.web.app`` is in
    ``sys.modules`` is a property of the whole process, and any earlier test in
    the session may have imported it.  A clean interpreter is the only place
    the question has a truthful answer.
    """
    import json
    import subprocess
    import sys

    probe = f"""
import json, sys, subprocess
seen = []
def _fake_call(args, env=None, **kw):
    seen.append("molbuilder.web.app" in sys.modules)
    return 0
subprocess.call = _fake_call

# Stub Flask.run BEFORE invoking: the unsupervised path falls straight through
# to app.run(), which binds a socket and blocks forever.  Imported here rather
# than at the top so the module list stays honest -- flask is going to be
# imported by the app anyway on any path that reaches it, and the assertion
# above is about molbuilder.web.app, which this does not pull in.
import flask
flask.Flask.run = lambda self, *a, **k: None

from click.testing import CliRunner
from molbuilder import cli
CliRunner().invoke(cli.cli, {argv!r})
print(json.dumps([bool(seen), (seen[0] if seen else False)]))
"""
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_serve_supervises_by_default():
    """Plain ``molbuilder serve`` runs supervised.

    The flag was opt-in until 2026-08-04, which meant the Reload button was
    absent for anyone who had not read the help text -- and the reason to
    supervise (a restart is possible at all) applies to every ordinary run.
    """
    forked, _ = _serve_probe(["serve", "foreground", "--port", "0"])
    assert forked, "serve did not become a supervisor without --no-supervise"


def test_the_parent_forks_before_importing_the_application():
    """The property the design rests on, tested where it actually happens.

    ``test_the_supervisor_does_not_import_the_app_it_restarts`` only proves
    that IMPORTING the CLI is clean.  It never ran the parent branch -- and the
    branch was not clean: ``from .web.app import create_app`` sat one line into
    ``cmd_serve``, above the fork, so the supervisor imported the whole app and
    Flask before spawning anything.  A child that failed to import would have
    taken the parent with it, which is the exact failure the supervisor exists
    to prevent.
    """
    forked, app_imported = _serve_probe(["serve", "foreground", "--port", "0"])
    assert forked
    assert not app_imported, (
        "the supervisor imported molbuilder.web.app before forking; a broken "
        "app now kills the parent too and no reload can fix it"
    )


def test_no_supervise_runs_in_one_process():
    """The opt-out still opts out -- systemd, Docker and gunicorn already own
    restarts, and a supervisor inside one of them is a second answer to a
    question that has one."""
    forked, _ = _serve_probe(["serve", "foreground", "--port", "0", "--no-supervise"])
    assert not forked, "--no-supervise still forked a supervisor"


def test_debug_turns_supervision_off_on_its_own():
    """Werkzeug's reloader respawns its child on ANY exit, including the
    sentinel the reload route uses to ask for a fresh server -- so the request
    would never reach our supervisor.  Debug is single-process by nature."""
    forked, _ = _serve_probe(["serve", "foreground", "--port", "0", "--debug"])
    assert not forked, (
        "--debug started a supervisor; its Reload button would be swallowed "
        "by Werkzeug's own reloader"
    )
