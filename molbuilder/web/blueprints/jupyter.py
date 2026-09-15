"""The Notebook tab's control surface: start, stop, and where to point a frame.

Contract: `docs/web/jupyter.md`; the exposure rule is that document's § 6.
(It said `access-control.md` § 6 until 2026-09-14, which is about
`/api/admin/reload` and contains no mention of a notebook -- the rule that
actually ships, including the loopback clause below, was only ever in this
docstring.)

**WHY THESE ROUTES ARE ADMIN-ONLY, and why they can be absent entirely.**
A live kernel is arbitrary code execution as the account running the server --
the same capability `POST /api/admin/reload` guards, arrived at from the other
direction.  So it follows the same rule, and for the same reason:

* **no supervisor, no control.**  The notebook is held by `serve`'s supervisor
  (§ 3.4), so without one there is nobody to start or stop it, and a button
  that cannot work is worse than an absent one.

  **ONE GATE, ASKED PER REQUEST** (`plan.md` § 5n, J3).  There were two until
  2026-09-15.  Registration itself sat behind the SUPERVISED env var, read at
  import -- and that variable means *somebody can respawn me*, which
  `serve foreground` also sets while writing no pidfile and installing no
  handlers.  So `_supervised()` was added per request to ask the real
  question, and the module ended up with two gates on two different facts for
  one rule.  The import-time one had a second cost: it made the app's URL MAP
  depend on the environment `create_app()` happened to run in, so no test
  could reach these routes at all.

  The routes are therefore always registered, and `_refuse()` answers **404**
  when `_supervised()` says no.  The same thing a client sees, decided when
  the answer is knowable -- and now with a sentence saying which run mode
  this is, instead of Flask's bare HTML page.
* **the `admin` list decides who may press it.**  Signing in is not enough:
  reaching a session already required being in a provider's ``allowed_users``,
  but *running code on the server* is the privilege § 5 separates.
* **404 rather than 403** for the absent case, because a misconfiguration then
  reads as *the button is missing* -- which is what it is -- and never as
  *anyone may start a kernel here*.

`status` is different and is always registered: answering *"is there a
notebook, and may I use it"* is not itself a capability, and the page needs an
honest answer to decide what to draw.  It reports, and it never starts
anything.
"""
from __future__ import annotations

import os

from flask import Blueprint, jsonify

from ...reload_protocol import SUPERVISED_ENV

bp = Blueprint("jupyter", __name__)


def _supervised() -> bool:
    """Is there a supervisor that can actually be ASKED to hold a notebook?

    `SUPERVISED_ENV` alone is the wrong question, and answering it was a real
    defect.  `reload_protocol`'s own docstring says what that flag means --
    *somebody can respawn me* -- and TWO supervisors set it: `serve start`'s
    `serve_daemon.supervise`, which writes a pidfile and installs the two USR
    handlers, and `serve foreground`'s `cli._supervise_forever`, which does
    neither.  Under the second, this returned True, the tab drew a
    Start button, and the click came back `not running (no pidfile at ...)` --
    which reads as a broken molbuilder rather than as a run mode that has no
    notebook.  Found in review 2026-09-14.

    So the probe is the thing the Start button actually depends on: a serve
    pidfile naming a live serve of ours, which is exactly what
    `signal_supervisor` will go looking for.
    """
    if os.environ.get(SUPERVISED_ENV) != "1":
        return False
    from ...serve_daemon import pid_state, read_pid
    return pid_state(read_pid(_serve_port())) == "ours"


def _may_control() -> bool:
    """May THIS request start and stop the notebook?

    Two ways to yes, and the second is not a loosening of the first.

    1. **The `admin` list says so** -- the rule `POST /api/admin/reload`
       follows, and the one that applies whenever this server has sign-in.
       Running code on the machine everyone shares is the privilege
       `access-control.md` § 5 separates from merely being signed in.

    2. **There is NO sign-in configured and the request came from this
       machine.**  On such a server the admin list can never answer yes --
       `is_admin_request` requires a session and no session can exist -- so
       the capability would be unreachable from the browser on every
       developer's own machine, which is where notebooks are actually used.
       It is not a hole: a client on the loopback interface has a shell on
       this account already and can run `molbuilder jupyter start` (or
       anything else) directly, so the button grants nothing it did not
       have.  And with no sign-in configured every file endpoint is open to
       that same client regardless.

    The second way is deliberately narrow: it requires BOTH conditions.  A
    server with sign-in falls to rule 1 even on loopback, and a server
    without sign-in gives nobody off-machine the button -- which matters
    because "no providers" does NOT force a loopback bind (only the explicit
    `--no-auth` flag does), so an unauthenticated molbuilder can legitimately
    be listening on a network interface.

    ``request.remote_addr`` is the socket peer, so no header can forge it --
    **and that holds for a narrower reason than "this app installs no
    ProxyFix", which is what this said until 2026-09-15.**  `auth.trust_proxy`
    DOES install one (`web/auth.py`).  It is still true here, by the
    condition rule 2 already requires: ProxyFix is installed by `init_auth`,
    which only runs when providers are configured, and rule 2 only runs when
    there are NONE.  So the two cannot coexist -- but the reason is the
    condition, not the absence of the middleware
    (`plan.md` § 5n.8).

    The one deployment this misreads is a reverse proxy on the same host in
    front of an UNAUTHENTICATED molbuilder -- where every file endpoint is
    already exposed, which `_enforce_tls_for_remote_bind` calls "two attacks
    in one".
    """
    import ipaddress

    from flask import current_app, request

    from ..admin import is_admin_request

    if is_admin_request():
        return True
    from ..auth import sign_in_is_configured
    if sign_in_is_configured(current_app):
        return False        # sign-in IS configured: rule 1 is the only way
    try:
        peer = ipaddress.ip_address(request.remote_addr or "")
    except ValueError:
        return False
    return peer.is_loopback


def _serve_port() -> int:
    """The port THIS server is on -- the notebook's is derived from it.

    ONE HOME, in `web.app`.  This parsed `request.host` itself and fell back
    to **80** while `app.py` parsed the same header and fell back to **0** --
    two answers to one question, and the notebook feature simply does not
    work behind the reverse proxy `deployment.md` recommends, because the
    public host's port is not the port the supervisor is keyed by.
    """
    from ..app import serve_port
    return serve_port()


@bp.get("/api/jupyter/status")
def api_jupyter_status():
    """What the tab needs to draw itself, and nothing it does not.

    **NOTHING IS STARTED BY ASKING** (`jupyter.md` § 4: nothing runs until
    asked).  This reports; the page decides what to offer, and a person
    decides whether to start anything.

    Four states the tab has to tell apart, which is why the env is probed
    here rather than inferred from a failure:

    * the env is not installed -- an OPT-IN env (`recipes.py`), so this is an
      ordinary state and the answer is an install command, not a Start button
      that would fail;
    * installed, nothing running -- offer to start it;
    * running, not answering yet -- starting, or wedged; the log says which;
    * answering -- frame it.

    **The token is included ONLY for a caller who may control the notebook**
    -- see the gate at the end of this function.  Reaching a live kernel is
    the same code execution as starting one, so it is the same privilege.
    It is not written into the page by the server and not logged; the tab puts
    it in the iframe URL, which is where Jupyter expects it.
    """
    from ...diagnostics import get_capabilities
    from ...envs.hints import fix_cmd
    from ...envs.recipes import effective_name, recipe_by_name
    from ...jupyter import status

    port = _serve_port()
    # ONE CALL EACH.  `_supervised()` reads a pidfile and stats /proc, and it
    # was asked twice per request on a polling endpoint.
    supervised = _supervised()
    may_control = bool(supervised and _may_control())
    # The token and the open-notebook paths are never even PRODUCED for a
    # caller who may not control the notebook -- see `status`.
    st = status(port, include_private=may_control)

    # THE ENV, PROBED -- the browser cannot ask conda anything.
    recipe = recipe_by_name("molbuilder-jupyternb")
    caps = get_capabilities()
    env_name = effective_name(recipe, caps)
    installed = bool(caps.env_available(env_name))
    if not installed:
        # ASK THE MANAGER AGAIN before saying "not installed".
        #
        # The snapshot is bound ONCE per process (`diagnostics`), and the web
        # server is long-lived: an env created after it started is invisible
        # to it.  Measured 2026-09-14 -- the env was installed, a fresh probe
        # saw it, and this endpoint still said no, so the tab told a person to
        # install what they had just installed.
        #
        # Only on the NEGATIVE, and that is the whole economy of it: when the
        # snapshot says yes there is nothing to correct, so the polling path
        # costs nothing, and the answer that would send somebody to a terminal
        # for no reason is the one that gets checked.  `conda_env_prefixes` is
        # the one registry reader (M2).
        from ...diagnostics import conda_env_prefixes
        if caps.conda_binary:
            try:
                installed = env_name in conda_env_prefixes(caps.conda_binary)
            except Exception:  # noqa: BLE001 - a manager that will not answer
                pass           # leaves the snapshot's answer standing
    from ...jupyter import port_clash

    # THE WHOLE PAYLOAD, IN ONE EXPRESSION (`plan.md` § 5n, J5).
    #
    # Nine keys come from `jupyter.status()`, six are added here and `ok` is
    # the envelope -- and until 2026-09-15 the six arrived as scattered
    # `st[...] = ` mutations, so the shape a page depends on existed in no
    # single place and in no document.  `jupyter.md` § 5 is the contract;
    # this is the one assembly, and the two are meant to be read together.
    #
    # TWO THINGS ARE WITHHELD, and not by deleting them afterwards.
    # `include_private=False` means `token` and `open` are never PRODUCED
    # for a caller who may not control the notebook -- the difference between
    # a credential that was never built and one whose safety depends on a
    # later line in a function that will grow.  The token authenticates a
    # browser to a LIVE KERNEL, which is the same arbitrary code execution
    # the Start button hands out; returning it to every caller while refusing
    # them the button inverted the gate, and the old justification ("already
    # behind the sign-in gate") does not hold in the case `_may_control`
    # itself contemplates -- an unauthenticated molbuilder legitimately bound
    # to a network interface has no sign-in gate at all.
    #
    # `port_clash` is gated too: it names another server on this machine.
    return jsonify({
        "ok":              True,
        **st,                       # running · pid · pid_state · port ·
                                    # answering · url · token · open ·
                                    # workspace_saved
        "supervised":      supervised,
        "may_control":     may_control,
        "env_name":        env_name,
        "env_installed":   installed,
        "install_command": fix_cmd("install", recipe.name, "--yes"),
        "port_clash":      port_clash(port) if may_control else None,
    })


def _refuse():
    """The ONE refusal for the two control routes, or ``None`` to proceed.

    Returns a ready ``(body, status)`` -- 404 when there is no supervisor to
    ask, 403 when there is one and this caller may not press the button.

    **One sentence, not two.**  `start` used to answer with three lines
    naming `molbuilder.json`'s `admin` section while `stop` answered
    ``"admin auth required"`` -- the same gate, the same condition, two
    answers, and the short one told a person nothing about what to do
    (`plan.md` § 5n, J4).
    """
    if not _supervised():
        return jsonify({
            "ok": False,
            "error": ("this molbuilder has no supervisor to hold a notebook: "
                      "it was started with `serve foreground` or "
                      "`--no-supervise`.  `molbuilder serve start` gives it "
                      "one."),
        }), 404
    if not _may_control():
        return jsonify({
            "ok": False,
            "error": ("admin auth required: a live kernel runs code as the "
                      "account serving this page.  Sign in -- or, if an "
                      "`admin` section in molbuilder.json names addresses, "
                      "sign in as one of them."),
        }), 403
    return None


@bp.post("/api/jupyter/start")
def api_jupyter_start():
    """Ask the supervisor to start it.  Admin only; idempotent."""
    import signal

    from ...serve_daemon import signal_supervisor

    refused = _refuse()
    if refused is not None:
        return refused
    ok, msg = signal_supervisor(_serve_port(), signal.SIGUSR1)
    return jsonify({"ok": ok, "message": msg}), (202 if ok else 409)


@bp.post("/api/jupyter/stop")
def api_jupyter_stop():
    """Stop it, and its kernels with it (`jupyter.md` § 3.2).

    The kernels go because the SERVER goes -- Jupyter collects them
    itself -- not because any signal of ours reaches them.
    """
    import signal

    from ...serve_daemon import signal_supervisor

    refused = _refuse()
    if refused is not None:
        return refused
    ok, msg = signal_supervisor(_serve_port(), signal.SIGUSR2)
    return jsonify({"ok": ok, "message": msg}), (202 if ok else 409)
