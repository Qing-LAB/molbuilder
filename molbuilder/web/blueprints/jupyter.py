"""The Notebook tab's control surface: start, stop, and where to point a frame.

Contract: `docs/web/jupyter.md`; the exposure decision is
`docs/ops/access-control.md` § 6.

**WHY THESE ROUTES ARE ADMIN-ONLY, and why they can be absent entirely.**
A live kernel is arbitrary code execution as the account running the server --
the same capability `POST /api/admin/reload` guards, arrived at from the other
direction.  So it follows the same rule, and for the same reason:

* **no supervisor, no routes.**  The notebook is held by `serve`'s supervisor
  (§ 3.4), so without one there is nobody to start or stop it, and a button
  that cannot work is worse than an absent one.
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
    return os.environ.get(SUPERVISED_ENV) == "1"


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

    ``request.remote_addr`` is the socket peer: this app installs no
    ProxyFix, so no header can forge it.  The one deployment this misreads
    is a reverse proxy on the same host in front of an UNAUTHENTICATED
    molbuilder -- where every file endpoint is already exposed, which
    `_enforce_tls_for_remote_bind` calls "two attacks in one".
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

    Read from the request's own host rather than configured: a second
    molbuilder on another port has its own notebook, and asking the request
    is how each one finds its own (`jupyter.jupyter_port`).
    """
    from flask import request
    try:
        return int((request.host or "").rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 80


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
    st = status(port)
    st["may_control"] = bool(_supervised() and _may_control())
    st["supervised"] = _supervised()

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
    st["env_name"] = env_name
    st["env_installed"] = installed
    st["install_command"] = fix_cmd("install", recipe.name, "--yes")

    # THE TOKEN IS GATED BY THE SAME RULE AS START AND STOP.
    #
    # It authenticates a browser to a LIVE KERNEL, which is arbitrary code
    # execution as the account serving this page -- the same capability the
    # Start button hands out.  Returning it to every caller while refusing
    # them the button inverted the gate: a client who may not START a
    # notebook could USE one that was already running.  The earlier
    # justification here ("already behind the sign-in gate") does not hold in
    # the case `_may_control` itself contemplates -- an unauthenticated
    # molbuilder may legitimately be bound to a network interface, and then
    # there is no sign-in gate at all.  Found in review 2026-09-14.
    #
    # The tab needs it only to build the iframe URL, which is exactly the
    # thing a caller who may not control the notebook has no business doing.
    if not st["may_control"]:
        st["token"] = ""
    return jsonify({"ok": True, **st})


if os.environ.get(SUPERVISED_ENV) == "1":

    @bp.post("/api/jupyter/start")
    def api_jupyter_start():
        """Ask the supervisor to start it.  Admin only; idempotent."""
        import signal

        from ...serve_daemon import signal_supervisor

        if not _may_control():
            return jsonify({
                "ok": False,
                "error": ("admin auth required: a live kernel runs code as "
                          "the account serving this page.  Sign in -- or, if "
                          "an `admin` section in molbuilder.json names "
                          "addresses, sign in as one of them."),
            }), 403
        ok, msg = signal_supervisor(_serve_port(), signal.SIGUSR1)
        return jsonify({"ok": ok, "message": msg}), (202 if ok else 409)

    @bp.post("/api/jupyter/stop")
    def api_jupyter_stop():
        """Stop it, and every kernel with it (`jupyter.md` § 3.2)."""
        import signal

        from ...serve_daemon import signal_supervisor

        if not _may_control():
            return jsonify({"ok": False, "error": "admin auth required"}), 403
        ok, msg = signal_supervisor(_serve_port(), signal.SIGUSR2)
        return jsonify({"ok": ok, "message": msg}), (202 if ok else 409)
