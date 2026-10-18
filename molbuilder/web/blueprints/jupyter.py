"""The Notebook tab's control surface -- start, stop, and where to point a
frame -- and the notebook itself, passed through at ``/jupyter/``.

Contract: `docs/web/jupyter.md`; the pass-through is that document's § 2, the
exposure rule its § 6.

**WHY THESE ROUTES ARE ADMIN-ONLY, and why they can be absent entirely.**
A live kernel is arbitrary code execution as the account running the server --
the same capability `POST /api/admin/reload` guards, arrived at from the other
direction.  So it follows the same rule, and for the same reason:

* **no supervisor, no control.**  The notebook is held by `serve`'s supervisor
  (§ 3.4), so without one there is nobody to start or stop it, and a button
  that cannot work is worse than an absent one.

  **ONE GATE, ASKED PER REQUEST** (`plan.md` § 5n, J3).  The routes are
  always registered, and `_refuse()` answers **404** when `_supervised()`
  says no -- with a sentence saying which run mode this is, instead of
  Flask's bare HTML page.
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

**THE PASS-THROUGH ASKS THE SAME GATE** (`jupyter.md` § 2, § 6): reaching a
running kernel is the same code execution as starting one, so every request
under ``/jupyter/`` is refused exactly as the buttons are before anything
reaches Jupyter.
"""
from __future__ import annotations

import os

from flask import Blueprint, Response, jsonify, request

from ...jupyter import BASE_PATH
from ...reload_protocol import SUPERVISED_ENV

bp = Blueprint("jupyter", __name__)


def _supervised() -> bool:
    """Is there a supervisor that can actually be ASKED to hold a notebook?

    `SUPERVISED_ENV` alone is the wrong question.  `reload_protocol`'s own
    docstring says what that flag means --
    *somebody can respawn me* -- and TWO supervisors set it: `serve start`'s
    `serve_daemon.supervise`, which writes a pidfile and installs the two USR
    handlers, and `serve foreground`'s `cli._supervise_forever`, which does
    neither.

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

    ``request.remote_addr`` is the socket peer, so no header can forge it.
    `auth.trust_proxy` DOES install a ProxyFix (`web/auth.py`); it holds
    here by the condition rule 2 already requires: ProxyFix is installed by
    `init_auth`,
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
    return peer.is_loopback and _asked_by_a_loopback_name()


def _asked_by_a_loopback_name() -> bool:
    """Was this page asked for by a LOOPBACK NAME -- ``localhost``, or a
    loopback address -- as the browser's own ``Host`` says?

    A loopback PEER is not enough on its own: a web page from anywhere whose
    name its owner points at 127.0.0.1 (DNS rebinding) is fetched by the
    browser on this machine, so its requests arrive from loopback too -- and
    rule 2 would hand it the notebook.  What tells it apart is the name it
    was asked by, which is that page's own (`jupyter.md` § 6).
    """
    import ipaddress
    from urllib.parse import urlsplit

    from flask import request
    name = urlsplit(f"//{request.host}").hostname or ""
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _serve_port() -> int:
    """The port THIS server is on -- the notebook's is derived from it.

    ONE HOME, in `web.app`.
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

    **The notebook's token is not in it.**  The pass-through presents it to
    Jupyter on every request it lets through (`jupyter.md` § 2), so molbuilder
    never hands it to the browser -- not here, not in the frame's address,
    not in its log.
    """
    from ...diagnostics import get_capabilities
    from ...envs.hints import fix_cmd
    from ...envs.recipes import effective_name, recipe_by_name
    from ...jupyter import status

    port = _serve_port()
    # ONE CALL EACH.  `_supervised()` reads a pidfile and stats /proc, on a
    # polling endpoint.
    supervised = _supervised()
    may_control = bool(supervised and _may_control())
    # The open-notebook paths are never even PRODUCED for a caller who may
    # not control the notebook -- see `status`.
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
        # to it.
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
    # Eight keys come from `jupyter.status()`, seven are added here and `ok`
    # is the envelope.  `jupyter.md` § 5 is the contract;
    # this is the one assembly, and the two are meant to be read together.
    #
    # `open` IS WITHHELD, and not by deleting it afterwards:
    # `include_private=False` means it is never PRODUCED for a caller who may
    # not control the notebook -- the paths of somebody's open notebooks.
    # `port_clash` is gated too: it names another server on this machine.
    return jsonify({
        "ok":              True,
        **st,                       # running · pid · pid_state · port ·
                                    # answering · url · open ·
                                    # workspace_saved
        "base":            BASE_PATH,
        "supervised":      supervised,
        "may_control":     may_control,
        "env_name":        env_name,
        "env_installed":   installed,
        "install_command": fix_cmd("install", recipe.name, "--yes"),
        "port_clash":      port_clash(port) if may_control else None,
    })


def _refuse():
    """The ONE refusal for the control routes and the pass-through, or
    ``None`` to proceed.

    Returns a ready ``(body, status)`` -- 404 when there is no supervisor to
    ask, 403 when there is one and this caller may not control the notebook,
    or the request comes from another page (`jupyter.md` § 6).

    **One sentence for every route it guards** (`plan.md` § 5n, J4): the
    same gate and the same condition give the same answer.
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
    if not _from_this_page():
        return jsonify({
            "ok": False,
            "error": ("refused: this request comes from another page "
                      "(its Origin), and only this one may drive the "
                      "notebook."),
        }), 403
    return None


def _from_this_page() -> bool:
    """Does the request come from this page -- no ``Origin``, or one naming
    the host it was sent to?

    Jupyter makes this check itself, and skips it for a request it sees as
    token-authenticated -- every request the pass-through forwards -- so the
    gate makes it (`jupyter.md` § 6).
    """
    from urllib.parse import urlsplit
    origin = request.headers.get("Origin")
    return origin is None or urlsplit(origin).netloc == request.host


@bp.post("/api/jupyter/start")
def api_jupyter_start():
    """Ask the supervisor to start it.  Admin only; idempotent."""
    import signal

    from ...serve_daemon import signal_supervisor

    refused = _refuse()
    if refused is not None:
        return refused
    # THE FILE THE SHEPHERD WILL READ, read first -- as `jupyter start` does
    # on the command line (`jupyter.md` § 5): a broken molbuilder.json is
    # refused here in its words, not in the notebook's log after the signal.
    from ...runtime_config import RuntimeConfigError, read_config
    try:
        read_config()
    except RuntimeConfigError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
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


# --------------------------------------------------------------------- #
#  The pass-through -- the notebook on molbuilder's own address          #
# --------------------------------------------------------------------- #

#: Headers about ONE connection, never forwarded (RFC 9110 § 7.6.1) -- the
#: upgrade a WebSocket asks for is carried by the join, not by these.
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate",
                         "proxy-authorization", "te", "trailer",
                         "transfer-encoding", "upgrade"})

_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

#: How long a forwarded request may wait on Jupyter.  A request, not a
#: kernel: the kernel's traffic is the WebSocket, which has no such bound.
_FORWARD_TIMEOUT_S = 300


@bp.route(f"{BASE_PATH}/", defaults={"rest": ""}, methods=_METHODS)
@bp.route(f"{BASE_PATH}/<path:rest>", methods=_METHODS)
def pass_through(rest):
    """Every ordinary request under ``/jupyter/``, passed to the notebook on
    this machine (`jupyter.md` § 2) -- refused first, as the buttons are
    (§ 6)."""
    refused = _refuse()
    if refused is not None:
        return refused
    from ...jupyter import jupyter_port
    return _forward(jupyter_port(_serve_port()))


# A WEBSOCKET IS ITS OWN ROUTE: the router answers a WebSocket handshake
# that matches only an ordinary route with 400 before any view runs
# (Werkzeug's `WebsocketMismatch`), so the kernel's channels need a rule
# that says it takes one.
@bp.route(f"{BASE_PATH}/<path:rest>", websocket=True,
          endpoint="pass_through_websocket")
def pass_through_websocket(rest):
    """A WebSocket under ``/jupyter/`` -- the kernel's channels -- joined to
    the notebook on this machine (`jupyter.md` § 2), refused first (§ 6)."""
    refused = _refuse()
    if refused is not None:
        return refused
    from ...jupyter import jupyter_port
    return _join(jupyter_port(_serve_port()))


def _target() -> str:
    """The path and query the browser asked for, as it spelled them -- the
    server's raw URI, so an escaped ``/`` in a file name stays one."""
    return request.environ["RAW_URI"]


#: Headers a browser may send that would speak for this pass-through: who is
#: asking (``Authorization``, presented here as the notebook's own token) and
#: the scheme and address Jupyter is told to trust from it alone.
_SAID_HERE = frozenset({"authorization", "x-forwarded-proto",
                        "x-forwarded-for", "x-forwarded-host", "x-real-ip",
                        "x-scheme"})


def _headers_for_jupyter():
    """The request's headers as Jupyter is given them: the hop-by-hop ones
    dropped, molbuilder's own session cookie taken out of ``Cookie`` -- it
    is molbuilder's credential, not Jupyter's -- the notebook's token
    presented, since this request has passed § 6's gate, and the scheme the
    browser used said (``X-Forwarded-Proto``, which Jupyter is told to
    trust: nothing but this pass-through reaches it)."""
    from flask import current_app

    from ...jupyter import read_runtime
    ours = current_app.config.get("SESSION_COOKIE_NAME") or "session"
    out = []
    for name, value in request.headers.items():
        low = name.lower()
        # `Content-Length` is the client library's own, from the body sent.
        if low in _HOP_BY_HOP or low in _SAID_HERE or low == "content-length":
            continue
        if low == "cookie":
            kept = [c for c in value.split(";")
                    if c.strip().split("=", 1)[0] != ours]
            if not kept:
                continue
            value = ";".join(kept).strip()
        out.append((name, value))
    token = str(read_runtime(_serve_port()).get("token") or "")
    if token:
        out.append(("Authorization", f"token {token}"))
    out.append(("X-Forwarded-Proto", request.scheme))
    return out


def _not_answering():
    return jsonify({
        "ok": False,
        "error": ("no notebook is answering on this machine -- start it "
                  "from the JupyterNB tab, or `molbuilder jupyter start`."),
    }), 502


def _framed(headers) -> None:
    """Every answer passed through may be framed by its own origin, which is
    molbuilder's page, and by nothing else (`jupyter.md` § 2): Jupyter's
    ``frame-ancestors`` where it sent none, and ``X-Frame-Options:
    SAMEORIGIN`` in place of the ``DENY`` molbuilder's own pages send."""
    if "Content-Security-Policy" not in headers:
        headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    headers["X-Frame-Options"] = "SAMEORIGIN"


def _forward(port: int):
    """An ordinary request: sent to Jupyter, and its answer streamed back
    with its status and headers."""
    import http.client

    from werkzeug.datastructures import Headers

    from ...jupyter import LOOPBACK
    conn = http.client.HTTPConnection(LOOPBACK, port,
                                      timeout=_FORWARD_TIMEOUT_S)
    try:
        conn.request(request.method, _target(),
                     body=request.get_data() or None,
                     headers=dict(_headers_for_jupyter()))
        answer = conn.getresponse()
    except OSError:
        conn.close()
        return _not_answering()

    def body():
        while chunk := answer.read1(65536):
            yield chunk

    # NOTHING OUTSIDE THE PAGE IS ANSWERED: Jupyter grants a token-
    # authenticated caller cross-origin access, and every request here is
    # one (`jupyter.md` § 2).
    headers = Headers([(k, v) for k, v in answer.getheaders()
                       if k.lower() not in _HOP_BY_HOP
                       and not k.lower().startswith("access-control-")])
    _framed(headers)
    passed = Response(body(), status=answer.status, headers=headers)
    passed.call_on_close(conn.close)        # closed on every path
    return passed


class _Joined(Response):
    """What a joined WebSocket hands back: nothing to write.  The connection
    was the browser's and Jupyter's while they talked, and it is over --
    raising ``ConnectionError`` is how the server it runs on is told a
    request's connection is gone, so it writes no answer of its own onto it
    (Werkzeug's ``connection_dropped``)."""

    def __call__(self, environ, start_response):
        raise ConnectionError("the WebSocket was joined and has ended")


def _join(port: int):
    """A WebSocket: the browser's own connection joined to one to Jupyter,
    the upgrade request sent on, and bytes copied both ways until either end
    closes (`jupyter.md` § 2)."""
    import socket

    from ...jupyter import LOOPBACK
    client = request.environ["werkzeug.socket"]
    try:
        upstream = socket.create_connection((LOOPBACK, port), timeout=10)
    except OSError:
        return _not_answering()
    upstream.settimeout(None)
    head = [f"{request.method} {_target()} HTTP/1.1"]
    head += [f"{k}: {v}" for k, v in _headers_for_jupyter()]
    head += ["Connection: Upgrade", "Upgrade: websocket"]
    upstream.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
    _copy_both_ways(client, upstream)
    return _Joined()


def _copy_both_ways(client, upstream) -> None:
    """Copy bytes between the two connections until either closes.

    A TLS connection may hold bytes it has already decrypted, which the
    readiness check cannot see -- so those are read first."""
    import selectors
    import socket
    import ssl

    sel = selectors.DefaultSelector()
    sel.register(client, selectors.EVENT_READ)
    sel.register(upstream, selectors.EVENT_READ)
    try:
        while True:
            if isinstance(client, ssl.SSLSocket) and client.pending():
                ready = [client]
            else:
                ready = [key.fileobj for key, _ in sel.select()]
            for src in ready:
                dst = upstream if src is client else client
                try:
                    data = src.recv(65536)
                except ssl.SSLWantReadError:
                    continue
                if not data:
                    return
                dst.sendall(data)
    except OSError:
        return
    finally:
        sel.close()
        upstream.close()
        # AND THE BROWSER IS TOLD: the server it runs on would otherwise
        # read the next bytes on this connection as a new HTTP request, and
        # a browser whose kernel went away would never hear of it.
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
