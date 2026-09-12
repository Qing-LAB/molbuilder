# Jupyter — a live notebook tab, and the lifecycle that keeps it honest

**Role:** contract
**Domain:** web
**Companions:** [`ops/deployment.md`](?doc=ops/deployment.md) § 1.0a–1.0c — the
supervisor discipline this copies (pidfile, verify-before-signal, a status that
answers two questions); [`ops/installation.md`](?doc=ops/installation.md) — the
per-backend env model and the recipe registry;
[`ops/access-control.md`](?doc=ops/access-control.md) — what exposing a kernel
means; [`overview.md`](?doc=web/overview.md) — the tab registry.

## The short version

| rule | where |
|---|---|
| **Flask cannot carry the kernel.** The server is plain WSGI; Jupyter kernels talk WebSocket. The tab is an **iframe to a separately-run Jupyter**, and the socket goes browser→Jupyter directly | § 2 |
| **The kernel is a grandchild.** Killing the Jupyter server alone leaves `ipykernel` processes holding memory and GPUs. Every stop takes the **process group** | § 3.2 |
| **`kill -9` runs no handler.** `PR_SET_PDEATHSIG` is the only mechanism that survives it; the pidfile reconciliation is what catches the rest | § 3.1, § 3.3 |
| **Parented to the SUPERVISOR, not the server child.** A code reload must not kill your notebook; stopping molbuilder must | § 3.4 |
| **Nothing runs until asked.** No kernel on page load; idle kernels are culled by Jupyter's own timeouts | § 4 |
| **A live kernel is arbitrary code execution on a network port.** That is a decision, not a side effect of adding a tab | § 6 |

```
   browser ──HTTP──▶ molbuilder (Flask/WSGI, :8888)   tab shell, control API
      │
      └──WebSocket──▶ jupyter server (:8889)  ◀── started/stopped by molbuilder
                            │
                            └── ipykernel, ipykernel, …   (the grandchildren)
```

---

## 1. What this document owns

The **shape** of the integration: which process talks to which, who starts and
stops what, and what must be true when molbuilder exits. It does not own
Jupyter's own configuration surface, nor the recipe's package list — that is
`recipes.py` and `installation.md`.

## 2. Why an iframe, and not a proxy

molbuilder serves through Werkzeug's `ThreadedWSGIServer`
(`cli.py`, `app.run(ssl_context=…)`). **WSGI has no WebSocket**, and there is
no `flask-sock`, socketio or gevent in the tree. Jupyter's kernel protocol is
WebSocket. So Flask cannot proxy the kernel connection, and the tab must point
a frame at Jupyter directly.

Two consequences that are not optional:

* **TLS on the Jupyter port too.** The app page is HTTPS, so an `http://` frame
  is blocked as mixed content. Jupyter reuses the cert/key `serve` already
  loads.
* **Framing must be allowed.** Jupyter sends `X-Frame-Options` / a CSP
  `frame-ancestors` by default and will refuse to be framed;
  `ServerApp.tornado_settings` has to name molbuilder's origin.

The rejected alternative — replacing the WSGI server with gunicorn+gevent or an
ASGI stack — collides with the TLS hardening bolted onto `ThreadedWSGIServer`
(`cli.py`, `_molbuilder_tls_hardened`), and buys nothing the iframe does not.

## 3. The lifecycle

Copied in discipline from `serve_daemon.py`, which already solves this shape for
molbuilder itself: a pidfile, a verify-before-signal check, and a `status` that
answers *process up* and *answering* separately.

### 3.1 `PR_SET_PDEATHSIG` — the only thing that survives `kill -9`

A graceful exit can run handlers. **`SIGKILL` cannot**, and the child is
reparented to init and survives. The Linux `prctl(PR_SET_PDEATHSIG, SIGTERM)`
is set in the child, so the kernel signals it when its parent dies — including
when the parent was killed outright. Linux-only, which matches the project's
stated platform.

### 3.2 A process group, because the kernels are grandchildren

The Jupyter **server** is the child; the `ipykernel` processes it spawns are
grandchildren, and they are what hold memory and GPUs. Stopping the server
alone routinely leaves them. The child is started in its own session
(`start_new_session=True`) and stopped with `killpg`, so the whole tree goes.

This composes with § 3.1: `PR_SET_PDEATHSIG` keys on *parent death*, not on
group membership, so a new session does not disable it.

### 3.3 Reconciliation at startup

Layers 1 and 2 cannot cover a machine crash, or a survivor that was re-parented
before the signal landed. On startup molbuilder reads the pidfile and, **only
if the pid is alive and its `/proc/<pid>/cmdline` shows a Jupyter we started**,
stops it. Same rule `serve_daemon` states: a stale file whose pid was recycled
is reported stale, never signalled.

### 3.4 Parented to the supervisor

molbuilder's supervisor **respawns the server child on `RELOAD_EXIT_CODE`**. If
Jupyter were a child of that server process, `PDEATHSIG` would kill it on every
reload — an unrelated code change would silently destroy a notebook's state.

So Jupyter is parented to the **supervisor**:

| stopping | Jupyter |
|---|---|
| a code reload | **survives** |
| `serve stop` / supervisor exit | **dies** |
| `kill -9` of the supervisor | **dies** (§ 3.1) |

## 4. Nothing runs until asked

No kernel starts on page load. The tab starts Jupyter on first open or on an
explicit control action, and Jupyter's own settings shrink the idle window
rather than molbuilder hand-rolling one:
`MappingKernelManager.cull_idle_timeout` reaps idle kernels, and
`ServerApp.shutdown_no_activity_timeout` stops the server when nobody is using
it.

## 5. The control surface

Mirrors the verbs that already exist, so it is debuggable without a browser:

```
molbuilder jupyter start | stop | restart | status
```

plus a blueprint the tab calls. `status` answers **two** questions separately —
*process up* and *answering its own endpoint* — for the reason
`deployment.md` gives: the wedge worth catching is a server that is up and not
answering.

## 6. What this exposes

A live kernel is **arbitrary code execution**, reachable on a network port, in
an env with the project's science stack. That is a deliberate decision and
belongs in `access-control.md` beside the rest of the exposure story — not an
implication of having added a tab. Minimum: bind to loopback unless the
operator states otherwise, and keep Jupyter's own token auth on.

## 7. The env

`molbuilder-jupyternb`, one recipe in the registry like every other backend
(`installation.md` § 1). It is not the host env: the notebook stack pins its
own dependencies, and the isolation rule is the whole point of the per-backend
model.

## 8. Status

**Designed, not built** *(2026-09-11)*. The decision recorded here is § 3.4 —
parented to the supervisor — taken by the user against the two alternatives
(dying with the server child, or fully independent with its own pidfile).
