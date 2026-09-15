# Jupyter — the JupyterNB tab, and the lifecycle that keeps it honest

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
| **Stopping means stopping the SERVER.** The tree is supervisor → shepherd → the manager's `run` → jupyter-server, so a stop takes the shepherd's whole process group. The kernels are not in that group and do not need to be: Jupyter collects its own | § 3.2 |
| **`kill -9` runs no handler.** `PR_SET_PDEATHSIG` is the only mechanism that survives it; the pidfile reconciliation is what catches the rest | § 3.1, § 3.3 |
| **Parented to the SUPERVISOR, not the server child.** A code reload must not kill your notebook; stopping molbuilder must | § 3.4 |
| **Nothing runs until asked.** No kernel on page load; idle kernels are culled by Jupyter's own timeouts | § 4 |
| **A live kernel is arbitrary code execution on a network port.** That is a decision, not a side effect of adding a tab | § 6 |

```
   browser ──HTTP──▶ molbuilder (Flask/WSGI, :8888)   tab shell, control API
      │                       │
      │                       └── shepherd ── <mgr> run ── jupyter server
      │                           └───────────────────────────┘
      │                             one process group: a stop takes all three
      │
      └──WebSocket──▶ jupyter server (:8889)  ◀── started/stopped by molbuilder
                            │
                            └── ipykernel, ipykernel, …
                                each in its OWN session -- Jupyter collects
                                them, molbuilder's signal does not reach them
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

* **The notebook takes the SAME scheme the app page is served with.** A
  browser blocks an `http://` frame inside an `https://` page as mixed
  content, so the two must match — Jupyter reuses the cert and key `serve`
  resolved, and runs plain http when `serve` has none. It does **not** refuse
  to start without TLS; this section said it did until 2026-09-14, while a
  plain-http molbuilder had been running one the whole time. What the rule
  does require is that the resolution be the SAME on both sides: `serve start`
  resolves TLS from the `tls` block in `molbuilder.json` as well as the
  command line, and until the same day it handed the notebook only the raw
  flags — so a machine configured in the file served an https page and framed
  an http notebook.
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

### 3.2 A process group, so the stop reaches the SERVER

The tree is four deep: molbuilder's supervisor starts a **shepherd**, the
shepherd enters the env through the manager's `run`, and that runs
**jupyter-server**. Signalling any one of those is not signalling the others,
so the shepherd is started in its own session (`start_new_session=True`) and
takes its whole group down at once.

**The group does NOT reach the kernels — and it does not need to.**
`jupyter_client` launches every kernel with `start_new_session=True`, so each
kernel is its own session and group leader and no `killpg` of molbuilder's
ever touches it. Measured 2026-09-14: a kernel's process group was 2152481
against the shepherd's 2141249.

Jupyter owns that half, and owns it twice. A graceful `SIGTERM` makes the
server shut its own kernels down. If the server dies with no handler at all,
`ipykernel`'s parent poller sees `JPY_PARENT_PID` vanish and each kernel exits
by itself — measured the same day by `kill -9` on the server, with the kernel
gone in about a second and nothing orphaned.

So what this layer guarantees is that **the server dies**, and the server
dying is what collects the kernels. *(This section claimed the group reached
the kernels directly, and a `/proc`-walking reaper was drafted to make that
claim true before the measurement showed the backstop already works. Building
it would have been a third copy of something Jupyter does and passes.)*

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

## 4. Nothing runs until asked — **not even by looking**

Opening the tab **probes and reports**; it starts nothing. Starting a notebook
runs code as the account serving the page, so it is a thing a person asks for,
with a button. *(Decided with the user 2026-09-13, against the earlier "starts
on first open" written here: a tab that runs code because you clicked its name
is a tab you cannot look at safely.)*

That gives the tab exactly four states, and the person moves between them:

| State | What the tab shows |
|---|---|
| the env is not installed | the install command — the env is **opt-in**, so this is an ordinary state and not an error |
| installed, nothing running | one sentence saying what starting gets you, and the Start button — **unless** this molbuilder has no supervisor to hold a notebook, or the viewer may not control one, in which case the sentence says which and there is no button |
| starting | the same row, polling, until the port answers |
| running | the framed JupyterLab, and a Stop button. A viewer who may not control it gets neither: the token that reaches a kernel is withheld from them (§ 6), so there is nothing to frame |

Once it is running, Jupyter's own settings shrink the idle window rather than
molbuilder hand-rolling one: `MappingKernelManager.cull_idle_timeout` reaps
idle kernels, and `ServerApp.shutdown_no_activity_timeout` stops the server
when nobody is using it.

### 4.1 What the framed Lab looks like, and what it remembers

A Lab in a frame is not a Lab in a window, and four of its defaults are wrong
here. molbuilder sets them in `_LAB_OVERRIDES` (`jupyter.py`) — **defaults, not
values**: every one is still a control the person can change inside Lab.

| Setting | Why |
|---|---|
| multi-document mode (`startMode: multiple`) | Lab's document **tab bar**, so several notebooks are open at once. Single-document mode was tried first, to be rid of Lab's file browser — it takes the tab bar with it, and reading two notebooks side by side is worth more than losing the panel is *(decided with the user 2026-09-14)* |
| `kernelShutdown: true` | Closing a notebook shuts its kernel down. Lab keeps it running by default so you can reopen with your variables; inside a tab of another application a kernel nobody can see is memory — and on a GPU box a device — held for no one. A page RELOAD is not a close, so reopening reconnects |
| dark theme + dark scrollbars | Lab renders light by default, inside an application that is dark everywhere else. The frame read as a different program pasted into the page |
| `fetchNews: false`, `checkForUpdates: false` | Jupyter asks each viewer whether it may fetch its news feed, in a popup over the frame. A tab inside molbuilder is not where that is answered, and the answer is a network call from a machine that may have no route out |
| **its own settings home** (`config_dir.jupyter_lab_home`) | `app_settings_dir` · `user_settings_dir` · `workspaces_dir`, all separate from `~/.jupyter`. Lab writes a user setting the first time it resolves one and a user setting BEATS an override, so a shared home let the framed Lab adopt whatever the person's standalone Lab had written — and let molbuilder's choices leak back into it |

**And it remembers no LAYOUT.** The frame URL carries Jupyter's `?reset`,
which resets the *workspace* — which documents were open, which side panel was
showing — and nothing else. That restore argued with the one thing this tab
decides: it disagreed with the folder the projects sidebar had selected.
**The notebook file is the state worth keeping, and it is on disk.**
*(Decided with the user 2026-09-14.)*

The four settings above are **user settings**, not workspace, so `?reset` does
not touch them: they live in `user-settings/` and persist across loads by
design — which is the point of the separate settings home, and why a person's
own change inside Lab sticks. *(This section claimed `?reset` restored all four
until the claim was checked.)*

### 4.2 No `.ipynb_checkpoints` in the projects tree

Jupyter writes a `.ipynb_checkpoints/` directory **beside every notebook it
saves**. In a projects tree that is a directory in every folder somebody has
opened a notebook in — swept up by result scans, carried along by every copy to
a cluster, and holding a stale duplicate of work nobody asked it to keep.

Jupyter has no switch for it, and the obvious workaround is worse than the
problem: `FileCheckpoints.checkpoint_dir` only *renames* the directory, and
pointing it at one shared absolute path makes two `Untitled.ipynb` in different
folders write the same `Untitled-checkpoint.ipynb`, so a restore hands back the
wrong file. What the contents manager *does* take is a `checkpoints_class`, and
jupyter-server ships no no-op one — so molbuilder writes one, into a generated
`jupyter_server_config.py` passed as `ServerApp.config_file`. A Jupyter config
file is executed Python, which is why the class can live there rather than on
`PYTHONPATH`. Being an absolute path, it is loaded *instead of* searching the
config path, so the framed server does not read a personal
`~/.jupyter/jupyter_server_config.py` either — the same isolation § 4.1 gives
the settings home.

Restoring **refuses** rather than quietly doing nothing: with the checkpoint
list empty Lab offers nothing to restore, and a path that could still be
reached must never silently discard an edit. *(Asked for by the user
2026-09-14; verified the same day — a checkpoint POST answered
`{"id": "no-checkpoint"}`, the list came back empty, and no directory
appeared.)*

### 4.3 Where a notebook is saved, said out loud

**Lab's own file browser decides.** Its current folder is what the Launcher
creates in, and molbuilder cannot see or set it once the frame is live —
re-pointing the frame means reloading it, which discards every open document.
The projects sidebar still chooses where Lab *opens* (the frame URL is
`/lab/tree/<selected folder>`), and after that the person is driving.

So the tab states the fact it can know rather than the one it would like to.
`jupyter.open_notebooks` asks Jupyter's own `GET /api/sessions` and the control
row lists **the full path of every notebook Lab has open**. The row used to
claim "new notebooks are saved in `<the folder molbuilder selected>`", which is
true for exactly as long as it takes to click a folder inside Lab — and with
several notebooks open it is usually wrong. Naming the control that decides
beats impersonating it. *(Asked for by the user 2026-09-14: "I want to make
where those notebooks are saved clear and explicit".)*

**Closing a notebook shuts its kernel down** — `kernelShutdown` in § 4.1.
*(This paragraph said no such setting existed, until searching every shipped
schema on 2026-09-14 found it.)*

A page RELOAD is not a close: the workspace reset closes documents without Lab
treating it as one, so reopening the same notebook reconnects to its kernel
with its variables. What catches the rest is the timeout culling above, and its
generosity is deliberate: the tab is an **iframe**, so switching to Results
closes the kernel's socket. Culling promptly on a closed connection would mean
a five-minute look at another tab costs you every variable in memory. Nothing
leaks regardless — the whole tree dies with molbuilder (§ 3).

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

A live kernel is **arbitrary code execution** as the account serving the page,
reachable on a network port. That is a deliberate decision, not an implication
of having added a tab, so here is the rule that ships.

**Starting one, stopping one, and being handed the token that reaches one are
the same privilege**, and `_may_control` (`web/blueprints/jupyter.py`) grants
all three together:

1. an **admin** request may (`web/admin.py` — with no `admin` section
   configured, that is anyone who can sign in, per `access-control.md` § 5);
2. otherwise, if **sign-in is configured at all**, nobody else may;
3. otherwise — an unauthenticated molbuilder — a **loopback** peer may.

Rule 3 is the one worth reading twice: "no sign-in configured" does not imply
a loopback bind, so an unauthenticated molbuilder can legitimately be listening
on a network interface, and there the check is the peer address and nothing
else.

**The token follows the same rule.** `/api/jupyter/status` is readable by
anyone who can reach the page — it has to be, the tab draws itself from it —
but it returns `token: ""` to a caller who may not control the notebook, and
the tab then refuses to frame Lab rather than showing Jupyter's login page.
Until 2026-09-14 the token went to every caller while the Start button was
gated: the gate was inverted, since *using* a running kernel is the same code
execution as starting one.

Jupyter's own token auth stays on, and the server binds to loopback unless the
operator says otherwise.

## 7. The env

`molbuilder-jupyternb`, one recipe in the registry like every other backend
(`installation.md` § 1), and it holds **everything a notebook needs**: the
JupyterLab server, the `ipykernel` a cell runs on, and the analysis stack —
`numpy`, `scipy`, `pandas`, `matplotlib`. Jupyter is installed, activated and
runs here. **No other env carries notebook tooling.**

It deliberately holds no science backend — no `ase`, `sisl`, `rdkit` or
`pyscf`. Those would be a second copy to keep in step with the first, and the
day the two drift a notebook stops reproducing what a calculation does,
silently. This env *reads* what a calculation wrote and plots it; it is not a
second place to run one.

*(That is the user's design, stated 2026-09-14. The earlier text here — the
stack in the host env, each calculation env offering itself as a kernel — was
mine and was wrong: to be a kernel an env must carry `ipykernel`, which drags
`debugpy`, `ipython`, `jupyter_client`, `pyzmq`, `tornado` and six more into an
env whose only job is reproducible calculation. A job env stays a job env.)*

## 8. Status

**Built** *(2026-09-14)*, and § 3.4 — parented to the supervisor — is the
decision the user took against the two alternatives (dying with the server
child, or fully independent with its own pidfile).

Where it lives: `molbuilder/jupyter.py` (the lifecycle, the shepherd, and the
generated Lab settings), `serve_daemon` (the two signals, the stop-on-exit,
and the startup reconciliation — done with its own helpers, because it may
import nothing of the application), `web/blueprints/jupyter.py` (status ·
start · stop), and the **JupyterNB** tab at `/jupyternb`.

There is deliberately **no kernel search path**. One stood here until
2026-09-14, from the design where every env offered itself as a kernel; once
that was withdrawn it contributed zero kernels and one cross-env leak — the
host env's `share/jupyter` went on `JUPYTER_PATH` first, handing the framed Lab
a `jupyterlab-plotly` extension built against a `plotly` the notebook env does
not have. The kernel lives in the same prefix the server runs in, so Jupyter
finds it through `sys.prefix` with no search path at all.

Verified in a browser on 2026-09-14: Start → supervisor → shepherd → the
manager's own `run` → `jupyter lab`, framed, rooted at the projects tree, and
Stop taking the whole process group with it.

**Known gap.** An env built before the analysis stack was added to the recipe
has a kernel that cannot `import numpy`. The verb is
`repair molbuilder-jupyternb`, **not `install`**: on an env that already
exists `install` skips the create step and goes straight to verify, so it adds
no missing conda package. `repair` is what closes what the package audit
reports. *(Measured 2026-09-14, after `install` reported FAILED with
`No module named 'numpy'` and had installed nothing.)*
