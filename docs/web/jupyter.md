# Jupyter — the JupyterLab tab, and the lifecycle that keeps it honest

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
| **One port.** The browser reaches JupyterLab only through molbuilder's own address, under `/jupyter/`: molbuilder passes each request through and joins the kernel's WebSocket through the same connection. Jupyter listens on `127.0.0.1` only | § 2 |
| **Stopping means stopping the SERVER.** The tree is supervisor → shepherd → the manager's `run` → jupyter-server, so a stop takes the shepherd's whole process group. The kernels are not in that group and do not need to be: Jupyter collects its own | § 3.2 |
| **`kill -9` runs no handler.** `PR_SET_PDEATHSIG` is the only mechanism that survives it; the pidfile reconciliation is what catches the rest | § 3.1, § 3.3 |
| **Parented to the SUPERVISOR, and stopped with the server.** A restart of the server — `serve restart`, the page's Reload — stops a running notebook; stopping molbuilder stops it too | § 3.4 |
| **Jupyter's own port is `serve + 1`, on loopback.** Two molbuilders on adjacent ports collide by construction, and molbuilder says so before it starts rather than after | § 2.1 |
| **Nothing runs until asked.** No kernel on page load; idle kernels are culled by Jupyter's own timeouts | § 4 |
| **A live kernel is arbitrary code execution.** Starting one, stopping one and reaching one through `/jupyter/` are one privilege. That is a decision, not a side effect of adding a tab | § 6 |

```
   browser ──HTTP(S)─▶ molbuilder (:8888)   the tab, its control API, and
                           │                /jupyter/… passed through
                           │
                           ├──HTTP + WebSocket──▶ jupyter server (127.0.0.1:8889)
                           │                           │
                           │                           └── ipykernel, ipykernel, …
                           │                               each in its OWN session --
                           │                               Jupyter collects them
                           │
                           └── shepherd ── <mgr> run ── jupyter server
                               └───────────────────────────┘
                                 one process group: a stop takes all three
```

---

## 1. What this document owns

The **shape** of the integration: which process talks to which, who starts and
stops what, and what must be true when molbuilder exits. It does not own
Jupyter's own configuration surface, nor the recipe's package list — that is
`recipes.py` and `installation.md`.

## 2. Through molbuilder's port

*(The user, 2026-10-17: "the jupyternb never need an extra port from the
outside and the embedded frame talk through the same webui of molbuilder".)*

**The browser talks to molbuilder and to nothing else.** Four parts, each
with one job:

| part | owns |
|---|---|
| **molbuilder's server** — the front door | the one public port and its TLS; the pages; `/jupyter/…` passed to the notebook |
| **the gate** — `_refuse` (`web/blueprints/jupyter.py`) | who may reach the notebook: § 6's rule, asked of every `/jupyter/` request before anything is passed on |
| **the notebook** — JupyterLab, in its own env | notebooks and kernels; it listens on `127.0.0.1` only, under `/jupyter/` (`ServerApp.base_url`), its token auth on |
| **the supervisor** | starting and stopping the notebook (§ 3) |

**The pass-through** is two routes on one path, because the server routes a
WebSocket handshake only to a rule declared for one (Werkzeug's
`websocket=True`):

* **an ordinary request** is forwarded to Jupyter — its method, path, query,
  body and headers — and the answer streamed back with its status and
  headers;
* **the kernel's WebSocket** is joined: the server hands the route the
  request's own connection (Werkzeug's `werkzeug.socket`), the route opens one
  to Jupyter, sends it the same handshake, and copies bytes both ways until
  either end closes.

On the way in, the pass-through **speaks for itself**: molbuilder's own
session cookie and any `Authorization` or forwarding header the browser sent
are dropped, and in their place it presents the notebook's token and says the
scheme the browser used (`X-Forwarded-Proto`, which Jupyter trusts: nothing
but the pass-through reaches it). **molbuilder never hands the browser the
token** — not in its pages, not in the frame's address, not in its log;
JupyterLab's own page carries it for Lab's scripts, and only a viewer past
the gate is served that page. On the way out, Jupyter's `Access-Control-*`
headers are dropped: nothing outside the page is answered.

**The frame is the page's own origin**: `<base>/lab/tree/<folder>`, `<base>`
being `/jupyter`. Every answer passed through carries `X-Frame-Options:
SAMEORIGIN`, in place of the `DENY` molbuilder's own pages send, and
`frame-ancestors 'self'` where Jupyter sent no policy of its own — so the page
may frame it and nothing else may. The JupyterLab page's security policy is
every page's.

### 2.1 Jupyter's own port, and what collides with it

`jupyter_port(p) = p + 1` — **derived, not configured**, so the pair moves
together when a second molbuilder runs on another port. It is a port on
`127.0.0.1`: molbuilder's pass-through and `answering()` reach it there, and
nothing outside the machine does.

**The cost of deriving it is that adjacent ports collide.** A molbuilder on
6006 wants 6007 for its notebook; a molbuilder on 6007 wants 6007 for its
*web server*. Whichever starts second loses, and which one that is decides
the symptom — the notebook refuses to start, or `serve start` cannot bind.

Three things make that honest rather than mysterious:

1. **`ServerApp.port_retries = 0`** (`data/jupyter.toml`). jupyter-server
   otherwise tries fifty more ports and picks a random free one, while the
   runtime file, the pass-through and `answering()` all keep naming `p + 1` —
   so the tab would report *wedged* over a perfectly healthy notebook. Measured 2026-09-15 against a live notebook on 6007:
   *"ERROR: the Jupyter server could not be started because port 6007 is not
   available"*, exit 1.
2. **It is said in advance where it is knowable.** `jupyter.port_clash()`
   asks two questions in order — is that port another molbuilder's *serve*
   port (attributable from the pidfiles, and it names the other server), and
   failing that, is anything listening there at all. `serve start` prints it
   before detaching, the `serve status` survey prints it per server, and the
   tab's status payload carries it so a give-up message can name the cause
   instead of listing things to go and check.
3. **It is a warning, not a refusal.** The web server on that port is
   perfectly startable; only its notebook is doomed. Refusing the verb would
   be deciding for somebody who may not want a notebook at all.

**The derivation does not change.** `p + 1000` moves the collision without
removing it, and letting Jupyter choose freely makes the runtime file
authoritative for a port the pass-through derives — which is the design
`port_retries = 0` was chosen against. *(User question 2026-09-15;
[`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) § 5n, J15.)*

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
dying is what collects the kernels. *(A reaper of molbuilder's own would be
a third copy of something Jupyter does and passes.)*

This composes with § 3.1: `PR_SET_PDEATHSIG` keys on *parent death*, not on
group membership, so a new session does not disable it.

### 3.3 Reconciliation at startup

Layers 1 and 2 cannot cover a machine crash, or a survivor that was re-parented
before the signal landed. On startup molbuilder reads the pidfile and, **only
if the pid is alive and its `/proc/<pid>/cmdline` shows a Jupyter we started**,
stops it. Same rule `serve_daemon` states: a stale file whose pid was recycled
is reported stale, never signalled.

**The shepherd's polite phase ends before reconciliation stops waiting.** A
stop asks the shepherd's group to go and waits a grace before it takes it
(`jupyter._STOP_GRACE_S`, 4 s); reconciliation's stop of the shepherd waits
its own (`serve_daemon.stop_by_pidfile`, 5 s) and reports *"(forced)"* when it
had to take it. Equal, the two would expire together and every polite stop of
a server slow to go would read *forced*; so the shepherd's stays strictly
under (`test_jupyter_contract.py` keeps them apart).

### 3.4 Parented to the supervisor, stopped with the server

Jupyter is parented to molbuilder's **supervisor**, the process that holds
the notebook's start command and answers the Start and Stop buttons — the
server child runs the page and cannot hold a process past its own restart.

**A restart of the server stops the notebook** *(the user, 2026-10-17:
"restart server should restart notebook with it or shutdown notebook
associated with it")*: a restart is how new code takes effect, and a
notebook left running would be one the server no longer started. The tab
then offers Start, on the code the server now runs.

| what happens to the server | Jupyter |
|---|---|
| `serve restart`, or the page's Reload | **stopped** — its kernels end with it |
| its child crashed and was respawned (the hung-child repair) | **runs on** — nothing was restarted on purpose |
| `serve stop` / supervisor exit | **stopped** |
| `kill -9` of the supervisor | **dies** (§ 3.1) |

## 4. Nothing runs until asked — **not even by looking**

Opening the tab **probes and reports**; it starts nothing. Starting a notebook
runs code as the account serving the page, so it is a thing a person asks for,
with a button. *(Decided with the user 2026-09-13: a tab that runs code
because you clicked its name is a tab you cannot look at safely.)*

### 4.0 The states — the table the tab is built from

The code is a literal table (`STATES` in `jupyternb/index.js`) and this is
the same list in the same order. **First match wins**, so the order is the
contract.

| # | State | When | Shows | Waits |
|---|---|---|---|---|
| 1 | **env-missing** | the env is not installed | the install command — the env is **opt-in**, so this is ordinary and not an error | 60 s |
| 2 | **not-yours** | running and answering, and this viewer may not control it | that using a kernel is an admin action (§ 6). Nothing is framed: every `/jupyter/` request would be refused | 30 s |
| 3 | **opening** | answering, **nothing to restore**, and the projects sidebar has not resolved its root | *"Opening JupyterLab…"* | the sidebar's answer (`projects.md` § 5) → 4; when the sidebar cannot resolve the root, 4 at the projects root |
| 4 | **framed** | running and answering | the framed JupyterLab, what it has open, and a Stop button *if this viewer may control it* | 30 s heartbeat |
| 5 | **starting** | the process is up but not serving | *"…up but not answering yet"* | 1.5 s, **30 s** → wedged, with **Check again** and Stop |
| 6 | **asked** | Start was clicked and nothing is up yet | *"Starting…"* | 1.2 s, **15 s** → the port clash if there is one (§ 2.1), else *"none started"* naming **both** logs, with **Check again** |
| 7 | **unsupervised** | no supervisor to hold a notebook | that `serve start` is what gives it one. **Not** `jupyter start` — that verb signals the supervisor, so it is the one command guaranteed to fail here | 30 s |
| 8 | **no-control** | nothing running, and this viewer may not start one | that starting runs code on this machine | 30 s |
| 9 | **idle** | installed, nothing running, and the person may ask | what starting gets you, and the Start button | 15 s heartbeat |
| 10 | **unreachable** | the status endpoint could not be asked | the error, and a **Try again** button | that button |

**EVERY STATE EITHER POLLS OR OFFERS A BUTTON**: a state that does neither
leaves a notebook started from a *terminal* invisible to an open tab, and
turns a budget that runs out into a dead end.
`env-missing` polls slowest on purpose: it is the one poll that costs a
subprocess, because a negative answer re-asks the env manager rather than
trusting a snapshot bound at process start.

**WAITING HAS ONE RULE.** A state declares how long it may last; entering it
starts the clock, and entering is the only thing that resets it — so a
notebook that goes away later never makes the tab report a failed start that
worked. Budgets are in **milliseconds, not polls**, so each state keeps the
wake-up that suits it.

State 3 is the one to understand: it exists because Lab's Launcher creates a
notebook **in the folder the frame is showing**, so framing before the root is
known put every new notebook in `projects/` however carefully a folder had
been picked. It waits on the sidebar's answer rather than polling — each
poll costs the server two blocking round-trips to Jupyter — and needs no
budget, because that answer always comes: the root, or why the sidebar's
start could not resolve it ([`projects.md`](?doc=web/projects.md) § 5), and
then the tab opens at the projects root. When there is no such door on the
page at all, state 3 simply does not match and state 4 takes it.

Once it is running, Jupyter's own settings shrink the idle window rather than
molbuilder hand-rolling one: `MappingKernelManager.cull_idle_timeout` reaps
idle kernels, and `ServerApp.shutdown_no_activity_timeout` stops the server
when nobody is using it.

### 4.1 What the framed Lab looks like, and what it remembers

A Lab in a frame is not a Lab in a window, and four of its defaults are wrong
here. molbuilder sets them in **`molbuilder/data/jupyter.toml`'s
`[lab.overrides]`** — **defaults, not values**: every one is still a control
the person can change inside Lab.

| Setting | Why |
|---|---|
| multi-document mode (`startMode: multiple`) | Lab's document **tab bar**, so several notebooks are open at once. Single-document mode was tried first, to be rid of Lab's file browser — it takes the tab bar with it, and reading two notebooks side by side is worth more than losing the panel is *(decided with the user 2026-09-14)* |
| `kernelShutdown: true` | Closing a notebook shuts its kernel down. Lab keeps it running by default so you can reopen with your variables; inside a tab of another application a kernel nobody can see is memory — and on a GPU box a device — held for no one. A page RELOAD is not a close, so reopening reconnects |
| dark theme + dark scrollbars | Lab renders light by default, inside an application that is dark everywhere else. The frame read as a different program pasted into the page |
| `fetchNews: "false"` (a quoted string enum — Jupyter's own schema), `checkForUpdates: false` (a real boolean) | Jupyter asks each viewer whether it may fetch its news feed, in a popup over the frame. A tab inside molbuilder is not where that is answered, and the answer is a network call from a machine that may have no route out |
| **its own settings home** (`config_dir.jupyter_lab_home`) | `app_settings_dir` · `user_settings_dir` · `workspaces_dir`, all separate from `~/.jupyter`. Lab writes a user setting the first time it resolves one and a user setting BEATS an override, so a shared home let the framed Lab adopt whatever the person's standalone Lab had written — and let molbuilder's choices leak back into it |

**It remembers your LAYOUT while the server runs, and forgets it when the
server restarts.** *(The user, 2026-09-15: "we would like to have this
persistent when switching tab, but do not need this when server get shutdown
and restarted.")*

Lab saves a **workspace** — which documents were open, which side panel was
showing — and molbuilder empties that directory **once, when the notebook
server starts** (`jupyter.prepare_lab_home`). So:

| | |
|---|---|
| switch to another molbuilder tab and come back | your notebook is still open, and its kernel never stopped |
| reload `/jupyternb` | the same |
| `jupyter restart`, `serve stop` + `start`, a new notebook server | clean |

**The reset happens when the server starts, never when the page loads.**
Leaving `/jupyternb` destroys the iframe, so a tab switch is a load, and a
reset there would throw away the notebook you had open **while its kernel is
still running**. *(Reported by the user 2026-09-15.)*

**The workspace is PER SERVER.** `config_dir.jupyter_lab_home()` is not
port-keyed, because the defaults do not differ between servers; session state
does, so the layout lives in `workspaces/<serve port>/`, and `settings/` and
`user-settings/` stay shared. A shared workspace would let starting B's
notebook empty A's layout, so A's next tab switch would lose the notebook
whose kernel is still running. B's saved layout would also make A open at the
projects root instead of the selected folder.

**The folder is carried exactly once.** `status` reports `workspace_saved`,
and the tab uses it to decide the URL: with no workspace this is the first
framing since the server started, so the URL carries the selected folder
(`<base>/lab/tree/<folder>`) and Lab opens there; with one, the URL carries no tree
path at all and nothing competes with the restore. Asking the **server** —
rather than remembering in the browser — is what keeps the two halves from
disagreeing: the wipe and the flag read the same directory, and nothing has
to survive a page load. It also means a tree path and a restore are never
sent together, so Lab's own precedence between them, which molbuilder has
**not** measured, decides nothing here.

The four settings above are **not workspace**, so the wipe never touches
them — but note *where* they are. molbuilder writes them to the **app**
settings directory, `settings/overrides.json`, rewritten at every start; a
person's own change to one of them is what Lab saves into `user-settings/`,
and a user setting beats an override, so it sticks across every restart.
Debugging *"why does my framed Lab look like this"* starts at
`settings/overrides.json`.

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
jupyter-server ships no no-op one — so molbuilder ships one, as
**`molbuilder/data/jupyter_server_config.py`**, copied into the Lab home at
every start and passed as `ServerApp.config_file`. A Jupyter config file is
executed Python, which is why the class can live there rather than on
`PYTHONPATH`. **A file under `data/`, not a module and not a string literal:**
it must subclass jupyter-server's `AsyncCheckpoints` at class-definition time,
and the shepherd runs in the HOST env, which does not have jupyter-server and
must not need it — so it cannot be imported, and `inspect.getsource`, which
reads an imported class, cannot reach it. A string literal is invisible to
every tool: an edit can delete it with nothing to catch it. Being an absolute path, it is loaded *instead of* searching the
config path, so the framed server does not read a personal
`~/.jupyter/jupyter_server_config.py` either — the same isolation § 4.1 gives
the settings home.

Restoring **refuses** rather than quietly doing nothing: with the checkpoint
list empty Lab offers nothing to restore, and a path that could still be
reached must never silently discard an edit. *(Asked for by the user
2026-09-14; verified the same day — a checkpoint POST answered
`{"id": "no-checkpoint"}`, the list came back empty, and no directory
appeared — and asked of a real notebook server, started as the supervisor
starts it, by `test_jupyter_contract.py`.)*

### 4.3 Where a notebook is saved, said out loud

**Lab's own file browser decides.** Its current folder is what the Launcher
creates in, and molbuilder cannot see or set it once the frame is live —
re-pointing the frame means reloading it, which discards every open document.
The projects sidebar still chooses where Lab *opens* (the frame URL is
`<base>/lab/tree/<selected folder>`), and after that the person is driving.

So the tab states the fact it can know rather than the one it would like to.
`jupyter.open_notebooks` asks Jupyter's own `GET /api/sessions` and the control
row lists **the full path of every notebook Lab has open**. For new
notebooks it says only that they are saved in the folder Lab's own file
browser is showing, and names no folder: a named one would hold only until a
folder is clicked inside Lab, and with several notebooks open it is usually
wrong.
Naming the control that decides beats impersonating it. *(Asked for by the user 2026-09-14: "I want to make
where those notebooks are saved clear and explicit".)*

**Closing a notebook shuts its kernel down** — `kernelShutdown` in § 4.1.

A page RELOAD is not a close: Lab restores the workspace and reconnects each
document to its kernel, variables and all. What catches the rest is the timeout culling above, and its
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

**`start` and `restart` read `molbuilder.json` before they signal**, and so
does the tab's Start button (§ 5.2), because the shepherd reads it to find the
notebook env: a broken file is refused in its own words (`Error: <path>: …`,
exit 2, on the command line; a 409 on the button), and `restart` never stops a
working notebook for a start that cannot happen — the rule `serve restart`
keeps for the server ([`ops/deployment.md`](?doc=ops/deployment.md) § 1.0b).

### 5.1 `GET /api/jupyter/status` — the payload, stated once

**Sixteen keys, authored in two modules and read by a third**
([`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) § 5n, J5). `jupyter.status()` produces eight, the blueprint adds
seven in one expression, `ok` is the envelope. `jupyternb/index.js` reads
**fourteen** of them — `pid_state` and `url` are produced for the CLI's
`jupyter status`, and the tab deliberately ignores `url` (see its row). This table is the
contract; the assembly at the end of `api_jupyter_status` is the one place
the shape is built.

| key | from | meaning |
|---|---|---|
| `ok` | blueprint | the envelope. `false` never happens here. The tab shows *unreachable* for a refusal or an unreadable answer (`ask`, `lib/server.js`) |
| `running` | `status()` | a pidfile naming a live shepherd of ours |
| `pid` · `pid_state` | `status()` | which process, and `ours` / `foreign` / `dead` |
| `port` | `status()` | Jupyter's own port on `127.0.0.1` — `serve + 1` (§ 2.1) |
| `answering` | `status()` | the **second** question: it replied over HTTP |
| `url` | `status()` | where it listens on this machine, for `molbuilder jupyter status`. The tab frames `/jupyter/` (§ 2) and does **not** use it |
| `open` | `status()` | the paths Jupyter says are open. **Withheld unless `may_control`** |
| `workspace_saved` | `status()` | has Lab saved a layout since this server started (§ 4.1) |
| `base` | blueprint | the path the notebook is passed through at — `/jupyter`, `jupyter.BASE_PATH` (§ 2); the tab frames `<base>/lab…` |
| `supervised` | blueprint | is there a supervisor that can be asked at all |
| `may_control` | blueprint | may **this request** start and stop it |
| `env_name` · `env_installed` | blueprint | the opt-in env, probed — the browser cannot ask conda |
| `install_command` | blueprint | what to run when it is not installed |
| `port_clash` | blueprint | why a start will fail, before it is tried (§ 2.1). **Withheld unless `may_control`** |

**Two are withheld, and by never being produced.** `include_private=False`
means `open` is not built at all for a caller who may not control the
notebook — somebody's project paths. `port_clash` names another server on
this machine. **The token is not in the payload at all**: it never leaves the
server (§ 2).

### 5.2 The two control routes

`POST /api/jupyter/start` and `POST /api/jupyter/stop` are **always
registered** and answer:

| | |
|---|---|
| **404** | no supervisor to ask — the button is absent, and a misconfiguration reads as *missing*, never as *anyone may start a kernel here* |
| **403** | there is one, and this caller may not run code on this machine |
| **202** | the signal was delivered |
| **409** | the supervisor refused it — or, on `start`, `molbuilder.json` cannot be read, in the file's own words (§ 5) |

One gate, one refusal, both in `_refuse()`. They are always registered, so
the app's URL map never depends on how the process was started, and a test
reaches them.

## 6. What this exposes

A live kernel is **arbitrary code execution** as the account serving the page.
That is a deliberate decision, not an implication of having added a tab, so
here is the rule that ships.

**Starting one, stopping one and reaching one through `/jupyter/` are the
same privilege**, and `_may_control` (`web/blueprints/jupyter.py`) grants all
three together:

1. an **admin** request may (`web/admin.py` — with no `admin` section
   configured, that is anyone who can sign in, per `access-control.md` § 5);
2. otherwise, if **sign-in is configured at all**, nobody else may;
3. otherwise — an unauthenticated molbuilder — a **loopback** peer that asked
   for the page by a **loopback name** (`localhost`, `127.0.0.1`, `::1`) may.

Rule 3 is the one worth reading twice. "No sign-in configured" does not imply
a loopback bind, so an unauthenticated molbuilder can legitimately be
listening on a network interface, and there the check is the peer. And the
peer alone is not enough: a web page from anywhere whose name its owner
points at `127.0.0.1` is fetched by the browser on this machine, so it
arrives from loopback too — the name it was asked by is that page's own, and
is what refuses it.

**The pass-through asks the same gate as the buttons** — `404` with no
supervisor to hold a notebook, `403` for a caller who may not control it — on
every request, before anything reaches Jupyter, and answers `502` when no
notebook is running. Behind it Jupyter's token auth stays on, presented by
the server (§ 2), and Jupyter listens on `127.0.0.1` only.

**And the gate refuses another page's request.** A request whose `Origin`
names any page but this one is refused, `403`. A request Jupyter sees as
token-authenticated skips its own cross-site checks — the `Origin` and XSRF
checks — and the pass-through presents the token on every request, so the
gate makes the check Jupyter would have made: without it, a page elsewhere
visited by someone who may control the notebook could drive the kernel in
their name.

**The token travels in the environment, not on the command line**: `/proc`
keeps `environ` owner-only and `cmdline` world-readable, so it is
`JUPYTER_TOKEN`, and no `--ServerApp.token` flag — an explicit trait would
beat the environment.
`test_jupyter_contract.py` asserts the token appears nowhere in the argv, over
the whole list rather than one spelling, so a row reintroducing it under
another trait name fails too.

## 7. The env

`molbuilder-jupyternb`, one recipe in the registry like every other backend
(`installation.md` § 1), and it holds **everything a notebook needs**: the
JupyterLab server, the `ipykernel` a cell runs on, and the analysis stack —
`numpy`, `scipy`, `pandas`, `matplotlib`. Jupyter is installed, activated and
runs here. **No other env carries notebook tooling.** **It starts as this
env's own Python running JupyterLab** — `python -m jupyterlab`, which the door
names as `<prefix>/bin/python` ([`ops/installation.md`](?doc=ops/installation.md)
M6), never a `jupyter` the activation's `PATH` finds: where this env lacks its
Python nothing starts, said in the notebook log, and where it lacks JupyterLab
its own Python says so.

**And the file-level science libraries: `ase` and `sisl`** *(user ruling,
2026-09-24)*. **A notebook has to read
the MD and trajectory formats calculations produce** — extended-XYZ, ASE's
own `.traj`, plain xyz, pdb, cif, and the engine outputs ASE's readers cover
— and `ase` is that reader; `sisl` is the same for what SIESTA and TBtrans
write. A frame set for transport is never made **with `ase`**
([`engines/transport.md`](?doc=engines/transport.md) § 2a.9): it is a data set
made through molbuilder's own Structure API — the motion toolkit
([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md)), which
imports with this env's packages and so makes frame sets here, as its example
notebook does, or a person's script, `with_frames` and `set_customized` —
written by the codec with its sidecar (`model/structure.md` § 2.2e). A file `ase` writes carries no
sidecar, and its comment line is never metadata
([`model/structure.md`](?doc=model/structure.md) § 2.3), so it reads as atoms
and coordinates only.

**And molbuilder itself — the one its server runs** *(the user, 2026-10-23:
"this has to be part of the module … that the user would be able to import
and use as a unified layer for post-processing")*. The notebook server is
started with molbuilder's own source folder on its Python path
(`PYTHONPATH`, set in the server's environment by `jupyter.run_shepherd`),
which a kernel it starts inherits — so a cell imports the molbuilder that
wrote the files it reads, `from molbuilder.postprocess import parse,
atom_index_map` ([`model/parse.md`](?doc=model/parse.md) § 3,
[`model/overview.md`](?doc=model/overview.md) § 2.2b): nothing installed into
this env, no second copy to drift. *(Measured 2026-10-23: outside the
checkout, `import molbuilder` fails in this env; with the source on the path,
`molbuilder.parse`, `runs`, `structure`, `atom_permutation` and both data
files' readers import.)* **The layer imports with this env's packages** —
numpy, netCDF4, sisl and xarray, the recipe's (xarray listed for its own reason — the user's yes, 2026-10-23) — and a reader that needs more is not part of
it.

**And molbuilder's example notebooks**, in the projects tree's
`notebooks/examples/` (their one rule,
[`execution/project-layout.md`](?doc=execution/project-layout.md) § 2.6): the
tab opens them there — Lab's file browser starts at the projects root, its
`root_dir` (§ 4.3) — each on this env's kernel: the motion toolkit's on a
vibration run molbuilder carries
([`engines/vibration-frames.md`](?doc=engines/vibration-frames.md) § 6.1),
transport's by hand on a model chain and on a transport calculation of the
person's ([`engines/transport.md`](?doc=engines/transport.md), *The script,
and the notebook*).

**It still holds no calculation engine** — no `pyscf`, `siesta` or `rdkit`.
Running one from a notebook would be a second place to run a calculation,
off the record. That half of the 2026-09-14 ruling stands. The cost that
ruling named — a second `ase` that can drift from the host env's — is real
and is **not checked** by the audit: both are bare conda-forge specs solved
on the same channel, and the formats it reads are stable, so the exposure
is bounded, not zero.

*(No calculation env offers itself as a kernel: to be one, an env must carry
`ipykernel`, which drags `debugpy`, `ipython`, `jupyter_client`, `pyzmq`,
`tornado` and six more into an env whose only job is reproducible
calculation. A job env stays a job env.)*

## 8. Status

**Built** *(2026-09-14)*, and § 3.4 — parented to the supervisor — is the
decision the user took against the two alternatives (dying with the server
child, or fully independent with its own pidfile).

Where it lives: `molbuilder/jupyter.py` (the lifecycle, the shepherd, and the
generated Lab settings), `serve_daemon` (the two signals, the stop-on-exit,
and the startup reconciliation — done with its own helpers, because it may
import nothing of the application), `web/blueprints/jupyter.py` (status ·
start · stop · the `/jupyter/` pass-through), and the **JupyterLab** tab at
`/jupyternb`.

There is deliberately **no kernel search path**. The kernel lives in the
same prefix the server runs in, and **the notebook is told to take it first**
(`JUPYTER_PREFER_ENV_PATH=1`, in `jupyter.notebook_environment`, beside the
token and molbuilder's source on `PYTHONPATH`): Jupyter takes an env's kernels
ahead of a person's own folders only where the account owns the env's prefix —
a lab-shared env, or a service account, does not — and a `--user` kernel
installed once, on another Python, then answered (measured 2026-11-21;
[`ops/installation.md`](?doc=ops/installation.md) M6). A
search path would be a cross-env leak: with the host env's `share/jupyter`
first on `JUPYTER_PATH`, the framed Lab would get a `jupyterlab-plotly`
extension built against a `plotly` the notebook env does not have.

Held on the road by `tests/test_notebook_through_one_port_e2e.py`: `serve
start` and `jupyter start` as a person types them, then a real browser on
molbuilder's one port — the tab frames JupyterLab from its own origin and a
cell run in it answers, and Jupyter's own port answers this machine only.

**An env built before the analysis stack was added to the recipe** has a
kernel that cannot `import numpy`. Either verb adds what such an env lacks:
`molbuilder envs install molbuilder-jupyternb` installs the recipe's conda
packages into an env that exists whenever it is short of one
([`ops/env-framework.md`](?doc=ops/env-framework.md) § 4.4), and
`molbuilder envs repair molbuilder-jupyternb` installs what the package audit
reports missing.
