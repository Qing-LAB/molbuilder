# Reloading a running server — a plan

**Role:** plan (**complete — A–E all landed 2026-08-03**; archived 2026-08-19
per the plans-folder rule — a finished plan leaves `plans/`; kept as the record of
what was decided and what was rejected. The shipped behaviour is documented in
[`deployment.md`](?doc=ops/deployment.md) § 1 and § 4.)
**Domain:** ops
**Started:** 2026-08-03
**Companions:** [`access-control.md`](?doc=ops/access-control.md) — the gate this
route sits behind, and why it answers 404 rather than 403;
[`deployment.md`](?doc=ops/deployment.md) — how the server is started and
exposed.

---

## 1. The problem, split in two

Changing the code and seeing the change costs a manual restart today. But the two
halves of "the code" are not the same problem and must not get one answer:

| What changed | Why the change is not visible | What it actually needs |
|---|---|---|
| **Python** | modules were imported once, at startup | a **fresh interpreter** — nothing else will do it |
| **JS / CSS / templates** | the browser is serving its cached copy, unasked | the browser to **ask** before reusing it (§ 2) |

Templates need neither: Jinja re-reads from disk on request. Static files are read
per request too — so **only the browser's cache makes them look stale**, and only
Python genuinely needs the process replaced.

## 2. Half one — static assets revalidate ✅ done 2026-08-03

**Not a version on the URL. That was this plan's first answer and it was wrong.**

Flask's default caches a static file for **12 hours with no check**, so a changed
`.js` or `.css` keeps loading from the browser's copy — while the server has been
serving the new bytes the whole time. That is the entire reason a front-end change
has looked like it "needs a restart".

`SEND_FILE_MAX_AGE_DEFAULT = 0` makes it `Cache-Control: no-cache`, which does not
mean *do not store* — the browser keeps its copy and **asks** whether it is still
good. Unchanged files come back **304 with no body**, so the saving that mattered
is kept and the staleness is gone.

**Why the version guard was the wrong answer.** It only reaches URLs the *server*
builds. 119 of this app's asset references come through `url_for('static')` — but
**51 more are ESM imports written inside the JavaScript** (`export { mount } from
"./mount.js"`), which no template sees and no `url_for` can rewrite. Versioning
the entry point would have left the whole module graph behind it on cached copies:
the half that actually breaks. And the identity it would have used —
`/api/health`'s `__version__` — is the *package* version, which does not move when
a file is edited, so it could not have signalled a change at all.

Revalidation is a property of how a file is **served**, so it covers all 170
references with no build step and no bookkeeping.

**It does not weaken the rate limiter, and that was checked rather than assumed.**
It multiplies requests per page load, but `rate_limit.py:310` reads
`if not (400 <= status_code < 500): return` — **only 4xx feeds the counter**. A 200
or a 304 is discarded before reaching any buffer. Pinned by
`tests/test_static_revalidates.py`, including a guard on that predicate: if the
limiter is ever widened to count 3xx, the assumption behind this change breaks and
the test says so.

**A `/static` exemption was considered and rejected.** The idea was to make
`threshold_total` usable again — but that counter counts *all* requests, not 4xx,
so exempting static from the 4xx path would not have helped it. Worse, a scanner
probing `/static/.env`, `/static/config.php` generates exactly the 4xx storm the
limiter exists to catch, on that very path — the exemption would have created a
namespace where enumeration is free. Legitimate static traffic already passes
freely because it returns 200/304; **there is no legitimate static 4xx to
protect.**

## 3. Half two — an *enforced* reload, not a watched one

### 3.1 Why not a file watcher

Werkzeug's reloader watches every file in `sys.modules` — stat-polled once a
second in this environment (`watchdog` is not installed, so `auto` resolves to
`StatReloaderLoop`).

**It fires on any mtime change, at a moment nobody chose.** An editor writing in
chunks, a `git checkout` touching fifty files, a partially-flushed save — each is
a reload against a source tree that is momentarily inconsistent, and the child
comes back up importing a half-written module. The failure looks like a code bug
and is not one.

**So the trigger should be a person saying "it is ready now."** That removes the
race completely rather than narrowing it.

### 3.2 What is worth keeping from the reloader: the parent

The valuable half of Werkzeug's design is not the watcher, it is the **process
shape**:

```python
while True:
    exit_code = subprocess.call(args, env={**os.environ, "WERKZEUG_RUN_MAIN": "true"})
    if exit_code != 3:
        return exit_code
```

A **parent that never imports application code**, whose only job is to respawn a
child that exits with a sentinel. Any other exit code — Ctrl-C, a crash — ends it
properly instead of respawning forever.

That is what makes it robust, and it is exactly what a restart endpoint inside the
server cannot give itself: a process re-execing itself has no one to catch it if
the new code fails to import. **The parent has no opinions and no imports, so it
cannot be broken by the code it restarts.**

### 3.3 The shape

```
molbuilder serve                    the parent: spawn, wait, respawn on <sentinel>
        └── child                   the app, as today
                POST /api/admin/reload   →  answer 202, then exit(<sentinel>)
```

1. The button POSTs to an **admin-gated** route.
2. The route replies **before** exiting, so the browser knows it was accepted, then
   schedules `os._exit(<sentinel>)` on a short timer so the response is flushed.
3. The parent sees the sentinel and spawns a fresh child — new interpreter, every
   module imported again.
4. The browser polls **`/api/health`** (it already exists) until it answers, then
   reloads the page — which picks up new JS through § 2's revalidation.

**Without a supervisor the route does not exist.** A server with no one to
restart it would, on pressing the button, leave a dead site with no way back from
the browser. Supervision is what makes the promise true.

**It became the default on 2026-08-04.** Opt-in was the wrong shape: the reason
to supervise — that a restart is possible at all — applies to every ordinary
run, and as a flag it meant the button was missing for anyone who had not read
the help text. `--no-supervise` remains for the case where something else
already owns restarts (systemd, Docker, gunicorn), where a supervisor inside is
a second answer to a settled question. `--debug` turns it off on its own:
Werkzeug's reloader respawns its child on *any* exit, so it would swallow the
sentinel and the button would silently do nothing.

The same change closed a defect that made the whole design a claim rather than a
fact. `from .web.app import create_app` sat above the fork in `cmd_serve`, so the
parent imported the entire app before spawning anything — and a parent that
imports the broken module dies with it, which is precisely what the supervisor
exists to prevent. The import now happens in the child, and
`test_the_parent_forks_before_importing_the_application` runs the parent branch
in a clean interpreter to keep it that way.

## 4. Who may press it ✅ decided 2026-08-03

**This was the part to decide before any of it was built.** All three conditions
below were adopted as written, and the first is pinned by
`tests/test_admin_reload.py`.

At the time, `rate_limit.py` shipped `admin_emails: []` and its own comment said
what that meant: *"Empty list = ANY logged-in session is admin."* The server
binds `0.0.0.0` behind OAuth.

**So with that default, a restart button next to the user's email would hand
every person who can log in the ability to disconnect everyone else** — mid
calculation, mid edit. Workspace writes are fire-and-forget, so in-flight ones
are lost. (That default is gone: see the note below.)

Three conditions, and the first is not optional:

1. **A real admin gate.** Somebody must be named for the route to exist at all —
   an empty list disables the endpoint rather than admitting everybody. The
   failure mode of a mistake is then "the button is missing", not "anyone can
   restart the server". (Originally an inversion *for this route only*; since
   2026-08-03 it is simply what the one admin list means everywhere.)
2. **The button is hidden when the route is absent**, so a non-admin never sees a
   control they cannot use.
3. **A confirm that names the cost** — *"this disconnects everyone using this
   server and drops saves that are still in flight."*

**What building it exposed — and it was fixed 2026-08-03, not left.** One key
gated two unrelated subsystems: `rate_limit.admin_emails` decided both who may
read the rate-limit table and who may stop the process, and the two read the
*same empty default in opposite directions* — "everybody" there, "nobody" here.
`create_app` also reached the list through `app.extensions["rate_limiter"]`, so
turning the limiter off quietly moved the admin list.

The admin identity has its own section now — top-level `admin: {emails: […]}`,
read through `web/admin.py` — and **absent or empty means anyone who signed in, to every
subsystem that asks**. So this route no longer inverts anything; it reads the
same list the same way. See [`access-control.md`](?doc=ops/access-control.md)
§ 5.

## 5. Order — all landed 2026-08-03

| | Step | Note |
|---|---|---|
| **A** ✅ | ~~The version guard~~ **static revalidation (§ 2)** | independent of everything else, no new surface, useful on its own |
| **B** ✅ | the parent loop (`--supervise`, on by default since 2026-08-04) | `cli.py::_supervise_forever`; the sentinel lives in `molbuilder/reload_protocol.py`, a **leaf module that imports nothing** |
| **C** ✅ | The admin gate: route exists only when somebody is named an admin | `app.py`, before the route does anything |
| **D** ✅ | `POST /api/admin/reload` — reply 202, then exit with the sentinel | `os._exit` on a short timer, so the response is already on the wire |
| **E** ✅ | The button + the poll-and-reload, shown only when the route exists | `_app_header.html` + `static/lib/app-reload.js`; availability read from `/api/admin/reload/available` |

**Where the protocol constants live is load-bearing, and the first placement was
wrong.** They started at `web/reload_protocol.py`, where importing them ran
`web/__init__.py` → `app.py` → Flask — which destroys the one property the design
rests on: *the parent never imports application code*, so a child that fails to
import leaves the supervisor alive to fix it. Moved to `molbuilder/reload_protocol.py`
and pinned by `test_the_supervisor_does_not_import_the_app_it_restarts`, which runs
a subprocess and checks `sys.modules`.

**Tests:** `tests/test_admin_reload.py` (12) — both halves of the gate as **404,
not 403**; the availability answer in four configurations; the respawn loop
(sentinel only, and the child is told it is the child, or `--supervise` forks
forever); one sentinel value read by both sides; and the import-isolation probe.
Step A is pinned separately by `tests/test_static_revalidates.py`.

## 6. What this does not do

- **It does not make the dev server a production server.** `app.run()` is still
  Werkzeug's dev server; supervision does not change that, and
  [`deployment.md`](?doc=ops/deployment.md) still governs how it is exposed.
- **It does not reload without a restart.** There is no module-swapping and no
  partial reload — a fresh child is the whole mechanism, which is why it cannot
  leave half the app on old code.
- **It does not preserve in-flight work.** A restart drops it, which is why § 4.3
  says so out loud rather than hiding it behind a spinner.
