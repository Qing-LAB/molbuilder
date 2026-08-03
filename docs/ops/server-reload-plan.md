# Reloading a running server — a plan

**Role:** plan (step A done; B–E need the § 4 decision)
**Domain:** ops
**Started:** 2026-08-03
**Companions:** [`deployment.md`](?doc=ops/deployment.md) — how the server is
started and exposed, and the auth/admin model this leans on.

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
molbuilder serve --supervise        the parent: spawn, wait, respawn on <sentinel>
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

**Without `--supervise` the route does not exist.** A server started plainly has
no one to restart it, and an endpoint that stops it would leave a dead site with
no way back from the browser. The flag is what makes the promise true.

## 4. Who may press it

**This is the part to decide before any of it is built.**

`rate_limit.py` ships `admin_emails: []`, and its own comment says what that means:
*"Empty list = ANY logged-in session is admin (the implicit default that ships)."*
The server binds `0.0.0.0` behind OAuth.

**So with the shipping default, a restart button next to the user's email hands
every person who can log in the ability to disconnect everyone else** — mid
calculation, mid edit. Workspace writes are fire-and-forget, so in-flight ones are
lost.

Three conditions, and the first is not optional:

1. **A real admin gate.** `admin_emails` must be non-empty for the route to exist
   at all — an empty list disables the endpoint rather than admitting everybody.
   That inverts the current default *for this route only*, and it is the safe
   direction: the failure mode of a mistake is "the button is missing", not
   "anyone can restart the server". (Task #15 already carries an admin-gate
   review.)
2. **The button is hidden when the route is absent**, so a non-admin never sees a
   control they cannot use.
3. **A confirm that names the cost** — *"this disconnects everyone using this
   server and drops saves that are still in flight."*

## 5. Order

| | Step | Note |
|---|---|---|
| **A** ✅ | ~~The version guard~~ **static revalidation (§ 2) — done 2026-08-03** | independent of everything else, no new surface, useful on its own |
| **B** | `serve --supervise` — the parent loop, no route yet | testable alone: start it, kill the child by hand, watch it come back |
| **C** | The admin gate: route exists only when `admin_emails` is non-empty | before the route does anything |
| **D** | `POST /api/admin/reload` — reply, then exit with the sentinel | |
| **E** | The button + the poll-and-reload, shown only when the route exists | |

**A is worth doing whether or not B–E are**, and it is the half that removes the
confusion people actually hit.

## 6. What this does not do

- **It does not make the dev server a production server.** `app.run()` is still
  Werkzeug's dev server; supervision does not change that, and
  [`deployment.md`](?doc=ops/deployment.md) still governs how it is exposed.
- **It does not reload without a restart.** There is no module-swapping and no
  partial reload — a fresh child is the whole mechanism, which is why it cannot
  leave half the app on old code.
- **It does not preserve in-flight work.** A restart drops it, which is why § 4.3
  says so out loud rather than hiding it behind a spinner.
