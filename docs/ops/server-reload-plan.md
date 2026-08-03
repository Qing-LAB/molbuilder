# Reloading a running server — a plan

**Role:** plan (proposed, not started)
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
| **JS / CSS / templates** | the browser is serving its cached copy | a **new URL** for the changed file |

Templates need neither: Jinja re-reads from disk on request. Static files are read
per request too — so **only the browser's cache makes them look stale**, and only
Python genuinely needs the process replaced.

## 2. Half one — the version guard on static assets

Static URLs carry a version, so a changed file is a different URL and the browser
fetches it. Nothing is restarted and nothing is invalidated by hand.

The version is the app's own build identity — the same string `/api/health`
already reports — so one value moves every asset at once and there is no per-file
bookkeeping to get wrong.

**Why this is not "cache-busting as a workaround".** Today there is no
`SEND_FILE_MAX_AGE` and no version, so correctness depends on the user knowing to
hard-reload. A URL that changes when the bytes change is the honest statement of
what happened.

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
   reloads the page — which picks up new JS through § 2's version.

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
| **A** | The version guard (§ 2) | independent of everything else, no new surface, useful on its own |
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
