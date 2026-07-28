# Notifications — the app-wide message framework

**Role:** contract
**Domain:** web
**Companions:** [`runtime.md`](?doc=web/runtime.md) — the registry it registers
with, and the other shared building blocks; [`workspace.md`](?doc=web/workspace.md)
— the one built-in source today (a failed save); [`overview.md`](?doc=web/overview.md)
— the module registry; [`roadmap.md`](?doc=roadmap.md) — the pending ES-module pass.

`notify` (`lib/app-notifications.js`, `window.molbuilder.notify`) is the **one
consistent surface for system-level messages** that matter regardless of which
tab you're on — a stack of notifications you clear individually. It is built as a
**general framework** any part of the app can call; today it has a single caller
(a failed background save), but the API is open to all.

## 1. The API — any caller, any level

```js
const h = notify.show({ level: "error", message: "…", id: "save-failed" });
// … later, programmatically:
h.dismiss();
```

| Call | What it does |
|---|---|
| `show({ level, message, id?, detail? })` → `{ dismiss }` | Push (or update) a message; returns a handle to dismiss it. |
| `clear(id)` | Remove one message. |
| `clearAll()` | Remove all. |
| `list()` | A snapshot of the current messages (for tests / introspection). |

- **Levels** are `error`, `warn`, `info` (an unknown level falls back to `info`).
- **Dedup by id.** A repeated `id` **updates** its existing row (new message + a
  `×N` counter) instead of stacking — so a burst of the same failure is one row,
  not fifty.
- It **registers with the runtime** (`register("notify", …)`), so a consumer can
  `whenReady("notify")` regardless of load order.

The persist-error toast is *not* baked into the API — it's just one caller (§3)
layered on top of this general surface.

## 2. Where it shows — one host, always present

Messages render into a single host, `#app-notifications`. The important design
point: **the host, its stylesheet, and the script all live together in the shared
`_app_header.html`**, which every tab page includes. So the module and its host
*travel together* — a page can never call `show()` into a missing host, and
messages are never silently dropped.

The host is a `role="region"` live region, `hidden` until there's something to
show. Within it:

- **newest on top**; each row has a **×** dismiss button; a **Clear all (N)**
  button appears once there are 2 or more.
- **Accessibility:** an `error` row is `role="alert"` (assertive — announced at
  once); `warn`/`info` are `role="status"` with `aria-live="polite"`. Escape
  dismisses the focused row.

## 3. The one built-in source — a failed save

The workspace persistence layer, when a save to disk fails, dispatches a
`molbuilder:persist-error` DOM event (`lib/workspace/dispatcher.js`). `notify`
listens for it and shows a single, stable-id error row —
*"Couldn't save state to disk … your edits are kept in memory, but
retract / crash-recovery history may be incomplete."* A burst of failures updates
that one row rather than stacking. This is the whole reason the framework exists
so far; everything else in the API is there for the callers to come.

## 4. Where the module stands (current → target ESM)

Today `notify` is a **classic `window.molbuilder.*` script**. Converting it to an
**ES module** — and, in the same pass, building it out as the framework it's meant
to be — is tracked as **task #105**:

- `export` the API; keep a transitional global door + the `persist-error` wiring
  for classic callers until they `import`.
- **Add an optional auto-dismiss (`ttl`)** so transient `info`/`warn`/success
  messages can expire on their own — today *nothing* auto-dismisses, every message
  persists until you clear it, which is right for errors but heavy-handed for a
  passing "saved" note.
- Decide the `detail` field (surface it, or drop it — it's accepted but unused
  today) and whether to add a `success` level.

When that lands, this section is removed.

## 5. Test map

- `test_app_notifications_e2e.py` — the notification bar end to end (show, dedup,
  dismiss, clear-all, the persist-error source).
