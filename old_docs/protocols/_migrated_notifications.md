# System notifications — the app-wide notification bar (contract)

> **ONE consistent surface for system-level messages that matter regardless of
> which tab the user is on.** It debuts carrying non-blocking persistence
> failures (workspace-contract.md §4.7), and is open to any future cross-tab
> system message. Before this, every subsystem rolled its own `aria-live` region
> (projects lock banner, trajectory `#forces-status`, modify `#status`, …), so a
> workspace persist failure had **nowhere consistent** to surface — it went to
> `console.error` + a DOM event and no further. This is that missing place.

It is deliberately **not** a replacement for transient, tab-local status lines
(structure "Generating…", per-op validation advisories, the projects lock
banner). Those describe one in-tab action and keep their local surfaces. The
notification bar is for messages that **outlive a single tab action** and are
relevant no matter where the user navigates.

---

## §1 Placement

- A single container **`#app-notifications`** lives in the shared app shell
  (`templates/_app_header.html`, which every tab template `{% include %}`s), so it
  renders on **every page**, directly under the tab nav.
- The module **`lib/app-notifications.js`** is loaded on every page (alongside
  `molbuilder-runtime.js`, which every template already loads). Styling lives in a
  shared stylesheet loaded everywhere (`lib/app-notifications.css`).
- The container is empty (and visually absent) until the first notification.

## §2 Behavior — a SEQUENCE the user clears individually

- The area holds a **stack of 0..N notifications** — newest on top. Concurrent
  messages are shown as a **list, never collapsed into one**. A second, different
  message does not replace the first; both are visible until dismissed.
- **Each notification is individually dismissible by the user:** a **×** button on
  a row clears **that** notification; the rest remain. (Plus a "Clear all" affordance
  when 2+ are present.)
- **Levels:** `error` | `warn` | `info`, each severity-styled (semantic color,
  separate from the accent). Error/warn report a real problem; info is incidental.
- **Persistence:** `error` and `warn` stay until the user dismisses them (they name
  a problem the user must know about). `info` MAY auto-expire after a timeout.
  **Errors never auto-dismiss.**
- **Dedup by `id`:** `show({id})` with an id that is already present **updates that
  row in place** (message + a repeat counter + timestamp) instead of stacking
  identical copies — so a burst of the same failure (e.g. repeated persist errors
  during a server outage) is one row that says "×N", not dozens.

## §3 API — `molbuilder.notify`

| Call | Meaning |
|---|---|
| `show({ level, message, id?, detail? }) → { dismiss }` | Add — or, if `id` is already present, UPDATE — a notification. `level` defaults to `"info"`. Returns a handle whose `dismiss()` removes it. |
| `clear(id)` | Remove the notification with that `id` (no-op if absent). |
| `clearAll()` | Remove every notification. |
| `list()` | The current stack `[{ id, level, message, count }]` — a test affordance + lets a consumer reflect state. |

`molbuilder.notify` is registered on `window.molbuilder` and via the runtime
(`runtime.register("notify", …)`), so consumers `whenReady("notify")`.

## §4 Sources

- **Persistence failures (the first consumer).** The module listens for the
  `molbuilder:persist-error` DOM event (workspace-contract.md §4.7 — the
  non-blocking/error-explicit state write) and calls `show()` with a **stable id**
  (`"persist-error"`) at `level:"error"`, so a burst updates one row. Message:
  *"Couldn't save state to disk (…). Your edits are kept in memory, but retract /
  crash-recovery history may be incomplete."*
- **Any consumer** may call `molbuilder.notify.show(...)` for a system-level message.
- **NOT** for transient per-tab status — those keep their local status lines.

## §5 Accessibility + styling

- Container: `role="region"`, `aria-label="System notifications"`. Each **error**
  row is `role="alert"` (assertive); **warn/info** are `aria-live="polite"`.
- The **×** dismiss button is keyboard-focusable with a visible focus ring;
  **Esc** dismisses the focused notification.
- Per [`ui-design-contract.md`](ui-design-contract.md): tokens-only, **ONE** shared
  `.app-notification` class + `--error`/`--warn`/`--info` severity modifiers (grep
  before adding a second), no per-row bespoke spacing; respects
  `prefers-reduced-motion` (no slide animation when set). Semantic severity color is
  distinct from the accent.

## §6 Test affordances

- `molbuilder.notify.list()` returns the current stack for assertions.
- **e2e:** dispatch `molbuilder:persist-error` → exactly one `error` row appears; a
  second dispatch with the same id updates in place (still one row, count 2);
  clicking its **×** removes it; a different `show({id:"x"})` adds a second row that
  survives dismissing the first.
- **node/source pin:** the module registers `notify` + wires the
  `molbuilder:persist-error` listener at top-level scope.

---

## Decisions log

| Date | Decision |
|---|---|
| 2026-07-14 | Created as the app-wide system-notification surface. Motivated by the workspace persist-error contract (§4.7) having no consistent UI home — each subsystem had its own `aria-live` region and a cross-tab system message had nowhere to go. Chosen over a Modify-tab-only banner because persistence (and future system messages) span tabs. Design: a **stack** the user clears **individually** (× per row), dedup by `id`, errors persist until dismissed. Lives in the shared app shell so it is genuinely app-wide. |
