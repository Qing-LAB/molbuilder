# `molbuilder-runtime.js` — module registry + ready Promises

`lib/molbuilder-runtime.js` is the **module init contract**. It
solves the "classic-script consumer runs before `type="module"`
producer initializes" problem: instead of polling for
`window.molbuilder.foo`, consumers `whenReady("foo")` and resolve
on a Promise.

This doc is the sole source of truth for the registry's contract.

**New to it?** Start with the plain-language [`runtime-registry-guide.md`](../runtime-registry-guide.md) — the developer on-ramp (why it exists, register/whenReady patterns, gotchas).
The implementation is `molbuilder/web/static/lib/molbuilder-runtime.js`;
the node-driven unit tests are
`tests/test_molbuilder_runtime_js.py` (19 tests, see § 5 below).

---

## 1. Why this exists

Mixing classic `<script src=…>` tags with `<script type="module"
src=…>` in a single template means:

- Classic scripts execute synchronously during HTML parsing.
- Module scripts defer to end-of-parse, then execute in order.
- **A classic script that does `(window.molbuilder || {}).projects`
  at IIFE time sees `undefined`** because the projects module
  hasn't run yet.

Before the registry, consumers polled (`if window.molbuilder &&
window.molbuilder.projects`) or guarded each callsite. The
registry inverts this: producers `register("name", api)`;
consumers `whenReady("name").then(api => …)`. No polling, no
script-tag-order superstitions.

---

## 2. Public API

`window.molbuilder.runtime` exposes five functions. All are
documented in the module's header docstring; this table is the
contract.

| Method | Signature | Behavior |
|---|---|---|
| `register` | `(name: string, api: any) → void` | Register `api` under `name`. Throws `TypeError` for empty/non-string name OR `null`/`undefined` api. Falsy-but-non-null (0, "", false, {}) accepted. Re-registration **warns** to `console.warn` + replaces (programmer bug; page doesn't break). Drains pending `whenReady` waiters synchronously on the microtask queue. |
| `whenReady` | `(name: string) → Promise<any>` | Returns a Promise that resolves with `api` when `name` is registered. If already registered, resolves on next microtask. Rejects (not throws) with `TypeError` on empty/non-string name — uniform sync/async error handling means callers always get a Promise. |
| `get` | `(name: string) → any \| undefined` | Synchronous peek. Returns `undefined` if not registered. Prefer `whenReady` in production code; `get` is for devtools + tests. |
| `listRegistered` | `() → string[]` | Sorted snapshot of registered names. **Not a live view** — mutating the returned array does not affect the registry. |
| `listPending` | `() → string[]` | Sorted snapshot of names with outstanding `whenReady` waiters but no registration yet. Diagnoses "consumer hung forever" bugs. |

---

## 3. Load order

The IIFE auto-mounts on `window.molbuilder.runtime` (also
`globalThis` in non-DOM environments). It is **idempotent**: if
loaded twice (template double-includes the script), the second
IIFE no-ops and the first registry instance persists, so modules
registered before the second load survive.

`molbuilder-runtime.js` **MUST be loaded first** — before any
other molbuilder script — so `register` and `whenReady` are
defined when other modules start their IIFEs. Convention: include
it right after the 3Dmol vendor script in every template (`build`,
`modify`, `spectra`, `results`).

---

## 4. Registered modules (as of 2026-06-02)

The naming scheme is **flat, dotted, lowercased**.

| Name | Source file | Producer side |
|---|---|---|
| `projects` | `lib/projects/projects-sidebar.js` | type=module |
| `selection.panel` | `lib/selection-panel.js` | classic IIFE |
| `selection.viewerAdapter` | `lib/selection/viewer-adapter.js` | classic IIFE |
| `modify.handle` | `modify/viewer.js` | classic IIFE (per-tab); embed handle, not raw 3Dmol viewer |
| `modify.loadStructureText` | `modify/viewer.js` | classic IIFE (per-tab) |
| `inspectors` | `lib/inspectors/registry.js` | classic IIFE |

Adding a new module-with-a-global: pick a name in this scheme,
call `register()` at the END of your IIFE, add the name to this
table.

### Non-registered globals (synchronous IIFE-only)

These attach to `window.molbuilder.<x>` without going through the
registry because they load as classic synchronous scripts BEFORE
any `type="module"` consumer could ask for them:

- `formSchema` — `lib/form-schema.js`
- `path` — `lib/path-utils.js`

Consumers that need them can read `window.molbuilder.<x>`
directly — the load-order guarantee is enforced by the `<script>`
tag order in templates.

**MolView embed helpers — no longer globals (ESM migration).**
`lib/viewer/mol-style.js` (`style`) and `lib/viewer/mol-format.js` (`fmt`) used to
sit in the list above, but they are now pure ES modules in the
MolView embed graph and publish **no** global — the embed `import`s
`spec` / `formula` (and `axes` from `lib/viewer/mol-axes.js`) directly.
`lib/viewer/mol-viewer.js` is also an ES module now (imported via
`lib/molview/index.js`) but still publishes `window.molbuilder.viewer`
on purpose: it is the shared-embed **seal** (`.create` / `.embed`),
a live door other subsystems read, not migration scaffolding. See
`docs/protocols/molview-esm-finalization.md`.

---

## 5. Test coverage

`tests/test_molbuilder_runtime_js.py` (19 node-driven tests, audit
task #191, 2026-06-02) pins the contract end-to-end:

- Surface presence (5 methods).
- Idempotence on double-load.
- `register`: round-trip via `get`; sorted `listRegistered`;
  empty/non-string name rejection; null/undefined api rejection;
  falsy-but-non-null acceptance; re-register warns + replaces.
- `whenReady`: immediate resolution if registered; deferred
  resolution on late register; multiple consumers all resolved by
  one register; empty name rejects (not throws); ONE consumer
  throwing in `.then()` does NOT block OTHER consumers (Promise
  microtask isolation).
- `list*`: sorted snapshot; `listPending` tracks waiters; mutating
  returned array doesn't poison the registry.
- `get`: round-trip; `undefined` for unregistered name.

---

## 6. Subtleties + load-bearing details

**Consumer-error isolation comes from Promise microtask semantics,
NOT from the registry's try/catch.** The registry wraps
`resolve(api)` in a try/catch as belt-and-braces, but `resolve()`
itself doesn't throw. If a consumer's `.then(handler)` throws,
that's an unhandled rejection — other consumers' `.then()`
handlers still run independently because each lives in its own
microtask. The test that exercises this scenario catches its own
throw (`.catch(() => {})`) so Node doesn't crash on the unhandled
rejection; in production callers are expected to attach their own
`.catch()`.

**Re-register replaces silently with a `console.warn`** rather
than throwing. The reasoning: a double-register is a programmer
bug (two modules claiming the same name), but throwing would
break the whole page; warning + replacing keeps the page
functional and surfaces the issue to devtools.

**`whenReady("missing")` resolves on next microtask** (not the
current one) when the name is already registered. This keeps
timing consistent across "already registered" vs "registered
later" — consumers can always assume `.then` runs at microtask
time.

---

## 7. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-21 | Land `lib/molbuilder-runtime.js` as the canonical init contract; require it as the FIRST script in every template. | Build's `.pdb` sidebar pick was silently doing nothing because a classic-script consumer captured `window.molbuilder.projects` at IIFE time, before the deferred `type="module"` sidebar had initialised. Polling fixes the symptom; the registry fixes the structure. |
| 2026-05-21 | The five `window.molbuilder.<x>` utilities (`viewer`, `style`, `fmt`, `formSchema`, `path`) do NOT register with the runtime; they remain plain globals because their load-order is enforced by classic-script ordering. | Going through the registry adds ceremony without solving a real bug — these utilities never race their consumers in practice. |
| 2026-06-02 | Add the 19-test node-driven unit suite (`test_molbuilder_runtime_js.py`) covering surface presence, idempotence, register/whenReady contracts, error isolation, and list-snapshot stability. | The registry had zero direct tests before this; consumers indirectly exercised it via tab e2e. A regression in `register` draining waiters would have surfaced as "the next tab refresh hangs" not "a test fails" — this commit closes the gap. |
