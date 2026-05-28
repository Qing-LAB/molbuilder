# Projects sidebar — sole source of truth

**Status**: canonical reference for the Projects sidebar.  Everything
another module needs to consume the sidebar — API, data shapes, CSS
classes, integration patterns, lifecycle, visual states — lives here.

**Contract**: when this doc and the code disagree, ONE of them is
wrong.  Fix the disagreement in the same commit; never let one outrun
the other.  Reviewers reject PRs that change sidebar behaviour
without a matching doc update.

**Related (referenced, not duplicated)**:
* [`selection.md`](selection.md) — the cursor model + tab integration
  contract.  This doc embeds its main concepts inline (so you can
  read this alone) but defers to selection.md for boundary cases
  (multi-tab semantics, anti-pattern history).
* [`web-api.md`](web-api.md) — the `/api/files/*` backend protocol.

**Code map** (file → ownership; see § 3 for details):

```
molbuilder/web/templates/_projects_sidebar.html      template
molbuilder/web/static/lib/projects-sidebar.css       styles
molbuilder/web/static/lib/projects-sidebar.js        entry / bootstrap
molbuilder/web/static/lib/projects/api.js            HTTP wrappers
molbuilder/web/static/lib/projects/state.js          state + public API
molbuilder/web/static/lib/projects/list.js           list + breadcrumb + lock UI
molbuilder/web/static/lib/projects/forms.js          create / mkdir / upload forms
molbuilder/web/static/lib/projects/preview.js        preview modal
molbuilder/web/blueprints/files.py                   /api/files/* backend
molbuilder/web/blueprints/projects.py                /api/projects/create backend
tests/test_web_files.py                              backend tests
tests/test_projects.py                               project-create tests
tests/test_sidebar_lock_api.py                       Playwright: lock + visibility
```

---

## 1. Goal

A single, **content-agnostic** file browser pinned to the left of every
main tab.  It owns:

* **One cursor pair** — `(current_dir, current_file)` — that any tab
  can read or subscribe to.
* **Primitives** — list / read / write / mkdir / upload / delete /
  create-project — backed by the `/api/files/*` and
  `/api/projects/*` HTTP endpoints.
* **A sidebar-wide lock** — long-running multi-step pipelines (Save
  FDF + install pseudos + install wrapper) acquire it so the user
  can't re-navigate mid-pipeline and silently retarget a downstream
  step.

The sidebar is **passive**: it publishes state changes; tabs subscribe
and decide what to do.  The sidebar never triggers tab loaders,
navigates between tabs, or knows which tab consumes which file type.

---

## 2. Scope (what it does / does not do)

| Does | Does NOT |
|---|---|
| Browse `projects/` (the only allowed root) | Browse anywhere outside `projects/` |
| Hold the current cursor in sessionStorage | Hold any per-tab state |
| Provide `read/write/mkdir/upload/delete/createProject` API | Implement format-specific logic (XYZ, FDF, PDB parsing) |
| Lock itself during multi-step save pipelines | Track per-tab busy state |
| Show a preview modal for `getCurrentFile()` | Mount tab-specific UIs (3Dmol, Plotly, etc.) |
| Render its own list / breadcrumb / forms / modal | Auto-load any file into any tab |
| Fire publish events when state changes | Poll for changes (subscribers must opt in) |

---

## 3. Architecture

### 3.1 Module split

| Module | Owns | Imports | Imported by |
|---|---|---|---|
| `projects-sidebar.js` | bootstrap order; mounts `window.molbuilder.projects` | api, state, list, forms, preview | template `<script type="module">` |
| `projects/api.js` | HTTP wrappers; no state, no DOM | nothing | state, list, forms |
| `projects/state.js` | sessionStorage cursor; module-private subscriber sets; lock state; public Inquire API | api | list, forms, preview, sidebar |
| `projects/list.js` | breadcrumb + entry list DOM; per-entry buttons; `openDir`; **lock-UI subscription (`initLockUI`)** | state, api, preview | sidebar |
| `projects/forms.js` | New project / New subdir / Upload forms; depth-aware visibility | state, api, list | sidebar |
| `projects/preview.js` | file preview modal | state | sidebar, list |

**Hard rule**: no module reads another module's "private" closure
state.  All cross-module communication goes through the exported APIs
in § 6.

### 3.2 Where data lives

```
                  ┌──────────────────────────────────────────────────┐
sessionStorage    │ molbuilder.current_dir   molbuilder.current_file │
(persistent       └──────────────────────────────────────────────────┘
 per tab)                       ▲                  ▲
                                │ setShared        │ setShared
                                │                  │
                  ┌──────────────────────────────────────────────────┐
state.js          │   selectionSubscribers (Set<cb>)                 │
(module           │   lockSubscribers      (Set<cb>)                 │
 closure)         │   lockState            (null | {reason, ...})   │
                  │   projectsRoot         (string)                  │
                  │   refreshHandler       (1 slot)                  │
                  └──────────────────────────────────────────────────┘
                                ▲ publish*Change      ▲ lock/unlock
                                │                     │
                  ┌──────────────────────────────────────────────────┐
DOM (derived)     │  .ps-entry.is-selected   .projects-sidebar       │
                  │  .ps-lock-banner[hidden] .is-locked              │
                  └──────────────────────────────────────────────────┘
```

**The DOM is derived state.**  Authoritative state lives in
sessionStorage (cursor) and the closure (lock + subscribers + root).
Any new DOM update must trace back to a state mutation that published
a change — never the reverse.

---

## 4. Initialization lifecycle

### 4.1 Module load (synchronous, before `DOMContentLoaded`)

1. `state.js` evaluated → `projects` object created; subscriber sets
   created (empty).
2. `projects-sidebar.js`: `window.molbuilder.projects = projects` —
   **public API is reachable from this instant**, before `init()`
   runs.  `lock()` / `getCurrentDir()` / `onChange()` all work; only
   DOM-rendered effects of state changes are gated on init.
3. Optional: registered in `window.molbuilder.runtime` (for
   `whenReady("projects")`).

### 4.2 `init()` (async, on `DOMContentLoaded`)

```js
async function init() {
  const sidebar = document.getElementById("projects-sidebar");
  if (!sidebar) return;       // (a) partial not included on this page

  initLockUI();               // (b) UNCONDITIONAL DOM wiring

  const roots = await apiRoots();
  if (roots.length === 0) {
    return;                   // (c) "no projects root" UX path
  }
  setProjectsRoot(roots[0].path);

  initList();                 // (d) data-dependent DOM wiring
  initForms();
  initPreview();

  await openDir(start);       // (e) first directory listing
  restoreSelection();
}
```

### 4.3 The load-bearing rule

> **DOM wiring that must work regardless of project-root state belongs
> in step (b) or earlier.  DOM wiring that genuinely depends on
> project-listing data belongs in step (d).**

Lock UI is in (b) so a Save pipeline can lock the sidebar even if no
`projects/` root exists.  Selection rendering + create-forms are in
(d) because they have no meaning without a project root.

**When in doubt, default to (b).**  Wiring something unconditionally
that turns out unnecessary costs nothing; gating something on data
that doesn't arrive produces a silent "click does nothing" — the bug
class we keep hitting (2026-05-28 sighting).

### 4.4 Pages that include the sidebar

`index.html` (Build), `modify.html`, `spectra.html`, `results.html`.
Each does:

```jinja
{% include "_projects_sidebar.html" %}
...
<script type="module" src="{{ url_for('static', filename='lib/projects-sidebar.js') }}"></script>
```

`<body class="has-projects-sidebar">` is set **server-side** in each
template (NOT toggled by JS) so CSS reserves left-padding from first
paint.

---

## 5. Data model — all shapes

### 5.1 sessionStorage keys

```
molbuilder.current_dir    string  absolute path, always inside projects/
molbuilder.current_file   string  absolute path, "" when only browsing
```

Always coherent: `current_file === ""` OR `current_file` is a file
whose parent is `current_dir`.  Mutating one without the other through
a path other than `setShared(dir, file)` is forbidden.

### 5.2 Subscription payloads

```ts
// onChange(cb)  — fires on cursor mutations + once on register
SelectionPayload = {
  dir:  string,   // current_dir
  file: string,   // current_file ("" when only browsing)
}

// onLockChange(cb)  — fires on lock transitions + once on register
LockPayload = {
  locked: boolean,
  reason: string,   // "" when unlocked, else the message passed to lock()
}
```

### 5.3 Lock state (internal)

```ts
LockState =
  | null                              // unlocked
  | {
      reason: string,
      cancelers: Array<() => void>,
    }
```

`null` means unlocked.  `lock()` creates the `{reason, cancelers}`
object; `unlock()` resets to `null`.  Re-entry (`lock()` while
`lockState !== null`) **throws** — nested locks would tangle Cancel
semantics.

### 5.4 Return shapes from the public API

```ts
// writeFile / saveToWorkspace success
WriteOk = {
  ok:      true,
  path:    string,    // absolute path written
  relPath: string,    // path shortened to projects/...
  size:    number,    // bytes written
  mtime:   number,    // unix epoch (server's clock)
}

// writeFile / saveToWorkspace failure
WriteErr = { ok: false, error: string }

// saveToWorkspace also returns null when atRoot() is true (silent skip)
SaveToWorkspaceReturn = WriteOk | WriteErr | null

// readCurrentFile
ReadResult = { path: string, text: string } | null
```

### 5.5 Backend `/api/files/*` envelopes (relevant subset)

```ts
// GET /api/files/roots
RootsResp = { roots: Array<{ path: string, /* …other meta… */ }> }

// GET /api/files/list?path=…
ListResp = {
  ok:      boolean,
  path:    string,            // canonical resolution of input path
  entries: Array<{
    name: string,
    kind: "file" | "directory" | "symlink",
    size: number | null,      // null for non-files
  }>,
  error?:  string,
}

// GET /api/files/read?path=…
ReadResp = { ok: true, path: string, text: string } | { ok: false, error: string }

// POST /api/files/mkdir
MkdirResp = { ok: true, path: string } | { ok: false, error: string }
```

Full backend contract: see [selection.md § 6](selection.md) and
[web-api.md](web-api.md).

---

## 6. Public API reference

All methods live under `window.molbuilder.projects.*`.  Stable
surface — adding / changing a method requires updating this section
AND, if cursor semantics change, updating [selection.md § 3](selection.md).

### 6.1 Selection (read)

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `getCurrentDir()` | `() => string` | `current_dir` | Always set after init. `""` only before init OR if no projects root. |
| `getCurrentFile()` | `() => string` | `current_file` | `""` when only browsing. |
| `getProjectsRoot()` | `() => string` | `projectsRoot` | `""` until `apiRoots()` resolved. |
| `atRoot()` | `() => boolean` | true iff `current_dir` is unset or equals `projectsRoot` | Use this — not raw `!!dir` — for "can saveToWorkspace land here". |
| `relativeToProjects(path)` | `(string) => string` | display-shortened path | Shortens an absolute path to its `projects/…`-relative tail. |
| `isLocked()` | `() => boolean` | true iff a lock is currently held | |
| `getLockReason()` | `() => string` | the message passed to `lock()` | `""` when unlocked. |

### 6.2 Selection (write)

| Method | Signature | Returns | Effect |
|---|---|---|---|
| `writeFile(path, text, opts?)` | `(string, string, {overwrite?, expected_mtime?}) => Promise<WriteOk \| WriteErr>` | success or error | Writes to exact path via `/api/files/write`. Auto-refreshes sidebar if the write landed in current dir. |
| `saveToWorkspace(text, filename, opts?)` | `(string, string, {overwrite?, expected_mtime?}) => Promise<SaveToWorkspaceReturn>` | success / error / `null` | Convenience: writes to `<current_dir>/<filename>`. Returns `null` (no error) when `atRoot()` is true — callers fall back to a local Download. |
| `readCurrentFile()` | `() => Promise<ReadResult>` | `{path, text}` or `null` | Convenience wrapper over `/api/files/read`. |
| `refresh()` | `() => Promise<void>` | `void` | Re-list the current directory. No-op + console warning if `setRefreshHandler` wasn't called (init order broken). |

### 6.3 Subscriptions

Both subscriptions follow the **fire-once-immediately** rule: `cb()`
is invoked synchronously at registration with the current state, so
subscribers can initialise without a separate `getCurrent*()` call AND
can't miss the "first" event by subscribing too late.

| Method | Signature | Fires with |
|---|---|---|
| `onChange(cb)` | `(cb: (SelectionPayload) => void) => UnsubscribeFn` | Cursor mutations (via `setShared`) + initial state |
| `onLockChange(cb)` | `(cb: (LockPayload) => void) => UnsubscribeFn` | Lock transitions (`lock` / `unlock`) + initial state |

`UnsubscribeFn = () => void` — call it to remove the subscriber.

### 6.4 Lock

| Method | Signature | Behaviour |
|---|---|---|
| `lock(reason, cancelers)` | `(string, Array<()=>void>) => LockState` | Acquires the lock. **Throws** if already locked. `cancelers` is a list of zero-arg functions the Cancel button will invoke. |
| `unlock()` | `() => void` | Releases the lock. **Idempotent** — safe to call when already unlocked. |
| `cancelLockedOperation()` | `() => void` | Runs cancelers in order, swallowing per-canceler exceptions. **Does NOT itself unlock** — the operation's own `try { } finally { unlock() }` is responsible after abort unwinds. Safe no-op when unlocked. |

### 6.5 Calling conventions

* All async methods return Promises that **never throw** — failures
  come back in the result envelope (`{ok: false, error}`).  Callers
  display `error` verbatim; backend messages are already actionable.
* Synchronous methods on the API surface never throw EXCEPT `lock()`
  (reentry; intentional fail-fast).
* `onChange` / `onLockChange` callbacks may throw — the publish loop
  catches per-subscriber and continues so one bad subscriber can't
  break the rest.

---

## 7. Interaction with other modules

Every page that loads the sidebar can consume it the same way.  Three
canonical integration patterns:

### 7.1 Read-on-demand (any tab)

```js
const proj = window.molbuilder.projects;
const path = proj.getCurrentFile();
if (path) {
    const r = await proj.readCurrentFile();
    if (r) doSomethingWith(r.text);
}
```

Use for one-shot "give me what's selected right now" — Build form's
psml_lib live-resolution, viewer3D's "load this file" on click, etc.

### 7.2 Subscribe (Build, Modify, Spectra, Watch)

```js
const proj = window.molbuilder.projects;

const unsubscribe = proj.onChange(({dir, file}) => {
    // Re-render any tab UI that depends on what's selected.
    refreshSaveButtonAvailability(dir);
    updateWorkspaceLabel(dir);
});

// On page teardown (rarely needed for full-page reloads):
// unsubscribe();
```

Use for any tab that displays "where will my Save go?" or gates a
button on whether the current dir / file is suitable.  Subscribers
fire on EVERY cursor mutation AND once immediately on register; the
callback should be safe to run on the initial empty cursor state.

### 7.3 Multi-step save pipeline (Build, Spectra)

```js
const proj = window.molbuilder.projects;

async function savePipeline() {
    const abort = new AbortController();
    proj.lock("Saving FDF + pseudos + wrapper…",
              [() => abort.abort()]);
    try {
        // Step 1: write the .fdf
        const w = await proj.saveToWorkspace(text, filename, {overwrite: true});
        if (!w?.ok) return;
        // Step 2: pseudos
        await fetch("/api/siesta/install-pseudos", {
            method: "POST", body: JSON.stringify({...}),
            signal: abort.signal,    // Layer B
        });
        // Step 3: wrapper
        await fetch("/api/run/install-wrapper", {
            method: "POST", body: JSON.stringify({...}),
            signal: abort.signal,
        });
    } finally {
        proj.unlock();               // Layer A
    }
}
```

**Mandatory pattern** when chaining ≥ 2 backend calls that target the
same workspace.  See § 12 for the lock model's full 3-layer recovery.

### 7.4 Reference implementations (live code; do not reinvent)

* `molbuilder/web/static/viewer.js` — Build tab: `save-fdf` +
  `save-pyscf` click handlers.
* `molbuilder/web/static/lib/spectra/core.js` — Spectra:
  `saveSpectraToCurrentDir`.
* `molbuilder/web/static/lib/projects/forms.js` — internal subscriber
  use (depth-aware section hide).

---

## 8. CSS classes — full catalogue

All sidebar-internal classes use the `ps-` prefix.  Page-level
stylesheets MUST NOT redefine these classes.

### 8.1 Structural (set once in template)

| Class | Element | Role |
|---|---|---|
| `.projects-sidebar` | `<aside id="projects-sidebar">` | root container |
| `.ps-header` | header bar | "Projects" title |
| `.ps-title` | `<h2>` inside header | title text |
| `.ps-create-section` | `<details>` blocks | foldable form sections |
| `.ps-create-summary` | `<summary>` | section toggle |
| `.ps-create-form` | `<form>` | the form itself |
| `.ps-field-label` | `<label>` | field label |
| `.ps-field-hint` | `<small>` | "(in <dir>)" hint |
| `.ps-create-actions` | row | submit + reset buttons |
| `.ps-mkdir-error`, `.ps-create-note` | error / note text | inline messages |
| `.ps-breadcrumb` | `<nav>` | path crumbs |
| `.ps-crumb` | `<span>` | one crumb segment |
| `.ps-crumb-sep` | `<span>` | "/" separator |
| `.ps-list` | `<ul>` | entry list |
| `.ps-entry` | `<li>` | one directory entry |
| `.ps-entry-icon` | `<span>` | leading glyph (▸ / → / ·) |
| `.ps-entry-name` | `<span>` | entry display name |
| `.ps-entry-meta` | `<span>` | file size (right-aligned) |
| `.ps-entry-action` | `<button>` | per-entry hover button |
| `.ps-entry-preview` | preview button | "view" |
| `.ps-entry-delete` | delete button | "×" |
| `.ps-actions` | `<section>` | bottom status bar |
| `.ps-selection` | `<p>` | "Selected: <name>" |
| `.ps-preview-modal` | `<div role="dialog">` | preview modal root |
| `.ps-preview-backdrop` | backdrop layer | click to close |
| `.ps-preview-window`, `.ps-preview-header`, `.ps-preview-title`, `.ps-preview-close`, `.ps-preview-meta`, `.ps-preview-body`, `.ps-preview-error`, `.ps-preview-footer`, `.ps-preview-close-footer` | modal innards | |
| `.ps-lock-banner` | `<div role="status">` | lock banner root |
| `.ps-lock-icon` | `<span>` | ⏳ glyph (bob animation) |
| `.ps-lock-message` | `<span>` | reason text |
| `.ps-lock-cancel` | `<button>` | Cancel button |

### 8.2 State modifiers (toggled by JS)

| Class | Applied to | Means |
|---|---|---|
| `.is-selected` | `.ps-entry` | this entry equals `current_file` |
| `.is-empty` | `.ps-list` | directory has zero children |
| `.is-current` | `.ps-crumb` | last crumb (current dir) |
| `.is-locked` | `.projects-sidebar` | a lock is currently held — fades + disables every direct child except the banner + header |

### 8.3 Body marker (server-rendered)

| Class | Applied to | Means |
|---|---|---|
| `.has-projects-sidebar` | `<body>` | reserve left-padding for the sidebar (set in template, NOT by JS — avoids initial-paint races) |

### 8.4 Z-index reservations

| Layer | z-index |
|---|---|
| `.projects-sidebar` | 5 |
| `.ps-preview-modal` | 100 |

No sidebar-adjacent UI may use z-index ≥ 5 without ensuring its
stacking context doesn't overlap the sidebar.

---

## 9. Visual states catalogue

Every observable UI state of the sidebar, when it appears, and what it
looks like.

| State | When | Visual |
|---|---|---|
| **Idle, empty cursor** | After fresh page load before any click | Breadcrumb shows `projects`; entry list shows projects' children; "No file selected." in actions area. |
| **Browsing a directory** | After clicking a dir crumb / entry | Breadcrumb shows full path; entry list shows that dir's children; no `.is-selected`. |
| **File selected** | After clicking a file entry | That entry gets `.is-selected` (highlight); actions area shows "Selected: <basename>". |
| **Empty directory** | A directory has zero children | `.ps-list.is-empty` displayed; "(empty)" placeholder or nothing. |
| **Listing error** | `apiList` returned `{ok:false}` | Error row inside `.ps-list`; breadcrumb still shows attempted path; `current_dir` cleared to attempt path; `current_file = ""`. |
| **Sidebar locked** | Active lock held | `.projects-sidebar.is-locked` — every child except banner + header at 40% opacity, `pointer-events: none`. Lock banner visible with ⏳ + reason + Cancel. |
| **Preview modal open** | User clicked the "view" button on a file entry | Modal floats at z-index 100; backdrop covers the page. Body shows file text; Save button is disabled (not implemented). |
| **No projects root** | `apiRoots()` returned `[]` | `.ps-list.is-empty` with a red error line. List + forms NOT wired. Lock UI IS wired (so save-pipeline locks still work). |
| **New-project / New-subdir form expanded** | User clicked `<summary>` of a `<details>` section | Form fields revealed; context label shows current dir; submit → backend → openDir(new path) on success. |
| **Form error** | Backend returned `{ok:false}` on a mutation | `.ps-mkdir-error` shows `j.error` verbatim. Form NOT reset (user can edit + retry). |

---

## 10. Visibility model — `[hidden]` versus author CSS

### 10.1 The trap

The UA stylesheet's `[hidden] { display: none }` has specificity
`(0,1,0)`.  Any author CSS rule `.foo { display: <non-none> }` ALSO
has `(0,1,0)`.  On a specificity tie, **author CSS wins** by cascade
order — so any element with `class="foo"` AND `hidden=""` is rendered
VISIBLE despite the attribute.

### 10.2 The rule (load-bearing)

> Every author CSS rule that sets `display: <non-none>` on a class
> whose DOM element may carry the `hidden` attribute **MUST be paired
> with**:
>
> ```css
> .foo[hidden] { display: none; }
> ```

Specificity `(0,2,0)` beats both `[hidden]` and `.foo`.  No
`!important`, no ordering trick required.

### 10.3 Adding a new `display:` rule — checklist

1. Does the targeted element's `hidden` attribute ever get set
   (template OR `el.hidden = true/false` in JS)?
2. If yes: write the paired `.foo[hidden] { display: none; }` rule
   **in the same diff hunk**.  Reviewers reject changes that add
   `display:` without the paired guard.
3. If unsure: add the guard anyway.  Cost is one line; missing it
   produces a silent always-visible bug.

### 10.4 Currently-guarded rules (do not remove)

```
.ps-lock-banner[hidden]      projects-sidebar.css
.ps-preview-modal[hidden]    projects-sidebar.css
.scf-banner-row[hidden]      lib/trajectory-inspector.css
.plot[hidden]                lib/trajectory-inspector.css
.source-panel[hidden]        style.css
.edit-panel .op-row[hidden]  modify/style.css
.modes-table th[hidden]      spectra/style.css
.modes-table td[hidden]      spectra/style.css
.lock-reason[hidden]         style.css
.tab-panel[hidden]           style.css
.phase-indicator[hidden]     spectra/style.css   (2026-05-28)
.nucleic-row[hidden]         style.css           (2026-05-28)
.issues-panel[hidden]        style.css           (2026-05-28)
.issues-panel[hidden]        spectra/style.css   (2026-05-28)
```

### 10.5 Future: `.is-hidden` convention

The cleanest end state is a single `.is-hidden { display: none
!important }` class + a CI grep banning `\bhidden=` in templates
(allow `aria-hidden`).  Removes the bug class entirely.  Tracked as
gap G3 (§ 16).

---

## 11. Lock model

### 11.1 Why

Multi-step Save pipelines (Build's "Save FDF" emits the .fdf, copies
pseudos, drops the wrapper) read `getCurrentDir()` at each step.
Without a lock, the user can re-navigate the sidebar between steps and
silently retarget downstream steps to a different directory.

### 11.2 Three layers (independent; do not collapse)

```
Layer A   try { ... pipeline ... } finally { unlock() }
          Releases on success AND on throw.

Layer B   AbortController + signal: on every fetch in the locked window
          Bounds the network call's duration; layer C can trigger this.

Layer C   Cancel button in the lock banner runs registered cancelers
          Last-resort user escape hatch when A and B both failed.
```

The independence matters: a bug in any one layer doesn't strand the
sidebar.  Forgotten `finally` → Cancel still works.  Backend hang →
Cancel triggers abort → fetch rejects → `finally` runs → unlock.  JS
exception in the canceler → caught per-canceler, doesn't block other
cancelers; lock state is unaffected.

### 11.3 CSS contract

```css
.projects-sidebar.is-locked > :not(.ps-lock-banner):not(.ps-header) {
    opacity: 0.4;
    pointer-events: none;
    user-select: none;
}
.projects-sidebar.is-locked > .ps-header {
    pointer-events: none;
}
```

Banner + header stay visible; everything else fades and stops
receiving clicks.  Header keeps full opacity (visual anchor); future
header controls (search box, etc.) will be locked automatically.

### 11.4 Lock UI wiring lifecycle

`initLockUI()` (in `list.js`) wires:

* The `onLockChange` subscriber that toggles `.is-locked` + shows the
  banner.
* The Cancel button click handler, via **delegated dispatch on
  `document`** (not `addEventListener` on the button).  Delegation
  survives any future re-rendering of the partial.

`initLockUI()` is called UNCONDITIONALLY in `projects-sidebar.js`'s
`init()`, BEFORE `await apiRoots()`.  **Do NOT move this call.**
Moving it back into `initList()` re-couples the lock UI to project-
listing success and resurrects the 2026-05-28 Cancel-does-nothing bug.

---

## 12. State synchronization rules

### 12.1 Mutation paths

Every state mutation must go through a designated function so the
publish events fire:

| State | Mutate via | Publishes |
|---|---|---|
| `current_dir`, `current_file` | `setShared(dir, file)` in `state.js` | `publishSelectionChange()` → all `onChange` subscribers |
| `lockState` | `lock(reason, cancelers)` / `unlock()` | `_publishLockChange()` → all `onLockChange` subscribers |
| `projectsRoot` | `setProjectsRoot(root)` | No subscribers today (gap G4) |
| `refreshHandler` | `setRefreshHandler(handler)` | No subscribers (1 slot only) |

### 12.2 Forbidden patterns

* Writing `sessionStorage.setItem("molbuilder.current_dir", …)` directly
  from any module.  Use `setShared`.
* Mutating `lockState` directly from any module other than `state.js`.
  Use `lock()` / `unlock()`.
* Reading another module's closure state directly.  Use the exported
  getters.
* Driving DOM from inside an event handler when a publish event could
  drive a subscriber instead.  See gap G1.

### 12.3 The publish-then-react flow

```
1. User action / API call mutates state via the designated function.
2. The function calls publish*Change() before returning.
3. publish*Change() iterates subscribers, calling each in try/catch.
4. Each subscriber updates its own DOM region (or other side effect).
5. Subscribers that throw are caught; loop continues; lock state unchanged.
```

This is the contract.  Any new sidebar state belongs in a closure
variable + a `publishXChange` + an `onXChange` subscription API.

---

## 13. Backend contract (summary)

Full version in [selection.md § 6](selection.md) and
[web-api.md](web-api.md).

| Endpoint | Method | Status |
|---|---|---|
| `/api/files/roots` | GET | shipped |
| `/api/files/list` | GET | shipped |
| `/api/files/read` | GET | shipped |
| `/api/files/stat` | GET | shipped |
| `/api/files/mkdir` | POST | shipped |
| `/api/files/write` | POST | shipped |
| `/api/files/upload` | POST | **501 stub** (UI wired) |
| `/api/files/delete` | DELETE | **501 stub** (UI wired with hover ×) |
| `/api/files/rename` | POST | not built |
| `/api/projects/create` | POST | shipped |

All `apiX()` wrappers in `api.js` return either the JSON body or a
synthesised `{ok: false, error: …}` on network failure.  Callers
never need a try/catch around `apiX()` calls.

---

## 14. Testing strategy

| File | Layer | What it pins |
|---|---|---|
| `tests/test_web_files.py` | backend (Flask test client) | endpoint contracts: status codes, JSON shapes, path safety, depth-aware name rules |
| `tests/test_projects.py` | backend | `/api/projects/create` atomicity, conflicts, naming |
| `tests/test_sidebar_lock_api.py` | Playwright | lock contract (acquire / release / re-entry throw); subscribers fire-on-register; Cancel runs cancelers in order + safe when unlocked; **DOM rendered visibility via `getComputedStyle().display`** — catches the § 10 specificity-trap regression directly. |
| `tests/test_inspector_registry_e2e.py` | Playwright | sidebar selection → results-tab inspector dispatch |

### 14.1 Adding tests for new sidebar features

| Change kind | Required test |
|---|---|
| New DOM state class | Playwright test using `getComputedStyle()` (NOT just `el.classList.contains` — does not catch CSS bugs) |
| New `window.molbuilder.projects.*` method | Playwright test driving it via `page.evaluate`, asserting subscribers fire, returns shape matches § 5 |
| New backend endpoint | `tests/test_web_files.py` test pinning status codes + JSON envelope shape |
| New CSS `display:` rule | Confirm paired `[hidden]` guard per § 10 |

---

## 15. Anti-patterns (do not reintroduce)

| Anti-pattern | Why retired |
|---|---|
| Wiring UI behaviour in an init function gated on a network call's success | 2026-05-28 Cancel-does-nothing bug. UI wiring belongs in `init()` step (b); data-dependent wiring in step (d). |
| Setting `display:` on a class without a paired `[hidden]` guard | 2026-05-28 multi-site bug. See § 10. |
| Reading another module's closure state directly | Module ownership boundary. Use the exported APIs in § 6. |
| `OPEN_TARGETS` hard-coded extension → tab map in the sidebar | Tabs handle their own extension matching via `onChange`. |
| `window.molbuilderTabAutoLoad` per-tab auto-load on `DOMContentLoaded` | Implicit action — races user clicks. Explicit pull (button) instead. |
| Calling `lock()` re-entrantly | Tangles Cancel semantics. Compose pipelines in a single outer lock. |
| Updating DOM from an event handler when a publish event could drive a subscriber | Hand-coordinated DOM → state drift. See gap G1. |
| Pointing the sidebar at a path outside `projects/` | `projects/` is the source of truth scope; outside files come in via upload. |
| Triggering a tab's loader from sidebar code | Sidebar is content-agnostic; tabs subscribe to selection and decide. |

---

## 16. Identified gaps (roadmap)

Severity: **B** = should fix soon (real correctness/UX risk),
**F** = future (no current cost).

| ID | Sev | Gap | Proposed fix |
|---|---|---|---|
| G1 | B | `_markSelected` + `_renderSelectionStatus` are called inline from event handlers, NOT from an `onChange` subscriber. Any mutation through a different path leaves DOM out of sync. | Introduce `renderSidebar(state)` subscribed to `onChange`; remove inline calls. Task #166. |
| G2 | F | No cross-tab sync. sessionStorage is per-tab; opening two tabs leaves them drifting on cursor. | `window.addEventListener("storage", …)` in `state.js` calling `publishSelectionChange()`. |
| G3 | F | The `[hidden]` + author-`display:` trap is mitigated case-by-case (§ 10). | Convert to a single `.is-hidden` class + CI grep banning `\bhidden=` in templates. |
| G4 | F | `setProjectsRoot` has no subscribers (today no consumer needs to react). | Add `onProjectsRootResolved` if a real consumer arrives, OR fold root into `onChange`'s payload. |
| G5 | F | `refreshHandler` is a single slot — multiple consumers would each need their own wrap. | Convert to a Set if/when a second consumer arrives. |
| G6 | B | Upload + delete + rename backends are 501 stubs. UI surface is wired. | Implement in `web/blueprints/files.py` against the existing endpoint shapes (selection.md § 6). Mechanical work; naming-rule + path-safety helpers already exist. |
| G7 | F | Preview modal's Save button is permanently disabled (`title="coming soon"`). | Either ship the edit-and-save flow against `/api/files/write` or remove the button. |
| G8 | F | No telemetry / diagnostic logging for lock acquire/release. A misbehaving subscriber that throws is silently swallowed. | Optional: emit a custom event on `document` for each lock transition that DevTools can capture. |

---

## 17. Change protocol

1. Open this doc.  Find the section that covers what you're about to
   change.
2. Edit the doc to describe the new behaviour (in the same diff).
3. Make the code change.
4. Verify tests pass; add tests where the spec changed (§ 14).
5. Commit doc + code + tests together.  Reviewer rejects any one of
   the three missing.

If you find a code-vs-doc discrepancy: file an issue, do NOT silently
"fix" the doc to match buggy code (or vice versa).  The doc is design
intent; the code is implementation.  Decide which is wrong, then fix
in one commit that aligns them.
