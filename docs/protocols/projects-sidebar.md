# Projects sidebar — architectural design

**Status**: design reference.  Defines the target architecture — the
mission, principles, capabilities, lifecycle, failure-mode coverage —
that the sidebar should embody.  Current code is transitional; the
migration plan from current shape to designed shape lives in § 13.

When this doc and existing code disagree, **the doc wins** until a
deliberate design change updates the doc.  Code changes are reviewed
against the doc, not against other code.

**Related design surfaces**:
* [`selection.md`](selection.md) — the selection-cursor contract; this
  doc references it for cursor semantics and tab integration history.
* [`web-api.md`](web-api.md) — backend endpoint envelope shapes.

---

## 1. Mission

A single, content-agnostic file browser pinned to the left of every
main tab.  It owns the user's notion of "where am I working" and
mediates every filesystem mutation that touches `projects/`.

**The sidebar is passive.**  It publishes; tabs subscribe.  It never
triggers a tab's loader, navigates between tabs, or knows which tab
consumes which file type.  Tabs read the cursor when they need to and
react to changes when they want to.

**The sidebar is scoped to `projects/`.**  Files outside that root
enter via upload; files leave only by being moved out of the tree
manually.  The picker has no concept of arbitrary filesystem
browsing.

---

## 2. Architectural principles

The six load-bearing rules.  Every design decision below traces to
one of these.

1. **One cursor pair.**  `(current_dir, current_file)` covers every
   v1 workflow.  Multi-slot cursors (`input_structure`,
   `compare_file`, …) are a future extension only when a real
   multi-input workflow earns it.  See [selection.md § 7](selection.md).

2. **Pull, don't push.**  The sidebar exposes inquire-and-subscribe
   APIs; tabs read what they need when they need it.  The sidebar
   never reaches into a tab.  Adding a new tab is zero sidebar code.

3. **State is authoritative; DOM is derived.**  Cursor lives in
   sessionStorage.  Lock + projects-root + subscribers live in
   in-memory variables.  DOM is a function of state — rebuilt from
   subscribers when state changes.  Never the other way around.

4. **Mutators publish.**  Every state mutation goes through a
   designated function that writes the store AND fires subscribers.
   No mutation skips the publish; no DOM update happens without a
   state event behind it.

5. **UI wiring decouples from data wiring.**  Anything that must
   work regardless of project-root state — lock UI, future
   diagnostics, future cross-tab listener — wires unconditionally
   at module load.  Anything that has no meaning without a project
   root wires only after `apiRoots()` resolves.  The two phases
   must not share an `init()` step.

6. **Uniform envelopes; no thrown errors at the public surface.**
   Every async public method returns `{ok: true, ...}` or
   `{ok: false, error}` (or `null` for documented no-ops).  Tabs
   NEVER need `try/catch`.  Every async public method also accepts
   an optional `AbortSignal` so a lock's Cancel button can abort
   anything in flight.

---

## 3. Where the sidebar sits

```mermaid
flowchart LR
    user(("user"))

    subgraph page["Page (one per tab)"]
        tabs["Tab UI<br/>(Build / Modify /<br/>Spectra / Results)"]
        sidebar["Projects sidebar"]
    end

    runtime["molbuilder.runtime<br/>(module registry)"]
    backend["Flask backend<br/>/api/files/* + /api/projects/*"]
    fs[(projects/<br/>filesystem)]

    user -->|click| sidebar
    user -->|click| tabs
    sidebar -.publishes.-> tabs
    tabs -.reads cursor.-> sidebar
    sidebar -.HTTP.-> backend
    backend --> fs
    sidebar -->|register| runtime
    tabs -->|whenReady| runtime
```

Tabs and the sidebar live in the same page; the sidebar exposes a
global API (`window.molbuilder.projects.*`) and registers itself in
the runtime module registry so tab JS can `whenReady("projects")`
instead of polling.  The backend is the source of truth for what's
on disk; the sidebar holds a current cursor + a cache of one
directory listing at a time.

---

## 4. State model

The sidebar holds exactly three things.

### 4.1 Cursor

Where the user is looking.  Two slots, always coherent:

* **`current_dir`** — the directory being shown.  Always inside the
  resolved root.
* **`current_file`** — the highlighted file, or empty when only
  browsing.

Lives in `sessionStorage` so it survives reloads within a tab.  Set
exclusively by the cursor mutator; mutation fires `onChange`.

### 4.2 Resolved root

The absolute path of the project tree the backend is serving (e.g.
`/home/.../projects`).  Resolved once at init by asking the
backend; immutable thereafter for the page's lifetime.

Empty before init; non-empty after.  Tabs that need it early use
the runtime registry's `whenReady("projects")` instead of polling.

### 4.3 Lock state

`null` (unlocked) OR `{reason, cancelers}` (a multi-step pipeline is
in progress).  Mutated only via `lock()` / `unlock()`; mutation
fires `onLockChange`.  Re-entry forbidden: a second `lock()` while
held throws.

```mermaid
flowchart TB
    subgraph store["sessionStorage<br/>(persistent per browser tab)"]
        cursor["current_dir<br/>current_file"]
    end

    subgraph mem["state.js — in-memory variables<br/>(per page load)"]
        root["projectsRoot"]
        lock["lockState<br/>(null | {reason, cancelers})"]
        subs["subscribers<br/>(selection + lock)"]
    end

    subgraph dom["DOM — derived<br/>(rebuilt from state)"]
        listEntries["entry list +<br/>.is-selected highlight"]
        banner["lock banner +<br/>.is-locked overlay"]
    end

    cursor -.publishes.-> subs
    lock -.publishes.-> subs
    subs -->|update| listEntries
    subs -->|update| banner
```

**DOM is derived from state.**  Every visual update traces back to a
state mutation that published a change.

---

## 5. Capabilities (what tabs can do)

A tab interacting with the sidebar can do exactly six things.
Implementation-level method signatures are in the source modules;
what matters architecturally is the six capability buckets.

| # | Capability | What it covers |
|---|---|---|
| C1 | **Read the cursor** | Where am I?  What's selected?  Is the sidebar at the root?  What's the resolved projects root?  Is a lock held?  What's the lock reason? |
| C2 | **Subscribe to changes** | Be notified when the cursor changes.  Be notified when the lock state changes. |
| C3 | **Read or write files** | Read the currently-selected file's text.  Write a generated file to the current dir.  Refresh the visible listing. |
| C4 | **Filesystem operations** | Create a project (with full subdir skeleton).  Create a subdir.  Upload a file.  Delete an entry.  Rename an entry.  Navigate the sidebar into an arbitrary path. |
| C5 | **Acquire/release the lock** | Begin a multi-step pipeline; declare cancelers; release in `finally`.  Query lock state. |
| C6 | **Drive the Cancel button** | Run registered cancelers (no-op if unlocked).  Used by the in-banner Cancel button automatically; tab code rarely calls it directly. |

**All six are reachable from `window.molbuilder.projects.*`.**  Each
async method returns a uniform `{ok, ...}` envelope (Principle 6).
Each accepts an optional `{signal: AbortSignal}` so Cancel covers
every in-flight call (Principle 6).

### 5.1 Capability boundaries

Things tabs explicitly **cannot** do via the sidebar:

* Browse outside `projects/` — there is no API.
* Multi-file selection — `current_file` is one slot.
* Per-tab state — sessionStorage holds the cursor only.
* Tab-aware behaviour — the sidebar has no notion of which tab is
  active.

If a future workflow needs one of these, propose a design change to
this doc first.  Do not back-door it through `localStorage` or
custom events.

---

## 6. Subscribe model

The contract for every subscriber API on the sidebar:

```mermaid
sequenceDiagram
    autonumber
    participant Caller as user click<br/>OR tab code
    participant Mut as designated mutator
    participant Store as authoritative store
    participant Subs as subscribers
    participant DOM

    Caller->>Mut: invoke mutator
    Mut->>Store: write new value
    Mut->>Subs: publish(newPayload)
    loop for each subscriber
        Subs->>Subs: try { cb(payload) } catch (swallow)
        Subs->>DOM: subscriber updates its region
    end
    Note right of Subs: one bad subscriber NEVER<br/>breaks the loop — others still fire
```

Three load-bearing properties:

* **Fire-once-immediately on register.**  Every subscribe API calls
  the new callback synchronously with the current state.  Removes
  the "subscribed too late, missed the first event" trap; lets
  subscribers initialise without a separate `getCurrent*()` call.
* **Per-subscriber error isolation.**  The publish loop wraps each
  callback in try/catch.  A throwing subscriber doesn't break the
  rest and never affects lock or cursor state.
* **Unsubscribe returns a closure.**  Subscribers that outlive the
  page can stop receiving events.  Subscribers tied to the page's
  lifetime may discard the closure (and usually do).

---

## 7. Lifecycle

### 7.1 Two-phase init

```mermaid
sequenceDiagram
    participant Browser
    participant Sidebar
    participant State
    participant Backend

    Note over Browser,State: Module load (synchronous)
    Browser->>State: import projects
    Browser->>Sidebar: import init
    Sidebar->>Sidebar: window.molbuilder.projects = projects
    Note right of Sidebar: PUBLIC API REACHABLE FROM HERE

    Browser->>Sidebar: init() (on DOMContentLoaded)

    rect rgb(232, 248, 232)
    Note over Sidebar: PHASE 1 — UNCONDITIONAL UI WIRING
    Sidebar->>Sidebar: wire lock UI + Cancel delegation
    end

    Sidebar->>Backend: GET /api/files/roots
    Backend-->>Sidebar: roots
    alt no projects root
        Sidebar-->>Browser: bail (lock UI still wired!)
    end

    rect rgb(232, 240, 252)
    Note over Sidebar: PHASE 2 — DATA-DEPENDENT UI WIRING
    Sidebar->>Sidebar: wire breadcrumb / list / forms / preview
    Sidebar->>Backend: initial directory listing
    end
```

### 7.2 The load-bearing rule

> **UI wiring that must work regardless of project-root state belongs
> in Phase 1.  UI wiring that genuinely needs the project-listing
> data belongs in Phase 2.**

Phase 1 wiring: lock UI, future cross-tab listener, future
diagnostics.  Phase 2 wiring: entry list, breadcrumb, create-forms,
preview modal.

**When in doubt, default to Phase 1.**  The cost of wiring
unconditionally is one extra DOM lookup per page load.  The cost of
gating on data that doesn't arrive is a silent UI failure with no
errors — the bug class we keep hitting.

---

## 8. Lock model

### 8.1 Why

Multi-step save pipelines (Build's "Save FDF" emits the .fdf, copies
pseudos, drops the wrapper) read `current_dir` at each step.  Without
a lock, the user can re-navigate the sidebar between steps and
silently retarget downstream steps to a different directory.  The
lock blocks navigation visually + functionally for the pipeline's
duration.

### 8.2 Three-layer recovery (independent; do not collapse)

| Layer | Mechanism | Triggers when |
|---|---|---|
| **A** — `try/finally` | Release on success AND on throw | Pipeline completes (normally or by exception) |
| **B** — AbortSignal threaded through every async call | Bounds network duration; cancellable | A hung backend; user-triggered cancel via layer C |
| **C** — Cancel button in banner | Runs registered cancelers | A + B both failed (bug or backend deadlock); user wants out |

The independence matters: a bug in any one layer doesn't strand the
sidebar.  Forgotten `finally` → Cancel still works.  Backend hang →
Cancel triggers abort → fetch rejects → `finally` runs → unlock.
Canceler throws → caught per-canceler; lock state unaffected; user
can still click Cancel again or wait for natural unlock.

**Layer B applies to every async call inside the lock window —
including the workspace-write step.**  No "first step is special".

### 8.3 State machine

```mermaid
stateDiagram-v2
    [*] --> Unlocked

    Unlocked --> Locked : lock(reason, cancelers)
    Locked --> Unlocked : unlock()<br/>(also: pipeline finally{} runs unlock)

    Locked --> Locked : Cancel button<br/>runs cancelers — lock STAYS HELD<br/>(pipeline's own abort + finally release it)
    Locked --> Locked : lock(...) — THROWS<br/>(re-entry forbidden)

    Unlocked --> Unlocked : unlock() — no-op
    Unlocked --> Unlocked : Cancel — no-op
```

### 8.4 Visual contract

While locked, the sidebar fades every direct child except the
banner + header, sets `pointer-events: none` on them, and the banner
stays fully interactive so Cancel is always reachable.  Header keeps
full opacity as a visual anchor; any future header controls are
locked automatically because they inherit the disabled pointer
events.

---

## 9. Visibility model

### 9.1 The trap (and the rule we follow because of it)

The browser's `[hidden] { display: none }` rule and any author
`.foo { display: <non-none> }` rule have the **same specificity**.
On a tie, author CSS wins by cascade order.  An element with
`class="foo"` AND `hidden=""` is rendered VISIBLE despite the
attribute.

**Today's rule**: every author `display:` rule on a class whose
element may carry `hidden` MUST be paired with a `.foo[hidden]
{ display: none }` guard.  Higher specificity wins the tie.

### 9.2 Design direction

The current pattern is fragile — it requires every CSS contributor
to remember the guard rule and every reviewer to check for it.  The
design's end state is:

* One global helper class — `.is-hidden { display: none !important }`.
* HTML `hidden=` is banned in sidebar templates; replaced by
  `class="is-hidden"` toggled via a tiny `setVisible(el, bool)`
  helper.
* A CI grep flags any new use of `hidden=` so the bug class can't
  re-enter.

Until that migration completes, the guard rule (§ 9.1) is the
contract.

---

## 10. Visual states

The conceptual UI states the design recognises.  Every state must
be reachable from a known state mutation; no state may appear
"because of CSS alone".

| State | When | What the user sees |
|---|---|---|
| **Idle** | Page load, cursor unset | Breadcrumb at root; project list; "no file selected" |
| **Browsing** | After navigating | Breadcrumb shows path; entries for that dir |
| **File selected** | After clicking a file | Highlight on entry; "Selected: <name>" status |
| **Empty directory** | Dir has no children | Empty list area with a `.is-empty` modifier |
| **Listing error** | Backend returned `{ok:false}` | Inline error row in the list; cursor reset to the attempted path |
| **Locked** | A pipeline holds the lock | Sidebar contents faded + non-interactive; banner with reason + Cancel |
| **No project root** | Init's `apiRoots` returned empty | List replaced with a "no roots configured" message; lock UI still functional |
| **Preview open** | User clicked preview on a file | Modal over the page; closes on ESC / backdrop / button |
| **Form open** | User expanded a creation section | Form visible with context label; submits navigate the sidebar |
| **Form error** | Backend rejected the form submit | Inline error verbatim; form keeps its current value for retry |

---

## 11. Failure modes

Every failure the design anticipates and how it should be handled.
"Should" — not necessarily "does today".

| Failure | Design response |
|---|---|
| Backend down at init (`/api/files/roots` 5xx or unreachable) | Sidebar shows "no roots configured" + offline indicator.  Lock UI works.  All public-API calls return `{ok:false, error}`.  No exceptions reach tab code. |
| Backend down mid-session | Next `apiX` call returns `{ok:false, error}`.  Sidebar stays usable (last-known listing visible); next refresh shows the error.  No exceptions reach tab code. |
| User clicks Cancel during step 1 of a save | The `AbortSignal` passed to `saveToWorkspace` aborts the fetch; the promise rejects with `AbortError`; the pipeline's `finally` unlocks; sidebar returns to idle. |
| User navigates during a lock | Visually impossible: clicks pass through `pointer-events: none`.  No state mutation occurs. |
| Subscriber callback throws | Caught per-subscriber.  Other subscribers still fire.  Lock + cursor state unchanged. |
| Two tabs in the same project | Each holds its own cursor in sessionStorage.  The design includes a `storage` event listener so navigation in tab A reflects in tab B; current code doesn't (gap M2). |
| Network race on rapid clicks | The sidebar uses per-fetch AbortControllers for navigation too: clicking a second directory aborts the first listing.  Last click wins; partial UI never appears. |
| Cancel clicked when no lock held | No-op.  Documented behaviour.  Cancel button is normally hidden when unlocked, but the API tolerates a stale click. |
| Reentry: `lock()` while already locked | Throws.  Intentional fail-fast — nested locks tangle Cancel semantics.  Pipelines must compose into one outer lock or sequence with unlock between. |
| Subscriber registered after first state change | Initial-fire-on-register means the subscriber gets the current state immediately; the "missed event" trap is structurally impossible. |
| HTTP wrapper got non-JSON response (501 stub era, browser interstitial, etc.) | Wrapper synthesises `{ok:false, error: "<status / message>"}`.  Never throws. |

---

## 12. Backend contract (capability level)

The backend offers seven file-system primitives plus one
project-bootstrap operation.  All operate exclusively inside the
configured `projects/` root.

| Capability | HTTP shape | Notes |
|---|---|---|
| List a directory | `GET /api/files/list` | Optional extension filter |
| Read a file | `GET /api/files/read` | Size-capped; UTF-8 only; binary rejected |
| Stat a path | `GET /api/files/stat` | Single-path metadata |
| Write a file | `POST /api/files/write` | `expected_mtime` for edit-conflict detection |
| Create a directory | `POST /api/files/mkdir` | Depth-aware name validation |
| Upload a file | `POST /api/files/upload` | Multipart; 409 on conflict (no implicit overwrite) |
| Delete a path | `DELETE /api/files/delete` | Canonical-topic dirs protected; recursive flag required for non-empty |
| Bootstrap a project | `POST /api/projects/create` | Atomic; rolls back on partial failure |
| Rename a path | `POST /api/files/rename` | Designed; not yet built (gap M5) |

Every endpoint returns a uniform `{ok: true, …}` or `{ok: false,
error: string, …}` envelope.  HTTP status codes classify
(200 / 4xx / 5xx) but the body shape doesn't change.  Exact field
lists per endpoint live in [`web-api.md`](web-api.md).

---

## 13. Migration plan (current code → design)

The current code implements most of the design but with rough edges.
The migration items, ordered by impact.

### High impact (blocks the design's coverage of failure modes)

| ID | Drift | Migration |
|---|---|---|
| M1 | `apiWrite` / `saveToWorkspace` don't accept `AbortSignal`.  Layer B of the lock recovery doesn't cover the first step of save pipelines (the actual write). | Add `signal` to the `apiX` write surface; thread through `writeFile` / `saveToWorkspace`.  Update Build + Spectra save handlers to pass the lock's signal. |
| M2 | DOM updates in `list.js` (`_markSelected`, `_renderSelectionStatus`) are called inline from event handlers, not from `onChange` subscribers.  Violates Principle 3. | Introduce a single `renderSidebar(state)` subscribed to `onChange`; remove inline calls. |
| M3 | `apiRoots` / `apiList` / `apiStat` / `apiRead` / `apiMkdir` / `apiCreateProject` don't wrap network failures uniformly — they throw on network errors.  Violates Principle 6. | Wrap each in the same try/catch pattern `apiWrite` already uses; synthesise `{ok:false, error}`. |

### Medium impact (architectural completeness)

| ID | Drift | Migration |
|---|---|---|
| M4 | The public API doesn't expose `createProject`, `mkdir`, `upload`, `deleteEntry`, `rename`, `navigateTo`.  These exist as `forms.js` + `list.js` internals only.  Violates Capability C4. | Promote each to `window.molbuilder.projects.*`; keep internal helpers as the implementation. |
| M5 | Backend `rename` endpoint not built.  Capability C4 is incomplete. | Implement `POST /api/files/rename` against the existing path-safety + naming-rule helpers. |
| M6 | No cross-tab `storage` event listener.  Failure mode "two tabs in the same project" is unaddressed. | Add `window.addEventListener("storage", …)` in `state.js` that fires `publishSelectionChange` when cursor keys change. |
| M7 | Visibility relies on the case-by-case `[hidden]` guard pattern.  Brittle (one missed guard = silent bug). | Migrate to the `.is-hidden` class; ban `hidden=` in templates via CI; update the guard rule. |

### Low impact (cosmetic / future-proofing)

| ID | Drift | Migration |
|---|---|---|
| M8 | `setProjectsRoot` has no subscribers.  `onProjectsRootResolved` not on the public API; tabs that need root early use `runtime.whenReady("projects")` instead. | Either add an explicit subscribe API or document that tabs use the runtime registry; pick one and stick to it. |
| M9 | `refreshHandler` is a 1-slot register; multiple consumers would need their own wrapping. | Convert to a `Set` if/when a second consumer arrives. |
| M10 | Preview modal Save button is permanently disabled. | Either implement edit-and-save via the shipped `/api/files/write` endpoint or remove the button. |
| M11 | Internal naming inconsistency: `publishSelectionChange` (no prefix) vs `_publishLockChange` (underscore prefix). | Pick one convention (recommend underscore-prefix for all internal-publish functions) and apply across the module. |

---

## 14. Testing strategy

The design's testing principles:

* **Backend endpoints** — pin envelope shape, status code, path-safety,
  naming-rule outcomes.  Flask test client; no browser.
* **Public-API behaviour** — pin the contract every tab depends on:
  uniform envelopes, fire-once-immediately subscribers, lock state
  machine transitions, AbortSignal propagation through every async
  method.  Playwright + `page.evaluate`.
* **Rendered visibility** — pin `getComputedStyle(el).display`, not
  just `el.classList.contains(...)`.  Catches the § 9.1 specificity
  trap directly.
* **Failure-mode coverage** — for each row in § 11, at least one
  test that triggers the failure and asserts the design response.

When adding a new sidebar feature: the test file already exists for
the layer the feature lives in.  Add tests there in the same commit.
Reviewers reject feature commits without the matching test diff.

---

## 15. Anti-patterns (architectural)

The patterns that consistently produce bugs and have no acceptable
use case in the sidebar.

| Anti-pattern | Why it's banned |
|---|---|
| **UI wiring inside data-dependent init** | Silent "click does nothing" bugs when the data path fails.  Violates Principle 5. |
| **Author `display:` rule without `[hidden]` guard** | Same-specificity tie loses to UA stylesheet.  Element renders visible despite `hidden`.  Multi-site bug class.  Violates Principle 9. |
| **Reading another module's closure state directly** | Breaks module ownership.  Hides the dependency from imports.  Use the exported API. |
| **Tab-specific knowledge in the sidebar** | Sidebar is content-agnostic.  Hardcoded extension-to-tab maps reintroduce the coupling we deliberately removed (see [selection.md § 8](selection.md)). |
| **Per-tab auto-load on `DOMContentLoaded`** | Races user clicks.  Implicit action without consent.  Explicit pull (button) instead. |
| **Reentrant `lock()`** | Tangles Cancel semantics — whose cancelers do we run?  Compose into one outer lock; sequence with unlock between if you must. |
| **DOM updates from event handlers when a publish event could drive a subscriber** | Hand-coordinated DOM updates drift from state.  Violates Principle 3. |
| **Pointing the sidebar at a path outside `projects/`** | Out of scope — files outside come via upload.  Mission boundary. |
| **Triggering a tab's loader from sidebar code** | Sidebar is passive — push violates Principle 2. |
| **Throwing from a public async method** | Tab code never knows to `try/catch`.  Violates Principle 6. |

---

## 16. Change protocol

```mermaid
flowchart TD
    start([sidebar change requested]) --> revisit["1. Does the design itself need to change?"]
    revisit -->|yes| editdoc["2a. Update this doc first."]
    revisit -->|no| editcode["2b. Implement against the existing design."]
    editdoc --> editcode
    editcode --> runtests{"3. Tests pass?"}
    runtests -->|no| editcode
    runtests -->|yes| addtests["4. Add tests for the new design or feature."]
    addtests --> commit["5. Commit doc + code + tests + new tests<br/>in ONE commit."]
    commit --> review{"Reviewer: all four present?"}
    review -->|missing any one| reject([reject])
    review -->|all present| land([merge])

    style commit fill:#dde9ff
    style reject fill:#fdd
    style land fill:#dfd
```

If you discover a code-vs-design drift not already in § 13: add it to
§ 13 in the same commit.  Don't silently align the doc to buggy code,
and don't silently align the code to a stale doc.  Decide which is
right, then update both.
