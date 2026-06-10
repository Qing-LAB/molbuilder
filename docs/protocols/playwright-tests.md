# Playwright test design — patterns and anti-patterns

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.

**Status**: protocol document, persistent reference for all new browser
tests under `tests/test_*_e2e.py` and `tests/test_pages_*.py`.

**Audience**: anyone adding or rewriting a Playwright test in this repo.

**Why this exists**: the 2026-06-01 flake-cluster audit (task #178)
found 24 failing `test_molbuilder_e2e.py` tests across three distinct
failure modes — and almost all of them were caused by NOT applying the
patterns codified here. A future change to the selection panel CSS,
the inspector registry, or the modify HTML must not re-introduce them.
This doc is the playbook that says how Playwright tests in molbuilder
should be written and what they should never do.

---

## 1. Test layering: pick the lowest layer that covers the contract

A `tests/test_*_e2e.py` Playwright test is the **most expensive** kind
of test in the repo: it boots a Flask server, launches Chromium, loads
JS, runs an interaction script, screenshots on failure. One e2e test
costs roughly:

| Layer | Cost per test | What it can verify |
|-------|---------------|--------------------|
| Python unit | ~5 ms | Pure functions, dataclasses, parsers |
| Backend API (Flask test client) | ~50 ms | JSON contracts, request validation, RBAC |
| Node-driven JS unit | ~300 ms | Pure JS helpers, schema shape, contract surface |
| Playwright contract (`page.evaluate`) | ~3 s | JS-side runtime contract against a loaded `/` page |
| Playwright UX flow | ~10-30 s | End-to-end user interaction across DOM + JS + backend |

**Rule**: a test belongs at the LOWEST layer that can verify what it's
trying to verify. A test that asserts "calling `pickResult('foo.out')`
returns the trajectory inspector" does NOT need a UX flow — it needs
a Playwright contract test that opens `/results` and calls
`page.evaluate("() => window.molbuilder.inspectors.pickResult('foo.out').name")`.

### Concrete examples from this repo

- ✅ `tests/test_results_file_picker_js.py` — node-driven JS unit tests
  on pure helpers (`parseDir`, `filterToResultFiles`, `groupResultFiles`).
  Costs ~300 ms total for 27 tests. No browser needed.
- ✅ `tests/test_selection_store_js.py` — node-driven JS unit tests
  on the selection store: API surface, initial state, synchronous
  mutators, subscribe contract.  Uses
  `window.molbuilder.selection._createStore()` (test-only entry
  point) to spin up a FRESH store per test, isolating from the
  module's auto-mounted singleton.  ~3 s for 34 tests.  See § 1.1
  "Module singletons and test isolation".
- ✅ `tests/test_inspector_registry_e2e.py::TestPickerContract` —
  Playwright but contract-only via `page.evaluate(...)`. No UI clicks.
  Verifies `isResult` / `pickResult` / `resultCategory` against the
  LIVE registered inspectors. ~30 s for 16 tests.
- ✅ `tests/test_molbuilder_e2e.py::test_panel_mode_swap_preserves_selection`
  — Playwright UX flow: real clicks on real elements, then asserts on
  observable state. ~5-10 s per test.
- ❌ A Playwright test that just calls `page.evaluate("() => 1+1")` —
  that's a JS unit test in disguise; pay the 3 s of browser startup
  for nothing. Move it to `tests/test_*_js.py` driven by `node`.

### 1.1 Module singletons and test isolation

JS modules that mount a SINGLETON on `window.molbuilder.*` at load
time (e.g. `selection/store.js`, `inspectors/registry.js`) are a
testing trap: every test that mutates the singleton leaks state to
the next.  The canonical fix in this repo is to EXPORT a factory
under a `_create*` prefix that builds a fresh instance:

```js
// At the bottom of the module:
root.molbuilder.selection = root.molbuilder.selection || {};
if (!root.molbuilder.selection.store) {
    root.molbuilder.selection.store = _create();  // production singleton
}
root.molbuilder.selection._createStore = _create;  // test factory
```

Tests then call `_createStore()` per test for isolated state:

```python
out = _run_node(
    "const store = window.molbuilder.selection._createStore();\n"
    "store.setSelection([1, 2, 3]).then(() => "
    "  console.log(JSON.stringify(store.getState().selection))"
    ");"
)
```

Underscore-prefix the factory so it's not part of the production
public surface (the registry's `_clear()` follows the same
convention).

### Routing heuristic

1. Is the thing under test a **pure function or schema**? → Python or
   node JS unit test.
2. Is it a **server contract** (HTTP endpoint, JSON shape)? → pytest
   with `flask.test_client`.
3. Is it a **JS-side runtime contract** (the registry surface, module
   global init order)? → Playwright contract test using
   `page.evaluate(...)` — no UI interaction.
4. Is it a **real user flow** (click X, then Y, expect Z visible)?
   → Playwright UX flow.
5. Is it about **layout / visual regressions**? → Playwright with
   explicit `set_viewport_size(...)` and DOM-shape assertions, not
   pixel diffs (we don't have a visual-diff harness).

---

## 2. Locator strategy: click what the user clicks, not what the DOM says

The 2026-06-01 audit found **at least 3 tests** clicking on
`#selection-mode-filter`, an `<input type="radio">` styled as:

```css
.selection-mode-option input[type="radio"] {
    position: absolute;
    opacity:        0;
    width:          0;
    height:         0;
    pointer-events: none;
}
```

That's the standard "hide the radio, click the label" CSS pattern.
A real user clicks the wrapping `<label class="selection-mode-option">`.

Playwright's `locator("#selection-mode-filter").click(force=True)`
fails with:

```
playwright._impl._errors.Error: Locator.click: Element is outside of the viewport
Call log:
  - waiting for locator("#selection-mode-filter")
    - locator resolved to <input type="radio" ... />
  - attempting click action
    - scrolling into view if needed
    - done scrolling
```

Diagnostic: "scrolled into view but element is still outside the
viewport". A zero-width / zero-height element has no on-screen position
to scroll to. `force=True` skips actionability checks but not the
final viewport-position requirement.

### Right patterns

**Pattern A — set the state via JS + dispatch `change` (RECOMMENDED for hidden inputs)**:

```python
# Don't go through Playwright's click pipeline at all.  Set the

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# checked/value state via page.evaluate + dispatch the change event

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# the JS listener actually cares about.

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("""(sel) => {
    const el = document.querySelector(sel);
    el.checked = true;
    el.dispatchEvent(new Event('change', { bubbles: true }));
}""", "#selection-mode-filter")
```

Notes:
* **`.check(force=True)` is NOT sufficient** for a width=0 / height=0
  input.  Playwright's `check()` and `click()` BOTH require the
  element to have an on-screen rect for the final click action, even
  with `force=True`.  `force=True` skips actionability checks
  (visible / enabled / stable / receives events) but the click
  action itself still resolves to a click coordinate that has to
  land somewhere on screen.  A zero-area element fails with
  `Element is outside of the viewport`.
* Setting `.checked` then dispatching `change` does what a click
  WOULD have done internally — fires the same event the JS handler
  listens for, without needing a hit-testable position.
* This repo's canonical helpers `_set_selection_mode(page, mode)`
  (radio) and `_set_checkbox(page, sel, value)` (checkbox) in
  `tests/test_molbuilder_e2e.py` implement Pattern A.

**Pattern B — click the visible interactive parent**:

```python
# Click the label that wraps the hidden radio.  Use :has() to find the

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# label whose descendant is the radio with the desired value.

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.locator('label.selection-mode-option:has(#selection-mode-filter)').click()
```

This mirrors what a real user does and exercises the full event chain
(label-click → forwarded radio change → JS handler).  Choose this
when you want to verify the click chain itself works (not just the
state mutation).

### When to use which

| Scenario | Use |
|----------|-----|
| You only need the state mutation; the visible-click path is tested elsewhere | Pattern A (set state + dispatch `change`) |
| You want to exercise the actual click event chain | Pattern B (click the label) |
| The element is genuinely visible and ≥ 1 px in both dimensions | Direct `click()` / `check()` |

### Other hidden-element gotchas to watch for

- **File inputs**: `<input type="file">` is often hidden behind a
  styled button. Use `page.locator("button#upload").click()` then
  `page.locator("input[type=file]").set_input_files(path)`.
- **Custom-styled selects**: similar pattern; click the styled
  trigger, then assert on the underlying `<select>` value via
  `select_option(...)`.
- **`display: none`** is different from `opacity: 0`. Playwright will
  refuse `click()` on `display: none` even with `force=True` because
  the element has no layout box at all. For state mutation, use the
  appropriate API method (`check`, `select_option`, `fill`).

---

## 3. Assertions: prefer `expect(locator).to_*` over reading values

The 2026-06-01 audit found tests like:

```python
assert page.locator("#atom-count").inner_text() == "2 atoms"
```

These fail with:

```
playwright._impl._errors.TimeoutError: Locator.inner_text: Timeout 30000ms exceeded.
Call log:
  - waiting for locator("#atom-count")
```

Two problems:

1. **Brittle equality**. The locator's `inner_text()` returns the text
   at the moment of the call. If the assertion runs before the DOM
   update has landed (Plotly draw, Vue render, fetch resolve), the
   string comparison fails on what is otherwise a correct UI.
2. **Bad diagnostic**. If `#atom-count` doesn't exist, the test
   produces a 30 s timeout on EVERY run with no per-call context.

### Right pattern: `expect()` from `playwright.sync_api`

```python
from playwright.sync_api import expect

expect(page.locator("#atom-count")).to_have_text("2 atoms")
```

`expect(...)` is:

- **Retry-aware**. Re-evaluates the assertion every ~100 ms until it
  passes or hits the timeout (default 5 s, not 30 s).
- **Better diagnostic**. Failure shows the latest observed value, not
  just "timeout".
- **Cheap**. Polling is fast; the test still returns in ~100 ms when
  the state arrives quickly.

### When to use plain `.locator()` reads

Only when you need the VALUE (to compare against a derived expected
value the assertion library can't express). Example:

```python
n_frames = int(page.locator("#frame-tot").inner_text())
assert n_frames >= 10, "expected at least 10 frames, got " + str(n_frames)
```

Even here, prefer:

```python
expect(page.locator("#frame-tot")).to_have_text(re.compile(r"^\d+$"))
n_frames = int(page.locator("#frame-tot").inner_text())
```

so the wait happens first.

---

## 4. Waits: prefer state-based waits over time-based waits

`page.wait_for_timeout(N)` is a code smell unless the test is
deliberately measuring a specific time-based effect (debounce, throttle,
animation duration). Every other wait should be condition-based.

### Wait types ordered by preference

1. **`expect(locator).to_*`** — implicit wait via retry. Use when
   asserting state.
2. **`page.wait_for_function("() => ...", timeout=5000)`** — explicit
   wait on arbitrary JS condition. Use when you need to wait BEFORE
   asserting (e.g., for the inspector to finish mounting).
3. **`page.wait_for_selector("#x", state="visible")`** — wait for an
   element to appear / disappear. Cheaper than `wait_for_function` for
   pure DOM presence.
4. **`page.wait_for_load_state("networkidle")`** — wait for all
   network requests to settle. Use sparingly; flaky on pages with
   live polling.
5. **`page.wait_for_timeout(N)`** — wait N ms unconditionally. Last
   resort.

### When `wait_for_timeout` IS legitimate

- Testing that a debounced handler hasn't fired before its delay
  expires.
- Testing that an animation has reached a specific intermediate state.
- A measurement: "did X happen within the configured window?"

For "give the page time to settle" — never `wait_for_timeout`. Use
`wait_for_function` on the actual readiness signal.

---

## 5. Test independence and ordering

Each Playwright test should:

1. Navigate to a known URL (no implicit "the last test left me on
   `/foo`").
2. Bring the page to a known state via fixtures or explicit setup.
3. Clean up its own listeners / global state in teardown (if it
   registered any).

### Shared state to watch out for in molbuilder

- **Inspector registry** (`window.molbuilder.inspectors`) is
  process-wide. Tests that call
  `window.molbuilder.inspectors.register(...)` MUST clean up via
  `_clear()` in a `try/finally`, OR the test runner's page-reload
  between tests resets the global. Check.
- **`projects` sidebar selection** persists in `sessionStorage`. A
  test that calls `projects.setShared(dir, file)` leaks that
  selection to subsequent tests sharing the same page context.
  Reset via `sessionStorage.clear()` in teardown.
- **`pytest_playwright`'s `page` fixture** is per-test by default —
  each test gets a fresh page. But the Flask server fixture
  (`flask_server` in `test_inspector_registry_e2e.py`) is
  module-scoped — its state persists across tests in the same file.

---

## 6. Viewport: explicit when the layout matters

Default Playwright viewport is **1280×720**. Tests that exercise
narrow / mobile layouts MUST set the viewport before navigating:

```python
page.set_viewport_size({"width": 360, "height": 720})
_open_modify(page, flask_server)
```

Setting it AFTER navigation works but may miss the responsive
breakpoint (CSS resize observer doesn't always re-fire reliably).

### Tests that depend on visible position

If a test needs to click an element that lives "below the fold" at
default viewport, **set the viewport tall enough** rather than scroll
manually:

```python
page.set_viewport_size({"width": 1280, "height": 1400})
```

---

## 7. Diagnostics: every failure should point at the root cause

A failing test should make it easy to identify WHY it failed without
running it under a debugger. Bad:

```python
page.locator("#x").click()
assert page.locator("#y").inner_text() == "ok"
```

If the second assert fails, the message is just "AssertionError" with
no context. Good:

```python
expect(page.locator("#y")).to_have_text(
    "ok",
    use_inner_text=True,
)
```

`expect()` reports the OBSERVED value, the expected value, the locator
selector, and the elapsed retry time.

### Diagnostic checklist for new tests

- [ ] Every assertion uses `expect(...)` so failures show the
      observed state, not a raw `AssertionError`.
- [ ] Plain `assert x == y` blocks include the diagnostic string:
      `assert x == y, f"got {x!r}, want {y!r}"`.
- [ ] On a timeout, the failing test name + the locator selector
      tell you which DOM contract changed.

---

## 8. Routinely-painful anti-patterns

This list compiles things that the 2026-06-01 audit ran into AND
patterns observed across other Playwright test suites. Avoid all of
these.

### A1. Clicking hidden form inputs

```python
# BAD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.locator("#selection-mode-filter").click(force=True)
```

```python
# ALSO BAD -- check(force=True) has the same viewport requirement as

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# click(force=True).  force=True bypasses actionability checks (visible,

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# stable, enabled) but the click action's final coordinate still has

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# to land on the element's bounding rect.  A width=0 / height=0

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
# element fails with "Element is outside of the viewport".

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.locator("#selection-mode-filter").check(force=True)
```

```python
# GOOD — set state via JS + dispatch the event the listener cares about

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("""(sel) => {
    const el = document.querySelector(sel);
    el.checked = true;
    el.dispatchEvent(new Event('change', { bubbles: true }));
}""", "#selection-mode-filter")
```

See § 2 "Locator strategy" for the full discussion, and
`tests/test_molbuilder_e2e.py::_set_selection_mode` /
`tests/test_molbuilder_e2e.py::_set_checkbox` for the canonical helpers
this repo uses.

### A2. Reading text without retry

```python
# BAD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
assert page.locator("#atom-count").inner_text() == "2 atoms"
```

```python
# GOOD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
expect(page.locator("#atom-count")).to_have_text("2 atoms")
```

### A3. Sleep instead of wait

```python
# BAD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("() => myAsyncOp()")
page.wait_for_timeout(2000)
expect(page.locator("#result")).to_be_visible()
```

```python
# GOOD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("() => myAsyncOp()")
page.wait_for_function("() => window.myOpComplete === true")
expect(page.locator("#result")).to_be_visible()
```

### A4. Coupling to internal class names

```python
# BAD — `.is-active` is an implementation detail

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
expect(page.locator(".tab-button.is-active")).to_have_text("Build")
```

```python
# GOOD — ARIA attribute is a user-visible contract

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
expect(page.locator('[role="tab"][aria-selected="true"]')).to_have_text("Build")
```

### A5. Tests that don't fail loudly

```python
# BAD — assertion never runs if the locator can't be evaluated

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
try:
    assert page.locator("#x").is_visible()
except Exception:
    pass
```

Never swallow exceptions in tests. If you need to handle "X might or
might not be present", check first:

```python
if page.locator("#x").count() > 0:
    expect(page.locator("#x")).to_be_visible()
```

### A6. Assuming the inspector registry is empty

```python
# BAD — assumes nothing's registered

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("""() => {
    window.molbuilder.inspectors.register({name: 'fake', ...});
    return window.molbuilder.inspectors.list().length;
}""")  # returns 5 (4 real + 1 fake)
```

```python
# GOOD — clear then register, then assert

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("""() => {
    window.molbuilder.inspectors._clear();
    window.molbuilder.inspectors.register({name: 'fake', ...});
    return window.molbuilder.inspectors.list().length;
}""")  # returns 1
```

### A7. Reading text from a node that's about to be replaced

```python
# BAD — race between text update and read

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.click("#refresh")
n_after = int(page.locator("#counter").inner_text())  # racing
```

```python
# GOOD — wait for the new value via expect()

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.click("#refresh")
expect(page.locator("#counter")).not_to_have_text(str(n_before))
```

### A8. Locators that aren't unique

```python
# BAD — page may have multiple `.tab` elements

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.locator(".tab").click()
```

```python
# GOOD — disambiguate by id or attribute

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.locator('button[role="tab"][data-tab="build"]').click()
```

### A9. Long timeouts to "be safe"

```python
# BAD

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
expect(page.locator("#x")).to_be_visible(timeout=60000)
```

A 60-second timeout means a real failure waits 60 s before reporting,
making the test feel hung. Default timeouts (5 s for `expect`, 30 s
for `wait_for_*`) are well-chosen. Override only when the operation
GENUINELY takes longer (e.g., a heavy parse on a large file).

---

## 9. Patterns specific to molbuilder

### 9.1 The `_open_modify` / `_open_results` / `_open_watch` helpers

Every modify/results/watch e2e test should use a shared helper that:

1. Sets up the pageerror + console.error listeners (so a JS crash on
   page load fails the test).
2. Navigates to the URL.
3. Waits for a known "page-ready" signal (e.g., the inspector registry
   reports >= 4 inspectors).
4. Returns the errors list so the test can assert on it.

Existing examples: `_open_results` in
`tests/test_inspector_registry_e2e.py`.

### 9.2 Loading a structure for /modify

Tests that need a loaded structure should use the `water_xyz_file`
fixture and call `_load_water(page, water_xyz_file)`. Don't try to
reach into the /modify Flask routes directly from the test.

### 9.3 Selection-store interaction

The selection store (`lib/selection/store.js`) exposes its state via
`window.molbuilder.selection.store`. Test helpers like
`_set_selection(page, [0, 2])` and `_get_selection(page)` go through
that API rather than poking the DOM checkboxes — DOM is the rendered
view of the store, not the source of truth.

### 9.4 Inspector ready signal

The `/results` inspectors dispatch `molbuilder:inspector:ready` on
`document` after their first paint. Tests waiting for an inspector
to be fully loaded should use:

```python
page.wait_for_function(
    "() => window.__inspectorReady === true",
    timeout=10000,
)
# Plus a one-time listener installed BEFORE the file load:

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.
page.evaluate("""() => {
    document.addEventListener('molbuilder:inspector:ready', () => {
        window.__inspectorReady = true;
    });
}""")
```

### 9.5 Tab-level picker contention — drive via `setShared`, not `reg.mount`

The `/results` tab-level file picker
(`lib/results/file-picker.js`, since commit 6633c4e) subscribes to
the projects-sidebar `onChange` and AUTO-MOUNTS the most recent
result file in the current sidebar directory.

A test that calls `reg.mount(host, file, ctx)` DIRECTLY (the
pattern several watch tests used pre-2026-06-01) races the picker:
if the sidebar auto-resolves to its default root (`projects/`),
which contains real `.out` / `.molwatch.log` files, the picker
will see those, auto-pick the most recent, and replace the
inspector the test just mounted.

The canonical fix is `_load_watch_log` in
`tests/test_molbuilder_e2e.py`:

```python
def _load_watch_log(page, base_url, log_path):
    page.goto(f"{base_url}/results", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4",
        timeout=5000,
    )
    import os
    page.evaluate(
        "(args) => window.molbuilder.projects.setShared("
        "  args.dir, args.file)",
        {"dir": os.path.dirname(log_path), "file": log_path},
    )
```

Driving via `projects.setShared(dir, file)` IS the canonical
sidebar publish.  `viewer.js`'s `onChange` then disposes any
prior inspector and mounts the trajectory inspector for
`log_path` — and the picker, subscribing to the SAME
`onChange`, sees the same selection and doesn't fight it.

Watch fixtures that create their `.molwatch.log` in `tmp_path`
ALSO need `_register_tmp_as_picker_root(tmp_path, monkeypatch)`
(see § 9.7) so the picker scans `tmp_path`, not the projects/
root.

### 9.6 Second-visit + external-change pattern (the #192 bug class)

The 2026-06-02 /results stale-dropdown bug (#192) revealed a
systematic test-coverage gap: every Playwright test up to that
point did ONE `page.goto()` per scenario. The bug only manifested
on a SECOND visit to the same page when external state had
changed between visits — the picker's "lastScannedDir" sentinel
bailed because the dir was the same, while the browser HTTP cache
returned the stale `/api/files/list` response. Neither the
node-driven unit tests, the Flask middleware tests, nor any
single-`page.goto()` Playwright test could have caught it.

This pattern is the canonical regression-test recipe for the bug
class. Use it whenever the tab's UI is driven by a subscriber
that fires on STATE CHANGE rather than STATE RECONFIRM.

**Pattern:**

```python
def test_external_change_refreshes_view_on_revisit(
        self, page, flask_server, fixture_with_one_file):
    proj_dir, dir_str = fixture_with_one_file
    # 1. Set the dir on a non-target tab.
    _setup_dir_via(page, flask_server, "/modify", dir_str)
    # 2. Visit the target tab; assert initial state.
    page.goto(f"{flask_server}/target_tab")
    _wait_for_target_data_loaded(page)
    initial = _read_target_state(page)
    # 3. Navigate away.
    page.goto(f"{flask_server}/modify")
    # 4. Mutate external state (file on disk, sidecar JSON, mtime).
    (proj_dir / "newfile.out").write_text("...")
    time.sleep(0.5)  # mtime resolution + flush
    # 5. Return to the target tab.
    page.goto(f"{flask_server}/target_tab")
    _wait_for_target_data_loaded(page)
    # 6. Assert the refresh landed.
    refreshed = _read_target_state(page)
    assert refreshed != initial, (
        "target tab did not refresh on revisit -- "
        "the subscriber-bails-on-same-key bug pattern is present"
    )
```

**Companion: synthetic `pageshow` dispatch.**

For pinning the underlying handler at the JS-event level (without
needing two `page.goto()` calls), dispatch the event manually:

```python
page.evaluate("""() => {
    window.dispatchEvent(new PageTransitionEvent("pageshow", {
        persisted: true,
    }));
}""")
```

This is faster than the full revisit and is the right size for
unit-style pinning of "the handler is wired". See
`test_results_file_picker_e2e.py` and
`test_inspector_pageshow_refresh_e2e.py` for working examples.

**Tabs already covered (2026-06-02):**

- `/results` file-picker — `test_results_file_picker_e2e.py`
- `/results` trajectory inspector — `test_inspector_pageshow_refresh_e2e.py`

**Tabs still at risk (apply this pattern when adding/refactoring):**

- `/build` form schema, viewer's sidebar-pick load.
- `/modify` selection panel, atom-list refresh after external structure edit.
- `/spectra` form (when /spectra grows an inspect side).

A future "/build re-renders form on schema change" or "/modify re-fetches atoms when the structure file changes on disk" bug would have the same shape as #192 and need the same test pattern.

### 9.7 Registering a tmp_path as a Capabilities picker root

Tests that load a file from `tmp_path` and then drive
`store.setSourceFile(...)` or anything that hits
`/api/selection/atoms?path=...` need `tmp_path` registered as a
file-picker root — otherwise the endpoint returns 403 and the
panel's atoms list never populates, and any
`wait_for getNAtoms()` will time out.

The reusable helper at
`tests/test_molbuilder_e2e.py::_register_tmp_as_picker_root` does this
via `monkeypatch.setattr` so the registration auto-reverses on
test teardown:

```python
def my_test(page, flask_server, tmp_path, monkeypatch):
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "diag.xyz"
    p.write_text("...")
    _load_file(page, str(p), expected_atoms=N)
```

Existing fixtures in this repo that bundle this registration:
`water_xyz_file`, `ss_pair_xyz_file`, `watch_log_file`,
`watch_log_file_finished`, `watch_log_file_with_forces`.

---

## 10. Test review checklist

When adding or modifying a Playwright test, run through:

- [ ] Is this test at the right layer? Could it be a JS unit or
      backend test?
- [ ] Does every `click()` target a visible interactive element?
- [ ] Does every text-read assertion use `expect(...)`?
- [ ] Is every wait either condition-based or accompanied by a
      comment explaining why time-based is correct?
- [ ] Is the test independent of order (no implicit state from a
      previous test)?
- [ ] If layout-dependent, is the viewport set explicitly?
- [ ] Do failures produce a useful diagnostic?
- [ ] Are inspector / registry / sessionStorage mutations cleaned up?
- [ ] No catch-and-ignore exceptions?
- [ ] No 60+ second timeouts unless the operation genuinely takes
      that long?

---

## 11. Historical reference: the 2026-06-01 cluster

The 24 failures in `test_molbuilder_e2e.py` audited on 2026-06-01 broke
down by root cause:

| Cause | # tests | Pattern violated | Fix |
|-------|---------|------------------|-----|
| Clicking hidden radios `#selection-mode-*` (`opacity:0; w:0; h:0`) | 3 | A1 | `_set_selection_mode` helper dispatches `change` event directly |
| Custom fixtures missing picker-root registration → /api/selection/atoms returns 403 → atoms list never populates → wait_for getNAtoms times out | 3 (3 diag fixtures) + 2 (send_to_build via ss_pair_xyz_file) + 5 (electrode tests) | 5 (test independence) | Extract `_register_tmp_as_picker_root(tmp_path, monkeypatch)`; call from each fixture / inline test that loads a custom XYZ |
| Watch-tab tests bypass the canonical sidebar publish; /results file-picker auto-mounts a different file from the sidebar's default projects-root scan, replacing the inspector the test just mounted | 6 watch_inspect tests | 5 (test independence) + 9.4 (inspector ready signal) | `_load_watch_log` drives `projects.setShared(dir, file)` instead of calling `reg.mount` directly; watch_log_file fixtures call `_register_tmp_as_picker_root` so the picker scans the test's tmp_path (not projects/ root) |
| UI text change (`Ongoing` → `Running` on run-state badge) | 2 watch_run_state tests | (not a flake; UI spec change) | Update expected text + add note explaining the server-side `run_state` is still `"ongoing"` |
| Asserting on removed DOM id (`#atom-count`) | 1 | A2 (stale id) | Drop the redundant assertion (the canonical check via `getNAtoms()` already covered it) |
| Asserting on viewer-side data that doesn't reach the viewer (server-side `residue_name='MOD'` drops in the XYZ round-trip) | 1 (apply_add_atom) | A4 (test on user-visible state) | Drop the residue assertion; explain in a docstring why it's not observable |
| Section-order list out of date (`Parallel execution` moved last) | 1 | (not a flake; intentional UX change) | Update expected list to match `SiestaConfig._form_section_order` |
| Custom-styled checkbox (form-schema's flex layout collapses native input to w=0) — same shape as hidden-radio issue | 1 | A1 | Generic `_set_checkbox` helper; sets `checked` + dispatches `change` |
| Racing the debounced preflight: the test's `wait_for_function` checks for "any issue", which fires on the stale issue from a prior call before the new mesh_cutoff one arrives | 1 | A7 (reading text that's about to be replaced) | Strengthen the wait to look for the SPECIFIC expected warning text |
| **Real CSS responsive bug**: sidebar doesn't collapse at viewport width 360 px, body overflows | 1 (modify_layout_phone_width) | (not a flake) | Marked `xfail`; tracked separately for a sidebar narrow-viewport fix |

Of 24 failures, 23 were genuine test-quality issues addressed via the
patterns in this doc; 1 (the layout regression) is a real CSS bug
correctly captured by an xfail-marked test. The wave landed alongside
this document so the patterns and the exemplars match.

The work also extracted two reusable test-suite helpers:

* `_register_tmp_as_picker_root(tmp_path, monkeypatch)` — register a
  test's tmp_path as a Capabilities picker root so the selection
  panel + file picker can scan it.
* `_set_selection_mode(page, mode)` and `_set_checkbox(page, sel, v)`
  — set hidden form inputs by dispatching `change` directly.

Future tests that need either capability should call these helpers
rather than re-implementing the dispatch.
