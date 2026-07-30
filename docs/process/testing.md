# Testing — the strategy, the layers, and the browser tests

**Role:** reference
**Domain:** process
**Companions:** [`conventions.md`](?doc=process/conventions.md) — the guard tests
that enforce the conventions; [`package-layout.md`](?doc=process/package-layout.md)
— where `tests/` sits; [`ops/installation.md`](?doc=ops/installation.md) — the
conda envs the backend tests run against.

The suite is ~275 test files. Most are fast Python unit/module tests you'll write
constantly; a smaller set drives a real browser. This doc is the map: the pyramid,
where tests go, how the front-end JS is tested without a browser, and the handful
of Playwright patterns that keep the e2e tests from being flaky.

## 1. The pyramid — pick the lowest layer that covers the contract

Tests are marked by **layer**, and the marker is orthogonal to the directory (a
`tests/spectra/` file can hold unit *and* integration tests). The markers, from
`pyproject.toml`:

| Marker | Layer | What it tests |
|---|---|---|
| `unit` | L1 | a pure helper — no I/O, no globals (microsecond cost) |
| `module` | L2 | one submodule's public surface, end to end |
| `interface` | L3 | the contract between two modules (a registry, a severity map, a shape) |
| `integration` | L4 | several subsystems agreeing on one shared fact |
| `smoke` | — | subprocess-runs a *generated script* (slow; needs pyscf) |
| `e2e` | — | browser-driven Playwright (slow; needs chromium) |
| `slow` | — | > 1 s (full runs include these; pre-commit skips them) |
| `capture_on_fail` | — | dump browser state + console to `test-artifacts/` on failure |

The rule of thumb: **cover a contract at the lowest layer that can see it.** A
severity-map bug is an `interface` test, not an e2e click-through. e2e is for the
things only a real browser exposes.

Config worth knowing (`pyproject.toml`): `testpaths=["tests"]`,
**`pythonpath=["."]`** (so the in-tree package imports without `pip install -e`),
`addopts="-ra"`.

## 2. Where tests go, and the one structural invariant

`tests/` is **flat at the top** with a few topic subdirs (`tests/parse/`,
`tests/spectra/`, `tests/validation/`, `tests/watch/`); fixtures live in
`tests/data/`. Naming is `test_*.py`.

The one *structural* guarantee is **layering** — enforced, not just documented, by
`tests/test_layering.py`. It AST-walks every `molbuilder/*.py`, classifies each
module into a layer, and asserts imports only point down:

- **L1** (core types — `structure.py`, `chemistry.py`, …) imports nothing higher;
- **L2** (domain verbs — builders, engines, parse, …) may import L1, not L3;
- **L3** (the two *surfaces*, `cli.py` and `web/`) may import anything.

It also asserts *every* top-level name is classified, so a new module can't slip
past the boundary silently. This is what lets `cli` and `web` share one API
without circular imports — see the thin-shell note in
[`conventions.md § 3`](?doc=process/conventions.md).

## 3. Design tests *around* the envs, never the reverse

molbuilder dispatches into per-backend conda envs
([`installation.md`](?doc=ops/installation.md)), and the test rule follows from
that: **a test must not require an env to be reshaped to pass.** The whole suite
runs in the host `molbuilder` env; tests that genuinely need a backend
(`smoke` needs pyscf, `e2e` needs chromium) are **marked and gated** so they skip
cleanly when that backend is absent, rather than failing. You design the test to
fit the environment model — you never edit an env to make a test pass (that rule is
load-bearing elsewhere too).

## 4. Testing the front-end JS without a browser

Most front-end logic is tested **in Node, no browser** — the `*_js.py` tests
(~49 of them) drive `tests/_node_esm.py`. Its `run_node(files, snippet)` loads an
ordered list of module files via dynamic `import()` and then runs a JS snippet
against them.

The clever part is that it spans the ESM migration: a classic IIFE file publishes
its `window.molbuilder.*` global as a side effect, and a converted ES module
publishes the *same* global **and** exposes exports — a dynamic `import()` runs
either kind. So a test that reads through the **global** (`window.molbuilder.X`)
passes *before and after* its module converts to ESM — no per-module test churn at
conversion time. (This is why the ESM tasks #103–#107 don't drag a test rewrite
behind them.)

Reach for a JS unit test for anything you can express as "load these modules, call
this, assert the result." Save the browser for what needs the DOM + 3Dmol.

## 5. The browser tests (Playwright / e2e)

The ~19 `*_e2e.py` tests use **Playwright** against a real headless Chromium. Each
spins up a **live Flask server** in a fixture (`flask_server`) built from
`create_app`, and drives the `page` fixture (`page.goto(f"{base}/molbuilder")`, …).
The default `web_client` fixture builds the app with **rate-limiting disabled**
(`create_app(config={"rate_limit": {"enabled": False}})`) so the limiter never
trips the test client; the rate-limit tests build their own enabled client.

These are the durable patterns — follow them and the e2e tests stay stable:

- **Locate what the *user* clicks, not what the DOM says.** Target the visible
  label/button, not a hidden input by id. Where a control is backed by an invisible
  element (a hidden radio, a 3Dmol atom), don't fight the click pipeline — set state
  via `page.evaluate(...)` and dispatch the event the JS listener actually cares
  about. (3Dmol atoms live inside a canvas — there's nothing to click; you go
  through `page.evaluate`.)
- **Wait on *state*, not time.** Prefer `page.wait_for_function(...)` /
  `expect(locator).to_*` over `sleep`. A time-based wait is either flaky or slow;
  a state-based wait is neither.
- **Assert with `expect(locator).to_*`**, which auto-retries, rather than reading a
  value once and asserting on the snapshot.
- **Set the viewport explicitly when layout matters.** A `force=True` click bypasses
  the actionability checks but the final coordinate still must land on the element's
  box — a zero-size or off-screen element fails "outside of the viewport." If layout
  is under test, pin the viewport.
- **Every failure should point at the root cause.** Wire `page.on("pageerror", …)`
  and `page.on("console", …)` so a JS error or console message surfaces in the test
  output; use `capture_on_fail` to dump browser state + console to
  `test-artifacts/` for the intermittents.

## 6. A few named test patterns

- **Source-text invariant tests** — assert a property over the *source*, not a
  runtime. E.g. `test_no_inline_scripts.py` scans every served template for an inline
  `<script>` (the CSP `script-src 'self'` rule would break at runtime otherwise), and
  `test_negative_body_assert_lint.py` AST-lints the *test suite* so a "body lacks X"
  assertion is always paired with a status check. Cheap, and they catch a class of
  bug no unit test would.
- **State-composition tests** — the molview class of bug: a value is correct in
  isolation but wrong once composed with a sibling piece of state. These get an
  explicit test that exercises the *combination*, not each part alone.

## 6.1 How to actually run the suite — `tools/testrun.py`

Use the project's runner. It exists because a bare `pytest` gives you nothing
until it exits ~25 minutes later, and piping it to a file or through `tail` is
worse than nothing: the output buffers until the process ends, and a `tail`
silently truncates the failure list, so you draw conclusions from a fraction of
the failures.

```bash
python tools/testrun.py run none2e   # every non-e2e test (run it in the background)
python tools/testrun.py run e2e      # the Playwright batch
python tools/testrun.py run lf       # rerun ONLY the last run's failures
python tools/testrun.py status              # live summary of every batch
python tools/testrun.py status --fails      # every failed id + its reason
python tools/testrun.py failed none2e       # bare node-ids, to feed back to pytest
```

`tools/progress_plugin.py` streams **each test outcome** to
`.test-progress/<batch>.jsonl`, flushed per test, so `status` is accurate *while
the run is in flight* and readable from any other shell or session — the path is
stable and git-ignored, never a job-specific temp file. The two batches are
single-process, so `none2e` and `e2e` can run concurrently on a multi-core box
without contention.

The working loop this enables: launch a batch, read `status --fails`, fix, then
`run lf` to verify just those failures instead of paying for a whole re-run, and
finally one clean full batch before committing.

> **Why this is documented so emphatically.** On 2026-07-29 a session ran three
> full sweeps as `pytest … | tail -14`, saw 7 of 31 failures, inferred the rest
> from whichever files it was already touching, and pushed two regressions as a
> result — a half-migrated per-slot map in `warning-modal.js`, and a new embed
> handle method missing from its documented-surface list. Both were sitting in
> the 24 failures the `tail` had cut off. `status --fails` shows all of them in
> one line each.

Two habits that go with it: never conclude "the suite is green" from a truncated
view, and when an unfamiliar test fails, check it against `HEAD` in a throwaway
`git worktree` before assuming it is someone else's problem — that is how you
tell a regression you just pushed from breakage that was already there, and it
takes a few seconds.

## 7. What gates a commit

There is **no CI** — enforcement is the **pre-commit hook**
([`conventions.md § 1`](?doc=process/conventions.md)): `pytest -m "not slow"` (which
deliberately keeps `e2e` in), pyflakes, and a `node -c` syntax check on changed
`*.js`. So the suite you run locally *is* the gate.

A ready-to-use GitHub Actions workflow sits beside this doc at
`process/github-workflows-test.yml`. It is a **template, not a live workflow** —
copy it to `.github/workflows/test.yml` yourself if you want CI; it is
deliberately not installed there.

## 8. Test map (the meta-tests)

- `test_layering.py` — the import-direction + full-classification invariant (§2).
- `_node_esm.py` — the Node ESM load-sim harness the `*_js.py` tests use (§4).
- `test_no_inline_scripts.py`, `test_negative_body_assert_lint.py` — the
  source-text invariant lints (§6).
- `conftest.py` — the `web_client` fixture (rate-limit-off) + the shared fixtures;
  each e2e file carries its own `flask_server`.
