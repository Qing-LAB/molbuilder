# Spec — `/results` tab (post-merge unified inspector)

<!-- ROUTE-RENAME-BANNER -->
> **Route names updated 2026-06-07.** Occurrences of `/build`, `/modify`,
> `/spectra` in this doc refer to PAGE routes that have been renamed to
> `/structure-optimization`, `/molbuilder`, `/spectrum-calculation`
> respectively.  `/api/build/*`, `/api/modify/*`, `/api/spectra/*`
> BACKEND prefixes are unchanged — only the page routes moved.  See
> [`tabs/architecture.md`](../tabs/architecture.md) § 3 for the canonical
> route table.

**Status**: canonical spec for task #58 (signed off 2026-05-16).
See § 2 for the locked-in decisions.

**Module(s) the spec affects** (after merge):

  * `molbuilder/web/templates/results.html`           — new
  * `molbuilder/web/static/results/viewer.js`         — new (or split into modules)
  * `molbuilder/web/templates/spectra.html`           — shrinks (Inspect panel moves out)
  * `molbuilder/web/templates/watch.html`             — becomes legacy-only
  * `molbuilder/web/blueprints/{watch,spectra}.py`    — endpoints reused, no new HTTP routes
  * `tests/test_pages_no_js_errors.py`                — add `/results` to the route list

---

## 1. Why this tab exists

Today the user has two tabs for "look at what a finished computation
produced":

  * **Watch** — live + post-mortem inspection of a `.molwatch.log`
    trajectory (frames, energies, SCF history, max-force plot).
  * **Spectra** — generates AND inspects `.spectra.json` (config form
    + script generator + results viewer + 3D eigenmode animation +
    issues panel + CSV export).

The user-facing workflow ("I have a finished run; open it") doesn't
care which kind of run produced it.  The two tabs do file-type
dispatch the user has to do mentally ("this is a trajectory; click
Watch.  this is a spectrum; click Spectra.").  The merge makes the
sidebar's existing file-selection the dispatch mechanism: the user
clicks a file in the projects sidebar, `/results` shows the right
inspector for that file type.

A side effect of this merge is that **Spectra becomes generate-only**:
form + script-generator + Save-to-workspace, nothing else.  Inspection
moves to `/results`.

---

## 2. Locked-in decisions

### 2.1 Route topology

**Decision: new top-level `/results`; keep `/watch` working but
remove from the primary tab nav.**

Considered alternatives (rejected):
* `/results` absorbs `/watch` via 301 — would break existing
  bookmarks and external links to `/watch` during the cutover.
* Three-way Spectra subtabs (`Generate` / `Trajectory` / `Spectra
  results`) — forces every trajectory-inspection workflow through
  the Spectra tab, which is the wrong mental model for non-spectra
  trajectories.

Future release (not this task) will 301 `/watch` → `/results` and
delete the legacy templates.

### 2.2 File-type dispatch

`/results` opens whatever file the sidebar has selected.  Dispatch
table:

| Selected file | Inspector | Notes |
|---|---|---|
| `*.molwatch.log` | trajectory view | the current Watch UI, moved into the results tab |
| `*.spectra.json` | spectra results view | the current Spectra Inspect panel, moved into the results tab |
| `*.fdf.out`, `*.log` (SIESTA) | (later) | scope creep; defer |
| `*.xyz` | "structure preview" with `Open in Modify` button | thin; uses existing 3Dmol viewer |
| `*.pdb` | same as `*.xyz` | |
| `*.fdf`, `*.py` (inputs) | source listing + "Open in Build" CTA | one-screen; no editing |
| anything else | "No inspector for this file type" with `Preview` (text view) fallback | parallels the existing sidebar preview |

**Out of scope for v1**: comparing two runs side-by-side, multi-file
selection, anything that needs simultaneously-loaded trajectories.

### 2.3 What Spectra keeps vs sheds

After the merge, Spectra is **generate-only**:

| Spectra surface today | After merge |
|---|---|
| Parameter form (`spectra-form-container`) | KEEP |
| Methods modal | KEEP |
| "Generate script" button + script preview / download / copy | KEEP |
| "Save to workspace" | KEEP |
| Issues panel | KEEP (script-side issues only — runtime issues move to /results) |
| Live polling for `.spectra.json` updates | MOVE to /results |
| Mode-eigenvector 3D animation viewer | MOVE to /results |
| Per-mode CSV export | MOVE to /results |
| Equilibrium-geometry XYZ download | MOVE to /results |

Spectra's left-side workspace selection ("Load from current
selection") stops being a load-into-this-tab operation; it becomes a
"jump to /results with this file" shortcut (or just disappears,
since the sidebar selection is already global state).

### 2.4 Handoff direction

| From | To today | After merge |
|---|---|---|
| Build "Send to Watch" | `/watch?path=...` | **dropped** (no equivalent — user opens /results via sidebar) |
| Build "Send to Modify" | `/modify` (sessionStorage) | KEEP unchanged |
| Modify "Send to Build" | `/` (sessionStorage) | KEEP unchanged |
| Spectra "Load from current selection" → Inspect | within-tab | becomes "open in /results" (since Spectra no longer inspects) |

Net effect: no tab has an explicit "Send to /results" button.  The
projects sidebar IS the path into /results; clicking a file there
in /results context loads it.

### 2.5 `/watch` legacy lifetime

Leave the `/watch` route + templates + JS alone for one release
cycle.  Remove from the primary tab nav at merge time.  In the
next release, redirect `/watch` to `/results` and delete the
files.  Legacy timeline preserved in
[`docs/archive/2026-06-02-tabs-watch.md`](../archive/2026-06-02-tabs-watch.md).

### 2.6 Sidebar integration

`/results` uses the same `_projects_sidebar.html` partial every
other tab does.  The sidebar already publishes
`molbuilder.current_file`; `/results` reacts to changes (via the
existing `projects.onChange` subscriber API) by re-rendering the
inspector for the new file.

---

## 3. Structural plan (assumes the recommendations above)

### 3.1 New files

Implementation pivoted from "four pre-rendered panels with show/hide"
to a SINGLE `#inspector-host` element whose contents are exclusively
owned by the active inspector — see the Inspector Registry
foundation (task #78).  The registry's `pick(file)` returns the
matching inspector module; the dispatch in `results/viewer.js` calls
`previous.dispose()` then `next.mount(host, file, ctx)` on every
sidebar selection change.  Each inspector creates its own DOM inside
the host; no template-side panel ids are required.

* `molbuilder/web/templates/results.html` — page shell + sidebar
  include + one `<section id="inspector-host">` mount point.
* `molbuilder/web/static/lib/inspectors/registry.js` — the
  Inspector Registry (match → pick → mount → dispose contract).
* `molbuilder/web/static/lib/inspectors/{source,structure,trajectory,spectra}.js`
  — one module per file-type, each implementing the registry's
  Inspector interface.  Source and structure are pure JS (build
  DOM via createElement).  Trajectory delegates to a shared core
  module (`lib/trajectory/core.js`), see § 3.2.
* `molbuilder/web/static/results/viewer.js` — dispatch + glue.
  Loads on `DOMContentLoaded`; subscribes to
  `window.molbuilder.projects.onChange`; calls
  `registry.pick(file)` and `mount`/`dispose` accordingly.
* `molbuilder/web/static/results/style.css`
* `molbuilder/web/blueprints/results.py` — `GET /results` + `GET
  /partials/trajectory-inspector` (server-rendered partial HTML
  consumed by the trajectory inspector wrapper).  All other data
  comes via existing `/api/watch/*` + `/api/spectra/*` +
  `/api/files/*` endpoints.

### 3.2 Moved JS — three-layer modularization

Trajectory inspector (step 1, done):
  * `static/lib/trajectory/core.js`  — THE shared trajectory-
    inspector implementation.  Exports
    `window.molbuilder.trajectoryInspector.mount(host, opts)`.
    Self-contained (no auto-bootstrap on page load); safe to
    include on any page.  Used by both consumers below.
  * `static/watch/viewer.js`         — `/watch` page bootstrap
    only.  Calls `trajectoryInspector.mount(document)` on
    `DOMContentLoaded` so the inspector mounts against the page's
    loader-bar elements.
  * `static/lib/inspectors/trajectory.js` — registry adapter.
    Fetches the inspector partial from
    `GET /partials/trajectory-inspector`, assigns to the host's
    innerHTML (trusted server render), calls
    `trajectoryInspector.mount(host, {file})`.

Spectra inspector (step 2, DONE):
  * Same three-layer pattern as trajectory.  The shared
    `static/lib/spectra/core.js` carries the entire inspector body
    (form, modes table, Plotly chart, 3Dmol mode-animation viewer,
    live-watch poller) wrapped in `mountInspector(rootEl, opts)`.
    `static/spectra/viewer.js` is a 47-line bootstrap that calls
    `window.molbuilder.spectraInspector.mount(document)` on
    DOMContentLoaded; `static/lib/inspectors/spectra.js` is the
    registry adapter that fetches `_spectra_inspector.html` from
    `GET /partials/spectra-inspector` and mounts the core into the
    host element.
  * `init()` inside the core gates on
    `hasGenerateSide = Boolean(els.formContainer)` and
    `hasInspectSide = Boolean(els.watchPath)` so the same module
    mounts cleanly on either consumer: /spectra exposes the
    generate-side form (form rendering, render-script flow, methods
    preview) and the inspect side is absent; /results' inspector
    host has the inspect side only.

The shared-core pattern (rather than a thin wrapper that re-exports
the legacy module) means:

* Dependency direction reads correctly: inspectors → trajectory/,
  not inspectors → watch/.
* A bug fix in `lib/trajectory/core.js` benefits both consumers
  immediately; no fork.
* The `/watch` page bootstrap can't accidentally regress the
  inspector — it lives in its own file with a tiny surface.

### 3.3 Endpoints unchanged

No new HTTP routes.  `/results` reuses:

  * `/api/watch/load` + `/api/watch/data` (trajectory loading)
  * `/api/spectra/*` (script generation lives here too — the
    Spectra-tab form posts to the same endpoints)
  * `/api/files/*` (file reads for source / structure previews)

### 3.4 Template + nav

  * `_app_header.html` (or wherever the tab nav lives) — add
    `Results` link; remove `Watch` from the primary nav (but keep
    it routable).
  * `_projects_sidebar.html` — no changes; the sidebar is the
    dispatch mechanism.

### 3.5 Tests

  * `tests/test_pages_no_js_errors.py` — add `/results` to the
    parametrized route list; add a `results-form-container`-style
    ready selector.
  * `tests/test_results_dispatch.py` (new) — for each file
    extension in the dispatch table, assert the right inspector
    container becomes visible (Playwright E2E, gated on the same
    importorskip as the existing E2E tests).
  * Existing Spectra blueprint + Watch blueprint tests UNCHANGED —
    the endpoints don't move; only the UI does.

---

## 4. Migration order

1. **Extract trajectory inspector** (DONE) — `lib/trajectory/core.js`
   shared module; `static/watch/viewer.js` shrunk to /watch
   bootstrap only; `lib/inspectors/trajectory.js` mounts via the
   shared core after fetching `_trajectory_inspector.html` from
   the new `/partials/trajectory-inspector` endpoint.
2. **Extract spectra inspector** (DONE 2026-05-18; sub-stages
   2.1..2.6 mirror the trajectory lift):
   * 2.1 (DONE) — extract inspect-side DOM into
     `_spectra_inspector.html`; spectra.html includes it; add
     `GET /partials/spectra-inspector` endpoint with the same
     wire contract as the trajectory partial.
   * 2.2 (DONE) — extract inspect-side JS from
     `static/spectra/viewer.js` into `static/lib/spectra/core.js`,
     wrapped in `mountInspector(rootEl, opts) -> handle`.  Returns
     `{dispose()}` so the registry can tear down timers + Plotly
     listeners between inspector swaps.
   * 2.3 (DONE) — load `lib/spectra/core.js` on /results (script
     tag in results.html, before the inspector adapter).
   * 2.4 (DONE) — `lib/inspectors/spectra.js` rewritten as the real
     adapter (fetches the partial, mounts the core, chains dispose).
   * 2.5 (DONE) — `{% include %}` dropped from `spectra.html`;
     /spectra is now generate-only.  `static/spectra/page.js`
     deleted (its load-from-selection + workspace-indicator wiring
     concerned ids that are no longer on /spectra); `viewer.js`
     reduced to a thin bootstrap.
   * 2.6 (DONE) — design.md changelog entry; this file updated;
     tests/spectra/test_blueprint.py id-pins repointed at the
     partial endpoint + the core module (44 tests pass).
3. Add `results.py` blueprint + `results.html` template + `results/`
   static dir (DONE).
4. Wire dispatch: subscribe to `projects.onChange`, route file
   extension to inspector, mount the inspector module (DONE for
   trajectory; pending for spectra).
5. Remove inspect-side UI from Spectra tab (form-only).
6. Remove `Watch` from primary nav; keep route working.
7. Add E2E dispatch tests + extend `tests/test_pages_no_js_errors.py`.
8. Update `docs/tabs/`: add `results.md`, mark `watch.md` as
   legacy, mark `spectra/spec.md` as form-only.

Each step is independently shippable — if step 4 breaks, the
inspectors still work via legacy `/watch` and `/spectra`.

---

## 4.5 Trajectory inspector plots (2026-06-12 polish)

The trajectory inspector renders four plots in a `.plots-row`
grid (`auto-fit, minmax(280px, 1fr)`): energy, max force, SCF
energy, SCF gnorm.  Two adjustments landed on 2026-06-12:

### Dual-trace max-force plot

When the loaded run has frozen atoms — meaning either the
SIESTA parser captured `Max <val> constrained` lines or the
PySCF parser found `frozen_atoms` in the sidecar to mask the
qdata gradient — the force plot renders **two** traces:

| Trace | Series | Style | Meaning |
|---|---|---|---|
| `all atoms` | `data.max_forces` | red, dashed, thin | Max \|F\| including frozen atoms.  Informational. |
| `free atoms` | `data.max_forces_constrained` | green, solid, thick | Max \|F\| excluding frozen atoms.  Engine's actual convergence-gating signal. |

When `max_forces_constrained` is `[]` (runs without frozen
atoms; the JSON layer collapses an all-`None` list to `[]`), the
plot falls back to the single solid-red `Max |F|` trace it
rendered before this change.

The legend is horizontal, centered, anchored BELOW the plot area
so it doesn't compete with the y-axis label for lateral space.
Single-trace renders skip the legend entirely.

### Export-all-plot-data CSV button

A `↓ Export all plot data (CSV)` button in a small
`.plots-toolbar` above the plots row generates a self-describing
CSV bundling every column drawn across the four plots.  Header
block (lines starting with `#`):

* `# generated:` — ISO-8601 generation timestamp (browser local)
* `# source path:` — server-resolved absolute path of the file
  loaded (the value `applyNewData` stores on `state.path` from
  the `/api/watch/load` response)
* `# parser:` — `state.format` (`siesta`, `pyscf`, `molwatch`, …)
* `# label:` — display label, often the same as `parser`
* `# source mtime:` — ISO-8601 of the source file's mtime
* `# n_frames:` — frame count
* `# Column legend:` — per-column unit + meaning + a one-line
  schema reminder for whoever opens the file later

Data columns (one row per frame):

```
step, energy_eV,
max_force_eVperA, max_force_constrained_eVperA,
scf_cycle, scf_cycle_energy_eV, scf_cycle_gnorm_eVperA
```

Empty cells where the engine didn't emit the value (matches
Plotly's `connectgaps: false` skipped-marker visuals).  Numbers
use `Number.toString()` to preserve IEEE precision (lossless
round-trip).  Filename is `<sanitised-label>_plots.csv`.

---

## 4.6 Convergence-targets summary + threshold lines (2026-06-13)

The trajectory inspector renders a small text band BETWEEN the
run-state badge and the plots row that names the convergence
targets the run was configured to chase and tells the user how far
the current step is from them.  The matching plots gain a green
dashed horizontal threshold line at the target value.

**Data shape.**  `Trajectory.runtime_info["convergence_targets"]`
(a nested dict, populated by the parsers):

| Key | Units | Read by which parser |
|---|---|---|
| `max_force_tol_eV_per_A` | eV/Å | SIESTA (`redata: Force tolerance`); molwatch (`# convergence.*`); PySCF emitter passes each enabled stage's `gmax` after Ha/Bohr→eV/Å conversion |
| `dm_tolerance` | dimensionless | SIESTA (`redata: DM tolerance for SCF`); molwatch |
| `scf_energy_tol` | Hartree | molwatch (PySCF runs, from each enabled stage's `conv_tol`); not in SIESTA echo |
| `scf_grad_tol` | eV/Å | molwatch (geomeTRIC gmax-style); not in SIESTA |
| `max_scf_iter` | integer | SIESTA (`redata: Max. number of SCF Iter`); molwatch |
| `max_geom_iter` | integer | molwatch (PySCF: each enabled stage's `max_steps`) |
| `max_displ_ang` | Å | SIESTA (`redata: Max atomic displ per move`); molwatch |
| `source` | string | One of `"siesta_input_echo"`, `"molwatch_header"`, `"geomeTRIC_log"` — drives the italic provenance label in the UI |

Each parser populates what its source actually carries; missing
keys are tolerated by the inspector (it renders only the rows
present).  When the `convergence_targets` subdict is entirely
absent (older runs, runs from non-molbuilder scripts), the band
falls back to a short hint pointing at how to get the lines next
time: "load the source .fdf / .py file next to the run, or rerun
via molbuilder for self-describing output."

**Plot recolor (load-bearing).**  Pre-2026-06-13 the "free atoms"
convergence-gating force trace was green (`#1f9d55`).  Adding the
green threshold line on the same plot meant trace + target would
sit indistinguishable.  The recolor moves:

| Element | Before | After | Source |
|---|---|---|---|
| Free-atoms force trace | green `#1f9d55` | blue (`--accent` via `_themeColors()`) | trajectory/core.js |
| All-atoms force trace | red `#d62728` | unchanged | trajectory/core.js |
| SCF gnorm residual trace | amber/green | orange `#fb923c` | trajectory/core.js |
| Threshold line (force + SCF) | — | green `--success` dashed | trajectory/core.js |
| Stage-marker dashed verticals | `#888` | `--text-muted` via theme helper | trajectory/core.js |

`_themeColors()` reads CSS custom properties via
`getComputedStyle(document.documentElement)` once per `makePlots()`
call so a future theme retune in `lib/tokens.css` repaints traces
+ threshold lines together.  Non-themable plot-convention colours
(red for "all atoms" informational; orange for SCF gnorm) stay
literal — pinning them to tokens would couple "scientific plotting
convention" to "site theme" in a way that doesn't make sense.

**Auto log y-axis.**  When `max(force_history) / max_force_tol_eV_per_A > 50`
the force plot switches to a log y-axis so the threshold line +
the trace are both legible at the same time.  Below that ratio
the linear/`rangemode: "tozero"` view shows the target line at
the bottom of the plot, where the eye naturally lands on "the
floor we're approaching."

**Pinned by:** `tests/test_live_poll_invariants_audit.py::TestConvergenceTargetsAndPlotColors`
(parser extracts targets, `_themeColors()` helper present, no
hardcoded `#1f9d55` outside comments).

---

## 4.7 SCF wall-time annotation (2026-06-15)

Trajectory inspector's `#scf-status` line carries a per-iter
wall-time annotation so the researcher can answer "is this run
progressing at a reasonable pace?" without having to compare timer
lines by hand.  Three-tier precedence ladder, in order from
"freshest measurement" to "best available":

| Tier | Source | Annotation suffix |
|---|---|---|
| **STAGE 2a** | Server-side mtime delta between successive polls (`watch.py::_attach_iter_walltime`); survives browser reload | `~Xs/iter (from refresh delta, N iters in last Ms)` |
| **STAGE 2b** | Client-side `state.scfPollHistory` rolling buffer; fills the first 1–2 polls before the server has paired samples | `~Xs/iter (from poll estimate, N iters in last Ms)` |
| **STAGE 1 ladder** | Parser-attached `cumulative_walltime_s` from SIESTA's once-per-run `timer: IterSCF` line: walks `history` newest-first for the most relevant cycle | `~Xs/iter (from current step report` \| `from last step report` \| `from SIESTA iter-1 timer; refresh delta pending)` |
| (none) | No SCF data at all | no annotation |

The transition between tiers is automatic and self-healing:

* The mtime-delta path produces a fresh measurement once two polls
  span ≥1 iter in the same geom step.  Cross-step deltas are
  deliberately skipped (the DM extrapolation + mesh rebuild between
  steps is not iter cost), so when a step boundary falls between
  two polls the annotation briefly falls back to the snapshot
  label until the next iter completes.
* The client buffer (`scfPollHistory`) resets on file switch in
  `loadByPath()` — without that reset, the rolling average would
  delta file A's `totalIters` against file B's and surface a bogus
  `(from poll estimate, N iters in last Ms)` for one full buffer
  cycle.  See `core.js` line ~2615.
* The Refresh button on the file picker dispatches a custom
  `molbuilder:results:refresh` event that the trajectory inspector
  listens for in `startPolling`; clicking Refresh runs an immediate
  `pollOnce()` instead of waiting up to 60 s for the scheduled
  tick.  Camera + animation state are preserved (no remount).

The provenance label is load-bearing UX: a user reading "~16.2s/iter
(from SIESTA iter-1 timer; refresh delta pending)" knows the number
is a snapshot from start-of-run, not a measurement of the current
iter — and that a fresh measurement will arrive once two polls have
been compared.  Labels are spelt out (not slug-coded) so the meaning
doesn't require reading the source.

Wire shape pinned in [`web-api.md § 8.4`](./web-api.md#84-apiwatchdata--polling-endpoint).
Algorithm in `web/blueprints/watch.py::_attach_iter_walltime`.
L1 tests in `tests/test_watch_iter_walltime.py`.

---

## 4.8 System load monitor — bottom strip (2026-06-15)

Persistent fixed bottom strip on every page, mounted from
`_app_header.html`.  Polled at 1 Hz from `/api/system/load`
(see [`web-api.md § 9`](./web-api.md#9-apisystemload--server-load-snapshot-2026-06-15)).
Four canvas sparkline cells:

| Cell | Metric | Hover tooltip |
|---|---|---|
| `CPU` | `cpu_pct` aggregate across logical CPUs | `~N/M cores busy` (`cpu_pct × cpu_count_physical / 100`) + loadavg 1m/5m/15m + `[over-subscribed: load > physical cores]` footer when load > phys |
| `RAM` | `ram_pct` | `X.X / Y.Y GB used` |
| `GPU` | `gpus[0].util_pct` | NVIDIA name + util across multi-GPU hosts |
| `VRAM` | `gpus[0].mem_pct` | `X.X / Y.Y GB used` per GPU |

Behavior rules:

* **CPU-only hosts**: server returns `gpus: []`; widget hides GPU
  + VRAM cells.  Strip collapses to two cells.
* **Pause on backgrounded tab**: `document.visibilitychange` →
  `stopTimer()`.  Resume on visibility return.  Zero requests
  while the user is on another tab.
* **Collapse toggle** (`≡` pill): persisted in `sessionStorage`,
  intra-session only.  Per-session UI preference, not a stored
  setting.
* **Color thresholds** read from CSS tokens (`--load-ok` <50%,
  `--load-warn` 50–80%, `--load-bad` ≥80%) so dark / light themes
  pick the right contrast.

The strip is intentionally NOT scoped to the Results tab — it's on
every page, because the load monitor is most useful when the user
is on the Modify tab building the next structure WHILE a SIESTA
job runs in the background.

JS: `lib/system-load-monitor.js`.  CSS: `lib/system-load-monitor.css`.
Backend: `web/blueprints/system_load.py` (psutil core dep,
`nvidia-ml-py` in the `[gpu]` extra with graceful import guard).
L1 tests in `tests/test_system_load.py`.

---

## 5. What this design does NOT include

* No multi-file comparison ("diff two runs"). Bigger feature; later.
* No re-running a script from `/results`. The user goes back to
  Spectra / Build / cli to re-run.
* No new file format support beyond what /watch and /spectra already
  read.  Adding `.fdf.out` parsing is its own task.
* No write operations from `/results` itself (other than the
  existing CSV / XYZ export buttons that move with the inspectors).

---

## 6. References

* Sidebar selection contract: [`./selection.md`](./selection.md)
* Job-layout: [`./job-layout.md`](./job-layout.md)
* Spectra spec (will shrink to generate-only after the merge):
  [`../tabs/spectra/spec.md`](../tabs/spectra/spec.md)
* Watch legacy spec (archived 2026-06-02): [`../archive/2026-06-02-tabs-watch.md`](../archive/2026-06-02-tabs-watch.md)
