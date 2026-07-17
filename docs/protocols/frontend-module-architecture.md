# Frontend module architecture — the target the tabs migrate onto

The **code**-structure counterpart to [`ui-design-contract.md`](ui-design-contract.md)
(which governs UI/CSS) and the companion to [`web-ui-coherence.md`](web-ui-coherence.md)
(data coherence). This doc records the **target** shape of the browser-side JavaScript —
what a *concealed module* is, how it is delivered (ES module), and the order the tabs
migrate onto the shared modules — so the per-tab migration has one spec to follow.

It does **not** re-open the per-module designs: MolView's API lives in
[`molview-module.md`](molview-module.md), VibrationView in
[`vibrationview.md`](vibrationview.md), the workspace in
[`workspace-contract.md`](workspace-contract.md), the sidebar in
[`projects-sidebar.md`](projects-sidebar.md). The open-work tracker for the MolView
core is [`molview-migration-plan.md`](molview-migration-plan.md). This doc sits *above*
those: the shape they all converge to.

> **Status (2026-07):** the design principles + CSS unification are in place
> (`ui-design-contract.md` §1–§9), and **every web tab's viewer now mounts the
> concealed MolView module** — Modify, Transport, Spectra, structure-optimization, and
> both Results inspectors (structure + trajectory, task #34). What remains is the
> *delivery* target below (ES-module conversion + `window` shims + node-test
> `import()`), not the viewer migration itself.

---

## 1. Three senses of "module" — keep them straight

| Term | What it is | In this repo |
|---|---|---|
| **ES module** | A JS *language* feature: `<script type="module">` + `import`/`export`. Private scope by default; explicit wiring; deferred, dependency-ordered load. | `projects/api.js`, `list.js`, `preview.js`, `dialogs.js`, `projects-sidebar.js` |
| **Classic script** | The old `<script src>` with no `import`/`export`; every file shares the global `window`; files talk via `window.molbuilder.*`; load order is hand-managed. | `molview/*`, `workspace/*`, `vibrationview/*`, `spectra/viewer.js`, `viewer.js` |
| **Concealed (architectural) module** | A *design* unit: owns its DOM + CSS + a small public API; a host wires it, never reaches inside. Independent of the two above. | MolView, VibrationView, workspace, projects sidebar, selection panel |

The last is the design concept from `ui-design-contract.md` §7. The first two are *how
the browser loads the code*. **The target below makes each concealed architectural
module an ES module**, so the seal is enforced by the language instead of by
convention.

## 2. Why ES modules are the delivery target

Today a "concealed" module like MolView dumps its whole surface onto
`window.molbuilder.molview.*`, and "don't reach inside" is enforced only by nobody
doing it. An **ES module makes the seal real**:

- **Enforced encapsulation** — a top-level `const`/helper is invisible outside the
  file unless `export`ed. Only the small public API is reachable. Concealment stops
  being a rule to remember.
- **Explicit wiring replaces fragile ordering** — `import { mountMolView } from
  ".../molview/index.js"` resolves load order from the dependency graph, retiring the
  hand-sorted `<script>` list and its "MUST precede …" comments.
- **One mechanism, not two** — ends the ES-modules-*and*-classic-globals split that is
  itself an inconsistency (`ui-design-contract.md` §2.3).

**Target shape per concealed module:** one public ES entry (`<module>/index.js`) that
re-exports a *small* API; every other file in the module is a **private ES submodule**
that `import`s its siblings. Consumers `import` from the index **only** — they cannot
reach an internal file. Vendor globals (`window.$3Dmol`) stay classic; an ES module
reads them normally.

## 3. The transition rule — never big-bang

ES modules load **deferred** (after classic scripts, in dependency order). The instant
a module becomes pure-ES it leaves `window` and runs at a different time, which
**breaks every classic consumer** still calling `window.molbuilder.<name>.*`. So the
conversion is **coupled to its consumers**, one module at a time:

1. Convert a concealed module to ES **together with** the tab(s) that use it — the tab
   becomes an ES module that `import`s it.
2. Keep a **thin temporary `window.molbuilder.<name>` shim** on the module (its public
   API re-exposed on the global) until its **last** classic consumer is migrated, then
   delete the shim.
3. Move that module's node tests from "concatenate the file + read the global" to
   dynamic `import()` (the pattern `test_projects_api_envelope_js.py` already uses).
4. Vendor stays classic; nothing about 3Dmol changes.

A half-converted module with a live classic consumer and no shim is the failure mode
this rule exists to prevent.

## 4. The concealed modules + their one public door

| Module | Public door | Owns | Doc |
|---|---|---|---|
| **MolView** | mount + `molview.data` accessors + view-controls | the 3D viewer **and** the in-memory structure/atoms/frames model; every tab's viewer | [`molview-module.md`](molview-module.md) |
| **VibrationView** | its own 3Dmol seal (normal-mode animation) | spectrum normal-mode playback; a **sibling** of MolView, not a branch inside it | [`vibrationview.md`](vibrationview.md) |
| **workspace** | `persist` / `readState` / `pruneStatesAbove` … | session state + **format-blind** file-bytes persistence (push-only, write-ordered) | [`workspace-contract.md`](workspace-contract.md) |
| **projects sidebar** | `api.js` (`apiRead/Write/…`) + sidebar mount | format-blind file/project layer; **already ES modules** | [`projects-sidebar.md`](projects-sidebar.md) |
| **selection panel** | mount-panel + accessors | atom-selection UI + region model | `ui-design-contract.md` §8.2 |

**File I/O has one door: `api.js`.** Classic-script consumers currently raw-fetch
`/api/files/*` because they can't `import` the ES module (`ui-design-contract.md` §7
gap). This resolves **for free** as those consumers become ES modules in step 3 above —
so build **no** transitional `window.molbuilder.filesApi` bridge for code that is about
to become an ES module anyway.

## 5. Per-tab migration order

Current state (updated 2026-07-17):

| Tab (route) | Viewer today | Migration |
|---|---|---|
| **Modify** (`/molbuilder`) | MolView (concealed) | **done** — the reference consumer |
| **Transport** (`/transport-calculation`) | MolView (`mode:"modify"`) | **done** — mounts on commit + sources gen labels from `molview.data` |
| **Spectra** (`/spectrum-calculation`) | MolView (`mode:"readonly"`) | **done (Card 1)** — read-only inspect card via the shared include; VibrationView left Phase-1 (borrows its own embed) |
| **structure-optimization** (`/structure-optimization`, `index.html`) | MolView (`mode:"readonly"`) | **done (display)** — full read-only card via the shared include; structure into `molview.data`.  k-grid stays a `SiestaConfig` form field (not on the Structure).  Follow-up: source Generate from `molview.data` + drop the raw `/api/files/read` (like Transport increment 2) |
| **Results** (`/results`) | structure + trajectory inspectors on MolView | **done** — structure inspector + **trajectory inspector** (task #34) both on MolView; spectra inspector uses VibrationView (Phase-1) |

Order: Transport (done) → Spectra (done) → structure-optimization (done) →
**Results** structure inspector (done) → **Results trajectory inspector** (done,
task #34) — every web tab's viewer is now MolView.

> **Results trajectory inspector (task #34) — the mechanism + interface (shipped).**
> MolView owns trajectory rendering: a trajectory is the SAME render pipeline with a
> `frame-select` step at the front (§14.5 of [`molview-module.md`](molview-module.md)); a
> single structure is the `frameCount()===1` case. So the inspector does not reimplement
> playback — it **feeds** MolView and is a pure DATA FEEDER
> (`lib/trajectory/core.js`): `mv.mount('#viewer-host', ws, {mode:"readonly",
> owner:"results:trajectory"})`, then `data.openMolecule(frame0)` +
> `data.reloadFrames(coords)` on load and `data.addFrames(newCoords)` for the live-poll
> tail-append. MolView owns the whole view — the frame bar (playback + speed + loop),
> unit-cell display (via the periodicity handed to `openMolecule`), atom-index labels,
> selection + measurement + picking, and atom hiding (its selection/isolate). The tab
> keeps ONLY the **force-vector producer** control: it builds arrows from the parsed
> per-frame forces + its knobs and hands them to MolView via `handle.setArrows` on
> `handle.onChange` (so they re-derive per frame). Per-frame scalars (energy/max-force)
> stay in the inspector's Plotly charts; per-frame structure export is MolView's Export
> knob (current-frame-correct). `#hide-frozen` is a pure force-arrow filter now.
`_molview_scripts.html` is the shared component-stack include (Transport + Spectra use
it; migrating Modify / Results / molview-demo onto it is a follow-up).  Remaining per
tab: the ES-module conversion + `window` shim + node-test `import()` re-point (deferred
— the tabs are still classic scripts that reach `window.molbuilder.molview`).

## 6. Guardrails (every migration step)

- **Don't regress the CSS unification** — the tab keeps `page-shell` + `form-components`
  + its own sheet; no new shared component in a page sheet (`ui-design-contract.md`
  §2.2), one class/token owner (§2.1–§2.3).
- **One door per concern** (`ui-design-contract.md` §7) — data via `molview.data`
  accessors (defensive copies), files via `api.js`, persistence via the workspace;
  no raw `fetch` for `/api/files/*` once the tab is ES.
- **Don't reintroduce the two-writer race** — single-authority mount restore
  (`workspace-contract.md`); persistence stays push-only + write-ordered.
- **VibrationView stays a separate seal** — spectra mounts it *beside* MolView, never
  folds normal-mode playback into MolView.
- **Verify at the page level** — `test_pages_no_js_errors` + the tab's E2E must stay
  green (there is no visual-regression net; a console error is the tripwire).

## References

- [`ui-design-contract.md`](ui-design-contract.md) — UI/CSS + the §7 module-boundary principles this doc realizes in JS.
- [`molview-module.md`](molview-module.md) · [`vibrationview.md`](vibrationview.md) · [`workspace-contract.md`](workspace-contract.md) · [`projects-sidebar.md`](projects-sidebar.md) — the per-module designs.
- [`molview-migration-plan.md`](molview-migration-plan.md) — the MolView core open-work tracker.
- [`web-ui-coherence.md`](web-ui-coherence.md) — the data-coherence companion.
- [`../architecture.md`](../architecture.md) — the top-level subsystem index.
