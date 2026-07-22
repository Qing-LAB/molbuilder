# MolView ESM finalization — principles + plan (LIVING guidance)

**Purpose.** This is the standing guidance for finishing MolView's conversion to a **fully
concealed, clean ES module**. It exists so the principles below do not have to be restated every
session. It is a *working plan* — update the status checklist as steps land; when the job is done,
this doc collapses into a short "done" note and the design lives in
[`molview-module.md`](molview-module.md).

> **Design source of truth: [`molview-module.md`](molview-module.md).** That doc is authoritative for
> what MolView *is* and its public API. This doc only tracks *how we finish the ESM conversion to
> match it*. Where code diverges from `molview-module.md`, the code is wrong.

---

## 1. Principles (non-negotiable — do NOT relitigate)

1. **Architecture, not patching.** Do not paper over a broken global read with a call-time getter or
   a local shim. The fix is to convert the consumer to `import` from the module. Band-aids are
   forbidden; they leave the obsolete pattern in place.
2. **Import the stateless pieces; LOOK UP the live data.** This is the core rule (molview-module.md
   §D.0). Two different things, never conflated:
   - **`mount` and `formula`** are stateless → `import { mount, formula } from
     "/static/lib/molview/index.js"`.
   - **`molview.data`** is MolView's live internal state → **looked up at runtime**
     (`window.molbuilder.molview.data`), **never imported or injected into a reader**. Reading is
     decoupled from loading: loading/writing is an explicit op (`installMolecule`/`applyOp`/`save`);
     reading returns *what MolView has right now* (including nothing). A fixed `import { data }`
     wires the reader into MolView's internals and couples reading to loading — forbidden.
     (Tests may inject a data stub through a `configure`/`_bind` seam; production always looks up.)
3. **Dump the transitional shims that are pure scaffolding — but NOT the design seams.** Publishes
   that only existed to bridge the migration (e.g. `window.molbuilder.fmt` once every reader
   `import`s `formula`) get deleted. But **`window.molbuilder.molview.data` is a permanent front
   door for reading** (rule 2), **not** scaffolding — it stays. Likewise `molview.data.selection`
   (read the same way) and the projects↔molview lookup in `parser.js` (deliberate decoupling).
4. **No blind moves / no blind sed.** Before moving or renaming, distinguish what belongs to the
   **whole web framework** from what belongs to **one tab**. Verify each change file-by-file; never a
   blanket namespace `sed`. (See [[feedback_no_blind_sed_namespace_rename]].)
5. **Symmetrical, logical structure.** Every tab isolates its own code + UI + CSS in its own dir
   (`modify/`, `spectra/`, `results/`, `structure-optimization/`). Shared/framework rules live in
   `lib/` shared sheets. Nothing tab-specific sits loose in the public static root; nothing
   framework-wide sits inside a tab dir.
6. **MolView vs Workspace are distinct concealed modules — never mix them.**
   - **Workspace** = persistent **file operations in a project subdirectory** (`.molbuilder_workspace/`),
     format-blind byte store. Its own module, its own doc ([`workspace-contract.md`](workspace-contract.md)).
   - **State timeline** = a **MolView submodule** (retract of edit state). It owns the *policy* and is
     **built on top of** the Workspace's persistent-file primitive via the Workspace's public API —
     correct layering (mechanism vs policy), NOT a leak. Do **not** move the `/api/state-timeline/*`
     transport into MolView. (See [[reference_timeline_on_workspace_layering]].)
7. **k-grid is not MolView.** It is an FDF/SIESTA reciprocal-space sampling parameter on
   `SiestaConfig`. MolView has no k-grid render step and stores no k-grid.
8. **Verify against reality.** Every step ends green on: `node --check`, the node ESM harness
   (`tests/_node_esm.py`), the affected pytest suites, **and a real-browser check of the tab**. Unit
   tests can pass while the UI is broken — the browser check is mandatory.

---

## 2. Target architecture

- `lib/molview/` + the `mol-*.js` embed files are ES modules aggregated by `index.js`.
- `index.js` **exports the full public API** consumers need — currently `mount`, `data`, `formula`
  (add others only if a consumer legitimately needs them; keep the surface minimal).
- Every consumer is a `<script type="module">` that imports from the door. **No consumer reads
  `window.molbuilder.molview` / `.fmt` / `.style` / `.axes` / `.viewer`.**
- Internal MolView cross-module references are `import`s, not `window.molbuilder.*` reads.
- Node tests `import()` the modules (no global-read seams).
- **Zero transitional shim publishes remain.**
- Vendor globals (`window.$3Dmol`) stay classic — that is a third-party seal, not ours.

---

## 3. The plan — safe phased order (never leaves the app broken)

The migration is done in an order where **every intermediate state works**, because the shims stay
published until the very last step:

- **Phase A — the door exports everything (additive, safe).** Ensure `index.js` re-exports the full
  public API. Purely additive; nothing breaks. *(Status: `mount`, `data`, `formula` exported.)*
- **Phase B — convert every reader to `import` (additive, safe).** Consumers, internal cross-module
  reads, and node tests move from `window.molbuilder.*` reads to `import`. Each still works because
  the shims are still published, so this is safe to do incrementally, one file at a time, verifying
  each. Classic consumers become `<script type="module">`.
- **Phase C — delete the shim publishes (the payoff).** Only after Phase B leaves **no** reader of a
  given global, delete that `window.molbuilder.* = …` line. Do it per-global, re-grepping for readers
  before each deletion.
- **Phase D — docs + full verify.** Update `molview-module.md` (no globals; single import door),
  `web-module-map.md`, and this checklist; run the full suite + browser-verify every tab.

---

## 4. Consumer inventory (Phase B work list)

Readers of MolView globals that must become `import` consumers (grep-verified this session):

| Consumer | Tab | Kind today | Reads |
|---|---|---|---|
| `structure-optimization/viewer.js` | structure-opt | classic → **make module** | `molview`, `fmt.formula` |
| `modify/viewer.js` | Modify | classic → **make module** | `fmt.formula`, `molview.data` |
| `modify/periodicity.js` | Modify | classic → **make module** | `molview` |
| `modify/selection-bootstrap.js` | Modify | module | `molview`, `molview.data.selection` |
| `spectra/viewer.js` | Spectra | module | `fmt`, `molview` |
| `lib/trajectory/core.js` | Results | module | `molview`, `molview.data` |
| `lib/transport/core.js` | Transport | module | `molview` |
| `lib/inspectors/structure.js` | Results | module | `molview` |
| `modify/structure/{file,save,page}.js` | Modify source panels | classic → **make module** | `molview.data.installMolecule`, `fmt` |
| `molview/demo.js` | demo | module | `molview` |

Internal MolView cross-module global-reads to convert to imports (Phase B): `data-model.js`
(`molview.selection.store`), `selection/panel.js` (`molview.data`), `mol-viewer-embed.js`
(`molbuilder.fmt`), plus the node-test seams.

### 4.1 Kill the `window.molbuilder.loadStructureText` global

`modify/viewer.js` publishes `window.molbuilder.loadStructureText(text, filename)` — a thin alias
that just calls `molview.data.installMolecule({text, filename})` (+ a status message). The Sources-card
generators (`modify/structure/{smiles,dna,rna,peptide,name}.js`) consume it as their injected
`viewerLoader` (a dependency-injection seam, kept via `configure({viewerLoader: …})` + a
`_lazyResolve` fallback). This is a transitional `window.molbuilder.*` shim and comes out in Phase B/C:

- The generators become ES modules and inject **`data.installMolecule`** (imported from the door,
  bound) as their `viewerLoader` — the DI pattern stays, the *global* goes.
- `modify/viewer.js` stops publishing `window.molbuilder.loadStructureText`; the global disappears.
- Net: the load path is `generator → data.installMolecule` through imports, no global alias.

---

## 5. The structure-optimization dir move (symmetry — principle 5)

The structure-opt tab was the last one with its page files loose in the public static root. They move
into `static/structure-optimization/`, matching `modify/` etc.:

- `static/viewer.js` → `static/structure-optimization/viewer.js` (its page controller — becomes a
  `type="module"` import consumer per Phase B).
- `static/style.css` → `static/structure-optimization/style.css`, **stripped to structure-opt-only**:
  keep `#analyze-card`, `#generate-card`, `#generate-fdf/pyscf`, the generate-card sub-`.tabs`; **do
  NOT keep** framework classes already owned by shared sheets the page loads (`.status` → `page-shell`,
  `.app-grid` → `form-components`, `.workflow-group` → `form-schema`). Only `index.html` ever loaded
  `style.css`, so other tabs are unaffected — verified.
- Update every reference (templates, tests, docs). `watch/viewer.js`, `modify/viewer.js`,
  `spectra/viewer.js` are **different files** — do not touch.

---

## 6. Status checklist (update as steps land)

**Read/load rule + Phase-C shim cleanup (2026-07):**
- [x] All consumers apply the read/load rule: `import { mount, formula }`; **look up** `molview.data`
      (never import it). Composition-root data injection reverted; `page.js` looks up at call time.
- [x] `window.molbuilder.fmt` shim DROPPED (pure helper; every reader imports `formula`).
- [ ] `window.molbuilder.style` / `.axes` shims — droppable the same way: the embed still reads them
      via the global (`mol-viewer-embed.js` ~L2271/2713/2756/2880); convert to
      `import { spec } from "./mol-style.js"` / `import { axes } from "./mol-axes.js"`, then delete
      the publishes. **Touches the embed render path — verify a browser render (style + axes toggle).**
- [ ] `window.molbuilder.molview.mount` shim — no readers (all `import { mount }`); droppable, but
      mind `runtime.register("molview", …)` which reads the global.
- [ ] KEEP (design seams, NOT shims): `molview.data` / `.selection` (live read doors, §D.0);
      `molbuilder.viewer` (shared embed seal); the node-test-seam publishes.

---

### (historical) original phased checklist

- [x] Phase A: `index.js` exports `mount`, `data`, `formula`.
- [x] structure-opt dir created; `viewer.js` + `style.css` `git mv`'d into it; `index.html` links/tag repointed to the new path + `type="module"`.
- [x] structure-opt `viewer.js` converted to `import { mount, data, formula }`; ALL molview/fmt global reads deleted; **browser-verified** (card mounts, `#info-formula`=`C6H4S2Au432` compact — root-fixed, no getter band-aid).
- [x] structure-opt `style.css` stripped to tab-only (`.status` reduced to its layout tweak; page-shell owns the base — zero visual change).
- [x] Test path refs updated for the move; affected suites green (`test_format_fetch_error`/`test_form_state_persistence`/`test_viewer_structure_path`/`test_in_body_labels`/`test_live_poll_invariants` = 49 pass; `test_web`/`test_xss_audit`/`test_css_no_duplicate_selectors` = 515 pass).
- [x] Phase B (already-module consumers): `demo.js`, `modify/selection-bootstrap.js`, `spectra/viewer.js`, `lib/transport/core.js`, `lib/inspectors/structure.js`, `lib/trajectory/core.js` — all `window.molbuilder.molview`/`.fmt` reads replaced with `import { data, formula }`; syntax-clean; demo browser-verified (card mounts, water loads).
- [x] Phase B (Modify classic files): `modify/viewer.js` + `modify/periodicity.js` → `import { data, formula }`; modify.html tags → `type="module"`. **Browser-verified**: card mounts, `title-readout`=`H2O` (imported formula), `loadStructureText("water")`→ok (Sources-card loader path works with viewer.js deferred, via `_lazyResolve`).
- [x] Phase B (classic consumers → module): all 8 `modify/structure/*` converted — **browser-verified** (SMILES `CCO`→9 atoms, title `C2H6O`, generator's own status fires; no load-order breakage) — `smiles/dna/rna/peptide/name` import `data` + inject a `_loadText = (t,f)=>data.installMolecule({text,filename})` adapter as `viewerLoader` (no more `loadStructureText` read); `file/save/page` import `data` (was `molview.data`). modify.html's 8 generator tags → `type="module"`. node-clean. **Pending browser-verify after restart** (biggest load-order risk: 8 classic→deferred modules reading `structurePage`/`warningModal` globals). `loadStructureText` publish RETAINED (E2E hook + Modify status) with a follow-up note to remove it.
- [ ] Phase B: internal cross-module reads (`data-model.js`, `selection/panel.js`, `mol-viewer-embed.js`) + node-test seams converted.
- [ ] Phase C: all transitional shim publishes deleted (per-global, re-grep first).
- [ ] Phase D: `molview-module.md` / `web-module-map.md` updated; full suite + every tab browser-verified.
