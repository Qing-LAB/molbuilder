# molbuilder — documentation

**This is the ONE index.** Every document under `docs/` is listed here, one
line each. If a doc is not listed here it does not exist (test-enforced:
`tests/test_docs_structure.py`).

> **Migration in progress (started 2026-07-26).** The previous docs tree is
> frozen at [`../old_docs/`](../old_docs/) and is being migrated here piece
> by piece, each doc reconciled against the code at its move. The ledger —
> what lives where, what is pending — is [`MIGRATION.md`](?doc=MIGRATION.md).
> Until a doc migrates, its old_docs copy remains the source of truth.

## Structure — domains, not document kinds

Documents are grouped by **domain** (the subsystem a reader works on), not
by kind of contract. Kind is expressed in the filename suffix and the
header, inside the domain.

| Folder | Domain |
|---|---|
| *(root)* | The spine: this index, `design.md` (mission · principles · decisions), `architecture.md` (the reuse map: task → tool), `roadmap.md` (THE one plan) |
| `model/` | The data model (L1): Structure, periodicity, annotations, codecs & load/save, sidecars, region labels, selection grammar, chemistry, the parse stack, the data vocabulary |
| `science/` | Scientific correctness: validation machinery, chemistry correctness, pseudopotential standards, parameter tuning |
| `engines/` | Per-engine emitter specs: SIESTA, PySCF, transport/TranSIESTA, builders, GPU build recipe |
| `execution/` | Running jobs: script generation & wrappers, deployment config, SLURM, the JobSet framework (bundles/ladders/sweeps), benchmarks, run layout & decoding, checkpoints, workflow handoff |
| `web/` | The whole front end: tabs, MolView, workspace, projects sidebar, results/inspectors, forms, UI/CSS contract, web API |
| `ops/` | Installing and serving the app: install model, deployment, auth/rate-limit |
| `process/` | How we work: code conventions, test strategy, audit playbook, CLI conventions, package layout |
| `archive/` | History, date-prefixed. **Not a source of truth.** |

## The rules

- **R1 — one index, never lagging.** Every non-archive doc has exactly one
  line here, added **in the same commit** that adds, moves, merges, or
  archives the doc. The master index is updated at EVERY step — it is never
  allowed to drift from the tree (the structure test fails on unindexed or
  dangling entries; the old design.md §0 index rotted to ⅔ coverage because
  updating it was a separate chore).
- **R2 — provenance header.** Every doc starts with a header block naming
  its **Role** (`contract` | `guide` | `overview` | `plan` | `process`), its
  **Domain** (folder), and its **Companions** (linked related docs). A
  contract is the sole source of truth for its surface; a guide explains it
  in plain language and never contradicts it.
- **R3 — contracts don't hold plans.** Durable decisions stay in contracts.
  Phasing, status and open work live in `roadmap.md` (one pointer allowed in
  the contract).
- **R4 — one archive.** Superseded content moves to `archive/` with a
  `YYYY-MM-DD-` prefix; the archive README says what superseded it.
- **R5 — names carry the vocabulary.** File names use the system's canonical
  terms (see `model/data-vocabulary.md` once migrated) and must not collide
  across meanings — e.g. the run→next-calculation handoff is
  `execution/handoff-bundle.md`, never plain "bundle", which the JobSet
  framework owns. **Sub-documents share the master's filename as a prefix**, so
  the hierarchy is visible in the name itself: a master `structure.md` has subs
  `structure-periodicity.md`, `structure-annotations.md`, … (a filename prefix,
  not a subdirectory — the name alone shows the parent).
- **R6 — born here.** New documents are created in this structure only; the
  old tree is frozen (test-enforced against the ledger).
- **R7 — internal links use the document-module convention.** The Documents
  tab serves docs through the module (`/documents?doc=<path>`), **never** as a
  raw `.md` path — a raw relative `.md` href 404s in the rendered view. So a
  link between docs points its target at `?doc=` followed by the
  docs-root-relative path — e.g. [`model/structure.md`](?doc=model/structure.md).
  Index links follow the same form.
  Enforced by `tests/test_docs_structure.py`. (A doc that has not migrated yet
  is named in inline code, not linked — no target exists to point at.)
  **No `#` heading-anchor links** (browser-verified 2026-07-26): the renderer
  forces `target="_blank"` on every in-doc link *and* does not honor `#` anchors,
  so an anchor link opens a stray blank tab and never scrolls. For a same-doc
  section pointer use plain text ("see § 8"), not `[…](#…)`.

### The doc rule (carried over — still the point of all of this)

> Tests must be derivable from the spec without reading the implementation.
> Code reviews must verify code matches spec, not code matches reviewer's
> expectations.

## Editorial rules — how documents are written and merged

These bind every write to this tree — migration reconciles AND ordinary
edits afterwards. They are re-read at every migration gate.

- **E1 — structure first, never just append.** Before merging or updating
  a document, map the overall structure — its topics, its table of
  contents, and its siblings' — and reorganize so the information sits
  logically. Adding a new section onto old scaffolding because it is easy
  is exactly how the previous tree rotted. A merge is a re-architecting of
  the combined content, not a concatenation.
- **E2 — Mermaid diagrams wherever they explain.** Data structures
  (`classDiagram`), dependencies and workflows (`flowchart`), API designs,
  protocol sequences (`sequenceDiagram`) — if a picture explains it, the
  picture is required, and it is Mermaid (renders in the Documents tab).
  ASCII diagrams are converted when a doc is touched.
- **E3 — plain language, full rigor.** Translate jargon into simple
  language with concrete scenarios ("when you click X, Y happens") — but
  lose NOTHING: every constraint, number, edge case, and decision
  rationale survives the translation. Simplify the words, never the
  content. A coined shorthand may only be used after the sentence that
  defines it.
- **E4 — scientific content is evidence-based, and enriched.** Where a doc
  makes a scientific argument (defaults, validation thresholds, method
  choices), the physical/chemical foundation and the reference trail are
  respected — preserved on merge, and ENRICHED where thin: state the basis,
  cite the method/literature, keep defensible defaults *with their
  justification*. A scientific claim without its foundation is drift
  waiting to happen.
- **E5 — organize by the reader's surface, with worked examples.** When a
  doc (especially a merge) serves both the **Python/CLI backend** and the
  **JS/user frontend**, organize its body by those surfaces so each
  developer lands directly in their part — a clearly-labelled backend
  section (the dataclass/codec/CLI use) and a frontend section (the JS
  module / UI use), rather than one undifferentiated contract. Give each
  surface **runnable example code** (a Python snippet, a JS snippet, a CLI
  line — real, not pseudocode) and a **diagram** (E2) of how the two sides
  exchange data across the wire. The test of a good doc here is that a
  backend dev and a frontend dev each find "how do I use this, with an
  example" in under a minute. (Single-surface docs keep their natural
  structure — this binds the two-surface ones.)

## Migration protocol (per doc)

Moving a doc from `old_docs/` is a **review gate**, not a file move.
The editorial rules above apply to every step in full:

1. Read it against the current code — fix drift or archive it.
2. Map the structure first (E1): this doc's TOC + its target-domain
   siblings'; decide the merged/reorganized shape before writing.
3. Merge overlaps with sibling docs instead of carrying duplicates.
4. Extract any plan/status content into `roadmap.md` (R3).
5. Rewrite where needed: plain language (E3), Mermaid diagrams where they
   explain (E2), scientific foundations preserved + enriched (E4).
6. Add the provenance header (R2) and the index line here (R1) — same
   commit, always.
7. Repoint inbound references — other docs, code comments, tests — to the
   new path (grep-verified, per file; no blind rewrite). **Under keep-and-mark
   this is a closeout task, not per-doc:** the old file stays resolvable (just
   `_migrated_`-prefixed) until `old_docs/` is deleted, and merges shift
   section numbers, so bulk comment-repointing is done once at closeout
   (Wave 10) rather than churning every code file on each doc move.
8. Mark the ledger row `moved` (or `merged-into <doc>` / `archived`), **and**
   mark the old file done by renaming it with the `_migrated_` prefix
   (keep-and-mark — the old tree is kept intact for the closeout
   cross-check, never deleted mid-migration). Same commit as the move.

**Order (see [`MIGRATION.md`](?doc=MIGRATION.md) for the wave plan):** components
first, bottom-up (the data model, then what builds on it, then the surfaces);
the summary docs — `design.md` (a concise outline that points at the detailed
docs) and `architecture.md` (the reuse map) — are composed **last**, over the
settled tree, so they never summarize a moving target. `roadmap.md` (the
forward plan) leads and is done early.

## Index

*(grows as documents migrate, bottom-up by domain; the summary spine —
`design.md`, `architecture.md` — lands last)*

| Doc | Role | Owns |
|---|---|---|
| [`roadmap.md`](?doc=roadmap.md) | plan | The ONE plan: every open feature/backend workstream + the closed-work log |
| [`model/overview.md`](?doc=model/overview.md) | overview | The model domain's **start-here** map (every model doc + when to open it) + the atom-index convention (0-based internal / 1-based user-facing / per-engine) |
| [`model/structure.md`](?doc=model/structure.md) | contract | The `Structure` object (master): the L1 codec (`to_dict`/`from_dict`/`to_wire`), geometry I/O, the L2 paired-file door, and the JS load/save doors |
| [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) | contract | *(sub of structure)* Per-axis box behaviour: `cell` · `cell_origin` · `axis_kind` · derived `pbc` · `vacuum` (k-grid is a `SiestaConfig` knob, not here) |
| [`model/structure-annotations.md`](?doc=model/structure-annotations.md) | contract | *(sub of structure)* The per-atom channel model (`tag`/`flag`/`value`; `regions`/`frozen` built-ins), persistence, engine translation, and the region-label vocabulary |
| [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md) | contract | *(sub of structure)* The `.molstruct.json` save file: envelope (`schema_version`/`structure_hash`/…), schema versioning (v3–v6), the codec, and the `.xyz`↔sidecar pairing rule |
| [`model/chemistry.md`](?doc=model/chemistry.md) | contract | The chemistry helpers on a `Structure`: net-charge resolution (phosphate heuristic + override), protonation, `add_hydrogens`, clash relief, dipole (correctness machinery → `science/`) |
| [`model/parse.md`](?doc=model/parse.md) | contract | The unified read stack: three ABCs (File/Text/Dir parsers), the frozen `ParseResult` hierarchy, the registry (`detect`/`parse`/`parse_dir`/`parse_text`), package layout, plugin + composer contracts |
| [`science/overview.md`](?doc=science/overview.md) | overview | The science domain's **start-here** map + the two cross-cutting rules: validation is advisory-while-editing / enforcing-at-generation (`report()` is the gate), and the validation-pass check catalog (geometry / cell / k-sampling / spin-charge / field-range) with rationale |
| [`science/validation.md`](?doc=science/validation.md) | contract | The runtime scientific-validation machinery: the engine-agnostic chemistry analyzer (`analyze_structure`→`ChemistryAnalysis`), the 3 noble-metal categories, the per-engine adapter registry, the consumers (`check_open_shell_metal` + `/api/structure/analyze`), and the `validation/` package layout |
| [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) | contract | The chemistry control surface end to end: the 5 points where chemistry can go wrong, the `(charge, spin)` science (conventions + coordination-dependent spin), the pure primitives (`total_electrons`/`check_spin_charge_parity`/`detect_open_shell_metals`/`explain_metal_spin`), the hemeC-dithiol post-mortem, and the per-step audit checklist |
| [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) | contract | The `.psml` pseudopotential coverage checks (C1–C6): the two entry points (SIESTA preflight + `molbuilder pseudo check`), the ERROR/WARN severity model (incl. the XC family-vs-author split), the dead-KB-projector guard, what is *not* checked, and the KB/ONCVPSP/PseudoDojo references |
| [`engines/siesta.md`](?doc=engines/siesta.md) | contract | The SIESTA `.fdf` emitter (`render_fdf`/`convert`): the 14 output sections, the charge / spin / cell-padding contracts, k-grid & lattice, the `Diag.Algorithm`/ELPA/GPU emit contract, staged optimization, and the sibling `.molwatch.log` / Makov-Payne / memory modules |
| [`engines/pyscf.md`](?doc=engines/pyscf.md) | contract | The PySCF `.py` script emitter (`render_script`): output files, the logging / optimizer / spin-method / charge contracts, the engine-agnostic molwatch-log format, in-script staged optimization, and the publication-quality parameter guide (tiers, basis/functional, methods-section template + citations) |
| [`engines/transport.md`](?doc=engines/transport.md) | contract | The TranSIESTA / NEGF transport workflow: the one-device→three-coupled-runs model, the NEGF physics, the `molbuilder transport` CLI, region-label-driven electrode discovery (`TS.Elec`/`TS.ChemPots` emit), the cross-run consistency **invariant set** (I1–I13 → preflight gates), the scientific baseline, and the Brandbyge/Papior/Xiao references |
| [`engines/builders.md`](?doc=engines/builders.md) | contract | The structure-synthesis contract: the five builders (peptide / DNA / RNA / SMILES / name), the sequence-notation grammar (`[SEP]` brackets, 5′/3′), the pluggable nucleic-acid backend registry (threedna / amber / rdkit), backend-quirk repairs, the mismatch-clash policy, the tri-state hydrogen control, and the CLI + Python surfaces |
| [`engines/tuning.md`](?doc=engines/tuning.md) | contract | The cross-engine optimization-quality dial: the four-tier framework (screening → publishable → tight), per-parameter guidance (optimizer, force/SCF tolerance, mesh/basis/k-grid, step caps), the shipped SIESTA + PySCF three-stage ladders, the cross-engine parameter map, restart strategy, and the scientific references behind each default |
| [`engines/overview.md`](?doc=engines/overview.md) | overview | The engines-domain **map** (what siesta / pyscf / transport / builders / tuning each own) plus the three cross-engine contracts it owns: the shared **script-contract wrapper** (provenance / ATOM-METADATA / user-custom), the **UI→config→script boundary-condition contract** (no silent absorption — the spectra frozen-atom reference instance + the A/B/C preflight), and the shared **staged-optimization / non-convergence policy**; plus the engine registry for adding a new engine |
| [`execution/overview.md`](?doc=execution/overview.md) | overview | The execution-domain **start-here** map (which of the four execution docs answers what) + **the current→target transition** — single-task works everywhere, the JobSet framework works from the CLI, the job system in the browser is the target — with a shipped-vs-planned × CLI-vs-web **status matrix** |
| [`execution/job-contracts.md`](?doc=execution/job-contracts.md) | contract | The stable on-disk formats + shared vocabulary every surface rests on: the **run-directory layout** (one-job basename rule, file catalogue, multi-stage suffixing, the directory discovery chain, the `project/topic/structure` tree + the nine canonical topics, the `.run.sh`/`.sbatch` wrapper), the **generated-script reserved blocks** (HEADER/PROVENANCE/BENCH-MARKS/ATOM-METADATA `v4`/USER-CUSTOM + the per-engine emit matrix), **warm/cold restart** (the four behaviours, per-engine warm-file inventories, the status banner), the workflow **handoff bundle** (`BundleResult` + materialisation), and the **data vocabulary** (persisted-artifacts registry, the `@major` schema convention, the config↔scheduler parameter translation, identifier conventions) |
| [`execution/running-a-job.md`](?doc=execution/running-a-job.md) | guide | The usable single-job path (web single-task + CLI): the self-contained **standalone / detection contract** (T/M/C baked at generate, only allocation + hardware read at runtime), **runtime resource resolution** (MPI/OMP precedence + the `n_atoms` rank clamp, GPU load-balance / MPS / NUMA pinning), the wrapper flags, **watching a run** (the backgrounded monitor + SCF timing + the propor/IMAX hint + the shipped `decode_run_dir` decoded-run view), **`molbuilder.json`** config (`script_generation` / `scheduler` / `execution` / `envs` + the `.sbatch` header sources), and **checkpointing** (`molbuilder snapshot` git + `.binsnapshots`) |
| [`execution/job-system.md`](?doc=execution/job-system.md) | guide | The JobSet batch/staged/HPC framework — **CLI-shipped, web-pending**: the `job-set@1` model (`JobSet`/`Job`/`Resources`/`Carry`, sweep vs ladder), the two producers (SIESTA staged ladder `stages_to_jobset` + benchmark sweep `sweep_to_jobset`; the PySCF ladder is an in-script loop, **not** a JobSet), the `molbuilder jobset prep/plan/submit/status` lifecycle + `fdf --jobset`, **SLURM deployment** (dependency chains, routing domains, submit-vs-direct), the **benchmark workflow** (`environment@1`/`bench-manifest@2`/`bench-result@1`, the `(G,K,c)` grid, winner + recommendation), and the **target web migration** (roadmap workstream 1, D7-gated) |
| [`web/molview.md`](?doc=web/molview.md) | contract | The embeddable 3D structure viewer used on every tab — **two surfaces in one doc**: a plain-language **user guide** (moving/styling, the toolbar toggles, click/filter selection + amber shape-glow, measurements, region labels, trajectory playback, the Export menu) and the **developer contract** (the one ES-module door + look-up-live rule + concealed-3Dmol seal, `mount()` and the 15-key handle, the `molview.data` model + `installMolecule`/`exportFile`, the `applyOp` ops-as-data registry, the four-tier render engine, the selection store, the session-state timeline, the `/api/build/load` + `/api/modify/*` wire); § 24 points to the **VibrationView** sibling (now its own doc) |
| [`web/vibrationview.md`](?doc=web/vibrationview.md) | contract | The **VibrationView** module — the concealed viewer that **animates a vibrational normal mode** (a *sibling* of MolView, mounted only by the spectra viewer): the `mount()` door + handle (`showMode`/`play`/`pause`/`setAmplitude`/`setSpeed`/`dispose`), the animation model (`pos = eq + amp·cos(φ)·disp`, live amplitude/speed, greyed frozen atoms), the eigenvector scatter (`mode-math.js`), the **semantic seal** (owns the clock/knobs/tick-math, drives a drawing surface through generic doors — not a second 3Dmol wrapper), spectra-tab wiring, and the **current → target** note: it still *borrows* MolView's shared embed via the transitional `molbuilder.viewer` global — full separation (own concealed seal + `lib/viewer/`→molview) is **task #104** |
| [`web/workspace.md`](?doc=web/workspace.md) | contract | The session-persistence module — saves a tab's in-progress work so a reload or Undo brings it back, in **two copies** (a fast `sessionStorage` mirror for same-tab reload + a numbered, crash-safe server **state timeline** for Undo history), written by one `persist()` call; the front door `dispatcher.js` + the sole-sessionStorage-owner `snapshot-io.js`; the public surface; the **boundary** (MolView owns *when/what* to save + the molecule, workspace owns only *where* the bytes go — zero data accessors); the mount-restore `hasRestorableSnapshot()` rule; and the note that a *file* save is the Modify Save panel + projects door, not the workspace |
| [`web/projects.md`](?doc=web/projects.md) | contract | The projects sidebar file browser — the one door (`window.molbuilder.projects`) tabs use to browse/select/open/save files: the **selection** (the two `current_dir`/`current_file` slots; single-click=preview→`onChange`, double-click=commit→`onCommit`), the **content-blind byte layer** vs the one **content-aware `parser` door** (`openMolecule` → `/api/build/load` → `molview.data`; `saveMolecule` → `/api/structure/save`, server owns the sidecar; the needs-overwrite handshake; why file-only), the **UI** (breadcrumb, tree + `⋯` menu, filter, header buttons, dialogs, the view/edit preview pop-up, the run-history checkpoint panel, layout), the full public surface, and the shipped-vs-planned split |
| [`web/presenters.md`](?doc=web/presenters.md) | contract | The Results-tab **file-viewer registry** — the switchboard that picks the right viewer for a file (`.xyz`→3D structure · `.molwatch.log`→trajectory movie · `.spectra.json`→spectrum · `.md`→markdown editor · `.log`/`.fdf`→text pane): the presenter **contract** (`name`/`displayName`/`match`/`mount` + `isResult`/`resultCategory`), the `register`/`pick`/`mount` surface + the shared `ctx` file reader, the pick→dispose→mount dispatch, thin-adapter vs heavy-engine (trajectory/spectra cores referenced), adding a viewer, and the **current → target** note (code is still classic `molbuilder.inspectors`; the ESM `presenters` rename is task #102) |
| [`web/runtime.md`](?doc=web/runtime.md) | contract | The **runtime registry** (`molbuilder.runtime` — `register(name,api)` + `whenReady(name)→promise`, the ask-don't-grab fix for the classic-`<script>`-vs-`type=module` load-order race; the five calls + the rules that bite) **and the shared building-block catalogue** (the notification bar, the discard-unsaved modal, the detection chip, the one markdown→safe-HTML renderer, path/constants helpers, the system-load strip; with one-line pointers to `form-schema.md`, the transport region-labels, and the already-ESM `xyz-io`), plus the current→target ESM note (the by-kind grouping + tasks #105/#106/#107) |
| [`web/notifications.md`](?doc=web/notifications.md) | contract | The **app-wide notification framework** (`molbuilder.notify`) — one consistent surface for system-level messages on any tab: the general **API** (`show({level,message,id?})→{dismiss}` / `clear` / `clearAll` / `list`, error/warn/info, **dedup-by-id + ×N**, runtime-registered — open to all callers), the **display** (host + CSS + JS co-located in the shared `_app_header.html` so a page can never `show()` into a missing host; live-region a11y, newest-on-top, ×, Clear-all), the **one built-in source** today (a failed save → the `molbuilder:persist-error` toast), and the **current→target** note (classic today → ESM + an auto-dismiss `ttl`, **task #105**) |
| [`web/form-schema.md`](?doc=web/form-schema.md) | contract | The engine-option **form builder** — the Build/Spectra/Transport config forms are **generated from the Python config dataclass** (`SiestaConfig`/`PySCFConfig`/…), not hand-written: the round-trip (dataclass → `dataclass_to_form_schema` → `GET /api/build/schema/<engine>` → `renderForm` → `collectForm` → the generated input), the four calls (`fetchSchema`/`renderForm`/`collectForm`/`setValues`), the nine field kinds → controls (checkbox/int/number/text/dropdown/tri-select/int-triple/comma-floats/stage-table), which tabs consume it, and the current→target ESM note |
| [`web/web-api.md`](?doc=web/web-api.md) | contract | The **server HTTP API** the browser calls — the shared contract + the one route catalogue: the uniform `{ok,…}`/`{ok:false,error}` **envelope** + the canonical structure wire shape (`workspace_payload`) + the client mirror (`_fetchEnvelope`); the **status-code map**; the **security posture** (CSP `default-src 'self'` / no inline JS / no CSRF token beyond same-origin+SameSite / the always-on per-IP rate limiter / the 50 MB cap); the complete `/api/*` **catalogue** grouped by domain, cross-linking each route's owner module doc; **full shapes** for the un-owned groups (app-level, build generate/validate, checkpoint, system, docs, admin, auth); a load round-trip; and the **removed-routes** list |
| [`web/ui-contract.md`](?doc=web/ui-contract.md) | contract | The cross-cutting **CSS/layout conventions** that keep every tab one consistent app: the **five stylesheet layers** (`tokens.css` → `page-shell.css` → `form-components.css` → module sheets → tab sheets) + the one-owner-per-shared-element rule; the **design tokens** (`lib/tokens.css` single source; `var(--token)` never raw hex; the `var(--token,#fallback)` embed pattern); **content-driven responsive** (`min()` grid floor, `.card-row`, the fused-card `@container`, the 641/640 dock↔drawer breakpoints); the visual-coherence conventions (`.status` severity one owner, the `[hidden]` guard gotcha); and the CSP style-allowed/script-banned boundary |
| [`web/results.md`](?doc=web/results.md) | contract | The **Results tab** — a dispatch shell that opens a finished calculation: the file **picker** (lists result-class files newest-first, auto-picks, the `fileSelected` event, Refresh re-scans), the tiny **controller** (pick → dispose → mount into one panel, the parsing-cover), the three viewers you land in (structure/trajectory/spectrum — cross-linked), what a **mounted viewer remembers** ("Refresh = open the same file again"), and the **bundle handoff** (`/api/results/bundle` writes a `.xyz`+`.molstruct.json` pair to feed the next stage) |
| [`web/trajectory.md`](?doc=web/trajectory.md) | contract | The **trajectory viewer** — a geometry-optimization run shown as a 3D movie + live convergence plots: the engine as a **data feeder** (MolView owns playback/cell/labels; the engine feeds frames + forces + the three force controls), the **four plots** (energy · max-force with the all-vs-free dual trace when atoms are frozen · SCF energy · SCF residual) + the green **convergence-target** lines (read from the run's own output) + auto-log-y, the **run badge** + the three-tier per-iter walltime line, **live polling** (`/api/watch/load` then poll `/api/watch/data` every 15 s, finish = two "finished" ticks, Refresh = clean reload), the CSV export, and the noted `error`/`errored` known-gap (crashed runs keep polling) |
| [`web/spectra.md`](?doc=web/spectra.md) | contract | The **spectra** surface — computing and reading a Raman vibrational spectrum: **two surfaces, one engine** (`lib/spectra/core.js`) — the standalone **Spectra tab** (`/spectrum-calculation`, compute→view: read-only MolView inspect card + auto-detect chemistry + schema form + `POST /api/spectra/render` builds a script) and the **Results-tab presenter** (view a `*.spectra.json`); the **chart** (frequency cm⁻¹ vs Raman activity Å⁴/amu · Lorentzian broadening · red imaginary modes · density-mode fallback), the sortable **mode table** + excited-state columns, **click-a-mode → VibrationView** 3D animation (a *different* viewer from the MolView inspect card), the **render vs load** doors, **2 s live polling**, and the current→target ESM note (engine + presenter still classic) |
| [`web/tabs.md`](?doc=web/tabs.md) | contract | **How the six tab pages compose the modules** (the consumer doc): the **tab roster + shared shell** (canonical `TABS` order, Molbuilder is the landing tab, the injected nav + projects sidebar); one section per page — **Molbuilder** (`/molbuilder`: MolView mounted by `selection-bootstrap`, ops via `applyOp`, and the **structure-generation in-gate** `loadIntoCanvas` — SMILES/name/DNA/RNA/peptide/upload), **Structure optimization** (read-only MolView + two schema forms + generate/save pipeline), **Transport** (a working TranSIESTA generator with a **current-vs-planned** box — the "placeholder" comments are stale; no in-app save yet), **Documents** (the `?doc=` reader that serves *these* docs); plus the two cross-cutting protocols — the **save out-gate** (`/api/structure/save`, server writes the `.xyz`+sidecar pair, 409 handshake) and the **data-coherence rule** (viewer-is-truth; labels validated against `n_atoms` server-side); ends on the tab-controller ESM posture |
| [`web/overview.md`](?doc=web/overview.md) | overview | **The web map** — start here: the **module-independence doctrine** (concealed, physically-separated, uniformly-reusable ES modules, with MolView as the exemplar — one module mounted read-only on 3 tabs + editable on 2), a **layers diagram** (tabs → modules → server), the **two universal patterns** (the `{ok,…}` envelope + the runtime registry), the **module registry** (every module → its own doc + a verified ESM status), the **ESM scorecard** (molview/vibrationview/workspace/projects/xyz-io full; trajectory/transport hybrid; presenters/spectra/results/form-schema/runtime classic → tasks #102/#103), and a "where to start reading" index by concern |
| [`ops/installation.md`](?doc=ops/installation.md) | guide | **Installing molbuilder + its science backends** — the **per-backend conda-env model** (host `molbuilder` + `molbuilder-siesta`/`-pySCF`/`-MDtools`/`-siesta-gpu`, dispatched via `conda run -n`, because the backends' pins are mutually unsatisfiable), the **one install path** (`scripts/install-env.sh bootstrap` → `molbuilder envs …`; run via `python -m molbuilder`, not pip-installed), the **backends table** (what lives where + how installed; X3DNA/pyberny manual), and the **GPU-SIESTA build appendix** (ELPA-tarball + SIESTA-cmake from source, the NFS-shmem hook, driver-optional). Reconciled to `recipes.py` — legacy-README drift corrected |
| [`ops/deployment.md`](?doc=ops/deployment.md) | guide | **Running the molbuilder web server** — `molbuilder serve` is a single-process **dev server** (no `--workers`; production = a reverse proxy in front terminating TLS + the **bind guard**: loopback-or-TLS); **opt-in SSO auth** (Google/GitHub/MS/ORCID/CAS — no passwords, an `allowed_users` allowlist, `secret_key_file` session key); the **security posture** (CSP `script-src 'self'`/no-inline-JS, HSTS-on-HTTPS, 50 MB cap, no CSRF token); the **rate limiter** (the threat model `web-api.md` defers here — 3 signals: attack-string signature · 404-storm · total-burst-off-by-default, 1 h cooldown, 429, admin routes); and **`molbuilder.json`** config (cwd-only). Reconciled to `cli.py`/`app.py`/`auth.py`/`rate_limit.py` |
| [`process/package-layout.md`](?doc=process/package-layout.md) | reference | **Where everything lives** — a domain-keyed map of the repo/package (model/science backends · engines · execution · web · ops), each area cross-linked to its owning doc; the top-level tree, the `molbuilder/` subpackages, the `web/` server+client (incl. every `static/lib/` module → its web doc), the flat-plus-subdir `tests/` layout, and packaging. Rewritten from the real tree (the legacy L1/L2/L3 folder story was stale) |
| [`process/conventions.md`](?doc=process/conventions.md) | reference | **How you write + invoke the code** — the **enforced** conventions first (L1/L2/L3 layering → `test_layering.py`; parse-layer no-I/O; pyflakes + full-suite **pre-commit**, no CI; `pythonpath` not `pip install -e`), each with its guard test; the module-provenance header flagged **advisory** (no guard); then the **CLI surface** — the thin-shell-over-the-web-API doctrine, the 15-command + 7-group catalogue (an index cross-linking execution/ops/tabs), and the exit-code / stdout-stderr / dataclass-`--help` conventions |
| [`process/testing.md`](?doc=process/testing.md) | reference | **How the project tests itself** — the **pyramid** (unit/module/interface/integration + smoke/e2e/slow markers), the **layering** structural invariant, **design-tests-around-the-envs**, testing the front-end **JS in Node without a browser** (the `_node_esm.py` harness that survives the ESM migration), the durable **Playwright** patterns (locate-what-the-user-clicks, state-based waits, viewport, `force=True` pitfalls, diagnostics), the source-text-invariant + state-composition patterns, and the pre-commit gate |
| [`process/code-audit.md`](?doc=process/code-audit.md) | reference | **The code-audit playbook** — how to run a systematic review: the audit **principles** (trust-but-verify, structural-audits-miss-behavioural-bugs, count-honestly, propagate-to-siblings, graduate-invariants-into-tests), the **dimensions**, the **five known traps** (`[hidden]`-precedence · per-feature option traversal · cross-source-of-truth gap · orphaned listener · same-shape-different-name endpoints) each with a DevTools smoking-gun diagnostic, and the per-dimension checklists — several invariants now enforced by source-text tests |
| [`MIGRATION.md`](?doc=MIGRATION.md) | index | The migration ledger: every old_docs file → target home + status |
| [`archive/README.md`](?doc=archive/README.md) | index | The archive's own index: what was archived when, and what superseded it |
