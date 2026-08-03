# molbuilder — documentation

**This is the ONE index.** Every document under `docs/` is listed here, one
line each. If a doc is not listed here it does not exist (test-enforced:
`tests/test_docs_structure.py`).

> **Migration complete (2026-07-28).** The previous docs tree was migrated to
> this domain-structured tree across Waves 0–9, each doc reconciled against
> the code at its move.  The old tree is archived under
> `archive/old_docs/` (see [`archive/README.md`](?doc=archive/README.md)).  The migration ledger —
> every source file → target home — is
> [`archive/MIGRATION.md`](?doc=archive/MIGRATION.md).

## Structure — domains, not document kinds

Documents are grouped by **domain** (the subsystem a reader works on), not
by kind of contract. Kind is expressed in the filename suffix and the
header, inside the domain.

| Folder | Domain |
|---|---|
| *(root)* | The spine: this index, `design.md` (mission · principles · decisions), `architecture.md` (the reuse map: task → tool), `roadmap.md` (THE one plan), [`audit-2026-07-28-document-migration.md`](?doc=audit-2026-07-28-document-migration.md) (migration closeout audit) |
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
  terms (the shared JSON vocabulary lives in
  [`model/overview.md`](?doc=model/overview.md) § 2 and
  [`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6) and must not collide
  across meanings — e.g. the run→next-calculation handoff is
  `execution/handoff-bundle.md`, never plain "bundle", which the JobSet
  framework owns. **Sub-documents share the master's filename as a prefix**, so
  the hierarchy is visible in the name itself: a master `structure.md` has subs
  `structure-periodicity.md`, `structure-annotations.md`, … (a filename prefix,
  not a subdirectory — the name alone shows the parent).
- **R6 — born here.** New documents are created in this structure only. (The
  legacy tree that predated it is archived under `archive/old_docs/`; the
  2026-07 migration that got us here is closed — see
  [`archive/MIGRATION.md`](?doc=archive/MIGRATION.md).)
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

## Reorganization protocol (any doc move or merge)

Moving or merging a doc is a **review gate**, not a file move. The editorial
rules above apply to every step in full:

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
   new path in the **same commit** (grep-verified, per file; no blind
   rewrite). `tests/test_no_retired_doc_paths.py` enforces that every
   `docs/**.md` path an active source cites exists on disk, so a move
   that strands references fails the suite.

*(This protocol ran the 2026-07 migration, whose migration-specific steps —
wave ordering, the freeze, keep-and-mark, the per-file ledger — are recorded
with the ledger at [`archive/MIGRATION.md`](?doc=archive/MIGRATION.md).)*

## Index

*(bottom-up by domain; the summary spine — `design.md`, `architecture.md`,
`backend-architecture.md` — landed last, in Wave 9)*

| Doc | Role | Owns |
|---|---|---|
| [`design.md`](?doc=design.md) | overview | The design north-star — mission, the assistant-not-nanny stance, the architecture in brief, the 8 load-bearing principles, the anti-patterns we refuse, and the decisions index (full log archived) |
| [`architecture.md`](?doc=architecture.md) | reference | The reuse map — **task → tool** + the subsystem index by layer (L1/L2/L3), every row routing to its authoritative domain doc; read before building anything |
| [`backend-architecture.md`](?doc=backend-architecture.md) | reference | The **same backend by functional concern** (data · construction · validation · execution) — which concern owns each module, the cross-concern pipeline, and where the concerns still leak into each other; the paired lens to `architecture.md` |
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
| [`web/molview-rework-plan.md`](?doc=web/molview-rework-plan.md) | plan | **How the MolView code is brought to its contract** (started 2026-07-30) — the target tree (28 files → 20, one sealed directory, one import, `3Dmol` in exactly one file), the eight phases bottom-up (one directory · the pure bottom · the master copy out of the renderEngine ★ · what a viewer holds · the doors and the read-only gate · the seal and the chrome · seal the entry + reconnect · the suite), and the rhythm: one unit at a time, its tests written from the contract, only those run — the suite stays out until the module is finished, and the breakage the rework causes is recorded, not chased. Retired at Phase 8 |
| [`web/molview-corrections-plan.md`](?doc=web/molview-corrections-plan.md) | plan | **What the finished MolView got wrong, and the agreed repair** (started 2026-07-31) — the review's findings against the contract and against the frozen tree, several of them features that were complete, correct and connected to nothing. One item is settled at a time (symptom · evidence · what the old code did · the code change · the document change · what stays open), so an item written down is ready to execute without re-deciding it. Retired when the last item lands |
| [`web/molview-css-namespace-plan.md`](?doc=web/molview-css-namespace-plan.md) | plan | **One prefix for MolView's stylesheet** (started 2026-08-01) — the module publishes 167 class names under 9 prefixes, 58 of them (35%) also defined outside it and 46 shared with the 3Dmol embed it is supposed to be independent of, so whichever sheet loads last styles the card. Everything becomes `molviewer-<area>-<part>`, spelled out rather than abbreviated, over the parts § 8.1 already names. Four phases ordered by risk (retire the dead frozen sheet · the private areas · the embed-shared names · the global ones), a guard test last, and the browser rather than the suite as the check — a stylesheet rename is invisible to stubs. Retired when the last phase lands |
| [`web/molview-integration-plan.md`](?doc=web/molview-integration-plan.md) | plan | **Wiring the tabs to the finished module** (started 2026-08-02) — the rework plan left everything outside MolView broken on purpose, and this is that bill: thirteen files still read a `window.molbuilder.molview` the module stopped publishing, `/molbuilder` never mounts a viewer at all, and the modify tab's cell editor asks for three names that are on no MolView surface plus an `isDefault` field that isn't there — every call guarded, so it renders "(default)" on every row instead of failing. The contract is the standard, so there is no design step and no adapter: § 9.3's sixteen needs table already says which door answers each need, the old names cease to exist rather than being mapped (`factsForRequest`, `isDirty`/`markSaved`, `getSource`, the `*Info` family), one owner per page mounts and hands the handle down, and five pages done whole, easiest first, each checked in a browser. Retired when `tabs.md` § 8 describes the finished state |
| [`web/css-system-plan.md`](?doc=web/css-system-plan.md) | plan | **Reconciling the tab CSS system** (proposed 2026-08-02) — 21 selectors are defined in two or more files, and the worst are not components but the document itself: `body` and `.status` live in `page-shell` **and** three page sheets, `html` in three, `*`/`header`/`.card` in three, so which one wins is decided by `<link>` order and no page states that it means to. The existing duplicate-selector guard cannot see it — it skips element-only selectors on purpose, which is exactly how the document tier drifted across four sheets while the suite stayed green. **The boundary comes first: a module owns its CSS the way it owns its JavaScript**, so `molview.css`, the sidebar, the trajectory and the inspectors are out of scope, and a page rule reaching at `molviewer-*` is deleted rather than edited. Four tiers (tokens · document · components · page vocabulary), one enforceable sentence — *a page sheet may contain only its own namespaced classes* — with two escapes so it stays liveable: scope under the page's own root, or promote a shared difference to a named variant. Ordered so the document tier is settled before any page's spacing is measured against it, `modify` already done as the worked example, and the guards last. Retired when the allowlist is empty and the three guards hold |
| [`web/modify-persistency-investigation.md`](?doc=web/modify-persistency-investigation.md) | investigation | **What the Molbuilder tab keeps, and what it drops** (2026-08-03) — molview.md § 11.3 names four things that look like saving; this tab has **six**. The fifth is *Save to project*, which is Export → Data arrived at from the other side — same pair of files, same `exportFile`, different door and a different destination — and the contract names only one of the two. The sixth is the tab's **own context**: which file is on the canvas, where it last saved, which panel was open — all in closure variables, **none of it kept**, because nothing under `modify/` writes to the workspace at all. That is not untidiness: after a genuine restore `_loadedFrom` is empty while a structure is plainly on screen, so the readout falls to *Picked:* and **the Load button re-enables against the file, inviting the user to discard the work just restored for them** — and a restore and a silent re-read of the file read identically on screen though one preserved unsaved edits and the other threw them away. § 11.2's own test (*state is the truth; what you are looking at is not*) sorts it: keep the file and the save target, drop the open panel and the form fields. The door already exists — a tag of the tab's own, exactly as `inspectors/structure.js` already does for `SHOWING_TAG` one directory away |
| [`web/vibrationview.md`](?doc=web/vibrationview.md) | contract | The **VibrationView** module — the concealed viewer that **animates a vibrational normal mode** (a *sibling* of MolView, mounted only by the spectra viewer): the `mount()` door + handle (`showMode`/`play`/`pause`/`setAmplitude`/`setSpeed`/`dispose`), the animation model (`pos = eq + amp·cos(φ)·disp`, live amplitude/speed, greyed frozen atoms), the eigenvector scatter (`mode-math.js`), the **semantic seal** (owns the clock/knobs/tick-math, drives a drawing surface through generic doors — not a second 3Dmol wrapper), spectra-tab wiring, and the **current → target** note: it still *borrows* MolView's shared embed via the transitional `molbuilder.viewer` global — full separation (own concealed seal + `lib/viewer/`→molview) is **task #104** |
| [`web/workspace.md`](?doc=web/workspace.md) | contract | The session-persistence module — saves a tab's in-progress work so a reload or Undo brings it back: **one place**, files on the server under the project directory, written by one `persist()` call through the front door `dispatcher.js` (a second copy in the browser's own storage was removed — nothing ever restored from it, so it cost a write per edit and bought nothing). The **tag** — how several savers share one page, and the id that is remembered per tag; the public surface; **§ 2a what it is NOT**, whose first row is the boundary that has actually cost time: the **timeline is MolView's** (`lib/molview/history.js`, molview.md § 11.2) and this module has no idea a sequence exists — it stores numbered *states* and never compares two indices. The server half is `/api/workspace-storage/*` in `workspace_storage.py`, renamed 2026-08-02 from a name that said *timeline*. A *file* save is the Modify Save panel + projects door, not the workspace |
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
| [`ops/server-reload-plan.md`](?doc=ops/server-reload-plan.md) | plan | **Seeing a code change without a manual restart** (2026-08-03; **complete — A–E landed**; shipped behaviour is in [`deployment.md`](?doc=ops/deployment.md) § 1 and § 4) — two halves that must not get one answer. **JS/CSS were never stale on the server**: Flask cached them in the BROWSER for 12 hours with no check, which is the whole reason a front-end change looked like it needed a restart. Now they revalidate (`no-cache` → 304, no body) — **not** the URL-version this plan first proposed, which reaches only the 119 `url_for` references and would have left the **51 ESM imports written inside the JavaScript** on cached copies. **Python** still needs a fresh interpreter, and a file watcher is rejected on purpose: Werkzeug stat-polls every imported module each second and fires on *any* mtime change, so a chunked write or a `git checkout` reloads against a half-written tree. What is kept from the reloader is the **process shape** — `serve --supervise`, a parent that never imports application code and respawns a child on a sentinel exit, which a route re-execing itself cannot give itself. **The gate, decided:** `admin_emails` ships empty meaning *any logged-in session is admin*, so this route **inverts that default** — with no supervisor or no named admins it is **absent (404), not refused (403)**, making a misconfiguration read as *the button is missing*. That one key now gating two subsystems is filed as task #49 |
| [`ops/deployment.md`](?doc=ops/deployment.md) | guide | **Running the molbuilder web server** — `molbuilder serve` is a single-process **dev server** (no `--workers`; production = a reverse proxy in front terminating TLS + the **bind guard**: loopback-or-TLS); **opt-in SSO auth** (Google/GitHub/MS/ORCID/CAS — no passwords, an `allowed_users` allowlist, `secret_key_file` session key); the **security posture** (CSP `script-src 'self'`/no-inline-JS, HSTS-on-HTTPS, 50 MB cap, no CSRF token); the **rate limiter** (the threat model `web-api.md` defers here — 3 signals: attack-string signature · 404-storm · total-burst-off-by-default, 1 h cooldown, 429, admin routes); and **`molbuilder.json`** config (cwd-only). Reconciled to `cli.py`/`app.py`/`auth.py`/`rate_limit.py` |
| [`process/package-layout.md`](?doc=process/package-layout.md) | reference | **Where everything lives** — a domain-keyed map of the repo/package (model/science backends · engines · execution · web · ops), each area cross-linked to its owning doc; the top-level tree, the `molbuilder/` subpackages, the `web/` server+client (incl. every `static/lib/` module → its web doc), the flat-plus-subdir `tests/` layout, and packaging. Rewritten from the real tree (the legacy L1/L2/L3 folder story was stale) |
| [`process/conventions.md`](?doc=process/conventions.md) | reference | **How you write + invoke the code** — the **enforced** conventions first (L1/L2/L3 layering → `test_layering.py`; parse-layer no-I/O; pyflakes + full-suite **pre-commit**, no CI; `pythonpath` not `pip install -e`), each with its guard test; the module-provenance header flagged **advisory** (no guard); then the **CLI surface** — the thin-shell-over-the-web-API doctrine, the 15-command + 7-group catalogue (an index cross-linking execution/ops/tabs), and the exit-code / stdout-stderr / dataclass-`--help` conventions |
| [`process/testing.md`](?doc=process/testing.md) | reference | **How the project tests itself** — the **pyramid** (unit/module/interface/integration + smoke/e2e/slow markers), the **layering** structural invariant, **design-tests-around-the-envs**, testing the front-end **JS in Node without a browser** (the `_node_esm.py` harness that survives the ESM migration), the durable **Playwright** patterns (locate-what-the-user-clicks, state-based waits, viewport, `force=True` pitfalls, diagnostics), the source-text-invariant + state-composition patterns, and the pre-commit gate |
| [`process/code-audit.md`](?doc=process/code-audit.md) | reference | **The code-audit playbook** — how to run a systematic review: the audit **principles** (trust-but-verify, structural-audits-miss-behavioural-bugs, count-honestly, propagate-to-siblings, graduate-invariants-into-tests), the **dimensions**, the **five known traps** (`[hidden]`-precedence · per-feature option traversal · cross-source-of-truth gap · orphaned listener · same-shape-different-name endpoints) each with a DevTools smoking-gun diagnostic, and the per-dimension checklists — several invariants now enforced by source-text tests |
| [`process/screenshots.md`](?doc=process/screenshots.md) | process | **The README screenshot manifest** — the capture guide for the 10 PNGs in `img/`: the one-project demo convention (`projects/BDT/`, so the README reads as one continuous Au–BDT–Au story), the pre-capture setup (window size, theme, no DevTools), a per-image table (URL · what to load · zoom region · what it must communicate), and the re-capture cadence. Carries the known-stale flag on the two nav shots (five tabs captured; six ship) |
| [`archive/MIGRATION.md`](?doc=archive/MIGRATION.md) | index | The migration ledger: every old_docs file → target home + status (closed — migration complete 2026-07-28) |
| [`archive/README.md`](?doc=archive/README.md) | index | The archive's own index: what was archived when, and what superseded it |
