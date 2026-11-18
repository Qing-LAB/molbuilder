# molbuilder — documentation

**This is the ONE index.** Every document under `docs/` is listed here, one
line each. If a doc is not listed here it does not exist. **That is a review
rule, not a gate.**

> **New here?** Start at
> [`ops/getting-started.md`](?doc=ops/getting-started.md) — from a fresh
> clone to a first calculation: which config files must exist on which
> machine (local, cluster, notifications, sign-in), in what order, each
> step handed to the document that owns it.

## Two words this repository uses in a specific way

Both come up constantly — and neither is standard vocabulary, so they are
glossed here rather than only where they are defined.

| word | what it means here | who enforces it |
|---|---|---|
| **floor** | **which ROLE owns a decision.** Numbered 1 (names & plain facts) to 7 (surfaces). *A floor may call down and return up; it may never reach across.* | review — `process/code-audit.md` § 1c (e), against `execution/architecture.md` § 2.1 (A7) |
| **layer** / `L1`–`L3` | **import depth** — which file may `import` which. A different axis. | review — `process/code-audit.md` § 1c (e) |

**They overlap without matching**, which is exactly why two words are needed:
`jobset` is one import tier (`L2`) and spans **six** floors (1 and 3–7). Reusing "layer"
for both is the confusion the split exists to prevent.

**Why "floor" and not "level" or "tier":** it is a storey of a building, and the
metaphor carries the rule. You can go *down* and come back with an answer; you
cannot go *up*, and you cannot cut sideways through a wall. A conventional word
would not have said the direction, and "layer" was already taken.

**It is not the same as the order things happen in.** `prep` walks floors 1→4
and is a *route*, not a floor: a route crosses floors, and each of its steps
is answered by the floor that owns it.

Both are defined in full in
[`execution/architecture.md`](?doc=execution/architecture.md) § 1 — **the owner
of both terms**; the floors and their modules are its § 2.1, which a module
docstring's *"Floor N"* names.

## Structure — domains, not document kinds

Documents are grouped by **domain** (the subsystem a reader works on), not
by kind of contract. Kind is expressed in the filename suffix and the
header, inside the domain.

| Folder | Domain |
|---|---|
| *(root)* | The spine: this index, `design.md` (mission · principles · decisions), `architecture.md` (the reuse map: task → tool), `workflow.md`, `configuration.md`, `backend-architecture.md`. **The one plan is [`plans/plan.md`](?doc=plans/plan.md)** — rule R3 |
| `model/` | The data model (L1): Structure, periodicity, annotations, codecs & load/save, sidecars, region labels, selection grammar, chemistry, the parse stack, the data vocabulary |
| `science/` | Scientific correctness: validation machinery, chemistry correctness, pseudopotential standards, parameter tuning |
| `engines/` | Per-engine emitter specs (SIESTA, PySCF, transport/TranSIESTA, builders, GPU build recipe) **and the cross-engine contracts**: tier values, what a stage is, and the template every engine's parameters travel in |
| `execution/` | Running calculations: the jobset protocol (describe, prep, launch, status), the job request, the folder on disk and the file formats, the run script at run time, runs of points, reports, saved history, the scheduler and GPUs, the benchmark — routed by `execution/overview.md` |
| `web/` | The whole front end: tabs, MolView, workspace, projects sidebar, results/inspectors, forms, UI/CSS contract, web API |
| `ops/` | Installing and serving the app: install model, deployment, auth/rate-limit |
| `process/` | How we work: code conventions, test strategy, audit playbook, CLI conventions, package layout |
| `plans/` | **One file: [`plans/plan.md`](?doc=plans/plan.md)** *(user: "We don't need ten plan files scattered")*; R3 names it |
| `archive/` | History, date-prefixed. **Not a source of truth.** |

## The rules

- **R1 — one index, never lagging.** Every non-archive doc has exactly one
  line here, added **in the same commit** that adds, moves, merges, or
  archives the doc. The master index is updated at EVERY step — it is never
  allowed to drift from the tree, and review looks for unindexed and dangling
  entries (an index kept as a separate chore rots).
- **R2 — provenance header.** Every doc starts with a header block naming
  its **Role** (`contract` | `guide` | `overview` | `plan` | `process` |
  `investigation` | `audit report` | `reference` | `record` | `index` — a
  doc may carry two, e.g. `guide and contract`; `reference` is a **snapshot
  of an external fact** like a
  facility's topology: consulted, never binding, no number from it
  hard-coded; `investigation` is a dated question answered against the
  code, and `audit report` a dated systematic review), its
  **Domain** (folder), and its **Companions** (linked related docs). A
  contract is the sole source of truth for its surface; a guide explains it
  in plain language and never contradicts it.
- **R3 — contracts don't hold plans, and there is exactly one plan.** Durable
  decisions stay in contracts; every open item lives in
  [`plans/plan.md`](?doc=plans/plan.md) and nowhere else.
  **A contract states the rule and points at the plan for status** — one
  pointer, and it names [`plans/plan.md`](?doc=plans/plan.md), never the
  archive.
- **R4 — one archive.** Superseded content moves to `archive/` with a
  `YYYY-MM-DD-` prefix; the archive README says what superseded it.
- **R5 — names carry the vocabulary.** File names use the system's canonical
  terms (the shared JSON vocabulary lives in
  [`model/overview.md`](?doc=model/overview.md) § 2 and
  [`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6) and must not collide
  across meanings — e.g. a calculation folder is a *calculation*, never a
  "bundle", though the CLI still addresses it as `--bundle`
  ([`execution/project-layout.md`](?doc=execution/project-layout.md) § 1.0). **Sub-documents share the master's filename as a prefix**, so
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
  The link form stands on review like the rest of this page.
  **No `#` heading-anchor links.** For a same-doc section pointer use plain
  text ("see § 8"), not `[…](#…)`.

### The doc rule (carried over — still the point of all of this)

> Tests must be derivable from the spec without reading the implementation.
> Code reviews must verify code matches spec, not code matches reviewer's
> expectations.

## Editorial rules — how documents are written and merged

These bind every write to this tree.

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
4. Extract any plan/status content into [`plans/plan.md`](?doc=plans/plan.md) (R3).
5. Rewrite where needed: plain language (E3), Mermaid diagrams where they
   explain (E2), scientific foundations preserved + enriched (E4).
6. Add the provenance header (R2) and the index line here (R1) — same
   commit, always.
7. Repoint inbound references — other docs, code comments, tests — to the
   new path in the **same commit** (grep-verified, per file; no blind
   rewrite). **Nothing enforces this, by choice**: a test that asserts the
   repository's shape rather than a result is not kept. So grep-verify a move
   in the same commit.

## Index

*(bottom-up by domain)* — one row per document. **Where a thing lives** — a
route, a module, a file a calculation writes, a constant or a citation — is
answered by the four indexes of the Documents tab's **Indexes** folder, built
from the code each time one is read ([`web/tabs.md`](?doc=web/tabs.md) § 5.6).

| Doc | Role | Owns |
|---|---|---|
| [`README.md`](?doc=README.md) | index | This index: the folder layout, the rules every document follows, how documents are written and moved, and one row per document. |
| [`workflow.md`](?doc=workflow.md) | contract | How the parts fit together: shared vocabulary, who reads and writes each file, step order, where answers enter, the whole road. |
| [`configuration.md`](?doc=configuration.md) | contract | Which files are configuration, who writes each and how, where each is looked for, every `molbuilder.json` key, and the machine-record rules. |
| [`design.md`](?doc=design.md) | overview | The mission, the stance toward the person, the architecture in brief, the load-bearing principles, refused anti-patterns, and the durable decisions. |
| [`architecture.md`](?doc=architecture.md) | reference | The reuse map: which existing tool serves each task, and every subsystem by import layer, each pointing to its owning document. |
| [`backend-architecture.md`](?doc=backend-architecture.md) | reference | The backend by functional concern: which concern owns each module, how the concerns chain, and where they leak into each other. |
| [`archive/2026-09-01-roadmap.md`](?doc=archive/2026-09-01-roadmap.md) | plan *(archived 2026-09-01 — superseded by [`plans/plan.md`](?doc=plans/plan.md); a record, not a source of truth)* | The old roadmap, the plan's home before there was one plan; its open items went into plans/plan.md. |
| [`model/overview.md`](?doc=model/overview.md) | overview | Start-here map of the model documents, and the atom-index convention: how atoms are counted in code, files, messages, display, engines. |
| [`model/structure.md`](?doc=model/structure.md) | contract | The `Structure` object: how it is serialized and sent, geometry files, and the load and save doors on server and browser. |
| [`model/structure-periodicity.md`](?doc=model/structure-periodicity.md) | contract | Per-axis box behaviour: the cell, where atoms sit in it, each axis's kind, vacuum padding, and how the box is made. |
| [`archive/2026-08-20-cell-plan.md`](?doc=archive/2026-08-20-cell-plan.md) | plan *(archived 2026-08-20 — all acceptance rows landed; live rules: `structure-periodicity.md` § 6.1–6.2, `molview.md` § 9.5)* | The plan that reconciled the unit-cell code and Cell page with the periodicity model; its rules live in model/structure-periodicity.md. |
| [`archive/2026-10-08-plan-consolidation.md`](?doc=archive/2026-10-08-plan-consolidation.md) | plan *(archive — the sections of `plans/plan.md` closed 2026-10-08, each with its verdict)* | The plan sections closed on 2026-10-08, each under its validation verdict, and the transport programme's record; plans/plan.md replaced them. |
| [`archive/2026-10-09-plan-consolidation.md`](?doc=archive/2026-10-09-plan-consolidation.md) | plan *(archive — the sections of `plans/plan.md` closed 2026-10-09, each with its verdict)* | The plan sections and rows closed on 2026-10-09, each under its verdict; plans/plan.md keeps what stayed open. |
| [`model/structure-annotations.md`](?doc=model/structure-annotations.md) | contract | Per-atom labels and values: the channel model, how they are saved, how they become engine input, and which labels mean electrodes. |
| [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md) | contract | The `.molstruct.json` file beside a saved structure: its layout, schema versions, the geometry check on reading, its pairing with the `.xyz`. |
| [`model/chemistry.md`](?doc=model/chemistry.md) | contract | Chemistry helpers on a `Structure`: the phosphate charge rule, protonation, adding hydrogens, species labels and order, clash relief, and the dipole. |
| [`model/datafiles.md`](?doc=model/datafiles.md) | contract | The framework for every NetCDF results file: self-describing values, writing, reading, tables and exports, and the catalogue of every name held. |
| [`model/parse.md`](?doc=model/parse.md) | contract | Reading output files into typed results: parser base class, result types, registry, a run's record and ending, adding a parser. |
| [`science/overview.md`](?doc=science/overview.md) | overview | The science domain's start-here map, when a validation finding blocks, and the catalogue of checks the validation pass runs. |
| [`science/validation.md`](?doc=science/validation.md) | contract | The validation machinery: a structure's chemistry facts, its one electronic state, and the checks, deck writers and card that read them. |
| [`science/normal-modes.md`](?doc=science/normal-modes.md) | contract | How many of a vibrational calculation's numbers are vibrations when atoms are held, and how the rest are removed. |
| [`science/refs/README.md`](?doc=science/refs/README.md) | index | The reference articles held behind the normal-modes contract, those not held, and what each claim rests on. |
| [`science/vibrational-averaging.md`](?doc=science/vibrational-averaging.md) | science reference | Averaging a junction's transmission over one vibrational mode's motion, thermal or light-driven, from a few frames, and its assumptions. |
| [`science/vibrational-optical-response.md`](?doc=science/vibrational-optical-response.md) | science reference | The method for a SIESTA vibration's infrared and Raman response to light along a chosen direction, its checks and assumptions. |
| [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) | contract | Where a calculation's chemistry can go wrong, the charge-and-spin science, the electronic state's rules, each species' needs, and the audit checklist. |
| [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) | contract | What molbuilder checks in a set of .psml pseudopotentials, why, at which severity, and what it does not check. |
| [`archive/2026-09-08-test-design-findings.md`](?doc=archive/2026-09-08-test-design-findings.md) | record | The 2026-09-08 audit of scientific tests that could pass for a physically wrong reason; its rule now lives in process/testing.md. |
| [`science/junction-cell.md`](?doc=science/junction-cell.md) | contract | How long a junction's or slab's box must be along z, and the stacking and registry its boundary must keep. |
| [`engines/siesta.md`](?doc=engines/siesta.md) | contract | The SIESTA .fdf deck writer: its sections, the charge, spin, cell, k-grid and eigensolver rules, and the files beside it. |
| [`engines/pyscf.md`](?doc=engines/pyscf.md) | contract | The PySCF script writer: its output files, its run rules, the shared progress-log format, and the publication-quality parameter guide. |
| [`engines/vibration.md`](?doc=engines/vibration.md) | contract | The vibration calculation on PySCF and SIESTA: what each computes, the road to its result file, its parameters, invariants and tests. |
| [`engines/vibration-postprocessing.md`](?doc=engines/vibration-postprocessing.md) | guide | Reading a finished vibration's results in Python — modes, infrared, Raman, thermochemistry, charges, optical calls — with worked examples. |
| [`engines/transport.md`](?doc=engines/transport.md) | contract | The transport calculation: stages derived from one cited junction, the physics, electrodes from region labels, the consistency invariants, the baseline. |
| [`engines/transport-postprocessing.md`](?doc=engines/transport-postprocessing.md) | guide | Reading a transport calculation's results in Python — transmission, conductance, I–V, DOS, convergence, atom indices — with worked examples. |
| [`engines/builders.md`](?doc=engines/builders.md) | contract | Building a 3-D structure from a sequence, SMILES or name: notation, nucleic-acid backends, clash and hydrogen controls, CLI and Python. |
| [`engines/tuning.md`](?doc=engines/tuning.md) | contract | What value each convergence and quality setting carries per tier, the shipped ladders, the cross-engine map, and their references. |
| [`engines/optimization-postprocessing.md`](?doc=engines/optimization-postprocessing.md) | guide | Reading an optimization's results in Python — finding it, its steps, energies, forces, SCF, final geometry, exported file — with worked examples. |
| [`engines/stages.md`](?doc=engines/stages.md) | contract | What a stage is, how its settings resolve, where each field lands, and the task.json file that describes the stages. |
| [`engines/template.md`](?doc=engines/template.md) | contract | A calculation's parameter file (the template): its format, each item's fields and owning layer, and rendering a deck from it. |
| [`engines/overview.md`](?doc=engines/overview.md) | overview | The engines map and its shared rules: deck blocks, held atoms and labels, the GPU choice, what ladders share, adding an engine. |
| [`execution/overview.md`](?doc=execution/overview.md) | overview | The way into the execution chapters: a calculation's steps, each chapter found by question, and every chapter's role. |
| [`execution/project-layout.md`](?doc=execution/project-layout.md) | contract | The calculation folder on disk: its levels, who creates each folder, a run's record files, every file's manifest, the invariants. |
| [`execution/job-contracts.md`](?doc=execution/job-contracts.md) | contract | Every format molbuilder writes and its name: the file catalogue, name grammar, run-script and deck blocks, warm files, persisted-file registry. |
| [`execution/architecture.md`](?doc=execution/architecture.md) | contract | Who decides what in the execution code: the floors, one door per fact, the declarations, the routes, where launch values are fixed. |
| [`execution/generator.md`](?doc=execution/generator.md) | contract | The benchmark: a run as a set of one, what bounds a sweep, its grid and trials, choosing the winner, its result file. |
| [`execution/script-preparation.md`](?doc=execution/script-preparation.md) | contract | Prep's steps to a launchable folder and why their order is forced; the questions an engine answers; the pipeline log. |
| [`archive/2026-08-18-preparation-backend-plan.md`](?doc=archive/2026-08-18-preparation-backend-plan.md) | plan | The plan that rebuilt prep's deck-writing step and moved both engines onto it; execution/script-preparation.md holds the rules. |
| [`execution/asu-sol.md`](?doc=execution/asu-sol.md) | reference | A dated snapshot of ASU Sol's published hardware, partitions and QoS, and the Sol examples the design was drawn against. |
| [`execution/checkpointing.md`](?doc=execution/checkpointing.md) | guide and contract | A calculation folder's saved history: the checkpoint commands and panel, when molbuilder saves, the rules a save and restore keep. |
| [`execution/preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md) | guide | Preparing a calculation here for another machine's record, what travels with it, and what to do when that machine changes. |
| [`execution/scheduler.md`](?doc=execution/scheduler.md) | contract | What a machine offers and where a job goes: the queue model, the fit, placement, the header and send line. |
| [`execution/submission.md`](?doc=execution/submission.md) | contract | The sending rules: what a person sees and is asked before a job goes, and how jobs reach a scheduler. |
| [`execution/gpu.md`](?doc=execution/gpu.md) | contract | How a GPU decision travels: whose question each GPU choice is, every GPU name, the rules, one calculation walked through. |
| [`execution/running-a-job.md`](?doc=execution/running-a-job.md) | contract | What a generated run script does on the node from start to end, and how a run's state is read back. |
| [`execution/runs-of-points.md`](?doc=execution/runs-of-points.md) | contract | One run holding many engine runs: its words and data, operations, the run's state, when a point reports, count and pace. |
| [`execution/run-reports.md`](?doc=execution/run-reports.md) | contract | How a running job reports: the monitor beside each run, its events and senders, the channels file, report cards, the listener. |
| [`execution/run-identity.md`](?doc=execution/run-identity.md) | contract | Whether a calculation continues: the label and its normalisation, the run id and what it is tied to, engines' restart groups. |
| [`execution/worked-example.md`](?doc=execution/worked-example.md) | guide | One two-stage calculation walked end to end, from describing it to saving and going back, on a workstation and cluster. |
| [`execution/job-system.md`](?doc=execution/job-system.md) | contract | The jobset protocol: its words, groups and their states, the next step, prep and launch, every verb and every refusal. |
| [`execution/job-request.md`](?doc=execution/job-request.md) | contract | job-request.json, how and where a calculation runs: every key, which stages share a job, what may change after launch, migrating it. |
| [`archive/2026-08-19-template-unification-plan.md`](?doc=archive/2026-08-19-template-unification-plan.md) | plan | The plan that made one template file describe a calculation for every engine; engines/template.md is its contract. |
| [`archive/2026-08-19-staged-runs-implementation-plan.md`](?doc=archive/2026-08-19-staged-runs-implementation-plan.md) | plan | The milestone-by-milestone build order of the staged-run system, now built; the execution contracts hold the design. |
| [`web/molview.md`](?doc=web/molview.md) | contract | The 3D structure viewer every tab embeds: what it holds, its layers and doors, how a frame is drawn, and its tests. |
| [`archive/2026-08-20-molview-css-namespace-plan.md`](?doc=archive/2026-08-20-molview-css-namespace-plan.md) | plan *(archived 2026-08-20 — all phases landed, guard test green; live rules: `molview.md` § 4/§ 8.1)* | The plan that gave MolView's stylesheet one class prefix, delivered; the rule lives in web/ui-contract.md. |
| [`archive/2026-08-20-molview-and-checkpoint-ui-plan.md`](?doc=archive/2026-08-20-molview-and-checkpoint-ui-plan.md) | plan *(archived 2026-08-20 — delivered; its walk-throughs, W12, were dropped on 2026-09-10 as a list of things to click rather than a defect)* | The plan for the checkpoint view, the viewer's kept view and a working Export, delivered; web/molview.md and web/projects.md hold the rules. |
| [`archive/2026-08-21-vibration-parameter-integration-plan.md`](?doc=archive/2026-08-21-vibration-parameter-integration-plan.md) | plan | The plan that made each shown vibration parameter reach the deck or be refused by name; engines/template.md and engines/vibration.md replaced it. |
| [`archive/2026-08-20-spectra-migration-plan.md`](?doc=archive/2026-08-20-spectra-migration-plan.md) | plan | The plan that made a vibrational calculation an ordinary described job, delivered; engines/vibration.md is its contract. |
| [`archive/2026-08-19-task-setup-tab-plan.md`](?doc=archive/2026-08-19-task-setup-tab-plan.md) | plan | The Task-setup tab's build plan and the two wrong-name mistakes it recorded; web/task-setup.md is the tab's contract. |
| [`archive/2026-09-21-modify-persistency-investigation.md`](?doc=archive/2026-09-21-modify-persistency-investigation.md) | investigation *(archived 2026-09-21 — its work is built and its last suggestion was already resolved; nothing outstanding)* | What the Molbuilder tab keeps across a reload, answered and built; the rule lives in web/tabs.md and web/workspace.md. |
| [`archive/2026-09-21-audit-2026-08-05-tab-ui.md`](?doc=archive/2026-09-21-audit-2026-08-05-tab-ui.md) | audit report *(archived 2026-09-21 — its open findings are `plans/plan.md` § 2 W1–W4 and W13)* | A static audit of every tab's UI on 2026-08-05, kept for its evidence; its surviving findings went into plans/plan.md. |
| [`web/vibrationview.md`](?doc=web/vibrationview.md) | contract | The normal-mode animator: its one door, where amplitude and the held-still atoms are decided, frame-based playback, and animation export. |
| [`plans/plan.md`](?doc=plans/plan.md) | plan | Every open item: where work stands, releases to the finish line, decisions taken, what comes after, where archived sections went. |
| [`web/spectrumchart.md`](?doc=web/spectrumchart.md) | contract | The spectrum chart: drawing the modes, mirroring the page's selection, turning a click into a mode, and its own stylesheet. |
| [`web/workspace.md`](?doc=web/workspace.md) | contract | Keeping a tab's unsaved work on the server so a reload or Undo brings it back, and what it never does. |
| [`web/projects.md`](?doc=web/projects.md) | contract | The projects sidebar: browsing and picking files, the byte-level file layer, and the one door that opens and saves structures. |
| [`web/presenters.md`](?doc=web/presenters.md) | contract | The Results tab's viewer registry: which viewer opens which file, the contract a viewer meets, and how to add one. |
| [`web/runtime.md`](?doc=web/runtime.md) | contract | How browser modules find each other whatever order they load in, and the catalogue of small helpers every page shares. |
| [`web/notifications.md`](?doc=web/notifications.md) | contract | The app-wide notification bar: its calls, where messages appear, and the one message source built into it. |
| [`web/form-schema.md`](?doc=web/form-schema.md) | contract | How every engine-option form is generated from the parameter catalogue: the round trip, the calls, field kinds and grouping. |
| [`web/handover-procedure.md`](?doc=web/handover-procedure.md) | contract | How a parameter tab hands its work to Task setup on disk: the steps, the hand-over file, and the cell gate. |
| [`web/task-setup.md`](?doc=web/task-setup.md) | contract | The Task setup tab: where it saves, the stage table, the measure and run cards, and the files it writes. |
| [`web/this-machine.md`](?doc=web/this-machine.md) | contract | The This machine tab: the machine's notification channels, its report listener, and how their secrets are kept. |
| [`web/jupyter.md`](?doc=web/jupyter.md) | contract | The JupyterLab tab: reaching it through molbuilder's own port, starting and stopping it, its controls, what it exposes, its env. |
| [`web/web-api.md`](?doc=web/web-api.md) | contract | The server's HTTP routes: the answer envelope, status codes, security settings, every route by domain, and the shapes no module owns. |
| [`web/ui-contract.md`](?doc=web/ui-contract.md) | contract | How every page looks and behaves as one app: stylesheet layers, colour tokens, responsive layout, dialogs, and reading a server answer. |
| [`web/results.md`](?doc=web/results.md) | contract | The Results tab: picking a folder's file, choosing and mounting its viewer, the Run panel, exports, following a live run. |
| [`web/trajectory.md`](?doc=web/trajectory.md) | contract | The optimization-run viewer: feeding frames to the 3D viewer, the convergence plots and targets, run status, live updates, exports. |
| [`web/spectra.md`](?doc=web/spectra.md) | contract | Describing a vibrational calculation and reading its spectrum: the chart, mode views, how big a mode is drawn, and the routes. |
| [`web/bench-summary.md`](?doc=web/bench-summary.md) | contract | A benchmark sweep on one page while it runs: the comparison charts, a card per trial, and where each figure comes from. |
| [`web/tabs.md`](?doc=web/tabs.md) | contract | How the tab pages are put together: the tab list, each page's wiring, the Documents reader and indexes, saving, label checks. |
| [`web/overview.md`](?doc=web/overview.md) | overview | The web map: the module rules, the layers, every module with its document and conversion status, and where to start. |
| [`ops/getting-started.md`](?doc=ops/getting-started.md) | guide | From a fresh clone to a first calculation: every setup file, which machine, in what order, each step's owning document. |
| [`ops/installation.md`](?doc=ops/installation.md) | guide | Installing molbuilder and its engines: the environment model, the one install command, which backend lives where, building the GPU SIESTA. |
| [`ops/env-framework.md`](?doc=ops/env-framework.md) | contract | The machinery under installation: the environment registry, recipes as data, the install pipeline and its order, audit, repair, the shell shim. |
| [`ops/access-control.md`](?doc=ops/access-control.md) | contract | Who reaches the server and what they may do: each gate and its question, the rules beneath them, the known sharp edges. |
| [`archive/2026-08-19-server-reload-plan.md`](?doc=archive/2026-08-19-server-reload-plan.md) | plan | How a code change reaches the running server without a manual restart, delivered; ops/deployment.md and ops/access-control.md hold the behaviour. |
| [`ops/deployment.md`](?doc=ops/deployment.md) | guide | Running the web server: the serve commands, the production shape behind a proxy, sign-in, security posture, rate limiter, molbuilder.json. |
| [`process/package-layout.md`](?doc=process/package-layout.md) | reference | Where everything lives: the repository and package mapped by domain, each area linked to its owning document, the tests' layout, packaging. |
| [`process/conventions.md`](?doc=process/conventions.md) | reference | How code is written and invoked: the conventions and what holds each, when the version number moves, the command line's conventions. |
| [`process/changelog.md`](?doc=process/changelog.md) | record | What each version changed in the files molbuilder writes, one dated line per version, newest first. |
| [`process/testing.md`](?doc=process/testing.md) | reference | How the project tests itself: tiers and layers, what earns a test, JavaScript and browser tests, the runner, the commit gate. |
| [`process/code-audit.md`](?doc=process/code-audit.md) | reference | How to run a systematic code review: principles, what review carries that tests cannot, the deletion protocol, known traps, checklists. |
| [`archive/2026-09-08-test-audit-findings.md`](?doc=archive/2026-09-08-test-audit-findings.md) | record | The non-science half of the 2026-09-08 test audit, each finding with its reasoning; its open rows went into plans/plan.md. |
| [`process/screenshots.md`](?doc=process/screenshots.md) | process | The README screenshots: how each image in docs/img is captured, the demo project they share, and when to re-capture. |
| [`archive/MIGRATION.md`](?doc=archive/MIGRATION.md) | index | The ledger of the 2026-07 documentation migration: every old document and its new home; closed. |
| [`archive/README.md`](?doc=archive/README.md) | index | The archive's own index: what was archived when, and what replaced it. |
