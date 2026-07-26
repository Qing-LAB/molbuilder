# molbuilder — documentation

**This is the ONE index.** Every document under `docs/` is listed here, one
line each. If a doc is not listed here it does not exist (test-enforced:
`tests/test_docs_structure.py`).

> **Migration in progress (started 2026-07-26).** The previous docs tree is
> frozen at [`../old_docs/`](../old_docs/) and is being migrated here piece
> by piece, each doc reconciled against the code at its move. The ledger —
> what lives where, what is pending — is [`MIGRATION.md`](MIGRATION.md).
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

**Order (see [`MIGRATION.md`](MIGRATION.md) for the wave plan):** components
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
| [`roadmap.md`](roadmap.md) | plan | The ONE plan: every open feature/backend workstream + the closed-work log |
| [`model/structure.md`](model/structure.md) | contract | The `Structure` object (master): the L1 codec (`to_dict`/`from_dict`/`to_wire`), geometry I/O, the L2 paired-file door, and the JS load/save doors |
| [`model/structure-periodicity.md`](model/structure-periodicity.md) | contract | *(sub of structure)* Per-axis box behaviour: `cell` · `cell_origin` · `axis_kind` · derived `pbc` · `vacuum` (k-grid is a `SiestaConfig` knob, not here) |
| [`MIGRATION.md`](MIGRATION.md) | index | The migration ledger: every old_docs file → target home + status |
| [`archive/README.md`](archive/README.md) | index | The archive's own index: what was archived when, and what superseded it |
