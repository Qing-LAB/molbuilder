# Archive

Historical documents. **Not a source of truth.** If you are reading a
date-prefixed file, you are reading history, not policy.

## Conventions (docs/README.md rule R4)

- Every archived doc is prefixed `YYYY-MM-DD-<original-name>.md` — the date
  it was archived, so the boundary between history and policy is visible in
  the filename itself.
- Every archived doc gets a row in the table below naming **what superseded
  it** (the canonical doc to read instead) — or "closed / historical" when
  it was a one-shot log whose work is complete. Superseded-by entries are
  doc *names*, not links: canonical docs move as the tree evolves, history
  doesn't chase them (find the current home in `../README.md`).
- **Substance-first rule** (learned the hard way, 2026-06-02): the substance
  of every still-live contract must be migrated into the canonical doc
  BEFORE the archive move. An archived file must never be the only place a
  live invariant lives. (The 2026-06-02 pass initially over-compressed and
  had to restore ~15 dropped contracts — see web-api.md § 16's gap audit.)
- Two sources feed this folder: **migration archiving** (docs from the
  frozen `old_docs/` tree whose reconcile gate found them superseded or
  retired — their ledger row in `../MIGRATION.md` says `archived`) and
  **ongoing archiving** (a live doc superseded later; move it here with the
  date prefix in the same commit that supersedes it).
- `audits/` holds point-in-time audit/analysis snapshots imported as whole
  directories — findings that were acted on; kept for the record.
- **Whole-tree snapshots are the one exception to the date-prefix rule**:
  `audits/…` directories and `old_docs/` (the pre-migration docs tree,
  86 files, snapshot 2026-07-26) are imported verbatim as directories.
  The snapshot's date and identity live in its Index row here — not in
  per-file prefixes, which would break the per-file map in
  `MIGRATION.md` and the trees' own internal links.

## Index

| Archived doc | Reason | Substance lives in |
|---|---|---|
| `2026-06-02-REVIEW_FINDINGS.md` | Closed — one-shot code-review log; work landed | n/a (historical) |
| `2026-06-02-watch-api.md` | Superseded — `/api/watch/*` HTTP reference | web-api.md § 8 (endpoint table, Mode A/B, `/api/watch/data` shape, `MOLBUILDER_WATCH_ROOT`, concurrency, security) |
| `2026-06-02-tabs-watch.md` | Superseded — legacy `/watch` UI spec; trajectory inspector lives on `/results` since 2026-05-19 | inspector-registry.md § 6; cross-cutting front-end conventions → web-api.md § 14.4 |
| `2026-07-03-embedded-viewer.md` | Superseded — standalone embedded-viewer contract, folded into the MolView module | molview-module.md (the viewer / handle) |
| `2026-07-03-atom-selection.md` | Superseded — standalone atom-selection spec, folded into the MolView module | molview-module.md (store, composition, measurement) |
| `2026-07-05-browser-data-contract.md` | Superseded — browser-owned working-copy contract; the "changed-underneath" hash-gate was removed | workspace-contract.md § 4 (persistence) |
| `2026-07-05-working-copy-persistence.md` | Superseded — load/edit/save working-copy persistence spec | workspace-contract.md § 4 + § 4.6 |
| `2026-07-06-molview-module.md` | Superseded — the 2026-07-03 standalone MolView design+contract snapshot | molview-module.md (viewer + selection + k-grid + measurement) |
| `2026-07-06-workspace-state.md` | Closed — the 2026-06-07 workspace-unification audit + Phases 1–9 migration log (all shipped); kept for the *why* | workspace-contract.md (the live model) |
| `audits/audit-2026-06-26/` | Closed — whole-repo audit snapshot (synthesis + top findings + CSS/UI + test depth) | findings acted on in code/tests |
| `audits/audit-2026-06-27/` | Closed — follow-up audit snapshot | findings acted on in code/tests |
| `audits/job-case-analysis/` | Closed — ANALYSIS-G1K1C4 job-case study (bench/job-execution milestone 2026-06-29) | conclusions folded into the bench/staged-execution contracts |
| `2026-07-28-decisions-log.md` | Closed — the verbatim 113-entry decisions log from the pre-migration `design.md` (Wave 9); kept for the *why* behind each decision | design.md § Decisions (indexes the load-bearing entries → each domain doc) |
| `old_docs/` | **Frozen legacy tree** — the pre-migration docs (86 files, snapshot 2026-07-26) migrated to the domain-structured `docs/` tree across Waves 0–9 (2026-07-26 to 2026-07-28) | `docs/README.md` (live index); `docs/archive/MIGRATION.md` (per-file map) |
| `2026-08-04-closed-tasks.md` | Closed — seven tasks finished 2026-08-02→04 (#41 the one label store, #44 MolView persistency, #48 server reload, #50 the app locking its own user out, #51 read-only tabs keep their structure, #52 label-chip colours, #53 opening a file with an unusable box), moved off the working list; kept for the decisions **and the corrections** — three record a thing believed, measured, and found false | the live contracts each one changed (`web-api.md` § 1, `molview.md` § 11.2a, `access-control.md` § 7, `structure-periodicity.md` § 8.2) |
| `2026-08-11-staged-runs-architecture.md` | Superseded — the staged-runs design written against the **flat** directory shape, before the hierarchical one existed (36 mentions of *flat* against 9 of *hierarchical*, in 1430 lines). It is the pre-job-system picture: one tab writing one flat directory, and unifying on the hierarchical shape is the whole of the work that replaced it. It never held durable decisions — its own status line said so and said the contracts win — but the domain map still routed readers to it for *"the design"*, so it kept being cited as authority. Its live substance left first: the five open questions to `stages.md` § 6b and `run-identity.md` § 6a, the work items already in the implementation plan under its own `Z` codes. **§ 8a–8b are dated code audits (2026-08-07) and are the reason to keep the file** | `execution/architecture.md` (who owns which decision), `execution/project-layout.md` (what a directory is, the two shapes, `prep`), `engines/stages.md` (a stage, the description), `plans/staged-runs-implementation-plan.md` (the items, the order, the gates), `execution/job-system.md` (the CLI grammar it called *step 1c*) |
| `2026-08-11-template-item-blocks.md` | Superseded — the template as an **`.fdf` with item blocks**: each parameter wrapped in `# === molbuilder item <field> BEGIN/END ===`, its declaration in a comment, and the deck line it produces as the block's payload. Retired 2026-08-11 on two findings: `prep` **rebuilds a config and renders** rather than substituting at anchors (which `engines/stages.md` § 4 and the section's own property 1 both already said), and once substitution was gone the argument for the engine's own format went with it — leaving only its cost, that each value was stored **twice** (declaration and payload) so the file could disagree with itself. The `kind=` vocabulary, `read_by=`, per-kind losslessness and the fingerprint were carried forward, not dropped | `engines/template.md` (the format, the items, the kinds), `job-contracts.md` § 3.7 (a pointer), § 6.1 and § 6.3 (the file's registry row and its name) |
| `2026-08-10-stage-chaining.md` | Superseded — a ladder was a SCHEDULER chain: `Job.depends_on` / `Job.dep_kind` edges, `Carry` symlinks laid before the producer ran, `carry_deref` to localize them, `--chain`, and SIESTA's `on_nonconvergence` whose whole effect was to pick the edge kind. Retired by user decision 2026-08-10 on scientific grounds — whether a later stage should pick up an earlier one cannot be settled without reviewing the earlier one's result — and an opt-in flag was rejected with it. Kept as the ONE home for the retired vocabulary, so the live contracts state what the system is rather than what it stopped being | `project-layout.md` § 1.6 (a person prepares each stage and names what it continues from), `job-system.md` § 2 decision 6 (a JobSet has no edges) and § 3 (`WarmFile` — what a job would take, never from whom) |
| `2026-08-16-structure-optimization-ui-plan.md` | Superseded — **the tab it plans is built.** It designed the Structure-optimization page's split: collect the physics, decide what varies, write the folder, and stop short of a deck. That landed 2026-08-15 (`d1c8a871`, `3a3a3b61`) — the tab now renders its form from the catalogue and produces no artifact — so the file was a second, drifting description of a shipped surface, still headed *"Status: a proposal. Nothing here is built."* Its § 7.5 also rested on two premises that were later measured false (any-ELPA-needs-the-GPU-build; `prep` proposing a `BlockSize`) | `web/form-schema.md` (how a catalogue item becomes a control), `engines/stages.md` § 1.2–1.3 (the promotion checkbox and the group default), § 5 (which fields reach the deck), `execution/project-layout.md` § 2.1–2.2 (why the browser cannot finish a deck) |
| `2026-08-16-task-setup-plan.md` | Superseded — the shared tab that finishes a description, planned under the working name *Task Setup*. The name is settled (**Task Setup**) and the design is agreed, so the plan became a second description of it. Its § 7 operations table — the nine edits that can each silently destroy a value — migrated in full before the move, as did § 3.1 (*holds no state of its own*) and § 4 (*"is this a calculation?" is answered once, by `checkpoint.py::_is_bundle_root`*). It carried two retired rules to the end: *"a one-stage calculation never opens this tab"* (`stages.md` § 6.5 now: a job always has at least one stage) and a `base:` key removed from `task.json` on 2026-08-07 | **`web/task-setup.md`** (the design, entire) |
| `2026-08-16-siesta-catalogue.md` | Closed — a working document, by its own header: *"not a contract… once the placements are agreed it becomes the input to T2, and the arguments here move into the contract or are dropped."* The placements were agreed and shipped into `data/catalogue.template.toml`. It records **44** fields where there are now 47, and catalogues `spin_polarized`, which no longer exists — spin became a four-state `spin_treatment` (`d15403a5`). Kept for the placement arguments, which is what it was written to hold | `engines/template.md` § 6.2 (the category vocabulary — the authority its own header names), and the catalogue file itself |
| `2026-08-17-generator-api-plan.md` | Closed — a one-shot plan whose five steps all landed the day it was written. It held the measurement behind two new architecture rules: `write_run_wrapper` took **eleven loose keyword arguments**, and its two callers passed ten and five, so `jobset/prep.py` wrote a `.sbatch` asking for `-c 8` beside a `.run.sh` baking an OMP default of `1`, while `web/blueprints/build.py` wrote the mirror image — each correct about one artifact, neither producing a correct pair. The same door had lost `max_memory_mb` to a hand-copied argument list four days earlier; that fix moved the field onto `Resources` and left the calling convention alone, which is why the class stayed open. **Everything durable migrated before the move**: the rule and its measurement to `architecture.md` § 3.1 + A8/A9, the step-order table to `generator.md` § 6.2, the `--cold` anchoring rule to `job-contracts.md` § 4.1, and its three out-of-scope items — the wrapper's string assembly, the twice-implemented GPU detection, and the open `bench`-in-`task@1` conflict — to `roadmap.md` § 6, so the one genuinely undecided question stays in a live document. Kept for the mis-diagnosis it records: *a diff shows what you have not committed, not what you have not caused* | **`execution/architecture.md`** § 3.1 and § 7 rules A8 · A9 (the rules), `execution/generator.md` § 6.2 (the step order), `execution/job-contracts.md` § 4.1 (the `--cold` exception), `roadmap.md` § 6 (what is left) |
| `2026-08-16-molview-rework-plan.md` · `2026-08-16-molview-integration-plan.md` · `2026-08-16-molview-corrections-plan.md` | Superseded — three plans covering one module: building it from the document, wiring the tabs to it, and correcting what shipped complete-but-unconnected. **MolView is done** (user, 2026-08-16), so all three describe finished work. Nothing but the docs index cited them | `web/molview.md` (the contract), `web/vibrationview.md` (the separate viewer it is often confused with) |

> Note (2026-07-26): the archived docs above predate the docs-tree
> reorganization, so *internal* relative links inside them point at the old
> tree layout and may dangle. That is expected — they are verbatim history
> (the link-integrity test exempts `archive/`).
