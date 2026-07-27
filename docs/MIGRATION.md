# Migration ledger — old_docs/ → docs/

## Wave plan

**Sequencing decision (user, 2026-07-26): the documentation migration
completes IN FULL before any system/feature work resumes** (including the
jobset web-merge Phase 1). Docs first, then the system.

Every wave, every doc: the [README migration protocol + editorial rules
E1–E4](?doc=README.md) apply in full — structure-first merging, Mermaid where a
picture explains, plain language with full rigor, scientific foundations
preserved and enriched, and the master index updated in the same commit as
every move.

### Order: components first, the summary map last (user, 2026-07-26)

The two summary docs — `design.md` (the concise outline that *points at* the
detailed docs) and `architecture.md` (the reuse map over the subsystems) —
**aggregate every component**, so they are composed **LAST**, once the
component docs they summarize have settled. Writing them first would mean
drawing the map before walking the territory: the summary would go stale the
moment the first component moved. So the order is **bottom-up** — the L1 data
model, then the layers that build on it, then the surfaces — and the spine
summary is the final pass over the finished tree. `roadmap.md` (the forward
*plan*, which leads rather than summarizes) is the one spine doc done early.

### Keep-and-mark: old_docs survives until the final cross-check (user, 2026-07-26)

Migrated old docs are **NOT deleted** during the migration. Each worked-on
file is **kept in `old_docs/` and renamed with the `_migrated_` prefix**
(e.g. `protocols/web-api.md` → `protocols/_migrated_web-api.md`). The frozen
tree therefore stays complete until closeout, when the new tree is
cross-checked against it to prove nothing was dropped in a reconcile/merge —
*then* `old_docs/` is deleted. The ledger records the outcome; the prefix is
the at-a-glance filesystem mirror. Enforced by `tests/test_docs_structure.py`
(prefix ⇔ non-`pending` status). *(Grandfathered exception: the Wave-H
archives were relocated verbatim before this policy, so their old files are
already gone — identity-cross-checkable in `docs/archive/`.)*

| Wave | Scope (~docs) | Status |
|---|---|---|
| 0 | Freeze + skeleton + ledger + rules + tests | **done** (2162e44, 6efb0ca) |
| H | ALL historical archives → `archive/` (+ `audits/`) | **done** (a3f9b82) |
| 1 | `roadmap.md` — THE plan (absorbs all scattered phasing); the forward north-star | **done** (8143bcf) |
| 2 | `model/` (~11): structure family, periodicity, annotations, sidecars, selection, region labels, chemistry, parse stack, data vocabulary; `types/` folds in (`parsers.md` → archive) | **next** |
| 3 | `science/` (~4): validation machinery, chemistry correctness, pseudopotential standards, tuning | pending |
| 4 | `engines/` (~8): SIESTA, PySCF, transport/TranSIESTA, builders, GPU recipe; `transiesta-workflow` + `transport-guide` move in | pending |
| 5 | `execution/` (~16): staged-execution first; job-execution master; config; script contracts; SLURM; guides deduped; `bundle-contract` → `handoff-bundle` rename | pending |
| 6 | `web/` (~31): MolView family, workspace, sidebar, results, UI contract, web-api; `tabs/` folds in | pending |
| 7 | `ops/` (~5): install / deployment / rate-limit + config examples | pending |
| 8 | `process/` (~7): conventions, test strategy, audit playbook, CLI, package layout | pending |
| 9 | **The spine summary, composed last:** `design.md` (concise outline over the settled tree) + `architecture.md` (reuse map) + the `backend-architecture.md` companion decision | pending |
| 10 | Closeout: **cross-check `old_docs/` (all `_migrated_`) against the new tree**, then delete `old_docs/`, retire the freeze rule, archive the ledger, sweep memory pointers | pending |

`img/` assets distribute with their owning docs along the way.

## Ledger

One row per file in the frozen `old_docs/` tree. **Targets are
proposals** — the per-doc reconcile gate (README § Migration protocol)
confirms or changes them; record the outcome in Status + Notes.

Status values: `pending` | `moved` | `merged-into <doc>` | `archived`.

The old tree is FROZEN: `tests/test_docs_structure.py` fails if a file
appears in `old_docs/` that is not in this ledger.

| old_docs/ file | proposed target (docs/) | status | notes |
|---|---|---|---|
| `README.md` | README.md (merge: index absorbed) | pending | |
| `README_install.md` | ops/ | pending | |
| `architecture.md` | architecture.md | pending | **Composed LAST (Wave 9)** with `design.md` — the reuse map is a summary over the subsystems, so it settles after they migrate. `backend-architecture.md` kept SEPARATE (concern-lens companion), not merged. |
| `archive/2026-06-02-REVIEW_FINDINGS.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/2026-06-02-tabs-watch.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/2026-06-02-watch-api.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/2026-07-03-atom-selection.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/2026-07-03-embedded-viewer.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/2026-07-06-workspace-state.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `archive/README.md` | archive/ | merged-into archive/README.md | supersession table carried over |
| `atom-selection-guide.md` | web/ | pending | |
| `audit-2026-06-26/README.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `audit-2026-06-26/SYNTHESIS.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `audit-2026-06-26/T1_top_findings.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `audit-2026-06-26/T3_css_ui.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `audit-2026-06-26/T4_test_depth.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `audit-2026-06-27/README.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `batch-workflow-overview.md` | execution/ | pending | |
| `checkpoints-guide.md` | execution/ | pending | |
| `config.md` | execution/ | pending | |
| `deployment.md` | ops/ | pending | |
| `design.md` | design.md (concise outline; §0 index → README, plans → roadmap) | pending | **Composed LAST (Wave 9)** as the concise map over the settled tree. §6 "Next steps" (tab-reorg = SHIPPED; transport B.3) already absorbed into `roadmap.md` 2026-07-26. The §0 index is retired (README.md owns it); design.md becomes a short outline that points at the detailed domain docs. |
| `engines/builders.md` | engines/ | pending | |
| `engines/optimization-tuning.md` | engines/ | pending | |
| `engines/pyscf-publication-guide.md` | engines/ | pending | |
| `engines/pyscf.md` | engines/ | pending | |
| `engines/siesta-gpu.md` | engines/ | pending | |
| `engines/siesta.md` | engines/ | pending | |
| `engines/transport.md` | engines/ | pending | |
| `examples/molbuilder.asu-sol.json` | ops/examples/ | pending | |
| `form-schema-guide.md` | web/ | pending | |
| `img/SCREENSHOTS.md` | (distribute with owning docs) | pending | |
| `img/hero-molbuilder.png` | (distribute with owning docs) | pending | |
| `img/molbuilder-workspace.png` | (distribute with owning docs) | pending | |
| `img/results-bundle-card.png` | (distribute with owning docs) | pending | |
| `img/results-spectra.png` | (distribute with owning docs) | pending | |
| `img/results-trajectory.png` | (distribute with owning docs) | pending | |
| `img/sidebar-projects.png` | (distribute with owning docs) | pending | |
| `img/spectrum-form.png` | (distribute with owning docs) | pending | |
| `img/structure-optimization-form.png` | (distribute with owning docs) | pending | |
| `img/tab-bar.png` | (distribute with owning docs) | pending | |
| `img/transport-form.png` | (distribute with owning docs) | pending | |
| `job-case-analysis/ANALYSIS-G1K1C4.md` | archive/audits/ | moved | archived-history pass 2026-07-26 |
| `job-execution.md` | execution/ | pending | |
| `jobset-infrastructure.md` | execution/ | pending | |
| `molbuilder.json.example` | ops/examples/ | pending | |
| `molviewer-guide.md` | web/ | pending | |
| `package-layout.md` | process/ | pending | |
| `protocols/archive/2026-07-05-browser-data-contract.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `protocols/archive/2026-07-05-working-copy-persistence.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `protocols/archive/2026-07-06-molview-module.md` | archive/ | moved | archived-history pass 2026-07-26 |
| `protocols/atom-annotations.md` | model/structure-annotations.md | merged-into model/structure-annotations.md | Wave 2 (2026-07-26), sub-doc of structure.md. The per-atom channel model. Verified vs code — corrected: sidecar is **schema v6** now (annotations added at v4); JS L1 moved `lib/workspace/` → `lib/molview/_atom-channels.js`. "IN PROGRESS phases" resolved: model + persistence + block-recovery + built-in translations SHIPPED; open (value-channel producer + `by_value`, generic fdf-strategy) → roadmap. Stale `§7` reference dropped. |
| `protocols/backend-architecture.md` | (root, companion to architecture.md) or merge into it -- decide at reconcile | pending | |
| `protocols/benchmark-workflow.md` | execution/ | pending | |
| `protocols/bundle-contract.md` | execution/handoff-bundle.md (RENAME: kills bundle vocab collision) | pending | |
| `protocols/chemistry-correctness.md` | science/ | pending | |
| `protocols/cli.md` | process/ | pending | |
| `protocols/code-audit.md` | process/ | pending | |
| `protocols/code-conventions.md` | process/ | pending | |
| `protocols/data-vocabulary.md` | **SPLIT** — atom-index → model/overview.md; the rest → execution/ | pending | Reclassified 2026-07-26: mostly an **execution** concern (config↔SLURM parameter vocabulary §2, run identifiers/paths §3, the persisted-artifacts registry §1). Only §3.1/§3.2 (the atom-index convention) is model → absorbed into `model/overview.md` (`_atom-index.js` path corrected to `lib/molview/`); §5 structure-metadata already in `structure.md § 2.2`. Stays **pending** until the parameter-vocabulary + identifiers + artifacts registry land in `execution/` (that wave); then mark merged + prefix. |
| `protocols/frontend-module-architecture.md` | web/ | pending | |
| `protocols/inspector-registry.md` | web/ | pending | |
| `protocols/job-decoder.md` | execution/ | pending | |
| `protocols/job-layout.md` | execution/ | pending | |
| `protocols/mobile-layout.md` | web/ | pending | |
| `protocols/molview-esm-finalization.md` | web/ (plan parts -> roadmap.md) | pending | Plan tail (Phase B-internal / C / D) absorbed into `roadmap.md` §3 2026-07-26; migrate the target-architecture + consumer-inventory design content in Wave 3 (web/). |
| `protocols/molview-migration-plan.md` | roadmap.md (merge; archive rest) | pending | Open-work tail (D3-tail / D4 / A3 / A4 / Step 6) absorbed into `roadmap.md` §3 2026-07-26; the shipped-orientation + anti-drift scaffolding archives when this row closes (Wave 3). |
| `protocols/molview-module.md` | web/ | pending | |
| `protocols/molview-render-streamline.md` | web/ | pending | |
| `protocols/notifications.md` | web/ | pending | |
| `protocols/parse-module.md` | model/parse.md | merged-into model/parse.md | Wave 2 (2026-07-26). The shipped parse-stack contract (3 ABCs, ParseResult hierarchy, registry, layout, plugin/composer/forbidden-pattern rules). Verified vs `parse/` — corrected the stale §5 layout: the "PENDING" coords parsers (`siesta_fdf.py`, `xyz.py`) were never added; added the real helper modules. The huge §8 migration plan (Phases A–H, all shipped 2026-06-21) → one History note (R3). |
| `protocols/playwright-tests.md` | process/ | pending | |
| `protocols/projects-sidebar-ui.md` | web/ | pending | |
| `protocols/projects-sidebar.md` | web/ | pending | |
| `protocols/pseudopotential-validation.md` | science/ | pending | |
| `protocols/rate-limit.md` | ops/ | pending | |
| `protocols/region-labels.md` | **SPLIT** — vocab → model/structure-annotations.md; emitter/bias/refs → engines/transport.md | pending | Vocabulary (the `-electrode` convention, canonical `L`/`R`/`bridge`/`interface`, `is_electrode_label`) absorbed into `structure-annotations.md § 5` (2026-07-26). Stays **pending** until the transport half — `_find_electrode_regions`, chempot/`TS.Elec` emit, bias direction, the NEGF references (Brandbyge/Stokbro/Reed/Solomon) — lands in `engines/transport.md` (Wave 4); then mark merged + prefix. |
| `protocols/results-state-contract.md` | web/ | pending | |
| `protocols/results-tab.md` | web/ | pending | |
| `protocols/run-checkpoints.md` | execution/ | pending | |
| `protocols/runtime-registry.md` | web/ | pending | |
| `protocols/save-flow.md` | web/ | pending | |
| `protocols/scientific-validation.md` | science/ | pending | |
| `protocols/script-contract.md` | execution/ | pending | |
| `protocols/script-execution.md` | execution/ | pending | |
| `protocols/selection.md` | web/ — **MERGE into `projects-sidebar`** (same module) | pending | Reclassified out of model/ 2026-07-26 (verified: it is the sidebar file-selection *cursor* — `current_dir`/`current_file`, Inquire API, sidebar mutation endpoints — backed by `files.py`/`projects/` JS, NOT the L1 data model). Merge into the projects-sidebar doc at Wave 6 (web/); it belongs to that module. |
| `protocols/sidecar-contract.md` | **SPLIT** — §11–12 (pairing + consumers) → model/structure-molstruct.md; §1–10 (the 3-stage boundary-condition contract) → engines/ | pending | §11 (rename/move/copy pair the sidecar) + §12 (consumers) absorbed into `structure-molstruct.md` (2026-07-26; the doc's envelope/codec/versioning is sourced from `sidecars/molstruct.py` directly). Stays **pending** until the 3-stage UI→config→script boundary-condition contract (frozen/regions → engine input, no-silent-absorption preflight, per-engine label table) lands in `engines/` (Wave 4); then mark merged + prefix. |
| `protocols/slurm-integration.md` | execution/ | pending | |
| `protocols/staged-execution.md` | execution/ | pending | §15.5 web-integration phasing (Phases 1-4 + D7 gate) absorbed into `roadmap.md` §1 2026-07-26; the framework contract itself migrates in Wave 2 (execution/). |
| `protocols/structure-authority.md` | model/structure.md | merged-into model/structure.md | Wave 2 (2026-07-26): the codec + doors + metadata authority + round-trip invariant. Facts re-verified vs code — corrected the "`_shared.structure_to_dict` deleted" claim (it is a retained back-compat wrapper over `to_wire()`); §3.4 sidecar-envelope detail → `model/sidecars.md`; §6 open CLI step → roadmap. |
| `protocols/structure-load-save-contract.md` | model/structure.md | merged-into model/structure.md | Wave 2 (2026-07-26): the JS doors + model primitives + SETTLE-BEFORE-READY + consumer map → the frontend surface of `model/structure.md`. Seam facts verified vs `build.py` (`/api/build/load`, `/api/structure/save`). |
| `protocols/structure-periodicity.md` | model/structure-periodicity.md | merged-into model/structure-periodicity.md | Wave 2 (2026-07-26), sub-doc of structure.md (filename-prefix convention, R5). Reconciled vs code — **major drift cleaned up**: `kgrid` is NOT a structure field (it's a `SiestaConfig` DFT knob; sidecar v5 dropped it) → its physics deferred to `engines/siesta.md` (Wave 4), preserved meanwhile in this kept source. Also fixed: isolated cell = `bbox + 2·vacuum` (not `+vacuum`); §9 "to-build" mostly SHIPPED (capture, calibrate, resolvers); `ws.*` accessors → `molview.data.*`; transport `cell_fdf` now a legacy fallback. |
| `protocols/test-strategy.md` | process/ | pending | |
| `protocols/transiesta-workflow.md` | engines/ | pending | |
| `protocols/ui-design-contract.md` | web/ | pending | |
| `protocols/vibrationview.md` | web/ | pending | |
| `protocols/web-api.md` | web/ | pending | |
| `protocols/web-module-map.md` | web/ | pending | |
| `protocols/web-ui-coherence.md` | web/ | pending | |
| `protocols/workspace-contract.md` | web/ | pending | |
| `results-tab-guide.md` | web/ | pending | |
| `roadmap.md` | roadmap.md (THE plan; absorbs scattered phasing) | pending | PLAN distilled into `docs/roadmap.md` (Wave 1, 2026-07-26) — that is now the plan authority. File retained ONLY for its shipped 3DNA detection/error/install + transport reference; relocates to `engines/` at Wave 6, then this row closes. |
| `runtime-registry-guide.md` | web/ | pending | |
| `science.md` | science/ | pending | |
| `staged-relaxation-guide.md` | execution/ | pending | |
| `structure-guide.md` | web/ | pending | |
| `tabs/architecture.md` | web/tabs.md (rename: avoid 3rd architecture.md) | pending | §9 phasing (Phases A-D) reconciled against code = ALL SHIPPED (6 tabs live); recorded closed in `roadmap.md` 2026-07-26. The cross-tab design content migrates in Wave 3 (web/). |
| `tabs/molbuilder.md` | web/ | pending | |
| `tabs/results.md` | web/ | pending | |
| `tabs/spectra/references.bib` | web/ | pending | |
| `tabs/spectra/spec.md` | web/ | pending | |
| `tabs/structure-optimization.md` | web/ | pending | |
| `templates/github-workflows-test.yml` | process/ | pending | |
| `transport-guide.md` | engines/ | pending | |
| `types/chemistry.md` | model/chemistry.md | merged-into model/chemistry.md | Wave 2 (2026-07-26). The source's scope (phosphate charge + protonation) fully migrated + verified vs `chemistry.py` (cutoffs 1.30/1.95, O–H 0.96 Å / 109.47°, idempotent `is`-return). Also mapped the rest of the module: charge helpers / `add_hydrogens` (OpenBabel→RDKit) / clash / dipole here; the **correctness** half (spin-charge parity, open-shell metals, ECP, `analyze_structure` + adapters) pointed to `science/` (documented there in `chemistry-correctness.md` / `scientific-validation.md`, science wave). |
| `types/parsers.md` | (retired — superseded by model/parse.md) | archived | Wave 2 (2026-07-26). The old file-level parser contract; its substance (the FileParser ABC + result types) is now in `model/parse.md` §§ 1–3. Retired stub, kept `_migrated_` for the closeout cross-check. |
| `types/structure.md` | model/structure.md | merged-into model/structure.md | Wave 2 (2026-07-26): the dataclass fields + invariants + geometry I/O (XYZ/PDB/PySCF/ASE) + TER handling + `molbuilder.load` → the L1 data-model + backend surface of `model/structure.md`. Method line-numbers verified vs `structure.py`. |
| `workspace-guide.md` | web/ | pending | |
