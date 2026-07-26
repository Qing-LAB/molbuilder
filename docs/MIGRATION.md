# Migration ledger — old_docs/ → docs/

## Wave plan

**Sequencing decision (user, 2026-07-26): the documentation migration
completes IN FULL before any system/feature work resumes** (including the
jobset web-merge Phase 1). Docs first, then the system.

Every wave, every doc: the [README migration protocol + editorial rules
E1–E4](README.md) apply in full — structure-first merging, Mermaid where a
picture explains, plain language with full rigor, scientific foundations
preserved and enriched, and the master index updated in the same commit as
every move.

| Wave | Scope (~docs) | Status |
|---|---|---|
| 0 | Freeze + skeleton + ledger + rules + tests | **done** (2162e44, 6efb0ca) |
| H | ALL historical archives → `archive/` (+ `audits/`) | **done** (a3f9b82) |
| 1 | The spine: `roadmap.md` (THE plan, absorbs all scattered phasing), `design.md` (slimmed), `architecture.md` (+ `backend-architecture.md` decision) | **in progress** — `roadmap.md` done (2026-07-26): all scattered phasing distilled into the one plan; `design.md` + `architecture.md` next |
| 2 | `execution/` (~16): staged-execution first; job-execution master; config; script contracts; SLURM; guides deduped; `bundle-contract` → `handoff-bundle` rename | pending |
| 3 | `web/` (~31): MolView family, workspace, sidebar, results, UI contract, web-api; `tabs/` folds in | pending |
| 4 | `model/` (~12): structure family, annotations, sidecars, parse stack, data vocabulary; `types/` folds in (`parsers.md` → archive) | pending |
| 5 | `science/` (~4) | pending |
| 6 | `engines/` (~8): + `transiesta-workflow`, `transport-guide` move in | pending |
| 7 | `ops/` (~5): install / deployment / rate-limit + config examples | pending |
| 8 | `process/` (~7): conventions, test strategy, audit playbook, CLI, package layout | pending |
| 9 | Closeout: `old_docs/` deleted, freeze rule retired, ledger archived, memory pointers updated | pending |

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
| `architecture.md` | architecture.md | pending | |
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
| `design.md` | design.md (slim: index+arch dups out) | pending | §6 "Next steps" (tab-reorg Phase 7 = SHIPPED; transport B.3) absorbed into `roadmap.md` 2026-07-26; migrate the rest (mission/principles/anti-patterns/decisions/process) in this Wave-1 pass. |
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
| `protocols/atom-annotations.md` | model/ | pending | |
| `protocols/backend-architecture.md` | (root, companion to architecture.md) or merge into it -- decide at reconcile | pending | |
| `protocols/benchmark-workflow.md` | execution/ | pending | |
| `protocols/bundle-contract.md` | execution/handoff-bundle.md (RENAME: kills bundle vocab collision) | pending | |
| `protocols/chemistry-correctness.md` | science/ | pending | |
| `protocols/cli.md` | process/ | pending | |
| `protocols/code-audit.md` | process/ | pending | |
| `protocols/code-conventions.md` | process/ | pending | |
| `protocols/data-vocabulary.md` | model/ | pending | |
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
| `protocols/parse-module.md` | model/ | pending | |
| `protocols/playwright-tests.md` | process/ | pending | |
| `protocols/projects-sidebar-ui.md` | web/ | pending | |
| `protocols/projects-sidebar.md` | web/ | pending | |
| `protocols/pseudopotential-validation.md` | science/ | pending | |
| `protocols/rate-limit.md` | ops/ | pending | |
| `protocols/region-labels.md` | model/ | pending | |
| `protocols/results-state-contract.md` | web/ | pending | |
| `protocols/results-tab.md` | web/ | pending | |
| `protocols/run-checkpoints.md` | execution/ | pending | |
| `protocols/runtime-registry.md` | web/ | pending | |
| `protocols/save-flow.md` | web/ | pending | |
| `protocols/scientific-validation.md` | science/ | pending | |
| `protocols/script-contract.md` | execution/ | pending | |
| `protocols/script-execution.md` | execution/ | pending | |
| `protocols/selection.md` | model/ | pending | |
| `protocols/sidecar-contract.md` | model/ | pending | |
| `protocols/slurm-integration.md` | execution/ | pending | |
| `protocols/staged-execution.md` | execution/ | pending | §15.5 web-integration phasing (Phases 1-4 + D7 gate) absorbed into `roadmap.md` §1 2026-07-26; the framework contract itself migrates in Wave 2 (execution/). |
| `protocols/structure-authority.md` | model/ | pending | |
| `protocols/structure-load-save-contract.md` | model/ | pending | |
| `protocols/structure-periodicity.md` | model/ | pending | |
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
| `types/chemistry.md` | model/ | pending | |
| `types/parsers.md` | archive/ (RETIRED; superseded by model/parse-module.md) | pending | |
| `types/structure.md` | model/ | pending | |
| `workspace-guide.md` | web/ | pending | |
