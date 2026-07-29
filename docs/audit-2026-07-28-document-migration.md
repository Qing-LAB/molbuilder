# Documentation-migration audit

**Role:** audit report
**Domain:** *(root — documentation migration closeout)*
**Audited:** 2026-07-28
**Scope:** 86 archived documents in `docs/archive/old_docs/`, 48 active
Markdown documents, and the owning Python, Flask, HTML, JavaScript, CSS, and
test surfaces.

This is closeout evidence, not a replacement for the domain docs. Active docs
are authoritative; `archive/` is historical evidence.

## Executive result

The migration preserved the subject coverage of every archived document. Every
legacy document has an active target, a deliberate split across active owners,
or an explicit archived/retired disposition. The active tree has coherent
ownership: `model/`, `science/`, `engines/`, `execution/`, `web/`, `ops/`, and
`process/`, with root-level design, architecture, and roadmap documents.

The closeout defect is **not missing prose**. Active code, templates, tests,
packaging metadata, and handoff notes contain **319 references in 181 files**
to retired `docs/protocols/*`, `docs/types/*`, `docs/README_install.md`, or
`docs/old_docs/*` locations. The new docs are generally correct, but those
embedded references make the archive appear authoritative and sometimes direct
users to a former path that no longer exists.

## Method and verification limit

1. Read every legacy document; compared title, section inventory, technical
   contract, and ledger disposition to the active owner(s).
2. Treated implementation as authority where legacy content conflicted with
   current code.
3. Cross-checked high-risk contracts against Python/CLI, Flask routes and
   templates, JavaScript modules, tests, the docs reader, index/TOC, roadmap,
   source TODOs, and obsolete-reference inventory.

`pytest` is unavailable in the inspected interpreter, so
`python -m pytest -q tests/test_docs_structure.py tests/test_docs_tab.py`
could not run (`No module named pytest`). This is an environment limitation,
not a test failure. Static inspection confirms root `old_docs/` is absent.

## Findings

### P0 — active files cite retired documentation

**Evidence:** 319 occurrences across 181 active files: 109 under
`molbuilder/`, 68 under `tests/`, plus `README.md`, `HANDOFF.md`,
`pyproject.toml`, and `scripts/install-env.sh`. Affected material includes
runtime/error guidance (`parse/registry.py`), JobSet and parser docstrings,
test metadata, and UI/template comments.

**Impact:** mostly non-functional, but maintainers are pointed to the wrong
contract and some diagnostics direct users to old paths. It contradicts the
new rule that archive content is not authoritative.

**Completion:** replace every reference with its active owner, or explicitly
say `archive/` where historical context is intended. Add a test that forbids
active references to the retired paths, with a narrow allowlist only for tests
that deliberately assert archive access behavior. Rewrite by domain, not by a
global replacement: several legacy docs split into multiple active owners.

### P1 — roadmap governance is incomplete

`roadmap.md` accurately tracks the major streams, but active source comments
still hold ungoverned or underspecified plans:

- `web/static/structure-optimization/viewer.js` polls for up to six seconds
  because form-schema has no render-complete callback.
- `web/static/lib/spectra/core.js` has a TODO for session-storage persistence
  of spectrum UI preferences; current docs correctly say preferences are
  per-mount, but the roadmap has no discrete item/test pin.
- VibrationView still borrows the transitional MolView embed/global boundary;
  the broad ESM stream only implies this work.
- TranSIESTA docstrings point to unnamed external `project_*.md` plans instead
  of active transport docs and the roadmap.

Add the first two as roadmap items, make the VibrationView boundary an explicit
ESM/concealment sub-item, replace external-memory pointers, and give each item
a concrete test pin.

### P1 — stale comments contradict current behavior

The active docs correctly describe Transport, but selected source comments do
not. `web/app.py` calls `/transport-calculation` a placeholder although its
form, `lib/transport/core.js`, and `/api/transport/render` implement working
zero-bias generation. `transport/transiesta.py` says electrode-FDF generation
is deferred and describes the retired manual path, while `transport/wizard.py`
and the CLI ship the wizard. Correct these comments in the reference-migration
pass; retain the true limitation: the web tab generates one zero-bias device
FDF, not a full multi-bias workflow bundle.

### P2 — docs tests were not runnable here

Install the project `test` extra or use the configured test environment, run
the two docs tests first, then focused tests for every changed source/test
reference.

## Code/design reconciliation

| Area | Verified current state | Conclusion |
|---|---|---|
| Docs delivery | `web/blueprints/docs.py` read-only serves `docs/`; TOC separates active docs/archive. | Sound; this audit must be indexed. |
| Structure | Structure codec, paired `.molstruct.json`, periodicity, annotations, and web codec load/save exist; CLI writes geometry only. | Coverage preserved; CLI-sidecar parity remains open. |
| Parse | Documented ABC/registry/composer stack exists under `parse/`. | Retiring `types/parsers.md` for `model/parse.md` is correct. |
| JobSet | CLI plan/prep/submit/status and SIESTA producer exist; no JobSet blueprint/API exists. | “CLI-shipped, web-pending” is accurate. |
| PySCF discovery | `watch.py` matches generated `JOB = "..."`. | Historical `job_name` mismatch is fixed; active doc is accurate. |
| SIESTA retry | `continue_retries` validates; PySCF implements retry loop, SIESTA chaining does not. | Active docs correctly distinguish the limitation. |
| Transport | Zero-bias render/preflight, wizard, CLI bundle/orchestration, and web single-FDF generation ship; parsing, multi-bias, and presenter do not. | Active transport docs/roadmap are accurate. |
| Web modules | MolView/workspace/projects are ESM; presenters, cores, runtime, and primitives remain transitional globals. | Current-to-target docs are accurate. |
| Spectra/trajectory | Spectra has two mounts/live watch; preferences are in memory; trajectory retains status/polling debt. | Coverage preserved; roadmap needs finer items. |
| Deployment | Docs correctly describe dev server, reverse proxy, TLS guard, optional SSO, and rate limit. | Legacy deployment material was corrected, not copied. |

## Pending product and refactoring plan

1. **Web JobSet integration:** wire the Structure-optimization stage table to
the SIESTA producer, then plan/status and checkpoint branch UI; pass a real
cluster ladder gate before transport/PySCF/spectra bundle modes. Pin byte
identity between equivalent CLI and web bundles.
2. **Transport results/multi-bias:** design `<job>.transport.json`, implement
`parse_output`, add transmission presentation, make JobSet Phase 3 emit one
input per bias/I-V aggregation, emit `TS.Atoms.Buffer`, and decide
multi-terminal chemical-potential UI. PySCF-NEGF/IETS remain later engines.
3. **Structure/persistence:** use `StructureCodec` for CLI load/save, decide
durable draft ownership, browser-verify timeline persistence, remove obsolete
Modify endpoints only after Results consumers are checked, then finish value
channels and generic annotation-to-FDF strategies.
4. **Frontend finalization:** finish MolView concealment; convert/rename
`inspectors` to `presenters` with trajectory/spectra cores; then Results,
runtime, and primitives, browser-verified per page. Include independent
VibrationView embedding, spectrum preference persistence, and form-schema
render completion.
5. **Ops/tests:** resolve micromamba bootstrap, review admin rate-limit auth
and user-level config, test security headers, make missing Chromium deselect
rather than fail e2e collection, maintain a skip census, and persist
multi-frame trajectories.
6. **Documentation closeout:** remove stale embedded paths, correct stale
comments, add a guard test, and run docs tests. Do not copy legacy prose back
into active docs simply to eliminate it.

## Per-document coverage matrix

**Covered** = live contract represented by active target(s). **Split** = live
sections intentionally placed in multiple owners. **Archived** = retired or
historical content, not missing guidance.

| Archived document | Active disposition | Status |
| `README.md` | README.md + process/conventions.md | merged-into README.md |
| `README_install.md` | ops/installation.md | merged-into ops/installation.md |
| `architecture.md` | architecture.md | moved |
| `atom-selection-guide.md` | web/molview.md | merged-into web/molview.md |
| `batch-workflow-overview.md` | execution/job-system.md | merged-into execution/job-system.md |
| `checkpoints-guide.md` | execution/running-a-job.md | merged-into execution/running-a-job.md |
| `config.md` | execution/running-a-job.md | merged-into execution/running-a-job.md |
| `deployment.md` | ops/deployment.md | merged-into ops/deployment.md |
| `design.md` | design.md | moved |
| `engines/builders.md` | engines/builders.md | merged-into engines/builders.md |
| `engines/optimization-tuning.md` | engines/tuning.md | merged-into engines/tuning.md |
| `engines/pyscf-publication-guide.md` | engines/pyscf.md (merged) | merged-into engines/pyscf.md |
| `engines/pyscf.md` | engines/pyscf.md | merged-into engines/pyscf.md |
| `engines/siesta-gpu.md` | **SPLIT** — GPU setting/perf → engines/siesta.md § 7.1; build/env recipe → ops/installation.md | merged-into engines/siesta.md + ops/installation.md |
| `engines/siesta.md` | engines/siesta.md | merged-into engines/siesta.md |
| `engines/transport.md` | engines/transport.md | merged-into engines/transport.md |
| `form-schema-guide.md` | web/form-schema.md | merged-into web/form-schema.md |
| `job-execution.md` | execution/running-a-job.md | merged-into execution/running-a-job.md |
| `jobset-infrastructure.md` | execution/job-system.md | merged-into execution/job-system.md |
| `molviewer-guide.md` | web/molview.md | merged-into web/molview.md |
| `package-layout.md` | process/package-layout.md | merged-into process/package-layout.md |
| `protocols/atom-annotations.md` | model/structure-annotations.md | merged-into model/structure-annotations.md |
| `protocols/backend-architecture.md` | backend-architecture.md | moved |
| `protocols/benchmark-workflow.md` | execution/job-system.md | merged-into execution/job-system.md |
| `protocols/bundle-contract.md` | execution/job-contracts.md § 5 | merged-into execution/job-contracts.md |
| `protocols/chemistry-correctness.md` | science/chemistry-correctness.md | merged-into science/chemistry-correctness.md |
| `protocols/cli.md` | process/conventions.md | merged-into process/conventions.md |
| `protocols/code-audit.md` | process/code-audit.md | merged-into process/code-audit.md |
| `protocols/code-conventions.md` | process/conventions.md | merged-into process/conventions.md |
| `protocols/data-vocabulary.md` | **SPLIT** — atom-index → model/overview.md; §5 → structure.md; §1–3 → execution/job-contracts.md | merged-into (split) |
| `protocols/frontend-module-architecture.md` | web/overview.md | merged-into web/overview.md |
| `protocols/inspector-registry.md` | web/presenters.md | merged-into web/presenters.md |
| `protocols/job-decoder.md` | execution/running-a-job.md | merged-into execution/running-a-job.md |
| `protocols/job-layout.md` | execution/job-contracts.md | merged-into execution/job-contracts.md |
| `protocols/mobile-layout.md` | web/ui-contract.md | merged-into web/ui-contract.md |
| `protocols/molview-esm-finalization.md` | web/molview.md | merged-into web/molview.md |
| `protocols/molview-migration-plan.md` | roadmap.md + web/overview.md | merged-into roadmap.md |
| `protocols/molview-module.md` | web/molview.md | merged-into web/molview.md |
| `protocols/molview-render-streamline.md` | web/molview.md | merged-into web/molview.md |
| `protocols/notifications.md` | web/notifications.md | moved → web/notifications.md |
| `protocols/parse-module.md` | model/parse.md | merged-into model/parse.md |
| `protocols/playwright-tests.md` | process/testing.md | merged-into process/testing.md |
| `protocols/projects-sidebar-ui.md` | web/projects.md | merged-into web/projects.md |
| `protocols/projects-sidebar.md` | web/projects.md | merged-into web/projects.md |
| `protocols/pseudopotential-validation.md` | science/pseudopotentials.md | merged-into science/pseudopotentials.md |
| `protocols/rate-limit.md` | ops/deployment.md | merged-into ops/deployment.md |
| `protocols/region-labels.md` | **SPLIT** — vocab → model/structure-annotations.md; emitter/bias/refs → engines/transport.md | merged-into (split) |
| `protocols/results-state-contract.md` | web/ (SPLIT: results.md + trajectory.md + spectra.md) | merged-into web/{results,trajectory,spectra}.md |
| `protocols/results-tab.md` | web/ (SPLIT: results.md + trajectory.md) | merged-into web/results.md + web/trajectory.md |
| `protocols/run-checkpoints.md` | execution/running-a-job.md | merged-into execution/running-a-job.md |
| `protocols/runtime-registry.md` | web/runtime.md | merged-into web/runtime.md |
| `protocols/save-flow.md` | web/tabs.md | merged-into web/tabs.md |
| `protocols/scientific-validation.md` | science/validation.md | merged-into science/validation.md |
| `protocols/script-contract.md` | execution/job-contracts.md | merged-into execution/job-contracts.md |
| `protocols/script-execution.md` | execution/job-contracts.md | merged-into execution/job-contracts.md |
| `protocols/selection.md` | web/projects.md | merged-into web/projects.md |
| `protocols/sidecar-contract.md` | **SPLIT** — §11–12 → model/structure-molstruct.md; §1–10 → engines/overview.md | merged-into (split) |
| `protocols/slurm-integration.md` | execution/job-system.md | merged-into execution/job-system.md |
| `protocols/staged-execution.md` | execution/job-system.md | merged-into execution/job-system.md |
| `protocols/structure-authority.md` | model/structure.md | merged-into model/structure.md |
| `protocols/structure-load-save-contract.md` | model/structure.md | merged-into model/structure.md |
| `protocols/structure-periodicity.md` | model/structure-periodicity.md | merged-into model/structure-periodicity.md |
| `protocols/test-strategy.md` | process/testing.md | merged-into process/testing.md |
| `protocols/transiesta-workflow.md` | engines/transport.md (merged) | merged-into engines/transport.md |
| `protocols/ui-design-contract.md` | web/ui-contract.md | merged-into web/ui-contract.md |
| `protocols/vibrationview.md` | web/vibrationview.md | moved → web/vibrationview.md |
| `protocols/web-api.md` | web/web-api.md | merged-into web/web-api.md |
| `protocols/web-module-map.md` | web/overview.md | merged-into web/overview.md |
| `protocols/web-ui-coherence.md` | web/ (SPLIT: ui-contract.md + tabs.md) | merged-into web/{ui-contract,tabs}.md |
| `protocols/workspace-contract.md` | web/workspace.md | merged-into web/workspace.md |
| `results-tab-guide.md` | web/{results,presenters}.md | merged-into web/{results,presenters}.md |
| `roadmap.md` | roadmap.md + ops/installation.md | merged-into roadmap.md |
| `runtime-registry-guide.md` | web/runtime.md | merged-into web/runtime.md |
| `science.md` | **SPLIT** — §2 → science/chemistry-correctness.md; §1/§3–6 → science/overview.md | merged-into (split) |
| `staged-relaxation-guide.md` | execution/job-system.md | merged-into execution/job-system.md |
| `structure-guide.md` | web/tabs.md | merged-into web/tabs.md |
| `tabs/architecture.md` | web/tabs.md (+ runtime.md, job-contracts.md, roadmap.md) | merged-into web/tabs.md |
| `tabs/molbuilder.md` | web/tabs.md | merged-into web/tabs.md |
| `tabs/results.md` | web/results.md | merged-into web/results.md |
| `tabs/spectra/spec.md` | web/spectra.md | merged-into web/spectra.md |
| `tabs/structure-optimization.md` | web/tabs.md | merged-into web/tabs.md |
| `transport-guide.md` | engines/transport.md (merged) | merged-into engines/transport.md |
| `types/chemistry.md` | model/chemistry.md | merged-into model/chemistry.md |
| `types/parsers.md` | (retired — superseded by model/parse.md) | archived |
| `types/structure.md` | model/structure.md | merged-into model/structure.md |
| `workspace-guide.md` | web/workspace.md | merged-into web/workspace.md |
The dated `protocols/archive/*` records are historical artifacts already in
`docs/archive/`; they are not missing active contracts.

## Closeout acceptance criteria

- Every matrix row remains mapped; no active source calls a retired path
  authoritative.
- The stale-reference scan returns zero except documented archive-test
  allowlist entries.
- `python -m pytest -q tests/test_docs_structure.py tests/test_docs_tab.py`
  passes in the project test environment.
- The active index and TOC expose this audit.
- Each source-level plan has either a roadmap item with a test pin or an
  explicit retirement decision.
