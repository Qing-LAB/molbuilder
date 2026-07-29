# molbuilder — the reuse map (task → tool)

**Role:** reference
**Domain:** *(root — the spine)*
**Companions:** [`backend-architecture.md`](?doc=backend-architecture.md) (the
same backend by *functional concern*, the paired lens to this layer index);
[`design.md`](?doc=design.md) (mission · principles · decisions — the narrative
sibling); [`README.md`](?doc=README.md) (the doc index + the rules);
[`roadmap.md`](?doc=roadmap.md) (open work);
[`process/package-layout.md`](?doc=process/package-layout.md) (where each file
lives); [`process/conventions.md`](?doc=process/conventions.md) (the L1/L2/L3
layering rule + the provenance header this map rests on).

> **Read this BEFORE building anything.** It is the single map of the major
> infrastructure, modules, and APIs that already exist — so we stop
> reinventing or patching around a tool we already own. It is an **index**:
> each entry gives the role, the layer, the public API entry point, and the
> **authoritative doc** to read for detail. When the linked doc and this
> summary disagree, the linked doc wins — keep this one thin and route to it.

---

## 1. The rule: find it here before you build it

Before writing a new helper, doctor, parser, launcher, config reader, or
persistence format, **find the capability in § 2 (task → tool) or § 3
(subsystem index) and reuse it.** If nothing fits, build the new thing as
**shared infrastructure** — a named module with its own doc and named
adopters — not a local patch. (This is the standing "don't reinvent wheels"
principle; the narrative rationale lives in [`design.md`](?doc=design.md).)

Two habits make the reuse rule work:

- **Read the provenance header first, never judge code dead from a glance.**
  The house style opens each code file with a `MODULE · ROLE · USED-BY`
  header naming who depends on it. It is **advisory today** (no test pins it;
  partial adoption — see [`process/conventions.md`](?doc=process/conventions.md)
  § 2), but it is still the first thing to read: trace the callers before you
  decide a function is removable. A surface glance is not evidence.
- **The two surfaces are the only top layer.** `cli.py` and `web/` (L3) call
  the *same* lower-layer verbs — never a private copy. If you are about to
  hand-roll logic in a blueprint or a CLI command, the verb you want almost
  certainly already exists one layer down. This is enforced
  (`tests/test_layering.py`); § 3 is organised by that layer.

---

## 2. Task → existing tool (check here first)

| I need to… | Use | Do NOT | Detail |
|---|---|---|---|
| Check an env is present / has the right GPU · CUDA · ELPA codepath | `molbuilder envs {advise,bootstrap,clean,doctor,install,list,repair,validate}` (`envs/`); `scripts/install-env.sh bootstrap` is the shell wrapper that hands off to `envs bootstrap` | write a new doctor / checker | [`ops/installation.md`](?doc=ops/installation.md) |
| Read `molbuilder.json` (scheduler · routing · activation · script-gen) | `runtime_config.get_scheduler` / `get_routing` / `get_script_generation` / `require_activation` / `write_config_scope` | re-parse the JSON yourself | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 5; server keys → [`ops/deployment.md`](?doc=ops/deployment.md) |
| Emit a run wrapper / `.sbatch` for a job | `runwrap.write_run_wrapper` / `render_sbatch` / `write_sbatch` | hand-write shell / sbatch | [`execution/running-a-job.md`](?doc=execution/running-a-job.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| Persist a versioned JSON artifact (`molbuilder/<name>@<major>`) | `persist.schema_major` / `check_schema_major` / `read_json` / `write_json` (atomic) | hand-roll the schema check + IO | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) (data vocabulary) |
| Persist the user's *unsaved* browser edits across tab switches / reload | the **workspace session state-timeline** (`lib/workspace/` → `/api/state-timeline/*`) — format-blind opaque snapshots | auto-save to, or directly bond an edit to, the project `.xyz` / `.json` | [`web/workspace.md`](?doc=web/workspace.md) |
| Save the edited structure to a project file (overwrite or save-as) | **`projects.parser.saveMolecule`** → `/api/structure/save` (server writes the `.xyz`+sidecar pair) | a bespoke second file stack | [`web/projects.md`](?doc=web/projects.md) |
| Parse an engine `.out` / dir / sidecar into typed data | `parse.registry.parse` / `parse_dir` / `parse_text`; `parse.dirs.job.decode_run_dir` (→ `JobResult`) | write a bespoke parser | [`model/parse.md`](?doc=model/parse.md); decoded-run view → [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |
| Run a *set* of related jobs (stage ladder / sweep) | the `jobset/` framework + `molbuilder jobset {plan,prep,status,submit}` | reimplement dir isolation / sbatch chaining | [`execution/job-system.md`](?doc=execution/job-system.md) |
| Benchmark GPU/CPU knobs on a target | `molbuilder bench {generate,prep,summarize,prep-run,…}` (`bench/`) | reinvent the sweep / adapters | [`execution/job-system.md`](?doc=execution/job-system.md) |
| Snapshot / restore / branch a run dir (with big-binary safety) | `checkpoint.Repo` + `molbuilder snapshot {init,checkpoint,tag,branch,restore,config,list}` | shuffle files by hand | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 6 |
| Hand a finished run to the next workflow (relax → transport / spectra) | `bundle_writer.write_bundle_as_handoff` (`.xyz` + `.molstruct.json`) | copy geometry ad hoc | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) (handoff bundle) |
| Build a structure (peptide / DNA / RNA / SMILES / name) | `molbuilder {peptide,dna,rna,smiles,name}` (`peptide.py` / `nucleic.py` / `smiles.py` / `pubchem.py` / `builders/backends/`) | new geometry code | [`engines/builders.md`](?doc=engines/builders.md) |
| Emit a SIESTA / PySCF input | `siesta.input.render_fdf` · `pyscf.input.render_script` from `SiestaConfig` / `PySCFConfig` | template strings | [`engines/siesta.md`](?doc=engines/siesta.md), [`engines/pyscf.md`](?doc=engines/pyscf.md) |
| Validate chemistry / open-shell / pseudos before emit | `validation/` (the per-engine `validate()` pass), `chemistry.analyze_structure`, `pseudos.check_coverage` | ad-hoc checks | [`science/validation.md`](?doc=science/validation.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md), [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) |
| Detect host capabilities (cores / GPU / env-for-category) | `diagnostics.get_capabilities().env_for_category(...)` | probe by hand | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) (runtime resolution) |

---

## 3. Subsystem index (by layer)

**Layer key.** **L1** core types (import nothing above them) · **L2** domain
verbs (may import L1) · **L3** surfaces (cli / web; may import both). This is
the load-bearing invariant — it is what stops the registry/abstraction tangle
from growing back — and it is enforced by `tests/test_layering.py`, which
classifies *every* top-level name so a new module can't silently escape a
layer decision.

```mermaid
flowchart TB
  subgraph L3["L3 · surfaces — the only top layer"]
    CLI["cli.py"]
    WEB["web/ (Flask blueprints)"]
  end
  subgraph L2["L2 · domain verbs"]
    ENG["siesta/ · pyscf/ · transport/ · builders/"]
    EXE["jobset/ · bench/ · runwrap · runtime_config · envs/"]
    RW["parse/ · sidecars/ · script_emit · bundle_writer · validation/"]
  end
  subgraph L1["L1 · core types (the lingua franca)"]
    T["structure · frame · selection · config/ · issues · persist · chemistry · pseudos · checkpoint"]
  end
  L3 -->|calls the same verbs| L2
  L2 -->|reads/writes| L1
```

The four core types (`Structure`, `Frame`, `Config`, `Issue`) are the wire
between subsystems: construction emits a `Structure`; validation reads a
`Structure`+`Config` and returns `List[Issue]`; the engine emitters render a
job from `Structure`+`Config`; data management owns the round-trip of all of
them to and from disk. (For the same backend seen through the *functional-
concern* lens — data · construction · validation · execution, and where those
concerns leak into each other — see
[`backend-architecture.md`](?doc=backend-architecture.md).)

### Execution & scheduling

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `jobset/` | L2 | engine-agnostic **staged execution**: a set of related jobs sharing a package | `stages_to_jobset`; `prep_jobset`; `submit_jobset(mode=…)`; `jobset_status`; `render_plan`; `JobSet.write` / `load`; CLI `molbuilder jobset {plan,prep,status,submit}` | [`execution/job-system.md`](?doc=execution/job-system.md) |
| `bench/` | L2 | portable **benchmark** sweep (detect → format → run → summarize); a `jobset` producer (`sweep_to_jobset`) | `molbuilder bench {generate,prep,summarize,prep-run,…}`; `sweep_grid` (shared grid) | [`execution/job-system.md`](?doc=execution/job-system.md) |
| `runwrap` | L2 | **launcher** emitter: `.run.sh` + `.sbatch` (env activation, MPI/OMP, mem, carry-localize) | `write_run_wrapper(…, carry_in=)`, `render_sbatch`, `write_sbatch` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md), [`execution/job-system.md`](?doc=execution/job-system.md) |
| `runtime_config` | L2 | reader for `molbuilder.json` (scheduler / routing / script-gen) | `get_scheduler`, `get_routing`, `get_script_generation`, `require_activation`, `write_config_scope` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 5 |
| `diagnostics` | L2 | host capability detection + env-for-category routing | `get_capabilities().env_for_category(...)` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |
| `monitor` | L2 | stdlib-only progress/utilization sampler shipped next to jobs (`mb_monitor.py`) | copied verbatim to targets | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) |

**The start-here map for *running* a molbuilder-generated job** on any target
(single-task everywhere · JobSet from the CLI · the browser job system as the
target) is [`execution/overview.md`](?doc=execution/overview.md) — the
current → target status matrix.

### Persistence, parsing, data exchange

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `persist` | L1 | shared **versioned-doc** schema check + atomic JSON IO | `schema_major`, `check_schema_major`, `read_json`, `write_json` | [`execution/job-contracts.md`](?doc=execution/job-contracts.md) |
| `parse/` | L2 | unified **read stack** (File / Text / Dir parsers → typed `ParseResult`) | `parse.registry.{parse,parse_dir,parse_text}`; `parse.dirs.job.decode_run_dir` (→ `JobResult`) | [`model/parse.md`](?doc=model/parse.md) |
| `sidecars/`, `script_emit`, `bundle_writer` | L2 | write-side JSON sidecars + run-bundle handoff | `sidecars.{to_dict,save,load,apply_to_structure}`; `script_emit.emit_*`; `bundle_writer.write_bundle_as_handoff` | sidecar → [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md); blocks + handoff → [`execution/job-contracts.md`](?doc=execution/job-contracts.md) |
| `config/` | L1 | the engine-knob **dataclasses** (`SiestaConfig` / `PySCFConfig` / `SpectraConfig` / `TransportConfig`) — the lingua franca | `config.siesta.SiestaConfig`, `config.pyscf.PySCFConfig`, … | [`engines/`](?doc=engines/overview.md); the JS form built from them → [`web/form-schema.md`](?doc=web/form-schema.md) |

### Safety, checkpoints, validation

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `checkpoint` | L1 | git-based run-dir **snapshot/restore** with sha-archived big binaries (engine-aware via parameterized glob tables, safety-critical) | `Repo.{init(engine=),checkpoint,tag,branch,restore,list_checkpoints,state,set_archive_globs}`; CLI `molbuilder snapshot …` | [`execution/running-a-job.md`](?doc=execution/running-a-job.md) § 6 |
| `validation/` | L2 | scientific-correctness analyzers + the per-engine `validate()` pass | `validation.validate(struct, cfg, prior=…)` (one gate per engine) | [`science/validation.md`](?doc=science/validation.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) |
| `pseudos` | L1 | PSML pseudopotential parse + coverage/version checks (C1–C6) | `pseudos.check_coverage` | [`science/pseudopotentials.md`](?doc=science/pseudopotentials.md) |
| `chemistry`, `residues` | L1 | structure analysis (open-shell, charge, residues) | `chemistry.analyze_structure` (→ `ChemistryAnalysis`) | [`model/chemistry.md`](?doc=model/chemistry.md), [`science/chemistry-correctness.md`](?doc=science/chemistry-correctness.md) |

### Environments, engines, builders

| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `envs/` | L2 | the **environments toolkit** (presence + verify-cmd + GPU / CUDA / ELPA readiness) | `molbuilder envs {advise,bootstrap,clean,doctor,install,list,repair,validate}` | [`ops/installation.md`](?doc=ops/installation.md); NEVER build a new doctor |
| `siesta/`, `pyscf/` | L2 | per-engine input emitters + stage rendering | `siesta.input.render_fdf` / `render_siesta_stage_fdfs`; `pyscf.input.render_script` | [`engines/siesta.md`](?doc=engines/siesta.md), [`engines/pyscf.md`](?doc=engines/pyscf.md) |
| `builders/`, `peptide/`, `nucleic`, `smiles`, `pubchem` | L2 | structure synthesis | `build_peptide` / `build_dna` / `build_rna` / `build_from_smiles` / `build_from_name` | [`engines/builders.md`](?doc=engines/builders.md) |
| `transport/` | L2 | TranSIESTA multi-run workflow + consistency preflight | `molbuilder transport …` | [`engines/transport.md`](?doc=engines/transport.md) |

### Core types (L1) & surfaces (L3)

- **L1 core types**: `structure`, `frame`, `issues`, `selection`,
  `runtime_info`, `trajectory_log` — the data model.
  See [`model/overview.md`](?doc=model/overview.md) and
  [`model/structure.md`](?doc=model/structure.md).
- **L3 surfaces**: `cli` (`molbuilder …`; the thin-shell-over-the-web-API
  doctrine + the full command catalogue →
  [`process/conventions.md`](?doc=process/conventions.md) § 3) and `web`
  (Flask blueprints → [`web/web-api.md`](?doc=web/web-api.md); the whole front
  end → [`web/overview.md`](?doc=web/overview.md)).

---

## 4. Persisted artifacts & schemas

The concentrated registry of on-disk names + the `molbuilder/<name>@<major>`
schema strings + the config↔scheduler parameter vocabulary is the **data
vocabulary** in [`execution/job-contracts.md`](?doc=execution/job-contracts.md).
New persisted artifacts MUST use `persist.check_schema_major` and be registered
there. The structure save file itself (`.molstruct.json`, its envelope and
schema versions v3–v6) is [`model/structure-molstruct.md`](?doc=model/structure-molstruct.md).

---

## 5. Where the deeper design lives

This index is deliberately thin — it routes you to the authoritative doc.

- **The narrative design** — mission, the L1/L2/L3 architecture, the design
  principles, the anti-patterns we refuse, and the decisions index — is
  [`design.md`](?doc=design.md), the narrative spine sibling.
- **The concern lens** — the same backend by *functional concern* (data ·
  construction · validation · execution), which concern owns each module, and
  where the concerns leak into each other — is
  [`backend-architecture.md`](?doc=backend-architecture.md), the companion to
  this layer index.
- **The domain docs are the authoritative per-subsystem source** for every row
  above: [`model/`](?doc=model/overview.md) (the L1 data model),
  [`science/`](?doc=science/overview.md) (correctness),
  [`engines/`](?doc=engines/overview.md) (the emitters + builders),
  [`execution/`](?doc=execution/overview.md) (running jobs),
  [`web/`](?doc=web/overview.md) (the front end + web API),
  [`ops/`](?doc=ops/installation.md) (install + serve),
  [`process/`](?doc=process/package-layout.md) (conventions · testing · audit ·
  package layout).
- **The forward plan** — every open feature/backend workstream, including the
  execution↔engine decoupling items and the front-end ESM conversions — is
  [`roadmap.md`](?doc=roadmap.md). Closed decisions live in [`design.md`](?doc=design.md).

Keep this map in sync when a **major** subsystem or public entry point is
added; per-detail changes belong in the linked docs, not here.
