# molbuilder — architecture index (the design-foundation reference)

> **Read this BEFORE building anything.** It is the single map of the major
> infrastructure, modules, and APIs that already exist — so we stop
> reinventing or patching without knowing the tools. It is an **index**: each
> entry gives the role, layer, the public API entry points, and the
> **authoritative doc** to read for detail. When in doubt, the linked doc
> wins over this summary.
>
> Companions: `design.md` (§ 0 doc index + narrative Architecture + roadmap +
> decisions), `README.md` (doc-folder map), `protocols/data-vocabulary.md`
> (the names/JSON formats exchanged between subsystems). The module→layer
> mapping here is enforced by `tests/test_layering.py`.

---

## 0. How to use this (the reuse-first rule)

Before writing a new helper, doctor, parser, launcher, config reader, or
persistence format: **find the capability in § 1 (task → tool) or § 2
(subsystem index) and reuse it.** If nothing fits, build the new thing as
**shared infrastructure** (named adopters, its own doc), not a local patch —
see `design.md` principle 8 ("Don't reinvent wheels").

---

## 1. Task → existing tool (check here first)

| I need to… | Use | Do NOT |
|---|---|---|
| Check an env is present / has the right GPU/CUDA/ELPA codepath | `molbuilder envs doctor/validate/advise/repair/install/bootstrap/list/clean` (`envs/`) | write a new doctor/checker |
| Read `molbuilder.json` (scheduler, routing, activation, script-gen) | `runtime_config.get_scheduler/get_routing/get_script_generation/require_activation` | re-parse the JSON yourself |
| Emit a run wrapper / `.sbatch` for a job | `runwrap.write_run_wrapper` / `render_sbatch` / `write_sbatch` | hand-write shell/sbatch |
| Persist a versioned JSON artifact (`molbuilder/<name>@<major>`) | `persist.check_schema_major/schema_major/read_json/write_json` | hand-roll the schema check + IO |
| Parse an engine `.out`/dir/sidecar into typed data | `parse.registry.parse/parse_dir/parse_text`; `parse.dirs.job.decode_run_dir` | write a bespoke parser |
| Run a *set* of related jobs (stage ladder / sweep) | the `jobset` framework + `molbuilder jobset plan/prep/status/submit` | reimplement dir isolation / sbatch chaining |
| Benchmark GPU/CPU knobs on a target | `molbuilder bench generate/prep/summarize/prep-run` (`bench/`) | reinvent the sweep/adapters |
| Snapshot / restore / branch a run dir (with big-binary safety) | `checkpoint.Repo` + `molbuilder snapshot init/checkpoint/tag/branch/restore/config` | shuffle files by hand |
| Hand a finished run to the next workflow (relax→transport/spectra) | `bundle_writer.write_bundle_as_handoff` (`.xyz` + `.molstruct.json`) | copy geometry ad hoc |
| Build a structure (peptide/DNA/RNA/SMILES/name) | `molbuilder peptide/dna/rna/smiles/name` (`builders/`, `peptide/`, …) | new geometry code |
| Emit a SIESTA/PySCF input | `siesta.input` / `pyscf.input` from `SiestaConfig`/`PySCFConfig` | template strings |
| Validate chemistry / open-shell / pseudos before emit | `validation/`, `chemistry.analyze_structure`, `pseudos.check_coverage` | ad-hoc checks |
| Detect host capabilities (cores/GPU/env-for-category) | `diagnostics.get_capabilities` | probe by hand |

---

## 2. Subsystem index (major infrastructure)

Layer key: **L1** core types (no L2/L3 imports) · **L2** domain verbs (may use
L1) · **L3** surfaces (cli/web). Enforced by `tests/test_layering.py`.

### Execution & scheduling
| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `jobset/` | L2 | engine-agnostic **staged execution**: a set of related jobs sharing a package | `stages_to_jobset`; `prep_jobset`; `submit_jobset(mode,domain,dry_run)`; `jobset_status`; `render_plan`; `JobSet.write/load`; CLI `molbuilder jobset {plan,prep,status,submit}` | `protocols/staged-execution.md`; user: `staged-relaxation-guide.md` |
| `bench/` | L2 | portable **benchmark** sweep (detect→format→run→summarize) | `molbuilder bench {generate,prep,summarize,prep-run,siesta-gpu,probe-scheduler}`; adapters `format_bench`/`format_run` | `protocols/benchmark-workflow.md` |
| `runwrap` | L2 | **launcher** emitter: `.run.sh` + `.sbatch` (env activation, MPI, mem, carry-localize) | `write_run_wrapper(...carry_in=)`, `render_sbatch`, `write_sbatch` | `protocols/slurm-integration.md`, `protocols/script-execution.md` |
| `runtime_config` | L2 | reader for `molbuilder.json` (scheduler/routing/script-gen) | `get_scheduler`, `get_routing`, `get_script_generation`, `require_activation`, `write_config_scope` | `config.md`, `protocols/slurm-integration.md` |
| `monitor` | L2 | stdlib-only progress/utilization sampler shipped next to jobs (`mb_monitor.py`) | `monitor` module (copied verbatim to targets) | `protocols/benchmark-workflow.md` § 9 |

### Persistence, parsing, data exchange
| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `persist` | L1 | shared **versioned-doc** schema check + JSON IO (atomic) | `check_schema_major`, `schema_major`, `read_json`, `write_json` | `protocols/data-vocabulary.md` § 1 |
| `parse/` | L2 | unified **parse stack** (File/Text/Dir parsers → typed `ParseResult`) | `parse.registry.{parse,parse_dir,parse_text}`; `parse.dirs.job.decode_run_dir` (→ `JobResult`) | `protocols/parse-module.md`, `protocols/job-decoder.md` |
| `sidecars/`, `script_emit`, `script_bundle`, `bundle_writer` | L2 | write-side JSON sidecars + run-bundle handoff | `bundle_writer.write_bundle_as_handoff` | `protocols/sidecar-contract.md`, `protocols/bundle-contract.md`, `protocols/script-contract.md` |
| `config/` | L1 | the **dataclasses** (SiestaConfig/PySCFConfig/spectra/transport) — the lingua franca | `config.siesta.SiestaConfig`, `config.pyscf.PySCFConfig`, … | `config.md`, `engines/*.md` |

### Safety, checkpoints, validation
| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `checkpoint` | L1 | git-based run-dir **snapshot/restore** with sha-archived big binaries (engine-aware, safety-critical) | `Repo.{init(engine=),checkpoint,tag,branch,restore,list_checkpoints,state,archive_globs,set_archive_globs}`; CLI `molbuilder snapshot …` | `protocols/run-checkpoints.md` (safety contract § 4.6, § 9) |
| `validation/` | L2 | scientific-correctness analyzers + engine adapters | `validation` (analyzer/adapters/validators) | `protocols/scientific-validation.md`, `protocols/chemistry-correctness.md` |
| `pseudos` | L1 | PSML pseudopotential parse + coverage/version checks (C1–C6) | `pseudos.check_coverage` | `protocols/pseudopotential-validation.md` |
| `chemistry`, `residues` | L1 | structure analysis (open-shell, charge, residues) | `chemistry.analyze_structure` | `protocols/chemistry-correctness.md` |

### Environments, engines, builders
| Module | L | Role | Public API entry points | Doc |
|---|---|---|---|---|
| `envs/` | L2 | the **environments toolkit** (presence + verify-cmd + GPU/CUDA/ELPA readiness) | `molbuilder envs {advise,bootstrap,clean,doctor,install,list,repair,validate}` | `README_install.md`; NEVER build a new doctor |
| `diagnostics` | L2 | host capability detection + env-for-category routing | `get_capabilities().env_for_category(...)` | `package-layout.md` |
| `siesta/`, `pyscf/` | L2 | per-engine input emitters + stage rendering | `siesta.input.render_fdf`/`render_siesta_stage_fdfs`; `pyscf.input` | `engines/siesta.md`, `engines/pyscf.md` |
| `builders/`, `peptide/`, `nucleic`, `smiles`, `pubchem` | L2 | structure synthesis | `molbuilder {peptide,dna,rna,smiles,name}` | `design.md` Architecture |
| `transport/` | L2 | TranSIESTA multi-run workflow + consistency preflight | `molbuilder transport …` | `protocols/transiesta-workflow.md`, `engines/transport.md` |

### Core types (L1) & surfaces (L3)
- **L1 core types**: `structure`, `frame`, `issues`, `selection`, `runtime_info`, `trajectory_log`, `script_contract` — see `design.md` Architecture › Core types.
- **L3 surfaces**: `cli` (`molbuilder …`, `protocols/cli.md`) and `web` (Flask blueprints, `protocols/web-api.md` — 69 routes; UI tabs in `tabs/`).

---

## 3. Persisted artifacts & schemas

The concentrated registry of on-disk names + `molbuilder/<name>@<major>`
schema strings + the config↔exchange parameter vocabulary is
**`protocols/data-vocabulary.md`**. New persisted artifacts MUST use
`persist.check_schema_major` and be registered there.

---

## 4. Where the deeper design lives

- **`design.md`** — mission, the L1/L2/L3 Architecture in narrative form, the
  10 design principles, anti-patterns, and the decisions log.
- **`protocols/*.md`** — the per-subsystem contracts (the authoritative
  source for each row above).
- **`README.md`** — the doc-folder map + the "tests derivable from spec" rule.

This index is deliberately thin — it routes you to the authoritative doc.
Keep it in sync when a **major** subsystem or public entry point is added;
per-detail changes belong in the linked docs, not here.
