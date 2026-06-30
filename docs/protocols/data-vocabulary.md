# Data vocabulary & persisted formats — the system's shared language

> **Concentrated, authoritative definition** of the names and JSON formats
> molbuilder subsystems use to **exchange key information**. The rule: if
> two files name the same concept, they use the name defined here. Every
> persisted artifact follows the schema convention here. This doc is the
> place to look up "what is this field called, system-wide?" — the other
> docs reference it rather than re-defining names.
>
> Maintained because the names DID drift (e.g. a job-set field once read
> `omp`/`walltime` while every other exchange file said `cpus_per_task`/
> `time`, 2026-06-30). One language prevents that.

---

## § 1 Persisted artifacts (the files that carry key info)

| Artifact | File | Schema string | Authoritative doc | Key top-level fields |
|---|---|---|---|---|
| User config | `molbuilder.json` / `.molbuilder.json` | *(validated, no `@N`)* | `config.md` | `scheduler{directives,gpu,defaults,routing}`, `execution`, `script_generation`, `envs` |
| Detected environment | `environment.json` | `molbuilder/environment@1` | `benchmark-workflow.md`, `bench/environment.py` | `scheduler`, `topology`, `site` |
| Benchmark manifest | `bench-manifest.json` | `molbuilder/bench-manifest@2` | `benchmark-workflow.md`, `bench/generate.py` | `points.{cpu,gpu}` |
| Benchmark result | `bench-result.json` | `molbuilder/bench-result@1` | `benchmark-workflow.md`, `bench/result.py` | `points`, `choice`, `recommend` |
| **Job-set plan** | `job-set.json` | `molbuilder/job-set@1` | `staged-execution.md` | `shared`, `jobs[]` |
| Decoded run | `decoded.json` | `schema_version: <int>` *(predates the `@major` convention)* | `job-decoder.md` | `schema_version`, decoded plots, job-type, triggers |
| Workflow handoff | `<stem>.xyz` + `<stem>.molstruct.json` | *(sidecar pair)* | `bundle-contract.md`, `bundle_writer.py` | geometry; `regions`/`frozen_atoms`/`structure_hash` |
| Checkpoint binary archive | `.binsnapshots/<sha>/MANIFEST` | *(3-col `<sha256> <bytes> <name>`)* | `run-checkpoints.md` § 10 | — |

**Schema-string convention.** `molbuilder/<name>@<major>`. Readers check the
**major only** (tolerate same-major minor bumps, reject a different major) —
implemented identically in `bench/environment.py`, `bench/result.py`, and
`jobset/model.py`. New persisted artifacts MUST follow this. The one
exception is `decoded.json`, which predates the convention and carries a
bare integer `schema_version` (`job-decoder.md`); not worth a breaking
change, but don't copy that pattern for anything new.

---

## § 2 The canonical parameter vocabulary

There are **two layers** with a deliberate, documented translation between
them. Within a layer, **one concept = one name**:

- **config layer** — the scientific dataclasses the user sets
  (`SiestaConfig` / `PySCFConfig`). Vocabulary tuned for the scientist.
- **exchange/scheduler layer** — the persisted artifacts (manifests,
  `job-set.json`) + the SLURM flags they become. Vocabulary tuned for the
  scheduler. Persisted files and `jobset.Resources` use THIS column.

| Concept | config-layer name | exchange / SLURM name | producer translates at |
|---|---|---|---|
| MPI ranks | `mpi_np` | `mpi_np` → `-n` | *(same name)* |
| OMP cores per rank | `omp_threads` | **`cpus_per_task`** → `-c` | `stages_to_jobset`, `render_sbatch` |
| Walltime | *(none; `defaults.time`)* | **`time`** → `-t` | — |
| Memory | `max_memory_mb` (cap) / `defaults.mem` | `mem` → `--mem` | `render_sbatch` (estimate) |
| Whole-node | `gpu.exclusive` | `exclusive` → `--exclusive` | — |
| Partition | `directives.partition` | `partition` → `-p` | resolved from `domain` |
| QoS | `directives.qos` | `qos` → `-q` | resolved from `domain` |
| Routing domain (named menu pick) | `scheduler.routing[].name`, `execution.domain` | `domain` (in `jobset.Resources`) | `--domain` → `-p`/`-q` |
| GPU request | `enable_gpu` + `diag_algorithm` | `gres` → `--gres` | derived from `.fdf` + GPU type |
| Eigensolver | `diag_algorithm` (`ScaLAPACK`/`ELPA-1STAGE`/`ELPA-2STAGE`) | `.fdf`: `Diag.Algorithm` | `render_fdf` |
| Non-convergence policy | `on_nonconvergence` (`proceed`/`continue`/`halt`) | `dep_kind` (`afterok`/`afterany`) | `stages_to_jobset` (§ 8) |

**The translation rule:** persisted/exchange files use the exchange-layer
name; a *producer* maps config→exchange at its boundary (e.g.
`stages_to_jobset` maps `SiestaConfig.omp_threads` → `cpus_per_task`). Never
mix the two within one file.

---

## § 3 Identifier & path conventions

| Convention | Form | Used for |
|---|---|---|
| **Project ID** | SIESTA `SystemLabel` / PySCF `JOB = "…"` | keys warm-restart files `<ID>.<ext>` (`script-execution.md`) AND the SLURM `-J` job name |
| **Warm-restart files** | `<ID>.XV` / `.DM` / `.CG` (SIESTA); `<ID>.chk` / `<ID>_optimized.xyz` (PySCF) | engine-native resume (`script-execution.md`) |
| **Per-job directory** | `point-<name>/` | benchmark `point-G<g>K<k>C<c>/`; stage ladder `point-stage<N>/` (`staged-execution.md`) |
| **SLURM job name** | `-J <ID>` (single) / `-J <ID>-G<g>K<k>C<c>` or `-J job-stage<N>` (per-job) | `squeue` differentiation (`slurm-integration.md` § 4.4) |
| **Dependency kind** | `afterok` / `afterany` | stage chaining (`staged-execution.md` § 8) |

---

## § 4 How to use this doc

- Adding a field to a persisted artifact → use the **exchange-layer** name
  from § 2 (or add a row if it's a new concept). Don't invent a synonym.
- Adding a new persisted artifact → follow the § 1 schema convention and
  add a row.
- A producer that reads config and writes an exchange file → it is the
  translation point; cite this doc at that boundary.
