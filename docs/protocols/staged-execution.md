# Staged execution — the relaxation ladder as a job-set

> **Design doc (PROPOSED, 2026-06-29).** How a multi-stage SIESTA
> optimization (the `cfg.stages` ladder, engines/siesta.md "Staged
> optimization") executes on a real scheduler by **reusing the
> benchmark's job-set machinery** — per-job directories, symlink-shared
> package files, and per-job scheduler resources (routing/exclusive/mem/
> `-J`). Status per piece is in § 0; vocabulary in § 1 (read first).
> Cross-references: `slurm-integration.md` (§ 4.3 routing, § 4.3.1
> exclusive/mem, § 4.4 per-point names), `benchmark-workflow.md`
> (the `_mb_point` isolation + adapters this generalizes),
> `engines/siesta.md` (the stage data model + the monolithic runner this
> extends), `job-execution.md` (the bundle / prep / submit contract).

---

## § 0 Status

| Piece | State |
|---|---|
| Stage data model (`SiestaStageSpec`, `cfg.stages`) | **built** (`config/siesta.py`) |
| Stage validation (empty / all-disabled / duplicate-name / knobs), wired into `validate()` | **built** (`validate_siesta_stages`; `validation/siesta.py`) |
| Per-stage `.fdf` rendering (one fdf/stage, shared `SystemLabel`) | **built** (`siesta/input.py::render_siesta_stage_fdfs`) |
| **Monolithic runner** — all stages in ONE job, sequential bash loop, in-place `.XV` auto-restart | **built** (`render_siesta_stages_runner`); becomes the `direct` mode (§ 6) |
| Benchmark job-set isolation (`_mb_point`: per-job dir + `ln -sfn` shared files) | **built** (`bench/adapters.py`) |
| Scheduler routing/exclusive/mem/`-J` per job | **built** (slurm-integration.md § 4.3–4.4) |
| **`jobset` framework — data model + `job-set@1` persistence** | **built** (`jobset/model.py`) |
| **`jobset` materialize engine** — per-job dirs + shared/carry symlinks | **built** (`jobset/materialize.py`) |
| **`jobset` plan engine** — STAGE-PLAN / BENCH-PLAN table | **built** (`jobset/plan.py`) |
| **SIESTA stage producer** — `cfg.stages` → ladder `JobSet` (deps + carry) | **built** (`siesta/stages.py::stages_to_jobset`) |
| Per-stage resources | **built** as `Job.resources` + the `resources_for` producer seam; a per-stage UI/CLI source is **PROPOSED** (§ 3) |
| **`jobset` submit engine** — dependency-threaded sbatch driver | **PROPOSED** (§ 4; `jobset/submit.py`) |
| `on_nonconvergence` → SLURM dependency mapping | **built** in the producer (`_dep_kind`); consumed by the submit engine (§ 5) |
| CLI / prep wiring + bench migration onto the framework | **PROPOSED** (§ 8 D4) |

---

## § 1 Vocabulary and scope

- **Stage** — one tier of the relaxation ladder (`SiestaStageSpec`): a
  `relax_type`/`relax_steps`/`relax_force_tol`/`relax_max_displ` set with a
  `name` and an `on_nonconvergence` policy.
- **Job-set** — a collection of related SIESTA jobs that share one package
  (pseudopotentials, geometry, monitor) and each run in their own
  directory with their own scheduler resources. The **benchmark sweep**
  (points `point-G<g>K<k>C<c>/`) is the first job-set; a **stage ladder**
  (`point-stage<N>/`) is the second.
- **Carry-forward** — copying/linking a finished stage's restart files
  (`.XV`, `.DM`, `.CG`) into the next stage's directory so SIESTA
  warm-restarts the geometry. (Benchmark points have NO carry-forward —
  they are independent.)
- **Monolithic runner** — today's single-job bash loop over all stages
  (`render_siesta_stages_runner`): one allocation, one resource spec.

**In scope:** how the stage ladder maps onto the job-set machinery; the
per-stage resource override; the dependency-chain + carry-forward submit
path; preserving the monolithic runner as the simple fallback.

**Out of scope:** changing the stage *science* (knobs, warm-restart
semantics) — that's engines/siesta.md. No new scheduler backend (reuses
the slurm adapter). The benchmark sweep is unchanged.

---

## § 2 The unifying model — a stage ladder IS a job-set

A stage ladder and a benchmark sweep are the same shape: **N related
SIESTA jobs, one shared package, each isolated in its own dir, each with
its own scheduler resources.** The benchmark already built every hard
part; staging reuses it.

| Property | Benchmark sweep | Stage ladder |
|---|---|---|
| per-job directory | `point-G<g>K<k>C<c>/` | `point-stage<N>/` |
| shared files | `ln -sfn ../*.psml`, `../mb_monitor.py` (`_mb_point`) | **same** |
| per-job `.fdf` | one `job-gpu.fdf` symlinked, varied by `-n`/`-c` | one `job_stage<N>.fdf` per stage (relax knobs differ) |
| per-job name | `-J job-gpu-G<g>K<k>C<c>` (§ 4.4) | `-J job-stage<N>` |
| routing/exclusive/mem | per point (§ 4.3, § 4.3.1) | **per stage** (§ 3) |
| **relationship** | **independent → parallel** | **dependent → chained** (§ 4) |
| **data flow** | none | **`.XV` carry-forward** (§ 4) |

The last two rows are the only genuinely new mechanics; everything above
is reuse.

---

## § 2.1 Framework architecture — the `JobSet` abstraction

This is **not** a stage-specific patch onto the bench. The bench sweep and
the stage ladder are two **producers** of one data structure — `JobSet` —
consumed by shared, engine-agnostic **engines** (materialize / submit /
plan). Each module has a single responsibility and no knowledge of the
others' internals (logical isolation).

```
 PRODUCERS (engine knowledge)        CORE (engine-agnostic)
 ───────────────────────────        ──────────────────────────────────
 bench:  format_bench  ───┐
                          ├──►  JobSet  ──►  materialize  (dirs+symlinks+carry)
 siesta: stages_to_jobset ┘     (data)  ──►  submit       (dependency-threaded sbatch)
                                        ──►  plan         (BENCH-PLAN / STAGE-PLAN)
                                        ──►  job-set.json  (persistence, job-set@1)
```

**Data model** (`molbuilder/jobset/model.py` — pure dataclasses, no IO,
no scheduler, no filesystem):

- **`Resources`** — a per-job scheduler ask: `domain` (routing name,
  slurm-integration.md § 4.3) / `walltime` / `exclusive` / `mem` / `gres`
  / `mpi_np` / `omp`. All `Optional` (`None` = inherit the job-level
  default / per-job estimate — *assistant, not nanny*).
- **`Carry`** — one inter-job data-flow rule: `pattern` (e.g. `"*.XV"`)
  carried `from_job` (the producing job's name). The materialize engine
  lays it as a symlink; this is the abstraction of "shared information
  produced at runtime", distinct from the static shared package.
- **`Job`** — one unit of work: `name` (unique → its dir `point-<name>/`
  and its `-J`), `script` (its input filename), `resources: Resources`,
  `depends_on: Optional[str]` (producer job name), `dep_kind`
  (`"afterok"`/`"afterany"`), `carry: List[Carry]`.
- **`JobSet`** — `name`, `engine`, `kind` (`"sweep"`/`"ladder"`),
  `shared: List[str]` (static package files symlinked into EVERY job dir
  — pseudos, geometry, monitor), `jobs: List[Job]`, `to_json`/`from_json`.

**Shared-information abstraction (the heart of the design).** Information
shared across jobs is modeled in exactly two forms, nothing implicit:
1. **static package** → `JobSet.shared` (same bytes for every job; one
   set of symlinks),
2. **runtime-produced** → `Carry` (one job's output feeds another; a
   symlink resolved after the producer runs).
A producer declares both; no job reaches outside these two channels.

**Persistence** (`job-set@1`): a `JobSet` serializes to `job-set.json` in
the bundle — the declarative source of truth that materialize/submit/plan
all read, same lifecycle as `environment.json` / `bench-manifest.json`
(written at prep, consumed downstream; engine-neutral, self-describing,
versioned). The whole execution plan is inspectable as data, not buried
in a bash script.

**Engines (one responsibility each, engine-agnostic):**

| module | input | output | knows about |
|---|---|---|---|
| `jobset/model.py` | — | the dataclasses + `job-set@1` (de)serialize | nothing (pure data) |
| `jobset/materialize.py` | `JobSet` + target dir | `point-<name>/` dirs, `shared` + `carry` symlinks | filesystem only |
| `jobset/submit.py` | `JobSet` + `SchedulerAdapter` | dependency-threaded submit driver | the scheduler adapter only |
| `jobset/plan.py` | `JobSet` | the human PLAN table | nothing (formats data) |

**Producers (the ONLY place engine knowledge lives):**

- `bench/adapters.format_bench` → a **sweep** `JobSet` (independent jobs,
  empty `carry`).
- `siesta/stages.py::stages_to_jobset(cfg)` → a **ladder** `JobSet`
  (chained jobs; `dep_kind` from `on_nonconvergence`, § 5; `carry` per
  § 8 D1).

This is the systematic framework: producers know engines, the core knows
data, and `Carry`/`shared` are the only sanctioned shared-information
channels. The bench migrates onto it (§ 8 D4) — it does not get a parallel
copy.

---

## § 2.2 Workflow — from `cfg.stages` to a running chain

The pipeline reuses the bundle lifecycle of `job-execution.md`
(**generate → prep → submit**); the `jobset` framework is the spine.
Each step has one owner module and one artifact, so the whole flow is
inspectable as data, not buried in a script.

```
 cfg.stages ──validate──► JobSet ──persist──► job-set.json ──ship──► TARGET
   (HOST)                (producer)          (job-set@1)            │
                                                                    ▼
                              materialize ──► point-<stage>/ dirs ──► submit ──► chain runs
                              (+ per-stage .sbatch, STAGE-PLAN.md)   (deps + carry)
```

| # | Step | Where | Owner | Artifact | Status |
|---|---|---|---|---|---|
| 1 | **Author + validate** — set `cfg.stages` (+ `execution.mode`); `validate()` blocks a broken ladder at the Build tab / CLI | HOST | `validation/siesta.py` → `validate_siesta_stages` | issues | **built** |
| 2 | **Produce** — `stages_to_jobset(cfg, shared=…)` → `JobSet`; render one `<label>_<stage>.fdf` per enabled stage | HOST (generate) | `siesta/stages.py`, `siesta/input.py::render_siesta_stage_fdfs` | `JobSet` + per-stage `.fdf` | **built** (producer); fdf render **built** |
| 3 | **Persist** — `JobSet.to_dict()` → `job-set.json` in the bundle | HOST (generate) | `jobset/model.py` | `job-set.json` (`job-set@1`) | **built** (model); bundle write **proposed** |
| 4 | **Ship** — copy the bundle (fdfs + `job-set.json` + shared package + entry shims) to the target | — | (scp / bundle) | bundle on target | reuses job-exec |
| 5 | **Prep** — detect env/topology; **resolve** each `Job.resources` (domain→`-p`/`-q`, gres from fdf+GPU type, walltime, exclusive/mem); `materialize()` the `point-<stage>/` dirs; bake per-stage `.sbatch`; render `STAGE-PLAN.md` | TARGET | `jobset/materialize.py` + `SlurmAdapter` + `jobset/plan.py` | dirs + `.sbatch` + `STAGE-PLAN.md` | materialize/plan **built**; resource-resolve + sbatch bake **proposed** |
| 6 | **Submit** — walk the `JobSet`, one `sbatch` per job with `--dependency` threaded from `depends_on`/`dep_kind`; carry symlinks resolve as each stage writes `.XV`/`.DM`/`.CG` | TARGET | `jobset/submit.py` | queued chain | **proposed** (§ 4) |
| 7 | **Monitor** — per-stage dir outputs + `mb_monitor`; `squeue` shows `-J <label>_<stage>` rows; `STAGE-PLAN.md` is the map | TARGET | reuses bench monitor | logs | reuses job-exec |

**`direct` mode** (`execution.mode=direct`, § 6) collapses steps 5–6 into
the existing monolithic runner: all stages in one allocation, in-place
`.XV` auto-restart — no per-stage dirs or dependency chain. The same
`JobSet` describes both; only the engine that consumes it differs.

So the framework is the *single description* (`job-set.json`) that both
the direct runner and the submit chain read — the workflow is "produce one
JobSet, then pick an engine to run it."

---

## § 3 Per-stage resources — `SiestaStageSpec.resources`

The point of staging on a cluster is that **stages want different
resources**: a loose CG warm-up is cheap (short walltime, CPU/ScaLAPACK,
`htc`), a tight final Broyden is expensive (longer walltime, GPU/ELPA,
more memory). A single sbatch can't express that; a job-set can.

Add an **optional** per-stage override (absent → inherit the job-level
config / detected default — *assistant, not nanny*: no surprise resource
choices):

```python
@dataclass
class StageResources:
    walltime:       Optional[str]  = None   # SLURM -t for this stage
    domain:         Optional[str]  = None   # scheduler.routing name (§ 4.3)
    exclusive:      Optional[bool] = None   # § 4.3.1
    diag_algorithm: Optional[str]  = None   # ScaLAPACK / ELPA-1STAGE / -2STAGE
    enable_gpu:     Optional[bool] = None   # GPU accelerator for this stage
    # mpi_np / omp / mem inherit job-level + per-job estimate; override
    # only when a stage genuinely differs.
# SiestaStageSpec gains:  resources: Optional[StageResources] = None
```

Resolution (per stage, at prep): `stage.resources.<field>` → else the
job-level `SiestaConfig` value → else the detected/estimated default.
This is why the `diag_algorithm`/`enable_gpu` **decoupling**
(engines/siesta.md § 13) matters here: a stage can switch *solver and
hardware* — e.g. `stage1` ScaLAPACK-CPU warm-up → `stage3`
ELPA-GPU final — and each routes to the correct env automatically
(`_fdf_requests_elpa`/`_fdf_requests_gpu`, slurm-integration.md § 13).

Memory stays **per-job estimated** from each stage's `.fdf`
(slurm-integration.md § 4.3.1) — no flat per-stage memory knob.

---

## § 4 The submit path — dependency chain + carry-forward

Replace the monolithic single job (in `submit` mode) with **one sbatch
per stage**, chained by SLURM dependency, restart files carried forward.
Each stage dir is built by the SAME `_mb_point` mechanism, plus one extra
symlink for the prior stage's restart files.

**File layout** (a 3-stage ladder, `stage3` enabled):

```
job/                                # the bundle (shared package)
  C.psml  S.psml  Au.psml  H.psml   mb_monitor.py
  stage-submit                      # entry shim (bootstraps env, like prep-bench)
  point-stage1/
    job_stage1.fdf   -> ../*.psml symlinked;  -J job-stage1
    job_stage1.sbatch                # -t/-p/-q/--mem/--gres from stage1 resources
  point-stage2/
    job_stage2.fdf   -> ../*.psml;  <label>.XV -> ../point-stage1/<label>.XV
    job_stage2.sbatch                # --dependency=afterok:<stage1 jobid>
  point-stage3/  …                   # --dependency=afterok:<stage2 jobid>
```

**Submit driver (pseudocode)** — mirrors `run-bench`, adds dependency
threading:

```
enabled = [s for s in cfg.stages if s.enabled]          # validated (§0)
prev_jobid = None
for n, stage in enumerate(enabled, 1):
    d = f"point-stage{n}"
    mb_point(d)                                          # mkdir + ln shared *.psml, mb_monitor.py
    if prev_jobid is not None:                           # carry-forward
        ln -sfn ../point-stage{n-1}/<label>.XV  d/<label>.XV   # (+ .DM/.CG if present)
    res = resolve_resources(stage, cfg, env)            # § 3
    dep = f"--dependency={dep_kind(stage_prev)}:{prev_jobid}" if prev_jobid else ""
    jobid = sbatch {dep} {routing(res.domain)} {excl/mem/-t from res} \
                   -J job-stage{n}  job_stage{n}.sbatch   # § 4.3/4.3.1/4.4
    prev_jobid = jobid
```

`<label>` is `cfg.system_label` (every stage shares it, so SIESTA's
auto-restart finds `<label>.XV` in its own dir — exactly the monolith's
contract, now satisfied by a symlink across dirs instead of staying in
one dir).

---

## § 5 `on_nonconvergence` → SLURM dependency

The per-stage policy maps directly onto the dependency *kind* of the
NEXT stage's submission (no bash polling needed — SLURM enforces it):

| stage `on_nonconvergence` | next stage dependency | meaning |
|---|---|---|
| `halt` | `afterok:<jobid>` | next runs only if this stage SUCCEEDS; a failure cancels the rest of the chain (SLURM `DependencyNeverSatisfied`) — the publication-defensible default |
| `proceed` | `afterany:<jobid>` | next runs regardless (warm-up tiers: refine from "not bad") |
| `continue` | resubmit THIS stage `afterany` up to `continue_retries`, then fall through to `halt` | extend a cheap stage that almost converged |

The **last enabled stage** forces `halt` (engines/siesta.md): the ladder's
contract is to produce a converged answer or fail loud. This matches the
monolithic runner's force-halt-last rule, now expressed as "no downstream
job depends on the last stage."

---

## § 6 Two execution modes (reuses `execution.mode`)

Keyed off the existing `execution.mode` (job-execution.md § 8.13):

- **`direct`** — the **monolithic runner** (today's `render_siesta_stages_runner`):
  all stages in one process/allocation, in-place `.XV` auto-restart, one
  resource spec. Right for a workstation or a short ladder. **Unchanged.**
- **`submit`** — the **chain** (§ 4): one sbatch/stage, per-stage resources,
  dependency + carry-forward. Right for a cluster where stages differ in
  cost/hardware or the whole ladder would exceed one walltime.

So no behavior changes for existing single-allocation users; the chain is
additive, gated by the same mode switch the benchmark uses.

---

## § 7 What is reused vs new

**Reused as-is** (no new wheels): `_mb_point` dir+symlink isolation;
`SlurmAdapter` routing (`--domain`, slurm-integration.md § 4.3);
exclusive/mem per job (§ 4.3.1); per-job `-J` (§ 4.4); env routing on
ELPA/GPU (§ 13); `write_run_wrapper`/`render_sbatch`; the env-bootstrap
entry-shim pattern (job-execution.md § 8.3); per-job memory estimate.

**New** (small, contained): `StageResources` + `SiestaStageSpec.resources`
(§ 3); the carry-forward symlink step in `_mb_point` for stages; the
dependency-threading submit driver (§ 4); the policy→dependency mapping
(§ 5).

---

## § 8 Decisions (resolved 2026-06-30)

- **D1 — carry-forward set: `.XV` always; `.DM` iff `use_save_dm`; `.CG`
  iff the consecutive stages share `relax_type`.**
  `.XV` is the relaxed geometry — the point of chaining, always carried.
  `.DM` (SCF density matrix) warm-starts the next SCF and is carried when
  `cfg.use_save_dm` (SIESTA's `DM.UseSaveDM`, default on); same basis/mesh
  across stages (shared `SystemLabel`) makes it valid. `.CG` is the
  **optimizer** history and is algorithm-specific: carrying it across a
  `relax_type` switch (the common `CG`→`Broyden` ladder) is at best
  ignored, at worst wrong — so `.CG` is carried **only** when stage N and
  N+1 use the same `relax_type` (same-algorithm continuation warm-starts
  the optimizer; an algorithm switch restarts it fresh from the carried
  geometry). This is a deliberate, documented improvement over the
  monolith (which keeps `.CG` in-dir regardless).

- **D2 — `continue` is an in-`.run.sh` retry, NOT extra jobs.**
  One sbatch per stage. A stage whose policy is `continue` re-enters
  SIESTA up to `continue_retries` times from its own `.XV` inside that
  stage's wrapper (reusing the monolith's retry logic), then exits
  success/failure per the fall-through-to-`halt` rule. Keeps the
  dependency graph exactly one job per stage — no self-dependency loops.

- **D3 — yes: emit `STAGE-PLAN.md` at prep.** Mirrors `BENCH-PLAN.md`
  (job-execution.md § 8.4): a table of each stage's resolved
  domain/walltime/solver/hardware + the carry-forward set + the
  dependency graph + the `on_nonconvergence` policy, printed before
  submit so the per-stage choices and the chain are visible up front.

- **D4 — the `jobset` framework IS the core (built first), not a
  follow-up.** Per the architectural directive (2026-06-30), build
  `jobset/{model,materialize,submit,plan}.py` as the foundation
  (§ 2.1); `siesta/stages.py::stages_to_jobset` is its first producer.
  The bench then **migrates** onto it — `format_bench` returns a `JobSet`
  instead of a `{filename: text}` dict — as a fast-follow (low risk: the
  framework is additive and the migration is a producer swap, with the
  existing bench tests as the safety net). No parallel copy of the
  isolation/submit logic is created at any point.
