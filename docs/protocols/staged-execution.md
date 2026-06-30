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
| **Per-stage resource overrides** (`SiestaStageSpec.resources`) | **PROPOSED** (§ 3) |
| **Stage submit driver** — one sbatch/stage, dependency chain, `.XV` carry-forward | **PROPOSED** (§ 4) |
| `on_nonconvergence` → SLURM dependency mapping | **PROPOSED** (§ 5) |

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

## § 8 Open decisions

- **D1 — carry-forward set.** `.XV` is mandatory (geometry). Also link
  `.DM` (SCF warm-start, big file) and `.CG` (optimizer history)? Default
  YES for `.DM`/`.CG` when present; they cut the next stage's first-SCF
  cost. Confirm.
- **D2 — `continue` retries as jobs.** Resubmit-self via a small `afterany`
  self-dependency loop, or a single job with an internal retry? Lean:
  internal retry inside that stage's `.run.sh` (the monolith already has
  the retry logic) so the chain stays one-job-per-stage.
- **D3 — resource defaulting visibility.** Like the benchmark's BENCH-PLAN,
  emit a `STAGE-PLAN.md` showing each stage's resolved domain/walltime/
  solver/hardware + the dependency graph, so the per-stage choices are in
  front of the user before submit (discoverability — § 8.4 of job-exec).
- **D4 — should the benchmark and stage drivers share one `job-set`
  module?** They now differ only in (independent vs chained) + (carry-
  forward). A common `job_set.py` (`mb_point`, routing, submit) with two
  thin callers would prevent drift. Lean YES, but stage it AFTER the
  stage driver works standalone.
