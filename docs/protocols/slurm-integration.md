# SLURM integration — sbatch submission as a use case of the script generator

**Status**: design — pre-implementation. No code referenced here exists
yet. This document is the authoritative design for the SLURM/sbatch
submission layer; it is reviewed point-by-point before any code lands.
First target cluster: **ASU Sol**.

**Audience**: anyone implementing the submission-script layer, or
configuring molbuilder for a SLURM cluster.

---

## 0. How to read this doc — and its place in the framework

This is **not** a new generator. It is one **use case** layered on top
of the existing script-generator framework. The framework already has
three contracts; SLURM integration adds a fourth concern (job
submission) *without altering* the first three:

| Existing doc | Owns | This doc |
|---|---|---|
| [`config.md`](../config.md) | wrapper contract + config schema (`preamble`, `activation`) + refuse-to-emit + runtime scheduler-var reads (§ 1.5) | **extends** — adds a `scheduler` block; reuses refuse-to-emit |
| [`script-execution.md`](script-execution.md) | warm/cold-restart + project-ID inside `.run.sh` | **no change** — launcher untouched |
| [`script-contract.md`](script-contract.md) | the `.fdf`/`.py` *input-file* structure | **no change** |

**One-sentence framing**: today the `.run.sh` launcher is
scheduler-agnostic and the user hand-writes the `#SBATCH` header
(`job-layout.md` § "Run wrapper" literally says users "paste chunks
into a SLURM script"). SLURM integration generates that header from
config so a fresh-login user can `sbatch` a correct job with zero
hand-editing.

| If you are … | Start at |
|---|---|
| Reviewing the design | § 1, § 3, § 9 (decisions) |
| Configuring a site | § 4 |
| Implementing the emitter | § 5, § 6 |
| Checking Sol compatibility | § 7 (the constrained-env walkthrough) |
| Worried about GPU/CUDA/MPI correctness | § 7.5, § 7.9, § 8 (hardware), § 11 (benchmark) |

---

## 1. Mission and scope

> When a user generates a job on a SLURM cluster, molbuilder must emit
> a submission script that runs **correctly from a fresh login** (mamba
> not yet on PATH), allocates the **right CPU/GPU/memory resources**,
> and activates the **right env** — with no hand-edited `#SBATCH`
> directives — and is **submitted through the batch system** (never run
> on a login node).

**In scope**: a generated `<basename>.sbatch` carrying the `#SBATCH`
header; a `scheduler` config block; a shipped **`asu-sol` preset**;
per-job resource values derived from the `.fdf`/CLI; a **benchmark /
validation mode** (§ 11) that proves correctness + sizes resources.

**Explicitly NOT in scope** (keep it focused): no partition-selection
logic engine; no job-array / dependency orchestration beyond the
benchmark sweep; no querying SLURM at generate time (`sinfo`/`sacctmgr`)
— generation works offline on a laptop; no change to the `.run.sh`
launcher internals; **no multi-node MPI in v1** (single-node only).

---

## 2. Principles

Inherited from `config.md`:

1. **Fail at generate time, not submit time.** Missing/contradictory
   scheduler config errors while the user is at a terminal — never
   after a job has queued for hours.
2. **Baked, not detected.** Every `#SBATCH` value is substituted at
   generate time. No runtime `sinfo` probes in the `.sbatch`.
3. **The launcher owns env activation.** The `.sbatch` delegates to
   `.run.sh`; it never re-implements `module load` / `source activate`.
4. **Readable plain bash.** Every line is human-legible and editable.

SLURM-specific:

5. **Stable-vs-variable separation** (§ 6). Site-stable values live in
   config; per-job values are derived or come from CLI flags. Configure
   the site **once**.
6. **Submission is mandatory, not optional** (§ 7.8). On a real cluster
   the job MUST go through `sbatch`; the launcher is not meant to run on
   a login node.
7. **The env is the MPI + CUDA *userspace* boundary** (§ 7.5, § 7.9).
   The job carries its own MPI runtime and CUDA toolkit; the only host
   dependencies are the SLURM client and the GPU *driver*.
8. **Trust the live cluster over the docs.** ASU's own wiki is
   demonstrably stale (§ 7.0); verified-live values win, and we record
   the drift with dates.

---

## 7.0 Verified facts & doc-drift log (ASU Sol, 2026-06-26)

Recorded because ASU's published docs contradict the live system. When
they disagree, the **live system wins**; cite this log.

| Fact | Value (verified) | Source / how |
|---|---|---|
| **Partition rename** | `general` is now **private** (owned nodes); **`public`** is the public partition. ASU's SBATCH-examples page still shows `-p general` — **stale**. | Live `srun … -p general` error: *"the 'general' partition now only contains privately-owned nodes … public partition created … As of May 2026 Sol Maintenance."* |
| Default if unspecified | `htc` partition + `public` QoS | Sol Partitions & QoS page |
| `public` QoS max time | 7 days | same |
| `debug` QoS | 15 min, `htc` | same |
| **Public GPU partition** | **`public`** (CONFIRMED live) | `srun -N1 -c1 -t5:00 -p public -q public -G 1 nvidia-smi` allocated an A100-SXM4-80GB |
| Short-job hint | jobs ≤ 4 h: SLURM suggests **`-p htc`** | live `srun` notice |
| GPU nodes are **shared** by default | other users' processes co-reside; use `--exclusive` for a dedicated node | live `nvidia-smi` showed foreign `python`/`gmx_mpi` procs |
| **MIG enabled on some A100s** | some A100s sliced into MIG instances; `--gres=gpu:a100:N` should give FULL 80 GiB GPUs — **verify, don't get a 20 GiB slice** | live `nvidia-smi` MIG table |
| Default GPU memory | **24000 MB / GPU** if `--mem` unset | live `srun` notice |
| GPU hardware | see § 8 | Sol Hardware page |
| Env MPI | conda **OpenMPI 5.0.10** (`h67ed482_1`), internal PMIx, `MCA ras:slurm` + `MCA plm:slurm` | `ompi_info --all`; identical build hash in both siesta envs |
| Env CUDA | toolkit/runtime **13.3** (`nvcc V13.3.33`, `libcudart.so.13.3.29`, cuBLAS 13) | `nvcc --version`, `ldd` |
| **Sol GPU driver** | **595.71.05**, max **CUDA 13.2** | live `nvidia-smi` header |
| **CUDA verdict** | toolkit 13.3 vs driver-max 13.2 — **same major (13) ⇒ minor-version compat should cover the gap**; confirm with the § 11 benchmark. `cuda-compat` likely NOT needed (that's a cross-major / older-driver fix). | derived from the two rows above |
| From-source SIESTA-GPU linkage | conda `libmpi.so.40` + conda `libpmix.so.2` + conda `libscalapack.so` + from-source `libelpa_openmp.so.19` + conda CUDA 13 | `ldd $CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta` |

---

## 3. The two-layer model

```
<basename>.sbatch          <-- NEW: submission script (scheduler concern)
  #SBATCH ... (header)          - partition, qos, resources, GPU, mail, export
  bash <basename>.run.sh "$@"   - delegates to the launcher, unchanged
        |
        v
<basename>.run.sh          <-- EXISTING: launcher (script-execution.md)
  module load mamba/latest      - preamble (block 3)
  source activate <env>         - activation (block 4)
  --cold / run-index / etc.     - blocks 5-8
  mpirun -np $_mpi_np siesta    - engine launch (block 10-11; see § 7.5)
```

**Why a separate `.sbatch` (decision D2, resolved → separate)**: the
`.run.sh` stays scheduler-agnostic and interactively runnable; a header
tweak never forces a launcher regeneration; a different scheduler (PBS,
local) reuses the same `.run.sh` with a different thin wrapper.

---

## 4. The `scheduler` config block + site presets

A new top-level key, sibling to `script_generation`, read at generate
time only (same lifecycle as `script_generation`, `config.md` § 3).

### 4.1 Schema (draft)

```json
{
  "scheduler": {
    "kind": "slurm",
    "directives": {
      "partition": "public",
      "qos": "public",
      "mail_type": "ALL",
      "mail_user": "%u@asu.edu",
      "export": "NONE"
    },
    "gpu": { "partition": "public", "default_type": "a100", "exclusive": true },
    "defaults": { "time": "0-04:00:00", "cpus_per_task": 8, "mem": null }
  }
}
```

- `directives` — stable site header values. **`partition: "public"`**
  (NOT `general` — § 7.0).
- `gpu.partition` — **`public`** on Sol (CONFIRMED live, § 7.0); the
  public GPU nodes live in the same `public` partition.
- `gpu.exclusive` — request a whole node for GPU jobs (Sol GPU nodes
  are shared by default, § 7.0); avoids GPU contention for production
  runs. Off for the benchmark sweep (partial allocations schedule
  faster).
- `gpu.default_type` — `a100` (most plentiful + largest memory on Sol;
  decision D3).
- `defaults` — per-job values the user MAY override via CLI; `time`
  default `0-04:00:00` is safe under every public QoS ceiling.
- **No `mpi_launch` knob** — launch is always the env's own `mpirun`
  (§ 7.5, decision D1). Not configurable, by design.

### 4.2 Site preset: `asu-sol`

The "group of default scripts in the beginning stages" — dropped in
once, shipped in `docs/molbuilder.json.example` (and a future
`molbuilder config init --site asu-sol`):

```json
{
  "script_generation": {
    "preamble":   "module load mamba/latest",
    "activation": "source activate"
  },
  "scheduler": {
    "kind": "slurm",
    "directives": {
      "partition": "public", "qos": "public",
      "mail_type": "ALL", "mail_user": "%u@asu.edu", "export": "NONE"
    },
    "gpu": { "partition": "public", "default_type": "a100", "exclusive": true },
    "defaults": { "time": "0-04:00:00", "cpus_per_task": 8, "mem": null }
  }
}
```

One block makes every generated Sol job correct. `gpu.partition` is
`public` (confirmed live, § 7.0).

---

## 5. The generated `<basename>.sbatch` — block by block

```bash
#!/bin/bash
# === molbuilder sbatch header (scheduler: slurm; site: asu-sol) ===
#SBATCH -J <basename>                       # job name = project basename
#SBATCH -N 1                                # single node (v1)
#SBATCH -n <ntasks>                         # MPI ranks   (derived, § 6)
#SBATCH -c <cpus_per_task>                  # cores/rank  (config / derived)
#SBATCH -t <time>                           # walltime    (CLI / default)
#SBATCH -p <partition>                      # public  (or gpu.partition)
#SBATCH -q <qos>                            # public
#SBATCH --gres=gpu:<type>:<n>               # ONLY if the .fdf requests GPU
#SBATCH --mem=<mem>                         # emitted only if set
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=<mail_type>
#SBATCH --mail-user="<mail_user>"
#SBATCH --export=NONE

# SLURM lands us in SLURM_SUBMIT_DIR = the project dir.  No cd
# (config.md § 1).  --export=NONE means a clean env -> the launcher's
# `module load mamba/latest` is load-bearing (§ 7.1).
bash <basename>.run.sh "$@"
```

- Body is **one line**: delegate to `.run.sh`.
- `"$@"` forwards `--cold` / `--continue` so `sbatch <basename>.sbatch
  --cold` does a clean restart.
- `--gres` and `--mem` lines are **conditionally emitted** (omitted for
  CPU jobs / when `mem` is null).

---

## 6. Value-source matrix (the tweaking-minimization rule)

| Directive | Source (first wins) | Notes |
|---|---|---|
| `-J` job name | `.fdf`/`.py` basename | the project ID |
| `-p` / `-q` | `scheduler.directives` (or `gpu.partition` for GPU) | stable per site |
| `--mail-*`, `--export` | `scheduler.directives` | stable |
| `-o` / `-e` | fixed `slurm.%j.{out,err}` | ASU convention |
| `-n` ntasks | CLI `--np` → `.fdf`-aware mpi_np selector → `scheduler.defaults` | **GPU jobs: 1 rank per GPU** (§ 8) |
| `-c` cpus/rank | CLI `--omp`/`--cpus` → derived `node_cores / ntasks` → `defaults.cpus_per_task` | feeds OMP for ELPA-OpenMP |
| `-t` time | CLI `--time` → `defaults.time` | only the user knows runtime |
| `--gres` | `.fdf` GPU request (`_fdf_requests_gpu`) + CLI `--gres <type>:<n>` → `gpu.default_type` | auto-on when chemistry wants GPU |
| `--mem` | CLI `--mem` → `defaults.mem` (null = scheduler default) | opt-in |
| `-N` nodes | fixed `1` (v1) | multi-node deferred |

**The user's real per-job surface**: *time*, and *GPU type/count when
relevant*. Everything else is site-config (once) or derived.

---

## 7. ASU Sol compatibility — the constrained-environment walkthrough

Each point asks: does this step actually work given Sol's remote access
+ cleaned environment + the partition migration?

### 7.1 Fresh login: mamba not on PATH + `--export=NONE`
The `.sbatch` sets `--export=NONE` (clean env); the *launcher* runs
`module load mamba/latest` then `source activate`. Bootstrap happens
once, in the right place. **Compatible** — matches ASU's own example.

### 7.2 Env already prepared
We assume molbuilder + the four backend envs are installed
(`README_install.md`). Scripts only *activate*, never *create*.
**Compatible.**

### 7.3 CPU allocation
`-N 1 -n <ntasks> -c <cpus>`. The launcher reads `SLURM_NTASKS` at
runtime, so header `-n` and `mpirun -np` agree by construction.
**Compatible — no double-source drift.**

### 7.4 GPU allocation + the partition migration
GPU requested via `--gres=gpu:a100:N` (or `-G N`) on **`-p public`**
(CONFIRMED live — § 7.0; post-May-2026 the stale `-p general` is now
private and rejected). Two live caveats to handle:
- **Shared nodes**: Sol GPU nodes carry other users' jobs by default.
  Production runs add `--exclusive` (preset `gpu.exclusive: true`);
  the benchmark sweep stays non-exclusive for faster scheduling.
- **MIG**: some A100s are sliced into MIG instances. `--gres=gpu:a100:N`
  should return FULL 80 GiB GPUs, but the benchmark's correctness gate
  (§ 11.1) must record `gpu_device` / GPU memory and flag if a 20 GiB
  MIG slice came back instead of a full A100.

**Compatible** — partition pinned to `public`.

### 7.5 MPI launch — our `mpirun`, never `srun --mpi=pmix` (decision D1)
**This is an ABI-correctness requirement, not a preference.** The
from-source SIESTA-GPU is dynamically linked against the **conda
OpenMPI 5.0.10** (`libmpi.so.40`) and that env's **internal PMIx**
(`libpmix.so.2`) — verified by `ldd`. Therefore:

- **Launch with the env's own `mpirun`.** It is SLURM-aware
  (`MCA ras:slurm` reads the allocation; `MCA plm:slurm` would place
  multi-node daemons via `srun`), so inside an sbatch allocation it
  picks up the nodes/cores automatically — using SLURM as a *resource
  manager*, with the MPI/PMIx wire-up staying entirely inside the
  matched conda stack.
- **Do NOT use `srun --mpi=pmix`.** That would hand our OpenMPI-5.0.10
  ranks to **Sol's slurmd PMIx server** (a different, independently
  built PMIx paired with ASU's `openmpi/4.1.5` module) — a cross-stack
  PMIx handshake whose version compatibility is fragile and would break
  on ASU upgrades. ASU recommends `srun --mpi=pmix` *for users who
  `module load openmpi/4.1.5`*; we deliberately don't (the four-env
  isolation), so that recommendation does not apply to us.
- **Never `srun mpirun …`** (double-launch).

For single-node (v1) this is maximally robust: `mpirun` forks ranks
inside the SLURM-allocated cpuset; `plm:slurm` isn't even exercised.
**Compatible — and `srun --mpi=pmix` is the *incorrect* path for this
build.**

### 7.6 Working directory / outputs
`SLURM_SUBMIT_DIR` = the dir you `sbatch` from = the project dir. The
launcher never `cd`s (`config.md` § 1), so outputs land beside inputs.
Convention: `cd <projdir>; sbatch <basename>.sbatch`. **Compatible.**

### 7.7 Walltime / QoS ceilings
`-q public` (≤ 7 days). The `0-04:00:00` default is safe under every
public ceiling. An over-long `-t` is rejected at submit — loud and
immediate. **Compatible.**

### 7.8 Submission is MANDATORY (login nodes cannot run the job)
On Sol you cannot `bash run.sh` interactively for a real job:
- **Login nodes have no GPUs** and reap long processes.
- An interactive `salloc`/`srun --pty` ties up the terminal and dies on
  disconnect.

The **only correct path** is `sbatch <basename>.sbatch` from the login
node: SLURM queues it, runs it on a compute node, and you can log out.
Output goes to **files** (`slurm.%j.out` + the launcher's `.out`
redirect + the runwrap log). Monitor with `squeue -u $USER`, post-mortem
with `seff <jobid>`, cancel with `scancel <jobid>`, watch with
`tail -f <basename>-run0.out`. **This is why the `.sbatch` layer is
required, not convenience** — without it the user is pushed toward the
two forbidden paths.

### 7.9 CUDA driver floor — the runtime-vs-driver split
We bundle the CUDA **userspace** (runtime/libraries/ELPA-CUDA kernels);
we **cannot** bundle the CUDA **driver** (`libcuda.so` + the kernel
module) — it is the host's, installed per GPU node by ASU. A CUDA
**13.3** runtime requires a **driver new enough to support CUDA 13.x**
(~580+ datacenter branch). Driver compat is one-directional: a driver
supports its CUDA version **and all older**, never newer.

- **The only host dependency for GPU** is the driver, and the only
  question is `Sol_driver_max_CUDA ≥ 13.3`.
- **Probe (the gate before any GPU job)**: on a Sol GPU node,
  `nvidia-smi` → top-right **"CUDA Version"** = the driver's max CUDA.
- **LIVE RESULT (2026-06-26)**: Sol driver **595.71.05**, max **CUDA
  13.2**. Our toolkit is **13.3** — the **same major (13)**, one minor
  ahead. CUDA **minor-version compatibility** allows a newer-minor
  runtime on an older-minor driver of the same major (as long as no
  13.3-only API is used — unlikely for SIESTA/ELPA's stable API
  surface). So the expectation is **runs as-is**; the § 11 benchmark
  confirms.
- **Fallback if it ever fails** (older cluster, or a 13.3-only symbol):
  Sol's GPUs are datacenter A100s, so NVIDIA's **`cuda-compat`**
  forward-compat package applies — a forward-compat `libcuda.so`
  prepended on `LD_LIBRARY_PATH`. Carry it *in the env* (self-contained)
  rather than asking admins to upgrade. **Likely NOT needed here**
  (that's a cross-major / much-older-driver fix; 13.3-on-13.2 is a
  0.1 minor gap).
- **Empirical proof**: the benchmark/smoke job (§ 11) either runs with
  `siesta_diag.elpa_gpu == True` or fails at the first GPU call.

**Verdict for Sol: expected compatible (minor-version compat covers
13.3-on-13.2); confirm via § 11.** See decision D7 (CUDA build-target).

### 7.10 Remote / offline generation
Generation needs no cluster reachability — artifacts are plain text,
generated on a laptop or login node, then `sbatch`'d. **Compatible.**

---

## 8. Sol GPU hardware reference (for resource sizing)

| Node | GPUs/node | GPU memory | CPU cores | RAM | Default mem |
|---|---|---|---|---|---|
| **A100** | 4× A100 | **80 GiB** ea | 48 (2× EPYC 7413) | 512 GiB | 24 GiB/GPU or 2 GiB/core |
| A30 | 3× A30 | 24 GiB ea | 48 | 512 GiB | — |
| MIG | 16× A100 slices | 10 / 20 GiB | 48 | 512 GiB | — |
| MI200 (AMD) | 2× MI200 | — | 24 (EPYC 9254) | 77 GiB | — |

Request syntax: `-G N` (any type) or `--gres=gpu:a100:N` (specific);
memory `--mem=120G`, whole node `--exclusive --mem=0`.

**Recommended SIESTA-GPU mapping (the efficient shape)**: ELPA-CUDA is a
*distributed* eigensolver — one MPI rank drives one GPU. So **1 rank ↔ 1
GPU**; cores split evenly (`-c = node_cores / ngpus`), `OMP_NUM_THREADS`
per rank for the ELPA-OpenMP CPU work. Whole-A100-node example (a slow
single-GPU relax → 4× A100):

```bash
#SBATCH -N 1 -n 4 -c 12 --gres=gpu:a100:4 --mem=0 --exclusive
#SBATCH -t 1-00:00:00 -p public -q public
```

Prefer **A100 over A30** (80 vs 24 GiB GPU memory; 4 vs 3 per node).
Notes from the live probe (§ 7.0): GPU nodes are **shared** by default
— `--exclusive` for a clean production run; **MIG** is enabled on some
A100s, so confirm the benchmark gets full 80 GiB GPUs (not 20 GiB
slices); for the short benchmark itself, **`-p htc`** (≤ 4 h) may
schedule faster than `public`.
**Validate scaling before the long run** (§ 11) — SIESTA-GPU scaling is
sublinear and system-size-dependent; over-requesting GPUs both wastes
allocation and lengthens the queue.

---

## 9. Open decisions

| # | Decision | Resolution / leaning |
|---|---|---|
| **D1** | MPI launcher | **RESOLVED → env's own `mpirun`** (ABI; `srun --mpi=pmix` rejected — § 7.5) |
| **D2** | Header location | **RESOLVED → separate `<basename>.sbatch`** (§ 3) |
| **D3** | Default GPU type | **RESOLVED → `a100`** (most plentiful + 80 GiB) |
| **D4** | `--export=NONE` | **always** under `asu-sol`; configurable via `directives.export` |
| **D5** | Multi-node MPI | **single-node v1**; multi-node (needs `plm:slurm` validation) deferred |
| **D6** | CLI surface | flags on `molbuilder run` (`--time`/`--gres`/`--mem`); `run` emits both `.run.sh` + `.sbatch` when `scheduler` is set |
| **D7** | **CUDA build-target** | **open, but lower urgency** — Sol's driver is **CUDA 13.2** (live), so our **13.3** build is expected to run via minor-version compat (§ 7.9). Still a tracked roadmap decision: bleeding-edge 13.x maximizes the chance of hitting an older-driver cluster *elsewhere*; a conservative 12.x build "runs almost anywhere." Decide deliberately at build time, not by default. |
| **D8** | **Public GPU partition name** | **RESOLVED → `public`** (confirmed live: `srun -p public -G 1` allocated an A100, § 7.0) |
| **D9** | **GPU node exclusivity** | **production: `--exclusive`** (shared nodes, § 7.0); **benchmark: non-exclusive** (faster scheduling). Preset `gpu.exclusive`. |

---

## 10. Refuse-to-emit, extended

- SLURM site (`scheduler.kind == "slurm"`) but `directives.partition` or
  `qos` missing → refuse the `.sbatch`, name the key, point here.
- A GPU job requested but `gpu.partition` null AND the fallback
  partition has no GPUs → refuse with a "set scheduler.gpu.partition
  (see § 7.0)" message rather than emit a header that won't allocate.
- `scheduler` absent → emit only `.run.sh` (today's behavior) + a
  one-line hint that no submission script was generated. Not an error —
  local/laptop users don't need a `.sbatch`.

---

## 11. Benchmark / validation mode

The first thing to submit on Sol — it proves the whole stack and sizes
resources in one short batch job, so we never guess Sol's driver or
scaling. **Built on the existing `molbuilder bench siesta-gpu`** (sweeps
`np/omp/BlockSize`, reports per-iter wall time) plus a correctness gate
and a resource sweep.

### 11.1 Correctness gate ("is the GPU actually doing the work?")
After a short capped run, assert from the parser's `runtime_info`
(captured by the §-5-of-#5 runtime probes):
- `siesta_diag.elpa_gpu == True` and `siesta_diag.gpu_device` populated
  → ELPA-CUDA engaged (catches the silent CPU-fallback **and** the
  § 7.9 driver problem),
- `siesta_build.parallelisations` includes `MPI` → ranks launched,
- final energies finite / SCF converging → numerically sane.

`elpa_gpu == False` ⇒ **fail loudly** — otherwise the job runs
slow-on-CPU while looking successful.

### 11.2 Resource sweep (find the right CPU/GPU/mem)
Emit a **short, capped** job (small `MaxSCFIterations` or a few relax
steps) as an **sbatch array** over a small grid: `{1, 2, 4}× A100`, OMP
threads, BlockSize. Each array task is one `.sbatch`, run through the
batch system. Matches ASU's own "test a small job first" advice.

### 11.3 Report + recommend
Fuse `seff <jobid>` (GPU/CPU/mem utilization) with the parser's
per-SCF-step wall time → table + recommended production config (the knee
of the scaling curve). Turns "Sol allows 4 GPUs" into "use N because
that's where *your* system stops scaling."

### 11.4 This is the smoke test
One short benchmark job simultaneously validates (a) the MPI-SLURM
launch (§ 7.5), (b) the CUDA driver compatibility (§ 7.9), and (c)
resource sizing — before committing the long production run.

---

## 12. Anti-patterns we refuse

- **No runtime scheduler probing** in the `.sbatch` (`sinfo`/`scontrol`
  at job start). All values baked.
- **No duplicated env activation** — the launcher owns it.
- **No `srun --mpi=pmix` / no `module load openmpi`** — would break the
  conda-MPI ABI (§ 7.5).
- **No partition-selection heuristics.** One configured default; user
  overrides explicitly.
- **No hardcoding partition/GPU names from ASU's docs** — they are
  stale (§ 7.0); pin from the live system.
- **No silent resource defaults that exceed a QoS ceiling.** Over-asks
  fail loudly at submit.
- **No `cd` in the `.sbatch`.** `SLURM_SUBMIT_DIR` is the contract.

---

## 13. Testing strategy

| Layer | Test |
|---|---|
| L1 | `scheduler` parse + merge + refuse-to-emit on missing partition/qos. |
| L2 | `.sbatch` golden: `asu-sol` preset emits `-p public` (NOT `general`), `-q public`, mail/export, `slurm.%j.out`. |
| L2 | GPU gating: GPU `.fdf` emits `--gres=gpu:a100:N` + 1-rank-per-GPU `-n`; CPU `.fdf` omits `--gres`. |
| L2 | Value sourcing: CLI `--time`/`--np`/`--gres` override config; `.fdf` mpi_np feeds `-n`. |
| L2 | Delegation: body is `bash <basename>.run.sh "$@"`; no re-implemented activation. |
| L3 | Benchmark mode: `elpa_gpu == False` fixture ⇒ correctness gate fails. |

---

## 14. References

- [`config.md`](../config.md), [`script-execution.md`](script-execution.md),
  [`script-contract.md`](script-contract.md), [`job-layout.md`](job-layout.md).
- `molbuilder bench siesta-gpu` (README § Performance benchmarking) —
  the benchmark foundation (§ 11).
- ASU RC Confluence (treat as **possibly stale** — § 7.0): [Using Slurm
  to Submit Jobs](https://asurc.atlassian.net/wiki/spaces/RC/pages/1905360902/),
  [SBATCH Job Scripts](https://asurc.atlassian.net/wiki/spaces/RC/pages/1905131604/),
  [Sol Partitions and QoS](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908867081/),
  [Sol Hardware - How to Request](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908998178/),
  [Managing Python Modules through mamba](https://asurc.atlassian.net/wiki/spaces/RC/pages/1905328428/).
