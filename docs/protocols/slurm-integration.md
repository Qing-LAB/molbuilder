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

#### 7.5.1 Multi-GPU binding — RUNTIME per-rank, never a hardcoded socket
**Mapping: 1 MPI rank ↔ 1 GPU** (ELPA-CUDA is a distributed eigensolver;
one rank drives each GPU). The non-obvious correctness point:

> **GPU→socket placement is NOT guaranteed.** Requesting `--gres=gpu:a100:N`
> on a SHARED node gives you N of the 4 GPUs *by availability* — they may
> be same-socket, split, or any combination. So a static `--map-by
> ppr:2:socket` binding is FRAGILE: if SLURM hands you socket-1 GPUs but
> the binding pins ranks to socket 0, every OMP thread is the far
> (distance-32) side of the NUMA divide.

**The binding must therefore be decided at RUN time, per rank**, from the
*actual* allocation. Each rank: (1) takes its GPU via
`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`, (2) reads that GPU's
NUMA node from `/sys/bus/pci/devices/<busid>/numa_node`, (3) binds itself
to that node with `numactl --cpunodebind=$numa --membind=$numa`, then
execs SIESTA. Robust to same-socket OR split allocations.

```bash
# per-rank launcher (emitted by the wrapper for a GPU job)
_lr=$OMPI_COMM_WORLD_LOCAL_RANK
export CUDA_VISIBLE_DEVICES=$_lr
_bus=$(nvidia-smi --id=$_lr --query-gpu=pci.bus_id --format=csv,noheader)
_numa=$(cat /sys/bus/pci/devices/${_bus,,}/numa_node 2>/dev/null)
[ "${_numa:--1}" -ge 0 ] && _bind="numactl --cpunodebind=$_numa --membind=$_numa" || _bind=""
exec $_bind siesta < in.fdf
```
Outer launch: `OMP_NUM_THREADS=<c> mpirun -np N ... <per-rank launcher>`.

#### 7.5.1.a One generalized model — `K` ranks per GPU (D10)
The "1 rank ↔ 1 GPU" framing was too narrow. The A100 is **heavily
under-utilized at 1 rank/GPU** (the 444-atom job uses <6 GiB of 80 GiB
and a fraction of the 108 SMs), and the user's *local* box already found
**4 ranks/GPU + MPS** to be the ELPA no-NCCL throughput optimum on a far
weaker card. So the real model is **`K` ranks per GPU**, with `K` an
empirical knob the benchmark sizes (§ 11.2) — exactly as `K=4` was found
locally. The old "shared-GPU/MPS" path and the "1-rank-per-GPU" path are
just `K>1, n_gpus=1` and `K=1, n_gpus≥2` special cases of it.

**Rank→GPU mapping (block-distributed, handles every case):**
```bash
_gpu=$(( OMPI_COMM_WORLD_LOCAL_RANK * _ngpu / OMPI_COMM_WORLD_LOCAL_SIZE ))
export CUDA_VISIBLE_DEVICES=$_gpu        # ranks 0..K-1 -> GPU 0, K..2K-1 -> GPU 1, ...
```
`n_gpus=1` ⇒ every rank → GPU 0 (the existing local behavior falls out).
`n_gpus=2, K=4` ⇒ 8 ranks, ranks 0-3 → GPU 0, 4-7 → GPU 1.

**MPS** is enabled per node whenever `ranks_per_gpu = mpi_np / n_gpus ≥ 2`
(one control daemon serves all GPUs on the node; each rank's
`CUDA_VISIBLE_DEVICES` selects its target). `K=1` ⇒ MPS suppressed (no
Hyper-Q sharing needed).

**Runtime fork** (topology, not SLURM presence): the generalized
per-rank path engages only for **genuine multi-GPU** (`n_gpus ≥ 2`); a
**single GPU always keeps the existing, battle-tested launch** (local OR
Sol — a 1-GPU job has no multi-GPU placement problem):
```bash
_ngpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
if [ "${_ngpu:-0}" -ge 2 ] && [ "$_mpi_np" -ge "$_ngpu" ]; then
    _multi_gpu=1   # generalized K-ranks/GPU per-rank launch (Sol)
else
    _multi_gpu=0   # single-GPU OR shared-1-GPU/MPS: existing path, unchanged
fi
```

**Mechanism**: the per-rank logic can't be a word-split `_launch_cmd`
string (the `bash -c` quoting won't survive). The wrapper writes a tiny
per-rank helper (`.mb-rank-launch-$$.sh`) that maps rank→GPU, exports
`CUDA_VISIBLE_DEVICES`, logs placement, then `exec siesta "$@"`; launch
is `mpirun -np N bash <helper> <fdf>`. Implementation specifics that the
fresh-eyes review pinned (2026-06-26):

- **Allocated-GPU set from `CUDA_VISIBLE_DEVICES`, not `nvidia-smi -L`.**
  SLURM sets `CUDA_VISIBLE_DEVICES` to the *allocated* GPUs; `nvidia-smi
  -L` over-counts on a shared node without device-cgroup isolation (it
  would list GPUs we don't own). The helper splits CVD into a list and
  indexes `list[ local_rank*ngpu/localsize ]`; it falls back to
  `nvidia-smi -L` only when CVD is unset (local / no scheduler). The
  parent's `_ranks_per_gpu` count uses the same CVD-first rule so the
  MPS gate agrees.
- **One unified `EXIT` trap.** A second `trap … EXIT` *replaces* the
  first in bash, so all teardown (per-rank helper file **and** the MPS
  daemon + its `/tmp` dirs) routes through a single `_mb_cleanup`
  function set once near the top; the MPS block only sets a
  `_mps_started=1` flag. (The earlier two-trap shape silently leaked the
  MPS daemon.)
- **`--dry-run`.** Every wrapper accepts `--dry-run`: it resolves +
  **logs** the launch command and (GPU mode) prints the per-rank
  rank→GPU/NUMA mapping for the current allocation, then `exit 0`
  WITHOUT launching SIESTA (and without starting MPS). This is the cheap
  pre-flight: `sbatch job.sbatch --dry-run` → read the log → confirm the
  command matches the SLURM allocation before spending a real run.

#### 7.5.1.b CPU-bind policy — P1 (trust SLURM) for v1, bench-gated (D11)
CPU/memory placement is left to **SLURM's cgroup cpuset**
(`--gres-flags=enforce-binding` + `-c`), NOT manual `numactl`. Reasons:
(1) on Sol's **NPS4** layout a GPU's sysfs `numa_node` is only **6
cores**, so a naive per-rank `numactl --cpunodebind=<that node>` would
crush a `-c 12` rank onto 6 cores; (2) at `K=4` the allocation already
lands one rank per 6-core NPS4 node anyway; (3) SLURM under
enforce-binding already cpusets each task to GPU-proximate cores +
local memory. So the helper does **not** bind CPU — it only sets
`CUDA_VISIBLE_DEVICES` and **logs each rank's GPU + NUMA node + actual
`Cpus_allowed_list`** so the benchmark REVEALS SLURM's placement. If the
bench shows SLURM is *not* binding near the GPU, the fallback is P2
(bind to the GPU's whole socket); we do not commit kernel-level
`numactl` speculatively. **OMP width per rank** honors
`SLURM_CPUS_PER_TASK` (the `-c` value) ahead of the local-probe default,
so the Sol allocation drives `OMP_NUM_THREADS` automatically.

**Cores/rank**: set by the chosen `K` (§ 11.2). On the 48-core / 2×A100
node
(2 sockets × 24 cores, **8 NUMA nodes** of 6 cores; within-socket NUMA
distance 12, cross-socket 32): **`-c 12`/rank**, 12 OMP threads. NEVER
exceed the GPU's socket — spilling OMP threads to the far socket
(distance 32) is slower than not having them. So for 2 GPUs → `-c 12`
(24 cores = one socket) is both the recommendation **and the ceiling**.
Add **`--gres-flags=enforce-binding`** to nudge SLURM toward co-locating
CPU+GPU. **The wrapper MUST log each rank's GPU + NUMA node** into the
timing header so a slow run is diagnosable (bad binding vs. bad config).
The existing `_gpu_runtime_defaults_block` is single-GPU/generate-time;
this per-rank/runtime probe is NEW logic.

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

**A100 node topology (live-probed `sg0NN`, 2026-06-26)**: 2× AMD EPYC
7413 = **48 cores, NO hyperthreading**; **8 NUMA nodes** (NPS4: 4/socket,
6 cores + ~64 GiB each); socket 0 = NUMA 0–3 / cores 0–23, socket 1 =
NUMA 4–7 / cores 24–47; within-socket NUMA distance **12**, cross-socket
**32**; GPUs NVLink-connected; **`TMPDIR=/tmp` (node-local → no OpenMPI
NFS-shmem fix needed)**.

**Mapping**: 1 rank ↔ 1 GPU; `-c 12`/rank (12 OMP threads for ELPA's
OpenMP host work); **binding decided per-rank at RUNTIME** (§ 7.5.1) —
NOT a static socket map, because GPU placement isn't guaranteed. 2-GPU
example (24 cores = one socket's worth):

```bash
#SBATCH -N 1 -n 2 -c 12 --gres=gpu:a100:2 --gres-flags=enforce-binding
#SBATCH -t 0-04:00:00 -p htc -q public
# launch: OMP_NUM_THREADS=12 mpirun -np 2 <per-rank NUMA launcher, § 7.5.1>
```

Prefer **A100 over A30** (80 vs 24 GiB), and **a100 over a100.40gb /
a100.20gb** (full 80 GiB vs MIG slice). **Never `l40`** (Ada, crippled
FP64 ~1.4 TFLOPS — same trap as a consumer card; § 11). Live-probe notes
(§ 7.0): nodes are **shared** (use `--exclusive` for a clean production
run); **MIG** is on some A100s (confirm full 80 GiB, not a 20 GiB slice);
**htc has the GPU fleet** (~47 `sg0NN` 4×A100 nodes) — short benchmark
jobs schedule there fastest. **Validate scaling before the long run**
(§ 11): SIESTA-GPU scaling is sublinear + system-size-dependent.

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
resources, so we never guess the driver or scaling. **Built on the
existing `molbuilder bench siesta-gpu`** plus a correctness gate and a
**CPU-vs-GPU head-to-head**.

**CPU vs GPU is an OPEN question — do NOT assume GPU wins.** Empirically
the user saw **CPU (np=20) beat single-GPU** on their workstation — but
that was an **RTX 3060 Ti**, a consumer card with **crippled FP64
(~0.25 TFLOPS)**; SIESTA/ELPA diagonalization is FP64-heavy and Sol's
**A100 does ~9.7 TFLOPS FP64 (40–80× more)**. So the local result does
NOT predict Sol — the benchmark must measure both, head-to-head, on Sol
hardware. (Also: **avoid `l40`** GPUs — Ada, crippled FP64 like a
consumer card.)

**Jobs are independent + parallel, NOT sequential.** A SLURM job has one
allocation, so CPU-only and GPU configs are *separate* `.sbatch` files
(they also differ in the `.fdf`'s `Diag.ELPA.GPU` flag → different env:
`molbuilder-siesta` vs `molbuilder-siesta-gpu`). Submit them all; SLURM
runs the CPU job on a CPU node and the GPU job on a GPU node concurrently.
Collect each job's `.scf-timing.log` and compare.

### 11.0 Iteration-count methodology (why a few, but not 1–2)
The ranking metric is **steady-state wall-time per SCF iteration**
(dominated by the fixed-size O(N³) eigensolve — identical every
iteration, convergence-independent). But **iteration 1–2 is one-time
warmup** (pseudo/basis/mesh setup; on GPU: CUDA context + cuBLAS/ELPA
handle + memory-pool alloc + first-kernel JIT — bigger for GPU, would
*penalize* it unfairly). So cap at **`MaxSCFIterations 5`**, single-point
(no relax), and **report the mean of iters 3–5** (the parser's
per-iteration delta cancels the one-time setup). 1–2 iters is wrong; ~5
is enough and bounds wall-time.

### 11.0b Per-iteration timing instrument — `.scf-timing.log`
SIESTA emits **no usable per-iteration wall time** (the `scf:` lines
carry energies + dDmax but no time; `timer: IterSCF` is cumulative). So
the wrapper MUST measure it: pipe SIESTA stdout through a filter that
**timestamps each `scf:` line** into a persistent
**`<basename>.scf-timing.log`** (`<epoch> <iter#> <dDmax>`); per-iter
time = consecutive-stamp delta. This is the benchmark's measurement
instrument and is **non-optional**. Forward-looking: the same filter
maintains a `<basename>.status.json` (current iter / per-iter time / ETA
/ config) — the machine-readable status a future **notifier** pushes
(webhook/email) so the user needn't log in to grep; this is the
front-end of the job-monitor/watcher surface.

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

### 11.1b Live local-GPU utilization probe (2026-06-26) — the extrapolation basis
Before sizing K we measured the LIVE phase-4 job (RTX 3060 Ti, np=4 +
MPS) for 5 min (observation only; the run dir is read-only):

| Metric | Value | Reading |
|---|---|---|
| GPU util | **mean 55 %, bursty 0↔100** | **compute-bound DURING the eigensolve** (hard 100 % spikes), idle between (CPU-side SCF phases) — NOT steady |
| VRAM | mean 2.8 GB, **peak 6.7 / 8 GB** | comfortable; memory is **not** the limit |
| mem-bw util | **~7 %** | nowhere near memory-bandwidth-bound |
| CPU | 4 ranks ≈ 4.4 cores | — |

**Extrapolation:** the bottleneck during the eigensolve is FP64 compute
(the weak card pegs at 100 %), but only ~55 % of the time. The A100's
~40× FP64 collapses each eigensolve burst to a blip and its 80 GB dwarfs
the 6.7 GB footprint, so a **single A100 is wildly under-fed at K=4** —
the ceiling shifts to **GPU compute throughput + CPU cores**, not VRAM.
This is why we push K **up** and **never test K=1**.

### 11.2 The configs to compare (single A100, swept high→low; K=1 excluded)
**Sequenced** (user 2026-06-26): run CPU baseline first, then the GPU
max-K, then one step down, and judge production sizing from the relation
between the three. All `MaxSCFIterations 5` single-point on the test
system (§ 16), all `-p htc -q public` (GPU fleet + fast scheduling).
**One GPU** (not two) for fast, predictable scheduling; ranks share it
via MPS (per-rank CVD map + `_ranks_per_gpu`-gated MPS, § 7.5.1); cores
kept on the GPU's 24-core socket.

| # | Config | `.fdf` flag | env | resources |
|---|---|---|---|---|
| 1 | **CPU np=64** | `Diag.ELPA.GPU .false.` | `molbuilder-siesta` | `-n 64` (no gres) — schedules immediately; the CPU reference |
| 2 | **GPU 1×A100, K=8 (max)** | `.true.` | `molbuilder-siesta-gpu` | `-n 8 -c 3 --gres=gpu:a100:1 --gres-flags=enforce-binding`, MPS |
| 3 | **GPU 1×A100, K=4 (down one)** | `.true.` | `molbuilder-siesta-gpu` | `-n 4 -c 6 --gres=gpu:a100:1 --gres-flags=enforce-binding`, MPS |

K=8 (c=3) and K=4 (c=6) both fit one 24-core socket; 8 MPS clients is
well within the A100's Hyper-Q limit. **K=1 is deliberately excluded**
(under-feeds the A100; § 11.1b). If K=8 ≫ K=4 the A100 wants even more
ranks (extrapolate to 2+ GPUs for production); if they tie, K=4 suffices.
The framework's general load-balance model (§ 7.5.1) already handles 2+
GPUs, so scaling out is a config change, not new code.

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

---

## 15. Implementation plan (the build — pin-pointed)

Build order; each is a scoped commit with host-env tests
(`conda run -n molbuilder python -m pytest`). Generation stays offline
(no cluster needed); the `.run.sh` launcher internals are unchanged
except the GPU/timing blocks.

**Status (2026-06-26): A, B, C DONE; D, E pending.**

- **A. `scheduler` config reader** — ✅ DONE (`get_scheduler` +
  `_validate_scheduler` + `_normalise` passthrough; 15 tests). Reads the
  `scheduler` block (server + project deep-merge), validates,
  **refuse-to-emit** if `kind=="slurm"` and `directives.partition`/`qos`
  missing (§ 10). Returns `None` when no block (emit only `.run.sh`).
- **B. `.sbatch` emitter** — ✅ DONE (`render_sbatch` + `write_sbatch` +
  `_parse_gres` + `_maybe_write_sbatch` wired into `write_run_wrapper`;
  23 tests). Produces the § 5 header (conditional
  `--gres`/`--gres-flags`/`--exclusive`/`--mem`) + body
  `bash <basename>.run.sh "$@"`, chmod 644, `bash -n` validated, emitted
  only when `scheduler` is configured (§ 10).
- **C. GPU load-balance + per-rank launcher + `--dry-run`** — ✅ DONE
  (the model GENERALIZED from "multi-GPU NUMA" to the load-balance model
  of § 7.5.1, per the 2026-06-26 design thread). Extracted, documented
  block-emitters: `_gpu_loadbalance_block` (ranks-per-GPU),
  `_gpu_per_rank_launcher_block` (CVD-derived rank→GPU map + SLURM-trust
  P1 binding), `_siesta_resolved_log_block` (always-on launch audit),
  `_siesta_dry_run_block`, `_bash_numa_from_gpu` (shared sysfs lookup).
  OMP honors `SLURM_CPUS_PER_TASK`; MPS gates on `_ranks_per_gpu` + not
  dry-run; single unified `_mb_cleanup` EXIT trap. 14 tests. **Note**:
  the original "`numactl --cpunodebind`/`-c 12`/socket-map" sketch was
  REPLACED by P1 (trust SLURM cpuset) after the NPS4 review (§ 7.5.1.b).
- **D. `.scf-timing.log`** — ⏳ PENDING. In the launch block, tee SIESTA
  stdout through a timestamping filter that appends `<epoch> <iter#>
  <dDmax>` for each `scf:` line to `<basename>.scf-timing.log` (§ 11.0b).
  Portable bash (`date +%s.%N` + awk). (`status.json`/notifier is a
  follow-up, not v1.)
- **E. Test-bundle generation** — ⏳ PENDING. Generate § 16 through the
  framework (A–D) into a temp project: one CPU `.fdf`
  (`Diag.ELPA.GPU .false.`) + one GPU `.fdf` (`.true.`), each with its
  `.run.sh`, and the three `.sbatch` of § 11.2 (CPU np=64, GPU K=8,
  GPU K=4). Gated on the conda envs existing on Sol.

**Tests** (host env): ✅ scheduler parse/merge + refuse-to-emit; ✅
`.sbatch` golden (asu-sol → `-p public`/`-q public`, conditional `--gres`,
`bash -n` clean); ✅ GPU `.fdf` → per-rank launcher + load-balance +
dry-run present, CPU `.fdf` → none; ✅ single-EXIT-trap; ⏳
`.scf-timing.log` emitter shape (with D).

---

## 16. Test-bundle spec (first Sol validation, do NOT touch the live run)

System: **444-atom BDT–Au(111) thiol junction** = `stage4` ("phase 4") of
`projects/BDT/optimization/TJ-BDT-Au111/` — that dir is the LIVE run,
**strictly read-only**. Build the test in a TEMP project
(`projects/test_runs/sol-bench-BDT-Au111/`).

Inputs (copy/reuse, framework-generated `.fdf`/`.run.sh`/`.sbatch`):
- Geometry: latest 444-atom coords (the dir's `.xyz` / `.XV`, updated last).
- Labels: `projects/BDT/structure/BDT_Au111_6_6_6_unoptimized.molstruct.json`
  — regions `L-electrode`/`bridge`/`R-electrode`/`BDT` + **216 frozen
  atoms** (atom indices stable → apply to current geometry). The SIESTA
  optimization consumes the frozen atoms (constraints) and warns on the
  region labels (pattern B, sidecar-contract).
- Pseudos: reuse the dir's `Au/C/H/S.psml`.
- Engine config: `SolutionMethod diagon`, `ELPA-1STAGE`, `MeshCutoff 400 Ry`,
  **`MaxSCFIterations 5`**, single-point (no relax / `MD.NumCGsteps 0`),
  `DM.UseSaveDM .false.`. GPU variant adds `Diag.ELPA.GPU .true.`.

Prerequisite gate (run-time, on Sol): the conda envs
(`molbuilder-siesta-gpu`, `molbuilder-siesta`) must be installed on Sol —
verify `module load mamba/latest && mamba env list | grep molbuilder`
before submitting. If absent, install them first (step zero). Then:
`cd <tempproj>; sbatch cpu-np20.sbatch; sbatch cpu-np64.sbatch;
sbatch gpu-2a100.sbatch` → compare each `.scf-timing.log` (mean iters 3–5).
