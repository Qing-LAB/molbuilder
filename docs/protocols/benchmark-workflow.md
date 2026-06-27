# Benchmark workflow — target-isolated, self-configuring

> **Design doc.** Authoritative for the end-to-end "benchmark, then run"
> flow that lets a user drop a portable bundle onto *any* compute machine
> — a single workstation or a shared supercomputer — and have it detect
> and configure itself, with **no hand-tuning and no hand-edited
> scripts.** Status of each piece is in § 0; vocabulary is in § 1 (read it
> first — the rest of the doc avoids unexplained jargon by relying on it).
> Cross-references: `slurm-integration.md` (the job-submission layer),
> `bench-generate-spec` (the bundle generator), `config.md` (script
> generation).

---

## § 0 Status

| Piece | State |
|---|---|
| `molbuilder bench generate` — portable bundle | **built** (`bench/generate.py`) |
| Monitor: progress + utilization sampling | **built** (`monitor.py`; § 9) |
| Memory estimator + runtime check | **built** (`siesta/memory.py`; slurm-integration.md § 11.0d) |
| GPU-to-CPU placement (same socket) | **built** (slurm-integration.md § 7.5.2) |
| Environment **probes** (detect → `environment.json`) | **built** (`bench/environment.py`) |
| Scheduler **adapters** — selection + `sweep_K` | **built** (`bench/adapters.py`); `format_*` next |
| `environment@1` schema (persistence) | **built** (`bench/environment.py`) |
| **prep-bench** — on-target detect + format | **proposed** (§ 7.2, § 4); detection done, formatting next |
| **summarize** — results → `bench-result.json` | **proposed** (§ 7.4, § 5) |
| **prep-run** — `bench-result.json` → run script | **proposed** (§ 7.5, § 4) |
| `bench-result@1` schema (persistence) | **proposed** (§ 5.3) |

This doc specifies the proposed pieces precisely enough to build them
independently, against stable interfaces (§ 4) and data formats (§ 5).

---

## § 1 Terminology

Plain definitions so the rest of the doc needs no inline glossing.

| Term | Meaning |
|---|---|
| **host** | the machine where the user *generates* the bundle (laptop, login node). Has molbuilder installed. |
| **target** | the machine that *runs* the calculation (a workstation, or a supercomputer's compute node). |
| **backend env** | the conda environment that holds the compute engine (e.g. `molbuilder-siesta-gpu`). It deliberately has **no** molbuilder/numpy — so anything that must run there is plain Python-stdlib or shell. |
| **scheduler** | the software that decides when/where a job runs. Two kinds here: **SLURM** (shared clusters — you submit a job and wait in a queue) and **workstation** (no scheduler — you just run the program). |
| **topology** | the target's hardware shape: **sockets** (physical CPU packages), **cores per socket**, **NUMA nodes** (memory regions local to a group of cores), and **GPUs**. Placing work on cores near its GPU/memory avoids slow cross-region access. |
| **engine** | the program doing the physics (here SIESTA), in a **CPU** variant (plain solver) or a **GPU** variant (GPU-accelerated solver). |
| **G / K / c** | GPU-run knobs: **G** = number of GPUs, **K** = MPI ranks (processes) sharing each GPU, **c** = CPU cores (compute threads) per rank. Total ranks = G·K; cores used per socket = K·c. |
| **sm%** | GPU compute utilization (0–100 %) — how busy the GPU's processors are. High = the GPU is the bottleneck; low = it is idle waiting on the CPU. |
| **GPU-bound / CPU-bound** | which resource limits the run. Sustained high sm% = **GPU-bound** (ideal). Low sm% while the CPUs are pegged = **CPU(host)-bound** (the GPU is starved). |
| **per-iter (steady-state)** | seconds per solver iteration once warm-up is past — the headline speed metric. |
| **bundle** | the portable set of files `bench generate` produces (§ 6). |
| **probe / adapter / Environment** | the pluggable detection + formatting machinery (§ 4). |

---

## § 2 Goal & philosophy — target isolation

A user should be able to:

1. generate a benchmark **bundle** once, on any host;
2. copy it (`scp`/`rsync`) to a target they may know little about;
3. run a couple of self-contained scripts on the target that **detect**
   the machine (scheduler? sockets? GPUs?) and **format** the run for it;
4. never open an editor to fix a queue name, a core count, a thread
   count, or a launch command.

**The machine-specific knowledge lives in the target-side scripts, not in
the user's head.** The bundle is *portable*; everything that depends on
the target is resolved **on** the target. This generalizes a pattern the
project already uses: `mb_monitor.py` and `job-gpu-sweep.sh` are
stdlib/shell, ship with the bundle, and run on the target with no
molbuilder install.

**Corollary (the self-contained rule):** any script that runs on the
target must import only Python-stdlib or be plain shell — the backend env
has nothing else (§ 1).

### § 2.1 Two targets, one workflow

The same steps (§ 3) run on a **workstation** and a **supercomputer**.
The *only* differences are confined to one pluggable piece — the
scheduler **adapter** and the **topology probe** (§ 4). Everything else is
shared and scheduler-agnostic.

| Stage | Workstation | Supercomputer (SLURM) |
|---|---|---|
| detect | no `sbatch` found → workstation | `sbatch` present → SLURM |
| topology | read locally (`lscpu`, `nvidia-smi`) — prep runs **on** the box | ask the scheduler (`scontrol show node`) — describes the *compute* node, from the login node |
| bench run | one script runs the points **one after another** (single box, shared GPUs) | one job **per point**, all **queued at once** |
| memory | estimator **informs** (prints estimate vs the machine's RAM); no hard limit | estimator **sets** the job's memory request (a hard limit) |
| environment entry | `conda activate <env>` | `module load …` + `source activate <env>` |
| launch | run the script directly | submit it to the queue |
| monitor / utilization / timing | **identical** | **identical** |

**Already workstation-ready (built):** the monitor (progress, utilization,
timing) is scheduler-free; the runtime memory check already has a
no-scheduler branch (reads the machine's RAM and just reports, see
slurm-integration.md § 11.0d); `bench generate` already emits a runnable
launch script when no scheduler is configured. The existing `molbuilder
bench siesta-gpu` command is the **seed of the workstation runner** — it
already runs a sweep of points *sequentially on the current machine*.

**Still needed (the workstation adapter, proposed):** make the sweep
helper emit direct run commands instead of queue submissions, and have
prep-bench/prep-run select the workstation vs SLURM adapter from the
detected Environment. **No engine, monitor, or estimator code changes** —
that is the entire point of confining the difference to the adapter.

---

## § 3 The workflow

```
   host (has molbuilder)              target machine
   ┌─────────────────────┐  copy   ┌──────────────────────────────────────┐
   │ molbuilder bench    │ ──────▶ │ 1. prep-bench  detect + format        │
   │ generate <fdf>      │         │       │        (scheduler + topology)  │
   │  → portable bundle  │         │       ▼                                │
   └─────────────────────┘         │ 2. run bench ──▶ monitor:             │
                                    │       │          timing + util.csv     │
                                    │       ▼                                │
                                    │ 3. summarize → bench-result.json       │
                                    │       │        (portable "hints")      │
                                    │       ▼                                │
                                    │ 4. prep-run <bench-result.json>        │
                                    │       │        format production run   │
                                    │       ▼                                │
                                    │ 5. submit / launch production          │
                                    └────────────────────────────────────────┘
```

| Step | What happens | Component (§ 7) |
|---|---|---|
| 0 | copy the bundle to the target | — |
| 1 | detect scheduler + topology; format the benchmark scripts | prep-bench (§ 7.2) |
| 2 | run the CPU + GPU points; monitor records timing + utilization | run-bench (§ 7.3) |
| 3 | read each point's outputs; write `bench-result.json` | summarize (§ 7.4) |
| 4 | from `bench-result.json` + the production input, format the run | prep-run (§ 7.5) |
| 5 | launch / submit production | — |

---

## § 4 Architecture — what is shared vs what is pluggable

### § 4.1 The single axis of variation
Across all targets, exactly **one thing varies: the scheduler**
(workstation vs SLURM vs a future PBS/cloud). It changes two concerns:
- **detection** — how you learn the topology + site facts;
- **formatting** — how you turn a job into a runnable/submittable script.

Everything else — transforming the input, launching, monitoring,
estimating memory, measuring timing, and the result format — is
**shared** and identical on every target. The design isolates the
varying concern behind two small interfaces so adding a target is adding
one adapter, not editing the core.

### § 4.2 Shared core (engine/scheduler-agnostic, mostly built)
- input transform (`bench/generate.py`: cold, capped, comparable fdfs),
- the launch wrapper + the self-contained monitor (timing + utilization),
- the memory estimator + runtime check,
- the result format (§ 5).

### § 4.3 Interfaces (the pluggable seam)
```
# A PROBE answers one question about the target.  Probes are ordered;
# the first that applies wins; explicit user flags override all.
Probe:
    applies() -> bool                 # usable on this machine?
    detect()  -> partial Environment  # contributes some fields

# An ADAPTER turns an abstract job into concrete scripts for one
# scheduler family.  One adapter serves BOTH bench and run (reuse).
SchedulerAdapter:
    name                                   # "slurm" | "workstation" | ...
    matches(env) -> bool
    sweep_K(topology) -> [int]             # core-efficient rank counts
    format_bench(bundle, env) -> [script]  # the benchmark scripts
    format_run(job, choice, env) -> [script]  # the production script
```
`Environment` is a plain data record (§ 5.2): `scheduler`, `topology`,
`site`. It is produced by probes and consumed by adapters — neither side
knows the other's internals, only this record.

### § 4.4 Resolution order
`resolve_environment()`:
1. **scheduler probe** — `sbatch`/`scontrol` on `PATH` or `SLURM_*` in the
   environment → SLURM; otherwise workstation.
2. **topology probe** — § 4.6, scheduler-aware.
3. **site probe** — scheduler-specific facts (SLURM: default queue from
   `sinfo`; workstation: none).
Explicit flags (`--scheduler`, `--cores-per-socket`, …) override any
detected value and are recorded as `source: "flag"` (§ 5.2).

### § 4.5 Built-in adapters
- **SlurmAdapter** — emits a thin queue-submission header that calls the
  shared launch script (slurm-integration.md § 3); fills queue, ranks,
  threads, GPUs, and memory request from the Environment; `sweep_K` =
  divisors of cores-per-socket.
- **WorkstationAdapter** — no scheduler; emits one script that launches
  each point directly, sized to the local cores/GPUs; the sweep runs
  points sequentially.

### § 4.6 Topology probes (must describe the *compute* node)
Detection has to describe where the job will **run**, not where prep
runs (a login node has a different CPU and usually no GPU). Source
priority, recorded in the data (§ 5.2):
1. **scheduler** — `scontrol show node <node-in-queue>` gives sockets,
   cores/socket, threads/core, and GPUs of the real compute node, askable
   from the login node;
2. **local** — `lscpu` + `nvidia-smi -L`, valid only when prep runs *on*
   the target (a workstation, or inside an interactive allocation);
3. **declared** — the `--cores-per-socket` / `--gpus-per-node` flags, or
   the bundle defaults. Always overridable.

### § 4.7 Extension guide
- **New scheduler** (PBS, LSF, cloud batch): implement one
  `SchedulerAdapter` (+ a topology probe if its source differs) and
  register it. `resolve_environment` and both prep scripts pick it up via
  `matches()`. **No change** to generate, run, monitor, estimator, or the
  data schemas.
- **New engine** (e.g. PySCF) or **new metric**: these are *not* the
  scheduler axis. They extend the shared core + the data schema (§ 5.5),
  which is versioned for exactly this. Keep them orthogonal to adapters.

**Self-contained constraint (restated).** The *shipped* adapters/probes
run on the target, so they are Python-stdlib or POSIX shell. The same
logic may also exist as a host-side molbuilder module; keep the two in
sync with a layout-invariant test, as `mb_monitor.py` already does.

---

## § 5 Persisted data model

The workflow's stages are decoupled by **files on disk**, not shared
code. Getting these formats right is what lets information move between
stages, between machines, and into other tools (plots, dashboards, the
web UI). This section is the contract.

### § 5.1 Principles
- **One format: JSON.** Machine- and human-readable, language-agnostic,
  diff-able, and trivially loadable by any downstream tool.
- **Versioned.** Every document starts with a `schema` string
  `"molbuilder/<name>@<major>"`. Consumers check the major version and
  fail loudly on the unknown; new optional fields do not bump major
  (§ 5.5).
- **Explicit units, in the field name.** `*_gb`, `*_s` (seconds),
  `*_pct`. No bare numbers whose unit you must guess.
- **Separation of concerns** — three independent record types, each
  reusable on its own:
  1. **Environment** — *where* (the machine). § 5.2.
  2. **BenchResult** — *what was measured* + the chosen mechanism. § 5.3.
  3. **RunChoice** — *what to do* (the portable decision). Embedded in
     BenchResult as `choice`, but defined separately so prep-run consumes
     only it.
- **Portable vs machine-specific, explicitly split** (§ 5.4) — so a
  consumer knows which fields transfer to another machine and which do
  not.
- **Provenance.** Each document records how it was produced (tool
  version, timestamp, detection source) so a result is interpretable and
  reproducible later.

### § 5.2 `environment.json` (record type: `environment`)
The detected target, written by prep-bench so every later stage and any
external tool reads the *same* machine description.
```json
{
  "schema": "molbuilder/environment@1",
  "detected_at": "2026-06-27T18:30:00Z",
  "scheduler": "slurm",                       // or "workstation"
  "topology": {
    "sockets": 2,
    "cores_per_socket": 24,
    "threads_per_core": 1,
    "numa_per_socket": 4,
    "gpus_per_node": 4,
    "gpu_type": "a100",
    "mem_total_gb": 503
  },
  "site": { "partition": "public", "qos": "public", "account": null },
  "source": {                                  // how each part was learned
    "scheduler": "path:sbatch",
    "topology": "scontrol",                    // scontrol | lscpu | flag
    "site": "sinfo"
  },
  "tool": "prep-bench@1"
}
```
Unknown/未detected fields are `null`, never omitted, so consumers can tell
"absent" from "unknown".

### § 5.3 `bench-result.json` (record type: `bench-result`)
Written by summarize; the *only* input prep-run needs.
```json
{
  "schema": "molbuilder/bench-result@1",
  "generated_at": "2026-06-27T22:00:00Z",
  "environment": { "...": "the § 5.2 record, embedded for self-containment" },
  "system": { "engine": "siesta", "n_atoms": 444, "n_orb": 14848,
              "n_kpoints": 16, "basis": "TZP" },
  "points": [
    { "label": "gpu-k8", "engine": "gpu",
      "knobs": { "gpus": 1, "ranks_per_gpu": 8, "cores_per_rank": 3,
                 "block_size": 256 },
      "metrics": { "s_per_iter": 1538, "iters_measured": 4,
                   "gpu_sm_mean_pct": 91, "cpu_mean_pct": 40,
                   "peak_rss_gb": 25.2 },
      "bound": "gpu",
      "state": "completed" },
    { "label": "gpu-k4", "engine": "gpu",
      "knobs": { "gpus": 1, "ranks_per_gpu": 4, "cores_per_rank": 6 },
      "metrics": { "s_per_iter": 1938, "bound": "gpu", "peak_rss_gb": 22.3 },
      "state": "completed" },
    { "label": "cpu-np64", "engine": "cpu",
      "knobs": { "ranks": 64 },
      "metrics": { "s_per_iter": null, "peak_rss_gb": 433.2 },
      "state": "timeout" }
  ],
  "choice": {                                  // the portable decision (§ 5.4)
    "engine": "gpu",
    "knobs": { "gpus": 1, "ranks_per_gpu": 8, "cores_per_rank": 3,
               "block_size": 256 },
    "rationale": "gpu-k8 fastest (1538 s/iter); GPU-bound, host has headroom"
  },
  "recommend": { "mem_gb": 472, "time": "0-12:00:00" },
  "tool": "bench-summarize@1"
}
```
Field meaning: `s_per_iter` = steady-state seconds/iteration from the
timing log (slurm-integration.md § 11.0); `gpu_sm_mean_pct`/`cpu_mean_pct`
+ `bound` from the utilization summary (§ 9); `peak_rss_gb` from
scheduler accounting (or the monitor's high-water mark on a workstation);
`recommend.mem_gb` from the estimator, validated against `peak_rss_gb`.

### § 5.4 What transfers across machines (the translation rule)
A result measured on machine A is consumed to plan a run on machine B
(re-run, or a different target). Fields split cleanly:

- **Portable** (transfer as-is): `system`, `choice.engine`,
  `bound`, and the *relative* ranking of mechanisms. *Which mechanism
  wins* (GPU vs CPU; more ranks/GPU helping) is usually stable across
  similar hardware.
- **Machine-specific** (do **not** transfer; re-derive from B's
  Environment): the absolute `s_per_iter`, `peak_rss_gb`, the recommended
  `mem_gb`, and the concrete `-n`/`-c`/GPU/memory values. prep-run takes
  the portable `choice` and *re-resolves* the concrete knobs from B's
  detected topology via B's adapter.

This is why `choice` is a separate record from `metrics`: the decision is
portable; the measurements are not. A consumer that only wants "what
should I run" reads `choice` and ignores the rest.

### § 5.5 Versioning & extension
- Add an **optional** field → no version bump; old consumers ignore it.
- Change/remove a field's meaning, or add a **required** field → bump the
  major (`@1` → `@2`) and provide a one-step migration note here.
- A new **engine** adds engine-specific `knobs`/`system` keys under its
  own names; the envelope (`schema`, `points`, `choice`) is unchanged, so
  a generic consumer (a plot of `s_per_iter` vs `label`) still works.
- A new **metric** is a new key under `metrics`; absent on older results,
  so consumers treat missing as "not measured."

---

## § 6 Output streams & file inventory

### § 6.1 Three separate streams (no tangling, no redundancy)
A run produces three **distinct kinds** of information that must **never
be mixed into one file**, because they have different owners, lifetimes,
and audiences. Each datum lives in exactly one stream — no duplication.

1. **Scientific (raw)** — the engine's own physics output: energies, SCF
   convergence, forces, geometry. Owned by the engine (SIESTA), authored
   to its native file. We do **not** write into it or copy from it.
2. **Performance & event (operational)** — *how the run executed*: the
   launch command and resolved placement, environment activation, the
   memory check, progress ticks, stalls, per-iteration timing, and
   machine utilization. This is the "ops" record; it carries **no**
   authoritative physics.
3. **Summaries (distilled)** — short machine-readable digests, themselves
   split by kind so they don't re-tangle:
   - **scientific summary** — the *result* of the physics (final energy,
     converged?, max force, …). Separate from the raw scientific log
     **and** from the performance log.
   - **performance summary** — the benchmark verdict (`bench-result.json`,
     § 5.3): timing, utilization, memory, and the chosen mechanism.

Rule of thumb: *raw vs distilled* on one axis, *science vs ops* on the
other — four quadrants, four destinations, no overlap.

| | Raw | Summary |
|---|---|---|
| **Science** | engine `.out` | `result.json` (scientific) |
| **Ops** | wrapper log, timing log, `util.csv`, `monitor.log` | `bench-result.json` (performance) |

The one datum to keep honest: the monitor's progress tick may show the
*latest* energy as a liveness/sanity glance, but it is explicitly **not**
the scientific record — the authoritative physics lives only in the raw
`.out` and the scientific `result.json` (§ 9).

### § 6.2 File inventory (what each is, and how to use it)

| File | Stream | What it is / how to use it |
|---|---|---|
| `*-runN.out` | science (raw) | engine stdout — energies, SCF, forces. The authoritative physics; parse for results, archive as the record of truth. |
| `*-runN.scf-timing.log` | ops (raw) | one timestamp per solver iteration. Compute steady-state s/iter (`total/N`); the timing instrument (slurm-integration.md § 11.0). |
| `*.util.csv` | ops (raw) | change-gated cpu%/mem/GPU-sm%/VRAM samples. Plot vs time to see GPU-bound vs CPU-bound; feeds the perf summary (§ 9). |
| `*.monitor.log` | ops (event) | progress ticks, stalls, `[UTIL-SUMMARY]`, notifications. Scan for liveness + the bound verdict; **not** the physics record. |
| wrapper/runwrap log | ops (event) | resolved launch command, env, ranks/threads, GPU placement, memory check, error diagnostics. The post-mortem of *how* it ran. |
| `result.json` *(proposed)* | science (summary) | distilled physics (final energy, converged, max force). Machine-readable result; what dashboards/UI read. Parsed from the `.out`, separate from it. |
| `bench-result.json` | ops (summary) | the benchmark verdict + portable `choice` (§ 5.3). prep-run's only input. |
| `environment.json` | (detected facts) | the target machine description (§ 5.2); shared by every later stage + external tools. |
| `job-{cpu,gpu}.{fdf,run.sh,sbatch}`, `job-gpu-sweep.sh`, `<prod>.{run.sh,sbatch}` | (scripts) | the generated/formatted run scripts (generate → prep-bench / prep-run). |

Every generated file should carry a one-line self-description in its own
header (the scripts already do; the logs get a first-line banner naming
the stream + a pointer to this section), so a user opening any file knows
what it is without consulting the docs.

---

## § 7 Components & responsibilities

### § 7.1 `molbuilder bench generate` (host, built)
Emits the portable bundle (`bench-generate-spec`): `job-cpu.fdf`,
`job-gpu.fdf` (cold, iteration-capped, `SCF.MustConverge` off so a capped
run still exits cleanly), the pseudopotentials, `job-gpu-sweep.sh`,
`README.md`. Topology flags it takes today become **fallbacks** once
prep-bench detects topology on the target (§ 4.6).

### § 7.2 prep-bench (target, proposed)
Self-contained. Runs `resolve_environment()` (§ 4.4), writes
`environment.json` (§ 5.2), and uses the matching adapter to format the
benchmark scripts + size the sweep. Prints what it detected and the
source; never silently guesses.

### § 7.3 run-bench (target, mostly built)
Launches the points (queue-submitted in parallel under SLURM; sequential
on a workstation). The launcher + monitor are built; the monitor emits
`util.csv` (§ 9). Per-point outputs: timing log, `util.csv`, peak memory.

### § 7.4 summarize (target, proposed)
Reads each point's timing + utilization + memory, writes
`bench-result.json` (§ 5.3) including the portable `choice`. Self-contained
so it runs on the target.

### § 7.5 prep-run (target, proposed)
Reads `bench-result.json` + the production input, takes the portable
`choice`, re-resolves the concrete knobs from the local Environment via
its adapter (§ 5.4), and writes the production run script. No hand editing.

---

## § 8 Detection rules (quick reference)

- **Scheduler:** `command -v sbatch` (or `$SLURM_*`) → SLURM; else
  workstation. `--scheduler` overrides.
- **Topology:** `scontrol show node` (SLURM, compute-node-correct) →
  `lscpu` + `nvidia-smi` (on the box) → flags. The chosen source is
  recorded in `environment.json`.
- **Sweep ranks (K):** divisors of cores-per-socket, so every point fully
  uses the socket (`K·c = cores`, `c = cores // K`). A K that does not
  divide leaves cores idle (e.g. 24/16 → c=1, 8 idle); the helper flags
  the utilization rather than silently wasting cores.

---

## § 9 Utilization & timing (built — the run-bench instrumentation)

The single monitor loop (slurm-integration.md § 11.0c, § 11.0e) also
samples utilization, so a post-run plot shows *why* a point was fast or
slow:
- wakes every few seconds (default 5 s); cheap reads of `/proc` +
  `nvidia-smi`;
- writes `util.csv` **change-gated** — a row only when a number moves
  ≥ 10 % from the last logged row (plus an occasional keepalive) — so the
  file stays small while timestamps keep the plot clean;
- at finish, a `[UTIL-SUMMARY]` line gives the **GPU-bound vs CPU-bound**
  verdict from the average GPU sm% / CPU% (§ 1).

This feeds `bound` and the utilization metrics in `bench-result.json`
(§ 5.3). Stdlib-only, so it ships in `mb_monitor.py` (the self-contained
rule, § 2).

`util.csv` and `monitor.log` are the **ops** stream (§ 6.1): timing,
utilization, progress, events — never the authoritative physics. The
progress tick's `energy=` is a non-authoritative liveness glance only;
the scientific record stays in the raw `.out` and the scientific
`result.json`.

---

## § 10 Open questions

- **summarize home:** extend `job-gpu-sweep.sh`, a standalone
  `bench-summarize.sh`, or a `molbuilder bench summarize` host command
  (with a shipped shell twin)?
- **prep language:** one stdlib-Python tool (cleaner adapter abstraction)
  with a thin shell launcher, vs pure shell (maximum portability). Leaning
  Python-stdlib + shell launcher.
- **multi-node production:** the estimator and the single-node assumption
  (`-N 1`) are documented limits (slurm-integration.md § 6); the data
  schema already carries `sockets`/node counts to grow into this.
