# Benchmark workflow — target-isolated, self-configuring

> **Design doc.** Authoritative for the end-to-end "benchmark then run"
> flow that lets a user drop a bundle onto *any* target machine and have
> it detect + format itself — **no hand-tuning, no hand-crafted scripts.**
> Cross-references: `slurm-integration.md` (the SLURM submission layer),
> `bench-generate-spec` (the bundle generator), `config.md` (script
> generation). Status of each piece is tracked in § 0.

---

## § 0 Status

| Piece | State |
|---|---|
| `molbuilder bench generate` (portable bundle) | **built** (`bench/generate.py`) |
| Monitor utilization sampling (cpu%/GPU-sm%) | **built** (`monitor.py`, § 8) |
| `--mem` estimator + runtime audit | **built** (`siesta/memory.py`, slurm-integration.md § 11.0d) |
| GPU↔CPU socket co-location | **built** (slurm-integration.md § 7.5.2) |
| **prep-bench** (target-side detect + format) | **proposed** (§ 3.2, § 4) |
| **prep-run** (bench-hints → production script) | **proposed** (§ 3.5, § 5) |
| Environment adapters (SLURM / workstation / …) | **proposed** (§ 4) |

This doc specifies the proposed pieces so they can be built modularly.

---

## § 1 Goal & philosophy — target isolation

A user should be able to:

1. generate a benchmark bundle **once**, anywhere (laptop, login node);
2. `scp` it to a target machine they may know little about;
3. run **two self-contained scripts** on the target that *detect* the
   machine (scheduler? sockets? GPUs?) and *format* the run correctly;
4. never open an editor to fix a partition name, a core count, a `-c`
   value, or an `mpirun` line.

**The machine-specific knowledge lives in the target-side scripts, not in
the user's head.** The generated bundle is *portable*; everything that
depends on the target is resolved **on** the target at prep time. This is
the same principle the launcher already uses (`mb_monitor.py`,
`job-gpu-sweep.sh` are stdlib/bash, ship with the bundle, and run on the
node with no molbuilder install) — generalized to the whole workflow.

Corollary: the prep scripts MUST be **self-contained** (stdlib python or
plain bash, no molbuilder/numpy import), because the target backend env
(e.g. `molbuilder-siesta-gpu`) has none of those.

### § 1.1 Two targets, one workflow

The **same six steps** (§ 2) must run on a **local workstation** and a
**remote supercomputer**. The only differences are confined to the
`SchedulerAdapter` + `TopologyProbe` (§ 4); everything else — the fdf
transform, the launcher, the monitor + `util.csv`, the timing instrument,
the `--mem` estimator, the bench-hints contract — is **scheduler-agnostic
and shared**.

| Stage | Workstation | Supercomputer (SLURM) |
|---|---|---|
| detect | no `sbatch` ⇒ `workstation` | `sbatch`/`$SLURM_*` ⇒ `slurm` |
| topology | `lscpu` + `nvidia-smi -L` (prep runs **on** the box) | `scontrol show node` (compute-node-correct, from login) |
| bench scripts | one `run-bench.sh` that `mpirun`s each point **sequentially** (one box, shared GPUs) | `.sbatch` per point, **submitted in parallel** to the queue |
| sweep | helper emits `./job-gpu.run.sh -n K*G …` lines (direct) | helper emits `sbatch --gres … -n K*G …` lines |
| memory | estimator **informs** (`[INFO] est vs node RAM`); no hard `--mem` | estimator **sizes** `#SBATCH --mem` (gates the cgroup) |
| env entry | config `activation` = `conda activate <env>` (no `module load`) | `module load mamba` + `source activate` |
| submit | `./run.sh` | `sbatch <prod>.sbatch` |
| monitor / util / timing | **identical** | **identical** |

**Already workstation-ready (built):** the monitor + `util.csv`/timing are
scheduler-free; the runtime `--mem` audit already has a no-SLURM branch
(reads `/proc/meminfo`, prints "node RAM ~XG (no SLURM --mem)",
slurm-integration.md § 11.0d); `bench generate` already emits a runnable
`.run.sh` when no scheduler is configured (§ 10 there). The existing
`molbuilder bench siesta-gpu` is the **seed of the workstation runner** —
it already sweeps points *sequentially on the current machine*.

**Needs the workstation adapter (proposed):** make `job-gpu-sweep.sh`
emit direct-exec lines (not `sbatch`) in workstation mode; have
prep-bench/prep-run pick `WorkstationAdapter` vs `SlurmAdapter` from the
detected Environment. No engine/monitor/estimator code changes — that is
the point of isolating the difference in the adapter.

---

## § 2 The workflow

```
   ┌─ host (has molbuilder) ─┐        ┌──────────── target machine ────────────┐
   │  molbuilder bench       │  scp   │  ./prep-bench      detect + format      │
   │  generate <fdf>         │ ─────▶ │       │            (scheduler+topology) │
   │   → portable bundle     │        │       ▼                                 │
   └─────────────────────────┘        │   run bench  ──▶  monitor + util.csv    │
                                       │       │            (timing + sm%/cpu%)  │
                                       │       ▼                                 │
                                       │   bench-result.json  (the "hints")      │
                                       │       │                                 │
                                       │       ▼                                 │
                                       │   ./prep-run <hints>   format prod run  │
                                       │       │                                 │
                                       │       ▼                                 │
                                       │   submit run                            │
                                       └─────────────────────────────────────────┘
```

Steps, verbatim from the design:

1. **copy all files over** — `scp`/`rsync` the bundle to the target.
2. **prep-bench** — run on the target: detect SLURM vs workstation +
   topology, format the benchmark scripts (partition, `-n`/`-c`/`--gres`,
   the topology-derived K sweep, or plain `mpirun` on a workstation).
3. **run bench** — submit/launch the CPU + GPU points; the monitor
   captures per-iter timing (`scf-timing.log`) and utilization
   (`util.csv`); accounting captures `MaxRSS`.
4. **wait for result** — points finish (`COMPLETED`, thanks to
   `SCF.MustConverge .false.`).
5. **prep-run** — run on the target: consume the **bench hints** (which
   config won) and format the **production** run script for this machine.
6. **submit run** — launch production with the chosen mechanism.

---

## § 3 Components & responsibilities

### 3.1 `molbuilder bench generate` (host, built)
Emits the **portable** bundle (`bench-generate-spec`): `job-cpu.fdf`,
`job-gpu.fdf` (cold, `MaxSCFIterations` capped, `SCF.MustConverge
.false.`), the pseudos, `job-gpu-sweep.sh`, `README.md`. Today it also
bakes topology defaults (`--cores-per-socket`, `--gpus-per-node`) — under
the new flow those become **fallbacks**, since prep-bench detects them on
the target (§ 4).

### 3.2 prep-bench (target, **proposed**)
Self-contained. Resolves the **Environment** (§ 4) and **formats** the
benchmark scripts for it:
- SLURM → `.sbatch` headers with the detected partition/qos, `-n`/`-c`/
  `--gres`, and the K sweep derived from detected cores/socket;
- workstation → a plain `run-bench.sh` that `mpirun`s each point directly
  (no scheduler), sized to the local core/GPU count.
Echoes what it detected and the source; never silently guesses.

### 3.3 run-bench (target, partly built)
Submits/launches the points. The launcher (`.run.sh`) + monitor are
already built; the monitor now emits `util.csv` (§ 8). Output per point:
`*-run0.scf-timing.log` (per-iter wall), `*.util.csv` (cpu%/sm%),
`MaxRSS` (accounting).

### 3.4 result aggregation → bench hints (**proposed**)
A small step (could fold into the sweep helper or a `bench summarize`)
that reads each point's timing + util + RSS and writes
**`bench-result.json`** (§ 5) — the machine-readable verdict:
winning engine (cpu/gpu), G/K/c, per-iter wall, GPU-bound vs host-bound,
peak RSS, recommended `--mem`/`--time`.

### 3.5 prep-run (target, **proposed**)
Self-contained. Reads `bench-result.json` + the **production** fdf and
formats the production run script for this Environment (same adapter
machinery as prep-bench, § 4) using the winning config. The user runs it;
no hand editing.

### 3.6 submit-run
`sbatch <prod>.sbatch` (SLURM) or `./run.sh` (workstation).

---

## § 4 The environment-probe/format architecture (modular, extensible)

The core requirement: **probing and formatting are modular and isolated,
with clear room for extension** (a new scheduler = a new adapter, nothing
else changes). Two small interfaces:

```
Environment = {
    scheduler:  "slurm" | "workstation" | <future>,
    topology:   Topology,          # sockets, cores_per_socket, threads,
                                   # gpus_per_node, gpu_type, numa_per_socket
    site:       {partition, qos, account, ...}   # scheduler-specific
}

# A PROBE answers one question about the target, in priority order.
class Probe:
    def applies(self) -> bool        # is this probe usable here?
    def detect(self) -> partial(Environment)

# An ADAPTER formats a job for one scheduler family.
class SchedulerAdapter:
    name: str
    def matches(self, env) -> bool
    def format_bench(self, bundle, env) -> [scripts]
    def format_run(self, fdf, hints, env) -> [scripts]
    def k_sweep(self, topology) -> [K]      # core-efficient K values
```

### 4.1 Resolution order (detection)
`resolve_environment()` runs probes in priority order, first hit wins,
explicit user flags override everything:

1. **SchedulerProbe** — `sbatch`/`scontrol` on PATH or `SLURM_*` env ⇒
   `slurm`; else `workstation`.
2. **TopologyProbe** (scheduler-aware, § 4.3).
3. **SiteProbe** — scheduler-specific site facts (SLURM: default
   partition/qos from `sinfo`; workstation: none).

### 4.2 Built-in adapters
- **`SlurmAdapter`** — emits the thin `.sbatch` header → `bash .run.sh`
  (slurm-integration.md § 3); fills partition/qos/`-n`/`-c`/`--gres`/
  `--mem` from the Environment; `k_sweep` = divisors of
  `cores_per_socket`.
- **`WorkstationAdapter`** — no scheduler; emits a `run.sh` that `mpirun
  -np <n>`s directly, sized to local cores/GPUs; `k_sweep` bounded by the
  single box.

### 4.3 Topology probes (scheduler-aware, target-correct)
**Detection must describe the COMPUTE node, not where prep runs** (a login
node has a different CPU and usually no GPU). Source priority:
1. **SLURM** → `scontrol show node <node-in-partition>` (gives `Sockets`,
   `CoresPerSocket`, `ThreadsPerCore`, `Gres=gpu:<type>:<n>`) — the real
   GPU node, queryable from the login node.
2. **local** → `lscpu` + `nvidia-smi -L` (only valid when prep runs *on*
   the node: a workstation, or inside an `salloc`).
3. **fallback** → the `--cores-per-socket`/`--gpus-per-node` flags /
   bundle defaults. Always overridable.

### 4.4 Extension — adding a scheduler (e.g. PBS/LSF/cloud)
Implement one `SchedulerAdapter` (+ a `TopologyProbe` if the topology
source differs) and register it. `resolve_environment` and the prep
scripts pick it up by `matches()`; **no change to generate / run / the
hints contract.** This is the isolation the design requires: scheduler
knowledge is confined to its adapter.

> **Self-contained constraint.** Because prep runs in the backend env, the
> reference adapters are stdlib python (shippable like `mb_monitor.py`) or
> POSIX sh. The same adapter *logic* may also exist as a molbuilder Python
> module for host-side use, but the **shipped** prep scripts must not
> import molbuilder. Keep the two in sync via a layout-invariant test
> (as `mb_monitor.py` already does).

---

## § 5 The bench-hints contract (`bench-result.json`)

The decoupling point between bench and run: prep-run consumes only this,
so the two stages don't share code paths.

```json
{
  "schema": "molbuilder/bench-result@1",
  "system": {"n_atoms": 444, "n_orb": 14848, "n_kpoints": 16},
  "winner": {"engine": "gpu", "gpus": 1, "ranks_per_gpu": 8,
             "cpus_per_task": 3, "block_size": 256},
  "points": [
    {"label": "gpu-k8", "engine": "gpu", "g": 1, "k": 8, "c": 3,
     "s_per_iter": 1538, "gpu_sm_mean": 91, "cpu_pct_mean": 40,
     "bound": "gpu", "max_rss_gb": 25.2, "state": "COMPLETED"},
    {"label": "gpu-k4", "s_per_iter": 1938, "bound": "gpu", ...},
    {"label": "cpu-np64", "s_per_iter": null, "state": "TIMEOUT", ...}
  ],
  "recommend": {"mem_gb": 472, "time": "0-12:00:00"}
}
```

`s_per_iter` is the steady-state `total/N` from `scf-timing.log` deltas
(slurm-integration.md § 11.0); `bound` comes from the `util.csv`
`[UTIL-SUMMARY]` (§ 8); `max_rss_gb` from `sacct`; `mem_gb` from the
estimator validated against that RSS.

---

## § 6 File inventory

| File | Stage | Producer |
|---|---|---|
| `job-{cpu,gpu}.{fdf,run.sh,sbatch}` | bench | generate / prep-bench |
| `job-gpu-sweep.sh` | bench | generate (topology baked by prep-bench) |
| `*-run0.scf-timing.log` | run-bench | launcher (`_mb_scf_tee`) |
| `*.util.csv` | run-bench | monitor (§ 8) |
| `*.monitor.log` (`[UTIL-SUMMARY]`) | run-bench | monitor |
| `bench-result.json` | aggregate | summarizer (§ 3.4) |
| `<prod>.{run.sh,sbatch}` | prep-run | prep-run |

---

## § 7 Detection rules (summary)

- **Scheduler:** `command -v sbatch` (or `$SLURM_JOB_ID`/`$SLURM_*`) ⇒
  SLURM; else workstation. Explicit `--scheduler` overrides.
- **Topology:** `scontrol show node` (SLURM, compute-node-correct) →
  `lscpu`+`nvidia-smi` (local) → flags. Echo source.
- **K sweep:** divisors of `cores_per_socket` so every point fully uses
  the socket (`K·c = cores`); `c = cores // K`. K that don't divide leave
  cores idle (e.g. 24/16 → c=1, 8 idle) — the helper flags utilization.

---

## § 8 Utilization & timing (built — the run-bench instrumentation)

The single monitor loop (slurm-integration.md § 11.0c) also samples
utilization (§ 11.0e there / `monitor.py`):
- wakes every `--interval` (default 5 s); cheap `/proc` + `nvidia-smi`.
- writes `util.csv` **change-gated**: a row only when a metric moves ≥10%
  from the last logged row (+ a keepalive), so the file stays small while
  timestamps keep the plot clean.
- `[UTIL-SUMMARY]` at finish → the **GPU-bound vs host/CPU-bound** verdict
  (sustained GPU sm% high ⇒ GPU-bound; low while cpu% pegged ⇒ host-bound).

This is how a bench point reports *why* it was fast/slow, which feeds
`bound` in the hints (§ 5).

---

## § 9 Open questions

- Where does the summarizer live — extend `job-gpu-sweep.sh`, or a new
  self-contained `bench-summarize.sh` / `molbuilder bench summarize`?
- prep-bench/prep-run language: one stdlib-python tool with sh shims, or
  pure sh? (python eases the adapter abstraction; sh maximizes
  portability.) Lean python-stdlib + a thin sh launcher.
- Multi-node production (the estimator + `-N 1` are single-node today,
  slurm-integration.md § 6 NIT B-6).
