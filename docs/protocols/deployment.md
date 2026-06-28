# Deployment & running operations

The single reference for *how a generated job actually runs* on a target
machine — the environment it needs, who prepares it, and how scheduler /
CUDA / system parameters are relayed into the job. Consolidates the
operational decisions; defers the sbatch block-by-block detail to
[`slurm-integration.md`](slurm-integration.md) and env *building* to
[`README_install.md`](../README_install.md).

---

## 0. The two phases (the mental model)

Deployment is **two separate phases**. Conflating them is the #1 source of
confusion.

| | Phase A — **env preparation** | Phase B — **job execution** |
|---|---|---|
| Who | the **user**, once per machine | the generated run scripts |
| What | builds + verifies the four envs (host + backends) | activates a prepared env and runs SIESTA |
| Tooling | `molbuilder envs install / bootstrap / doctor / repair` | `job-*.run.sh` / `*.sbatch` / `run-transport.sh` |
| Owns the env? | **yes — creates/repairs it** | **no — only consumes it** |

So all the `molbuilder envs …` machinery (install/doctor/repair) **is** the
user's preparation toolkit. The run scripts never build, doctor, or repair
an env — they assume it exists. They connect at exactly one point: if a run
script finds the needed env missing, it **stops and points the user back to
Phase A** (`molbuilder envs doctor <env>`).

---

## 1. Targets: workstation vs SLURM

The scheduler is probed (workstation = no scheduler on PATH; SLURM =
`sbatch` present / `SLURM_JOB_ID` set).

- **Workstation** — the user has full control. conda/mamba is on PATH; the
  run script can probe it directly. Jobs run via `bash job-*.run.sh`.
- **SLURM (e.g. ASU Sol)** — a *constrained* environment: the job lands in a
  **clean shell** (`--export=NONE`), so `mamba`/`conda` is **not** on PATH
  until `module load mamba` runs. That load step therefore **must** come from
  config (it can't be auto-detected). Jobs run via `sbatch *.sbatch`.

---

## 2. Environment resolution in the run script (the algorithm)

One rule: **config if present, else probe.**

```
1. LOAD conda/mamba:
     if .molbuilder.json `script_generation` is set
         -> run its `preamble` (e.g. "module load mamba")
         -> use its `activation` ("source activate" | "conda activate")
     else (workstation, no config)
         -> probe `mamba` then `conda` on PATH; use that

2. PICK the env (Phase-A env, already prepared by the user):
     molbuilder-siesta-gpu   FIRST   (the ELPA superset; ELPA is only here)
     molbuilder-siesta       only if the .fdf requests NO ELPA
                                      (precompiled fallback; user's job to
                                       confirm the .fdf is compatible)

3. CHECK it exists  (`conda env list` / `mamba env list`):
     present  -> continue
     missing  -> STOP with: "prepare <env> first: molbuilder envs doctor"
                 (Phase A — the script does not create it)

4. RUN:  conda run -n <env> siesta <fdf>     (PATH already correct;
                                               nothing to "locate")
```

This is what makes a single bundle portable: the **workstation** auto-probes,
the **HPC** target reads its config. The activation is the only
target-specific knob, and it lives in (or is detected for) one place.

> **Current state vs target.** Today the activation/preamble are *baked into*
> the `.run.sh` at generate time (from config → flags → auto-detected local
> conda). Moving steps 1–3 to *runtime* (so the exact same bundle runs on
> both a workstation and Sol without regeneration) is the agreed direction;
> until then, generate one bundle per target (§ 6).

---

## 3. `.molbuilder.json` — explicit, never implicit

This file is **not magic** and must never be treated as implicit context.

- **WHERE:** a project-level `<project-dir>/.molbuilder.json` that travels
  *with the bundle* (the run scripts read it from their own directory),
  deep-merged over an optional server-wide `molbuilder.json`. **Project
  wins.**
- **WHAT:** a `scheduler` block (SLURM directives + mem model) and a
  `script_generation` block (how to load + activate conda).
- **Workstation:** you can omit it entirely — activation is auto-detected
  from the local conda, and with no `scheduler` block only `.run.sh` is
  emitted (no `.sbatch`).

**Example (an HPC target — ASU Sol):**

```json
{
  "scheduler": {
    "kind": "slurm",
    "directives": { "partition": "public", "qos": "public",
                    "export": "NONE", "mail_type": "ALL",
                    "mail_user": "%u@asu.edu" },
    "gpu": { "partition": "public", "default_type": "a100",
             "exclusive": true, "mem": "64G" },
    "defaults": { "time": "0-04:00:00", "cpus_per_task": 8, "mem": null },
    "mem_model": { "node_mem_gb": 500, "safety": 1.3, "extra_gb": 0 }
  },
  "script_generation": {
    "preamble": "module load mamba",
    "activation": "source activate"
  }
}
```

**Example (a workstation):** none needed; or just

```json
{ "script_generation": { "activation": "conda activate" } }
```

`script_generation.preamble` is a free-form "run this before activation"
slot — put any environment-specific setup there (module loads, `export`s,
license-server vars, …).

---

## 4. How scheduler / CUDA / system parameters reach the job

Under `sbatch`, SLURM sets the allocation in the job's environment; the
wrapper **consumes** it and **relays** it into the `mpirun` launch. The
relay is the whole point — none of it is assumed.

| Allocation (SLURM sets) | Wrapper reads | Relayed into the job as |
|---|---|---|
| `-n` / `SLURM_NTASKS` | rank count | `mpirun -np N` |
| `-c` / `SLURM_CPUS_PER_TASK` | threads/rank | `OMP_NUM_THREADS` (+ `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) |
| `--gres=gpu:type:G` → `CUDA_VISIBLE_DEVICES` | allocated GPUs | a **per-rank launcher** assigns one GPU/rank + NUMA co-location |
| (K = ranks/GPU ≥ 2) | concurrency | **MPS** daemon (`CUDA_MPS_PIPE_DIRECTORY`) so ranks share the GPU |
| topology | binding | `mpirun --bind-to core --map-by package:PE=$OMP_NUM_THREADS` |
| `--export=NONE` | clean shell | the `module load mamba` preamble is **load-bearing** |

**CUDA is relayed per-rank, not globally.** The wrapper writes a small
helper (`.mb-rank-launch-*.sh`) and launches `mpirun … bash <helper> siesta`;
the helper reads `OMPI_COMM_WORLD_LOCAL_RANK` / `CUDA_VISIBLE_DEVICES` (the
SLURM-allocated set) and `export`s a **single** GPU for that rank, plus
GPU↔CPU socket co-location. So each rank gets exactly its GPU even on a
shared node. Override the socket pin with `MB_NO_SOCKET_PIN=1`.

Full detail: `slurm-integration.md` § 5 (sbatch block), § 7.5 (`mpirun` +
per-rank binding), § 7.5.1 (the K-ranks-per-GPU model), § 7.5.2 (socket
co-location), § 7.9 (CUDA driver floor).

Tuning knobs honored at run time (no regeneration): `MB_NP` /
`MOLBUILDER_MPI_NP`, `OMP_NUM_THREADS` / `MOLBUILDER_OMP_NUM_THREADS`,
`MOLBUILDER_USE_MPS`, `MB_NO_SOCKET_PIN`.

---

## 5. The CPU-vs-GPU benchmark must compare hardware, not solvers

For the `bench generate` bundle, both points use **ELPA**; the GPU point
just adds `Diag.ELPA.GPU .true.`. So the only variable is the CUDA toggle and
the measured difference is *hardware*, not *solver*. (A ScaLAPACK-CPU vs
ELPA-GPU comparison would conflate the two.) Because ELPA lives in
`molbuilder-siesta-gpu`, the CPU point also routes there (§ 2 rule).

---

## 6. Worked commands

**Workstation (conda on PATH → auto-detect; no config needed):**
```bash
molbuilder bench generate input.fdf --out bench-ws/
cd bench-ws/
./prep-bench --gpu-ks 1,2,4,8     # detect machine -> sweep
bash job-gpu-sweep.sh             # sequential; bash job-cpu.run.sh for CPU
./bench-summarize
```

**HPC / Sol (give it the activation it will use THERE):**
```bash
molbuilder bench generate input.fdf --out bench-sol/ \
    --activation "source activate" --preamble "module load mamba"
cd bench-sol/
./prep-bench --gpu-ks 8,16        # detects SLURM + A100
sbatch job-cpu.sbatch            # CPU point
bash job-gpu-sweep.sh            # GPU sweep (sbatch per point)
./bench-summarize
```

Transport is the same shape — `slurm-integration.md` for sbatch, this doc's
§ 2–4 for the env. Prerequisite for both targets: the **envs are already
prepared** (Phase A), and `molbuilder-siesta` / `molbuilder-siesta-gpu` show
up in `conda env list`.

---

## 7. Built vs proposed

| Piece | State |
|---|---|
| sbatch emission + per-rank CUDA relay + MPS + socket co-location | **built** (`runwrap.py`, `slurm-integration.md`) |
| activation/preamble from config → flags → auto-detected local conda | **built** (baked at generate time) |
| `.molbuilder.json` documented in `--help` (path + example) | proposed |
| **runtime** config-else-probe resolution (one bundle, both targets) | proposed (§ 2) |
| env-presence check → pointer to `molbuilder envs doctor` | proposed |
| CPU benchmark point on ELPA (apples-to-apples) | proposed (§ 5) |
