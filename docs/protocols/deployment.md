# Deployment & running operations

The single reference for *how a generated job actually runs* on a target
machine — the environment it needs, who prepares it, and how scheduler /
CUDA / system parameters are relayed into the job. Consolidates the
operational decisions; defers the sbatch block-by-block detail to
[`slurm-integration.md`](slurm-integration.md) and env *building* to
[`README_install.md`](../README_install.md).

**Reading this doc:** §§ 0–1, 4, 6 describe **what is built today**. § 2 (env
resolution), § 3's runtime read, and § 5 (CPU-on-ELPA) describe the **target
model** — the agreed direction not yet fully built. § 7 is the authoritative
built-vs-proposed ledger; every "target" claim below is tagged.

**Want copy-paste worked examples + `.molbuilder.json` templates** for a
workstation and a supercomputer (all situations)? See the cookbook:
[`deployment-examples.md`](deployment-examples.md).

---

## 00. The contract — what "standalone" means, the steps, assumptions, requirements

This section is the definition the rest of the doc depends on. Read it first.

**"standalone" does NOT mean "no molbuilder."** molbuilder **is installed** on
the target. Standalone means a generated script does its whole task **by
itself**: it **activates the correct conda env and runs the job with zero
manual steps**, so it works when **submitted remotely / headless** — `sbatch`,
a queue, cron; no terminal, nobody to type `conda activate`. Submit and forget.

### The three steps and their goals

| step | tool | goal | who owns the decision |
|---|---|---|---|
| **prep** | `prep-bench` / `prep-run` | detect THIS target (scheduler + topology) and **format** the scripts for it (node-sized sweep, `environment.json`). Pure machine-adaptation — no science. | automatic |
| **bench** | `job-*.run.sh` / the sweep | run the short, capped jobs to **measure** (CPU vs GPU, np scaling). Offers the harness + reports wall/iter; makes **no** recommendation. | the scientist decides |
| **run** | `run-production.sh` / `run-transport.sh` | **execute** the production calculation with the chosen mechanism. | the scientist |

### Assumptions (true before any generated script runs)

- **molbuilder is installed** on the target (the host env).
- The backend **envs are already prepared by the scientist** (Phase A,
  `molbuilder envs install`). Scripts never build, doctor, or repair them.
- The activation method (preamble + conda/mamba activate) is **known at
  prep/generate time** — from `.molbuilder.json` config or detection.

### Requirements (every generated run script MUST satisfy)

1. **Self-activating.** It bakes its own env activation (the same mechanism
   `runwrap` emits) and activates the env **itself** — there is **no** manual
   `conda activate` prerequisite, before or after.
2. **Headless-submittable.** `sbatch <script>` (or any queue, no terminal)
   runs it end to end.
3. **One task, start to finish.** activate → run → clean up; it assumes the
   scientist types **nothing**.

A script that requires a manual `conda activate` first (e.g. today's
`run-transport.sh`) **violates requirement 1 and is a bug**, not a design
choice. (Status: the `bench` `job-*.run.sh` meet this contract — they bake
activation via `runwrap`; `run-transport.sh` does **not** yet — § 7.)

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

## 2. Environment resolution — the model *(target; § 7)*

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

4. RUN:  activate <env>, then `mpirun -np N siesta <fdf>`
         (PATH is already correct once activated — nothing to "locate")
```

In this model a **single** bundle is portable: the workstation auto-probes,
the HPC target reads its config; activation is the only target-specific knob.

> **What's built today (vs the model above).** The activation + preamble are
> resolved *at generate time* (config → `--activation`/`--preamble` flags →
> auto-detected local conda) and **baked into** the `.run.sh`; step 2's
> gpu-first selection and step 3's presence-check are not yet emitted into the
> wrapper. Consequence: today you **generate one bundle per target** (§ 6).
> Moving steps 1–3 to *runtime* — so the same bundle runs on a workstation and
> on Sol without regeneration — is the agreed next step (§ 7).

---

## 3. `.molbuilder.json` — explicit, never implicit

This file is **not magic** and must never be treated as implicit context.

- **WHERE:** a project-level `<project-dir>/.molbuilder.json` that travels
  *with the bundle*, deep-merged over an optional server-wide
  `molbuilder.json`. **Project wins.** Today it is read **at generate time**
  (the generator bakes the result into the scripts); in the runtime model
  (§ 2) the run script reads it from its own directory.
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
                    "export": "NONE", "mail_type": "ALL" },
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

> **Gotcha — `mail_user`:** SLURM's `%u`/`%j` filename patterns expand **only**
> in `--output`/`--error`/`--input`, *not* `--mail-user`. A `"mail_user":
> "%u@asu.edu"` is sent literally and bounces. Omit `mail_user` (with
> `mail_type` set, SLURM mails the submitting user by default) or bake a real
> address.

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

(The sbatch header itself carries `--export=NONE` — a config directive, not an
allocation — so the job starts in a **clean shell**; that is exactly why the
`module load mamba` preamble is load-bearing, § 1/§ 3.)

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

## 5. The CPU-vs-GPU benchmark compares hardware, not solvers

Both points use the **same** solver, `Diag.Algorithm ELPA-1STAGE`; only the
GPU point adds `Diag.ELPA.GPU .true.`, so the only variable is the CUDA toggle
and the measured difference is *hardware*, not *solver* (ScaLAPACK-CPU vs
ELPA-GPU would conflate the two). ELPA lives only in `molbuilder-siesta-gpu`
(the precompiled `molbuilder-siesta` has no ELPA — verified via `conda list`),
so **both points run there**: the GPU point auto-routes via its `Diag.ELPA.GPU`
flag, and the CPU point **declares that env explicitly** (visible as
`Target env` in its `.run.sh`). This is an explicit, visible setup — the
generator does *not* silently re-route the global env detection (assistant,
not nanny — `design.md` § Stance).

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
| `.molbuilder.json` documented in `--help` (path + example) | **built** (`bench generate -h` § CONFIG FILE) |
| **runtime** config-else-probe resolution (one bundle, both targets) | proposed (§ 2) |
| env-presence check → pointer to `molbuilder envs doctor` | proposed |
| CPU benchmark point on ELPA (apples-to-apples) | **built** (`transform_fdf` + explicit env, § 5) |
| `bench` `job-*.run.sh` self-activate (contract req 1) | **built** (`runwrap` bakes activation) |
| `run-transport.sh` self-activates (contract req 1) | **BUG / proposed** — today it requires a manual `conda activate` (§ 00) |
