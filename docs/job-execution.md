# Job execution — the system, the workflow, the contract

> **Scope — job execution, not deployment.** This document is the master
> source of truth for **using the script-generator *module*** to **submit
> and run calculations** on a target (a workstation, or a supercomputer's
> compute node): the self-running wrapper, the `prep` step, activation, the
> `script_generation` / `scheduler` config keys. **Deploying molbuilder
> itself** (serving the web app, auth, TLS — the `tls` / `auth` / `envs`
> keys) is the separate concern in [`deployment.md`](deployment.md). One
> `molbuilder.json`, two key-owners.

This file holds the **big picture**, the **end-to-end workflow**, and the
**detection / standalone contract**. The precise config schema, the
per-engine restart rules, the on-disk naming, the SLURM specifics, and the
benchmark workflow each have a dedicated sub-doc — see the map in § 5.

---

## 1. The big picture (plain language)

molbuilder writes a script that **knows how to run itself on a target
machine**. That self-running script — the **runwrap** (`.run.sh` for a
workstation, `.sbatch` for SLURM) — does three things on its own:

1. **turns on the right conda environment** (so the engine binary is found);
2. **runs the engine** (SIESTA / PySCF / TranSIESTA) with the parameters the
   generator baked in;
3. **handles restarts** — picks up where a prior run left off (warm), or
   starts clean and moves the old files aside instead of deleting them
   (cold). The per-engine rules are in
   [`protocols/script-execution.md`](protocols/script-execution.md).

This is a **general system**, not a benchmark feature. A benchmark rides on
it; a transport calculation rides on it; a plain production run rides on it.
Same machinery underneath.

Before anything runs there is a **prep** step — the **front door of the
workflow**. You copy a portable bundle onto a target you may know little
about, and `prep` looks at the machine (scheduler? how many sockets / cores
/ GPUs?), fills in the right numbers, and **checks the machine is actually
ready** before you spend a queue slot. The machine-specific knowledge lives
in the target-side step, **not in your head**.

**Assistant, not nanny.** molbuilder *assists* — it generates the harness
and surfaces hints to cut your burden; **you** own the recipe, the env, and
the decisions. Nothing here auto-decides your environment or twists your
input: it offers a *setup to test*, and prep **points at** readiness checks
rather than running or installing on your behalf.

---

## 2. The workflow, end to end

| step | where it runs | what it does |
|---|---|---|
| **generate** | host (laptop / login node, molbuilder installed) | writes the portable bundle: the self-running wrapper + inputs + pseudopotentials. Resolves and **bakes** the host-side decisions (§ 3.2). |
| *copy* | — | `scp` / `rsync` the bundle to the target. |
| **prep** | **target**, before anything runs | detects the machine (scheduler + topology), writes `environment.json`, **formats** the run for *this* machine, and **surfaces the readiness checks** (§ 3.5). No hand-editing of queue names or core counts. |
| **run** | compute node, inside the wrapper | the wrapper activates its own env and runs the engine; handles warm / cold restart itself. |
| **summarize** *(benchmark only)* | target | ranks the swept points → `bench-result.json` with a portable `choice`. |
| **prep-run** *(benchmark only)* | target | takes the portable `choice`, re-resolves the concrete knobs for the local machine → `run-production.sh`. |
| **production run** | compute node | the chosen configuration, submitted to result with zero manual steps. |

The benchmark workflow (generate → prep → run → summarize → prep-run) is
specified in full in
[`protocols/benchmark-workflow.md`](protocols/benchmark-workflow.md);
worked copy-paste recipes for every target are in § 4 below.

---

## 3. The detection / standalone contract

> Moved here from `config.md § 9` — this is job-execution *behavior*, while
> `config.md §§ 1–8` own the config/wrapper *schema*. The worked cookbook
> that applies this contract is § 4 of this file; the SLURM/HPC specifics
> are in [`protocols/slurm-integration.md`](protocols/slurm-integration.md).

"The wrapper does no detection" is too coarse to be a spec. There are
several distinct things one might call "detection", and the rule differs for
each. This section is the authoritative definition of *standalone*,
*detection*, the *assumption* each target makes, and the *goal*.

### 3.1 Definition

**Detection = reading external state and choosing behaviour from it without
the user stating that choice explicitly.** The external state is one of five
things, and the rule depends on *what* is read and *when*:

| | WHAT is read | example |
|---|---|---|
| **T** | conda/mamba **tool** | `$CONDA_EXE`, `which mamba`, `$CONDA_PREFIX` |
| **M** | HPC **toolchain** (modules) | whether `module load mamba` is needed |
| **C** | **config** file | `molbuilder.json` / `.molbuilder.json` |
| **A** | scheduler **allocation** | `SLURM_NTASKS`, `CUDA_VISIBLE_DEVICES`, `SLURM_CPUS_PER_TASK` |
| **H** | **hardware** topology | GPU PCI bus → NUMA node → socket (sysfs) |

### 3.2 The three moments

| moment | where it runs | does what |
|---|---|---|
| **generate** | the machine where you run `molbuilder bench generate` / `run` | resolves T/M/C and **bakes** them into `.molbuilder.json` + the `.run.sh` (verbatim) |
| **prep / doctor** | **on the target** (`./prep-bench`, `molbuilder envs doctor`) | detects the scheduler + topology to *format* the sweep, **and verifies readiness honestly (every target)** — § 3.5 |
| **runtime** | the compute node, inside `.run.sh` | everything baked; reads only A/H (the scheduler's published contract, `config.md § 1.5`) |

### 3.3 The rule matrix

| WHAT | GENERATE-TIME | RUNTIME (inside `.run.sh`) |
|---|---|---|
| **T — conda/mamba tool** | ✅ **allowed, narrow** — `runtime_config.detect_conda_activation` probes the command on PATH → activation form. *The only tool-autodetect there is.* | ❌ **forbidden** — no `which conda`, no `conda info --base`, no PATH search. Baked verbatim. |
| **M — HPC toolchain** | ⛔ **impossible → must be explicit** — on a clean login shell mamba isn't on PATH; nothing to detect. From config `preamble`. | ❌ **forbidden** — baked verbatim (`module load mamba`). |
| **C — config file** | ✅ **required** — generator reads `molbuilder.json` / `.molbuilder.json`. | ❌ **forbidden** — the wrapper never reads a config file at runtime. |
| **A — allocation** | n/a (no allocation yet) | ✅ **required** — `SLURM_NTASKS`→`-np`, `CUDA_VISIBLE_DEVICES`→GPU map, `SLURM_CPUS_PER_TASK`→OMP. |
| **H — topology** | n/a (target node unknown) | ✅ **required** — per-rank launcher reads sysfs for GPU→NUMA→socket binding. |

**Precise restatement of "standalone":** the generated wrapper does **no
runtime detection of tools, config, or toolchain (T/M/C)** — those are
decided once at generate time and baked, which is what makes it standalone.
It **does** read the **allocation and topology (A/H)** at runtime, because
those are what the scheduler hands the job and adapting to them is the whole
point. At **generate time** the *only* tool-autodetect is **conda/mamba on a
workstation**; the HPC toolchain is never guessed.

### 3.4 Two different "detection" jobs (don't conflate them)

- **Job A — autodetect the activation *method*** (infer an unstated choice:
  which activation form / preamble). **Workstation: yes** (toolchain is on
  PATH, discoverable). **HPC: no** — explicit config.
- **Job B — doctor: verify the *truth* of prerequisites** (confirm stated
  facts: env present, toolchain loads, scheduler/GPU/driver there). **Every
  target, always** — a prep-time check on whatever target it is invoked on.
  **Doctor is prep-time, not the `.run.sh`** — the wrapper stays baked; if
  the env is missing at *run* time the wrapper's `set -euo pipefail` aborts
  loud anyway (`config.md § 1`).

  **The mechanism is the EXISTING `molbuilder envs` toolkit — do NOT build a
  new readiness/doctor checker** (it has been re-derived and rejected
  repeatedly):
  - **`molbuilder envs doctor`** — present/missing per recipe **+ runs each
    recipe's verify command** (invokes the engine binary in the env).
  - **`molbuilder envs validate <env>`** — post-install correctness probes;
    for `molbuilder-siesta-gpu`: binary links, **CUDA stack (`nvidia-smi` +
    `libcuda.so.1`)**, siesta ctest, and the load-bearing
    **ELPA-GPU-codepath probe** that catches ELPA silently falling back to
    CPU (slurm-integration.md § 7.9 driver floor + § 11.1 GPU correctness
    gate are both covered here).

  These run in the **host molbuilder env** (which *is* installed on the
  target — § 3.5), so they are not bound by the bundle's stdlib-only
  self-contained rule. Assistant-not-nanny: prep **surfaces / points at**
  these commands; the scientist runs them — molbuilder does not auto-install
  or auto-decide.

### 3.5 Per-target activation defaults + assumption

| target | activation form | required prerequisite (baked) | who supplies it |
|---|---|---|---|
| **workstation** | `conda activate` | `source "<base>/etc/profile.d/conda.sh"` (the conda hook — a non-interactive `bash job.run.sh` does **not** read `~/.bashrc`, so the `conda` function must be sourced) | **autodetected** + baked (`detect_conda_activation`) |
| **HPC** | `source activate` | `module load mamba` (puts the legacy `activate` shim on PATH) | **explicit** config / `asu-sol` preset |

Each form carries its own prerequisite; that is why the workstation default
(`conda activate`) *requires* baking the hook-source line — it is
load-bearing for standalone, not overreach. Override either with
`--activation` / `--preamble` (the explicit hatch).

- **The assumption that flips behaviour:** *is the machine that generates the
  script the same one that runs it, with conda already on PATH?* **Yes** →
  workstation → generate-time tool-autodetect is valid. **No / clean HPC
  shell** → nothing about the target is detectable → activation + preamble
  must be explicit config.
- **Env creation is always the user's** (workstation *and* HPC). Doctor
  verifies presence and stops with a pointer to `molbuilder envs doctor`; it
  never runs `envs install`.

### 3.6 The goal

A `.run.sh` / `.sbatch` that takes the job from **submit to result with zero
manual steps on the target** — every T/M/C decision resolved and baked at
generate time, every A/H decision adapted at runtime from what the scheduler
actually granted. Submit headless (`sbatch`), log out, collect the result.

---

## 4. Examples & templates (cookbook)

Copy-paste worked examples for submitting + running the generated workflows
(benchmark, transport) on **a workstation** and on **a supercomputer**, with
`.molbuilder.json` templates to study.

> **This is a setup to test, not a push-button.** molbuilder *assists* — it
> generates the harness and surfaces hints to cut your burden; **you** own
> the recipe, the env, and the decisions. Read the example, adapt it, run it.

> **Prerequisite for every example:** the envs are already prepared
> (Phase A — [`README_install.md`](README_install.md)) and the backend env
> (`molbuilder-siesta-gpu`) shows up in `conda env list`. If not, `prep` /
> doctor tells you (§ 3.4) — see § 4.1.4 / § 4.2.5.

> **One bundle per target (today).** Activation is resolved and baked at
> **generate time** (§ 3.2), so you generate a bundle *for the machine it
> will run on*. A workstation bundle bakes `conda activate`; a Sol bundle
> bakes `module load mamba` + `source activate`. (A future runtime-resolution
> mode — tasks #24/#25 — would make one bundle portable; § 4.3 and § 6.)

### 4.1 Workstation

Full control; conda/mamba on PATH; generate == run machine; jobs run via
`bash …run.sh`.

#### 4.1.1 Happy path — conda on PATH, env ready

```bash
molbuilder bench generate input.fdf --out bench-ws/   # activation auto-detected
cd bench-ws/
./prep-bench --gpu-ks 1,2,4,8       # detect THIS machine -> environment.json + sweep
bash job-cpu.run.sh                 # CPU point (ELPA in molbuilder-siesta-gpu)
bash job-gpu-sweep.sh               # GPU sweep (sequential)
./bench-summarize                   # rank CPU vs GPU -> winner
./prep-run --script-base myprod     # winner -> run-production.sh
```
No `.molbuilder.json` needed — `generate` autodetects your local conda
(Job A, § 3.4) and bakes **`conda activate`** plus the conda-hook source line
(`source "<base>/etc/profile.d/conda.sh"`, load-bearing for a
non-interactive script — § 3.5) into the wrappers.

#### 4.1.2 `mamba` instead of `conda`

Same as § 4.1.1 — autodetection probes `$MAMBA_EXE` / `mamba` too. To pin the
form, drop a workstation config (§ 4.4.1) or pass
`--activation "conda activate"`.

#### 4.1.3 GPU box vs CPU-only box

`prep-bench` detects GPUs. On a **1-GPU** box, scale *processes* with
`--gpu-ks` (K = ranks sharing the GPU), e.g. `--gpu-ks 1,2,4,8`
(oversubscription past cores/socket is allowed + flagged). On a **no-GPU**
box, run only `job-cpu.run.sh`. **VRAM caveat:** a large Au system in
ELPA-CUDA can OOM a consumer card (e.g. a 12 GB RTX 3060) — bench a smaller
structure if it won't fit.

#### 4.1.4 Env not prepared → what you see + the fix

`prep` / doctor reports a missing or unactivatable env and **stops with a
pointer** (it never builds the env — § 3.5). Fix it in Phase A:
```bash
molbuilder envs doctor                          # what's missing / broken
molbuilder envs install molbuilder-siesta-gpu   # or: molbuilder envs bootstrap
```

#### 4.1.5 Non-standard conda / custom activation

If autodetect can't find your conda, or you want a specific form, pass it
explicitly (written into the bundle's `.molbuilder.json`):
```bash
molbuilder bench generate input.fdf --out bench-ws/ \
    --activation "conda activate" --preamble 'source "$HOME/miniconda3/etc/profile.d/conda.sh"'
```

### 4.2 Supercomputer (SLURM, e.g. ASU Sol)

A *constrained* environment: the job lands in a **clean shell**
(`--export=NONE`), so `mamba`/`conda` is **not** on PATH until
`module load mamba` runs — that step **must** come from config/flags (it
can't be autodetected; § 3.3, row M). Jobs run via `sbatch`.

#### 4.2.1 Happy path — Sol

```bash
# generate WITH the activation Sol will use (or ship the config of § 4.4.2):
molbuilder bench generate input.fdf --out bench-sol/ \
    --activation "source activate" --preamble "module load mamba"
# copy bench-sol/ to Sol, then ON Sol's login node:
cd bench-sol/
./prep-bench --gpu-ks 8,16       # detects SLURM + A100 topology + verifies readiness
sbatch job-cpu.sbatch            # CPU point
bash job-gpu-sweep.sh            # GPU sweep (sbatch per point)
./bench-summarize
```

#### 4.2.2 Clean shell — why the preamble is load-bearing

Under `--export=NONE` the job starts with no `mamba`/`conda`. The wrapper
runs the baked `module load mamba` **before** `source activate <env>`;
without the preamble the activation fails immediately. This is exactly why a
Sol bundle **must** carry `script_generation` (§ 4.4.2) — there is nothing to
autodetect (§ 3.5).

#### 4.2.3 Doctor runs on the cluster too

`prep` on Sol's login node *runs the explicit `module load mamba`* to reach
the truth, then verifies the backend env (`mamba env list`), activation, the
scheduler, and the GPU driver ≥ the CUDA floor (slurm-integration.md § 7.9) —
§ 3.4. It tells you honestly whether the next `sbatch` will work before you
queue it.

#### 4.2.4 CPU vs GPU allocation

- **CPU:** `sbatch -n 64 job-cpu.sbatch` — `-n` → `SLURM_NTASKS` →
  `mpirun -np`.
- **GPU:** `--gres=gpu:a100:G` → SLURM sets `CUDA_VISIBLE_DEVICES`; the
  per-rank launcher gives each rank its GPU (+ MPS for K≥2). Run GPU
  `--exclusive` for clean timing.
- **CPU may time out:** plain `diagon` on a few-hundred-atom system can
  exceed the 4 h default before the capped iters finish — set `--cpu-time` at
  generate or `sbatch -t …`.

Both points run the **same ELPA-1STAGE solver in `molbuilder-siesta-gpu`** —
the comparison is hardware, not solver (slurm-integration.md § 11).

#### 4.2.5 Env not prepared on the cluster → error

Same as § 4.1.4, on Sol: `molbuilder envs doctor` then
`molbuilder envs install …`. The run scripts never create the env.

#### 4.2.6 A different cluster / module system

Autodetect doesn't apply on HPC. Provide that site's load+activate via flags
or config (§ 4.4.3): e.g.
`--preamble "module load anaconda3" --activation "conda activate"`, or
whatever the site documents.

### 4.3 Cross-machine (generate here, run there)

The common real case: molbuilder on your workstation, the job runs on Sol.
**Today:** generate a *Sol-configured* bundle (§ 4.2.1 — the activation is the
*target's*, not your workstation's), copy it, run. The workstation's conda is
irrelevant to that bundle.

**Future (tasks #24/#25):** runtime resolution would make one bundle portable
— the *same* `bench-*/` runs on the workstation (autodetect) and on Sol
(reads its `.molbuilder.json`), no per-target regeneration. Until then,
§ 4.1.1 and § 4.2.1 are separate generates. See § 6.

### 4.4 Templates — `.molbuilder.json` (copy, edit, drop in the OUT dir)

Lives in the bundle's OUT dir, merged over a server-wide
`~/.config/molbuilder/molbuilder.json` (project wins). Lookup + merge rules:
`config.md § 3`; schema: `config.md § 4`.

#### 4.4.1 Workstation (usually none needed; this only pins the activation)

```json
{ "script_generation": { "activation": "conda activate" } }
```

#### 4.4.2 ASU Sol (SLURM + GPU)

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
> Do **not** set `"mail_user": "%u@asu.edu"` — `%u` does NOT expand in
> `--mail-user` (only in `--output`/`--error`); it bounces. Omit it (SLURM
> mails the submitter) or use a literal address.

#### 4.4.3 Generic HPC (adapt the partition + module to your site)

```json
{
  "scheduler": {
    "kind": "slurm",
    "directives": { "partition": "<your-partition>", "export": "NONE" },
    "defaults": { "time": "0-04:00:00", "cpus_per_task": 8, "mem": null }
  },
  "script_generation": {
    "preamble": "<your module load line(s)>",
    "activation": "source activate"
  }
}
```

### 4.5 Quick decision guide

| situation | what to do |
|---|---|
| my workstation, conda on PATH | `bench generate … --out X` (no config) → `bash *.run.sh` |
| my workstation, weird conda | add `--activation`/`--preamble` (§ 4.1.5) or § 4.4.1 |
| SLURM cluster | `--activation "source activate" --preamble "module load mamba"` (or § 4.4.2/4.4.3) → `sbatch` |
| generate here, run on HPC | configure for the **target**, copy, run (§ 4.3) |
| env missing anywhere | `molbuilder envs doctor` → `… install` (Phase A) |
| GPU job won't fit | fewer ranks/GPU (`--gpu-ks`), smaller structure, or CPU-only |

Transport is the same shape: `transport bundle …` for the scripts,
`slurm-integration.md` for sbatch, this cookbook's templates for the env.

---

## 5. Where each detail lives (the map)

This file is the front door. Each specialized contract has one sub-doc that
owns it:

| Concern | Owner doc |
|---|---|
| **Config schema + wrapper contract** (`preamble`, `activation`, `scheduler`, refuse-to-emit, the generator contract) | [`config.md`](config.md) §§ 1–8 |
| **Self-running wrapper** — warm/cold restart inventory per engine, `--continue` / `--cold`, project-id consistency | [`protocols/script-execution.md`](protocols/script-execution.md) |
| **On-disk layout** — basename + `*-runN.out` convention | [`protocols/job-layout.md`](protocols/job-layout.md) |
| **SLURM specifics** — sbatch header line-by-line, Sol facts, CUDA driver floor, GPU correctness gate | [`protocols/slurm-integration.md`](protocols/slurm-integration.md) |
| **Benchmark workflow** — generate → prep → run → summarize → prep-run, probes/adapters, data formats | [`protocols/benchmark-workflow.md`](protocols/benchmark-workflow.md) |
| **Env install (Phase A)** — the four-env model, building the backends | [`README_install.md`](README_install.md) |
| **Deploying the app** (NOT job execution) | [`deployment.md`](deployment.md) |

---

## 6. Roadmap — open work

1. **Readiness gate at prep (#25).** `prep` should **surface** the
   readiness checks (`molbuilder envs doctor` + `molbuilder envs validate
   <gpu-env>`, § 3.4) in its output so the scientist runs them before
   submitting. Surface/point only — reuse the existing `envs` toolkit, do
   **not** build a new checker; do **not** auto-run or auto-install
   (assistant-not-nanny).
2. **One portable bundle (#24/#25).** Lift the "one bundle per target"
   limitation (§ 4.3): a runtime-resolution mode so the *same* bundle
   autodetects on a workstation **and** reads `.molbuilder.json` on an HPC
   node — no per-target regeneration. This relaxes the current "C forbidden
   at runtime" rule (§ 3.3, row C) for the activation decision only, and
   belongs in the `prep` step (the on-target detection step that already
   exists).
