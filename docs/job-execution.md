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
2. **runs the engine** (SIESTA / PySCF) with the parameters the
   generator baked in;
3. **handles restarts** — picks up where a prior run left off (warm), or
   starts clean and moves the old files aside instead of deleting them
   (cold). The per-engine rules are in
   [`protocols/script-execution.md`](protocols/script-execution.md).

This is a **general system**, not a benchmark feature. It is engine- and
calculation-agnostic: it runs *whatever* script was prepared for it — a
benchmark, a production run — regardless of which module produced the input.
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
| **generate** | host (laptop / login node, molbuilder installed) | writes the **portable** bundle: engine/calculation inputs + pseudopotentials + the prep driver. Bakes the *calculation* decisions (the `.fdf`); it does **not** bake a target-locked wrapper or resolve activation — the target is still unknown (§ 3.2). |
| *copy* | — | `scp` / `rsync` the bundle to the target. |
| **prep** | **target**, before anything runs (molbuilder installed — § 3.4) | detects the machine (scheduler + topology), writes `environment.json`, **validates** the env install + activation and **surfaces the readiness checks** (§ 3.5), then **resolves activation for *this* machine and generates the self-running wrapper** (§ 3.2). No hand-editing of queue names, core counts, or activation. |
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
| **generate** | the machine where you run `molbuilder bench generate` / `run` | bakes the *calculation* into the inputs (the `.fdf`); ships the prep driver. Does **not** resolve T/M/C or bake a wrapper (target unknown). May carry an explicit `.molbuilder.json` for an HPC target. |
| **prep / doctor** | **on the target** (`molbuilder bench prep`, `molbuilder envs doctor`; molbuilder installed — § 3.4) | detects the scheduler + topology, **verifies readiness honestly (every target)** (§ 3.5), then **resolves T/M/C and bakes them into `.molbuilder.json` + the `.run.sh` (verbatim) for *this* target** |
| **runtime** | the compute node, inside `.run.sh` | everything baked at prep; reads only A/H (the scheduler's published contract, `config.md § 1.5`) |

### 3.3 The rule matrix

| WHAT | RESOLVE-TIME (`prep`, on the target) | RUNTIME (inside `.run.sh`) |
|---|---|---|
| **T — conda/mamba tool** | ✅ **allowed, narrow** — `runtime_config.detect_conda_activation` probes the command on PATH *on the target* → activation form. *The only tool-autodetect there is.* | ❌ **forbidden** — no `which conda`, no `conda info --base`, no PATH search. Baked verbatim. |
| **M — HPC toolchain** | ⛔ **impossible → must be explicit** — on a clean login shell mamba isn't on PATH; nothing to detect. From config `preamble`. | ❌ **forbidden** — baked verbatim (`module load mamba`). |
| **C — config file** | ✅ **required** — `prep` reads `molbuilder.json` / `.molbuilder.json` *on the target*. | ❌ **forbidden** — the wrapper never reads a config file at runtime. |
| **A — allocation** | n/a (no allocation yet) | ✅ **required** — `SLURM_NTASKS`→`-np`, `CUDA_VISIBLE_DEVICES`→GPU map, `SLURM_CPUS_PER_TASK`→OMP. |
| **H — topology** | n/a (target node unknown) | ✅ **required** — per-rank launcher reads sysfs for GPU→NUMA→socket binding. |

**Precise restatement of "standalone":** the generated wrapper does **no
runtime detection of tools, config, or toolchain (T/M/C)** — those are
decided once **at prep time, on the target**, and baked, which is what makes
it standalone (a single self-contained file runnable from a clean bash). It
**does** read the **allocation and topology (A/H)** at runtime, because those
are what the scheduler hands the job and adapting to them is the whole point.
At **prep time** the *only* tool-autodetect is **conda/mamba on a
workstation**; the HPC toolchain is never guessed (explicit `preamble`).

> **What changed (2026-06-28, tasks #24/#25).** T/M/C resolution + wrapper
> baking **moved from generate-time (host) to prep-time (target)**, so ONE
> portable bundle now runs on a workstation *and* an HPC node — `prep`
> specialises it per machine. The standalone guarantee is **unchanged** (the
> wrapper is still fully baked, no runtime T/M/C reads — row C runtime rule
> intact); only the *moment* of baking moved. This relies on the contract
> that **molbuilder is installed on every target** (§ 3.4).

### 3.4 Two different "detection" jobs (don't conflate them)

> **Contract: molbuilder is installed on every target** — workstation *and*
> HPC compute environment — and the backend env (`molbuilder-siesta-gpu`)
> exists. This is a prerequisite, not something the toolbox engineers around.
> It is exactly what lets `prep` do its job on the target: validate the
> install + activation, then *generate the standalone wrapper there* (§ 3.2)
> using the same `molbuilder` machinery the host uses. (Phase A —
> [`README_install.md`](README_install.md) — prepares these envs; `prep` /
> `doctor` verify them, never create them.)

- **Job A — autodetect the activation *method*** (infer an unstated choice:
  which activation form / preamble), **at prep, on the target**. **Workstation:
  yes** (toolchain is on PATH, discoverable). **HPC: no** — explicit config.
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

  These run in the **molbuilder env installed on the target** (the contract
  above). Assistant-not-nanny: prep **surfaces / points at** these commands;
  the scientist runs them — molbuilder does not auto-install or auto-decide.

### 3.5 Per-target activation defaults + assumption

| target | activation form | required prerequisite (baked) | who supplies it |
|---|---|---|---|
| **workstation** | `conda activate` | `source "<base>/etc/profile.d/conda.sh"` (the conda hook — a non-interactive `bash job.run.sh` does **not** read `~/.bashrc`, so the `conda` function must be sourced) | **autodetected** + baked (`detect_conda_activation`) |
| **HPC** | `source activate` | `module load mamba` (puts the legacy `activate` shim on PATH) | **explicit** config / `asu-sol` preset |

Each form carries its own prerequisite; that is why the workstation default
(`conda activate`) *requires* baking the hook-source line — it is
load-bearing for standalone, not overreach. Override either with
`--activation` / `--preamble` (the explicit hatch).

- **The assumption that flips behaviour:** *on the target where `prep` runs,
  is conda/mamba already on PATH (a workstation login shell)?* **Yes** →
  workstation → prep-time tool-autodetect is valid. **No / clean HPC
  shell** → nothing about the target is detectable → activation + preamble
  must be explicit config (shipped in the bundle's `.molbuilder.json`).
  Because resolution now happens at **prep on the target** (§ 3.2), one
  portable bundle covers both — the host that generated it is irrelevant.
- **Env creation is always the user's** (workstation *and* HPC). Doctor
  verifies presence and stops with a pointer to `molbuilder envs doctor`; it
  never runs `envs install`.

### 3.6 The goal

A `.run.sh` / `.sbatch` that takes the job from **submit to result with zero
manual steps on the target** — every T/M/C decision resolved and baked at
**prep time, on the target**, every A/H decision adapted at runtime from what
the scheduler actually granted. Submit headless (`sbatch`), log out, collect
the result.

---

## 4. Examples & templates (cookbook)

Copy-paste worked examples for submitting + running a generated job
(benchmark / production) on **a workstation** and on **a supercomputer**, with
`.molbuilder.json` templates to study.

> **This is a setup to test, not a push-button.** molbuilder *assists* — it
> generates the harness and surfaces hints to cut your burden; **you** own
> the recipe, the env, and the decisions. Read the example, adapt it, run it.

> **Prerequisite for every example:** the envs are already prepared
> (Phase A — [`README_install.md`](README_install.md)) and the backend env
> (`molbuilder-siesta-gpu`) shows up in `conda env list`. If not, `prep` /
> doctor tells you (§ 3.4) — see § 4.1.4 / § 4.2.5.

> **One portable bundle (tasks #24/#25).** Activation is resolved and the
> wrapper baked at **prep time, on the target** (§ 3.2) — so the *same*
> bundle runs on a workstation (prep autodetects `conda activate`) **and** on
> an HPC node (prep reads the shipped `.molbuilder.json` → `module load mamba`
> + `source activate`). Generate once; `prep` specialises per machine. No
> per-target regeneration. (For an HPC target, ship the explicit activation
> in the bundle's `.molbuilder.json` — § 4.4.2 — since the HPC toolchain is
> not autodetectable; § 3.3 row M.)

### 4.1 Workstation

Full control; conda/mamba on PATH; generate == run machine; jobs run via
`bash …run.sh`.

#### 4.1.1 Happy path — conda on PATH, env ready

```bash
molbuilder bench generate input.fdf --out bench-ws/   # target-neutral bundle
cd bench-ws/
molbuilder bench prep --gpu-ks 1,2,4,8   # detect machine + autodetect conda +
                                         #   BAKE job-{cpu,gpu}.run.sh + sweep
bash job-cpu.run.sh                 # CPU point (ELPA in molbuilder-siesta-gpu)
bash job-gpu-sweep.sh               # GPU sweep (sequential)
./bench-summarize                   # rank CPU vs GPU -> winner
./prep-run --script-base myprod     # winner -> run-production.sh
```
No `.molbuilder.json` needed — **`prep`** autodetects your local conda
(Job A, § 3.4) and bakes **`conda activate`** plus the conda-hook source line
(`source "<base>/etc/profile.d/conda.sh"`, load-bearing for a
non-interactive script — § 3.5) into the wrappers. (`generate` no longer bakes
the wrappers — the bundle is portable; § 7.)

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
# generate a portable bundle; ship Sol's activation in it (clean shell on Sol
# can't autodetect it — § 3.3 row M).  Either pass the flags (persisted into
# .molbuilder.json) or drop the config of § 4.4.2 into the out dir:
molbuilder bench generate input.fdf --out bench-sol/ \
    --activation "source activate" --preamble "module load mamba"
# copy bench-sol/ to Sol, then ON Sol's login node (molbuilder installed):
cd bench-sol/
molbuilder bench prep --gpu-ks 8,16   # detect SLURM + A100 + readiness, and
                                      #   BAKE job-{cpu,gpu}.sbatch from the
                                      #   shipped activation + detected topology
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
The **same portable bundle** covers both (tasks #24/#25): activation is
resolved and the wrapper baked at **prep time, on the target** (§ 3.2), so —

1. `molbuilder bench generate …` once on your workstation → a portable bundle
   (for a Sol run, ship the explicit activation in `.molbuilder.json`, § 4.4.2,
   since the HPC toolchain isn't autodetectable — § 3.3 row M);
2. `scp` it to Sol;
3. `molbuilder bench prep` **on Sol** validates the env + activation and bakes
   the Sol-specific `.run.sh` / `.sbatch`;
4. `sbatch` and collect.

The same bundle prepped on your workstation instead would autodetect
`conda activate` and bake a `bash …run.sh` launcher — no regeneration. The
generating host's conda is irrelevant; `prep` resolves against the *target*.

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

1. **Readiness gate at prep (#25). — DONE.** `prep` now **surfaces** the
   readiness checks in its summary (`bench/prep.py::_readiness_lines`):
   `molbuilder envs doctor` always, plus `molbuilder envs validate
   molbuilder-siesta-gpu` when GPUs are detected (§ 3.4). Surface/point only
   — it reuses the existing `envs` toolkit, builds no new checker, and never
   auto-runs or auto-installs (assistant-not-nanny).
2. **One portable bundle (#24/#25). — DONE 2026-06-28** (design + code; see
   § 7). Lifted the "one bundle per target" limitation (§ 4.3) by moving
   T/M/C resolution **and wrapper baking** from generate-time (host) to
   prep-time (target). `generate` emits a portable bundle (calculation inputs
   + prep driver, no target-locked wrapper); `prep`, **using the molbuilder
   install that the contract guarantees on every target (§ 3.4)**, validates
   the env + activation, resolves activation (autodetect workstation conda /
   read `.molbuilder.json` on HPC), and bakes the standalone `.run.sh` /
   `.sbatch` for *this* machine.

   This **reuses the existing framework verbatim** — it relocates the
   `bench/generate.py::_ensure_activation` + `runwrap.write_run_wrapper` calls
   to the `prep` step; no new resolver.

   - **Standalone is preserved, NOT relaxed.** The wrapper is still fully
     baked and self-contained (a single file, clean-bash runnable); it does
     **no** runtime T/M/C reads. The earlier framing — "relax the row-C
     forbidden-at-runtime rule" — was **wrong**: config is read at *prep*
     (already allowed, § 3.3), not at runtime. Row C's runtime rule stays
     intact.
   - **Implementation surface:** `bench/generate.py` (stop baking the
     wrapper / hard-requiring activation; ship portable), `bench/prep.py` +
     the `molbuilder bench prep` CLI (resolve activation + call
     `write_run_wrapper` on the target), and reconcile the now-redundant
     "target needs no molbuilder install" property the stdlib-only prep-lib
     was built around (the contract says molbuilder *is* on the target).
   - **Tests:** the same bundle prepped on a workstation (autodetect) vs an
     HPC-config target (reads `.molbuilder.json`) bakes the correct activation
     with **no regeneration**.
   - Reconcile the § 4.1 / § 4.2 cookbook recipes to the prep-time flow in the
     same change.

---

## 7. Implementation design — one portable bundle (#24/#25)

> **Status: IMPLEMENTED (2026-06-28).** This section is the contract the
> framework honors: data flow, on-disk information structures, file
> ownership, pseudocode for each moved/added step, corner cases preserved,
> test plan. The code (`bench/generate.py`, `bench/_cli.py`,
> `tests/test_bench_{generate,prep}.py`) matches it; `§ 7.7` records the
> resolved decisions.

### 7.1 The one idea

The *only* thing that locks a bundle to a target is the **activation baked
into the run wrappers** (`job-cpu.run.sh` / `job-gpu.run.sh`). Move that
single decision — and therefore the wrapper baking that consumes it — from
**generate (host)** to **prep (target)**. Everything else already happens at
prep (topology detection, the sweep). The generated bundle becomes
**target-neutral**; `prep` specialises it on the machine it will run on.

This **reuses existing functions verbatim** — no new resolver, no new wrapper
renderer:
- `runtime_config.detect_conda_activation()` — workstation autodetect (T).
- `runtime_config.get_script_generation()` / `require_activation()` — read the
  shipped config (C) and enforce "no activation ⇒ refuse".
- `runwrap.write_run_wrapper()` — bake the standalone `.run.sh` / `.sbatch`
  (all its constrained-HPC handling: clean-shell `set +u` bootstrap, conda
  hook, MPS, socket-pin, mem estimate, warm/cold restart — untouched).

### 7.2 Data flow & file ownership

```
HOST (molbuilder installed)                     TARGET (molbuilder installed — § 3.4)
────────────────────────────                    ─────────────────────────────────────
molbuilder bench generate                       molbuilder bench prep
  reads:  input.fdf, *.psml,                      reads:  bench-manifest.json
          [.molbuilder.json if --activation]              .molbuilder.json (if shipped)
  writes: job-cpu.fdf  job-gpu.fdf                        input *.fdf, *.psml
          *.psml (copied)                         detects: scheduler + topology
          bench-manifest.json   ◄── NEW          writes:  environment.json
          job-gpu-sweep.sh (placeholder)                  .molbuilder.json (resolved activation)
          README.md                                       job-cpu.run.sh  job-cpu.sbatch   ◄── moved here
          mbbench/ + prep-bench/…(stdlib lib)              job-gpu.run.sh  job-gpu.sbatch   ◄── moved here
          .molbuilder.json (ONLY if explicit               mb_monitor.py (shipped by write_run_wrapper)
            --activation/--preamble given)                 job-gpu-sweep.sh (real, topology-sized)
  does NOT: autodetect host conda,
            bake any .run.sh                     →  bash job-cpu.run.sh / sbatch job-cpu.sbatch / ./job-gpu-sweep.sh
                                                 →  ./bench-summarize ; ./prep-run
        ── scp/rsync bundle ──►
```

**File ownership (who writes each):**

| file | written by | moment | notes |
|---|---|---|---|
| `job-cpu.fdf`, `job-gpu.fdf` | generate | host | the calculation (baked at generate — unchanged) |
| `*.psml/.psf/.vps` | generate | host | copied beside the fdfs (unchanged) |
| `bench-manifest.json` | generate | host | **NEW** — the benchmark knobs (§ 7.3) |
| `README.md` | generate | host | updated to the prep-bakes-wrappers flow |
| `mbbench/`, `prep-bench`, `bench-summarize`, `prep-run` | generate | host | stdlib prep-lib (unchanged) |
| `.molbuilder.json` | generate **(only if** `--activation`/`--preamble`**)**, else prep | host or target | merged through one helper; `script_generation` is the activation record |
| `environment.json` | prep | target | detected scheduler + topology (unchanged) |
| `job-{cpu,gpu}.run.sh` / `.sbatch` | **prep** | **target** | **moved** from generate — baked with the target's activation |
| `mb_monitor.py` | prep | target | shipped by `write_run_wrapper` (follows the wrapper) |
| `job-gpu-sweep.sh` | generate (placeholder) → prep (real) | both | unchanged |

### 7.3 Information structures

**`bench-manifest.json`** — *new* — the only generate→prep carrier of the
user's benchmark knobs (chosen via `bench generate` flags). Derived numbers
(`gres`, `-c`) are NOT stored; prep computes them from the *detected* topology.

```json
{
  "schema": "molbuilder/bench-manifest@1",
  "engine": "siesta",
  "jobs": {
    "cpu": { "script": "job-cpu.fdf", "mpi_np": 64,
             "cpus_per_task": 1, "time": null },
    "gpu": { "script": "job-gpu.fdf", "gpu_gpus": 1, "gpu_k": 4,
             "time": null, "exclusive": null }
  }
}
```

> **Engine-neutral by design (multi-engine readiness).** `engine` +
> per-job `script` (not `fdf`) keep the schema engine-agnostic: the
> job-execution layer (prep/bake/runwrap) dispatches by file *extension*
> (`.fdf`→siesta, `.py`→pyscf), so a future PySCF bench reuses this exact
> schema with `engine: "pyscf"` + `script: "job-cpu.py"` — no special
> casing. The science-specific part (how the inputs are produced —
> `transform_fdf` for SIESTA) stays in the engine's own module; only the
> *generic* run knobs live here.

| field | source (generate flag) | consumed at prep as |
|---|---|---|
| `cpu.mpi_np` | `--cpu-np` | `write_run_wrapper(mpi_np=…)` |
| `cpu.cpus_per_task` | `--cpu-c` | `write_run_wrapper(cpus_per_task=…)` |
| `cpu.time` / `gpu.time` | `--cpu-time` / `--gpu-time` | `write_run_wrapper(time=…)` |
| `gpu.gpu_gpus` | `--gpu-gpus` | ranks `= gpu_k*gpu_gpus`; `gres="<type>:gpu_gpus"` |
| `gpu.gpu_k` | `--gpu-k` | ranks + `cpus_per_task = cores//gpu_k` |
| `gpu.exclusive` | `--gpu-exclusive` | `write_run_wrapper(exclusive=…)` |
| `gres` GPU type | — (NOT stored) | **detected** `environment.topology.gpu_type` (was hardcoded `a100` at generate) |
| `cpus_per_task` (GPU) | — (NOT stored) | **detected** `environment.topology.cores_per_socket // gpu_k` (was a generate guess) |

**`.molbuilder.json` → `script_generation`** — the activation record
(schema owned by `config.md §§ 1–8`). One writer helper
(`_write_activation_config`) so generate-explicit and prep-resolved produce a
byte-identical shape:

```json
{ "script_generation": { "activation": "conda activate" | "source activate",
                         "preamble": "source \"<base>/etc/profile.d/conda.sh\"" | "module load mamba" } }
```

**`environment.json`** — detected scheduler + topology; schema unchanged
(`molbuilder/environment@1`), owned by `benchmark-workflow.md § 5.2`.

### 7.4 Pseudocode

**generate** — `bench/generate.py::generate_bench_bundle` (revised):

```text
generate_bench_bundle(fdf, out, *, cpu_np, gpu_gpus, gpu_k, …, activation, preamble):
    write job-cpu.fdf, job-gpu.fdf            # transform_fdf — unchanged
    copy pseudopotentials                     # unchanged
    if activation or preamble:                # explicit target config (HPC, § 4.4.2)
        _write_activation_config(out, activation, preamble)   # shared helper
    # NO host-conda autodetect, NO write_run_wrapper here.
    write bench-manifest.json                 # the knobs above (§ 7.3)
    write job-gpu-sweep.sh placeholder        # unchanged
    write README.md (prep-bakes-wrappers flow)
    ship mbbench/ + shims (stdlib)            # unchanged
    return out, written
```

Generate **no longer** resolves/raises on activation (the target is unknown);
the refuse-to-emit moves to prep, where the activation truly must exist.

**prep** — `bench/prep.py::run_prep_bench` is unchanged (stdlib: detect +
`environment.json` + sweep). The molbuilder-side baking is a **new step in
`bench/_cli.py::cmd_prep`**, factored as `generate.py::bake_target_wrappers`:

```text
cmd_prep(out, …overrides…):
    env, written = run_prep_bench(out, …)     # stdlib core — unchanged
    written += bake_target_wrappers(out, env) # NEW, molbuilder-side
    echo(summary)                             # incl. § 3.4 readiness pointer (task #2)

bake_target_wrappers(out, env):
    manifest = read_json(out/"bench-manifest.json")    # error if absent → "run generate first"
    # ── resolve activation FOR THIS TARGET (the detected scheduler gates
    #    BOTH branches — NOT _ensure_activation's config>autodetect order,
    #    which would bake a shipped HPC activation on a workstation) ──
    if env.scheduler == "workstation":
        det = detect_conda_activation()                 # T: login shell has conda
        if not det:
            raise RuntimeConfigError(                   # workstation w/o conda on PATH
                "activate conda on this workstation, or ship explicit "
                "script_generation in .molbuilder.json")
        _write_activation_config(out, det.activation, det.preamble)   # specialise THIS dir
    else:                                               # slurm / HPC
        sg = get_script_generation(project_dir=out)
        if not sg.activation:                           # clean job shell, nothing to autodetect (§ 3.3 row M)
            raise RuntimeConfigError(
                "ship .molbuilder.json with script_generation (§ 4.4.2)")
        # shipped config is already where write_run_wrapper reads it — leave as-is
    # ── bake the standalone wrappers (reuses write_run_wrapper) ──
    elpa_env = capabilities.env_for_category("siesta-gpu")
    # .sbatch is gated on the DETECTED scheduler, NOT the shipped config block
    # -- else a bundle carrying an HPC scheduler block, prepped on a
    # workstation, would emit stray SLURM .sbatch files (§ 7.5 #6).
    emit_sbatch = (env.scheduler == "slurm")
    cpu = manifest.jobs.cpu
    write_run_wrapper(out/cpu.script, env=elpa_env, emit_sbatch=emit_sbatch,
                      mpi_np=cpu.mpi_np, cpus_per_task=cpu.cpus_per_task, time=cpu.time)
    gpu = manifest.jobs.gpu
    cores = env.topology.cores_per_socket or 24
    gtype = env.topology.gpu_type or "a100"
    write_run_wrapper(out/gpu.script, env=elpa_env, emit_sbatch=emit_sbatch,
                      mpi_np=gpu.gpu_k*gpu.gpu_gpus,
                      gres=f"{gtype}:{gpu.gpu_gpus}",
                      cpus_per_task=max(1, cores//gpu.gpu_k),
                      time=gpu.time, exclusive=gpu.exclusive)
    return [the .run.sh/.sbatch/mb_monitor.py paths]
```

`write_run_wrapper` already reads `require_activation(project_dir=out)`
internally, so the `_ensure_activation` write immediately precedes it — the
wrapper comes out fully baked and standalone (no runtime config read).

### 7.5 Corner cases the framework must keep honoring

1. **Stdlib `./prep-bench` with no molbuilder** (`test_shipped_prep_lib_runs_
   with_no_molbuilder`): still does detection + sweep (it never baked
   wrappers). Wrapper baking is the molbuilder-side `molbuilder bench prep`
   step — valid because the contract guarantees molbuilder on the target
   (§ 3.4). The no-molbuilder path is detection-only by design.
2. **Clean HPC job shell (`--export=NONE`)**: unchanged — `write_run_wrapper`
   still bakes `module load mamba` + `source activate` verbatim; the wrapper
   self-bootstraps from a clean shell.
3. **Workstation vs HPC discriminator = the detected scheduler**, not a guess:
   `workstation` → autodetect conda (login shell has it); `slurm` → require
   explicit config (the *job* shell is clean, so autodetect on the login node
   would bake the wrong thing — § 3.5). This is why resolution must be
   scheduler-gated, not "autodetect-first".
4. **Refuse-to-emit preserved, relocated**: "no activation ⇒ stop with a
   pointer" now fires at prep on the real target (where it's actionable),
   not at generate on an unrelated host.
5. **One portable bundle, prepped per target.** A workstation prep *writes*
   the resolved `conda activate` into that dir's `.molbuilder.json`
   (specialising it); an HPC prep *uses* the shipped one. The supported flow
   is **copy a fresh bundle to each target, prep there** — re-prepping the
   *same* directory on a different machine is not supported (the prior
   target's resolved activation would linger); re-copy instead. This is why
   resolution is recomputed from `(scheduler, autodetect, shipped config)`
   every prep, never inherited from a previous machine's bake.
6. **`.sbatch` follows the DETECTED scheduler, not the shipped config.**
   `write_run_wrapper` keys `.sbatch` emission off the config's `scheduler`
   block; left alone, a portable bundle that ships an HPC `scheduler` block
   but is prepped on a workstation would emit stray SLURM `.sbatch` files that
   don't match the machine. `bake_target_wrappers` therefore passes
   `emit_sbatch=(env.scheduler == "slurm")`: workstation → `.run.sh` only;
   SLURM → `.run.sh` + `.sbatch`. (Validated 2026-06-28.)

### 7.6 Test plan

- `tests/test_bench_generate.py`: drop the generate-time wrapper/activation
  assertions; assert generate writes `bench-manifest.json` (correct knobs) and
  a target-neutral bundle (no `.run.sh`, `.molbuilder.json` only when explicit
  flags given). Keep transform/sweep-placeholder/prep-lib tests as-is.
- `tests/test_bench_prep.py`: same one bundle prepped twice —
  (a) `--scheduler workstation` (monkeypatched `detect_conda_activation`) bakes
  `conda activate` into `job-cpu.run.sh`; (b) `--scheduler slurm` with a shipped
  HPC `.molbuilder.json` bakes `source activate` + `module load mamba` — **no
  regeneration**; and (c) `--scheduler slurm` with no config raises the
  pointer. Assert `gres`/`-c` come from the detected topology.
- End-to-end (task #4): generate → prep(workstation) → bash run; the
  `bash -n` validity + standalone checks already in the suite.

### 7.7 Resolved decisions

1. **No draft wrapper at generate.** The bundle is target-neutral and not
   runnable until `prep` bakes the wrappers on the target — which the
   benchmark flow always does. (Implemented: generate writes no `.run.sh`.)
2. **Manifest at the bundle root** — `bench-manifest.json`, sibling of
   `environment.json`.
3. **README regenerated now** — `render_readme` documents the prep-bakes flow;
   the § 4 cookbook recipes were reconciled in the same change.

---

## 8. Workflow redesign — one entry point, full transparency (2026-06-28)

> **Status: IMPLEMENTED (2026-06-28).** Supersedes the entry-point / shim
> parts of § 7 (the prep-time-baking contract of § 7 stays). It existed
> because § 7's *implementation* shipped a confusing, opaque on-target
> experience; this section is the contract the redesigned workflow now honors.
> Code: `bench/generate.py` (`_ship_entry_shims`, `render_bench_plan`,
> manifest@2), `bench/_cli.py` (`cmd_prep` writes+prints `BENCH-PLAN.md`),
> tests in `test_bench_{generate,prep}.py`. § 8.9 decisions all = yes.

### 8.1 What was wrong (honest list)

1. **Two commands, one of them hidden.** `prep-bench` (a shell script in the
   bundle) did **detection + sweep only**; baking the run wrappers lived in a
   *separate* command, `molbuilder bench prep`, that the user had to already
   know. A shell entry point that can't do its own job and makes you memorise
   a Python entry point is backwards.
2. **Broken half-state.** Running `./prep-bench` left **no `job-cpu.run.sh` /
   `job-gpu.run.sh`**, while the sweep it wrote symlinked `job-gpu.run.sh` —
   so the bundle was un-runnable, with no message saying so.
3. **No plan / no transparency.** Nothing told the user *what* would be
   benchmarked: which points, in what order, what parameter varies, what is
   measured, or how to change it. You had to reverse-engineer it from
   `job-gpu-sweep.sh` (bash) + `bench-manifest.json` (terse) + the `.fdf`.
4. **CPU baseline hidden.** `job-cpu.fdf` *is* the CPU-only point, but it is
   **not in the sweep** and nothing says "run it too" — so "CPU vs GPU" looks
   like it has no CPU.
5. **Inconsistent knobs.** `manifest.gpu.gpu_k` (the single GPU point's ranks)
   ≠ the sweep's `--gpu-ks` list — two different things spelled the same, no
   explanation.
6. **Dead weight.** The bundle shipped a **stdlib copy of the prep-lib**
   (`mbbench/`) whose only reason to exist was "the target might not have
   molbuilder." **The contract (§ 3.4) says molbuilder *is* on every target**,
   so that copy is obsolete — and having two implementations is itself a
   source of the confusion above.

### 8.2 Root cause

The stdlib-only `mbbench/` + the prep split are leftovers from a pre-contract
assumption ("no molbuilder on the target"). Once § 3.4 fixed the contract to
**molbuilder installed on every target**, that whole layer should have been
retired. It wasn't. This redesign retires it.

### 8.3 The corrected model — one entry point per stage

The bundle ships **bash shims that SELF-BOOTSTRAP the molbuilder env and then
call the molbuilder machinery** — the user never activates anything by hand.
Each shim is the obvious entry point; none asks the user to know a second
command.

**The bootstrap (job-execution.md § 3.4 contract; `_shim_bootstrap`).** On a
fresh shell the shim makes the `molbuilder` CLI callable, mirroring the T/M
detection rule, then runs it:
1. *Activate the env* (only if molbuilder isn't already importable):
   **workstation** — conda/mamba is on PATH (the user guarantees it) → source
   the conda hook + `conda activate <env>`; **HPC clean shell** — conda/mamba
   is *not* on PATH → load it via the bundle's `.molbuilder.json` `preamble`
   (e.g. `module load mamba`), then activate. The host-env name is baked from
   the generating env; override with `MB_HOST_ENV`.
2. *Resolve the invocation* (`$_mb_run`), in order: **`MB_REPO=<path>`** (the
   explicit escape hatch — point it at a molbuilder checkout) → the
   `molbuilder` console script if installed → `python -m molbuilder` if
   importable → (dev checkout) the repo root found by walking up from the shim
   → `env PYTHONPATH=<repo> python -m molbuilder`. (So a bundle placed under
   the repo works in dev; a bundle copied to a bare HPC node needs molbuilder
   *importable* in that env — installed, or `MB_REPO` pointed at a checkout.)

| shim (in the bundle) | calls | does |
|---|---|---|
| **`./prep-bench`** | `molbuilder bench prep` | **the whole on-target prep**: detect topology → bake `job-cpu.run.sh` + `job-gpu.run.sh` (+`.sbatch` on SLURM) → write the sweep → write **`BENCH-PLAN.md`** → print the plan + next step |
| **`./run-bench`** | (runs the baked scripts) | run **every** point — CPU baseline **and** the GPU sweep — in order, each isolated; the one command to execute the benchmark |
| **`./bench-summarize`** | `molbuilder bench summarize` | rank the points → `bench-result.json` |
| **`./prep-run`** | `molbuilder bench prep-run` | winner → `run-production.sh` |

- **`mbbench/` (stdlib copy) is removed.** The shims bootstrap the env (above)
  then `exec $_mb_run bench <sub> "$@"`. One implementation, in molbuilder.
- **No half-state.** `./prep-bench` always produces a runnable bundle or
  fails loudly with the fix.

### 8.4 Transparency — `BENCH-PLAN.md` (the heart of the fix)

`prep` writes `BENCH-PLAN.md` **and prints it**. It enumerates the full test
matrix in plain language — what runs, the order, the one varied parameter,
what is measured, and exactly how to change each. Concrete shape:

```
BENCH PLAN — siesta, detected: workstation, 2×10 cores, 1×rtx GPU
Measured per point: SCF wall-time per iteration (s/iter), from <basename>-run0.scf-timing.log.
Varied parameter:  K = MPI ranks sharing one GPU (GPU points only).  CPU point is the baseline.
Run order (./run-bench runs them top to bottom; each in its own point-*/ dir):

  #  point        engine path              ranks  cores/rank  GPU      what it answers
  0  cpu          ELPA-1STAGE (no CUDA)    20     1           —        CPU baseline (s/iter)
  1  gpu-G1K1     ELPA-CUDA                1      10          1×rtx    1 rank/GPU
  2  gpu-G1K2     ELPA-CUDA                2      5           1×rtx    2 ranks/GPU (MPS)
  3  gpu-G1K5     ELPA-CUDA                5      2           1×rtx    5 ranks/GPU (MPS)
  4  gpu-G1K10    ELPA-CUDA                10     1           1×rtx    10 ranks/GPU (MPS)

How to change:
  • GPU K values     : ./prep-bench --gpu-ks 1,2,5,10   (re-runs prep)
  • CPU rank count   : ./prep-bench --cpu-np 20
  • SIESTA params    : edit job-cpu.fdf / job-gpu.fdf (MeshCutoff, kgrid, PAO.BasisSize, …),
                       then ./prep-bench again
Next step:  ./run-bench      (or run one point: bash job-cpu.run.sh / bash job-gpu.run.sh)
```

The plan is generated from the detected `environment.json` + the manifest, so
it always matches what will actually run (no probe/launch mismatch).

### 8.5 CPU folded into the matrix

The CPU baseline becomes **point 0 of the benchmark**, not a separate hidden
file. `./run-bench` runs the CPU point then the GPU sweep; `summarize` ranks
CPU vs the best GPU point from the same set. `job-cpu.run.sh` still exists for
running it alone.

### 8.6 Self-describing manifest

`bench-manifest.json` gains a top-level human-readable description and per-job
intent + units, and the two "K"s are disambiguated:

```json
{
  "schema": "molbuilder/bench-manifest@2",
  "engine": "siesta",
  "description": "CPU-vs-GPU SCF-timing benchmark; cold, capped, single-point.",
  "measured": "SCF wall-time per iteration (s/iter)",
  "points": {
    "cpu":  { "script": "job-cpu.fdf", "solver": "ELPA-1STAGE (no CUDA)",
              "mpi_np": 20, "cpus_per_task": 1, "role": "baseline" },
    "gpu":  { "script": "job-gpu.fdf", "solver": "ELPA-CUDA",
              "gpus": 1, "sweep_param": "K = ranks/GPU",
              "sweep_default": "divisors(cores_per_socket)" }
  }
}
```

### 8.7 Artifact model

| file | after `bench generate` (host) | after `./prep-bench` (target) |
|---|---|---|
| `job-{cpu,gpu}.fdf`, `*.psml` | ✅ | ✅ |
| `bench-manifest.json` (self-describing) | ✅ | ✅ |
| `prep-bench`/`run-bench`/`bench-summarize`/`prep-run` (thin shims) | ✅ | ✅ |
| `README.md` | ✅ | ✅ |
| `mbbench/` (stdlib copy) | ❌ removed | ❌ |
| `environment.json` | — | ✅ |
| `job-{cpu,gpu}.run.sh` / `.sbatch`, `mb_monitor.py` | — | ✅ baked |
| `job-gpu-sweep.sh` (placeholder → real) | placeholder | ✅ real |
| **`BENCH-PLAN.md`** | — | ✅ written + printed |

### 8.8 Implementation + test surface

- `bench/generate.py`: replace `_ship_prep_lib` (mbbench copy) with
  `_ship_entry_shims` (thin `molbuilder bench …` shims incl. `run-bench`);
  manifest@2 (self-describing); drop `_PREP_LIB_MODULES`.
- `bench/_cli.py`: `cmd_prep` writes+prints `BENCH-PLAN.md`, writes `run-bench`,
  folds in the CPU point; `prep-bench` is the documented single entry.
- `bench/adapters.py`: sweep includes the CPU baseline as point 0.
- Retire the obsolete "runs with no molbuilder" tests
  (`test_shipped_prep_lib_runs_with_no_molbuilder`,
  `test_shipped_summarize_and_prep_run_shims_standalone`,
  `test_generate_ships_prep_lib_verbatim`) → replace with "shim calls
  `molbuilder bench …`" + "prep writes BENCH-PLAN + run-bench + both run
  scripts" + "manifest@2 is self-describing" tests.
- `benchmark-workflow.md` § 7 reconciled to the single-entry model.

### 8.9 Resolved decisions (confirmed 2026-06-28 — all yes)

1. **`run-bench` runs CPU + full GPU sweep by default** — one command does the
   whole matrix; individual `job-*.run.sh` remain.
2. **`mbbench/` dropped entirely** — obsolete per the § 3.4 contract; the shims
   call molbuilder.
3. **Manifest bumped to `@2`** with the self-describing shape above (pre-1.0,
   no back-compat shim).

### 8.10 Fresh-eye review finding — the workstation sweep didn't sweep (FIXED 2026-06-28)

A post-implementation review of the *generated* scripts found that the
**workstation** GPU sweep set `MOLBUILDER_MPI_NP=K` / `MOLBUILDER_OMP_NUM_THREADS`
per point — but the baked `job-gpu.run.sh` carries an **explicit** `mpi_np`, so
those vars only set the *auto-mode* default, which the explicit value
**shadows**. Verified empirically: `MOLBUILDER_MPI_NP=8 → 4 ranks` (ignored),
`MB_NP=8 → 8 ranks` (honored). So every GPU sweep point ran the *same* baked K
— the sweep didn't sweep — and the same bug hit the workstation **production**
run (`prep-run`). (SLURM was fine: `sbatch -n` → `SLURM_NTASKS`, which the
wrapper honors.) Several tests had *locked in* the bug (one even commented "the
launcher actually honours these").

**Fix:** `WorkstationAdapter.gpu_launch_line` now emits `MB_NP` /
`OMP_NUM_THREADS` (the vars the wrapper's *launch* actually reads — matching
its own `cpu_launch_line`). Tests corrected to the honored vars.

### 8.11 Open follow-up — CPU baseline default rank count

The CPU baseline bakes `points.cpu.mpi_np = 64` (a Sol-sized default from
`bench generate`). On a target with fewer cores, `./run-bench`'s CPU point
asks `mpirun -np 64`, which OpenMPI **refuses by default** (not enough slots)
unless oversubscription is allowed — so the CPU point can fail to launch, not
merely thrash. `BENCH-PLAN.md` surfaces the value + how to change it (edit
`points.cpu.mpi_np`), but a cleaner default would be for `prep` to clamp the
CPU baseline to the detected core count (sockets×cores_per_socket) unless the
user overrode it — adapting resources to the machine, the same way GPU `-c` is
derived. **Not yet implemented** (deferred — it changes a default, the
scientist's call). Until then: set `points.cpu.mpi_np` before `./run-bench`.
