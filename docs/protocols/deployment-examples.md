# Deployment examples & templates (cookbook)

Copy-paste worked examples for running the generated workflows (benchmark,
transport) on **a workstation** and on **a supercomputer**, covering the
situations you actually hit, with `.molbuilder.json` templates to study.

- The *model* + rationale (env phases, resolution rule, SLURM/CUDA relay):
  [`deployment.md`](deployment.md).
- *Building* the envs (Phase A — your responsibility, once per machine):
  [`README_install.md`](../README_install.md).
- The *sbatch* block, line by line: [`slurm-integration.md`](slurm-integration.md).

> **Prerequisite for every example:** the envs are already prepared (Phase A)
> and `molbuilder-siesta` / `molbuilder-siesta-gpu` show up in `conda env list`.
> If not, that's the first error you'll see — see § A.4 / § B.4.

> **Today vs target.** Until the runtime-resolution work lands (deployment.md
> § 7, tasks #24/#25), activation is baked at **generate time**, so you
> **generate one bundle per target machine**. Every example below reflects
> that. § C explains the one-bundle-portable target.

---

## A. Workstation

You have full control; conda/mamba is on PATH; jobs run via `bash …run.sh`.

### A.1 Happy path — conda on PATH, env ready

```bash
molbuilder bench generate input.fdf --out bench-ws/   # activation auto-detected
cd bench-ws/
./prep-bench --gpu-ks 1,2,4,8       # detect THIS machine -> environment.json + sweep
bash job-cpu.run.sh                 # CPU point
bash job-gpu-sweep.sh               # GPU sweep (sequential)
./bench-summarize                   # rank CPU vs GPU -> winner
./prep-run --script-base myprod     # winner -> run-production.sh
```
No `.molbuilder.json` needed — `generate` detects your local conda and bakes
`conda activate` into the scripts.

### A.2 `mamba` instead of `conda`

Same as A.1 — auto-detection probes `$MAMBA_EXE` / `mamba` too. To force it,
or to pin the form, drop a workstation config (§ D.1) or pass
`--activation "mamba activate"`.

### A.3 GPU box vs CPU-only box

`prep-bench` detects GPUs. On a **1-GPU** box only `G1` sweep points are
emitted; scale *processes* with `--gpu-ks` (K = ranks sharing the GPU), e.g.
`--gpu-ks 1,2,4,8` (oversubscription past cores/socket is allowed + flagged).
On a **no-GPU** box, run only `job-cpu.run.sh`. **VRAM caveat:** a large Au
system in ELPA-CUDA can OOM a consumer card (e.g. a 12 GB RTX 3060) — bench a
smaller structure if it won't fit.

### A.4 Env not prepared → what you see + the fix

If `molbuilder-siesta-gpu` isn't built, the run aborts when it can't activate
it. Fix it in Phase A (it is *your* step, the script never builds it):
```bash
molbuilder envs doctor          # what's missing / broken
molbuilder envs install molbuilder-siesta-gpu   # or: molbuilder envs bootstrap
```

### A.5 Non-standard conda / custom activation

If auto-detect can't find your conda, or you want a specific activation, pass
it explicitly (it gets written into the bundle's `.molbuilder.json`):
```bash
molbuilder bench generate input.fdf --out bench-ws/ \
    --activation "conda activate" --preamble 'source "$HOME/miniconda3/etc/profile.d/conda.sh"'
```

---

## B. Supercomputer (SLURM, e.g. ASU Sol)

A *constrained* environment: the job lands in a **clean shell**
(`--export=NONE`), so `mamba`/`conda` is not on PATH until `module load mamba`
runs — that step **must** come from config/flags (it can't be auto-detected).
Jobs run via `sbatch`.

### B.1 Happy path — Sol

```bash
# generate WITH the activation Sol will use (or ship the config of § D.2):
molbuilder bench generate input.fdf --out bench-sol/ \
    --activation "source activate" --preamble "module load mamba"
# copy bench-sol/ to Sol, then ON Sol:
cd bench-sol/
./prep-bench --gpu-ks 8,16       # detects SLURM + A100 topology -> sbatch-able sweep
sbatch job-cpu.sbatch            # CPU point
bash job-gpu-sweep.sh            # GPU sweep (sbatch per point)
./bench-summarize
```

### B.2 Clean shell — why the preamble is load-bearing

Under `--export=NONE` the job starts with no `mamba`/`conda`. The wrapper runs
the baked `module load mamba` **before** `source activate <env>`; without the
preamble the activation fails immediately. This is exactly why a Sol bundle
**must** carry `script_generation` (§ D.2) — there is nothing to auto-detect.

### B.3 CPU vs GPU allocation

- **CPU:** `sbatch -n 64 job-cpu.sbatch` — `-n` → `SLURM_NTASKS` → `mpirun -np`.
- **GPU:** `--gres=gpu:a100:G` → SLURM sets `CUDA_VISIBLE_DEVICES`; the wrapper's
  per-rank launcher gives each rank one GPU (+ MPS for K≥2). Run GPU
  `--exclusive` for clean timing.
- **CPU may time out:** plain `diagon` on a few-hundred-atom system can exceed
  the 4 h default before the capped iters finish — set `--cpu-time` at
  generate or `sbatch -t …`.

### B.4 Env not prepared on the cluster → error

Same as A.4, on Sol: `molbuilder envs doctor` then `molbuilder envs install …`.
The run scripts never create the env.

### B.5 A different cluster / module system

Auto-detect doesn't apply on HPC. Provide that site's load+activate via flags
or the config (§ D.3): e.g. `--preamble "module load anaconda3" --activation "conda activate"`,
or whatever the site documents.

---

## C. Cross-machine (generate here, run there)

The common real case: you have molbuilder on your workstation, the job runs on
Sol. **Today:** generate a *Sol-configured* bundle (B.1 — the activation is the
*target's*, not your workstation's), copy it, run. The workstation's conda is
irrelevant to that bundle.

**Target (tasks #24/#25):** runtime resolution makes one bundle portable — the
*same* `bench-*/` runs on the workstation (auto-probe) and on Sol (reads its
`.molbuilder.json`), no per-target regeneration. When that lands, A.1 and B.1
collapse into one generate.

---

## D. Templates — `.molbuilder.json` (copy, edit, drop in the OUT dir)

Lives in the bundle's OUT dir, merged over a server-wide
`~/.config/molbuilder/molbuilder.json` (project wins). See deployment.md § 3.

### D.1 Workstation (usually none needed; this only pins the activation)

```json
{ "script_generation": { "activation": "conda activate" } }
```

### D.2 ASU Sol (SLURM + GPU)

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

### D.3 Generic HPC (adapt the partition + module to your site)

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

---

## E. Quick decision guide

| situation | what to do |
|---|---|
| my workstation, conda on PATH | `bench generate … --out X` (no config) → `bash *.run.sh` |
| my workstation, weird conda | add `--activation`/`--preamble` (§ A.5) or § D.1 |
| SLURM cluster | `--activation "source activate" --preamble "module load mamba"` (or § D.2/D.3) → `sbatch` |
| generate here, run on HPC | configure for the **target**, copy, run (§ C) |
| env missing anywhere | `molbuilder envs doctor` → `… install` (Phase A) |
| GPU job won't fit | fewer ranks/GPU (`--gpu-ks`), smaller structure, or CPU-only |

Transport is the same shape: `transport bundle …` for the scripts,
`slurm-integration.md` for sbatch, this cookbook's templates for the env.
