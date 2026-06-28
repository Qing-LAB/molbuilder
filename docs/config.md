# Configuration — wrapper contract + generator inputs

> **Status**: rewritten 2026-06-24 against the wrapper-independence
> contract.  Supersedes the previous version of this doc (which
> designed a config-driven wrapper — wrong shape, removed entirely).
>
> Cross-reference: [design.md](design.md) decision-log; any change to
> the wrapper's contract or the generator's inputs must update this
> file in the same commit.

---

## 1. The wrapper contract

The generated `<basename>.run.sh` is a **self-contained shell script**.
Given the `.fdf` (or `.py`) input sitting next to it and the env it
was generated for, it runs the job under SLURM (or directly).

**What the wrapper does at runtime, in this order:**

1. Open a log file in the **current working directory** named
   `<basename>.runwrap-<YYYYMMDD-HHMMSS>.log`.  Tees stdout +
   stderr to that file while keeping them on the original streams.
2. Run the **server preamble** verbatim (e.g. `module load mamba`).
   Baked at generate time from `molbuilder.json`.
3. Run **project additions** verbatim (env vars, extra setup).
   Baked at generate time from `.molbuilder.json` if present.
4. Run the **activation command** verbatim (e.g.
   `source activate molbuilder-siesta-gpu`).  Baked at generate
   time; both the form (`source activate` or `conda activate`) and
   the env name are decided by the generator.
5. Launch the engine: `mpirun -np $_mpi_np siesta < <basename>.fdf
   > <basename>-runN.out` (SIESTA) or the PySCF equivalent.

**What the wrapper does NOT do:**

* **Does not change cwd.**  SLURM lands the job in `SLURM_SUBMIT_DIR`
  by default; direct users `cd` to the project dir before
  invocation.  The caller's cwd is the contract.  Outputs land where
  the wrapper was invoked.
* **Does not read any external file** other than the engine input
  (`<basename>.fdf` / `<basename>.py`).  No config files, no
  `~/.bashrc`-style sourcing.
* **Does not read environment variables to alter activation or
  preamble behaviour.**  Specifically: there is no
  `MOLBUILDER_PREACTIVATE_CMDS` hook, no autodetect toggle, no
  way to swap the activation form at runtime.  See § 1.5 for the
  narrow set of vars the wrapper DOES read (scheduler vars for
  launch tuning — that's the scheduler's published contract, not
  user configuration).
* **Does not probe.**  No `conda info --base`, no `mamba info --base`,
  no `module avail`, no filesystem walks, no PATH searching.
  Everything is baked.
* **Does not have a fallback path.**  If `module load mamba` fails,
  if the env doesn't exist, if activation errors — the wrapper
  aborts (`set -euo pipefail`) with the real bash error in the
  log.  No silent rescue, no alternative path.

**What the wrapper assumes about its environment:**

* `$HOME` is set (SLURM inherits, doesn't strip).
* The target env is installed and reachable through the toolchain
  the server preamble loads.  molbuilder's `install-env.sh` /
  `molbuilder envs install` machinery guarantees this.
* The `.fdf` (or `.py`) is in the caller's cwd.

### 1.5 What the wrapper additionally does (allowed exceptions)

A small set of behaviours are NOT runtime discovery / config reads
but interact with externalities in a bounded way.  All four are
permitted by the contract:

| behaviour | what it reads | why consistent with the contract |
|---|---|---|
| **Run-index resolution** (picks `-run0`, `-run1`, ... for `.out` files) | sibling `<basename>-runN.out` files in cwd | reading the caller's cwd to choose its own output filename; not external config, not detection of tools |
| **Argument parsing** (`--continue` / `-c`, `--force` / `-f`, `-np N` / `--np N`, `-omp N`, `-h`) | the wrapper's own argv | explicit user input to a single invocation; not external state |
| **Scheduler-var read** (`SLURM_NTASKS`, `SLURM_GPUS_ON_NODE`, etc.) | env vars set by SLURM/PBS at job start | the scheduler's published contract for handing the job its rank count + GPU allocation; informs `mpirun -np`, never changes activation or preamble |
| **SIESTA propor post-mortem** (after SIESTA crashes, grep its `.out`/runwrap-log for `propor: ERROR: IMAX = 0` and print a multi-cause hint: defective/mismatched pseudopotential FIRST, then `-np` as a tunable, then `Spin.Total`) | the engine's own output file | post-mortem text-analysis on the engine's output; not a fallback path, not a retry, just a hint printed before exiting with SIESTA's exit code |

None of these read configuration files, probe for tools, or change
which env / preamble runs.  They are bounded reads of (a) the
caller's cwd contents, (b) the wrapper's own argv, (c) the
scheduler's documented env vars, (d) the engine's own output after
it ran.

### 1.6 Multi-stage SIESTA wrappers

A multi-stage SIESTA run emits N per-stage `.fdf`s plus an
orchestrator `<basename>.run.sh` that loops over them.

* Each per-stage `<basename>-stage<i>.run.sh`, if generated, is a
  single-engine wrapper conforming to this contract.
* The orchestrator `<basename>.run.sh` is a thin bash loop: it
  invokes each per-stage wrapper in turn under the same activated
  env (preamble + activation done once at the top of the
  orchestrator, then a loop).  It does not re-activate per stage.

Both shapes obey §§ 1 / 1.5: no config reads, no detection, no
fallback, no cd.

---

## 2. The generator contract

The generator is `molbuilder.runwrap.render_run_wrapper` plus the
config-reading helpers it calls.  At generate time it can read
anything it needs.  Its job is to produce a wrapper that satisfies
§§ 1 / 1.5 / 1.6.

**What the generator reads at generate time:**

* The target env name (from the recipe / capabilities snapshot).
* `script_generation.preamble` from `molbuilder.json` (server-wide).
* `script_generation.activation` from `molbuilder.json` (server-wide).
* `script_generation.preamble` from `.molbuilder.json` (project-scope,
  if present).
* Engine-specific knobs (mpi_np, omp, BlockSize) — unchanged from
  the existing implementation.

**What the generator emits:**

A `.run.sh` whose content is the result of substituting the read
values into a fixed template.  No conditionals at runtime, no
detection blocks, no fallback branches.

**Failure mode at generate time (refuse-to-emit on missing
essentials):**

The generator REFUSES to emit a wrapper if essential config is
missing.  Specifically:

* `script_generation.activation` is not set in either scope →
  generator prints a clear error naming the missing key + the doc
  reference (`docs/config.md § 4`) and exits non-zero.  No wrapper
  is written.
* `script_generation.preamble` is empty in BOTH scopes AND the
  generator detects it's being asked to produce a wrapper for an
  HPC-target env (siesta-gpu, siesta) → same: print error, point
  at docs, exit non-zero.  Pure CPU envs without HPC association
  (e.g. molbuilder host env runs locally) may proceed with an
  empty preamble.

The principle: a configuration mistake produces an error at GENERATE
time when the operator is still at a terminal in front of the
project — not at SUBMIT time when the job is queued on a remote
cluster.  "Broken on first run" is impossible by construction.

---

## 3. Where config lives

| scope | file | when read |
|---|---|---|
| **server-wide** | `molbuilder.json` | generate time only |
| **project** (optional) | `.molbuilder.json` | generate time only |

Both files share the same JSON schema.  Neither is read at wrapper
runtime.

### Lookup order

**Server-wide** (in priority order; first match wins, no merging
across):

1. `./molbuilder.json` (cwd of the `molbuilder run` invocation).
2. `~/.config/molbuilder/molbuilder.json` (XDG fallback; honours
   `$XDG_CONFIG_HOME` when set, else `$HOME/.config/`).

**Project** (single location, optional):

* `<project_dir>/.molbuilder.json`, where `<project_dir>` is the
  directory holding the `.fdf` / `.py` being generated.

**Merge across scopes** (project + server-wide):

* `preamble` strings concatenate: server-wide first, then project,
  joined by `"\n"`.  Either may be empty.
* `activation`: project value wins if set; else server-wide value.
  At least one scope must set it (per § 2's refuse-to-emit rule).

No section-level merging between the two server-wide candidates
(cwd file vs XDG file).  Only ONE of those is read.

---

## 4. Schema — the keys the generator needs

```json
{
  "script_generation": {
    "preamble":   "module load mamba",
    "activation": "source activate"
  }
}
```

| key | type | default | meaning |
|---|---|---|---|
| `preamble` | string (multi-line bash) | empty | Verbatim lines run at the top of the wrapper, before activation.  Typically `module load mamba` on HPC sites that gate the conda toolchain behind environment-modules; can include `export FOO=bar` lines for env vars. |
| `activation` | one of `"source activate"`, `"conda activate"` | **no default — must be explicitly set in at least one scope** | How the wrapper activates the env.  `source activate` is the legacy form that works whenever the `activate` script is on PATH (typical after `module load mamba`).  `conda activate` is the modern shell-function form that requires `conda.sh` to have been sourced. |

Both keys are valid in either scope.

There are no other keys under `script_generation`.  No
`autodetect_conda`, no `preactivate_format`, no `preactivate`.  Those
existed in a prior version of this doc and were removed.

**Why `activation` has no default:** picking one silently smuggles a
target-cluster assumption into every fresh deployment.  Sol uses
`source activate`; modern conda installs use `conda activate`.  The
operator picks during initial setup — once — and the generator
refuses to operate without it.  See § 2's refuse-to-emit rule.

**Why the enum is two values, not three (no `mamba activate`):**
on modern mamba + conda, `mamba activate <env>` and
`conda activate <env>` resolve to the same shell function loaded
from the same conda hook — the distinction is just which binary
appears on PATH, not a different activation mechanism.  If the
operator's binary is named `mamba`, use `source activate` (legacy,
binary-agnostic) or `conda activate` (modern shell-function form,
provided the conda hook is sourced).  No need for a third value
that would behave identically to one of the other two.

---

## 5. Other top-level keys in `molbuilder.json`

This doc is only about `script_generation`.  Other top-level keys
(`tls`, `auth`, `envs`, `rate_limit`, `secret_key_file`) are
unchanged.  Their schemas + usage are documented in
[`docs/deployment.md`](deployment.md).

They are also read at generate time / serve time only; none of them
change the wrapper's runtime behaviour.

---

## 6. What was removed from the previous version

The previous draft of this doc designed a config-driven wrapper.
Everything below was deleted because the wrapper now does no runtime
discovery:

* `script_generation.preactivate` (renamed to `preamble`; semantics
  also changed — the wrapper used to "respect" preactivate via a
  baked-in block + a runtime env-var hook; the env-var hook is gone)
* `script_generation.preactivate_format` (only one format existed
  anyway)
* `script_generation.autodetect_conda` (the 6-path detection is
  gone; this knob has no meaning anymore)
* `MOLBUILDER_PREACTIVATE_CMDS` runtime env var (gone)
* The 6-path conda detection block in the wrapper (gone)
* The "fail loud vs autodetect" decision (irrelevant under the new
  contract — the wrapper always does exactly what was baked)
* The per-run log "preactivate trace via `set -x`" idiom (gone; the
  baked preamble is plain lines)
* The cd at the wrapper top (gone)
* The multi-line activation-failed help heredoc (gone — `set -euo
  pipefail` aborts with the real bash error, which is more useful
  than a generic hint)

The wrapper itself shrinks substantially as a result.

---

## 7. Migration

* `script_generation.preactivate` → rename to `preamble`.  The
  config reader emits a one-time warning when it sees the old key
  and treats it as `preamble` for one release, then drops the
  alias.
* `script_generation.autodetect_conda` → drop entirely.  The
  reader silently ignores unknown keys, so this isn't load-bearing
  on the user side.
* `script_generation.preactivate_format` → drop entirely.
* `MOLBUILDER_PREACTIVATE_CMDS` → users who set this in shell init
  files should remove it.  The reader code does not consume it; the
  wrapper code does not consume it.  It's a no-op as of this rewrite.
* Fresh installs on HPC (e.g. ASU Sol): set
  `script_generation.activation: "source activate"` + a `preamble`
  appropriate to the cluster.  Without `activation` set, the
  generator refuses to emit a wrapper.

The example file `docs/molbuilder.json.example` will be rewritten
against this schema in the same commit as the code change.

---

## 8. Implementation plan

1. **`runwrap.py`** — strip the 6-path block, the env-var hook, the
   `autodetect_conda` plumbing, the cd, the multi-line error
   heredoc.  Replace with a small render that emits: log setup +
   verbatim preamble (server then project) + verbatim activation +
   engine launch.  Tests pinning the new shape; old tests pinning
   the discarded behaviours get deleted, not patched.
2. **`runtime_config.py`** — replace `get_script_generation` with
   the new schema (`preamble` + `activation`).  Add the
   refuse-to-emit guard described in § 2.  Keep
   `read_effective_config` / `write_config_scope` — they're sound.
3. **`docs/molbuilder.json.example`** — rewritten `script_generation`
   block matching the new schema.
4. **`README.md`** — deployment section's HPC-preflight subsection
   updated for the new key names + the wrapper-self-contained
   property.

UI work for editing config from the browser (Preflight card,
Generate-script preview modal) was in the previous draft.  It is
**not part of this doc** — those are how the user EDITS config, not
how the wrapper consumes it.  Whatever UI we build later, it writes
to the same `molbuilder.json` / `.molbuilder.json` files and the
schema above.

---

## 9. Detection model — what is detected, when, per target

> **Scope — this is job execution, not deployment.** This section (and
> config.md as a whole) documents *using the script-generator **module***
> to **submit and run calculations** on a target: standalone
> `.run.sh`/`.sbatch`, activation, the `script_generation` / `scheduler`
> keys.  **Deploying molbuilder itself** (serving the web app, auth, TLS,
> the `tls`/`auth`/`envs` keys) is the separate concern in
> [`deployment.md`](deployment.md).  One `molbuilder.json`, two key-owners.

"the wrapper does no detection" is too coarse to be a spec.  There are
several distinct things one might call "detection", and the rule differs
for each.  This section is the authoritative definition of *standalone*,
*detection*, the *assumption* each target makes, and the *goal*.  The
worked, copy-paste cookbook that applies this is
[`protocols/job-execution-examples.md`](protocols/job-execution-examples.md);
the SLURM/HPC specifics are in
[`protocols/slurm-integration.md`](protocols/slurm-integration.md).

### 9.1 Definition

**Detection = reading external state and choosing behaviour from it
without the user stating that choice explicitly.**  The external state is
one of five things, and the rule depends on *what* is read and *when*:

| | WHAT is read | example |
|---|---|---|
| **T** | conda/mamba **tool** | `$CONDA_EXE`, `which mamba`, `$CONDA_PREFIX` |
| **M** | HPC **toolchain** (modules) | whether `module load mamba` is needed |
| **C** | **config** file | `molbuilder.json` / `.molbuilder.json` |
| **A** | scheduler **allocation** | `SLURM_NTASKS`, `CUDA_VISIBLE_DEVICES`, `SLURM_CPUS_PER_TASK` |
| **H** | **hardware** topology | GPU PCI bus → NUMA node → socket (sysfs) |

### 9.2 The three moments

| moment | where it runs | does what |
|---|---|---|
| **generate** | the machine where you run `molbuilder bench generate` / `run` | resolves T/M/C and **bakes** them into `.molbuilder.json` + the `.run.sh` (verbatim) |
| **prep / doctor** | **on the target** (`./prep-bench`, `molbuilder envs doctor`) | detects the scheduler + topology to *format* the sweep, **and verifies readiness honestly (every target)** — § 9.4 |
| **runtime** | the compute node, inside `.run.sh` | everything baked; reads only A/H (the scheduler's published contract, § 1.5) |

### 9.3 The rule matrix

| WHAT | GENERATE-TIME | RUNTIME (inside `.run.sh`) |
|---|---|---|
| **T — conda/mamba tool** | ✅ **allowed, narrow** — `runtime_config.detect_conda_activation` probes the command on PATH → activation form.  *The only tool-autodetect there is.* | ❌ **forbidden** — no `which conda`, no `conda info --base`, no PATH search.  Baked verbatim. |
| **M — HPC toolchain** | ⛔ **impossible → must be explicit** — on a clean login shell mamba isn't on PATH; nothing to detect.  From config `preamble`. | ❌ **forbidden** — baked verbatim (`module load mamba`). |
| **C — config file** | ✅ **required** — generator reads `molbuilder.json` / `.molbuilder.json`. | ❌ **forbidden** — the wrapper never reads a config file at runtime. |
| **A — allocation** | n/a (no allocation yet) | ✅ **required** — `SLURM_NTASKS`→`-np`, `CUDA_VISIBLE_DEVICES`→GPU map, `SLURM_CPUS_PER_TASK`→OMP. |
| **H — topology** | n/a (target node unknown) | ✅ **required** — per-rank launcher reads sysfs for GPU→NUMA→socket binding. |

**Precise restatement of "standalone":** the generated wrapper does **no
runtime detection of tools, config, or toolchain (T/M/C)** — those are
decided once at generate time and baked, which is what makes it
standalone.  It **does** read the **allocation and topology (A/H)** at
runtime, because those are what the scheduler hands the job and adapting
to them is the whole point.  At **generate time** the *only* tool-autodetect
is **conda/mamba on a workstation**; the HPC toolchain is never guessed.

### 9.4 Two different "detection" jobs (don't conflate them)

- **Job A — autodetect the activation *method*** (infer an unstated
  choice: which activation form / preamble).  **Workstation: yes**
  (toolchain is on PATH, discoverable).  **HPC: no** — explicit config.
- **Job B — doctor: verify the *truth* of prerequisites** (confirm stated
  facts: env present, toolchain loads, scheduler/GPU/driver there).
  **Every target, always** — run by `prep` on whatever target it is
  invoked on.  On HPC, doctor *runs the explicit `module load mamba`* to
  reach the truth, then checks `mamba env list` / activation / driver; it
  uses the explicit preamble to verify, it does not method-detect.
  **Doctor is prep-time, not the `.run.sh`** — the wrapper stays baked.

### 9.5 Per-target activation defaults + assumption

| target | activation form | required prerequisite (baked) | who supplies it |
|---|---|---|---|
| **workstation** | `conda activate` | `source "<base>/etc/profile.d/conda.sh"` (the conda hook — a non-interactive `bash job.run.sh` does **not** read `~/.bashrc`, so the `conda` function must be sourced) | **autodetected** + baked (`detect_conda_activation`) |
| **HPC** | `source activate` | `module load mamba` (puts the legacy `activate` shim on PATH) | **explicit** config / `asu-sol` preset |

Each form carries its own prerequisite; that is why the workstation
default (`conda activate`) *requires* baking the hook-source line — it is
load-bearing for standalone, not overreach.  Override either with
`--activation` / `--preamble` (the explicit hatch).

- **The assumption that flips behaviour:** *is the machine that generates
  the script the same one that runs it, with conda already on PATH?*
  **Yes** → workstation → generate-time tool-autodetect is valid.
  **No / clean HPC shell** → nothing about the target is detectable →
  activation + preamble must be explicit config.
- **Env creation is always the user's** (workstation *and* HPC).  Doctor
  verifies presence and stops with a pointer to `molbuilder envs doctor`;
  it never runs `envs install`.

### 9.6 The goal

A `.run.sh` / `.sbatch` that takes the job from **submit to result with
zero manual steps on the target** — every T/M/C decision resolved and
baked at generate time, every A/H decision adapted at runtime from what
the scheduler actually granted.  Submit headless (`sbatch`), log out,
collect the result.

---

This file is the source of truth for the wrapper contract + the
config keys the generator reads.  Update it in the same commit as
any change to either.
