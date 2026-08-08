# Job contracts — the on-disk formats and shared vocabulary

**Role:** contract
**Domain:** execution

**Companions:**
[`execution/running-a-job.md`](?doc=execution/running-a-job.md) — how you
actually run and watch **one** job today (the run wrapper, `molbuilder.json`,
checkpoints, the decoded-run view);
[`execution/job-system.md`](?doc=execution/job-system.md) — the JobSet batch /
staged / HPC framework;
[`execution/overview.md`](?doc=execution/overview.md) — the map plus the
current → target status picture.

**Settled contracts this doc leans on:**
[`model/structure-molstruct.md`](?doc=model/structure-molstruct.md) (the
`.molstruct.json` sidecar it round-trips with),
[`model/structure-annotations.md`](?doc=model/structure-annotations.md)
(`regions` / `frozen_atoms` / annotation channels),
[`model/overview.md`](?doc=model/overview.md) (the 0-based-internal /
1-based-user atom-index rule), and
[`engines/overview.md`](?doc=engines/overview.md) (the UI → config → script
boundary contract and the script-wrapper contract that this doc's script
blocks physically implement).

This document is the sole source of truth for the **stable on-disk shapes**
that every other part of molbuilder rests on: where a run's files live and
what they are called, the reserved comment blocks inside a generated script,
how a script resumes (or refuses to resume) from a previous run, the object
that carries a finished run forward into the next calculation, and the
system-wide vocabulary for persisted files and parameters.

These formats are deliberately **surface-agnostic and workflow-agnostic**.
The same run directory, the same `.fdf` blocks, and the same handoff object
are produced whether you clicked *Generate* in the web UI, typed
`molbuilder fdf`, or let the JobSet framework fan a batch across a cluster.
That is the point of pinning them here once: the surfaces above (single-job,
JobSet, CLI, web) all read and write to this contract instead of inventing
their own.

---

## 1. What this document owns — a reader's map

| If you need to know… | Read § |
|---|---|
| Where a job's files go and what they are named | **§ 2 — The run directory** |
| The `project / topic / structure` folder tree and its fixed topic names | **§ 2.5** |
| The comment blocks molbuilder reserves inside a `.fdf` / `.py` / `.run.sh` | **§ 3 — The generated-script contract** |
| What "warm restart", `--continue`, and `--cold` actually do | **§ 4 — Warm & cold restart** |
| How a finished run flows into Transport / a continuation / a spectrum | **§ 5 — The workflow handoff bundle** |
| What a persisted file is called system-wide, and how a config value becomes a SLURM flag | **§ 6 — The shared data vocabulary** |

Two conventions bind everything below and are stated once here:

- **Atom indices are 0-based internally, 1-based only at the human edge.**
  Every JSON payload in this doc (`regions`, `frozen_atoms`, ATOM-METADATA,
  the sidecar) uses **0-based** indices. Engine input *coordinate blocks* use
  the engine's own convention (SIESTA `.fdf` is 1-based). The full rule and
  its single conversion boundary live in
  [`model/overview.md`](?doc=model/overview.md).
- **Schema versions are checked, never guessed.** Persisted artifacts carry a
  version; a reader that meets a newer major refuses with a clear message
  rather than mis-parsing. The one shared helper is `molbuilder/persist.py`
  (§ 6.2).

---

## 2. The run directory

### 2.1 What the engine assumes, and the two rules that serve it

Both rules below are consequences of one premise, so it is worth stating first.
Everything molbuilder does around a run exists to make it true.

> **The engine runs inside one directory, and that directory is its whole world.**
> It needs the right environment and every file it will read to be reachable from
> there. It has **no knowledge of how the directory came to be** — no notion of a
> stage, a description, a benchmark, a checkpoint or a previous run — and it does
> not care. It reads what is there and writes its output beside it.

Three things follow immediately, and each is a rule elsewhere in these documents:

| Because the engine… | …molbuilder must |
|---|---|
| has no notion of a **stage** | resolve every stage parameter **before** the deck is written. A deck that needed a reader to understand the word "stage" would not run (`engines/stages.md § 1`) |
| cannot fetch anything | put **everything the run will read in place before it starts** — which is why a carried restart file is copied at prep rather than resolved later (`project-layout.md § 1.6`) |
| assumes the environment is already correct | give the wrapper exactly two jobs — **activate, then exec** — and do every decision and arrangement in Python beforehand (`running-a-job.md § 2.2a`) |

**"Reachable from there" is the precise word, not "physically present."** A
symlink to `../Au.psml` opens as a real file from inside the directory, and the
engine cannot tell the difference — so sharing one copy of a large
pseudopotential across stages is invisible to it, and legitimate.

> ⚠ **Which is why the transportable unit is the calculation folder, not the run
> directory.** Completeness here is a property of *what resolves*, not of what
> sits in the directory. Archive a lone `run-0/` and its links to the deck and
> the shared package dangle — it was never self-contained, only self-contained
> *in place*. Move the whole calculation folder and every link stays inside it.

**Who puts the engine in that directory: the launcher, and only the launcher.**
Both deployment paths do the same one thing, and neither the wrapper nor the
engine ever navigates:

| Path | How the directory is established |
|---|---|
| **workstation** | the launcher runs `<id>.run.sh` **with its working directory set to the job's directory** |
| **SLURM** | the launcher runs `sbatch` **from** that directory, and SLURM lands the job in `SLURM_SUBMIT_DIR` — the same place |

So one rule covers both: **the caller's working directory is the contract.** The
wrapper inherits it and changes it for nothing; outputs land where the wrapper
was invoked. A wrapper that navigated would break the property the engine depends
on — that *here* is where everything is — and it would break it differently under
the two launchers, which is worse than breaking it consistently.

This is not a rule the design is asking for. It is how the shipped submit path
already works, in one line per mode, and the generated wrapper says so in its own
header: *"this wrapper does NOT change cwd… the caller's cwd is the contract."*
It is written down here because it was true everywhere and stated nowhere — and
because the one place it is currently broken (a rendered block that `cd`s into an
attempt it created) is being retired for exactly this reason
([`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md)
item 12a).

#### The third party in the directory: the monitor observes, and only observes

A wrapper activates and execs. An engine reads and writes. There is one more
thing in that directory, and it has the narrowest job of all.

**`mb_monitor.py` runs beside the engine and watches it.** It is backgrounded by
the wrapper at low priority, follows the launcher's **PID** — so it knows
authoritatively when the run ended, rather than guessing from output markers —
parses the run's artifacts as they grow, and appends what it learns to a log
beside them. It carries a **notifier hook** (`register_notifier`, or a webhook
via `MB_NOTIFY_URL`), which is the deliberate customization point: what should
happen when something notable occurs is the user's to decide, not molbuilder's.

**That hook is why nobody has to be at the cluster.** A run that ends at 3am can
say so.

> **And the boundary that makes it safe: the monitor observes and notifies. It
> never decides, and never mutates the calculation.**
>
> This is the same rule the wrapper has — activate and exec, nothing more — one
> layer further out, and the reason is the same. A compute node is where work
> happens, not where decisions are made. So the monitor does not take
> checkpoints, does not prepare the next stage, does not retry, and does not edit
> the description, even where it is the only thing present and would technically
> be able to. Something that both watches and acts would be deciding on your
> behalf, on a machine you are not at, about a calculation whose worth only you
> can judge.
>
> Three parties, three verbs: **the wrapper activates and execs, the engine reads
> and writes, the monitor watches and tells.** Everything that *decides* runs
> where the user is (`checkpointing.md § 4.1`).

**Rule 1 — one job per directory.** Every job lives in its own directory. A
directory may hold *several inputs* (one per stage of a staged relaxation,
§ 2.3) plus the engine's outputs and restart files, but never inputs for a
**different** job (a different molecule, a different `SystemLabel`). This is
not just tidiness — SIESTA's restart files (`<basename>.XV`, `.DM`, `.CG`)
are unprefixed within the directory, so a second job's `SystemLabel` would
overwrite them. PySCF inherits the same one-job rule through its inlined
trajectory writer and checkpoint file.

**Rule 2 — every file shares one basename.** Each file the generator writes
and a reader later opens carries the same **basename**:

- **SIESTA** — the basename **is** the `.fdf`'s `SystemLabel`.
- **PySCF** — the basename **is** the script's `JOB = "…"` literal.

The basename must be a single token matching `[A-Za-z0-9_-]+` — no spaces,
dots, or slashes. Generators reject anything else at the form / CLI boundary
(`molbuilder/projects.py`, `_NAME_PATTERN`). Across the stages of one staged
relaxation the basename **stays identical** (only parameters change); that is
exactly what lets SIESTA pick up `<basename>.XV` / `.DM` from the previous
stage. (The run *wrapper* accepts a slightly wider set, `[A-Za-z0-9._-]+`, at
`molbuilder/runwrap.py`, because a `SystemLabel` may legitimately carry a
dot.)

### 2.2 The file catalogue

For a job with basename `my-job` (`N` is the auto-advancing run index, § 2.6):

| File | Written by | Read by | Purpose |
|---|---|---|---|
| `my-job.fdf` | Build tab / `molbuilder fdf` | SIESTA | input deck (SIESTA) |
| `my-job.py` | Build tab / `molbuilder pyscf` | Python | input script (PySCF) |
| `my-job.run.sh` | `molbuilder run` | shell / SLURM | wrapper: activates the env and runs the engine (§ 2.6) |
| `my-job.sbatch` | `molbuilder run` (SLURM path) | `sbatch` | outer resource header that inner-execs the `.run.sh` (§ 2.6) |
| `my-job.molwatch.log` | both generators (initial preview) + live frames (PySCF's inlined emitter; SIESTA via the parser-on-stdout path) | the run viewer, `molbuilder watch parse` / `tail` | **canonical trajectory source** — preferred by every reader |
| `my-job-runN.out` | the SIESTA wrapper's stdout redirect | the run viewer (fallback) | SIESTA engine stdout, one file per run index |
| `my-job-runN.pyscf.log` | the PySCF wrapper's stdout redirect | the run viewer (fallback) | PySCF process stdout, one file per run index |
| `my-job.log` / `my-job_geom_<stage>.log` | the generated PySCF script | geomeTRIC parser fallback | geomeTRIC's own optimizer log |
| `my-job_geom_optim.xyz` | the generated PySCF script | trajectory parser fallback | PySCF trajectory frames |
| `my-job.STRUCT_OUT` | SIESTA | next stage / end user | final relaxed coordinates |
| `my-job.ANI` | SIESTA | external trajectory tools | per-step trajectory (SIESTA's own `.ANI` format) |
| `my-job.XV` / `.DM` / `.CG` | SIESTA | next stage (warm restart) | coords+velocities / density matrix / CG state |
| `my-job_optimized.xyz` | the PySCF script | next stage (warm restart) | latest converged geometry |
| `my-job.chk` | the PySCF script | next stage (warm restart) | SCF checkpoint |

The single `.molwatch.log` is **the** canonical trajectory. It is written at
file-emission time — the initial-geometry preview at "step 0" — *before* the
engine starts, so pointing a viewer at the directory works immediately after
you generate the input, before `siesta` / `python my-job.py` has finished one
SCF cycle.

> **Drift corrected (2026-07-27):** the old catalogue listed engine stdout as
> a static `my-job.out` / `my-job.log`. The wrapper now writes a **run-indexed**
> `my-job-runN.out` (SIESTA) and `my-job-runN.pyscf.log` (PySCF); the `.log`
> family is geomeTRIC's own logging (§ 2.6).

**Why PySCF's stdout is `.pyscf.log` and not `.out`.** It used to be `.out`, and
that collided with SIESTA's `.out` — which is not stdout at all, but a structured
text format SIESTA's Fortran writes. The Results tab picks a viewer by suffix, so
a PySCF log got handed to the SIESTA trajectory reader, which rendered garbage.
The rename fixes it three ways: `.log` is honest (the file *is* a capture of
stdout + stderr, not a calculation output), the `.pyscf.` infix pins which engine
produced it for anyone scanning a directory, and the distinct suffix lets the
viewer dispatch correctly. Worth knowing before anyone "tidies" the extension
back.

### 2.3 Multi-stage runs

A staged relaxation (coarse → tight) keeps its stages together, and the
`SystemLabel` / `JOB` basename stays **unsuffixed** — so SIESTA's `.XV` / `.DM`
/ `.CG` restart files transfer cleanly between stages (`MD.UseSaveXV`,
`DM.UseSaveDM`, `MD.UseSaveCG`). Only per-stage *derived* files carry a suffix,
and the codebase uses **two distinct suffix conventions** for two different
paths — do not conflate them:

- **The staged ladder** — the ladder from `task.json` (an engine config carries
  no stage list; that field was **deleted** 2026-08-07 —
  [`engines/stages.md`](?doc=engines/stages.md) § 1.1) rendered by
  `siesta/input.py::render_siesta_stage_fdfs` plus its `.run.sh` stage runner,
  and the `stages_to_jobset` JobSet producer — names each stage's input `.fdf`
  and stdout `.out` **`<label>_<stagename>`**: an **underscore** plus the
  stage's *name* (the default names are `stage1` / `stage2` / `stage3`):

  ```
  bundle-or-dir/
  ├── my-job_stage1.fdf   my-job_stage1.out
  ├── my-job_stage2.fdf   my-job_stage2.out
  ├── my-job.XV / .DM / .CG          ← unsuffixed, carried between stages
  └── my-job.STRUCT_OUT              ← final geometry, after the last stage
  ```

- **The single-stage overlay** (`molbuilder fdf --stage N`, which emits *one*
  stage's `.fdf` on its own) and the **molwatch log** basename use
  **`<label>-stage<N>`**: a **hyphen** plus the stage *number*. The log name is
  produced by `trajectory_log/format.py::molwatch_log_basename(system_label,
  stage)` → `<label>-stage<N>.molwatch.log`. The run decoder's stage regex
  (`parse/dirs/job.py::_STAGE_RE`) keys on this hyphen `-stage<N>` form.

  > **This second convention is being retired, not reconciled** (decided
  > 2026-08-07). **A run's log takes the basename of the deck that produced it** —
  > `<label>_<name>.molwatch.log` — so there is one naming instead of two and
  > nothing to keep in step. The number is the wrong half to keep: every other
  > artifact of a stage keys on the *name*, so the log is the only one that
  > cannot be read back to its stage without opening the description. The
  > reasoning is [`engines/stages.md`](?doc=engines/stages.md) § 7; the naming
  > table is [`execution/project-layout.md`](?doc=execution/project-layout.md)
  > § 4.1. `_STAGE_RE` changes with it. Until that lands, what is written above
  > is what the files are called.

**Two multi-stage execution shapes exist, deliberately:**

- **Per-stage processes (SIESTA run stage-by-stage, and the PySCF `cfg.stage`
  marker path)** — each stage is a separate process invocation writing its own
  `.molwatch.log`. A directory with **more than one** `.molwatch.log` is merged
  by the viewer: all logs are parsed in mtime order (oldest first) into one
  trajectory with a dashed boundary line per stage; live polling pins to the
  newest log.
- **PySCF in-script ladder (`cfg.stages`)** — all stages run inside **one**
  Python process (`for stage in STAGES:`), which writes a **single, unified**
  `<basename>.molwatch.log`. There is no per-stage suffix in this mode.

### 2.4 Resolving a directory — the discovery chain

When a reader (the Watch/run viewer) is handed a **directory** instead of a
specific file, it resolves the trajectory with this chain — first hit wins
(`molbuilder/web/blueprints/watch.py::_resolve_run_directory`):

1. `*.molwatch.log` — if several, the most recently modified.
2. `*.fdf` — parse `SystemLabel`; try `<label>.molwatch.log`, then
   `<label>.out`.
3. `*.py` — grep for a `job_name = "…"` assignment; try `<job>.molwatch.log`,
   then `<job>.log`, then `<job>_geom_optim.xyz`. *(Current generated PySCF
   scripts emit the label as `JOB = "…"`, not `job_name = "…"`, so this step
   matches none of them today — such a directory still resolves via step 1's
   `*.molwatch.log` or step 4's content-sniff. The regex/emit mismatch is a
   code follow-up.)*
4. `run.out` / `siesta.log` / `*.out` / `*_geom_optim.xyz` — content-sniff via
   the trajectory-parser registry.

> **Note on the `.out` / `.log` fallbacks (steps 2–3):** they look for the
> **non-indexed** `<label>.out` / `<job>.log`, not the run-indexed
> `<label>-runN.out` the wrapper now writes (§ 2.6). In practice the
> `.molwatch.log` at step 1 wins for any molbuilder-generated run, so the
> non-indexed fallbacks only match a hand-redirected legacy stdout. A reader
> that wants a specific run's stdout should open `<label>-runN.out` directly.

If nothing matches, the reader returns a clear error naming every filename it
tried. A path that resolves to a regular **file** is loaded directly — the
chain only runs for directories.

### 2.5 The project tree and the canonical topics

A single run directory sits at the bottom of a three-level tree under the
(git-ignored) `projects/` root:

```
projects/
└── <project>/            e.g. "Au-thiol-junctions"
    └── <topic>/          one of the fixed names below
        └── <structure>/  ← the one-job directory (§ 2.1–2.4 live here, flat)
```

The hierarchy is **organisational only**; the innermost `<structure>/` is
exactly the flat one-job-per-directory shape of § 2.1 (no sub-directories, no
nesting of restart files).

Each path segment must match `[A-Za-z0-9_-]+`, and `<topic>` must be one of a
**fixed set of nine** canonical topics (`molbuilder/projects.py::
CANONICAL_TOPICS`). An open topic vocabulary is rejected on purpose: it would
fragment the tree across users and break the "compare the same analysis
across structures" intuition that motivated topic-first ordering.

| Topic | Kind | Used for |
|---|---|---|
| `structure` | storage | flat store of `.xyz` / `.pdb` / `.cif` inputs |
| `pseudopotential` | storage | project-local cache of SIESTA `.psml` pseudos |
| `optimization` | run | geometry relaxation |
| `frequency` | run | Hessian / vibrational frequencies + rigid-rotor–harmonic-oscillator (RRHO) thermochemistry |
| `spectrum` | run | Raman / IR / UV-Vis at an optimised geometry |
| `transport` | run | non-equilibrium Green's function (NEGF) / TranSIESTA device calculations |
| `single-point` | run | energy at a fixed geometry |
| `scan` | run | potential-energy-surface scans |
| `user` | free-form | a workspace with no rules inside it |

> **Drift corrected (2026-07-27):** the source doc said "six" (the run topics
> only). The set is now **nine** — two storage topics (`structure`,
> `pseudopotential`) and a free-form `user` workspace were added.

`molbuilder/projects.py` exposes the tree API: `validate_name`,
`validate_topic`, `project_dir` / `topic_dir` / `structure_dir`,
`ensure_structure_dir` (mkdir -p), `list_projects` / `list_topics` /
`list_structures`, and `find_geom_candidates(project=…)`. The last scans the
tree for reusable geometries matching `*_optimized.xyz`, `*.STRUCT_OUT`, and
`*_geom_optim.xyz` (sorted newest-first) — deliberately **not** bare `*.xyz` /
`*.pdb`, which would sweep up user inputs and noise.

### 2.6 The run wrapper — `.run.sh` and `.sbatch`

`molbuilder run my-job.fdf` (or `.py`) emits a sibling `my-job.run.sh` that
activates the routed conda env and executes the tool
(`molbuilder/runwrap.py::render_run_wrapper`). Routing is by extension:

- **`.fdf` → `molbuilder-siesta`**, run as `mpirun -np N siesta …` (or serial
  if `N < 2`). A `.fdf` that requests **ELPA or GPU** eigensolving is re-routed
  to a third env, **`molbuilder-siesta-gpu`**.
- **`.py` → `molbuilder-pySCF`**, run as `python my-job.py` (OMP-only; the
  script writes its own `.molwatch.log` / `.pyscf.log`).

The wrapper is **plain, readable bash**. Two properties are load-bearing:

- **Activation is a configurable line, not `conda run`.** The wrapper emits an
  activation statement of its own (typically `conda activate <env>`) drawn
  from `runtime_config.require_activation`, so a site can substitute its own
  module-load / venv scheme. (The old illustrative `conda run -n … --no-capture-output`
  example is outdated.)
- **Outputs are run-indexed and never clobbered.** stdout goes to
  `my-job-runN.out` (SIESTA) / `my-job-runN.pyscf.log` (PySCF). The first run
  is `-run0`; **re-running auto-advances** to `max(N)+1` (default since
  2026-06-26), so running the script again never errors and never overwrites a
  prior result. `--force` restarts the sequence at `-run0` (clobbering it);
  `--continue` warm-resumes into the next index (§ 4).

  > **`--force` is retired under the staged layout** (proposed —
  > [`execution/project-layout.md`](?doc=execution/project-layout.md) § 1.2).
  > There, each invocation gets its own `run-<n>/` directory, immutable once
  > written, so there is nothing for a reset to overwrite: a redo is `run-2`.
  >
  > **`run-` becomes a reserved directory prefix there**, and its members are
  > numbers. Nothing else lives under it.
  >
  > **The attempt directory is created when the stage is *prepared*, in
  > Python** — not at submit and not by the wrapper
  > (`project-layout.md § 1.6`). The wrapper is launched *inside* it and is
  > otherwise
  > unchanged: it activates an environment and execs an engine in whatever
  > directory it was handed, which is what
  > [`running-a-job.md`](?doc=execution/running-a-job.md) § 2.2a states in
  > general.
  > A flag whose only purpose is to destroy a previous result has no place once
  > results cannot collide. A flat run directory keeps today's behaviour.

**Two-layer SLURM.** On a cluster the `.run.sh` is the *inner* launcher. The
same `molbuilder run` also emits an outer `my-job.sbatch` — the `#SBATCH`
resource header — which simply `bash`-execs the `.run.sh`. You submit the
outer file: `sbatch my-job.sbatch`. On a workstation there is no `.sbatch`;
you run the `.run.sh` directly (`bash my-job.run.sh`, or backgrounded with
`nohup`). This is one implementation: `runwrap.py::write_run_wrapper(…,
emit_sbatch=True)` → `render_sbatch`. **The JobSet framework reuses this exact
function** (`jobset/prep.py`) rather than reimplementing wrappers — see
`execution/job-system.md`.

molbuilder does **not** manage the launched process. Monitoring is by pointing
the run viewer at the directory (§ 2.4); the resource header's SLURM flags
come from your `molbuilder.json` (§ 6.3).

### 2.7 What the layout does not govern

- **Pseudopotential files** (`<Element>.psml`) sit next to the `.fdf`; their
  names follow the chemical element, not the basename, and are shared across
  jobs (a Au pseudo is the same everywhere). The `--psml-lib` CLI flag copies
  them into the run directory at generate time, but the layout does not
  *require* co-location.
- **Post-processing outputs** (`<basename>.MullikenPop`, `<basename>.bands`,
  PDOS files) follow SIESTA's own naming and inherit the basename
  automatically because they are SIESTA's own output.
- **Analysis pickle / cache files** a user creates after the run are out of
  scope.

---

## 3. The generated-script contract

molbuilder generates engine input that gets **copied** out of the edit
directory into project/run directories and travels onward — often away from
its originating `.molstruct.json` sidecar. To keep provenance and label
metadata attached to the script itself, molbuilder reserves **comment-block
regions** of every generated file for its own use, plus one clearly-marked
zone the user owns.

The payoff: `head -50 my-job.fdf` answers "which molbuilder made this, with
what defaults"; a `.fdf` carries the same region/frozen labels as the sidecar
that produced it (no coordination needed); tools read a stable contract
surface instead of scraping the engine body; and user edits survive
regeneration.

### 3.1 The reserved blocks

Blocks appear top-to-bottom in this order. **Every reserved block is
optional** — a file with none of them is still a valid engine input. Only the
ENGINE BODY is always present (it is the file's actual content, not a
"block"). A tool that needs a specific block refuses cleanly when it is
absent, rather than guessing.

```mermaid
flowchart TD
    H["HEADER  (reserved — defined but not emitted today)"]
    P["PROVENANCE  — who/when/what-defaults"]
    B["BENCH-MARKS  — which fields a tool may override"]
    A["ATOM-METADATA  — regions / frozen / annotations JSON"]
    E["ENGINE BODY  — the actual .fdf / .py content (always present)"]
    U["USER-CUSTOM  — your territory, preserved verbatim"]
    H --> P --> B --> A --> E --> U
```

Every reserved block is delimited by literal marker lines
(`molbuilder/script_emit.py`); parsers find blocks by scanning for them:

```
# === molbuilder <block-name> BEGIN ===
...comment-prefixed content...
# === molbuilder <block-name> END ===
```

**Which generator emits which block** (verified against code — not every
block is emitted by every engine):

| Block | SIESTA `.fdf` | PySCF `.py` | TranSIESTA `.fdf` | wrapper `.run.sh` |
|---|:--:|:--:|:--:|:--:|
| HEADER | — | — | — | — |
| PROVENANCE | ✅ | ✅ | — | ✅ |
| BENCH-MARKS | ✅ | — | — | — |
| ATOM-METADATA | ✅¹ | ✅¹ | ✅¹ | — |
| USER-CUSTOM | ✅ | ✅ | — | ✅ |

¹ Conditional — emitted only when the structure carries labels (§ 3.4).

> **HEADER is reserved but not currently emitted.** The grammar reserves a
> HEADER block and `script_emit.emit_header` exists, but no generator calls it
> today; run instructions instead ride in the engine-body banner. The slot is
> kept in the ordering so tools that *parse* for it degrade cleanly.
> **BENCH-MARKS is SIESTA-only today** — the PySCF `.py` bench block is a known
> gap (§ 3.3).

### 3.2 PROVENANCE — the generation snapshot

A static, always-parseable key/value snapshot of the generator state at
generation time:

```
# === molbuilder provenance BEGIN ===
#   generator-version    git e8a4f81
#   generated-at         2026-06-16T17:30:00-07:00
#   form-config-hash     sha256:7c4d…            # optional
#   resolved-defaults:
#     mpi_np            auto -> 4 (gpu+mps policy)
#     BlockSize         auto -> 256 (10 * 212 atoms / mpi_np, capped pow2)
#     kgrid             1x1x1 (auto-from-cell-vacuum)
# === molbuilder provenance END ===
```

- `generator-version` is the molbuilder git SHA (short); `git log <sha>` in the
  repo recovers the full generator state.
- `generated-at` is ISO-8601 with timezone.
- `resolved-defaults` lists a fixed set of parallel/resource knobs — `mpi_np`,
  `omp_threads`, `BlockSize`, `enable_gpu` (and the PySCF equivalents:
  `use_gpu`, `density_fit`, `threads`, `max_memory_mb`) — each annotated with
  either what the auto-policy chose (`auto -> 4`) **or** the user-set value
  (`user-set -> 256`, or the raw number). It is not a "what the user left on
  auto" list; the knobs always appear, tagged auto-or-user. Scientific
  keywords the user set live in the engine body where the engine reads them,
  not here. For a `.run.sh`, provenance carries form-state at generation time
  only; the *runtime*-resolved values (actual ranks after a hardware probe)
  belong to the wrapper's runtime banner, not here.
- Keys are additive and forward-compatible: PROVENANCE has no version tag, and
  an old parser simply ignores keys it does not know.

### 3.3 BENCH-MARKS — the override surface (SIESTA `.fdf`)

A machine-readable declaration of which engine-body fields a tool (e.g. the
benchmark generator, § `execution/job-system.md`) may override, and within
what limits:

```
# === molbuilder bench-marks BEGIN ===
#   version v1
#   n_atoms             212
#   n_orbitals_est      2120       # 10 * n_atoms, rough DZP heuristic
#   gpu_mode            true
#
#   field BlockSize        anchor=BlockSize        type=pow2  range=[16,256]  default=256
#   field MaxSCFIterations anchor=MaxSCFIterations type=int   default=500
#   field MD.NumCGsteps    anchor=MD.NumCGsteps    type=int   default=200
#   field MeshCutoff       anchor=MeshCutoff       type=float unit=Ry  default=400.0
#   field Diag.Algorithm   anchor=Diag.Algorithm   type=enum
# === molbuilder bench-marks END ===
```

- `version v1` is the block-format version; a higher version makes an old
  parser refuse rather than guess.
- Top-level keys (`n_atoms`, `gpu_mode`, …) are informational.
- `field <name> …` lines declare the **only** parameters a tool may override.
  `anchor=<text>` is the literal token a parser greps for at the start of an
  engine-body line (`^\s*<anchor>\b`) to find the override site — **anchor-based,
  not line-number-based**, so it survives layout drift above it. For a `.fdf`
  the anchor is the SIESTA keyword; for a `.py` it would be the Python
  identifier.
- `type` ∈ `{int, float, str, pow2, enum}` (`pow2` = power of two; `enum` was
  added for `Diag.Algorithm`). `range=[a,b]` and `unit=…` are advisory bounds
  for validating a requested override.

> **Gap:** the PySCF `.py` does not yet carry a BENCH-MARKS block. When it
> lands, its `field` declarations get listed here.

### 3.4 ATOM-METADATA — labels that ride with the script (`.fdf` / `.py`)

Embeds the region/frozen/annotation metadata that a `.molstruct.json` sidecar
carries next to an `.xyz`, so a script copied to a run directory does not
strand it. The payload follows the sidecar schema (see
[`model/structure-molstruct.md`](?doc=model/structure-molstruct.md)); this
block cites that schema rather than duplicating it.

```
# === molbuilder atom-metadata BEGIN ===
# format: molstruct-json/v4
# {
#   "schema_version": 4,
#   "n_atoms_total":  212,
#   "regions":     { "L-electrode": [11,12,…], "R-electrode": [200,…], "bridge": [60,…] },
#   "frozen_atoms": [88, 89, …, 211],
#   "annotations": { … },              # v4 channel; optional
#   "created_by":  "molbuilder modify",
#   "created_at":  "2026-05-20T14:23:00Z"
# }
# === molbuilder atom-metadata END ===
```

**Rules (reconciled to code — the block is now v4, not v3):**

- **Format is `molstruct-json/v4`, `schema_version: 4`.** v4 added the
  per-atom **`annotations`** channel additively (see
  [`model/structure-annotations.md`](?doc=model/structure-annotations.md)).
  The on-disk `.molstruct.json` sidecar itself is now schema **v6**; the
  read-side accepts sidecar/in-body versions `(3, 4, 5, 6)`
  (`molbuilder/parse/sidecars/molstruct.py`).
- **Emission is conditional.** The generator emits the block **only** when
  `regions` **or** `frozen_atoms` **or** `annotations` is non-empty. A
  label-free generation has *no* block at all (not an empty one) — absence is
  the honest signal "this generation had no labels", so it cannot later
  suppress a sidecar the user adds afterward. (The old rule said regions/frozen
  only; the annotations channel was added to the trigger with v4.)
- **Indices are 0-based** (matching the sidecar and `Structure.regions` /
  `Structure.frozen_atoms`). SIESTA's engine-body `%block Geometry.Constraints`
  is **1-based** by SIESTA convention. The two coexist in one file on purpose;
  a tool must not assume one indexing for both.
- **`structure_hash` is not emitted in-body.** The metadata and the
  coordinates are written by the same generator pass, so they cannot drift
  apart — a hash would be tautological.
- **In-body wins over the sidecar.** When a `.fdf` / `.py` with an
  ATOM-METADATA block sits next to a `.molstruct.json`, downstream code reads
  the in-body block and ignores the sidecar
  (`_shared.py::apply_companion_labels_if_present` runs before the sidecar
  branch of `apply_sidecar_if_possible`). The sidecar is the fallback for plain
  `.xyz` loads and for pre-contract scripts.

### 3.5 USER-CUSTOM — your territory

A zone molbuilder reads during regeneration only to learn where it is, then
copies **byte-for-byte** into the new output:

```
# === molbuilder user-custom BEGIN ===
# Your own additions go here. molbuilder preserves this section verbatim.
# === molbuilder user-custom END ===
```

molbuilder does not validate its contents (engine-invalid text there will be
rejected by the engine, not by molbuilder). The block may be missing; on
regenerate an empty one is emitted.

### 3.6 Versioning and what a tool may assume

Each structured block versions **independently**: BENCH-MARKS carries
`version v1`, ATOM-METADATA carries `format: molstruct-json/v4`, PROVENANCE is
additive-keys-only (no tag), HEADER is free-form prose. There is **no
autodetection and no silent upgrade** — a parser reads the version tag and
either handles it or refuses, pointing the user at "regenerate with the
current molbuilder". Given a conforming file, a tool may assume: PROVENANCE
answers who/when/what-defaults; BENCH-MARKS lists the overridable fields and
their bounds; ATOM-METADATA round-trips (its dict feeds the same
`apply_to_structure` path the sidecar uses); USER-CUSTOM survives
regeneration.

---

### 3.7 The template's item blocks — one file that is both the reference and the source

*Specified by the user, 2026-08-07.* The **template** (`engines/stages.md § 4`)
is the science backbone a generating tab writes: everything a script owns, with
the value the user set or the default they did not touch. This is its format.

**The rule in one line: every item appears exactly as it will be copied, wrapped
in markers, with its explanation beside it.**

```fdf
# === molbuilder item mesh_cutoff BEGIN ===
#   field mesh_cutoff  anchor=MeshCutoff  type=float  unit=Ry
#                      range=[50,2000]  default=300.0  group=stage
#   Mesh cutoff — the real-space integration grid, in Ry.  Higher is finer
#   and slower; convergence is checked, not assumed.
#   Tier ladder: 150 screening · 300 publishable · 500 tight.
MeshCutoff   300.0 Ry
# === molbuilder item mesh_cutoff END ===
```

**Each block carries a `field` declaration line and then prose**, and the
declaration is **the grammar § 3.3 already defines** — `field <name>
anchor=… type=… range=[a,b] unit=… default=…`, `type ∈ {int, float, str, pow2,
enum}` — extended with `group=` (the `workflow_group`) and `choices=` for enums.
Not a parallel notation: the same shape, in the same file, parsed the same way.

**That declaration is what makes the template enough on its own.** A surface
holding this file needs nothing else to work:

| From the declaration | What it lets a surface do |
|---|---|
| `type`, `choices` | pick the control — number box, dropdown, checkbox |
| `range`, `unit` | bound it and label it |
| `default` | show what "untouched" means, and **tell whether the payload is the user's or the default** — they are the same value or they are not, so no extra marker is needed |
| `group` | decide whether the *vary per stage* box starts ticked |
| `anchor` | **find the substitution site** when a stage overrides the item — anchor-based, not line-number-based, so it survives layout drift above it (§ 3.3's own rationale) |

**So constructing `task.json` from the UI needs only the template.** When a user
ticks an item, the tab already knows its type, its bounds and its default from
the block itself — enough to render the per-stage cells and to validate what is
typed into them, without asking the server what the field is. That is what makes
the whole package portable in the sense `project-layout.md § 2.1` means: the
calculation travels with its own catalogue.

Four properties, and each buys something specific:

1. **The payload is what lands in the final deck, and that is a *checked*
   property.** *(Decided 2026-08-07 — this rule was written as "producing that
   deck is a scan and a copy, never a re-render", and three fields of the
   shipped schema cannot be served that way.)*

   `prep` rebuilds an ordinary config from the blocks, resolves the stage
   (`stages.md § 4`), and renders through the **same emitter every other deck
   goes through**. A test then asserts that for every item no stage overrode,
   the rendered line is byte-identical to the template's payload. So *a value
   cannot change shape between what a person read and what the engine got*
   survives as the guarantee — it is enforced by a guard rather than by the
   copy being literal.

   > **Why the literal copy could not hold.** A stage that overrides
   > `relax_type` from `CG` to `Verlet` moves the step budget's site from
   > `MD.NumCGsteps` to `MD.FinalTimeStep` — the *anchor itself* is chosen by
   > another field's value, so there is no fixed site to substitute at.
   > `spin_total` writes **two** lines (`Spin.Fix` + `Spin.Total`) from one
   > field. And ten fields write **no** line at their defaults. Re-rendering
   > handles all three for free; substitution handles none of them.
   >
   > The alternative considered and rejected was to allow only
   > single-anchor, always-emitted fields to be varied — which would make
   > *which settings may vary* a fixed list again, the exact arrow § 1.2 of
   > [`engines/stages.md`](?doc=engines/stages.md) exists to reverse.

   **`anchor=` therefore stops being load-bearing.** It still says where the
   value lands, which is worth knowing and is what BENCH-MARKS uses it for
   (§ 3.3); nothing reads it to produce a deck.
2. **The markers carry the field's name, and the declaration carries its
   value.** That is what lets `prep` walk the file and rebuild an ordinary
   `SiestaConfig` — *without an fdf parser*. **This is the whole reason the
   design works**: nothing in molbuilder can read an `.fdf` back into a config,
   and with named blocks nothing needs to.

   The declaration gains **`value=`** beside `default=`, and the reader takes
   the value from there rather than from the payload. Same grammar, one more
   key — and it is what makes the read total: a payload can be absent (the
   field emits no line), several lines (`spin_total`), or a `%block`, and none
   of those change how the value is read. `default=` stays, because the pair
   is what tells a surface whether the user set this or left it alone.
3. **The block holds what we know about the item** — what it is, what it is
   validated against, how the engine uses it, any hint worth having. It is
   generated from the field's own metadata (`web/form-schema.md § 1a`:
   `help`, `range`, `unit`, `choices`, `engine_key`, `workflow_group`), so the
   documentation and the form are the same source and cannot drift.
4. **Every allowed, validated item has a place in the file** — not only the ones
   a user touched. The template is the engine's whole surface, instantiated.

   > **This is the premise, not an aspiration, and it settles a class of
   > question before it is asked.** The template is built from what molbuilder
   > *knows*: every field the engine's config declares, every one validated, each
   > with what we have learned about it. There is no "what about a keyword we do
   > not model" case to design for — a keyword molbuilder does not model is
   > **work not done yet**, and the answer is to model it, not to invent a slot
   > for it. § 3.5's USER-CUSTOM is not that slot either: it is a zone copied
   > **byte-for-byte and never validated**, for a user's own text.



**So one artifact serves four readers.** A person opens it and learns the
calculation *and* the reasoning. The UI renders it. `prep` extracts the deck.
The validator gets a real config out of it. That is why it is worth being a
real, readable `.fdf` rather than a serialised blob — which was the alternative,
and it would have been none of those four things but the last.

> **The marker convention is the one that already ships**, not a new one:
> `# === molbuilder <name> BEGIN ===` / `# === molbuilder <name> END ===`
> (`script_emit.py`), the same shape as HEADER, PROVENANCE, BENCH-MARKS and
> USER-CUSTOM above. The comment character is the engine's — `#` for `.fdf` and
> `.py`; an engine whose comment is `!` uses `!`.
>
> **Multi-line items come free.** A `%block ChemicalSpeciesLabel` or a coordinate
> block is several lines, and the markers delimit it exactly as they delimit a
> one-liner. A format that keyed on "one line per setting" could not have carried
> those at all.

**Both of the questions this format was leaving open are answered above**
(2026-08-07): `engine_key` is not an anchor for every field, and ten fields emit
no line at their defaults — and re-rendering makes both harmless, which is why
the rule changed rather than the schema.

**How a stage's override lands:** `prep` reads every block's `value=` into an
ordinary config, applies the stage's `overrides` on top (`stages.md § 4`), and
renders. A field the stage did not name keeps the template's value — that is
`overrides ⊆ varies` seen on disk, the quiet cell in the table.

> **Two things this format leaves to be decided when it is built**, recorded
> rather than guessed:
>
> - **Items whose default is *derived*, not literal.** Some defaults come from
>   the compute resources — `BlockSize` from the rank count is the example — and
>   the rank count is not known when the template is written. **These still get a
>   block**, and the rule is (user, 2026-08-07): **an explicit user setting is
>   honoured; otherwise the value is derived at generation, and at generation
>   time both are available.** So the block's declaration says the default is
>   derived rather than naming a number, and the payload is either what the user
>   set or supplied by `prep`. *How* that is spelled — a `default=derived` marker
>   with the payload line absent until `prep` writes it, or a placeholder payload
>   — is the remaining detail, and it is small.
>
>   ⚠ **`Diag.Algorithm` is not one of these**, and it was wrong to file it here.
>   It is **an ordinary explicit option**: a user chooses it, it gets a plain
>   block with a plain payload, and nothing derives it. **Whether the engine can
>   deliver the choice is the engine's business, not the generator's** — an
>   `.fdf` asking for an ELPA solver that the build does not have fails when
>   SIESTA runs, and that is the correct place for it to fail. The generator does
>   not check, and does not need to.
> - **A hand-edited payload that no longer matches its block name.** The file is
>   meant to be edited; someone will change `MeshCutoff 300.0` to `400.0` by
>   hand. Reading the value back out is what makes that work — but it also means
>   the payload, not the metadata, is the authority. Worth stating explicitly.
> - ~~**A user-set value that equals the default** is indistinguishable from an
>   untouched one~~ — **closed, no consequence** (user, 2026-08-07). **Every item
>   is explicitly instantiated**: § 7 of [`engines/stages.md`](?doc=engines/stages.md)
>   already requires *every value the description determined, written rather than
>   left to an engine default*, and § 3.7 extends that to the whole surface — every
>   allowed item has a block, with a payload. So a deliberate `300.0` and an
>   untouched `300.0` produce **the same bytes** in the deck. There is nothing for
>   the distinction to change.
>
>   **And the payload being the authority makes that hold over time, too.** If a
>   molbuilder release changes a field's default, an existing calculation
>   regenerates from *its template's payload*, not from the new schema — so the
>   number a user saw is the number they keep. A design where the deck was rebuilt
>   from defaults would have made this distinction load-bearing and dangerous;
>   this one makes it moot.
>
> **And one relationship to keep straight:** these item blocks and BENCH-MARKS
> (§ 3.3) share a grammar and answer different questions. The item blocks live in
> the **template** and declare **every** field. BENCH-MARKS lives in a
> **generated deck** and declares the subset a *tool* may override. **Both must be
> emitted from the same field metadata** (`web/form-schema.md § 1a`) — two hand-
> maintained copies of `default=` would drift, and the drift would be silent.

## 4. Warm & cold restart

This is the **per-job** resume contract, and it is designed to be the same to
the user across engines even though the machinery differs (SIESTA resumes
inside its binary from `.DM`/`.XV`; PySCF resumes via an `if exists →` branch
the generated script contains). Its extension to multi-stage / multi-job sets
— and the rule that *molbuilder informs but the user decides to continue* — is
owned by `execution/job-system.md`.

### 4.1 The four behaviors

| Behavior | What happens |
|---|---|
| **Project ID** | Every script declares its ID in one literal (`SystemLabel` / `JOB = "…"`). This ID keys all warm files as `<ID>.<ext>`. |
| **Warm-restart (auto)** | If warm files named by the ID exist in the directory, the engine resumes from them — no flag. Absent files ⇒ clean cold start. |
| **`--continue`** | Same as auto, but *asserts* the warm files must be present: if none exist it prints "…starting cold by necessity" rather than silently cold-starting. |
| **`--cold`** | Forces a clean start regardless of on-disk state. Warm files are **moved aside** (never deleted) into `<basename>-restart-aside-<UTC-timestamp>/`. |

The critical safety property of `--cold`: the move-aside glob **must cover
every file the warm-start branch reads**, or `--cold` silently leaks prior
state into a "clean" run. This is enforced per engine and pinned by tests
(`molbuilder/runwrap.py::_cold_restart_aside_block`).

### 4.2 Per-engine warm-file inventory

Every file below is in that engine's `--cold` move-aside glob.

**SIESTA (13 suffixes):** `.DM` (density matrix), `.CG` (CG optimizer state),
`.XV` (coords+velocities), `.LWF` (Wannier), `.ZM` (Z-matrix), `.Bonds`,
`.PARTIAL`, `.EIG`, `.HSX` (Hamiltonian+overlap, TranSIESTA restart), `.WFSX`
(saved wavefunctions), `.STRUCT_NEXT_ITER` (next-iter geometry), `.TSHS` /
`.TSDE` (TranSIESTA self-energy H / NEGF density). SIESTA reads these itself
when the matching `MD.UseSave*` / `DM.UseSaveDM` flags are set; the wrapper
only moves them aside for `--cold`.

**Those three flags are one group, and one field sets them.** A description
carries `restart` — `clean` or `continue` — and the renderer expands it into
`DM.UseSaveDM` / `MD.UseSaveCG` / `MD.UseSaveXV` together
(`run-identity.md § 4` rule 2). They are not individually settable, because the
two ways they can disagree with each other are both silent: the deck claiming a
resume the engine will not perform, and warm files sitting unread beside a run
that was told to start clean. The group is declared in code as
`config/siesta.py::SIESTA_RESTART_GROUP`, and PySCF's counterpart —
same idea, generated control flow instead of declared keys — as
`config/pyscf.py::PYSCF_RESTART_GROUP`.

**PySCF (5 files):** `<JOB>.chk` (SCF init guess), `<JOB>_optimized.xyz`
(latest converged geometry), `<JOB>_geom_optim.xyz` (geomeTRIC trajectory),
`<JOB>_geom_optim.tmp`, `<JOB>_geom.tmp` (geomeTRIC temporaries). Unlike
SIESTA, the *generated PySCF script* contains the warm-restart logic
explicitly:

```python
# SCF init guess from a prior checkpoint, if present:
mf.chkfile = _mb_outfile(JOB + ".chk")
_chk = _mb_outfile(JOB + ".chk")
if _os.path.exists(_chk) and _os.path.getsize(_chk) > 0:
    mf.init_guess = "chkfile"
    print(f"[molbuilder] continuation: SCF init guess from {_chk}")

# Geometry resume from a prior optimization, if present:
#   <JOB>_optimized.xyz overrides the literal _atom_block before gto.M(...)
```

> **Lesson pinned in code:** any new warm-restart hook must land its `--cold`
> glob entry **in the same commit** as its read-side. A parity test
> (`_PYSCF_WARM_RESTART_INVENTORY` in `tests/test_runwrap.py`) fails if a hook
> gains a read-side but forgets the glob.

### 4.3 Project-ID extraction

For `--cold` to move aside the *right* files, the wrapper must read the ID
from **inside** the script — `<basename>-stage2.fdf` may carry `SystemLabel
foo` (not `foo-stage2`). At runtime the wrapper `awk`s the `SystemLabel`
(SIESTA) or `JOB = "…"` (PySCF) line, **sanitizes** the value to
`[A-Za-z0-9._-]` before interpolation (blocking shell injection from a hostile
script), and falls back to the wrapper basename if it cannot parse one. The
`--cold` glob uses **both** the ID-derived name and the wrapper basename, so a
job where `SystemLabel == basename` is covered either way.

### 4.4 The status banner

Before the engine starts, the wrapper prints one MODE line so the user sees
what is about to happen. The wording is engine-agnostic (the same mode means
the same thing for SIESTA and PySCF):

| Mode line | Meaning |
|---|---|
| `initial-run (clean state)` | no warm files, no flags — cold from the script literal |
| `WARM-RESTART (silent; engine will load existing <files>. Pass --cold to discard them.)` | warm files present, no flag — auto-resume |
| `WARM-RESUME (--continue; engine will load <files>)` | `--continue` + warm files present |
| `WARM-RESUME REQUESTED but no prior state found -- starting cold by necessity` | `--continue` but no warm files — degraded to cold |
| `COLD (--cold; warm-start files moved aside)` | `--cold` — warm files moved to `-restart-aside-<UTC>/` |

(The flag spellings: `--continue` / `-c`; `--force` / `-f` resets the run
index to `-run0`; `--cold` / `--from-scratch`.)

---

## 5. The workflow handoff bundle

> **Vocabulary:** the object that carries a **finished run into the next
> calculation** is the *handoff bundle*. Plain "bundle" belongs to the JobSet
> framework (a bundled batch of jobs, `execution/job-system.md`) — do not use
> the bare word for this object.

### 5.1 Purpose

molbuilder writes an input script into a run directory; the run produces
outputs (SIESTA `.XV`, a PySCF `_optimized.xyz`). Historically the originating
`.xyz` + `.molstruct.json` were **not** copied into the run dir, so when a user
wanted to continue — Transport (needs `L`/`R`/`bridge` regions), a restart from
converged coords, a spectrum at the optimised geometry — the next stage had no
clean source for the labels that defined the run. The handoff bundle closes
that gap. It fuses:

1. the **final structure** (coords + elements) read from the converged engine
   output, with
2. the **labels** (regions, frozen atoms) from the originating script's in-body
   ATOM-METADATA (§ 3.4), and
3. **provenance** from that script.

Materialised to a destination, it writes an `.xyz` + `.molstruct.json` pair —
a format the next tab's existing load path already understands. No new load
primitive is needed downstream.

### 5.2 The handoff object

Assembly returns a typed, frozen `BundleResult`
(`molbuilder/parse/types.py`; produced by
`molbuilder/parse/dirs/bundle.py::BundleDirParser.parse(run_dir)`):

```python
@dataclass(frozen=True)
class BundleResult(ParseResult):      # + base: schema_version, parsed_at, parser_name, source
    structure:         Structure                 # final coords + elements
    cell:              Optional[...]             # lattice, when present
    regions:           Dict[str, List[int]]      # 0-based; may be {}
    frozen_atoms:      List[int]                 # 0-based; may be []
    source_engine:     Literal["siesta", "pyscf"]
    source_script:     Optional[str]             # abs path to the .fdf / .py that fed extraction
    final_coords_from: Literal["xv", "fdf-initial", "py-opt", "py-initial"]
    block_schema_versions: ...                   # the ATOM-METADATA versions seen
    notes:             List[str]                 # never None; diagnostics
```

`final_coords_from` is load-bearing: it tells a consumer whether the bundle
reflects a **converged** optimization (`"xv"`, `"py-opt"`) or **fell back** to
initial coordinates because the optimization output was missing
(`"fdf-initial"`, `"py-initial"`). `notes` carries non-fatal diagnostics
(schema-version mismatch, fallback reason, missing provenance) and is always a
(possibly empty) list.

> **Naming reconciled to code:** the old contract called this `RunBundle` with
> `user_custom_lines` + `provenance` fields and a free `assemble_from_run_dir`
> function. The shipped object is **`BundleResult`** (user-custom and
> provenance live on the sibling **`ScriptResult`**, the per-script
> extraction), and the entry point is the class method
> **`BundleDirParser.parse`** (the free `_assemble_from_run_dir` is private and
> returns a dict).

The per-script extraction feeding the bundle is `ScriptResult`
(`ScriptSourceTextParser.parse`), whose fields distinguish three states on
purpose: `None` = block absent/unparseable, `[]`/`{}` = block present but
deliberately empty. The bundle-layer convenience `_extract_script_source(text)
→ dict` is re-exported for back-compat as
`molbuilder.script_emit.extract_script_source`.

### 5.3 Source priority and conflict policy

**Final coordinates — first hit wins:**

| Engine | Source | Mark | When |
|---|---|---|---|
| SIESTA | `<SystemLabel>.XV` | `xv` | any run that wrote `.XV` (`SystemLabel` read from the in-body directive, not the filename); falls back to `<fdf-stem>.XV`, then to the sole `*.XV` in the directory (a `note` records the fallback; multiple `*.XV` are left ambiguous and drop to initial coords) |
| SIESTA | `.fdf` initial coords | `fdf-initial` | `.XV` absent/malformed — bundle still emits; `notes` records "NOT converged geometry" |
| PySCF | `<JOB>_optimized.xyz` | `py-opt` | geom-opt success (`JOB` read from the `JOB = "…"` line) |
| PySCF | `.py` initial atom-block | `py-initial` | `_optimized.xyz` absent — bundle still emits; `notes` records the fallback. Only the generator's whitespace atom-block is parsed; a hand-written list-of-tuple `.py` must be re-rendered through molbuilder first |

**Labels:** in-body ATOM-METADATA is authoritative; where a `.xyz` load has
both a sibling script *and* a `.molstruct.json`, **in-body wins** (§ 3.4).

**Conflict rules:** multiple scripts in a dir ⇒ pick the largest by atom count
(tie: lexicographic); both a `.fdf` **and** a `.py` present ⇒
`BundleError("dir contains both …; ambiguous")`; the script's
`n_atoms_total` not matching the final structure's atom count ⇒ `BundleError`
(no silent reconciliation). A left-handed cell (det < 0) assembles but adds a
loud `notes` warning (the check now lives in
`molbuilder/parse/dirs/_assembler_helpers.py`).

### 5.4 Materialisation and errors

```python
def write_bundle_as_handoff(bundle, target_dir, *, stem,
                            overwrite=False) -> Tuple[Path, Path]:
    """Write <target>/<stem>.xyz + <target>/<stem>.molstruct.json."""
```

(`molbuilder/bundle_writer.py`.) Each file is written atomically (tmp + fsync +
`os.replace`, mirroring the sidecar writer); the **pair** is best-effort
(`.xyz` lands first, then the sidecar). With `overwrite=False` (default) it
raises **`BundleWriteError`** if **either** the `.xyz` or the
`.molstruct.json` already exists at that stem — stricter than checking the XYZ
alone, because overwriting a stale sidecar that points at a different XYZ would
corrupt the structure↔sidecar pairing invariant. (Note the two distinct
exceptions: `BundleError` for *assembly* problems, `BundleWriteError` for
*write* problems.) The `overwrite=False` check is not a lock: two writers
racing on the same target stem can each pass the existence check before either
writes, so the pair could land mixed (`.xyz` from one, sidecar from the
other). The per-file atomic rename keeps each file internally consistent, and
the sidecar's `structure_hash` lets the next loader detect the mismatch — but
a UI that writes a handoff SHOULD warn before targeting a stem that already
exists.

On schema: the reader handles ATOM-METADATA **v3 and v4**; a version below 3
raises `BundleError` ("re-render with current molbuilder"); above 4 loads with
a `notes` warning ("molbuilder expects 4").

### 5.5 Surfaces

- **Web** — the Results panel's "Bundle for next stage →" button posts to
  `POST /api/results/bundle` (`molbuilder/web/blueprints/results.py`); the
  frontend wiring is `lib/results/bundle-handoff.js`. The endpoint resolves the
  target inside the project sandbox and then navigates the sidebar to it so the
  new pair appears without manual refresh.
- **CLI** — `BundleDirParser.parse` + `write_bundle_as_handoff` are the same
  entry points a script or future CLI command calls directly.

---

## 6. The shared data vocabulary

If two subsystems name the same concept, they use the name defined here; every
persisted artifact follows one schema convention. This section is the
"what is this called, system-wide?" reference — maintained because the names
*did* drift once (a job-set field read `omp`/`walltime` while every other
exchange file said `cpus_per_task`/`time`). One language prevents that.

### 6.1 Persisted artifacts

| Artifact | File | Schema string | Authoritative code | Key top-level fields |
|---|---|---|---|---|
| User config | `molbuilder.json` / `.molbuilder.json` | *(validated, no `@N`)* | `runtime_config.py` | `scheduler{kind,directives,gpu,defaults,mem_model,routing}`, `execution`, `script_generation`, `envs` |
| Detected environment | `environment.json` | `molbuilder/environment@1` | `bench/environment.py` | `scheduler`, `topology`, `site` |
| Benchmark manifest | `bench-manifest.json` | `molbuilder/bench-manifest@2` | `bench/generate.py` | `points.{cpu,gpu}` |
| Benchmark result | `bench-result.json` | `molbuilder/bench-result@1` | `bench/result.py` | `points`, `choice`, `recommend` |
| Job-set plan | `job-set.json` | `molbuilder/job-set@1` | `jobset/model.py` | `name`, `engine`, `kind`, `shared`, `jobs[]` |
| Task description | `task.json` | `molbuilder/task@1` | `task.py` | `engine`, `shape`, `run`, `structure`, `varies`, `stages[]` — **what changes**; what does not is in `<id>.fdf.template` |
| Workflow handoff | `<stem>.xyz` + `<stem>.molstruct.json` | *(sidecar pair, bare-int `schema_version` = 6)* | `bundle_writer.py`, `sidecars/molstruct.py` | geometry; `regions` / `frozen_atoms` / `structure_hash` |
| Checkpoint archive | `.binsnapshots/<sha>/MANIFEST` | *(3-col `<sha256>  <bytes>  <key>`)* | `checkpoint.py` | — |
| Run launch record | `<attempt>/run.json` | `molbuilder/run-launch@1` | *(proposed — `project-layout.md` § 1.6)* | `mode`, `command`, `job_id`, `launched_at`, `continued_from` |
| Checkpoint config | `.mbcheckpoint.json` | `molbuilder/checkpoint-config@1` | `checkpoint.py` | `engine`, `archive_globs` |
| Decoded run | *(served, not written to disk)* | bare-int `schema_version` | `parse/dirs/job.py` | see below |

> **The MANIFEST's third column is a repo-relative path, not a basename**
> (2026-08-06). It was a bare basename, and the parser rejected a separator. It
> could not be: `.gitignore` receives the raw archive globs (`*.DM`), and a
> gitignore pattern with no slash matches at **every** level, while the archive
> walk matched only the top one — so a big binary in a subdirectory was
> gitignored *and* unarchived, in no snapshot at all, and silently absent after a
> restore. The key space **widened**: a bare basename is a valid relative path,
> so every archive written before this reads unchanged. What stays rejected is
> anything that could direct a restore out of the run directory — absolute paths,
> `..` or `.` components, empty components, backslashes, and dot-prefixed
> components (which would reach `.git` or `.binsnapshots`). Pinned by
> `tests/test_checkpoint_nested_layout.py` and the parser cases in
> `tests/test_checkpoint_manifest_format.py`; the reasoning is
> [`execution/checkpointing.md`](?doc=execution/checkpointing.md), L2.

> **The decoded run is not a file.** `decode_run_dir(run_dir)` returns an
> in-memory `JobResult` dataclass served to the Results tab and consumed by
> `jobset/runstatus.py`; nothing writes a `decoded.json`. Its bare-integer
> `schema_version` predates the `@major` convention — do not copy that for
> anything new. Its full field set and the run-monitoring story live in
> `execution/running-a-job.md`.

**Schema-string convention:** `molbuilder/<name>@<major>`. A reader checks the
**major only** — tolerating same-major minor bumps, rejecting a different
major — through the single shared helper `molbuilder/persist.py`
(`schema_major`, `check_schema_major`, `read_json`, `write_json`), now adopted
by `bench/environment.py`, `bench/result.py`, and `jobset/model.py` (it was
hand-rolled three times with a subtle missing-`@` inconsistency before). New
persisted artifacts must use it. The two bare-integer exceptions
(`.molstruct.json` = 6, the decoded run = 1) predate the convention.

### 6.2 The parameter vocabulary — config ↔ scheduler

There are **two layers** with a deliberate, documented translation between
them; within a layer, one concept has exactly one name.

- **config layer** — the scientific dataclasses the user sets (`SiestaConfig` /
  `PySCFConfig`), vocabulary tuned for the scientist.
- **exchange / scheduler layer** — the persisted artifacts (`job-set.json`,
  manifests) and the SLURM flags they become, tuned for the scheduler.
  Persisted files and `jobset.Resources` use this column.

| Concept | config-layer name | exchange / SLURM name | translated at |
|---|---|---|---|
| MPI ranks | `mpi_np` | `mpi_np` → `-n` | *(same name)* |
| OMP cores / rank | `omp_threads` | **`cpus_per_task`** → `-c` | `siesta/stages.py::stages_to_jobset` |
| Walltime | `defaults.time` | **`time`** → `-t` | — |
| Memory | `max_memory_mb` / `defaults.mem` | `mem` → `--mem` | `render_sbatch` (estimate) |
| Whole-node | `gpu.exclusive` | `exclusive` → `--exclusive` | — |
| Partition | `directives.partition` | `partition` → `-p` | resolved from `domain` |
| QoS | `directives.qos` | `qos` → `-q` | resolved from `domain` |
| Routing domain | `routing[].name` / `execution.domain` | `domain` (in `jobset.Resources`) | `--domain` → `-p`/`-q` |
| GPU request | `enable_gpu` + `diag_algorithm` | `gres` → `--gres` | derived from `.fdf` + GPU type |
| Eigensolver | `diag_algorithm` (`ScaLAPACK` / `ELPA-1STAGE` / `ELPA-2STAGE`) | `.fdf`: `Diag.Algorithm` | `render_fdf` |
| Non-convergence policy | `on_nonconvergence` (`proceed`/`continue`/`halt`) | `dep_kind` (`afterany`/`afterok`) | `stages_to_jobset` |
| Warm-retry budget | `continue_retries` (1–5) | `continue_retries` — **not a SLURM flag** | `stages_to_jobset` |

> **One row in this table becomes no scheduler flag at all, and it is not an
> oversight.** `continue_retries` rides `jobset.Resources` because that is the
> road every *"field the deck never carries"* already rides
> (`engines/stages.md § 5`, third row, which groups it with `mpi_np` and
> `omp_threads`) — but where those two resolve to `-n` and `-c`, this one is
> **baked into the wrapper at install time** (`running-a-job.md § 3.5`) and
> never reaches an `sbatch` line.
>
> Decided 2026-08-07. The alternative was a second road from a stage to its
> wrapper, which would have meant two ways for a per-job value to travel and a
> mapping maintained by hand — the thing § 5's *"the routing is derivable, never
> a second list"* exists to prevent. Written down here **and** in `Resources`'
> docstring, because a field sitting in a class called *a per-job scheduler ask*
> is otherwise an invitation to render it into a directive.

**The translation rule:** persisted/exchange files use the exchange name; a
*producer* maps config → exchange at its boundary (e.g. `stages_to_jobset`
maps `SiestaConfig.omp_threads` → `cpus_per_task`). Never mix the two within
one file. `render_sbatch` is a *consumer* — it receives `cpus_per_task`
already translated and does not re-derive it from `omp_threads`. (In the
wrapper these are two distinct knobs that *coincide* on SLURM, where `-c` sets
`SLURM_CPUS_PER_TASK`, which the wrapper uses as its OMP default — the "one
concept, one name" framing here is the SLURM mapping, not a Python rename.)

The `jobset.Resources` dataclass holds exactly **seven** fields — `domain`,
`time`, `exclusive`, `mem`, `gres`, `mpi_np`, `cpus_per_task`. `partition` and
`qos` are **not** `Resources` fields; they are config `directives.*` resolved
from `domain` by the submit engine. `dep_kind` is a per-**job** edge field, not
a resource.

### 6.3 Identifier & path conventions — every name in the system

**This table is the cross-layer authority.** Other documents explain *why* a
name is shaped as it is; if any of them disagrees with a row here, this row wins
and the other is a bug.

#### The four separators, and what each one means

Read a molbuilder filename left to right and the punctuation tells you the
structure. That is not decoration — it is what lets a reader (or a glob, or a
parser) split a name without knowing what is in it.

| | Means | Example |
|:-:|---|---|
| `_` | **joins parts of one name.** Neither side names the thing on its own | `bdt_au_relax`, `<id>_<stage>`, `01_coarse` |
| `-` | **attaches a counter or qualifier** to a name that stands alone without it | `run-0`, `bench-G1K4C6`, `<id>-restart-aside-<UTC>` |
| `.` | **introduces a type suffix** — what the file *is* | `.fdf`, `.XV`, `.molwatch.log`, `.fdf.template` |
| `/` | **separates levels** — of a path, or of a history ref | `01_coarse/run-0/`, `<id>/<stage>/<UTC>` |

**This is why a stage name may not contain a hyphen** (`engines/stages.md § 2`):
a hyphen announces *"a counter follows"*, so one inside a name makes it
impossible to tell where the name ends. Names use `_`; the system uses `-` to
append to them.

#### Character sets

| What | Set | Fixed by |
|---|---|---|
| **run id** | `[A-Za-z0-9_-]+`, single token | `run-identity.md § 3` — normalised **once**, refused rather than truncated |
| **stage name** | `[A-Za-z0-9_]+` — **no hyphen** | `engines/stages.md § 2` |
| **project-tree path segment** | `[A-Za-z0-9_-]+`, topic from the fixed nine | § 2.5 |
| **basename the wrapper accepts** | `[A-Za-z0-9._-]+` — wider, because a `SystemLabel` may carry a dot; sanitising here also blocks shell injection | § 4.3 |

#### Files

| What | Flat | Hierarchical |
|---|---|---|
| **description** | `task.json` | `task.json` (at the calculation root) |
| **deck template** | `<id>.fdf.template` | `<id>.fdf.template` (at the root) |
| **deck** | `<id>_<stage>.fdf` | `<id>.fdf` — inside `<seq>_<stage>/` |
| **wrapper** | `<id>_<stage>.run.sh` / `.sbatch` | `<id>.run.sh` / `.sbatch` |
| **stdout** | `<id>_<stage>-run<N>.out` | `<id>.out` — inside `run-<n>/` |
| **trajectory log** | `<id>_<stage>.molwatch.log` | `<id>.molwatch.log` |
| **warm-restart state** | `<id>.XV` `.DM` `.CG` — **shared, unsuffixed** | `<id>.XV` `.DM` `.CG` — inside the attempt |
| **launch record** | — | `run.json` — inside the attempt |

**One rule generates the whole right-hand column: a name says what its location
does not.** In the hierarchy the directory already names the stage, so the deck
does not repeat it; flat has no such directory, so the suffix carries it. **The
trajectory log takes the deck's basename in both shapes**, which is why it needs
no convention of its own.

#### Directories

| What | Form | Why that shape |
|---|---|---|
| **calculation** | `<id>/` | the folder *is* the id, so a listing identifies its contents (`run-identity.md § 3`) |
| **stage** *(hierarchical)* | `<seq>_<stage>` — zero-padded to two digits | `seq` **orders**, so it pads and sorts; assigned once and never reassigned (`project-layout.md § 4.2`) |
| **attempt** *(hierarchical)* | `run-<n>` — **not** padded | a counter of invocations that happened, not a designed sequence; `run-` is reserved and its members are numbers, full stop |
| **benchmark** | `bench/` inside the stage it measures | a benchmark nests in what it measures (`project-layout.md § 3`) |
| **trial** | `bench-G<gpus>K<ranks-per-gpu>C<cores>` | a sweep has no order, so the name carries **what was tried** — which is what lets `summarize` map a directory back to its point |
| **warm state moved aside** | `<id>-restart-aside-<UTC>/` | `--cold` moves, never deletes (`checkpointing.md` I3) |

#### History

| What | Form | Example |
|---|---|---|
| **commit message** | `<id> · <stage> · <what happened>` | `bdt_au · tight · relaxation converged, 41 steps` |
| **stage-completion tag** | `<id>/<stage>/<UTC>` | `bdt_au/tight/20260806T221403Z` |
| **UTC stamp** | `YYYYMMDDThhmmssZ` — compact, because a ref forbids colons | `20260806T221403Z` |

The id's character set was chosen to survive a filename, a shell line and a
scheduler argument — and **it is therefore already git-ref-safe**, so no second
sanitiser exists for tags (`run-identity.md § 3`).

#### Scheduler

| What | Form |
|---|---|
| **SLURM job name** | a directly-submitted `.sbatch` carries `-J <script-stem>`; via the submit engine it is **overridden** per job on the command line — a stage is its bare stage name, a trial is `job-gpu-G<g>K<k>C<c>` / `job-cpu` |
| **Dependency kind** | `afterok` / `afterany` (`execution/job-system.md`) |

#### Persisted-file schema strings

`molbuilder/<name>@<major>`, checked **major-only** through `molbuilder/persist.py`
— a reader meeting a newer major refuses rather than mis-parsing (§ 6.1, § 6.2).

> **Two rows corrected 2026-08-07.** This table used to give the per-job
> directory as `point-<name>/` for everything — *"benchmark `point-G<g>K<k>C<c>/`;
> stage ladder `point-stage<N>/`"*. Both are now wrong. A stage directory is
> `<seq>_<stage>` (`01_coarse`), because a stage is ordered and a sweep point is
> not, and the two should not share a shape that hides the difference. And the
> trial prefix is **`bench-`**, not `point-`: a trial belongs to a benchmark, and
> *point* is grid vocabulary that names nothing a user would recognise in a
> directory listing. Both are renames with a parser cost — `summarize` maps trial
> directories back to their settings — and both are worth it, because this table
> is what other layers copy from.

---

## 7. Change process

A change to any format in this document requires, in the **same commit**: the
code change, a test pinning the new invariant, and this document updated to
match. A generator that changes a filename, a parser that changes a discovery
rule, or a new warm-restart hook without its `--cold` glob entry is a bug, not
a feature — the whole point of pinning these shapes here is that the surfaces
above can rely on them without re-checking the code.
