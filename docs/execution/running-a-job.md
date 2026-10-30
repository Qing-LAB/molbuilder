# Running a job — the usable single-job path

**Role:** guide
**Domain:** execution

**Companions:** [`execution/job-contracts.md`](?doc=execution/job-contracts.md)
— the on-disk formats this guide operates (the run directory, the wrapper
files, warm/cold restart, the config vocabulary);
[`execution/job-system.md`](?doc=execution/job-system.md) — the JobSet
framework that runs **batches** of jobs on top of this same wrapper;
[`execution/overview.md`](?doc=execution/overview.md) — the map and the
current → target status picture.

**This is the path that works today.** One task is
`molbuilder jobset prep` then `molbuilder jobset launch --mode direct` — **a
job set of one, through the same commands as a hundred** (there is no
`molbuilder run`; decided 2026-08-11). The browser's part is the description
(the hand-over + Task setup — [`web/task-setup.md`](?doc=web/task-setup.md))
and `prep` for a named machine, through the one entry the terminal calls
([`job-system.md`](?doc=execution/job-system.md) § 5.3); launching runs where
the machine is. *(A legacy web install-wrapper
endpoint survives as the low-level side door the described route supersedes —
`job-contracts.md § 2.6`'s note.)* Everything here — the
self-contained wrapper, the runtime resource resolution, `molbuilder.json`
config, checkpoints, and watching a run — is shipped and usable now. Running
**many** parameterised jobs (sweeps, staged ladders, HPC deployment,
benchmarking) is the JobSet framework's job; it builds directly on the wrapper
described here, and is documented in `execution/job-system.md`.

```mermaid
flowchart LR
    P["Prep<br/>(Task setup or the CLI, for its machine)<br/>.fdf or .py + wrapper, activation baked"]
    C["Copy<br/>scp the calculation to that machine,<br/>when it was prepared elsewhere"]
    R["Launch<br/>(on that machine)<br/>bash .run.sh / sbatch .sbatch"]
    W["Watch<br/>the Results tab: viewer + Run panel<br/>or molbuilder watch"]
    P --> C --> R --> W
```

The **host** where you prep need not be where you run: prep reads the target's
record ([`preparing-for-another-machine.md`](?doc=execution/preparing-for-another-machine.md)),
and a run directory is self-contained, so `scp -r my-job/` to a cluster and it
still works. That self-containment is a deliberate contract (§ 2).

---

## 1. What this guide owns — a reader's map

| If you need to… | Read § |
|---|---|
| Understand why the wrapper is self-contained and what it may read when | **§ 2 — The standalone contract** |
| Know how many MPI ranks / OMP threads a run actually uses, and how GPUs are pinned | **§ 3 — Runtime resource resolution** |
| See what flags `bash my-job.run.sh` accepts | **§ 3.4** |
| Watch a running job, read failure hints, or ask how a run is doing | **§ 4 — Watching a run, and reading it back** |
| Configure envs and how a shell enters them (`env_init`) in `molbuilder.json` | **§ 5 — Configuration** |
| Snapshot / restore a run directory | **§ 6 — Checkpointing** |

The on-disk shapes this guide drives — the run directory layout, the wrapper
files (`.run.sh` / `.sbatch`), warm/cold restart semantics, the config
vocabulary, and the persisted-artifact registry — are all defined in
[`execution/job-contracts.md`](?doc=execution/job-contracts.md). This guide
does not restate them; it explains how to *operate* them.

---

## 2. The standalone contract — a wrapper that runs anywhere

> **The wrapper's own files are named by `runfiles` too** — its stdout, its
> session log, the monitor's pair, the SCF timing log, the conclusion marker.
> All but the session log, which carries its start's stamp, carry the try's
> number, and `-run<N>` is not a literal anywhere: it is the one keyword `runfiles.QUALIFIERS` declares, so a
> second counter is a line there rather than an edit to whatever reads names.
> See [`job-contracts.md`](?doc=execution/job-contracts.md) § 2.2a, *"Which
> door to call"* (rule **A14**).

The single most important property of a generated run is that the wrapper is
**self-contained at runtime**: it reads no config, probes no toolchain, and has
no fallback path. Everything site-specific is **baked in** when the wrapper is
generated/prepared, so the compute node needs nothing but the files in the
directory. This is what lets you `scp` a run dir to a cluster, or hand it to a
collaborator, and have it run identically.

> **Verified in a pure shell, 2026-08-19.** Both engines' wrappers were run
> under `env -i` — no conda on `PATH`, no rc files, non-interactive — and
> bootstrapped from nothing but the baked hook line: sourced it, activated
> the routed env, resolved the launch (`--dry-run`, exit 0, both engines),
> and completed a real SIESTA run. The one assumption is the package manager
> `prep` found ([`workflow.md § 6.1`](?doc=workflow.md)).

### 2.0a Where it runs, and what that decides

Every rule in this section follows from one fact: **`<label>.run.sh` runs
unattended, in a non-interactive shell that reads no rc files** — so nothing
loads a module or activates an environment for it; it activates its own.
Under a scheduler it is started by `sbatch` on a **compute node** — a different
machine from the one you submitted from. Under `--mode direct` it is started
by `bash` locally. Nobody is watching either way.

Both starts pass the launching shell's variables on — the whole environment
direct, `--export ALL` under Slurm (the kind's send line, `scheduler/kinds.toml`). The rule agreed for
them *(W36 ⑦, 2026-09-27)*: keep inheriting, guarded — a job's size comes from
what prep wrote and `launch`'s flags, molbuilder's own override name (`MB_NP`)
still works and the wrapper log names it when used, and a bare
`OMP_NUM_THREADS` no longer sets it. A run `jobset launch` starts already takes
its size from the `-np` / `-omp` it is handed, which win (§ 3.2). Run by hand,
the run script honours `MB_NP` and `OMP_NUM_THREADS` ahead of the reservation
(§ 3.2, § 3.3's precedence), and its log names none of them — the person's
own shell, left as it is (the override log was dropped from the plan on
2026-10-17: never seen on a real run).

What follows, and is checked in the rendered script:

| the situation demands | the wrapper does |
|---|---|
| nothing is on `PATH` before it acts | bakes the preamble and the activation verbatim |
| `activate.d` hooks read unset variables | `set +u` across the bootstrap **only**, restored immediately after |
| `--help` must answer without a working env | the whole bootstrap sits inside the help guard |
| the scheduler kills with `SIGTERM` at the time limit | `trap … TERM INT` → clean up, exit 143 |
| the engine's exit code must be inspectable | the engine is **run, not `exec`'d**; `set +e` around it, `${PIPESTATUS[0]}` captured |
| a run must be watchable while it runs | stdout is **teed**, never swallowed |
| the engine env has no molbuilder in it | the monitor travels as one copied file, `mb_monitor.pyz`, not `python -m molbuilder monitor` — holding it and the framework modules it reads the run through, each a copy of its source (`runwrap.MONITOR_BUNDLE`, `run-reports.md` § 2.3); a PySCF script imports the molbuilder code it calls from `mb_pyscf.pyz` beside it, the same way (`engines/pyscf.md` § 3) |

> **`set -e` stays ON across the preamble, and that is deliberate.**
> A failed `module load` here means the job is about to run in the wrong
> environment — on an allocation, unattended, producing results nobody asked
> for and nobody is reading. Dying at the failed line is the cheap outcome;
> continuing is the expensive one.
>
> **Do not "fix" this by making the preamble tolerant.** The instinct comes
> from interactive shells, where a preamble that does not apply to the
> machine in front of you should not stop you working. This script has no
> such user: it is on a node, on a clock, spending an allocation. Same line
> of shell, opposite correct behaviour, because the environment is different
> — which is why it is written down here rather than left to be re-derived.

### 2.1 "Detection" — reading external state

"Detection" means reading state that is *not* in the run directory. There are
five kinds, and the contract governs **when** each may be read:

| Code | External state | Example |
|---|---|---|
| **T** | conda **T**ool availability | is `siesta` on `PATH` in the env? |
| **M** | HPC **M**odules | `module load mamba` |
| **C** | **C**onfig | the target's record (`env_init`, copied from that machine's `molbuilder.json`) and `molbuilder.json`'s `envs` |
| **A** | **A**llocation | `MB_NTASKS` (what the scheduler granted, in molbuilder's name), `CUDA_VISIBLE_DEVICES` |
| **H** | **H**ardware topology | physical core count, GPU→NUMA node |

There are three moments in a job's life, and the rule is:

- **Generate / prep** — T, M, and C are resolved here and **baked** into the
  wrapper as literals. (Prep is also where the *doctor* verifies prerequisites.)
- **Runtime (on the compute node)** — reading T/M/C is **forbidden**; only **A**
  (the scheduler's allocation) and **H** (the local hardware) may be read, and
  only to *tune* the launch (rank counts, GPU pinning) or to log — never to
  decide whether the job can run.

So at runtime the wrapper never runs `which conda`, never reads
`molbuilder.json`, and never uses `conda run`. It emits the activation form
**verbatim** (`conda activate <env>` or `source activate <env>`) inside a
clean-shell bootstrap, then launches the engine.

### 2.2 The two prep-time jobs

- **Bake the activation — declared, never detected.** prep writes the target
  record's `env_init` into the wrapper (§ 5.2): copied there by the probe from
  that machine's `molbuilder.json`, where `envs init-config` asked for it at
  install and offered a recommendation — `conda activate` plus a
  `source "<base>/etc/profile.d/conda.sh"` preamble where conda's hook is on
  disk (the hook line is baked because a non-interactive `bash job.run.sh` never
  sources `~/.bashrc`, so `conda activate` would otherwise be undefined); on HPC
  typically `source activate` after `module load mamba/latest`, because a login
  node's modules are not the compute node's.
- **Doctor (every target).** Verifies prerequisites — the engine env exists, the
  tool is present — and reports what is missing. It **never installs**; a
  GPU env the target's record lacks, for instance, is said at prep with its
  install command ([`gpu.md`](?doc=execution/gpu.md) G6), never degraded to
  the CPU build in silence.

### 2.2a What the wrapper may do — bash is a bootstrap, not a program

§ 2.1 already says it in passing: the wrapper emits the activation form verbatim
inside a clean-shell bootstrap, **then launches the engine**. The rule is worth
stating because it is easy to break by accretion.

> **The wrapper makes the environment right, runs the engine as its child, and
> reports on the run it started — its output, its monitor, how it ended. It
> creates no directory and arranges no file: that is Python's, on the host,
> before the wrapper is invoked.**
>
> **Naming the run's output is the one thing it lays down beside the engine**,
> because only the run knows its number — a warm retry's is decided on the
> node: its stdout, through the tee, and SIESTA's history, whose name SIESTA
> takes from its label — the run script points that name at the run's own
> file while the engine runs and removes the pointer when it exits
> ([`job-contracts.md`](?doc=execution/job-contracts.md) § 2.2, § 2.6).

**Why bash at all:**

- **Activation mutates the shell's own environment.** `conda activate` /
  `module load` change `PATH` and friends *in the calling process*. A Python
  child cannot do that for its parent, so this genuinely has to be shell.
- **The launcher must be the shell's direct child.** `mpirun` / `srun` want to
  inherit the activated environment and sit in the process tree where signals
  and scheduler accounting expect them.
- **The shell must outlive the engine.** The engine is run, not `exec`'d
  (§ 2.0a), so the wrapper can read its exit code, decide a warm retry (§ 3.5),
  **finish the calculation when the engine alone leaves no result** — each
  call of a SIESTA vibration runs its record step with `mb_vibration.pyz` on
  the job's own python, the force-constant call's deriving the modes and the
  last call's the finish, the result at the run's root ([`engines/vibration.md`](?doc=engines/vibration.md)
  § 5.11 § 3); its failure is the job's — and write the conclusion marker as its
  last act ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.3).

Everything else — resolving which directory to run in, creating it, arranging
files, recording the launch — is **decision and arrangement**, and none of it
needs the activated environment. It is Python's.

**The test, when adding to a wrapper:** *does this need the activated shell, or
this run's own ending?* If it computes, decides or arranges anything else, it
belongs upstream. Two decisions are made on the node, each because only the
node can answer it: the warm retry, which only this run's ending can make —
and the wrapper asks the framework's reader for that ending rather than
grepping for it (§ 3.5) — and, for a job with a finish, whether the finish
can run on the job's python, which only the activated environment can say
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.5). The finish
itself passes the same test: it needs the activated environment and this
run's own output.

> **Nothing arrives at a run directory needing to be resolved.** What a stage
> continues from is a **real file, copied in at its first launch** from the run it takes
> ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6) — present and
> local before the wrapper starts. There is nothing for bash to dereference,
> localize or wait for.

**This is forced, not stylistic.** Two facts make the compute node the wrong
place for logic:

- **molbuilder is not importable there.** It is deliberately never
  pip-installed into any env, so `import molbuilder` fails on the job's
  python. The wrapper is self-contained at run time by
  design (§ 2). One job runs molbuilder itself on the node: one that sends
  several groups launches each next group with
  `-m molbuilder jobset launch task --group N … --in-sequence`, run from the
  folder molbuilder's package is in, which the job's node must see. It runs
  it in molbuilder's own environment, entered as a run script enters its
  engine's: the machine record's preamble and activation
  (`runwrap.env_bootstrap`, § 5.2), then that environment's own python,
  `$_mb_py`, checked before the first group
  ([`job-system.md`](?doc=execution/job-system.md) § 6.0, *The walk*). Every
  other walk step runs a run script with `bash`, and the scheduler header
  runs no Python.
- **The job's python is its env's own, for every engine.** Right after the
  activation the wrapper sets `$_mb_py` to `$CONDA_PREFIX/bin/python3` — the
  activated env's interpreter, never the first `python3` on `PATH`, which a
  site's activation can leave ahead of the env's (`source activate` under a
  mamba 2.x module, Sol) — and logs it (`job python:`). Every python the job
  runs is that one: the monitor, the ending reader, a finish, a PySCF deck.
  Once the try's number is known, every job checks, before anything starts,
  that `CONDA_PREFIX` names its env and that python is there; otherwise it
  stops, saying which, its conclusion `no env python (<path>)`
  (`parse.dirs.job.NO_ENV_PYTHON`). The activation forms are conda's
  (`conda activate`, `source activate`, § 5.2), and every recipe pins
  python into its env (`envs/recipes.py`).

  Any logic written to run there is either shell, or a shipped bundle run on
  `$_mb_py`, or broken. **Three bundles ship beside a job**
  (`runfiles.MONITOR_BUNDLE`, `VIBRATION_BUNDLE`, `PYSCF_BUNDLE`):
  `mb_monitor.pyz`, beside every job, holds the monitor — a *subprocess of
  the running job*, watching it from inside — and the reader the wrapper asks
  how the run ended (`"$_mb_py" mb_monitor.pyz ending …`, § 3.5);
  `mb_vibration.pyz` is a SIESTA vibration's finish and the steps of its run
  in phases ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.5,
  § 5.11); `mb_pyscf.pyz` holds the molbuilder code a PySCF deck imports.
  **Only the monitor's is stdlib-only**, because it is the instrument that
  reports a failure: it must load when the env's own packages do not — a
  broken numpy is reported by it, never hidden by it. The vibration and
  PySCF bundles run on the same job python and import the env's own
  packages, numpy and netCDF4 among them (`runwrap.VIBRATION_COMPANIONS`,
  `runwrap.PYSCF_COMPANIONS`). None is a pattern to copy for anything that
  could run on the host instead.

**No generated wrapper contains a `cd`**, on either engine: the caller decides
the directory, and the layout layer — `jobset/materialize.py::prepare_attempt`
— makes and fills it ([`project-layout.md`](?doc=execution/project-layout.md)
§ 1.6, invariant 6a).

### 2.3 Env routing

The wrapper routes to a conda env by the script's extension, resolved at
generate time (`molbuilder/diagnostics.py`, `molbuilder/runwrap.py`):

- **`.fdf` → `molbuilder-siesta`**, launched with `mpirun -np N siesta`.
- **`.py` → `molbuilder-pySCF`**, launched with `"$_mb_py"`, the env's own
  python (§ 2.2a).
- **`.fdf` that requests GPU → `molbuilder-siesta-gpu`.** Only
  `Diag.ELPA.GPU true` re-routes. **CPU-ELPA does not** — the packaged SIESTA
  carries ELPA through ELSI and runs both stages on CPU (measured; see
  [`engines/siesta.md`](?doc=engines/siesta.md) § 7.2). The two envs differ by
  **provenance** — one installs from packages anywhere, the other must be built
  from source — so routing CPU-ELPA to the source build used to refuse a
  runnable calculation wherever compiling is not allowed. If the `.fdf` opts
  into GPU and the target's record lists no such env, prep says so with the
  install command and prepares the job as asked (`gpu.md` G6): the run script
  enters the env at its start and stops there without it. An env named
  explicitly always wins over the route.

The env **names** are overridable per category in `molbuilder.json` (`envs`,
§ 5.4); the categories and their defaults are
[`configuration.md`](?doc=configuration.md) § 4's — `diagnostics.DEFAULT_ENV_NAMES`.
The host env is always `molbuilder` (§ 2.1c there).

> The wrapper file shapes (`.run.sh` inner + `.sbatch` outer), the run-indexed
> output names, and warm/cold restart are defined in
> [`execution/job-contracts.md § 2.6, § 4`](?doc=execution/job-contracts.md).
> The sections below cover the *runtime behaviour* those files carry out.

---

## 3. Runtime resource resolution

Only the **launch** is assembled at run time (from A + H); everything else is
baked — the rank and thread counts included, as the run stated them. Here is
exactly which statement applies, and how the wrapper places a GPU run.

### 3.1 MPI ranks (SIESTA)

SIESTA is launched with `mpirun -np N` when the build probe reports MPI. The
rank count `N` is resolved by precedence, **highest wins**:

```
-np / --np flag   >   MB_NP   >   MB_NTASKS   >   the stated value, baked at prep
```

**The baked value is the one the run stated** — the run card's `mpi_np`, or
prep's `--np`. A run that states none is **refused at prep**, which names both
places (`architecture.md` § 5.2, which owns the rule). The wrapper has no
default of its own and neither has the `.sbatch` header — no target width, no
rank per GPU *(user, 2026-10-02: "explicit job config is the only way
allowed")*. `MB_NTASKS` is the ranks the scheduler granted, said in
molbuilder's name by the job's header (SLURM's `SLURM_NTASKS`,
[`scheduler.md`](?doc=execution/scheduler.md) § 6a) — the header's own rank
count, which is that same stated value, so the two agree by construction.

**Nothing lowers it** *(user ruling, 2026-09-03)*. Until then the default was
**clamped to the atom count**, and a user-set count above it was warned about,
both citing SIESTA's `propor IMAX=0` abort. That abort came from a **PSML**
problem, not from the system's size — so the rule was not science: it gave
right-sounding advice for a wrong reason where it fired, and refused perfectly
good rank counts everywhere else. *"Whether mpi is too big for a system is
none of your business."*

**The target's record checks a rank count; it never supplies one.** Until
2026-10-02 an unstated count became the target's width — the domain row's
widest node, else the record's topology — and before 2026-09-02 the width of
the box running `prep`. Each was a number nobody stated for that run, and the
second was about the wrong machine and looked exactly like a right one.

### 3.1a What the wrapper says instead — occupancy, as a notice

**SIESTA distributes ORBITALS across ranks**, not atoms, so the number that
says whether the ranks have anything to hold is

```
n_orbitals / mpi_np        # want > 1
```

At or below one, ranks are idle by arithmetic, and the wrapper says so at run
time — against the rank count **actually resolved**, not the one baked at
generation, so an `-np` override is included:

```
molbuilder: NOTE -- 200 ranks for ~100 orbitals is <= 1 orbital per rank.
  Your CPUs are not going to be fully used.  This is a notice, not a limit -- the run proceeds.
```

Both numbers are shown so the claim is checkable rather than a verdict, and
**nothing is refused or lowered**.

The orbital count is an **estimate and says so**: `10 × n_atoms`, the same
rough double-zeta-polarised figure the `BlockSize` bound uses and the deck's
BENCH-MARKS block already publishes as `n_orbitals_est` (`job-contracts.md`
§ 3.3). One estimate in one place, or the deck and the notice would disagree.
The true count needs every species' basis and is known only once SIESTA
starts. A deck with no `NumberOfAtoms` — it is optional in SIESTA — gets **no
notice at all** rather than one built on an invented number.

> **The header's `-n` is the stated rank count and nothing else** (F15,
> 2026-08-13; 2026-09-02; 2026-10-02). With no rank count stated it floored at
> `-n 1`, later took the target's width, and on a GPU job one rank per device
> — and inside the job the granted ranks **come from that very header**, so
> each of those outranked everything below it in the chain above: a 64-core
> node once ran a job on **one rank**. Prep refuses the unstated count now. The
> `--dry-run` inspection names each value's source.

> #### What `propor: IMAX = 0` depends on — and why no rank count is refused
>
> *(User ruling, 2026-09-03: "we had that problem because of a psml related
> issue, not a size issue.")*
>
> | | |
> |---|---|
> | **where it is raised** | `propor` (`Src/propor.f`), called only from `matel_table.F90`, which deduplicates the **radial-function tables** across MPI ranks — not in any BLACS distribution |
> | **when** | the table handed to it is all zeros |
> | **its two causes, in the order to check them** | 1. a **defective or XC-mismatched pseudopotential** — commoner and cheaper to check ([`science/pseudopotentials.md`](?doc=science/pseudopotentials.md); `molbuilder pseudo check`); 2. the **rank count** against the species count and radial-table size |
> | **not a cause** | `BlockSize` — the crash is identical at 1, 2 and 4 (`mpi_np` = 15, hemeC-dithiol); the atom count, a proxy that a few-species system exceeds safely and a many-species one crashes below; the spin — `propor` takes no spin argument and never sees the density matrix |
>
> So nothing lowers or refuses a rank count, and no species-aware bound
> replaces the atom clamp: whether a rank count suits a system is the user's
> judgement, and what the wrapper owes them is the objective number — orbitals
> per rank, above. The `.fdf` keeps `NumberOfSpecies` for a diagnostic.
> `BlockSize`'s bound is a different correction — **orbitals** over ranks, for
> matrix distribution ([`tuning.md § 2.11`](?doc=engines/tuning.md)).

### 3.2 OMP threads and BLAS

```
-omp / -t flag   >   OMP_NUM_THREADS   >   MB_CPUS_PER_TASK   >   the stated value, baked at prep
```

A run `jobset launch` starts always carries the flag, its own `-np` / `-omp`
([`job-system.md` § 6.1](?doc=execution/job-system.md)); the rest of the order
is for a script run by hand. The two variables are one list,
`runtime_info.THREAD_SOURCES` — `MB_CPUS_PER_TASK` the cores per rank the
scheduler granted, in molbuilder's name, as the job's header says it — and
the run script's chain is built from it, as is a PySCF script's own, run
without its run script, which ends on the node's physical cores instead
([`engines/pyscf.md`](?doc=engines/pyscf.md) § 3).

The stated value is the run card's cores per rank — SIESTA's `omp_threads`,
PySCF's `threads` — or prep's `--cpus-per-task`; a run that states none is **refused at
prep**, like a rank count. There is no policy default: SIESTA's `OMP=1` and
PySCF's *"the node's physical cores"* were the wrapper's own, and each was a
thread count nobody stated. *(Mainline SIESTA is not reliably OpenMP-aware, so
`omp_threads = 1` is the usual statement for a CPU build — a statement, made on
the run card.)* BLAS is always pinned to a single thread (`MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`) to keep MPI×BLAS from oversubscribing cores.

### 3.3 GPU mode: load-balance, MPS, and pinning

When the `.fdf` runs on GPU (ELPA-CUDA), the wrapper does substantially more at
launch (all in `molbuilder/runwrap.py`).  *This section is the MECHANICS; the
background — what actually offloads, why ranks share a device, what the
sources recommend, and how to design a benchmark matrix around it — is
[`engines/tuning.md § 2.12`](?doc=engines/tuning.md).*

- **Load-balance.** Counts allocated GPUs (preferring `CUDA_VISIBLE_DEVICES`
  over `nvidia-smi -L`), sets `ranks_per_gpu = mpi_np / ngpu` (≥ 1), and prints a
  `GPU load-balance` line. **The rank and thread counts are the stated ones**,
  as on CPU (§ 3.1, § 3.2); what to state for a GPU run is
  [`engines/tuning.md § 2.12`](?doc=engines/tuning.md)'s subject.
  `--mps`/`--no-mps` allow or forbid the MPS daemon and nothing else.
- **NVIDIA-MPS (Hyper-Q)** — the NVIDIA Multi-Process Service, which lets two or
  more MPI ranks share one GPU concurrently. **One decision**: on only when (a)
  `nvidia-cuda-mps-control` is on `PATH`, (b) the user did not opt out
  (`--no-mps` or `MOLBUILDER_USE_MPS`), and (c) the job sees at least one GPU
  and more ranks than GPUs — the moment sharing actually happens;
  single-GPU-per-rank runs need no MPS and get none. The banner and the dry run
  say that decision — on where the daemon started (in a dry run, would start),
  off otherwise, a start that timed out among them — and the run's log says
  why it is off. The per-job MPS daemon is torn down by a single `EXIT`
  trap.
- **Where each rank runs.** A generated helper assigns each rank a GPU
  (`CUDA_VISIBLE_DEVICES`), block-distributed (`rank × GPUs / ranks`), and,
  when the rank's cpuset spans more than one socket, pins it with `numactl` to
  the NUMA nodes of its GPU's socket that it owns — **the helper probes that
  mapping itself, per rank, at run time** (nvidia-smi + sysfs). Opt out of
  socket pinning with `MB_NO_SOCKET_PIN=1`. **Off a scheduler**, OpenMPI binds
  each rank to cores of its own (`--bind-to core --map-by
  package:PE=<cores per rank>`), the ranks spread over the sockets, and the
  **whole-job** `numactl` wrap of a single-GPU job (from a GPU→NUMA answer
  looked up **at run time**, on the machine the job runs on, from GPU 0's PCI
  bus id and sysfs — the per-rank helper's own lookup — and overridable with
  `MOLBUILDER_GPU_NUMA`) sets its memory on the GPU's node
  but does not hold its ranks there: OpenMPI's map overrides it (measured
  with OpenMPI 5.0.10), so a single GPU's second rank runs on the far socket.
  **Under a scheduler** the wrap and that binding are cleared, and OpenMPI
  places the ranks inside the cores the scheduler gave the job. **A rank whose
  cores lie on another socket than its GPU is warned about, never refused** —
  the helper says so in the run's log, and the run goes on paying the
  traffic (the user, 2026-10-17).

> **Override precedence, stated once:** the `-np` / `-omp` flags, then
> `MB_NP` / `OMP_NUM_THREADS`, then the scheduler's echo of the header
> (`MB_NTASKS`, `MB_CPUS_PER_TASK`), then the stated value baked at prep.

### 3.4 The flags a wrapper accepts

| Flag | Engines | Effect |
|---|---|---|
| `--run N` | both | **the try's number** — `0` from `launch`, which gives every run script it starts one, the next from a warm retry (§ 3.5; [`project-layout.md`](?doc=execution/project-layout.md) § 1.6.1, § 5.5 below); every file the try writes carries it as `-run<N>`. Without one — missing, or `--run` given no whole number — the script refuses to start, exit 2, naming the launch command; `-h` answers without one |
| `--continue` / `-c` | both | this start continues the try before it — a warm retry's (§ 3.5) — and the banner says so; whether the engine reads the warm files (`.DM`/`.CG`/`.XV`, `.chk`) is the deck's to say. The try's number is `--run`'s |
| `--force` / `-f` | both | what says *yes, overwrite* to `--cold`'s refusal |
| `--cold` / `--from-scratch` | both | start the engine from the deck alone, **removing** the prior state it names first (§ `job-contracts § 4`). Names the files and **refuses**; `--force` removes them — but under `--dry-run`, which changes nothing: it names them and says nothing was removed |
| `-np` / `--np N` | both | SIESTA: override MPI ranks. PySCF: accepted, and anything but 1 is said to be ignored — PySCF is OpenMP-only |
| `-omp` / `--omp N` | both | override OMP threads (SIESTA also takes `-t` / `--threads N`) |
| `--mps` / `--no-mps` | SIESTA (GPU) | allow / forbid MPS; it starts only where the ranks outnumber the GPUs (§ 3.3) |
| `--dry-run` | both | print the resolved launch command + rank→GPU/NUMA map, then exit 0 |
| `-h` / `--help` | both | usage |

> **`--cold` refuses before it overwrites, and that is the whole of the safety
> net.** It prints every file the run would overwrite, tells you to save the
> state first with `molbuilder checkpoint save`, and exits 1 having changed
> nothing. Run it again with `--force` and the same list is printed and the run
> proceeds. It *moved* those files into a timestamped `…-restart-aside-<UTC>/`
> folder until 2026-08-18; keeping a state is the checkpoint tool's job and it
> is never automatic ([`checkpointing.md § 2`](?doc=execution/checkpointing.md)).
>
> It works the same under `sbatch`: a refusal fails the job immediately with the
> reason in the log, where a prompt would simply hang.

Unrecognised arguments are **rejected** (the wrapper exits 1) — only the flags
above are accepted, for either engine. The `.sbatch` outer file forwards
`"$@"`, so a flag on the line that sends it — `jobset launch`'s, or the one
its own comment prints — still reaches the inner wrapper.

### 3.5 SIESTA auto-retry on non-convergence

A SIESTA wrapper **re-runs itself warm**, in the same run folder, with
`--continue` and the next try's number (`--run N+1`), when its try ended in
one of two ways a warm restart can fix. The budget is the run card's
`continue_retries` (*Warm-retry budget*, 0–5, default 1; a run setting, in
`job-request.json`, never the template's — `engines/stages.md` § 3), carried on
`Resources` (`job-contracts.md` § 6.2) and bounded by the exported
`MB_RETRY_N`; a benchmark trial's is `0`, one try whatever happens. **A
TBtrans run is never retried**: it runs no SCF and no relaxation, so neither
ending below can occur, and its script carries no retry and says so on its
banner's retry line whatever the budget *(F19b: it promised retries on
non-convergence)*.

| the try ended | SIESTA shows it by | the wrapper asks | the retry resumes from |
|---|---|---|---|
| its SCF did not converge, and SIESTA stopped on it | `SCF_NOT_CONV: … (required)`, then a nonzero exit (`SCF.MustConverge`: SIESTA's default, false in a benchmark trial's deck) | `_mb_ending stopped-by scf_not_conv` | the banked `.DM`, with a fresh SCF budget |
| it stopped at its wall (§ 3.6) | `OUT_OF_TIME: Time is up.`, after saving `.DM`/`.XV` | `_mb_ending stopped-by out_of_time:` — **never retried**: no time is left, and the next launch continues from the saved files | — |
| a relaxation used all its moves, unconverged | exit 0, and `outcoor: Final (unrelaxed) atomic coordinates` (a converged one prints `Relaxed…`) | `_mb_ending relaxation-capped` | the banked `.XV`/`.DM`/`.CG`, with a fresh step budget |

The wrapper **asks, never greps**: `_mb_ending` runs `"$_mb_py" mb_monitor.pyz
ending` over the output and SIESTA's stderr (`run-reports.md` § 2.3). **Never
retried**, because running again cannot fix it: a crash — `propor`'s
`IMAX = 0` (§ 3.1a), any abort, or a tolerated non-convergence followed by
one; a **diverged** SCF, whose retry resumes from the diverged density and is
the same run again — the monitor warns instead (`model/parse.md` § 5d.6)
*(not built — plan § 5t, P4: no divergence verdict or symptom is read yet, so a
diverged SCF that SIESTA stopped reads as the first row's and is retried)*; and
anything, when `mb_monitor.pyz` is not beside the job or does not load on the
job's python (the log says so).

**A run that cannot resume is retried as the budget says, and said to
re-run.** SIESTA re-runs a force-constant run from its reference (undisplaced)
step, reading back only the density it saved at that step
(`save_density_matrix.F90`: every iteration of the reference step, no other;
`m_new_dm.F90` reloads it before every displacement): an SCF that stopped the
reference step continues from it with a fresh budget, and one that stopped at a
displacement meets the same SCF again. The budget still travels — it is the
person's call (`job-contracts.md` § 6.2), as it is for a stage described
`clean` — and what changes is what the wrapper SAYS: the kind declares the fact
(`warm-files.toml`, `resumes = false`,
[`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a), `prep` bakes it
into the run script it renders (`Job.resumes`; a SIESTA vibration's, each
phase's own), and every wrapper text that speaks of a
retry — the banner's retry line, the retry's message, a retried run's Mode
line, the `--continue` help, the line after the budget, the line at the
wall (§ 3.6) — says it from one description (`runwrap._retry_texts`), a point
of a run of points told what its run's launch again does with it: the points
done taken over, this one run again as its run's rule starts a point not
done — a vibration's relaxation continues warm from its own last run, its
other calls and a transport point start over
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6,
[`engines/transport.md`](?doc=engines/transport.md) § 2a.11). *(They called it a resume until
2026-09-29 — the M11 review, SS-C3; the K6 review, R4/R5.)* *The divergence
verdict that tells a diverged SCF from a slow one is built by plan § 5t.3's P4;
until it lands every `SCF_NOT_CONV … (required)` stop is retried.*

```mermaid
flowchart LR
    P["a call of a run in phases:<br/>its plan step first"] --> X["SIESTA exits"]
    X --> Q{"how did it end?"}
    Q -->|"stopped on SCF_NOT_CONV, not diverged"| B{"budget left?"}
    Q -->|"a relaxation out of moves"| B
    Q -->|"a failure"| S["a call of a run in phases:<br/>its record step, told the exit"]
    S --> C["conclude: write -runN.concluded"]
    Q -->|"a clean exit"| F{"a finish to run?<br/>(Job.finish)"}
    F -->|"no"| C
    F -->|"yes: its exit status is the job's"| C
    B -->|"yes"| R["re-exec with --continue --run N+1"]
    B -->|"no: say so"| S
```

**A retry stays in its run**: the wrapper re-execs itself in the same
process and run folder, with the same `-np`/`--omp` and the next try's
number (`-run0` → `-run1`,
[`project-layout.md`](?doc=execution/project-layout.md) § 1.6.1), and only the
last try writes the conclusion marker. The monitor is stopped with SIGUSR1,
not an ending (§ 4.1). **Each try keeps its own session log**: the wrapper
hands back the stdout and stderr it was started with before it re-execs, so
the retry's output goes to its own log and to where the job's output goes,
never into the try before it *(F19b, 2026-10-13: a retry's output went on
into the first try's log, which held both)*.

**A retry writes no record of its own**: the run's `run.json` answers
for every try in it (`project-layout.md` § 1.6.3). Its line in the log reads
*… (retry 1/1) as try 1, with --continue*.

---


### 3.6 A run ends itself before the wall, and a file written as a job died is said

**The wall a job was sent with reaches its engine** *(the user, 2026-10-19:
"this should be now mandatory for all engines … fed to the engine through
their proper channel/parameter")*. The launch tells each run its wall in
seconds, `MB_WALL_S`, and the run script counts it from the start
`MB_JOB_STARTED` holds:

| launched | `MB_WALL_S` — told by `jobset launch` | `MB_JOB_STARTED` — counted from |
|---|---|---|
| one run, to a queue | on the send line, beside the launch claim ([`scheduler.md`](?doc=execution/scheduler.md) § 6a): the wall the header prep wrote carries, read off it by its kind's spelling (`Kind.directive_value`), or the launch's `--time` — the value `run.json`'s `placed_on.time` records (`submit._sbatch_request`) | the job's start, which its header says once ([`run-reports.md`](?doc=execution/run-reports.md) § 2.1b) |
| a walk, to a queue — a group, a run of points, a benchmark's shelf | the same line, the job's one wall for every member: its header's — a group's and a run of points' its members' walls summed (`group.envelope`), or the stage's one wall where a run's points share it (a vibration's calls, `jobset.points`), a shelf's the benchmark's own — or the launch's `--time` | the job's start: a later member counts what the members before it left |
| a sequence of groups, to a queue ([`job-system.md`](?doc=execution/job-system.md) § 6.0) | the same line, the job's one wall for every run of every group: its header's — every group's own wall summed (`group.sequence_request`) — or the launch's `--time`; each group's launch inside the job tells its runs nothing of their own (`submit._own_walls`) | the job's start: a later group counts what the groups before it left |
| one run, here | in its environment (`submit._go`), the wall its own job states (`Resources.time`: § 5.3.1's chain at prep) — none stated, nothing | its run script's own start |
| a walk, here | on each member's line of the walk script (`submit._walk_script`, `walls`), the wall that member's own job states — none stated, nothing | each member's run script's own start — but a run whose points share the stage's one wall (a vibration's calls, `jobset.points`), the walk's own start, exported once by the walk script (`MB_JOB_STARTED`), so its calls count from one start |
| a sequence of groups, here | each group's launch tells its runs as a run here or a walk here is told, above: each its own card's | each run script's own start |

From the two the run
script counts what is left when it starts the engine, and hands the engine
that and the calculation's **wall slack** — the catalogue's `wall_slack`
item on the run card, seconds, default 600 — through the engine's own
channel; the engine stops once less than the slack is left:

| engine | the channel | what the engine does |
|---|---|---|
| SIESTA, TranSIESTA | `-fdf MaxWalltime=<left>:s -fdf MaxWalltime.Slack=<slack>:s` on the command line, beside the deck — the words the catalogue's `wall_slack` item gives the program (`launch`, `template.md` § 5), never a line in the deck, which prep wrote before the launch stated the wall ([`engines/siesta.md`](?doc=engines/siesta.md) § 8.9) | checks the clock after every SCF iteration and every geometry step (`siesta_forces.F90`, `siesta_move.F`, 5.4.2) and, once less than the slack is left, saves its density and geometry, warns `SCF_NOT_CONV` or `GEOM_NOT_CONV`, and **dies with `OUT_OF_TIME: Time is up.`** — its own word for the stop (`siesta_grammar.OUT_OF_TIME_MARKER`). The run reads `failed`, the wall its cause in SIESTA's words; the wrapper does not warm-retry it (§ 3.5: no time is left); a launch again continues a single run from the files it saved, and starts a point of a run of points as its run's rule starts a point not done (§ 3.5) |
| TBtrans | nothing: the item names no words for it, it checks no clock | runs to the wall |
| PySCF | the deck reads `MB_WALL_S`, `MB_JOB_STARTED` and `MB_WALL_SLACK_S` from its environment and keeps a deadline ([`engines/pyscf.md`](?doc=engines/pyscf.md) § 3.9) | a relaxation stops at the first step past the deadline, keeps the geometry it reached (`_optimized.xyz`, as every step does) and ends normally with `# relaxed: no`; a vibration deck past the deadline before its spectrum is written stops with an error naming the wall, so the run reads `failed` with the reason and a launch starts it again |

A run with no wall stated — a run here whose job states none — hands the
engine nothing and runs until it is done. The monitor's `[WALL]` warning
(`run-reports.md` § 2.1b) stays for a job sent to a queue: it is the cue for a job
whose engine could not stop itself in time, a step longer than the slack.

**What a person sees of the wall**, word for word — a job sent with 4 hours,
the default slack:

| where | the line | when |
|---|---|---|
| the session log (`<basename>.runwrap-<stamp>.log`, § 4.1), the run script's banner | `Wall          : 14400s from the job's start; SIESTA stops 600s before it (running-a-job.md 3.6)` — a transmission's: `TBtrans checks no clock and runs to it (running-a-job.md 3.6)`; told no wall: `Wall          : none stated -- runs until done` | every start of SIESTA or TBtrans |
| the session log | `wall: 14382s left of 14400s; SIESTA stops 600s before it (running-a-job.md 3.6)` — what is left when SIESTA starts | a wall told, the program SIESTA |
| SIESTA's output | `OUT_OF_TIME: Time is up.` | SIESTA stopped itself at its wall |
| the session log, from the run script | `stopped at its wall (MaxWalltime): the restart state is saved; launch this stage again to continue` — the last clause says what a launch again does with this run, from the one description of it (§ 3.5, `runwrap._retry_texts`): a single run whose kind resumes, as shown; a point of a run of points — a vibration's call, a transport point — that the points done are taken over and this one is run again as its run's rule starts a point not done: a relaxation continues warm from its own last run, the other calls and a transport point start over ([`engines/vibration.md`](?doc=engines/vibration.md) § 5.11 § 6, [`engines/transport.md`](?doc=engines/transport.md) § 2a.11); a kind whose runs do not resume, that it re-runs from its first step | that stop — in place of a warm retry |
| the run's state (`jobset status`, the Results tab) | *failed* — `stopped before its end: OUT_OF_TIME: Time is up.` | that stop |
| the Results tab's *Reason* | `its wall (MaxWalltime): the engine stopped and saved its restart state` (`siesta_grammar.CAUSE_WORDS`) — the cause alone; what a launch again does is the session log's line above | that stop |
| a PySCF relaxation's output | `WARNING: stopped before the wall (the job's wall less its slack): the geometry reached is kept; launch this stage again to continue from it` — the last clause where launching again reads the geometry back (its `restart` is `continue`); then `# relaxed: no` in its trajectory log | the first step past the deadline |
| a PySCF vibration's output | `RuntimeError: stopped before the wall (the job's wall less its slack) before the spectrum was written: launch this stage again` | a displaced SCF due past the deadline |
| the monitor log | `[WALL] 21m left; the longest of 3 steps so far took 46m -- the step underway may not finish before the wall` ([`run-reports.md`](?doc=execution/run-reports.md) § 2.1b) | once, in a job sent to a queue |

**Why this is mandatory.** A scheduler's kill lands at a second of its own
choosing, and SIESTA writes its density in place after every iteration
(`m_iodm.F90`: open, write, close, no rename): a kill inside that write leaves
a file of the right shape with mixed numbers, which SIESTA then reads back
reporting success. Measured on Sol, 2026-10-18: the third launch of a
312-atom relaxation started from such a file at +4,251,896 eV with a density
residual of 694, and spent 37 SCF iterations climbing back to the energy the
run already had.

**A warm file written as the job ended is said at the launch that takes
it** (`materialize.written_as_ended`). Every launch that continues a stage
from a run's files compares each file with the second that run ended — a
stage launched again from its own last run, a first launch from the stage
before it, a run named with `--from`. The run's end is
its monitor log's closing record, the `[MONITOR] job ended` line
([`run-reports.md`](?doc=execution/run-reports.md) § 2.5;
`parse/instruments/monitor.py`, `ended_at`): the monitor writes it on the
signal that stops it — the scheduler's, at the wall or a cancel, or the run
script's own at its exit — within a fifth of a second, its time to the whole
second. A file whose write time (its modification time) is in that second
or after it is named. The inputs a transport rung gathers are not compared:
a gather takes only runs that finished. A run whose monitor wrote no closing
record — the node lost, the monitor killed outright — has no end to compare
with, and nothing is said.

The launch names the file in its own two lines — the preview before its
question, and the line once sent — and the ledger's `continues` entry
records it (`written_as_ended`: each file, with the same words); `status`
and the browser, which say what a launch would take, do not compare. A group
launched in a sequence of groups' job (`job-system.md` § 6.0) prints its
lines in the job's output. Word for word, a stage launched again after its
job was killed at the wall:

```text
  coarse: WOULD launch it again into run-1: it continues from 01_coarse/run-0 (its own latest run; …): would copy H2.XV, H2.DM -- WARNING: written as the job ended (13:58:12; the job ended 13:58:11), it may be incomplete; a start from it may need more iterations or fail; `--cold` starts without it
```

— once sent, *copied* in place of *would copy*. A fact shown, never a veto: the launch goes
as asked, and the person decides.

---

## 4. Watching a run, and reading it back

### 4.0 Which question, which reader

| the question | the reader | owned by |
|---|---|---|
| how is it going, right now? | the monitor, beside the job | [`run-reports.md`](?doc=execution/run-reports.md) § 2 |
| how is it doing — how did it end? | `parse.dirs.job.run_status` | § 4.2 |
| which engine ran? | `parse.contract.engine_of` | § 4.2 |
| what ran, with what, and how did it go? | the run record, `runs.folder_answer(dir)["record"]` | [`model/parse.md`](?doc=model/parse.md) § 5d |
| which file should a viewer open? | the run door, `runs.openable` — `parse.dirs.openable_in` for a folder no calculation claims | [`model/parse.md`](?doc=model/parse.md) § 5.2 |

### 4.1 The wrapper's own instruments

| instrument | file | engines | written | read by |
|---|---|---|---|---|
| **run banner** — host, cwd, env, engine binary and version, launch mode, threading, GPU resources | the session log | both | before the engine starts | a person; the run record |
| **session log** — the wrapper's stdout and stderr, SIESTA's stderr included, the engine's wall time on its `benchmark:` line, and the monitor's start — *starting*, then *started* or the error that stopped it ([`run-reports.md`](?doc=execution/run-reports.md) § 2.6) | `<basename>.runwrap-<stamp>.log`, one per start | both | from its first line | the run record; `_mb_ending` (§ 3.5) |
| **the engine's stdout**, teed | `<basename>-runN.out` · `-runN.pyscf.log` | SIESTA · PySCF | as it prints | `run_status` (§ 4.2); the viewers |
| **monitor** — utilisation every 10 s, progress, notifications | `-runN.monitor.log` · `-runN.util.csv` | both | start to end | a person; the run record |
| **SCF-timing tee** — every SCF row of both phases (`scf:`, `ts-scf:`) | `-runN.scf-timing.log` | SIESTA | as rows print | the timing instrument ([`model/parse.md`](?doc=model/parse.md) § 5c) |
| **failure hint** — how the run ended, where its output and log are; for `propor`, the causes (§ 3.1a) | the session log | SIESTA | on a nonzero exit | a person |
| **conclusion marker** — exit code and time, and the words that say why where a step of ours stopped it: the job had no python of its env (`…; no env python (<path>)`), or its finish — or a run in phases' plan or record step — failed or could not run (`…; finish failed (<bundle> <verb>)`, `…; finish cannot load (<bundle>)`) | `-runN.concluded` | both | the wrapper's last act, main line only — after the job's finish, when it has one | `run_status`, the launch gate, the run record ([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.3) |

**The monitor** is `mb_monitor.pyz`, one file beside every deck, reading the
run with the framework's own readers. What it reads, when it speaks (never
about a stall), what its percentages are fractions of — what the job **holds**,
never the node — its lifecycle and its switches are all
[`run-reports.md`](?doc=execution/run-reports.md) § 2's; each run states its
utilisation basis on a `[UTIL-BASIS]` line, to read before comparing two runs.

**It is told *when* the run ended, and never reads that from the output**: the
watched PID goes, or the wrapper stops it — SIGTERM at the job's end, SIGUSR1
for a warm retry, which is not an ending (`run-reports.md` § 2.4). *How* it
ended is `run_status`'s answer. An ending phrase can print before the run is
over (`siesta: Final energy`); a monitor that stopped on one would stop
sampling a job still holding its CPUs.

### 4.2 Reading a run directory back — `run_status` and `engine_of`

**This section owns a run directory's state** — its evidence, their order, and
each surface's words. A file's own ending is
[`model/parse.md`](?doc=model/parse.md) § 2b's.

#### The call

```python
run_status(run_dir, basename, *, launch=<not asked>, run=None, launched_in=None) -> RunStatus   # parse/dirs/job.py
```

| argument | meaning |
|---|---|
| `run_dir` | the attempt (`run-<n>/`) |
| `basename` | the run's stem, `<label>_<token>` (`runfiles.stem`): a run folder may hold several tries' files |
| `launch` | `launch_record(dir)` (`runrecord.py`) — the `run.json` dict, or `None`; one that does not read is an error naming the file, never an answer; left out, *not asked* |
| `run` | the try asked about; left out, the run's newest (`runrecord.newest_run`) |
| `launched_in` | the folder the launch record lies in when it is not `run_dir` — a run of points' one record, in the run's folder, answering for each point (`rundir.run_state_of` passes it; the scheduler's file is named from there) |

| field | holds |
|---|---|
| `state` | `pending` · `queued` · `running` · `finished` · `failed` · `unreadable`, a closed set — `unreadable` when the run's launch record does not read: its state cannot be told, the file is named in `detail`, and nothing follows it *(it read as the files alone until 2026-10-06 — `running` where nothing else was written)* |
| `detail` | one line for a person (the states, below) |
| `last_change_at` | the speaking file's mtime, ISO-8601 UTC — shown, never judged |
| `active_source` | the speaking file's name, or `None` |
| `concluded` | the conclusion marker's text, `rc=0 at …`, beside the state |
| `endings` | each output file's ending and its SCF phases' convergence — read by the run record and the monitor, so nothing scans twice |

`/api/results/dir` serves the first four ([`web/results.md`](?doc=web/results.md) § 2.3).

| caller | `launch` | `match` |
|---|---|---|
| `jobset/runstatus.py` — `jobset status`, the bench summary, the Results ladder | the run's, through `runs.concluded` (`state_of`: the run door, and the scheduler asked once where it can — § 4.2, *Asked once, written once*) — an attempt's, each point's for a bias scan's rung ([`engines/transport.md`](?doc=engines/transport.md) § 2a.11), or a trial's | the stage's stem |
| the run door, `runs.folder_answer` — `/api/results/dir`, the run record; `runs.run_answer` — the viewers | the run's, through the same door | the run's basename: the run the folder speaks for (`execution/architecture.md` § 3.2) |
| the monitor's closing line | not asked | its run's stem |

`run_status` has no answer *there is no run here*: asked without `launch`, a
directory with nothing written reads `running — no result file yet`. So the
directory door asks what a directory is first, and asks a container nothing
([`project-layout.md`](?doc=execution/project-layout.md) § 1.4a).

#### Which file speaks

An engine's captured stdout (`.out`, `.pyscf.log`) speaks, because it exists
only once the process started; a progress log (`.molwatch.log`) speaks once its
footer concludes, because prep seeds it before the engine starts. Each is read
by its role's scan for ending markers, `_run_ending.ending_of`
([`model/parse.md`](?doc=model/parse.md) § 5.4) — not parsed through the
registry. Of several, the newest try's speaks ([`model/parse.md`](?doc=model/parse.md)
§ 5.1); a file's time decides nothing, because a copied or
restored folder reorders times, never tries *(the newest mtime picked the
file until 2026-10-04, plan B11)*. The try's number picks *which* file, never
*whether* the run is alive — with the one bound a sent job's own wall gives
(*Past its wall*, below): its own files' newest write older than that wall
says the job is over, and nothing else about a time is judged.

#### The order of evidence

```mermaid
flowchart TD
    S{"does a file speak?"} -->|"yes"| E{"what ending does it state?"}
    E -->|"a stop, or out of memory"| FAIL["failed"]
    E -->|"its end, or none"| C{"did the run end on its own?"}
    C -->|"exit code 0"| FIN["finished"]
    C -->|"another exit code"| FAIL
    C -->|"not yet"| K{"did its scheduler end it?"}
    K -->|"its line for this job, or its recorded answer (ended)"| FAIL
    K -->|"none"| G{"its monitor's closing record?"}
    G -->|"job ended"| FAIL
    G -->|"none"| W{"past its wall?"}
    W -->|"its own newest write older than the wall it was sent with"| FAIL
    W -->|"no"| RUN["running"]
    S -->|"no"| C2{"did the run end on its own?"}
    C2 -->|"exit code 0"| FIN
    C2 -->|"another exit code"| FAIL
    C2 -->|"not yet"| K2{"did its scheduler end it?"}
    K2 -->|"its line for this job, or its recorded answer (ended)"| FAIL
    K2 -->|"none"| G2{"its monitor's closing record?"}
    G2 -->|"job ended"| FAIL
    G2 -->|"none"| L{"the launch record"}
    L -->|"no run.json"| PEN["pending"]
    L -->|"run.json"| Q["queued"]
    L -->|"not asked"| RUN
```

*Its end* is `>> End of run`, a PySCF deck's end line or a `# concluded:`
footer; *a stop* is a fatal marker, a Python traceback, an `# error:` footer or
an out-of-memory line (`model/parse.md` § 2b) — for a SIESTA run, in its output
or in the stderr its wrapper kept, the session log whose first section is the
run: a rank other than 0 that dies may say why only there. The detail quotes
the line.

**Whether the run ended on its own is one door's answer**, `runrecord.ending`
([`architecture.md`](?doc=execution/architecture.md) § 3.2), the same for
status, every hand-over and launch: molbuilder's conclusion marker, counted
only at the run's newest try, the highest index its files reached
([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.1) — an earlier
one, beside a newer output, is an earlier try's goodbye. Nothing else
answers: SIESTA's own `0_NORMAL_EXIT` says the engine ended, not the job
*(it answered for a run started by hand until 2026-10-03)*. **Finished is a run that ended on its own with exit code 0;
failed, one that ended with any other** — or whose output states a stop.
**An output that states its end is not a run that ended**: the job may still
be on its way out — a finish deriving its result, the wrapper's last lines —
so it reads `running` until it concludes, and `failed` once its scheduler's
own line says it ended the job, or the monitor's closing record says the
process went without concluding. The monitor's closing record —
`[MONITOR] job ended` in the latest run's `-runN.monitor.log`
([`run-reports.md`](?doc=execution/run-reports.md) § 2.5) — speaks wherever
the conclusion and the scheduler are silent: the process went, and nothing
recorded an exit. *(Until 2026-10-03 the output's end decided first: a run
whose output ended read `finished` with no conclusion, and beside an exit code
of 1, while every hand-over refused it — plan W38 F3.)*

**Whether its scheduler ended it is the scheduler's own word**
([`scheduler.md`](?doc=execution/scheduler.md) R16): for a job sent to a
queue that has not ended on its own, the line its scheduler writes in the
job's error file when it stops it — the file the run's launch record names
(`scheduler.stderr`), read by its kind's reader — reads `failed`, *stopped by
the scheduler: CANCELLED AT 2026-10-18T01:19:15 DUE TO TIME LIMIT
(slurm.65999920.err)*, the scheduler's words and its file. It is asked before
the monitor's closing record because it does not depend on a signal reaching
anything: on Sol's job 65999920 neither the wrapper's trap nor the monitor
recorded the SIGTERM at its wall, and only the scheduler's line said the job
was over. A run here is not asked; a record launched before it named the
file (2.12.0) is asked where its header put it, the run's own folder under the
file catalogue's name (`slurm.<id>.err`).
A run of points' point asks it with its run's one record, from the run's
folder, where the record lies and the file it names is named from
(`rundir.run_state_of`, given the run's folder — the one door every reader
of a point's state asks) — so a sweep the scheduler stopped reads `failed`
at every point not done, the ones the walk never reached too.

**A job with a finish is not done when its engine is**
([`engines/vibration.md`](?doc=engines/vibration.md) § 5.5, § 5.11 § 3): a
SIESTA vibration's last call's record step runs the finish after SIESTA ends,
and only then concludes. So an output that states its end speaks for the ENGINE, and
the job's own evidence decides: a marker naming a failed finish reads
`failed`; with no marker yet, the run's session log saying the step after the
engine began (`finish started:`, `wrapper_log.FINISH_STARTED` — a call's plan
step, before its engine, logs `plan step:` and never this) reads `running` — *the
engine ended; the job's finish is deriving the result* — and `failed` once the
scheduler's line or the monitor's closing record says the job went: a
walltime or a kill inside the finish, which leaves no marker. A marker `rc=0` is a finished job.

**Convergence never decides the state** (P-S2): an unconverged SCF is a fact
beside it, in `endings`. A run SIESTA *stopped* because its SCF had to converge
is `failed` by the stop; a capped benchmark that ran to its end is `finished`.

#### The states

| state | detail |
|---|---|
| `pending` | prepared, not launched (no run.json) |
| `queued` | queued as job N · launched (direct), no output yet — and, on the Results tab's ladder row, its **standing** where this machine sent the job — its address on this machine's jobs list, the scheduler asked on the cluster the job was sent to as the scheduler names it ([`job-system.md`](?doc=execution/job-system.md) § 6.0a; `joblist.address_of`, `asked_here`: the one rule `molbuilder jobs` asks by): `PENDING (Priority) · expected to start 2026-10-19T14:02:00 · priority 1234`, the start as the scheduler prints it, the kind's `standing` query ([`scheduler.md`](?doc=execution/scheduler.md) § 6a) through the run door (`runs.standing_of`); from any other machine the row says *ask on <machine>* |
| `running` | running · no result file yet · the engine's output ended; the job has not concluded · the engine ended; the job's finish is deriving the result — **where the run is**, in the reading pass's words where its output states it (`model/parse.md` § 2b, `doing`): *running · negf SCF iteration 12*, and until a device's first SCF row its set-up, *running · computing the R lead's surface Green's function (estimated 49.170 GB)* — the same words the monitor's status line says; a sent run's bare *running* says too how long since the job itself last wrote, *running · last wrote 2 h 10 min ago*: the one fact a folder gives where its scheduler cannot be asked · *the scheduler lists it: PENDING (Requeue)*, a requeued job past the wall rule's bound, on the machine it was sent to |
| `finished` | job_completed · concluded (rc=0 at …) |
| `failed` | stopped by the scheduler: *its words* (*its file*) · stopped before its end: *the line that stopped it* · out of memory: *its line* · concluded (rc=1 at …) · the engine's output ended, but the job exited with an error (rc=1 at …) · stopped before its end: no ending in its output and no exit recorded · stopped before its end: the scheduler no longer listed job N (asked on *machine* at *when*); no exit recorded · the scheduler listed job N ended: *its words* (asked on *machine* at *when*) · stopped before its end: nothing was written for longer than its wall (*wall*; last write *when*), so the job is over; how it ended was not recorded · the engine's output ended, but the job stopped before it concluded — no exit recorded · the engine ended, but the job's finish did not derive the result (*the marker*) · the engine ended and the job's finish began, but the job stopped before it concluded · concluded (rc=1 at …; *its words* — finish cannot load (*bundle*), no env python (*path*)) before any output |

**Silence is not death.** A healthy SIESTA SCF step can print nothing for over
twelve minutes, and a job the scheduler kills leaves no trace in its output, so
a file with no ending is `running` — not finished — however long it has been
quiet *(user, 2026-09-26: "It shows what it is")*. A forced stop writes no
marker; what says it is over is the scheduler's own line, where it wrote one
(R16 there), else the monitor's closing record, written when the monitor saw
its PID go or got its signal — *failed — stopped before its end: no ending in
its output and no exit recorded*, here and in its own `finish`
([`run-reports.md`](?doc=execution/run-reports.md) § 2.4). A stop neither
records — a lost node, a disk that filled under every writer at once (Sol's
job 66108241, 2026-10-19: the engine, the wrapper and the monitor all died
with nothing written) — would read `running` for ever from the folder alone.
Two more things say it is over, honestly *(user, 2026-10-19: "if the machine
knows a way to probe status of a job, put it down with a status record so it
does not have to be repeated; if not, just check time stamp")*:

- **Asked once, written once.** Where this machine sent the job — its
  address on this machine's jobs list, the scheduler asked on the cluster
  the job was sent to (`job-system.md` § 6.0a; `joblist.address_of`,
  `asked_here`: the one rule `molbuilder jobs` asks by, as the standing is
  asked, § 4.2 above) — every reader of a sent run that still
  reads live — `status`, a launch again, the Results tab — asks the
  scheduler whether it still lists the job (the kind's `listed` query,
  [`scheduler.md`](?doc=execution/scheduler.md) § 6a). A job it no longer
  lists, or lists ended, is **written into the run's launch record** (`ended`:
  the machine asked, when, and the scheduler's words — `runrecord.record_ended`,
  [`project-layout.md`](?doc=execution/project-layout.md) § 1.6.3) by that
  first look, and the run reads `failed` from then on, wherever the folder
  is copied: *stopped before its end: the scheduler no longer listed job N
  (asked on sol at …); no exit recorded* — the cluster as the scheduler
  names it, which the job's address records (a job whose cluster was not
  read when it went is not asked: another cluster sharing this home would
  answer for an id not its own). A run whose job has no address here —
  sent from another machine — is not asked and reads as its folder says;
  `molbuilder jobs status` removes an ended job's address only after its
  members' look, which writes it, and that one answer serves both
  (`joblist.asked_lately`). An answer that cannot be written — a full
  disk — is said on the run's line, never a crash. The question is asked only while
  the record holds no answer, so the folder is concluded once (`runs.concluded`,
  the door every reader of a run's state asks). A job the scheduler still
  lists is left as it reads; nothing is written for it.
- **Past its wall, wherever it is read.** A sent job cannot run longer than
  the wall it was sent with (`run.json`'s `placed_on.time`), and every file
  it writes is written while it runs — so a run whose **newest own write**
  is older than that wall is over, whichever machine reads the folder and
  whether or not its scheduler can be asked: *failed — stopped before its
  end: nothing was written for longer than its wall (1-00:00:00; last write
  2026-10-19T16:53:10Z), so the job is over; how it ended was not recorded*.
  The job's own writes are its outputs, the engine's files, its wrapper's
  and monitor's logs — the files of this run's label and stage
  (`runfiles.parse`, `split_stem`); never molbuilder's own records of it
  (`run.json`, `.continued-from`, `.gathered-from`) or the ledger. The bound is the
  scheduler's own, so this never calls a running job over — with one case
  the folder cannot tell: a job the scheduler **requeued** (a node lost;
  SLURM's `JobRequeue`) keeps its id and waits again with its first start's
  files in the folder. On the machine it was sent to the scheduler is asked
  for such a run too, and a job it still lists reads `running — the
  scheduler lists it: PENDING (…)`; from any other machine the wall rule
  stands until the job writes again. Before the wall, the run stays
  `running`, and its row says how long since the job last wrote — stalled
  or slow is yours to read, molbuilder states the fact. On a server the
  scheduler is asked at most once per ten seconds per job, however many
  points or members share it.

Launching it again is yours
to decide: `launch task` does it as asked, warm or `--cold`, however the last
run ended ([`job-system.md`](?doc=execution/job-system.md), *What molbuilder
does for you*, 3).

**Before the first output**, a direct launch reads `queued` (its `run.json` is
written as the process starts). One stopped before writing anything reads
`failed` by its scheduler's line or its monitor's closing record — a walk's
member the job never reached too, its job's line naming it — or `queued`
when nothing says so; an engine that merely died still reaches the wrapper's marker and reads
`failed`; a job with no python of its env, or one that cannot run its
finish, stops before its engine and says so in its marker (`no env python`,
`finish cannot load`), which reads `failed`.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> pending: prepared
    pending --> queued: run.json written
    queued --> running: first output
    queued --> failed: nonzero marker, the scheduler's line or the monitor's closing record, no output
    running --> finished: an end (with a finish, its marker rc 0), else marker rc 0
    running --> failed: a stop, a failed finish, else nonzero marker, else the scheduler's line, else the monitor's closing record
    note right of running: no clock moves a run out of here
```

#### The words on each surface

| surface | its words | read from |
|---|---|---|
| a file — `run_state` ([`model/parse.md`](?doc=model/parse.md) § 2b) | `running` · `ended` · `stopped` · `out_of_memory` · `unknown` | the file's markers |
| a run directory — `run_status` | `pending` · `queued` · `running` · `finished` · `failed` (and `unreadable`, asked through the run door with a launch record that does not read) | `ended` → `finished` — for a job with a finish, once its marker says `rc=0`: a failed finish → `failed`, a finish begun and unconcluded → `running`; `stopped`, `out_of_memory` → `failed`; otherwise the marker, else `running` |
| the Run panel ([`web/results.md`](?doc=web/results.md) § 3a) | `run_status`'s | the same scan |
| a ladder row — `jobset status`, the Results ladder ([`web/results.md`](?doc=web/results.md) § 2.4) | `run_status`'s for a prepared stage, plus `missing` (its folder is not on disk) and `unreadable` (its launch record is there and does not read); for a prepared stage not launched, the ready door's *ready to launch* — *(prepared by molbuilder X.Y.Z)* when another molbuilder prepared it — / *waits for*; *changed since prepared* and each value that differs ([`job-system.md`](?doc=execution/job-system.md), *Changed since prepared*); for one not prepared, `described` — or *waits before it can be prepared*, with what for (`job-system.md`, *The task*) | `jobset/runstatus.py` |
| the trajectory badge ([`web/trajectory.md`](?doc=web/trajectory.md) § 4) | Running · Finished · Stopped | the open file's `run_state`: `ended` → Finished; `stopped`, `out_of_memory` → Stopped; else Running |
| the monitor's closing line ([`run-reports.md`](?doc=execution/run-reports.md) § 2.3) | `finished` · `failed` | `run_status`'s, asked after it writes its closing record |

The badge reads a **file** and the panel the **run**, so they can differ: a
`-run0.out` that stopped reads Stopped while its warm retry keeps the attempt
`running`.

#### Worked examples

The attempt `01_coarse/run-0/` of `bdt`, as prep left it
([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.3), then as the
run goes:

| it also holds | `state` — `detail` |
|---|---|
| nothing more | `pending` — prepared, not launched (no run.json) |
| `run.json`, `"job_id": "481923"` | `queued` — queued as job 481923 |
| `bdt_01_coarse-run0.out`, no `>> End of run`, quiet for 40 minutes | `running` — running |
| the `.out` ends `>> End of run`; `-run0.concluded` reads `rc=0 at …` | `finished` — job_completed |
| the `.out` stopped on `SCF_NOT_CONV … (required)`; the warm retry's `-run1.out` is printing | `running` — `-run1.out` speaks: same stage, newer |
| killed at its wall: no ending in the `.out`, no `.concluded`; SLURM's `*** JOB 65999920 ON scc006 CANCELLED AT 2026-10-18T01:19:15 DUE TO TIME LIMIT ***` in `slurm.65999920.err`, which `run.json`'s `scheduler.stderr` names | `failed` — stopped by the scheduler: CANCELLED AT 2026-10-18T01:19:15 DUE TO TIME LIMIT (slurm.65999920.err); launching it again, warm or `--cold`, is yours |
| stopped with no scheduler line, its monitor's closing record there | `failed` — stopped before its end: no ending in its output and no exit recorded |
| no `.out` (SIESTA died before its first line); `-run0.concluded` reads `rc=1 at …` | `failed` — concluded (rc=1 at …) before any output |
| `-run0.concluded` is there and does not read (unreadable, or not text) | `failed` — the run concluded, the wrapper wrote it, with an exit code no one can tell: `rc=? (… does not read: …)`, as an empty marker's `rc=?` (`runrecord.ending`) |

#### Which engine ran — `engine_of`

`parse.contract.engine_of(run_dir)` answers `"siesta"` / `"pyscf"` /
`"unknown"`, and this section owns the rule.

**The engine is DECLARED when the script is generated**, because that is the
only moment it is known for certain, and a run directory gets copied away
from everything that knew. Two kinds of evidence exist, and they are not a
precedence list:

| tier | evidence | where it is written |
|---|---|---|
| **declaration** | the PROVENANCE `engine` key of any deck or wrapper | [`job-contracts § 3.2`](?doc=execution/job-contracts.md) |
| **declaration** | the `.molwatch.log` `# engine:` header | [`engines/pyscf.md`](?doc=engines/pyscf.md) § 4 |
| *fallback* | which files are present | only when nothing declared |

**The declarations are weighed TOGETHER.** One distinct answer among them is
the answer. Two is a run that contradicts itself, and that is `"unknown"` —
the same rule and the same reason as `contract_of`
([`model/parse.md`](?doc=model/parse.md) § 5b): a directory that says two
things cannot be made to say one by picking, and an answer that might be the
other engine's is worth less than no answer. No order is used, because a
first-hit list lets one stale `.run.sh` outrank two agreeing declarations.

**The sniff is consulted only when nothing declared**, for a directory
molbuilder did not write, and never overrules a declaration: files outlive the
run that wrote them, so a stale `.fdf` beside a freshly re-prepared PySCF deck
is litter, not a second opinion.


## 5. Configuration — `molbuilder.json`

**What `molbuilder.json` may hold, and what each key is for, is
[`configuration.md`](?doc=configuration.md) § 4** — the one list, with the
retired keys and what to do instead. It holds your preferences and nothing
else: **no value of a job** (its queue, wall, memory, ranks, cores per rank and
GPU count are the job's own — [`architecture.md`](?doc=execution/architecture.md)
§ 5.2) and **no fact about a machine** (its queues and topology are that
machine's record — `configuration.md` § 5) — except `env_init`, how a shell
enters an environment HERE, which the probe copies into every record it writes
(§ 5.2). An unknown
top-level key is refused with the known sections named, never ignored; a key
starting with `_` is a comment.

**Two parts of it reach a calculation**, both below: `launch` and `envs`. The
same file also configures the *server* — sign-in, TLS, the rate limiter, the
admin list ([`ops/deployment.md`](?doc=ops/deployment.md) § 5,
[`ops/access-control.md`](?doc=ops/access-control.md)).

### 5.1 Where config lives

- **Server-wide** `molbuilder.json` — **one file**, in the config directory:
  `$MOLBUILDER_CONFIG_DIR` if set, else `$XDG_CONFIG_HOME/molbuilder/`, else
  `~/.config/molbuilder/`. There is no search and no working-directory step; a
  `./molbuilder.json` is not read (`configuration.md` § 2.1a).

### 5.2 The activation — required to emit ANY wrapper, and a fact of the machine

How a shell enters an environment on a machine — the `preamble` (e.g.
`module load mamba/latest`, or sourcing conda's hook) and the `activation`
(`source activate` or `conda activate`) — is **a fact of that machine**, carried
by its record (`configuration.md` § 5 M-1). Every wrapper is rendered with the
TARGET's two, this machine included.

**They are declared once on each machine molbuilder is installed on**, in that
machine's own `molbuilder.json` — molbuilder needs them there before any record
exists *(user, 2026-10-02)*:

```json
"env_init": {"activation": "source activate",
                      "preamble": "module load mamba/latest"}
```

and **`jobset probe --write` copies them into every record it writes** — this
machine's `environment.json`, or a named `<name>.json` to carry to where you
prep. A copy that is wrong for the machine it describes is edited by hand, in
that record; an edit to `molbuilder.json` reaches a record at its next `probe
--write`, which asks about each difference (`--yes` takes them all; silence
keeps the record's).

`activation` must be `"source activate"` or `"conda activate"` and has **no
default**: a target whose record carries none is refused at prep, saying where
to declare it. On a fresh install that would bite a workstation first — which
is why **`envs init-config` asks** how this machine enters a conda env and writes
the answer into `molbuilder.json`, at the one moment it is both known and being
discussed ([`ops/installation.md`](?doc=ops/installation.md) § 2.1). It is
declared, never detected. `preamble` is arbitrary shell run before activation,
emitted verbatim.

### 5.3 The `.sbatch` header — every value stated

**There is no scheduler configuration.** A `scheduler` block in
`molbuilder.json` is refused by name (`configuration.md` § 4): each value in
it would be one some job received without stating it.

**What the `.sbatch` header carries** (`render_sbatch`): a fixed `-J <basename>`,
`-N 1`, `-o slurm.%j.out` / `-e slurm.%j.err`, and the job's own stated values —
`-n <ranks>`, `-c <cores per rank>`, `-t <wall>`, `--mem`, and `-p` / `-q` from
the queue it names, bound on the target's record; for GPU jobs
`--gres=gpu:<count>` and — unless the
calculation's `allocation.gpu_binding` is `false` — `--gres-flags=enforce-binding`
([`gpu.md`](?doc=execution/gpu.md) G9). Where each value is stated is
[`architecture.md`](?doc=execution/architecture.md) § 5.2, and prep refuses one
stated nowhere. The body says once what the scheduler tells the job, in
molbuilder's names (`MB_JOB_ID`, `MB_NTASKS`, …, [`scheduler.md`](?doc=execution/scheduler.md)
§ 6a), then a single line — `bash <basename>.run.sh "$@"` — because the inner
wrapper owns activation and launch, and reads only those names, so it is one
file with or without a scheduler. It is withheld only where the target's
record names a scheduler with no queue
([`job-system.md`](?doc=execution/job-system.md) § 6), and then only the
`.run.sh` is written.

#### 5.3.1 Memory and wall — the user states them, nothing else does

**No value is filled in** *(user decision, 2026-08-24)*: five Sol jobs
(62039301–05) OOM'd against scheduler defaults while a memory model sat
unconfigured and silent, and a model that answers a question the user was
never asked is the wrong shape regardless of its coefficients. What a run of
a stage measured is shown beside the ask — each stage's need, the run it was
measured on and how — and fills nothing
([`job-request.md`](?doc=execution/job-request.md) § 3.1a; the user,
2026-10-20).

**And no default exists either** *(user, 2026-10-02)*. A job's memory is, from
strongest: `--mem` at prep, the job request's `allocation.mem`. Its wall is,
for a submitted job, `--time` at launch, then `--time` at prep, the run card's
`time`, `allocation.time`. **Stated nowhere, prep refuses** on a target
with a scheduler. The scheduler's own default would stand in if prep let it
through, and that is a value nobody stated too.

`--mem 0` asks for the node's whole memory.

### 5.4 `launch` and `envs`

- **`launch`** — `{mode}`. `mode` is `direct` (run in place) or `submit`
  (through the scheduler); any other value, and any other key, is refused by
  name (`runtime_config.get_launch_mode`). This, not the detected scheduler, is what
  gates `.sbatch` submission.
- **`envs`** — overrides the conda env name per category
  (`{"siesta": "my-siesta-env", …}`); unset categories use the defaults
  (§ 2.3).

Config is written at mode `0600` by `write_config_scope` (deep-merge a patch
onto the one file, re-validate, then write it atomically —
`configuration.md` § 2.3).

### 5.5 The launch door — who may start a run, and how the run proves it

*Decided with the user, 2026-08-12. One section for the whole story: which
files decide the launch, how the decision reaches the running job, and what
the job's own log says about it.*

**Two files feed every launch — this machine's config and the target's
record — and each value knows where it came from:**

| file | scope | found where |
|---|---|---|
| `molbuilder.json` | **this machine** — `launch.mode`, the environment names, `env_init` | the config directory: `$MOLBUILDER_CONFIG_DIR`, else `$XDG_CONFIG_HOME/molbuilder/`, else `~/.config/molbuilder/` |
| the target's record | **the machine the job runs on** — its queues, topology, and the `env_init` copied from it | `environment.json`, or `environments/<name>.json` for a named target ([`configuration.md`](?doc=configuration.md) § 5) |

`prep` and `launch` print the provenance — every path consulted, found or
absent, and each effective value tagged with its source file — and `prep`
writes the same block into `STAGE-PLAN.md`, so a behaviour difference between
two machines is explained by the bundle itself. Secret sections (`auth`,
`tls`) are excluded by an allowlist, never by care.

**There is ONE launch door.** `molbuilder jobset launch` resolves the mode
(flag, else `launch.mode`, else a refusal — never the detected scheduler),
decides everything before it writes anything — what the run follows, and
the queue `prep` recorded for the work (a run
on a scheduler is refused at prep when it names none; launch places nothing
again) — then **shows the exact `sbatch` line and every value of the
header it sends, and asks** ([`submission.md`](?doc=execution/submission.md) S4; `--yes` skips
the question, never the output), records the attempt, and launches **one job
per invocation** — a grouped bench one per resource shelf
([`job-system.md`](?doc=execution/job-system.md) § 7). **It hands each run
script its first try's number**, `--run 0`
([`project-layout.md`](?doc=execution/project-layout.md) § 1.6.1): a run
script refuses to start without one, naming this door. The single-stage door
asks too: `--time` may restate a submitted job's wall after `prep`'s
printout, so the line as sent is seen only here. **A run here starts in the background**, its question offering the
foreground, `--foreground` the waited-on run
([`job-system.md`](?doc=execution/job-system.md) § 6.0). A flag the launch
would not read is refused by name: one the mode does not read — `--time`
under `--mode direct`, `--foreground` under `submit` or `ask` — with the
launch as typed for each mode; one the launch's kind does not read — a
benchmark's `--only` and `--trial-timeout` on `launch task`, a stage's
`--cold`, `--from` and `--gather` on `launch bench` — with *`launch <kind>`
does not read it*. One table, `commands.LAUNCH_FLAGS`
([`job-system.md`](?doc=execution/job-system.md) § 6.0, R-L4).

> **`--mode ask` submits nothing and tells you when it would start.** It walks
> the identical path `--mode submit` walks and inserts its scheduler kind's
> ask (SLURM's `--test-only`, [`scheduler.md`](?doc=execution/scheduler.md)
> § 6a), so the line asked about — the header its prep wrote — **is** the line
> that would be sent. The scheduler validates the request and predicts a start time; no job is
> created, nothing is recorded, and `status` sees nothing — because after it,
> nothing exists.
>
> It is for the minute before you commit: ask, and if the wait is poor, change
> the wall (`--time`) and ask again, or prepare it again on another queue or
> with less, or decide you can live with it and re-run with `--mode submit`. It needs a login node — **there is no
> prediction without the cluster**, and node counts are a poor proxy for it:
> 134 wide nodes are no help if all 134 are busy for two days.
>
> When SLURM declines to predict, that is reported as **unknown**, never as
> soon. A missing answer dressed as a good one is how you wait a day for a
> queue that looked instant. The reason it gives is printed, because it is
> often the whole answer — *"Requested node configuration is not available"*
> says the ask fits no machine in that queue.
>
> **It is not gated one-job-at-a-time, and `submit` is.** That rule exists
> because jobs queued together start together and contend — *"a rule about the
> scheduler, not about doing several things"*, which is why `--mode direct` is
> untouched too. Asking enqueues nothing, so none of that is reachable.
>
> A sweep is where it pays: a grid's shelves ask for different shapes, `G1`
> schedules sooner than `G4`, and seeing their waits side by side is what tells
> you which one to submit. It is asked **as `submit` sends it** — one question
> per resource shelf, each with the shelf's own header, its prep's. A stage
> launched before is asked about as the attempt its re-launch would open —
> nothing is opened by asking. The number of queries is capped for politeness, and
> anything past the cap is **named as unasked** — a partial answer that does
> not say it is partial reads as a complete one. When it launches, it stamps the claim
`MB_LAUNCHED_BY=jobset-launch` — into the child environment for a direct
run (inheritance survives `nohup` and backgrounding), and **explicitly on
the line** for a scheduler run — its kind's words, SLURM's `--export
ALL,MB_LAUNCHED_BY=jobset-launch` ([`scheduler.md`](?doc=execution/scheduler.md)
§ 6a), which beats any site export policy.

**Every `.run.sh` gates on that claim** before doing any work
(`job-contracts.md` § 2.6, the Launch-door gate row):

```mermaid
flowchart TD
    A[".run.sh starts"] --> B{"MB_LAUNCHED_BY set?"}
    B -- "yes (jobset-launch · manual · bench-runner)" --> C["log: launched-by: &lt;value&gt;<br/>proceed"]
    B -- no --> D{"interactive terminal?"}
    D -- yes --> E["warn, ask y/N"]
    E -- y --> C2["log: launched-by: manual<br/>proceed"]
    E -- "n / EOF" --> F["log: launched-by: NONE -- refused<br/>exit 2"]
    D -- "no (nohup · cron · hand-sbatch)" --> G["log: launched-by: NONE -- refused<br/>exit 2, message names the door<br/>and the override"]
```

**The verdict is in the job's own output log AND the runwrap log**: the
wrapper opens its per-start log (tee) *before* the gate, so every outcome —
proceed, manual yes, refusal — is a fact on disk even when nothing ran;
under sbatch the `launched-by:` line also lands in the job's `.out`. Either
log alone answers *"was this launched properly, run by hand on purpose, or
refused?"* — and nothing can ever sit waiting on a prompt, because a
non-interactive shell never prompts, it refuses (EOF at the prompt is a
refusal too, with the same verdict line). Four edges repaired 2026-08-12
(U10): the interactive **yes is `export`ed**, so the warm-retry re-exec of
the same wrapper does not re-prompt mid-retry; EOF falls to the refusal
instead of dying under `set -e` with no verdict; `-h`/`--help` is scanned
**before** the gate and runs none of the bootstrap — asking what a script
does needs neither a claim nor a working activation; and the refusal
reaches the runwrap log as above. The deliberate manual form
is `MB_LAUNCHED_BY=manual bash JOB.run.sh --run 0` (backgroundable), started
in the run's own folder, `run-<n>/`, where launch copied it — the script
needs its try's number by hand as well — and the value is logged so
the choice is on record. *(`bench-runner` was the transitional
claim of the old bench launchers; they died at step 6 u5, 2026-08-12, and no
shipped script emits it any more.)*

---

## 6. Saving a calculation — `molbuilder checkpoint`

A calculation folder can be put under a git-backed snapshot system, so any state
you saved is one you can come back to — rerun a stage, retune and try again, or
start over from it. Large binaries are handled beside git rather than inside it,
so a snapshot holds the density matrices too.

**This section is the guide: what to type, and what the buttons do.** The rules
it must not break — and which of them hold today — are
[`checkpointing.md`](?doc=execution/checkpointing.md); the file formats are
[`job-contracts.md § 6.1`](?doc=execution/job-contracts.md).

### 6.1 Three ideas

| | |
|---|---|
| a **state** | a saved snapshot of the whole folder: an id, a note you wrote, and the state it came from |
| a **tag** | a name you give a state so you can find it again |
| **where you stand** | the one state the folder is currently at |

**Where you stand is what makes the other two work.** It decides what "unsaved"
means — the folder differs from *that* state, never from the newest one — and it
decides where a new state hangs: `save` records where you stood as the new
state's parent. That is the whole of branching. There is **no branch verb**; you
go back to a state, save from it, and both attempts stay listed.

```mermaid
flowchart TD
    RD["the calculation folder"]
    G[".git/<br/>everything small:<br/>.fdf .out .XV .CG run.json"]
    B[".binsnapshots/&lt;digest&gt;/<br/>whole copies of everything large<br/>+ MANIFEST.do_not_edit"]
    S["an engine's scratch<br/>(TranSIESTA's *.TSGFL, *.TSGFR;<br/>an optical call's *.contrib.EPSIMG)<br/>in neither store; a restore removes it"]
    RD --> G
    RD --> B
    RD -.-> S
```

Which store a file goes to is decided by **measuring it** against a size limit —
10 MB by default, set in `molbuilder.json`. Every file is in exactly one of
the two stores, but an engine's scratch: the files the engines' own
vocabularies name `scratch` — every engine's, whatever engine the folder
records — recomputed by every run, are in neither, never reported unsaved,
and removed by a restore like every file the target does not hold
([`checkpointing.md`](?doc=execution/checkpointing.md) S1;
[`job-contracts.md`](?doc=execution/job-contracts.md) § 4.2a). `jobset
scratch` lists them and `--empty` removes them
([`job-system.md`](?doc=execution/job-system.md) § 5.3).

### 6.2 The verbs

```bash
molbuilder checkpoint init                                  # once, in the folder
molbuilder checkpoint save -m "stage 1 converged, 41 steps"  # the note is required
molbuilder checkpoint list                                   # what have I got?
molbuilder checkpoint tag stage1-good -m "geometry I trust"  # name one
molbuilder checkpoint restore 4f9ca71                        # or restore stage1-good
molbuilder checkpoint config                                 # which files count as big
```

Every verb takes `-p/--path` (default: the current directory).

- **`init`** — `--engine siesta|pyscf` names the engine whose config entry's
  always-large families skip the measuring, recorded in the folder's own
  `.git/config` — on a folder that is already a checkpoint folder too, when it
  records none. Omit it and every file is measured, which is always correct
  and merely slower; a history that a `prep`, Task setup's Save or a group's
  first launch begins records the calculation's engine with its first state
  (`checkpoint.save_before`, `Repo.name_engine`); an engine once recorded is
  never replaced. An engine's
  scratch is never stored, whatever the folder records. `--calculation` sets the
  name written into every state; it defaults to the folder's, and a name that
  would need repairing is **refused** rather than quietly fixed.
- **`save -m`** — the note is required and never generated. It is the only thing
  that answers the question you actually bring to a history a month later: *why
  did I stop here, and what was I about to do?* Says plainly when nothing
  changed rather than inventing a state.
- **`list`** — newest first, each state naming the one it came from. Two states
  showing the same parent are alternatives. It answers **cheaply**, from size and
  timestamp, and says so; `--check` compares content when you want certainty now.
- **`tag NAME -m`** — the note says why the state is worth returning to. Nothing
  tags on your behalf, so the namespace is yours alone. `--at` names a state
  other than where you stand.
- **`restore STATE`** — STATE is a state id or a tag. The **whole folder**
  returns; it is a rewind, not a fetch. To read one old file without moving
  anything, there are two commands and which one you want depends on the file's
  size. A small one is in git:

  ```bash
  git show <state>:<path>
  ```

  A large one is not in any commit — it lives in the side archive — so git
  answers `path '…' does not exist`. Read the state's message for the archive
  it names, then read the file straight out of it:

  ```bash
  git show -s --format=%B <state>      # the `Manifest-SHA256:` line is the archive
  cat .binsnapshots/<digest>/<path>
  ```
- **`config`** — prints the size limit, which families skip the measuring,
  and, on the line `in neither`, the engines' scratch — never saved, removed
  by a restore, named by the engines' own `warm-files.toml` and not by
  `molbuilder.json` — then where to change the first two. Read-only: the
  classification has one home.

**What a restore asks you.** It refuses first on things about the *target* — an
unknown state, or an archive that does not verify — because nobody should accept
a loss for an operation that then fails for another reason. Only then does it
name everything unsaved (changed, added and deleted alike) and ask. At a
terminal you answer; a script passes `--force`. **Say yes and it is gone**:
nothing is stashed, renamed or set aside. Files merely absent from the target are
removed without a warning — they are still in the state that holds them. An
engine's scratch is no file of any state, so a restore never names it as
unsaved and removes it like any file the target does not hold: a device run's
`tsgf/` comes back holding only its marker where the target holds that run,
and a run folder the target never had is removed whole (S1).

### 6.3 The panel, and the routes

The projects sidebar's run-history panel does the same work for a run directory:
a sensor pill reading `saved` or `N unsaved`, the states as a list or a graph
drawn from parentage, and buttons for init, save, tag and restore. **Refresh is
explicit** — it reads on directory-enter and when you press Refresh, which asks
the exact content question rather than the cheap one. There is no polling.

Over HTTP: `GET /api/checkpoint/state`, `list`; `POST
/api/checkpoint/init`, `save`, `tag`, `restore`
([`web-api.md`](?doc=web/web-api.md) § 5). The panel and the CLI go
through one class, so a rule proved on one holds on the other.

### 6.4 One rule for you: use the verbs, not bare git

The folder **is** a git repository — that is how the snapshots are made — so
nothing stops you, and molbuilder does not try to. But git alone sees half of
it: the big files live in the archive, which git is told to ignore, so a `git
checkout` of an older commit rewinds the text and leaves every large file where
it was. The folder is then in a state no save ever produced, and **that mess is
yours**. You will not be quietly fooled, though: the next restore checks content,
refuses, and names the files that differ.

### 6.5 What is not built, and what is not going to be

**Retired 2026-09-03 — decided, not deferred** *(user: "retire all of them")*.
These were named in the original design, never built, and had no recorded
decision either way, so they went on reading as *coming*. They are not:

- **`checkpoint diff`** — no verb, on any surface, and none is planned. The
  commit-graph viewer in the sidebar is how two states get compared.
- **`prune`** — under *"every saved state stays restorable"* almost nothing
  can be reclaimed, so the verb would mostly refuse. What no state holds — an
  engine's scratch an older save archived — a save clears from every archive
  by itself, at the end of the save, and records that it did, so the next
  saves skip the walk (`checkpoint.prune_scratch`,
  [`checkpointing.md`](?doc=execution/checkpointing.md) I1); nothing else is
  reclaimed. Delete the folder if you want the space.
- **The `snippets/` library** — reusable deck fragments. The template and its
  `overrides` are how a calculation varies; a second vocabulary for the same
  job is what this contract exists to prevent.
- **Wrapper-git "Path B"** — an alternative checkpoint mechanism, drafted
  beside the one that shipped. One mechanism is the point.

**Still open, and deliberately so:**

- **`checkpoint verify`** — the archive check exists and is reachable only by
  attempting a restore, which is the worst moment to learn an archive is gone
  (plan § 2, W45).

`prep` saves the folder's state before it writes, always
(`checkpoint.save_before`; [`job-system.md`](?doc=execution/job-system.md)
§ 5.0), so a rerun is never yours to remember to save before.

## 7. A note on the design that superseded the cookbook

The original single-job doc carried a benchmark/sweep cookbook and two large
implementation-design sections. Those have been overtaken: the batch/sweep,
HPC-deployment, and benchmarking workflow is now the **JobSet framework**
(`execution/job-system.md`), which drives the whole matrix through one entry
point rather than shipping a standalone `mbbench/` library or static
per-point scripts. The single-job wrapper on this page is the primitive that
framework runs; the framework, the `(G, K, c)` grid, routing domains, and
submit-vs-direct execution are documented there.  *(Until 2026-08-13 this
paragraph still promised a "self-bootstraps molbuilder on the target"
mechanism and a `bench-manifest@2` artifact — both retired; job-system § 7
records their retirement itself.)*
