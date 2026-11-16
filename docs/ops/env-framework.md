# The env framework — one registry, one runner, one outcome

**Role:** contract
**Domain:** ops
**Companions:** [`installation.md`](?doc=ops/installation.md) — the GUIDE a
person installing molbuilder reads; this page owns the machinery underneath it.
[`engines/template.md`](?doc=engines/template.md) — the catalogue, which is a
different registry for a different thing (calculation parameters, not env
packages) and must not be confused with this one.

## The short version

| rule | where |
|---|---|
| **Two state machines, and they answer different questions.** `EnvState` about the env before we touch it; `Outcome` about each step after we run it | § 2 |
| **The registry is the only place that says what an env is.** Recipes are data | § 3 |
| **The host env's package list is one file** — `molbuilder/envs/host-env.txt`, read by the bootstrap shim and by the registry; its `python` line is the python every env is built on | § 3.4 |
| **A fact about one package lives ON that package.** No parallel `optional_*` lists — membership-by-name silently mis-declares | § 3.2 |
| **A bare string is the ordinary case**, a record only when the package needs more than a name | § 3.1 |
| **The sequence is forced by dependency**, not by list order, and the document says which orderings may never move | § 4.1 |
| **Every situation is a FIELD on a record**, never a branch in the runner | § 4.2 |
| **Optionality means different things to a solver and to a per-package installer** — a non-fatal step for pip, a degraded attempt for conda | § 4.3 |
| **One door runs every step** — `run_step` — so no phase has a dispatch of its own | § 5 |
| **A step enters the env through the MANAGER's own mechanism**, and a manager's measured bug is worked around inside that one door, never on the default path | § 5.6 |
| **Five outcomes, one transition rule**, and every step in a result carries one | § 5.1 |
| **A step's success criteria ride on the STEP**, so the runner has no per-phase special cases | § 5.2 |
| **The verdict is DERIVED from the steps**, never tracked beside them | § 5.3 |
| **The audit reads disk, never the network**, and answers on identity *and* provenance | § 6 |
| **Repair re-reads the RECORD**, never a flattened copy of an instruction | § 7 |

```
recipes.py ── Recipe ──┬── CondaPackage(spec, optional, reason, opt_in)
  (the registry)       ├── PipPackage(name, source, extras, optional,
   § 3                 │              force, fallback_to_index,
                       │              reason, opt_in)
                       └── ExtraStep(argv, opt_in)              § 3.2a
                                    │
          ┌─────────────────────────┴──────────────────┐
   create_step_for · conda_packages_step_for ·        audit_packages
   pip_steps_for · pip_step_for · extra_steps_for ·   (reads disk only)
   verify_step_for · remove_step_for ·                           § 6
   conda_step_for                                            │
     conda_argv            pip_argv                          │
        (record  ->  InstallStep; the ONLY translators)      │
                    │                                        │
              plan_install            ── steps, no side      │
                    │                    effects (--dry-run) │
                    ▼                                        │
              ┌─ _run_steps ─────────────────┐               │
              │   one loop, called per       │               │
              │   stage; branches on         │               │
              │   StepRole, never on a label │               │
              └──────────┬───────────────────┘               │
                         ▼                                   ▼
                      run_step  ◄──────────────────────────  repair   § 7
                   (the one door)  ◄───────────────────  doctor verify
                         │                                         § 6
                   Outcome.decide      ── the one transition rule
                         │
                 OK · RECOVERED · DEGRADED · FAILED · SKIPPED     § 5.1
```

---

## 1. What this document owns

How an environment gets described, created, checked and repaired: the shape of
the registry, the execution model, and the contract between them. It does not
own which packages any particular env needs — that is the recipe — nor the
source-build machinery (`builds.py`), which has its own resume semantics.

## 2. The two state machines

Installing an env asks two different questions, and each has its own small,
closed set of answers. Keeping them apart is the point: one is about the env
*before* we touch it, the other about each command *after* we run it.

```
        ┌──────────────────── 1. EnvState ─────────────────────┐
        │   three observations  ──►  five states               │
        │                                                      │
        │     registry lists it?  ─┐      FRESH    PRESENT     │
        │     directory exists?   ─┼──►   ORPHAN   GHOST       │
        │     conda-meta/ there?  ─┘      BROKEN              │
        └───────────────────────┬──────────────────────────────┘
                                │  decides exactly ONE step
              ┌─────────────────┼──────────────────┐
           FRESH             PRESENT        ORPHAN/GHOST/BROKEN
              │                 │                   │
        dispatch create      SKIPPED              FAILED
                                      (ORPHAN, BROKEN: "rm -rf <prefix>";
                                       GHOST: "re-run --clean")
                                │
        ┌──────────────────── 2. Outcome ──────────────────────┐
        │   what became of ONE dispatched step                 │
        │                                                      │
        │     accepted, first attempt        ──►  OK           │
        │     accepted, a declared fallback  ──►  RECOVERED    │
        │     nothing accepted, optional     ──►  DEGRADED     │
        │     nothing accepted, required     ──►  FAILED       │
        │     deliberately not attempted     ──►  SKIPPED      │
        └──────────────────────────────────────────────────────┘
```

### 2.1 `EnvState` — is this env installable-into?

**The five are an enum (`EnvPresence`), and the label is derived from it**
*(2026-09-13)*: `can_resume` and `needs_cleanup` ask the enum, never a display
string, because a renamed label compared with `==` would turn `can_resume`
False, and the installer reads *"not resumable, not wreckage"* as "go ahead
and create".


Three cheap observations, combined into five states by `probe_env_state`. It is
a pure read, taken before any subprocess work, so a half-broken env is
diagnosed in a second instead of failing ten minutes into a `conda create`.

| registry | directory | `conda-meta/` | state | what the installer does |
|---|---|---|---|---|
| no | no | — | **FRESH** | dispatch `conda create` |
| yes | yes | yes | **PRESENT** | `SKIPPED` — resume into the existing env |
| no | yes | yes | **ORPHAN** | `FAILED` — the manager will neither create over it nor remove it; the remedy is the person's `rm -rf <prefix>`, then the install again |
| yes | no | — | **GHOST** | `FAILED` — a registry entry with no directory; `--clean` restores it (the removal is skipped, the create runs) |
| — | yes | no | **BROKEN** | `FAILED` — a directory that is not an env, which the manager will not remove; the remedy is the person's `rm -rf <prefix>`, then the install again |

**One reader of the registry, and it answers with prefixes** *(2026-09-12)*.
`diagnostics.conda_env_prefixes` is the only place `<mgr> env list --json` is
parsed, and it returns `{name: prefix}` — **with the manager's own names**, taken
from that document's `envs_details`, which also flags the base installation so it
is excluded by the manager's verdict rather than by a path pattern. A manager
that reports no such details has everything it lists taken at face value: an env
the registry names and this map omits reads as FRESH, and FRESH builds a second
env beside the real one. The probe, the capabilities snapshot and
`install._env_prefix` all ask this one reader, so they cannot disagree about
whether an env exists. `_env_prefix` consults the snapshot's reading first,
because a fresh read costs ~1.2 s and `doctor` asks once per recipe.

**"The directory" means the prefix the registry named**, not
`<envs_dirs>/<name>`. An env created with `--prefix` outside any of conda's
`envs_dirs` is listed by the registry *with* a perfectly good directory; probing
only the search path would call it GHOST and then print a removal command for
a healthy env, so the probe measures the registry's own prefix.

The five are **exhaustive over all eight combinations** of three booleans,
which is why the installer needs no default branch — and that exhaustivity is
load-bearing, not tidiness. A state nothing recognises would answer `False` to
both `can_resume` and `needs_cleanup`, and the installer reads that pair as
"go ahead and create" — the worst possible answer for an env that is already
wreckage. So the impossible case raises rather than returning a sixth label.

`--force-resume` overrides the probe: the operator asserts the env is usable
despite the reading, typically mid source-build where the directory exists but
`conda-meta/` has not been finalised yet.

### 2.2 `Outcome` — what became of one step?

Five states and **one transition rule**, which lives in `Outcome.decide`:

```
for each argv in (step.argv, *step.fallbacks):
    if the step ACCEPTS the result:
        return OK if this was the first argv else RECOVERED
return DEGRADED if the step is optional else FAILED
```

Naming the states keeps the rule in one place. Decided by nested conditions
instead (is the code zero, is it `None` because the process never launched,
are there alternatives, is the step optional), an edit loses a branch: a launch
failure skips the optional check, or a recovered step is reported under the
command that failed.

Two predicates read these states, and they answer **different questions**:

**One vocabulary for the five, on the state** *(2026-09-13)*: `Outcome.word` is
what both the live line and the recap print, with `Outcome.note` adding the
*why* where there is room, so the live line and the recap cannot name one
step's outcome differently.

| | asks about | answer for `DEGRADED` |
|---|---|---|
| `is_success` | did the env get what this STEP was for? | **False** — the package is honestly absent |
| `stops_the_install` | must everything after this be abandoned? | **False** — which is what `optional` buys |

`DEGRADED` is the only state where the two disagree, and that disagreement is
the entire point of optionality. Using the wrong predicate turns one
unavailable GPU wheel into a failed CPU env.

## 3. The registry

`molbuilder/envs/recipes.py` holds one `Recipe` per environment, and a recipe is
**data**: what the file says is what gets installed, and `molbuilder envs` reads
the registry and does not know how to guess.

**One fact of the host enters it: the CUDA version** the GPU packages target —
`MOLBUILDER_CUDA_VERSION`, else the major the driver reports (`nvidia-smi`),
else `13.*` (`_resolve_cuda_version`). `molbuilder-pySCF`'s `cupy-cuda<N>x` and
`gpu4pyscf-cuda<N>x`, and `molbuilder-siesta-gpu`'s `cuda-version=` pin and the
driver its precondition names, carry it — so those two recipes are functions
of it, and the registry is **`builtin_recipes()`, built on its first ask** and
kept for the process (`recipe_by_name` asks it). The
probe runs for a command that reads a recipe — `envs`, the notebook's start,
the prep of a sequence of groups (whose walk names molbuilder's own env by its
recipe) — and once at the server's start, so no request pays it. Nothing builds
it at import, so a command that reads no recipe never runs `nvidia-smi`.

### 3.1 One rule for both package kinds

A **bare string is the ordinary case** and is normalised into a record. A record
is written out only when the package needs something a name cannot say:

```python
conda_packages=(_PYTHON_SPEC, "pip", "numpy"),
pip_packages=("pubchempy",
              PipPackage("cupy-cuda13x", extras="[ctk]", optional=True,
                         reason="GPU only; the env is a full CPU env without it")),
```

Most of a recipe stays a readable list of names, and the entries that are
special **look** special. The rule is the same for both kinds, so a reader
learns it once.

`Recipe.conda_specs` / `.pip_specs` give the plain strings back for the places
that want them — conda's command line, the solver, the audit's pin parsing.

### 3.2 Optionality is a field, never a second list

There is no second list of optional names beside `conda_packages` or
`pip_packages`, for two reasons:

* **A membership list matched by name fails silently.** An entry that matches
  nothing leaves its package REQUIRED, and the first anyone hears is the audit
  reporting it missing with nothing to explain why.
* **Optionality has to reach the installer, not only the audit.** In one
  combined pip command, a single unavailable GPU wheel would abort a healthy
  CPU env however optional the recipe called it.

`optional` on the record governs **both**: the audit reports absence as
informational, and the installer does not let its failure abort the install:
a non-fatal step of its own for pip, the fallback solve without it for conda
(§ 4.4).

#### 3.2b Two packages are in EVERY recipe, and that is a rule

`python` and `git` are declared by every recipe, and a test holds it
(`test_every_recipe_declares_the_uniform_packages`). Both are about what is
available at RUN time on a machine this process cannot probe, which is why
neither can be left to "whatever that env happened to need":

* **`python`** — every python a generated job runs is the activated env's
  own, `$CONDA_PREFIX/bin/python3` (`running-a-job.md` § 2), and a job whose
  env holds none stops before it starts. Its version is `host-env.txt`'s
  `python` line (§ 3.4). See [`installation.md`](?doc=ops/installation.md)
  *"Choosing the Python every env is built on"*.
* **`git`** — `checkpoint.py`'s `GitNotInstalledError` tells the person to
  activate a molbuilder env *because* "every molbuilder env ships git as a
  conda_packages entry". That is a promise the registry has to keep, and HPC
  sites' system git versions are inconsistent enough that the env's is the only
  one under our control. When a save is taken is `checkpointing.md` § 9's
  rule — every prep that writes, Task setup's Save and a group's first launch,
  never a running job's run script.

*The rule is written here and held by a test, because a package every recipe
declares by hand, with no rule written, is how one recipe comes to lack it.*

### 3.2a `opt_in` is a third thing, and it is not `optional` *(2026-09-18)*

`optional` means **attempted, but failure does not abort**. `opt_in` means
**not attempted at all unless asked**. Both are string-or-flag fields on the
same records, and conflating them would be expensive: four packages depend on
today's meaning of `optional` — `PeptideBuilder` and `pubchempy` in the host
env, `cupy-cuda13x` and `gpu4pyscf` in pySCF — and all four *are* installed on
a default run. Redefining it would silently strip the peptide builder, PubChem
lookup and GPU support out of every fresh install.

`opt_in` carries a **string reason**, exactly as `Recipe.opt_in` does and for
the same reason: a bare `True` leaves every surface inventing its own wording
for *why am I not getting this*. **`doctor`'s `opt-in:` row is the surface that
renders it today** — the flag's help and `--dry-run`'s plan do not (the plan
simply omits an opt-in step, saying nothing about it). Stated as what is,
rather than as three surfaces that would have to be built.

**It applies to all three kinds**, which is what forced `extra_steps` to become
records (`ExtraStep`). The host env's test tooling is the case that needed it:
a conda package (`nodejs` — the Node every Node test runs on, the env's own
whatever else is on `PATH`), two pip packages (`playwright`,
`pytest-playwright`) and two steps — `playwright install chromium`, and the JS
test packages (below) — that **cannot** be gated by the packages alone: run
either in an env without its tool and it simply fails. A bare argv tuple still normalises into an `ExtraStep`, so a step
with nothing extra to say still reads as an argv.

**Why the test tooling is in the host env and not its own.** The `e2e` tests
start the real server *in-process*, so the process running them needs
molbuilder's whole import stack, which a browser-tooling-only env lacks — and a
second *full* env is no answer either: the suite runs under whatever interpreter invokes
`tools/testrun.py`, so a dev env would sit unused while `e2e` kept failing on
"playwright not installed".

**Confinement is part of the declaration, not a footnote.** `playwright
install` writes to `~/.cache/ms-playwright` by default — measured at 1.3 GB
there, outside every env and surviving `conda env remove`. The step points
`PLAYWRIGHT_BROWSERS_PATH` at `$CONDA_PREFIX/share/ms-playwright` and writes
the `activate.d` hook that keeps it pointed there for the pytest process too
(the manager performs the env's own activation — § 5.6). **No step may require
elevated privilege**: `playwright install-deps` installs distro packages as
root and is not run; a host missing those shared libraries is a host
prerequisite, like conda itself. A test runs the step against a stand-in
prefix and asserts where the browser and the activation hook land, and that
nothing reaches `~/.cache`.

**The JS test packages are a step of the same kind** *(the user, 2026-11-14:
"self-contained ... so that we don't have to contaminate the system")*: the
page scripts' Node tests are to run on a real DOM implementation, `jsdom`,
rather than the DOMs written by hand in test files today. What is
installed is pinned by a lockfile kept as data beside the host env's list —
`molbuilder/envs/js/package.json` and `package-lock.json` — and the step
installs exactly that, `npm ci --ignore-scripts` (no package's install script
runs), into the env's `recipes.JS_TEST_PACKAGES_HOME`
(`$CONDA_PREFIX/share/molbuilder-js`): inside the env, removed with it. npm's
cache is kept inside the prefix by the step's environment, beside pip's
(`env_for_step`, § 5.6); the person's own npm configuration — a site's
registry mirror — is honoured, as pip's is; the packages are `dependencies`,
not `devDependencies`, so no `omit` setting of the person's can leave them out.
No hook is needed: a test that uses them reads them from that folder under the
prefix of the env it runs in. A pin is changed in `package.json`, and the lockfile is
rewritten in that folder by the host env's own npm, its cache in that env:

```
conda activate molbuilder      # its npm, and its $CONDA_PREFIX
cd molbuilder/envs/js
npm_config_cache="$CONDA_PREFIX/var/cache/npm" npm install --package-lock-only --ignore-scripts
```

`doctor` does not check what an install step puts down — the browser or these
packages — only the declared packages: a test that needs either skips, naming
the command that installs it.

**An opt-in package that is absent is a choice, not a defect.** The audit
reports it under its own kind (`*-missing-opt-in`) and it does not fail the
health check — otherwise `doctor` would be red on every machine that simply did
not want the tooling. It is still *reported*, with the command that adds it,
because an absence nobody can see is one nobody acts on.

### 3.3 What a pip record can say that a string cannot

A spec string conflates three things that only coincide for an indexed package:

| | | breaks when |
|---|---|---|
| **identity** | what `dist-info` records — the audit's matching key | the spec is a URL |
| **source** | which index or URL supplies it | it isn't PyPI |
| **staleness** | whether comparing versions proves it current | two trees share a version |

`pyscf-properties` splits all three; `installation.md` § 3.1 tells that story.
The fields that follow from it are `source`, `force` (the installed version
cannot prove it is the declared build, so install unconditionally) and
`fallback_to_index` (an unreachable source must not cost us the package).

**`fallback_to_index` carries no flags, deliberately.** Its job is to make sure
the package EXISTS, not to replace what is there — with a force flag, an
unreachable source during an unrelated install would overwrite a good tree with
a worse one.

### 3.4 The host env's package list is one file, and two readers read it *(2026-10-04)*

Every env but one is installed by molbuilder's Python layer from its recipe.
The exception is the host env, `molbuilder`: it is the env that layer runs in,
so on a fresh machine it is created by the bash shim before any Python exists
(§ 8) — and bash cannot read `recipes.py`. So the host env's list is **data,
in one file both read**:

```
molbuilder/envs/host-env.txt           one entry per line; `#` is a comment
    python 3.12                        the python EVERY env is built on
    conda numpy                        a conda package, one spec
    pip PeptideBuilder <reason>        a pip package, optional, with why

scripts/install-env.sh     ── reads it ──►  creates the host env (`bootstrap`)
molbuilder/envs/recipes.py ── reads it ──►  `_HOST`'s packages; every recipe's python
```

* **One list, so nothing has to be kept in line.** Adding or removing a host
  package is one line in the file: a fresh bootstrap and every `molbuilder
  envs` verb (`install`, `doctor`, `repair`) then agree on it. The reason a
  package is there is the comment beside it, in the file.
* **A line holds a kind and a name — nothing a record says.** A host package
  that ever needs a source, a force flag or a fallback (§ 3.3) is a record in
  `recipes.py`, as is the **opt-in test tooling** (`nodejs`, `playwright`,
  `pytest-playwright`, the chromium step and the JS test packages' step,
  § 3.2a): the shim performs a
  default install, which leaves those out, so they are not on the list it
  reads.
* **The `python` line is the python every env is built on**, not only the
  host's (§ 3.2b) — the shim needs it before any Python exists, so it lives
  where the shim can read it. `--python` / `MOLBUILDER_PYTHON` overrides it
  for both readers.
* **A line neither reader understands is refused**, naming the file — by the
  shim before it creates anything, by `recipes.py` at import.

*(User: "both bash and recipes.py can read one ... well formatted file that
has one ... list" — no second copy is kept equal by a test.)*

## 4. The pipeline

### 4.1 The sequence, and which orderings may never move

A recipe's steps are not run in list order because a list has an order — most
of the sequence is **forced by dependency**, and those orderings are rules:

| # | stage | why it sits here |
|---|---|---|
| 1 | `conda create` | **Forced.** Nothing installs into an env that does not exist. It is also the one step needing no env prefix. It carries the recipe's **python spec and nothing else** — see 1b. |
| 1b | `conda install` | **Forced after create**, and a step of its own so that an env which ALREADY EXISTS can still receive it. Welded to `create`, which is skipped for an env that exists, the declared set could be applied only once, so a recipe that gained a package would never reach a machine that had already bootstrapped. Dispatched only when the env is short of something; see § 4.4. |
| 2 | pip, batched then per-package | **Forced after conda.** A forced pip install carries `--no-deps` *because* conda supplies numpy / scipy / h5py. Reorder this and `--no-deps` quietly installs a package with nothing under it. |
| 3 | `extra_steps` | **Forced after packages.** They run tools the packages provide. |
| 4 | `build_spec` | **Forced after conda+pip.** It needs the toolchain the solve installed. |
| 5 | verify | **Forced last.** It checks the final artifact, which is why it is pulled out of the plan and re-appended after the build. |

Only one ordering is a *choice* rather than a constraint: the batched plain-pip
step before the per-package ones, because the batch is the common case and
fails fast.

### 4.2 One shape, every situation

```python
# The pipeline, in pseudocode.  Everything else is detail.

def plan_install(recipe):               # pure: builds steps, runs nothing
    return [create_step_for(recipe),       # the interpreter, and nothing else
            conda_packages_step_for(recipe),   # the declared conda set (§ 4.4)
            *pip_steps_for(recipe),        # ...this is what --dry-run prints
            *extra_steps_for(recipe),
            verify_step_for(recipe)]


def run_step(step, prefix):             # THE ONE DOOR
    for n, argv in enumerate((step.argv, *step.fallbacks)):
        rc, output = dispatch_into_env(argv, prefix, env=env_for_step(prefix))
        if step.accepts(rc, output):              # the step's own criteria
            return step.decided(Outcome.decide(ok=True,
                                               first_attempt=(n == 0),
                                               fatal=step.fatal),
                                argv=argv)        # the argv that RAN
    return step.decided(Outcome.decide(ok=False, fatal=step.fatal))


def run_steps(steps, dispatcher):       # ONE loop, called once per stage
    for step in steps:
        if step.role is CREATE:                   # the EnvState machine
            decided = create_decision(dispatcher)
            if decided is not None:               # SKIPPED, or FAILED
                record(decided)
                return False if decided.stops_the_install else CONTINUE
        elif dispatcher.ensure_prefix() is None:
            record(undispatched(step, FAILED, "cannot resolve the prefix"))
            return False
        record(run_step(step, dispatcher.prefix))
        if the record stops the install:
            return False
    return True


def run_install(recipe):
    steps = plan_install(recipe)
    ok = run_steps(steps.before_verify, dispatcher)     # stages 1-3
    if ok and recipe.build_spec:
        ok = adapt(run_build_spec(...))                 # its own executor
    if ok:
        run_steps(steps.verify, dispatcher)             # the SAME loop
    return not any(s.outcome.stops_the_install for s in recorded)
```

`run_steps` is called twice with different step lists, and nothing it is
passed changes what it *does* — the one stage-shaped parameter, `tag`, only
labels the output. That is the test of whether the unification is real: with
policy moved onto the step, rather than read off the *recipe* at execution
time, verify is an ordinary step.

It branches on `StepRole`, never on `label`: the label is display text, and a
branch on it would let a rename for clarity silently disable the create-skip,
so every install would re-attempt `conda create` on an existing env.

### 4.3 How a field covers each situation

Each row is something an env actually needs. **None of them is a branch in the
runner** — every one is a field on a record that a translator turns into a
step the one door can run.

| the situation | expressed as | what happens | outcome when it goes wrong |
|---|---|---|---|
| an ordinary package | a bare string in the list | batched into one `pip install` | `FAILED` |
| a package whose version cannot prove it is the declared build | `force=True` | `--force-reinstall --no-deps`, its own step, every time | `FAILED` |
| a pip package the env is usable without | `optional=True` | its own step, `fatal=False` | `DEGRADED`, install continues |
| a conda package the env is usable without | `optional=True` | **a degraded attempt on the packages step** — see below | `RECOVERED` |
| a package not on PyPI | `source="git+https://…"` | PEP 508 `name @ url` | `FAILED` |
| …whose source may be unreachable | `fallback_to_index=True` | the indexed build as a declared alternative, **flagless** | `RECOVERED` |
| a tool that exits 0 while the thing we asked for is missing | `expect_contains="…"` | checked in addition to the exit code | `FAILED`, naming the substring |
| the env already exists | — (`EnvState`) | create is not dispatched | `SKIPPED`, and no exit code |
| the env is wreckage | — (`EnvState`) | nothing is dispatched | `FAILED`, naming `rm -rf <prefix>` (ORPHAN, BROKEN) or `--clean` (GHOST) |
| a binary that must be compiled | `build_spec=BuildSpec(…)` | `builds.py`'s own executor, results adapted | `FAILED` per phase |
| "tell me, do not do it" | `--dry-run` | `plan_install` only; nothing dispatches | no outcome at all |

That last row is why an outcome is `Optional`: a step with no outcome is a step
that is still a *plan*. Everywhere else, every step carries one.

### 4.4 The conda set is delivered, not conferred at birth *(2026-09-17)*

`conda create -n <env> pkg…` does two jobs in one command, and `create` is
skipped for an env that exists, so a conda list carried only by the birth
command never reaches an existing env while `install` reports success. A recipe
says *this env contains these packages*, not
*this env was born with these packages*.

> **`conda create` carries the recipe's python spec and nothing else. Every
> other declared package is delivered by `conda install`, which runs whenever
> the env is short of something — whether it was just created or has existed
> for months.**

**The gate is free, and that is what makes it affordable on every install.**
`doctor.absent_conda_specs` reads `conda-meta/` off disk — no subprocess, no
solve, no network — and is the same reading § 6's audit takes. Nothing absent →
the step is `SKIPPED` and the env is untouched.

**Why gated rather than unconditional.** Measured on conda 26.7.1: a full-list
`conda install` into a **satisfied** env still costs 26 s and wants to update
`ca-certificates` and `openssl` to newer builds. Running it every time would
drift a healthy env on every re-run of a command whose whole promise is
idempotence.

**Why the argv is the full set and not the absent subset.** `plan_install` is
pure, so `--dry-run` has already printed this line; narrowing it at dispatch
would print one command and run another. The runner decides *whether*, never
*what*.

**Absence only.** A version or build mismatch is `repair --include-version-fix`,
deliberately: those rebuilds are destructive and opt-in. `install` closes
absence. And for a recipe with a `build_spec` a version mismatch is not
repairable at all — the conda packages there are the toolchain the binaries were
compiled against, so re-pinning the package recompiles nothing; `doctor` names a
rebuild for that case (measured: `sysroot_linux-64` pinned 2.17, env at 2.34,
`objdump -T` on the built `siesta` demanding `GLIBC_2.34`).

**Optionality means something different to a solver.** pip installs one package
at a time, so an optional pip package is its own non-fatal STEP. conda solves
everything at once — an optional conda package cannot be its own step without
paying a *second solve*, and a second solve may legitimately change versions for
packages the first one already placed. So conda optionality is an **attempt**,
not a step property: `conda_packages_step_for` tries the full solve, and on
failure the same solve without the optional specs. The env lands `RECOVERED` — created,
minus something the recipe said it could live without.

**One degradation step, not a search.** With *n* optional specs, "find the
largest subset that still solves" is 2ⁿ solves at minutes each, so the rule is
all of them or none. Nothing is lost silently: the audit then names exactly
which optional packages are absent, with the recipe's `reason` beside each, and
`repair --include-optional` installs them in one batched `conda install`,
paying the second solve there because the operator asked for it. A recipe with
no optional conda package gets no fallback.

**`fallback_to_index` carries no flags, deliberately.** Its job is to make sure
the package EXISTS, not to replace what is there — with a force flag, an
unreachable source during an unrelated install would overwrite a good git tree
with a worse indexed one.

## 5. Execution

`plan_install` builds `InstallStep`s and runs nothing. `run_step` runs exactly
one step. **Every step goes through it** — every install phase, `repair`, and
`doctor`'s verify probe. A private dispatch re-derives the same four things
(prefix resolution, the way into the env, the launch-failure branch, the output
trim), and each copy drifts; a bare pip command would know nothing of a
package's alternatives.

A record becomes a step in exactly one place per kind — `create_step_for` for
the env itself, `conda_packages_step_for` for the recipe's conda set,
`pip_steps_for` for its pip packages (the plain ones batched into one step, each
other one through `pip_step_for`), `extra_steps_for` for its extra steps,
`verify_step_for` for its verify fields, `remove_step_for` for `--clean`'s
removal, and `conda_step_for` for `repair`'s batched conda install. One
place writes each command line: `pip_argv` and `conda_argv`. So force flags,
optionality, fallbacks and verify criteria are each read off one record by one
reader, and the channel flags cannot sit before the specs in one caller and
after them in another.

### 5.1 What the five outcomes mean in a result

The states and the transition rule are § 2.2 — this is what they imply for the
`InstallResult` a caller reads.

`rc is None` is not a special case in that rule. It is simply *not accepted*.

**Every step in an `InstallResult` carries an outcome** — including the ones
decided *without* being dispatched (`_undispatched`): a skipped create, an env
too broken to install into, an unresolvable prefix. Those keep `returncode =
None`, because a step that did not run has no exit code, and "I did not do
this" must not read as "I did this and it worked". The only step with no outcome is one that is still a *plan*, which is
what `--dry-run` prints.

`run_step` returns the argv that **actually ran**, so a recovered step can never
be reported under the attempt that failed.

### 5.2 What counts as success rides on the step

An exit code alone is not enough. **Every** recipe adds a substring its verify
output must contain (`expect_contains`) — a tool can exit `0` while the thing we
asked about is missing. **The exit code is never waived.** A tool that exits
non-zero from a healthy start is given input that makes it stop cleanly —
`tleap` gets a `quit` — rather than having its exit code ignored, which would
print a tool's `*** Error` under `verify: OK`. The rule rides on the
**step** and is checked by `InstallStep.accepts`.

On the step, verify is an ordinary step and the runner needs no knowledge of
which phase it is serving. A process that never
launched is never accepted.

### 5.3 The install's verdict is derived

`succeeded` is `not any(step.outcome.stops_the_install ...)`, computed from the
steps at the end. A verdict tracked beside the steps can drift from them (a
failure announced as "SKIPPED"), so there is none.

Note **which** predicate. `is_success` asks about one STEP — and is `False` for
`DEGRADED`, because an optional package that could not be installed is honestly
absent. `stops_the_install` asks about the INSTALL. `DEGRADED` is exactly where
the two answers differ, and that difference is the whole point of `optional`.

### 5.4 What is deliberately outside the door

`install --clean`'s env removal is an install step like any other:
`remove_step_for(env_name, conda)`, beside `create_step_for`, dispatched
through `run_step`, so the wipe carries an `Outcome` and a line in the result
the verdict is derived from (§ 5.3).

`validate.py`'s post-install probes (SIESTA's `ctest`, ELPA's `make check`)
dispatch on their own. They are not install steps and do not become them: they
answer "is this build scientifically sane", they need a working directory and
their own output parsing, and they report a `ProbeResult`. The rule is that every
`InstallStep` goes through `run_step`, not that no subprocess may exist.

### 5.5 Source builds keep their own executor

`builds.run_build_spec` is not a sequence of `InstallStep`s and does not become
one: it has sentinel-based resume, which no install step has. What the installer
does is **adapt** its results — each build phase becomes a step whose outcome
is read from the build's own status by one mapping (ok → OK, fail → FAILED,
skip → SKIPPED; a phase never reached is not recorded) — while `builds.py`
keeps its own verdict, which the installer reads rather
than re-derives.

### 5.6 How a step enters the env *(2026-09-12)*

`run_step` decides *whether* a step is accepted; this is about *how* its argv
reaches the env's packages — and the rule is that molbuilder does not answer
that question itself.

**The manager activates.** `_run_argv` is the one spelling —
`<mgr> run -n <env> --no-capture-output …`, re-addressed at `--prefix` once the
prefix is known — and the detected binary
performs the activation its own envs expect, including `etc/conda/activate.d`
hooks, which source-built recipes depend on (siesta-gpu puts binaries under
`<prefix>/opt/…/bin`, not `<prefix>/bin`). Re-creating that by exporting `PATH`,
`LD_LIBRARY_PATH` and `CONDA_PREFIX` and globbing the hook scripts is a
reimplementation of someone else's product, and it is wrong by construction on
any manager whose activation does something we did not copy.

**A manager's bug is MEASURED, and it is not a step outcome.** Some mambas' `run`
(1.x; Sol's `mamba` module 2.9.0 too, 2026-10-18) emits a stub whose `exec --` bash
rejects. Two things follow:

* **Measured, not predicted.** The first dispatch that fails with that
  signature switches the process to `activation_wrapper` and says so once. A
  version check would mean tracking someone else's release history; the stub
  announces itself, and it dies *before* the inner command starts, which is what
  makes the retry safe rather than a second half-install. One wasted launch per
  process on a broken manager, none at all on a working one — **and none on a
  machine whose record remembers it** *(2026-10-18)*: the dispatch that
  measures it saves it in this machine's record there and then
  (`builds.remember_manager_run`, `manager_run`,
  [`configuration.md`](?doc=configuration.md) § 5 M-2; `jobset probe` makes
  the same measurement for a record it writes); the door reads this machine's
  record before the first try and takes the record's word for the manager it
  names. A record about another manager, or none to write into: the try, the
  note and the switch, as before.
* **Not `RECOVERED`.** A declared alternative on a step means *this package may
  come from another source* (§ 4.3), and an `Outcome` describes what became of a
  package. A broken manager is a property of the **machine**: reporting every
  step on such a cluster as `RECOVERED` would state something false about
  every package in the recipe. So the fallback lives inside the one door, is
  reported once, and the step's outcome is whatever the step's own criteria say.

**Environment, not activation.** What a step legitimately needs beyond activation
— `TMPDIR`, `PIP_CACHE_DIR` and npm's cache (`npm_config_cache`) kept inside
the prefix so `env remove` really cleans up, and `builds.build_subprocess_env()`'s stripping of host `CPATH` /
`CFLAGS` / `CUDA_HOME` / `OMPI_*` leakage — is computed by `builds.env_for_step`
inside `run_step` (and by `builds._build_env` for a build phase). A step does not
choose its environment; `run_step` takes none from its caller.

**What this does NOT cover: a generated job script.** `runwrap.py` writes the
activation line into a wrapper **verbatim from the target's record** (its
`env_init`, copied by `jobset probe` on that machine from its
`molbuilder.json`) and must keep
doing so. That script runs on a machine this process cannot probe -- a cluster
node, after `module load`, possibly minutes or days later -- so the only
correct source is what was declared for that machine
([`configuration.md`](?doc=configuration.md) § 5 M-1).
[`installation.md`](?doc=ops/installation.md) M1 is about the env manager **on
this machine, now**; the wrapper's activation is a declared fact
of the target, and detecting it here would be the same error in the other
direction.

**A prefix is asked for, never derived**, and a printed remedy names the
detected manager and never proposes destroying the env the process is running
from: [`installation.md`](?doc=ops/installation.md) § *"One door for RUNNING"*
M2, M3, M5.

**Where it lives.** `dispatch_into_env`, `activation_wrapper`, `env_for_step`
and the run speller are in `builds.py`, beside `run_streaming` and
`build_subprocess_env` — the lowest layer, because every caller must enter an
env identically: the step runner (`install.run_step`), the build executor
(`builds._run_build_phase`) and its installed-and-working gate, the tool router
(`_dispatch.run_in_env`), and the notebook's start (`jupyter.py`, which passes
no pipe and so gives up the automatic fallback). One place writes the command line (`_run_argv`) and there
are two public addresses — `conda_run_argv` by name, for the planner, which has
no prefix because `plan_install` runs nothing; `conda_run_prefix_argv` by prefix,
for a caller that already has one and must not re-derive a name from a directory
to use it.

Where the manager's `run` works, `envs doctor` verifies every present env
through it, with no generated shell.

## 6. The audit

`audit_packages` reads `conda-meta/*.json` and `site-packages/*.dist-info`
**off disk**. No subprocess, no `conda list`, no network. The source of truth is
the metadata the package manager itself wrote at install time.

It answers two questions:

* **is it there** — matched on IDENTITY alone, so a URL-sourced package is never
  reported permanently missing;
* **is it the one we asked for** — matched on PROVENANCE, read from PEP 610
  `direct_url.json`, because a version string cannot always tell two trees
  apart.

A provenance mismatch the recipe *accepts* (`fallback_to_index`) is
informational: the env is degraded, not broken.

It iterates **records**, both kinds, so every issue names the package it came
from and carries that package's `reason`. Walking spec strings would need a
name-set of the optional ones (the mechanism § 3.2 forbids for failing
silently), would leave `name` empty so `repair` could not map a conda issue back
to its record, and would keep `CondaPackage.reason` from reaching a user. `optional` suffixes every conda kind, not only absence: an
optional package resolving to the wrong build must not fail the health check for
an env the recipe calls usable without it.

`doctor` also runs the recipe's **verify** step, and it runs the same step the
installer does — built by `verify_step_for`, dispatched by `run_step`, judged by
`accepts`. That is the only way the health report can agree with what `install`
just said. It trims its excerpt tighter (2 KiB against the installer's 4) because
it prints one block per env, six at most; that is a display choice,
not a second rule.

## 7. Repair

`repair` acts on audit issues, and for each one it **maps back to the record**
by name and runs the step the installer would. It does not build commands and
does not carry a flattened copy of the instruction — an issue says what is
WRONG; the recipe says what to do about it.

This is why `PackageAuditIssue` carries `name` and not `install_flags`: a second
copy of an instruction can only drift from the first.

**Both kinds map back; they differ in GRANULARITY, and that follows from § 4.3.**
pip is repaired per package, through `pip_step_for`, so each one gets its
source, its force flags and its declared alternative. conda is repaired in one
batched `conda_step_for`, because every conda command is a whole solve — and
that is also why `install` does not do it this way: there, an optional conda
package is dropped from the packages solve instead. `repair` is the place that
pays a second solve, because the operator asked for one.

## 8. The shell shim

`scripts/install-env.sh` solves one chicken-and-egg problem — you cannot run
`molbuilder envs` until the host env exists — and forwards everything else.

Because it owns that one install, it has to honour one flag it otherwise just
forwards: **`--dry-run` means nothing gets installed, including by the shim.**
Creating the host env is the largest install in the flow, and planning needs
that env, so where it is absent `bootstrap --dry-run` refuses and names the
command that makes it work.
The shim *consumes* two flags — `--gcc` and `--python` — and this one it reads
and also passes on.

**The flag reference is Python's.** The shim's help names only what Python
cannot know — `--gcc` and `--python`, which it consumes because `recipes.py`
reads `MOLBUILDER_GCC` and `MOLBUILDER_PYTHON` at import time, and the two
flags it reads and forwards — and
points at `<subcommand> --help` for the rest. **A consumed flag is parsed
before anything reads it**: `--python` reaches the bash array of the host env's
packages, built before any python exists, so a later parse would build the host
env on one python and every other env on another. `--gcc` is read only
Python-side. The help does not re-specify Python's flags: a second copy of a
flag surface nothing can keep in sync drifts, down to flags the Python layer
prints as the command to run.

**A PLAN IS NOT A CHANGE** *(2026-09-17)*. `--dry-run`, `--check` and `--help`
change nothing, so the arg scan implies `--yes` for each (otherwise every way CI
or a pipe asks what this program would do stops at a confirmation with no TTY)
and none of them performs the `--clean` wipe. The detected manager is still
announced, so nothing is hidden.

**`bootstrap --clean` is the shim's third owned concern** *(2026-09-17)*, and it
is the host-env chicken-and-egg again rather than a new job: `dispatch` runs
`<host env>/bin/python`, so the Python layer cannot wipe the env its own
interpreter lives in (`installation.md` M5, which is why `install --clean`
refuses that env outright). This shell is not in that env — `${ENV_MGR}` is the
manager's own binary — so `env remove` here is an ordinary conda command
followed by the `create` the shim already owns. The window between them is real
and is named: the recovery command is printed *before* the removal, not after a
failure. Python applies `--clean` to every other recipe and says so for the one
it skips.

**The env-manager confirmation is asked only by the verbs that CHANGE an env**
*(2026-09-12)*. `require_conda` prompts *"use this env manager?"* — which exists
for the case where an old `~/anaconda3` is picked over the Miniforge you meant —
and that only matters when something is about to be created with it. So it is
asked only by `bootstrap` and `install` (`_verb_changes_envs`). Both accept
`--yes`, so the remedy printed with no TTY is runnable. A read-only verb
(`doctor`, `list`, `validate`, `repair`) defines no `--yes` and needs none, so a
health check runs in CI.

**`--help` never installs.** It implies `--yes`, so with no TTY the pointer
still prints, and `MB_WANT_HELP` stops the `bootstrap` arm when the host env is
absent (creating a multi-GB env to answer a question about flags is the install
nobody asked for) and says why per-verb help needs that env (the help comes from
molbuilder itself, which lives in it).

**It reads the host env's package list from the one file the registry reads**,
`molbuilder/envs/host-env.txt` (§ 3.4), and creates the host env from it — it
holds no list of its own.

## 9. What stays asymmetric, and why

Three differences are deliberate. Each traces to something the two package
managers actually do differently, not to how the code grew.

**The spec's shape.** `CondaPackage` carries `spec` — conda's own grammar, with
its version and build pin — while `PipPackage` carries a parsed `name` plus
`source` and `extras`. conda's spec language is rich and its own, and the audit
parses it; pip's had to be taken apart because identity and source could not
share one string (§ 3.3).

**Where optionality lives.** A non-fatal step for pip, a degraded attempt for
conda (§ 4.3). pip installs one package; conda solves all of them.

**Repair's granularity.** Per package for pip, batched for conda (§ 7), for the
same reason.

What is symmetric, and must stay so: both kinds are a
record per package, both normalise from a bare string, both carry `optional` and
`reason`, both are audited by iterating records, and both reach `repair` through
a lookup by name.
